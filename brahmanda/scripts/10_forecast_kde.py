"""
10_forecast_kde.py
===================
KDE forecast + ridge plot from a retrieval JSON produced by script 09/09b.

What this does (per the EventCast PDF Stage 4)
----------------------------------------------
  1. Load the retrieval JSON for one base cluster + its top-K analogues.
  2. For each analogue, fit a Gaussian KDE on its `realised_path` (daily
     log-returns over first_seen .. last_seen + post_days).
  3. Build the alpha-aggregated prior:
        pi(r) = sum_i ( w*_i / sum w*  ) * KDE_i(r)
     evaluated on a discretised grid of returns.
  4. Sample N cumulative h-day return paths from the weighted analogue mixture
     for each horizon h in {1, 3, 7, 15}; report VaR(95/99%) and ES(95/99%).
  5. Render an interactive Plotly ridge plot (one ridge per analogue, the
     aggregated prior in bold at top, the base cluster's own realised path
     marked as a vertical line for visual calibration).

Reads
-----
  --retrieval_json   any analogues_<cluster_id>.json from script 09 / 09b
  --cluster_index    cluster_memory*.parquet (to look up the BASE cluster's
                     realised_path for the calibration overlay; optional)

Writes (default: alongside the retrieval JSON)
----------------------------------------------
  forecast_<cluster_id>.json   per-horizon: prior_quantiles, VaR, ES, samples
  ridge_<cluster_id>.html      interactive ridge plot

Run
---
    python 10_forecast_kde.py \
        --retrieval_json clusters_full_history/retrievals/2026_w15/analogues_c15d_00103.json \
        --cluster_index  clusters_full_history/cluster_memory_MERGED_2001_2026.parquet \
        --horizons 1,3,7,15 \
        --n_samples 10000

Importable for the API
----------------------
    from forecast_kde import load_retrieval, fit_prior, horizon_forecast
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.stats import gaussian_kde
except ImportError:
    sys.exit("ERROR: scipy required. pip install scipy")

try:
    import plotly.graph_objects as go
except ImportError:
    sys.exit("ERROR: plotly required. pip install plotly")


# ---------------------------------------------------------------------------
# DEFAULTS
# ---------------------------------------------------------------------------
DEFAULT_HORIZONS  = (1, 3, 7, 15)
DEFAULT_N_SAMPLES = 10000
DEFAULT_BANDWIDTH = "scott"   # 'scott' | 'silverman' | float
GRID_POINTS       = 401


# ---------------------------------------------------------------------------
# DATA LOADERS
# ---------------------------------------------------------------------------
def load_retrieval(path: Path) -> Dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def load_base_realised_path(cluster_index_path: Optional[Path],
                            base_cluster_id: str) -> Optional[List[float]]:
    """Look up the base cluster's own realised_path from the cluster index
    (so we can overlay it on the ridge plot for calibration)."""
    if cluster_index_path is None or not cluster_index_path.exists():
        return None
    try:
        import pandas as pd
        if cluster_index_path.suffix == ".parquet":
            df = pd.read_parquet(cluster_index_path,
                                 columns=["cluster_id", "realised_path"])
        else:
            df = pd.read_pickle(cluster_index_path)[["cluster_id", "realised_path"]]
        row = df[df["cluster_id"] == base_cluster_id]
        if row.empty:
            return None
        rp = row["realised_path"].iloc[0]
        if isinstance(rp, str):
            rp = json.loads(rp)
        return list(rp)
    except Exception as e:
        print(f"[warn] could not load base realised_path: {e}")
        return None


# ---------------------------------------------------------------------------
# KDE FITTING
# ---------------------------------------------------------------------------
def fit_kde(returns: List[float], bandwidth: Any = DEFAULT_BANDWIDTH
            ) -> Optional[gaussian_kde]:
    """Fit Gaussian KDE on a 1-D array of log-returns. Returns None if
    insufficient data (<5 points or zero variance)."""
    arr = np.asarray(returns, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 5:
        return None
    if arr.std() < 1e-9:
        return None
    try:
        return gaussian_kde(arr, bw_method=bandwidth)
    except Exception:
        return None


def fit_prior(analogues: List[Dict[str, Any]], bandwidth: Any = DEFAULT_BANDWIDTH
              ) -> Tuple[List[Optional[gaussian_kde]], np.ndarray]:
    """Returns (per-analogue KDEs, normalised weights)."""
    kdes, raw_w = [], []
    for a in analogues:
        rp = a.get("factors", {}).get("realised_path", [])
        kde = fit_kde(rp, bandwidth=bandwidth)
        kdes.append(kde)
        raw_w.append(float(a.get("scores", {}).get("composite_weight", 0.0)))
    raw_w = np.asarray(raw_w, dtype=np.float64)
    # Mask out failed KDEs
    mask = np.array([k is not None for k in kdes])
    if not mask.any():
        return kdes, np.zeros(len(analogues))
    # Normalise weights over valid analogues; re-weight uniform if any all-negative
    w_valid = raw_w[mask]
    if w_valid.sum() <= 0:
        w_valid = np.ones_like(w_valid)
    w_valid = w_valid / w_valid.sum()
    weights = np.zeros(len(analogues))
    weights[mask] = w_valid
    return kdes, weights


def aggregated_density(kdes: List[Optional[gaussian_kde]], weights: np.ndarray,
                       grid: np.ndarray) -> np.ndarray:
    """pi(r) = sum w_i * KDE_i(r), evaluated on grid."""
    out = np.zeros_like(grid, dtype=np.float64)
    for k, w in zip(kdes, weights):
        if k is None or w <= 0:
            continue
        out += w * k(grid)
    return out


# ---------------------------------------------------------------------------
# HORIZON FORECASTING — sample-based
# ---------------------------------------------------------------------------
def horizon_forecast(analogues: List[Dict[str, Any]], weights: np.ndarray,
                     horizon: int, n_samples: int = DEFAULT_N_SAMPLES,
                     rng: Optional[np.random.Generator] = None
                     ) -> np.ndarray:
    """Sample N cumulative h-day log-returns from the weighted analogue mixture.
    For each sample: choose an analogue ~ weights, then take a random contiguous
    h-day chunk from its realised_path."""
    rng = rng or np.random.default_rng(42)
    valid = [(a, w) for a, w in zip(analogues, weights)
             if w > 0 and len(a.get("factors", {}).get("realised_path", [])) >= horizon]
    if not valid:
        return np.array([])
    valid_w = np.array([w for _, w in valid])
    valid_w /= valid_w.sum()
    samples = []
    while len(samples) < n_samples:
        i = rng.choice(len(valid), p=valid_w)
        path = np.asarray(valid[i][0]["factors"]["realised_path"])
        path = path[np.isfinite(path)]
        if path.size < horizon:
            continue
        start = rng.integers(0, path.size - horizon + 1)
        samples.append(float(path[start:start + horizon].sum()))
    return np.asarray(samples)


def risk_measures(samples: np.ndarray, alpha: float = 0.05) -> Dict[str, float]:
    """VaR and ES at the given alpha-tail. Returns are log-returns; positive
    losses correspond to negative returns. VaR is reported as a positive
    number representing the loss magnitude."""
    if samples.size == 0:
        return {"VaR": float("nan"), "ES": float("nan"),
                "median": float("nan"), "mean": float("nan")}
    var = -float(np.quantile(samples, alpha))
    tail = samples[samples <= -var]
    es  = -float(tail.mean()) if tail.size else float("nan")
    return {
        "VaR": var, "ES": es,
        "median": float(np.median(samples)),
        "mean":   float(samples.mean()),
        "std":    float(samples.std()),
        "n":      int(samples.size),
    }


# ---------------------------------------------------------------------------
# RIDGE PLOT — Plotly, one ridge per analogue + aggregated prior at top
# ---------------------------------------------------------------------------
def build_ridge(retrieval: Dict[str, Any],
                kdes: List[Optional[gaussian_kde]],
                weights: np.ndarray,
                grid: np.ndarray,
                base_realised_path: Optional[List[float]] = None) -> go.Figure:
    analogues = retrieval["analogues"]
    n = len(analogues)
    fig = go.Figure()

    # Spacing between ridges
    row_height = 1.0
    max_density_per_ridge = []

    # Pre-compute densities (and find max for vertical scaling)
    densities = []
    for kde in kdes:
        if kde is None:
            densities.append(None)
            max_density_per_ridge.append(0.0)
        else:
            d = kde(grid)
            densities.append(d)
            max_density_per_ridge.append(float(d.max()))
    overall_max = max(max_density_per_ridge) if max_density_per_ridge else 1.0
    if overall_max <= 0:
        overall_max = 1.0
    scale = 0.9 / overall_max   # each ridge fits ~90% into its row

    # Color ramp by composite_weight: brighter = higher weight
    palette = [
        "#08306b", "#08519c", "#2171b5", "#4292c6", "#6baed6",
        "#9ecae1", "#c6dbef", "#deebf7", "#cccccc",
    ]

    # Ridges (bottom to top) — best-ranked at top
    for i, (a, density, w) in enumerate(zip(analogues, densities, weights)):
        if density is None:
            continue
        y_base = (n - 1 - i) * row_height
        rank = a.get("rank", i + 1)
        color = palette[min(rank - 1, len(palette) - 1)]
        label = f"#{rank}  {a['cluster_id'][:14]}  {a['first_seen'][:10]}  ({a.get('label','')[:40]})"
        # filled ridge
        fig.add_trace(go.Scatter(
            x=grid, y=y_base + density * scale,
            mode="lines",
            line=dict(color=color, width=1),
            fill="tonexty",
            fillcolor=_with_alpha(color, 0.45),
            name=label,
            hoverinfo="text",
            hovertext=[f"{label}<br>return={r:.4f}<br>density={d:.2f}"
                       for r, d in zip(grid, density)],
            showlegend=False,
        ))
        # baseline (zero) trace per ridge so fill='tonexty' works
        fig.add_trace(go.Scatter(
            x=grid, y=[y_base] * len(grid),
            mode="lines", line=dict(color="rgba(0,0,0,0)", width=0),
            hoverinfo="skip", showlegend=False,
        ))
        # left-side label
        fig.add_annotation(
            x=grid[0], y=y_base + 0.3,
            text=label, showarrow=False, xanchor="right",
            font=dict(size=9, color="#333"), xshift=-5,
        )

    # Aggregated prior — bold ridge at the very top
    pi = aggregated_density(kdes, weights, grid)
    if pi.max() > 0:
        y_base_top = n * row_height + 0.5
        scale_top = 1.4 / pi.max()
        fig.add_trace(go.Scatter(
            x=grid, y=y_base_top + pi * scale_top,
            mode="lines",
            line=dict(color="#cb181d", width=2.5),
            fill="tonexty",
            fillcolor="rgba(203, 24, 29, 0.30)",
            name="α-aggregated prior",
            hoverinfo="text",
            hovertext=[f"prior  return={r:.4f}  density={d:.2f}" for r, d in zip(grid, pi)],
        ))
        fig.add_trace(go.Scatter(
            x=grid, y=[y_base_top] * len(grid),
            mode="lines", line=dict(color="rgba(0,0,0,0)", width=0),
            hoverinfo="skip", showlegend=False,
        ))
        fig.add_annotation(
            x=grid[0], y=y_base_top + 0.7,
            text="<b>α-AGGREGATED PRIOR</b>", showarrow=False,
            xanchor="right", font=dict(size=11, color="#cb181d"), xshift=-5,
        )

    # Base cluster's own realised path — vertical line + marker per day
    if base_realised_path:
        rp = np.asarray(base_realised_path, dtype=np.float64)
        rp = rp[np.isfinite(rp)]
        if rp.size:
            mean_realised = float(rp.mean())
            fig.add_vline(x=mean_realised,
                          line=dict(color="#000000", width=2, dash="dash"))
            fig.add_annotation(
                x=mean_realised, y=(n + 1) * row_height + 1.0,
                text=f"<b>BASE realised mean = {mean_realised:.4f}</b>",
                showarrow=True, arrowhead=2, ax=40, ay=-30,
                font=dict(size=11, color="#000000"),
                bgcolor="rgba(255,255,255,0.85)",
            )

    # Layout
    base_id = retrieval.get("query", {}).get("cluster_id", "?")
    base_label = retrieval.get("query", {}).get("label", "")
    fig.update_layout(
        title=dict(
            text=(f"<b>KDE forecast — base {base_id}</b><br>"
                  f"<sub>{base_label}  ·  "
                  f"K={n} analogues  ·  ridges sorted by rank "
                  f"(top = aggregated prior)</sub>"),
            x=0.02, y=0.98, font=dict(size=14, color="#212529"),
        ),
        xaxis=dict(title="daily log-return", showgrid=True, gridcolor="#e1e7ee",
                   zerolinecolor="#bbb"),
        yaxis=dict(showticklabels=False, showgrid=False, range=[-0.5, n + 3]),
        height=max(450, 30 * n + 200),
        margin=dict(l=260, r=20, t=80, b=40),
        plot_bgcolor="#fafbfd", paper_bgcolor="white",
        hoverlabel=dict(bgcolor="white"),
    )
    return fig


def _with_alpha(hex_color: str, alpha: float) -> str:
    """Convert '#rrggbb' to 'rgba(r,g,b,a)'."""
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha:.2f})"


# ---------------------------------------------------------------------------
# DRIVER
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--retrieval_json", required=True,
                    help="analogues_<cluster_id>.json from script 09 / 09b")
    ap.add_argument("--cluster_index", default=None,
                    help="cluster_memory*.parquet (for base realised_path "
                         "calibration overlay; optional)")
    ap.add_argument("--output_dir", default=None,
                    help="defaults to alongside the retrieval JSON")
    ap.add_argument("--horizons", default=",".join(str(h) for h in DEFAULT_HORIZONS))
    ap.add_argument("--n_samples", type=int, default=DEFAULT_N_SAMPLES)
    ap.add_argument("--bandwidth", default=DEFAULT_BANDWIDTH,
                    help="'scott' | 'silverman' | a float")
    ap.add_argument("--grid_lo", type=float, default=-0.10,
                    help="lower bound of return grid (default -0.10 = -10%)")
    ap.add_argument("--grid_hi", type=float, default=0.10)
    args = ap.parse_args()

    rj_path = Path(args.retrieval_json).resolve()
    out_dir = Path(args.output_dir).resolve() if args.output_dir else rj_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {rj_path.name}")
    retrieval = load_retrieval(rj_path)
    base_id = retrieval.get("query", {}).get("cluster_id", "unknown")
    analogues = retrieval.get("analogues", [])
    if not analogues:
        sys.exit(f"ERROR: retrieval JSON has no analogues for {base_id}")
    print(f"[load]   base={base_id}  K={len(analogues)} analogues")

    # Bandwidth: try float else string
    try:
        bw = float(args.bandwidth)
    except ValueError:
        bw = args.bandwidth

    # Fit KDEs + prior
    print(f"[fit]    fitting KDEs (bandwidth={bw})")
    t0 = time.time()
    kdes, weights = fit_prior(analogues, bandwidth=bw)
    n_valid = sum(1 for k in kdes if k is not None)
    print(f"[fit]    {n_valid}/{len(kdes)} analogues with valid KDE  "
          f"(in {time.time()-t0:.1f}s)")
    if n_valid == 0:
        sys.exit("ERROR: no analogues had a fittable KDE (all had <5 returns)")

    # Density grid
    grid = np.linspace(args.grid_lo, args.grid_hi, GRID_POINTS)
    pi = aggregated_density(kdes, weights, grid)
    pi_norm = pi / np.trapz(pi, grid) if pi.sum() > 0 else pi

    # Horizon forecasts + risk measures
    print(f"[forecast] sampling N={args.n_samples} per horizon")
    horizons = [int(h) for h in args.horizons.split(",")]
    rng = np.random.default_rng(42)
    forecast_summary: Dict[str, Any] = {"horizons": {}}
    for h in horizons:
        samples = horizon_forecast(analogues, weights, h, args.n_samples, rng)
        r95 = risk_measures(samples, alpha=0.05)
        r99 = risk_measures(samples, alpha=0.01)
        forecast_summary["horizons"][f"h{h}"] = {
            "n_samples":  int(samples.size),
            "mean":       r95["mean"],
            "std":        r95["std"],
            "median":     r95["median"],
            "VaR_95":     r95["VaR"],   "ES_95": r95["ES"],
            "VaR_99":     r99["VaR"],   "ES_99": r99["ES"],
            "quantiles":  {q: float(np.quantile(samples, q)) for q in
                           (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
                           if samples.size},
        }
        print(f"[forecast] h={h:>2}  n={samples.size:>5}  "
              f"mean={r95['mean']:+.4f}  std={r95['std']:.4f}  "
              f"VaR95={r95['VaR']:.4f}  ES95={r95['ES']:.4f}")

    # Save forecast JSON
    forecast_summary["query"]    = retrieval.get("query", {})
    forecast_summary["config"]   = {
        "n_analogues": len(analogues), "n_valid_kde": n_valid,
        "bandwidth": str(bw), "n_samples": args.n_samples,
        "horizons": horizons,
        "grid_lo": args.grid_lo, "grid_hi": args.grid_hi,
    }
    forecast_summary["weights"]  = {a["cluster_id"]: float(w)
                                    for a, w in zip(analogues, weights)}
    out_json = out_dir / f"forecast_{base_id}.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(forecast_summary, f, ensure_ascii=False, indent=2,
                  default=str)
    print(f"[save] {out_json}")

    # Base cluster's realised path for calibration overlay
    cidx = Path(args.cluster_index).resolve() if args.cluster_index else None
    base_rp = load_base_realised_path(cidx, base_id)
    if base_rp:
        print(f"[load]   base realised_path: {len(base_rp)} days, "
              f"mean={np.mean(base_rp):+.4f}")

    # Ridge plot
    print(f"[plot] building ridge plot")
    fig = build_ridge(retrieval, kdes, weights, grid, base_realised_path=base_rp)
    out_html = out_dir / f"ridge_{base_id}.html"
    fig.write_html(str(out_html), include_plotlyjs="cdn",
                   full_html=True, auto_open=False)
    print(f"[save] {out_html}")

    # Brief stdout summary
    print(f"\n--- Summary ---")
    print(f"  base cluster:       {base_id}")
    print(f"  analogues used:     {n_valid}/{len(analogues)}")
    print(f"  horizons forecast:  {horizons}")
    if base_rp:
        rp_arr = np.array(base_rp)
        for h in horizons:
            if rp_arr.size >= h:
                cum = float(rp_arr[:h].sum())
                fc  = forecast_summary["horizons"][f"h{h}"]
                # Where does the realised cum sit in the prior?
                # Use mean+/-std as a quick signal
                z = (cum - fc["mean"]) / max(fc["std"], 1e-12)
                print(f"  h={h:>2}  realised cum_logret={cum:+.4f}  "
                      f"prior mean={fc['mean']:+.4f}  z={z:+.2f}")


if __name__ == "__main__":
    main()
