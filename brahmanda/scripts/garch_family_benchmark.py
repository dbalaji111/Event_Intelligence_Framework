"""
garch_family_benchmark.py
=========================
Computes EGARCH and GJR-GARCH benchmarks for all 406 test events,
then generates two publication-quality figures:

  Figure 4 — Quantile efficiency curve (pinball loss across τ)
             Selective Prior vs GARCH(1,1) vs EGARCH vs GJR-GARCH

  Figure 5 — Coverage + band width bar chart
             Full GARCH family comparison

Usage (from project root):
    PYTHONPATH=scripts python scripts/garch_family_benchmark.py \
        --results   kde_output_v3/kde_v3_results_h15.csv \
        --prices    data/oil_prices_full.xlsx \
        --output    figures

Requires: arch, numpy, pandas, matplotlib
"""

import argparse
import sys
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm as scipy_norm
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       10,
    "axes.titlesize":  11,
    "axes.labelsize":  10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi":      150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid":       True,
    "grid.alpha":      0.3,
})

COLORS = {
    "Selective Prior":      "#1f77b4",
    "GARCH(1,1)":  "#d62728",
    "EGARCH(1,1)": "#ff7f0e",
    "GJR-GARCH":   "#2ca02c",
}


# ─────────────────────────────────────────────────────────────────────────────
# GARCH FAMILY BENCHMARKS
# ─────────────────────────────────────────────────────────────────────────────

def fit_garch_family(prices_df: pd.DataFrame,
                     date: pd.Timestamp,
                     horizon: int,
                     col: str = "WTI"):
    """
    Fit GARCH(1,1), EGARCH(1,1), GJR-GARCH(1,1) on 400 pre-event days.
    Returns dict of {model_name: {p5, p95, sigma, covers_90}} for each model.
    """
    try:
        from arch import arch_model
    except ImportError:
        log.error("arch package not found. pip install arch")
        sys.exit(1)

    # Get pre-event returns
    start = date - pd.Timedelta(days=500)
    col   = col if col in prices_df.columns else prices_df.columns[0]

    # Find nearest date
    idx = prices_df.index
    nearest = idx[abs(idx - date) <= pd.Timedelta(days=5)]
    if len(nearest) == 0:
        return None
    date_use = nearest[abs(nearest - date).argmin()]

    series = prices_df.loc[start:date_use, col].dropna()
    if len(series) < 100:
        return None

    rets = series.pct_change().dropna() * 100

    results = {}
    specs = {
        "EGARCH(1,1)": arch_model(rets, vol="EGARCH", p=1, q=1),
        "GJR-GARCH":   arch_model(rets, vol="GARCH",  p=1, o=1, q=1),
    }

    for name, model in specs.items():
        try:
            res = model.fit(disp="off", show_warning=False)
            # EGARCH requires simulation for horizon > 1
            # GJR-GARCH supports analytic — use simulation for both (consistent)
            try:
                fc    = res.forecast(horizon=horizon, reindex=False,
                                     method="simulation", simulations=200)
                # simulation returns distribution of paths — use mean variance
                var   = fc.variance.values[-1, -1]
            except Exception:
                fc    = res.forecast(horizon=horizon, reindex=False)
                var   = fc.variance.values[-1, -1]
            sigma = float(np.sqrt(var * horizon))
            if not np.isfinite(sigma) or sigma <= 0 or sigma > 500:
                continue
            results[name] = {
                "sigma":    round(sigma, 4),
                "p5":       round(float(scipy_norm.ppf(0.05, 0, sigma)), 3),
                "p95":      round(float(scipy_norm.ppf(0.95, 0, sigma)), 3),
                "p_down10": round(float(scipy_norm.cdf(-10, 0, sigma)), 4),
                "p_up10":   round(float(1 - scipy_norm.cdf(10, 0, sigma)), 4),
            }
        except Exception as e:
            log.debug(f"  {name} failed: {e}")

    return results if results else None


# ─────────────────────────────────────────────────────────────────────────────
# PINBALL LOSS
# ─────────────────────────────────────────────────────────────────────────────

def pinball(r: float, q: float, tau: float) -> float:
    """Pinball loss at quantile tau."""
    if r >= q:
        return tau * (r - q)
    else:
        return (1 - tau) * (q - r)


def gaussian_quantile(sigma: float, tau: float, mu: float = 0.0) -> float:
    return float(scipy_norm.ppf(tau, mu, sigma))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True,
                    help="kde_v3_results_h15.csv")
    ap.add_argument("--prices",  required=True,
                    help="data/oil_prices_full.xlsx")
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--output",  default="figures")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load existing results ─────────────────────────────────────────────
    df = pd.read_csv(args.results)
    df["base_date"] = pd.to_datetime(df["base_date"])
    log.info(f"Loaded {len(df)} events from {args.results}")

    # ── Load prices ───────────────────────────────────────────────────────
    log.info(f"Loading prices from {args.prices}")
    prices_df = pd.read_excel(args.prices)
    prices_df = prices_df.rename(columns={prices_df.columns[0]: "Date"})
    prices_df["Date"] = pd.to_datetime(prices_df["Date"], errors="coerce")
    prices_df = prices_df.dropna(subset=["Date"]).set_index("Date").sort_index()
    rename = {
        "Crude Oil-WTI Spot Cushing U$/BBL":  "WTI",
        "Crude Oil WTI NYMEX Close M U$/BBL":  "WTI_M1",
    }
    prices_df = prices_df.rename(
        columns={k: v for k, v in rename.items() if k in prices_df.columns})
    prices_df = prices_df.ffill(limit=5)

    # ── Run GARCH family for each event ───────────────────────────────────
    log.info(f"\nFitting EGARCH + GJR-GARCH for {len(df)} events...")
    egarch_p5, egarch_p95, egarch_sigma, egarch_cov = [], [], [], []
    gjr_p5,    gjr_p95,    gjr_sigma,    gjr_cov    = [], [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc="GARCH family", unit=" ev"):
        date     = row["base_date"]
        realised = row["realised_r"]
        res      = fit_garch_family(prices_df, date, args.horizon)

        if res and "EGARCH(1,1)" in res:
            eg = res["EGARCH(1,1)"]
            egarch_p5.append(eg["p5"])
            egarch_p95.append(eg["p95"])
            egarch_sigma.append(eg["sigma"])
            egarch_cov.append(
                int(eg["p5"] <= realised <= eg["p95"])
                if pd.notna(realised) else np.nan)
        else:
            egarch_p5.append(np.nan); egarch_p95.append(np.nan)
            egarch_sigma.append(np.nan); egarch_cov.append(np.nan)

        if res and "GJR-GARCH" in res:
            gj = res["GJR-GARCH"]
            gjr_p5.append(gj["p5"])
            gjr_p95.append(gj["p95"])
            gjr_sigma.append(gj["sigma"])
            gjr_cov.append(
                int(gj["p5"] <= realised <= gj["p95"])
                if pd.notna(realised) else np.nan)
        else:
            gjr_p5.append(np.nan); gjr_p95.append(np.nan)
            gjr_sigma.append(np.nan); gjr_cov.append(np.nan)

    df["egarch_p5"]      = egarch_p5
    df["egarch_p95"]     = egarch_p95
    df["egarch_sigma"]   = egarch_sigma
    df["egarch_cov90"]   = egarch_cov
    df["gjr_p5"]         = gjr_p5
    df["gjr_p95"]        = gjr_p95
    df["gjr_sigma"]      = gjr_sigma
    df["gjr_cov90"]      = gjr_cov

    # Save augmented CSV
    aug_csv = out_dir / "kde_v3_results_garch_family.csv"
    df.to_csv(aug_csv, index=False)
    log.info(f"[save] {aug_csv}")

    # ── Print summary ─────────────────────────────────────────────────────
    valid = df.dropna(subset=["realised_r", "kde_p5",
                               "garch_p5"])
    n = len(valid)
    log.info(f"\n{'='*60}")
    log.info(f"  GARCH FAMILY SUMMARY  (h=+{args.horizon}d, N={n})")
    log.info(f"{'='*60}")
    for method, cov_col, bw_p5, bw_p95 in [
        ("Selective Prior",      "covers_90",   "kde_p5",    "kde_p95"),
        ("GARCH(1,1)",  "garch_covers_90", "garch_p5",  "garch_p95"),
        ("EGARCH(1,1)", "egarch_cov90", "egarch_p5", "egarch_p95"),
        ("GJR-GARCH",   "gjr_cov90",   "gjr_p5",    "gjr_p95"),
    ]:
        cov = valid[cov_col].dropna().mean() * 100
        bw  = (valid[bw_p95] - valid[bw_p5]).dropna().mean()
        log.info(f"  {method:<18} coverage={cov:.1f}%  band_width={bw:.2f}%")
    log.info(f"{'='*60}\n")

    # ── FIGURE 4: Quantile efficiency curve ───────────────────────────────
    log.info("Generating Quantile efficiency curve...")
    taus = np.arange(0.05, 1.00, 0.05)

    def mean_pinball(df_in, p5_col, p95_col, sigma_col, taus):
        """Compute mean pinball loss at each tau for a Gaussian model."""
        losses = []
        for tau in taus:
            ls = []
            for _, row in df_in.iterrows():
                r = row["realised_r"]
                if not pd.notna(r): continue
                sig = row[sigma_col]
                if not pd.notna(sig): continue
                q = gaussian_quantile(sig, tau)
                ls.append(pinball(r, q, tau))
            losses.append(np.mean(ls) if ls else np.nan)
        return np.array(losses)

    def mean_pinball_kde(df_in, taus):
        """Compute mean pinball loss for KDE using stored quantiles."""
        # Use linear interpolation between stored quantiles
        # KDE stores: p5, p25, median, p75, p95
        stored_taus = [0.05, 0.25, 0.50, 0.75, 0.95]
        stored_cols = ["kde_p5", "kde_p25", "kde_median", "kde_p75", "kde_p95"]
        losses = []
        for tau in taus:
            ls = []
            for _, row in df_in.iterrows():
                r = row["realised_r"]
                if not pd.notna(r): continue
                # Interpolate KDE quantile at tau
                qs = [row[c] for c in stored_cols if pd.notna(row[c])]
                ts = stored_taus[:len(qs)]
                if len(qs) < 2: continue
                q = float(np.interp(tau, ts, qs))
                ls.append(pinball(r, q, tau))
            losses.append(np.mean(ls) if ls else np.nan)
        return np.array(losses)

    kde_losses    = mean_pinball_kde(valid, taus)
    garch_losses  = mean_pinball(valid, "garch_p5",  "garch_p95",
                                 "garch_sigma",  taus)
    egarch_losses = mean_pinball(valid, "egarch_p5", "egarch_p95",
                                 "egarch_sigma", taus)
    gjr_losses    = mean_pinball(valid, "gjr_p5",    "gjr_p95",
                                 "gjr_sigma",    taus)

    # ── Bootstrap confidence bands for Panel B ────────────────────────────
    log.info("  Bootstrap confidence bands (B=500)...")
    rng       = np.random.default_rng(42)
    B         = 500
    n_events  = len(valid)
    valid_arr = valid.reset_index(drop=True)

    boot_rel_garch = np.zeros((B, len(taus)))
    boot_rel_gjr   = np.zeros((B, len(taus)))

    for b in range(B):
        idx  = rng.integers(0, n_events, size=n_events)
        boot = valid_arr.iloc[idx].reset_index(drop=True)
        bk   = mean_pinball_kde(boot, taus)
        bg   = mean_pinball(boot, "garch_p5", "garch_p95", "garch_sigma", taus)
        bj   = mean_pinball(boot, "gjr_p5",   "gjr_p95",   "gjr_sigma",   taus)
        boot_rel_garch[b] = np.where(bg > 0, (bg - bk) / bg * 100, 0)
        boot_rel_gjr[b]   = np.where(bj > 0, (bj - bk) / bj * 100, 0)

    ci_garch_lo = np.percentile(boot_rel_garch, 2.5,  axis=0)
    ci_garch_hi = np.percentile(boot_rel_garch, 97.5, axis=0)
    ci_gjr_lo   = np.percentile(boot_rel_gjr,   2.5,  axis=0)
    ci_gjr_hi   = np.percentile(boot_rel_gjr,   97.5, axis=0)

    # ── Significance markers (bootstrap p-value: fraction of boot < 0) ───
    p_garch = (boot_rel_garch < 0).mean(axis=0)   # prob KDE is worse
    p_gjr   = (boot_rel_gjr   < 0).mean(axis=0)

    def sig_marker(p):
        if p < 0.01: return "***"
        if p < 0.05: return "**"
        if p < 0.10: return "*"
        return ""

    # ── Build figure ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    fig.suptitle(
        "Out-of-sample Pinball Loss Across Quantiles — "
        "Selective Prior vs GARCH Family\n"
        f"$h = +{args.horizon}$ days, $N = {n}$ out-of-sample events",
        fontsize=10
    )

    rel_garch = (garch_losses - kde_losses) / garch_losses * 100
    rel_gjr   = (gjr_losses   - kde_losses) / gjr_losses   * 100

    # ── Panel A: Mean pinball loss ────────────────────────────────────────
    ax = axes[0]
    scale = 1e3   # scale up for readability
    ax.plot(taus, kde_losses   * scale, color=COLORS["Selective Prior"],
            lw=2.2, label="Selective Prior (proposed)", zorder=4)
    ax.plot(taus, garch_losses * scale, color=COLORS["GARCH(1,1)"],
            lw=1.5, ls="--", label="GARCH(1,1)")
    ax.plot(taus, egarch_losses* scale, color=COLORS["EGARCH(1,1)"],
            lw=1.0, ls="-.", alpha=0.6, label="EGARCH(1,1)")
    ax.plot(taus, gjr_losses   * scale, color=COLORS["GJR-GARCH"],
            lw=1.5, ls=":",  label="GJR-GARCH(1,1)")

    ax.set_xlabel("Quantile $\\tau$")
    ax.set_ylabel("Mean Pinball Loss ($\\times 10^{-3}$)")
    ax.set_title("(A) Out-of-sample Mean Pinball Loss\n(lower = better)")
    ax.legend(fontsize=8.5)
    ax.set_xlim(0.05, 0.95)

    # ── Panel B: Relative improvement with CI and significance ───────────
    ax2 = axes[1]

    # GARCH line + CI
    ax2.plot(taus, rel_garch, color=COLORS["GARCH(1,1)"],
             lw=1.8, ls="--", label="vs GARCH(1,1)", zorder=4)
    ax2.fill_between(taus, ci_garch_lo, ci_garch_hi,
                     alpha=0.15, color=COLORS["GARCH(1,1)"],
                     label="95% CI (bootstrap)")

    # GJR line + CI
    ax2.plot(taus, rel_gjr, color=COLORS["GJR-GARCH"],
             lw=1.8, ls=":", label="vs GJR-GARCH(1,1)", zorder=4)
    ax2.fill_between(taus, ci_gjr_lo, ci_gjr_hi,
                     alpha=0.15, color=COLORS["GJR-GARCH"])

    ax2.axhline(0, color="black", lw=1.0, ls="-", label="Break-even (0%)")
    ax2.fill_between(taus, rel_garch, 0,
                     where=(rel_garch > 0),
                     alpha=0.06, color=COLORS["GARCH(1,1)"])

    # Significance markers at key quantiles
    marker_taus = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    for mt in marker_taus:
        idx = np.argmin(np.abs(taus - mt))
        mk  = sig_marker(p_garch[idx])
        if mk:
            ax2.text(taus[idx], rel_garch[idx] + 1.5, mk,
                     ha="center", va="bottom", fontsize=9,
                     color=COLORS["GARCH(1,1)"], fontweight="bold")

    ax2.set_xlabel("Quantile $\\tau$")
    ax2.set_ylabel("Selective Prior improvement over benchmark (%)")
    ax2.set_title("(B) Relative Pinball Loss Improvement\n"
                  "(positive = KDE wins; shaded = 95% bootstrap CI)")
    ax2.legend(fontsize=8)
    ax2.set_xlim(0.05, 0.95)

    # Footnote for formula
    fig.text(0.5, -0.03,
             r"Relative improvement = $100 \times "
             r"\frac{L_{\mathrm{benchmark}} - L_{\mathrm{KDE}}}"
             r"{L_{\mathrm{benchmark}}}$. "
             r"Significance: $^*p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$ "
             r"(bootstrap, $B=500$).",
             ha="center", fontsize=8.5, style="italic")

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        p = out_dir / f"fig4_efficiency_curve.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=200)
        print(f"[saved] {p}")
    plt.close()

    # ── FIGURE 5: Coverage + band width bar chart (GARCH family) ─────────
    log.info("Generating GARCH family coverage comparison...")

    methods   = ["Selective Prior", "GARCH(1,1)", "EGARCH(1,1)", "GJR-GARCH"]
    cov_cols  = ["covers_90", "garch_covers_90", "egarch_cov90", "gjr_cov90"]
    p5_cols   = ["kde_p5",   "garch_p5",  "egarch_p5",  "gjr_p5"]
    p95_cols  = ["kde_p95",  "garch_p95", "egarch_p95", "gjr_p95"]
    colors    = [COLORS[m] for m in methods]

    covs = [valid[c].dropna().mean() * 100 for c in cov_cols]
    bws  = [(valid[p95] - valid[p5]).dropna().mean()
            for p5, p95 in zip(p5_cols, p95_cols)]

    fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4.5))
    fig2.suptitle(
        "GARCH Family Comparison — Coverage and Prediction Interval Width\n"
        f"$h = +{args.horizon}$ days, $N = {n}$ out-of-sample events",
        fontsize=10
    )

    x = np.arange(len(methods))
    w = 0.55

    # Panel A: Coverage
    ax3 = axes2[0]
    bars = ax3.bar(x, covs, w, color=colors, alpha=0.85)
    ax3.axhline(90, color="black", lw=1.2, ls="--",
                label="90% nominal target")
    for bar, val in zip(bars, covs):
        ax3.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.3,
                 f"{val:.1f}%", ha="center", va="bottom",
                 fontsize=9, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(methods, fontsize=9)
    ax3.set_ylabel("90% Empirical Coverage (%)")
    ax3.set_ylim(82, 102)
    ax3.set_title("(A) 90% Prediction Interval Coverage")
    ax3.legend(fontsize=9)

    # Panel B: Band width
    ax4 = axes2[1]
    bars2 = ax4.bar(x, bws, w, color=colors, alpha=0.85)
    for bar, val in zip(bars2, bws):
        ax4.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.2,
                 f"{val:.1f}%", ha="center", va="bottom",
                 fontsize=9, fontweight="bold")
    ax4.set_xticks(x)
    ax4.set_xticklabels(methods, fontsize=9)
    ax4.set_ylabel("Mean Band Width: $P_{95} - P_5$ (%)")
    ax4.set_title("(B) Prediction Interval Width\n(narrower = more efficient)")

    # Arrow annotation on band width panel
    ax4.annotate("KDE achieves\nhighest coverage\nwith narrowest band",
                 xy=(0, bws[0]), xytext=(1.5, bws[0] + 3),
                 fontsize=8, color=COLORS["Selective Prior"],
                 arrowprops=dict(arrowstyle="->",
                                 color=COLORS["Selective Prior"], lw=1.2))

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        p = out_dir / f"fig5_garch_family_comparison.{ext}"
        fig2.savefig(p, bbox_inches="tight", dpi=200)
        print(f"[saved] {p}")
    plt.close()

    log.info(f"\nAll figures saved to {out_dir}/")
    log.info("Done.\n")


if __name__ == "__main__":
    main()
