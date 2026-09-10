"""
compute_fei_by_event_type.py
============================
Computes Forecast Efficacy Index (FEI) by event type for all
three GARCH benchmarks (GARCH, EGARCH, GJR-GARCH).

FEI = divergence(benchmark, actual) / divergence(KDE, actual)
    > 1  KDE wins (benchmark is further from actual)
    < 1  Benchmark wins

Three divergence metrics: KL, JS, Wasserstein
Four event types (from 2024-2026 corpus):
    Geopolitical News, Macroeconomic News, Price Movement, Supply Shocks

Usage:
    python scripts/compute_fei_by_event_type.py \
        --kde_dir  kde_output_v3/per_event \
        --results  figures/kde_v3_results_garch_family.csv \
        --metadata memory_2024_2026/df_major_metadata_2024_2026.csv \
        --output   figures
"""

import argparse, json, sys, warnings, logging
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm as scipy_norm
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

# ── Event type mapping ────────────────────────────────────────────────────────
EVENT_TYPE_MAP = {
    "geopolitical-tension": "Geopolitical News",
    "crisis":               "Geopolitical News",
    "trade-tensions":       "Geopolitical News",
    "embargo":              "Geopolitical News",
    "grow-strong":          "Macroeconomic News",
    "slow-weak":            "Macroeconomic News",
    "movement-up-gain":           "Price Movement",
    "movement-down-loss":         "Price Movement",
    "movement-flat":              "Price Movement",
    "cause-movement-up-gain":     "Price Movement",
    "cause-movement-down-loss":   "Price Movement",
    "position-high":              "Price Movement",
    "negative-sentiment":         "Price Movement",
    "oversupply":           "Supply Shocks",
    "nan":                  None,   # excluded
}

EVENT_TYPES = [
    "Geopolitical News",
    "Macroeconomic News",
    "Price Movement",
    "Supply Shocks",
]

BENCHMARKS = [
    ("GARCH(1,1)",  "garch_sigma"),
    ("EGARCH(1,1)", "egarch_sigma"),
    ("GJR-GARCH",   "gjr_sigma"),
]

# ── Divergence functions ──────────────────────────────────────────────────────

def gaussian_density(x, mu, sigma):
    d = scipy_norm.pdf(x, mu, sigma)
    d /= (np.trapz(d, x) + 1e-10)
    return d

def kl_div(p, q, x):
    """KL(p || q)"""
    p = np.clip(p, 1e-10, None)
    q = np.clip(q, 1e-10, None)
    return float(np.trapz(p * np.log(p / q), x))

def js_div(p, q, x):
    """Jensen-Shannon divergence"""
    p = np.clip(p / (np.trapz(p, x) + 1e-10), 1e-10, None)
    q = np.clip(q / (np.trapz(q, x) + 1e-10), 1e-10, None)
    m = 0.5 * (p + q)
    return float(0.5 * np.trapz(p * np.log(p / m), x) +
                 0.5 * np.trapz(q * np.log(q / m), x))

def wasserstein(p, q, x):
    """Wasserstein-1 distance via CDF integral"""
    dx = x[1] - x[0]
    cdf_p = np.clip(np.cumsum(p) * dx, 0, 1)
    cdf_q = np.clip(np.cumsum(q) * dx, 0, 1)
    return float(np.trapz(np.abs(cdf_p - cdf_q), x))

def compute_fei_all_benchmarks(
    x, kde_density, realised,
    garch_sigma, egarch_sigma, gjr_sigma
):
    """
    Compute FEI for all three benchmarks vs KDE.
    Reference distribution = narrow Gaussian at realised return.
    FEI = divergence(benchmark, ref) / divergence(KDE, ref)
    """
    sigma_ref = max(abs(realised) * 0.05, 0.2)
    ref = gaussian_density(x, realised, sigma_ref)

    kde_norm = kde_density / (np.trapz(kde_density, x) + 1e-10)

    # KDE divergences from actual
    kde_kl   = kl_div(kde_norm, ref, x)
    kde_js   = js_div(kde_norm, ref, x)
    kde_wass = wasserstein(kde_norm, ref, x)

    results = {}
    for name, sigma in [
        ("garch",  garch_sigma),
        ("egarch", egarch_sigma),
        ("gjr",    gjr_sigma),
    ]:
        if not (pd.notna(sigma) and sigma > 0):
            results[name] = {"kl": np.nan, "js": np.nan, "wass": np.nan,
                             "wins_kl": np.nan, "wins_js": np.nan, "wins_wass": np.nan}
            continue

        g_dens = gaussian_density(x, 0, sigma)
        g_kl   = kl_div(g_dens, ref, x)
        g_js   = js_div(g_dens, ref, x)
        g_wass = wasserstein(g_dens, ref, x)

        fei_kl   = g_kl   / kde_kl   if kde_kl   > 1e-10 else np.nan
        fei_js   = g_js   / kde_js   if kde_js   > 1e-10 else np.nan
        fei_wass = g_wass / kde_wass if kde_wass > 1e-10 else np.nan

        results[name] = {
            "kl":       fei_kl,
            "js":       fei_js,
            "wass":     fei_wass,
            "wins_kl":   int(fei_kl   > 1) if np.isfinite(fei_kl)   else np.nan,
            "wins_js":   int(fei_js   > 1) if np.isfinite(fei_js)   else np.nan,
            "wins_wass": int(fei_wass > 1) if np.isfinite(fei_wass) else np.nan,
        }
    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kde_dir",  required=True)
    ap.add_argument("--results",  required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--output",   default="figures")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load results (has all garch sigma columns)
    results = pd.read_csv(args.results)
    log.info(f"Loaded {len(results)} results")

    # Load metadata and map event types
    meta = pd.read_csv(args.metadata)
    meta["event_type_mapped"] = meta["Event_type"].str.lower().map(
        lambda x: EVENT_TYPE_MAP.get(str(x).strip(), "Other")
        if pd.notna(x) else None
    )
    results = results.merge(
        meta[["event_id", "event_type_mapped", "Spread_M1_M2"]],
        on="event_id", how="left"
    )
    log.info(f"\nEvent type distribution:\n"
             f"{results['event_type_mapped'].value_counts()}")

    # ── Compute FEI per event ─────────────────────────────────────────────
    kde_dir = Path(args.kde_dir)
    rows = []

    for _, row in tqdm(results.iterrows(), total=len(results),
                       desc="FEI computation", unit=" ev"):
        event_id   = row["event_id"]
        realised   = row.get("realised_r")
        event_type = row.get("event_type_mapped")
        spread     = row.get("Spread_M1_M2", np.nan)

        # Skip excluded / unknown types
        if event_type is None or event_type == "Other":
            continue
        if not pd.notna(realised):
            continue

        json_path = kde_dir / f"{event_id}_kde_v4.json"
        if not json_path.exists():
            continue

        with json_path.open() as f:
            data = json.load(f)

        x_grid  = np.array(data.get("x_grid",  []))
        density = np.array(data.get("density", []))
        if len(x_grid) == 0:
            continue

        fei = compute_fei_all_benchmarks(
            x_grid, density, realised,
            row.get("garch_sigma"),
            row.get("egarch_sigma"),
            row.get("gjr_sigma"),
        )

        base = {
            "event_id":    event_id,
            "event_type":  event_type,
            "spread_m1m2": spread,
            "realised_r":  realised,
        }
        for bname, bvals in fei.items():
            for metric, val in bvals.items():
                base[f"{bname}_{metric}"] = val

        rows.append(base)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "fei_by_event_type.csv", index=False)
    log.info(f"\n[save] fei_by_event_type.csv  ({len(df)} events)")

    # ── Print summary table ───────────────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"  FEI BY EVENT TYPE — All GARCH Benchmarks  (h=+15d, N={len(df)})")
    print(f"  FEI > 1 = KDE wins  |  FEI < 1 = benchmark wins")
    print(f"{'='*90}")

    bench_labels = [("GARCH(1,1)", "garch"), ("EGARCH(1,1)", "egarch"), ("GJR-GARCH", "gjr")]
    metric_labels = [("KL", "kl"), ("JS", "js"), ("Wass", "wass")]

    for blabel, bkey in bench_labels:
        print(f"\n  vs {blabel}")
        print(f"  {'Metric':<8} " + "  ".join(f"{et:<22}" for et in EVENT_TYPES))
        print("  " + "-"*80)
        for mlabel, mkey in metric_labels:
            vals = []
            for et in EVENT_TYPES:
                sub = df[df["event_type"] == et]
                col = f"{bkey}_{mkey}"
                wcol = f"{bkey}_wins_{mkey}"
                if col not in sub.columns or len(sub) == 0:
                    vals.append("N/A")
                    continue
                mean_fei = sub[col].dropna().mean()
                win_pct  = sub[wcol].dropna().mean() * 100
                vals.append(f"{mean_fei:.3f} ({win_pct:.0f}%)")
            print(f"  {mlabel:<8} " + "  ".join(f"{v:<22}" for v in vals))

    print(f"\n{'='*90}\n")

    # ── LaTeX table ───────────────────────────────────────────────────────
    ET_SHORT = {
        "Geopolitical News": "\\makecell{Geopolitical\\\\News}",
        "Macroeconomic News": "\\makecell{Macro-\\\\economic}",
        "Price Movement": "\\makecell{Price\\\\Movement}",
        "Supply Shocks": "\\makecell{Supply\\\\Shocks}",
    }

    latex = []
    latex.append(r"\begin{table}[htbp]")
    latex.append(r"\centering")
    latex.append(r"\caption{Forecast Efficacy Index (FEI) by benchmark, divergence metric,")
    latex.append(r"and event type, $h = +15$ days, $N = 377$ out-of-sample events")
    latex.append(r"(29 events with ambiguous type excluded).")
    latex.append(r"FEI $= $ divergence(benchmark, actual) / divergence(KDE v3, actual);")
    latex.append(r"FEI $> 1$ indicates KDE v3 outperforms the benchmark.")
    latex.append(r"Percentages in parentheses: fraction of events where KDE v3 wins.}")
    latex.append(r"\label{tab:fei_event_type}")
    latex.append(r"\small")
    latex.append(r"\begin{tabular}{llcccc}")
    latex.append(r"\toprule")
    latex.append(r"Benchmark & Metric & " +
                 " & ".join(ET_SHORT[e] for e in EVENT_TYPES) + r" \\")
    latex.append(r"\midrule")

    for blabel, bkey in bench_labels:
        latex.append(r"\midrule")
        for i, (mlabel, mkey) in enumerate(metric_labels):
            cells = []
            for et in EVENT_TYPES:
                sub = df[df["event_type"] == et]
                col  = f"{bkey}_{mkey}"
                wcol = f"{bkey}_wins_{mkey}"
                if col not in sub.columns or len(sub) == 0:
                    cells.append("---")
                    continue
                mean_fei = sub[col].dropna().mean()
                win_pct  = sub[wcol].dropna().mean() * 100
                cell = f"{mean_fei:.3f} ({win_pct:.0f}\\%)"
                if mean_fei > 1:
                    cell = f"\\textbf{{{mean_fei:.3f}}} ({win_pct:.0f}\\%)"
                cells.append(cell)

            if i == 0:
                latex.append(f"  \\multirow{{3}}{{*}}{{{blabel}}} & {mlabel} & "
                             + " & ".join(cells) + r" \\")
            else:
                latex.append(f"   & {mlabel} & " + " & ".join(cells) + r" \\")

    latex.append(r"\bottomrule")
    latex.append(r"\multicolumn{6}{l}{\small \textbf{Bold}: FEI $> 1$ (KDE v3 outperforms benchmark).} \\")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    latex_str = "\n".join(latex)
    (out_dir / "fei_table_eventtype.tex").write_text(latex_str)
    log.info(f"[save] fei_table_eventtype.tex")
    print(latex_str)
    log.info("\nDone.\n")


if __name__ == "__main__":
    main()