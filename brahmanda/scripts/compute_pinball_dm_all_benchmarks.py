"""
compute_pinball_dm_all_benchmarks.py
======================================
Extends validate_framework.py's Diebold-Mariano pinball-loss test
(currently hardcoded to garch_p5/garch_p95, i.e. GARCH(1,1) only) to
all three GARCH-family benchmarks, using the same pinball_loss and
diebold_mariano functions, applied identically to each.

Reads from the garch_family output CSV (produced by
garch_family_benchmark.py), which carries kde_p5/kde_p95/realised_r
plus egarch_p5/egarch_p95/gjr_p5/gjr_p95 alongside the original
garch_p5/garch_p95 -- so this is the correct single source for all
three comparisons, not the base kde_v4_results file which only has
GARCH(1,1).

Usage:
    python compute_pinball_dm_all_benchmarks.py \
        --results ..\\results_run\\garch_family_v4_final\\kde_v3_results_garch_family.csv \
        --output ..\\results_run\\pinball_dm_v4
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def pinball_loss(tau: float, realised: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    """Verbatim from validate_framework.py."""
    errors = realised - forecast
    return np.where(errors >= 0, tau * errors, (tau - 1) * errors)


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray):
    """Verbatim from validate_framework.py."""
    d = loss_a - loss_b
    d = d[np.isfinite(d)]
    if len(d) < 10:
        return np.nan, np.nan
    d_bar = np.mean(d)
    d_var = np.var(d, ddof=1)
    if d_var <= 0:
        return np.nan, np.nan
    dm_stat = d_bar / np.sqrt(d_var / len(d))
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return float(dm_stat), float(p_value)


BENCHMARKS = {
    "GARCH(1,1)": ("garch_p5", "garch_p95"),
    "EGARCH(1,1)": ("egarch_p5", "egarch_p95"),
    "GJR-GARCH": ("gjr_p5", "gjr_p95"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True,
                     help="garch_family output CSV with kde_p5/kde_p95/"
                          "realised_r plus all three benchmarks' p5/p95")
    ap.add_argument("--output", default=".")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.results)
    print(f"Loaded {len(df)} events from {args.results}")
    print(f"Columns available: {df.columns.tolist()}\n")

    required_base = {"kde_p5", "kde_p95", "realised_r"}
    if not required_base.issubset(df.columns):
        missing = required_base - set(df.columns)
        print(f"ERROR: missing required base columns: {missing}")
        return

    rows = []
    for name, (p5_col, p95_col) in BENCHMARKS.items():
        if p5_col not in df.columns or p95_col not in df.columns:
            print(f"SKIP {name}: columns {p5_col}/{p95_col} not found in this file")
            continue

        sub = df.dropna(subset=["kde_p5", "kde_p95", "realised_r", p5_col, p95_col])
        n = len(sub)
        if n < 10:
            print(f"SKIP {name}: only {n} valid events, too few for a DM test")
            continue

        realised = sub["realised_r"].values

        kde_loss_p5 = pinball_loss(0.05, realised, sub["kde_p5"].values)
        bench_loss_p5 = pinball_loss(0.05, realised, sub[p5_col].values)
        kde_loss_p95 = pinball_loss(0.95, realised, sub["kde_p95"].values)
        bench_loss_p95 = pinball_loss(0.95, realised, sub[p95_col].values)

        dm_p5, p_p5 = diebold_mariano(kde_loss_p5, bench_loss_p5)
        dm_p95, p_p95 = diebold_mariano(kde_loss_p95, bench_loss_p95)

        print(f"{name} (N={n}):")
        print(f"  tau=0.05  KDE mean={np.nanmean(kde_loss_p5):.4f}  "
              f"{name} mean={np.nanmean(bench_loss_p5):.4f}  "
              f"DM={dm_p5:.3f}  p={p_p5:.4f}")
        print(f"  tau=0.95  KDE mean={np.nanmean(kde_loss_p95):.4f}  "
              f"{name} mean={np.nanmean(bench_loss_p95):.4f}  "
              f"DM={dm_p95:.3f}  p={p_p95:.4f}\n")

        rows.append({
            "benchmark": name, "n": n,
            "dm_tau05": round(dm_p5, 3), "p_tau05": round(p_p5, 4),
            "dm_tau95": round(dm_p95, 3), "p_tau95": round(p_p95, 4),
        })

    out_df = pd.DataFrame(rows)
    out_path = out_dir / "pinball_dm_all_benchmarks.csv"
    out_df.to_csv(out_path, index=False)
    print(f"[saved] {out_path}")
    print("\n" + out_df.to_string(index=False))


if __name__ == "__main__":
    main()
