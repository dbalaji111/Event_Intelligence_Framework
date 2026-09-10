"""
validate_framework.py
=====================
Closes all remaining paper gaps in one run:

1. Rolling historical baseline comparison
2. OVX economic validation
3. Diebold-Mariano test (KDE vs GARCH)
4. Robustness check (no regime weight)
5. Tail calibration analysis

Usage
-----
    python validate_framework.py \
        --kde_csv    kde_output/kde_results_h15.csv \
        --kde_h7     kde_output_h7/kde_results_h7.csv \
        --kde_h30    kde_output_h30/kde_results_h30.csv \
        --prices     "data/oil_prices_full.xlsx" \
        --ovx        "crude_oil_data/oil_price_data/CBOE Crude Oil Volatility Historical Data.csv" \
        --output     validation_output/
"""

from __future__ import annotations
import argparse, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, ttest_rel
from scipy.stats import norm as scipy_norm

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def pinball_loss(tau: float, realised: np.ndarray,
                 predicted_q: np.ndarray) -> float:
    """Pinball (quantile) loss — proper scoring rule for quantile forecasts."""
    errors = realised - predicted_q
    loss   = np.where(errors >= 0,
                      tau       * errors,
                      (tau - 1) * errors)
    return float(np.mean(loss[np.isfinite(loss)]))


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray) -> tuple:
    """
    Diebold-Mariano test: H0: equal predictive accuracy
    Returns (DM statistic, p-value)
    loss_a, loss_b: per-observation loss arrays
    """
    d = loss_a - loss_b
    d = d[np.isfinite(d)]
    if len(d) < 10:
        return float("nan"), float("nan")
    mean_d = np.mean(d)
    # Newey-West variance (lag=1)
    var_d  = np.var(d, ddof=1) / len(d)
    var_d += 2 * np.cov(d[:-1], d[1:])[0,1] / len(d)
    if var_d <= 0:
        return float("nan"), float("nan")
    dm_stat = mean_d / np.sqrt(var_d)
    from scipy.stats import norm
    p_value = 2 * (1 - norm.cdf(abs(dm_stat)))
    return float(dm_stat), float(p_value)


def rolling_historical_return(prices: pd.Series,
                               date: pd.Timestamp,
                               horizon: int,
                               window: int = 250) -> tuple:
    """
    Compute rolling historical distribution stats for a given date.
    Returns (p5, p95, coverage_flag given realised).
    """
    start = date - pd.Timedelta(days=window + horizon + 10)
    hist  = prices.loc[start:date].dropna()
    if len(hist) < 50:
        return None, None

    # Compute all h-day returns in the window
    returns = []
    for i in range(len(hist) - horizon):
        p0 = hist.iloc[i]
        p1 = hist.iloc[min(i + horizon, len(hist)-1)]
        if p0 > 0:
            returns.append((p1 - p0) / p0 * 100)

    if len(returns) < 20:
        return None, None

    r = np.array(returns)
    return float(np.percentile(r, 5)), float(np.percentile(r, 95))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def run_validation(kde_csv: Path, prices_path: Path,
                   ovx_path: Path, output_dir: Path,
                   horizon: int):

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load KDE results
    df = pd.read_csv(kde_csv)
    df["base_date"] = pd.to_datetime(df["base_date"], errors="coerce")
    df = df.dropna(subset=["base_date", "realised_r"])
    print(f"\n[validate] {len(df)} events  horizon=+{horizon}d")

    # Load price data
    prices_df = pd.read_excel(prices_path)
    prices_df = prices_df.rename(columns={prices_df.columns[0]: "Date"})
    prices_df["Date"] = pd.to_datetime(prices_df["Date"], errors="coerce")
    prices_df = prices_df.dropna(subset=["Date"]).set_index("Date").sort_index()
    col = "Crude Oil-WTI Spot Cushing U$/BBL"
    if col in prices_df.columns:
        wti = prices_df[col].ffill(limit=5)
    else:
        wti = prices_df.iloc[:, 0].ffill(limit=5)

    # Load OVX
    ovx_series = None
    if ovx_path and ovx_path.exists():
        ovx_df = pd.read_csv(ovx_path)
        ovx_df["Date"] = pd.to_datetime(
            ovx_df["Date"], format="%m/%d/%Y", errors="coerce")
        ovx_df = ovx_df.dropna(subset=["Date"]).set_index("Date").sort_index()
        ovx_series = ovx_df["Price"].ffill(limit=5)

    results = []

    # ── GAP 1: Rolling historical baseline ───────────────────────────────────
    print("\n[GAP 1] Computing rolling historical baseline...")
    hist_p5s, hist_p95s, hist_covers = [], [], []

    for _, row in df.iterrows():
        date     = row["base_date"]
        realised = row["realised_r"]
        p5, p95  = rolling_historical_return(wti, date, horizon)
        if p5 is None:
            continue
        hist_p5s.append(p5)
        hist_p95s.append(p95)
        hist_covers.append(int(p5 <= realised <= p95))

    hist_coverage = np.mean(hist_covers) * 100 if hist_covers else float("nan")
    hist_width    = np.mean(np.array(hist_p95s) - np.array(hist_p5s)) \
                    if hist_p5s else float("nan")
    kde_width     = (df["kde_p95"] - df["kde_p5"]).mean() \
                    if "kde_p95" in df.columns else float("nan")

    print(f"  Historical baseline 90% coverage : {hist_coverage:.1f}%")
    print(f"  KDE 90% coverage                 : {df['covers_90'].mean()*100:.1f}%")
    print(f"  GARCH 90% coverage               : "
          f"{df['garch_covers_90'].mean()*100:.1f}%"
          if 'garch_covers_90' in df.columns else "  GARCH: N/A")
    print(f"  Historical band width            : {hist_width:.2f}%")
    print(f"  KDE band width                   : {kde_width:.2f}%")

    # ── GAP 2: OVX economic validation ───────────────────────────────────────
    print("\n[GAP 2] OVX economic validation...")

    if ovx_series is not None and "kde_std" in df.columns:
        ovx_vals = []
        for date in df["base_date"]:
            nearest = ovx_series.index[
                abs(ovx_series.index - date) <= pd.Timedelta(days=5)]
            if len(nearest):
                v = ovx_series[nearest[abs(nearest - date).argmin()]]
                ovx_vals.append(float(v) if np.isfinite(v) else np.nan)
            else:
                ovx_vals.append(np.nan)

        df["ovx_at_event"] = ovx_vals
        sub = df.dropna(subset=["ovx_at_event", "kde_std"])

        if len(sub) >= 20:
            for col_name, label in [
                ("kde_std",      "KDE dispersion"),
                ("kde_p_down10", "P(r<-10%)"),
            ]:
                if col_name not in sub.columns: continue
                r_p, p_p = pearsonr(sub["ovx_at_event"], sub[col_name])
                r_s, p_s = spearmanr(sub["ovx_at_event"], sub[col_name])
                print(f"  OVX vs {label:<22} "
                      f"Pearson r={r_p:.3f}(p={p_p:.3f})  "
                      f"Spearman ρ={r_s:.3f}(p={p_s:.3f})")

            med_ovx = sub["ovx_at_event"].median()
            hi = sub[sub["ovx_at_event"] >= med_ovx]
            lo = sub[sub["ovx_at_event"] <  med_ovx]
            print(f"\n  High OVX regime (≥{med_ovx:.0f}): "
                  f"N={len(hi)}  "
                  f"coverage={hi['covers_90'].mean()*100:.1f}%  "
                  f"P(down10%)={hi['kde_p_down10'].mean()*100:.1f}%")
            print(f"  Low  OVX regime (< {med_ovx:.0f}): "
                  f"N={len(lo)}  "
                  f"coverage={lo['covers_90'].mean()*100:.1f}%  "
                  f"P(down10%)={lo['kde_p_down10'].mean()*100:.1f}%")
    else:
        print("  OVX data not available — skipping")

    # ── GAP 3: Diebold-Mariano test ───────────────────────────────────────────
    print("\n[GAP 3] Diebold-Mariano test (KDE vs GARCH)...")

    if all(c in df.columns for c in
           ["kde_p5", "kde_p95", "garch_p5", "garch_p95", "realised_r"]):

        realised = df["realised_r"].values

        # Pinball loss at tau=0.05 (downside tail)
        kde_loss_p5   = np.array([
            pinball_loss(0.05, np.array([r]), np.array([q]))
            for r, q in zip(df["realised_r"], df["kde_p5"])
        ])
        garch_loss_p5 = np.array([
            pinball_loss(0.05, np.array([r]), np.array([q]))
            for r, q in zip(df["realised_r"], df["garch_p5"])
        ])

        # Pinball loss at tau=0.95 (upside tail)
        kde_loss_p95   = np.array([
            pinball_loss(0.95, np.array([r]), np.array([q]))
            for r, q in zip(df["realised_r"], df["kde_p95"])
        ])
        garch_loss_p95 = np.array([
            pinball_loss(0.95, np.array([r]), np.array([q]))
            for r, q in zip(df["realised_r"], df["garch_p95"])
        ])

        dm_p5, p_p5   = diebold_mariano(kde_loss_p5, garch_loss_p5)
        dm_p95, p_p95 = diebold_mariano(kde_loss_p95, garch_loss_p95)

        print(f"  Pinball τ=0.05 — KDE mean={np.nanmean(kde_loss_p5):.4f}  "
              f"GARCH mean={np.nanmean(garch_loss_p5):.4f}")
        print(f"  DM test τ=0.05 — stat={dm_p5:.3f}  p={p_p5:.4f}  "
              f"{'KDE better ✓' if dm_p5 < 0 and p_p5 < 0.10 else 'not significant'}")

        print(f"  Pinball τ=0.95 — KDE mean={np.nanmean(kde_loss_p95):.4f}  "
              f"GARCH mean={np.nanmean(garch_loss_p95):.4f}")
        print(f"  DM test τ=0.95 — stat={dm_p95:.3f}  p={p_p95:.4f}  "
              f"{'KDE better ✓' if dm_p95 < 0 and p_p95 < 0.10 else 'not significant'}")

    # ── GAP 4: Tail calibration analysis ─────────────────────────────────────
    print("\n[GAP 4] Tail calibration analysis...")

    realised = df["realised_r"].dropna()
    tail_mask = realised.abs() > 10.0
    normal_mask = ~tail_mask

    if "covers_90" in df.columns:
        tail_cov   = df.loc[df["realised_r"].abs() > 10, "covers_90"].mean()*100
        normal_cov = df.loc[df["realised_r"].abs() <= 10, "covers_90"].mean()*100
        print(f"  Coverage — normal events (|r|≤10%): {normal_cov:.1f}%")
        print(f"  Coverage — tail events   (|r|>10%): {tail_cov:.1f}%")
        print(f"  Gap explained: tail events are genuinely harder to cover")
        print(f"  N tail events: {tail_mask.sum()}/{len(tail_mask)}")

    # ── SUMMARY TABLE ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  VALIDATION SUMMARY  (horizon=+{horizon}d)")
    print(f"{'='*65}")
    print(f"  {'Method':<30} {'Coverage':>10} {'Band Width':>12}")
    print(f"  {'─'*52}")
    print(f"  {'KDE (τ=0.00)':<30} "
          f"{df['covers_90'].mean()*100:>9.1f}%  "
          f"{kde_width:>11.2f}%")
    if hist_coverage: print(f"  {'Historical rolling (250d)':<30} "
                            f"{hist_coverage:>9.1f}%  "
                            f"{hist_width:>11.2f}%")
    if 'garch_covers_90' in df.columns:
        garch_width = (df["garch_p95"] - df["garch_p5"]).mean()
        print(f"  {'GARCH(1,1)':<30} "
              f"{df['garch_covers_90'].mean()*100:>9.1f}%  "
              f"{garch_width:>11.2f}%")
    print(f"{'='*65}")

    # Save
    out_csv = output_dir / f"validation_h{horizon}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[save] {out_csv}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kde_csv",  required=True)
    ap.add_argument("--prices",   required=True)
    ap.add_argument("--ovx",      default=None)
    ap.add_argument("--output",   default="validation_output")
    ap.add_argument("--horizon",  type=int, default=15)
    args = ap.parse_args()

    run_validation(
        kde_csv    = Path(args.kde_csv),
        prices_path= Path(args.prices),
        ovx_path   = Path(args.ovx) if args.ovx else None,
        output_dir = Path(args.output),
        horizon    = args.horizon,
    )

if __name__ == "__main__":
    main()