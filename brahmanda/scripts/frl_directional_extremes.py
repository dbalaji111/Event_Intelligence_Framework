"""
frl_directional_extremes.py
==============================
Splits |r|>10% into two directionally distinct outcomes:
  ExtremeUp:   r_{t+h} > +10%
  ExtremeDown: r_{t+h} < -10%

Runs the same beyond-dispersion logistic regression (bimodal flag +
kde_std) separately for each direction, to test whether bimodality's
relationship with extreme moves is symmetric or direction-specific.

Usage:
    python frl_directional_extremes.py \
        --kde_results rolling_kde_combined_h15_primary.csv \
        --threshold 10.0
"""
import argparse
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def run_direction(df, outcome_col, label):
    print(f"\n{'='*70}")
    print(f"{label}")
    print(f"{'='*70}")
    n_pos = int(df[outcome_col].sum())
    print(f"N={len(df)}, events={n_pos} ({n_pos/len(df)*100:.1f}%)")

    X1 = sm.add_constant(df[["kde_std"]])
    y = df[outcome_col]
    m1 = sm.Logit(y, X1).fit(disp=0)

    X2 = sm.add_constant(df[["kde_std", "kde_is_bimodal"]])
    m2 = sm.Logit(y, X2).fit(disp=0)

    beta = m2.params["kde_is_bimodal"]
    se = m2.bse["kde_is_bimodal"]
    p = m2.pvalues["kde_is_bimodal"]
    or_val = np.exp(beta)
    or_ci = (np.exp(beta - 1.96*se), np.exp(beta + 1.96*se))

    lr_stat = 2 * (m2.llf - m1.llf)
    lr_p = stats.chi2.sf(lr_stat, df=1)

    print(f"Bimodal coefficient: {beta:.4f} (SE={se:.4f}, p={p:.4f})")
    print(f"Odds ratio: {or_val:.4f}  95% CI: [{or_ci[0]:.4f}, {or_ci[1]:.4f}]")
    print(f"LR test vs dispersion-only: chi2={lr_stat:.3f}, p={lr_p:.4f}")

    return {"direction": label, "n": len(df), "n_events": n_pos,
            "beta_bimodal": round(beta, 4), "se": round(se, 4), "p": round(p, 4),
            "odds_ratio": round(or_val, 4), "or_ci_low": round(or_ci[0], 4),
            "or_ci_high": round(or_ci[1], 4), "lr_p": round(lr_p, 6)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kde_results", required=True)
    ap.add_argument("--threshold", type=float, default=10.0)
    ap.add_argument("--output", default="directional_extremes_results.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.kde_results).dropna(
        subset=["kde_is_bimodal", "kde_std", "realised_r"])

    df["extreme_up"] = (df["realised_r"] > args.threshold).astype(int)
    df["extreme_down"] = (df["realised_r"] < -args.threshold).astype(int)
    df["extreme_either"] = (df["realised_r"].abs() > args.threshold).astype(int)

    print(f"Total N={len(df)}")
    print(f"Extreme UP (r > +{args.threshold}%):   {df['extreme_up'].sum()} events")
    print(f"Extreme DOWN (r < -{args.threshold}%):  {df['extreme_down'].sum()} events")
    print(f"Combined |r|>{args.threshold}% (for reference): {df['extreme_either'].sum()} events")

    results = []
    results.append(run_direction(df, "extreme_up", f"UPSIDE SHOCKS: r > +{args.threshold}%"))
    results.append(run_direction(df, "extreme_down", f"DOWNSIDE SHOCKS: r < -{args.threshold}%"))
    results.append(run_direction(df, "extreme_either", f"COMBINED (reference): |r| > {args.threshold}%"))

    out_df = pd.DataFrame(results)
    out_df.to_csv(args.output, index=False)

    print(f"\n{'='*70}")
    print("SUMMARY: is the bimodal effect symmetric across direction?")
    print(f"{'='*70}")
    print(out_df[["direction", "n_events", "beta_bimodal", "odds_ratio", "p"]].to_string(index=False))
    print(f"\n[saved] {args.output}")


if __name__ == "__main__":
    main()
