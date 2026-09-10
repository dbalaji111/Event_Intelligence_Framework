"""
frl_robustness_v3.py
======================
Addresses reviewer points 3 (marginal effects), 4 (multiple extreme
thresholds), 5 (multiple horizons), and 6 (clustered standard
errors) for the FRL bimodal-signal paper.

Requires h7, h15, h30 kde_results files, all with kde_is_bimodal
and kde_std columns.

Usage:
    python frl_robustness_v3.py \
        --kde_h7  results_run/kde_output_v4/kde_v4_results_h7.csv \
        --kde_h15 results_run/kde_output_v4/kde_v4_results_h15.csv \
        --kde_h30 results_run/kde_output_v4/kde_v4_results_h30.csv \
        --output frl_v4
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def run_logit_with_margins(df, extreme_threshold, cluster_col=None):
    """Fit dispersion-only and +bimodal models, return coefficients,
    odds ratios, AME, and optionally cluster-robust SEs."""
    df = df.copy()
    df["extreme"] = (df["realised_r"].abs() > extreme_threshold).astype(int)

    X2 = sm.add_constant(df[["kde_std", "kde_is_bimodal"]])
    y = df["extreme"]

    if cluster_col is not None and cluster_col in df.columns:
        m2 = sm.Logit(y, X2).fit(disp=0,
                                  cov_type="cluster",
                                  cov_kwds={"groups": df[cluster_col]})
        se_note = f"cluster-robust (by {cluster_col})"
    else:
        m2 = sm.Logit(y, X2).fit(disp=0)
        se_note = "standard (non-clustered)"

    beta1 = m2.params["kde_is_bimodal"]
    se1 = m2.bse["kde_is_bimodal"]
    p1 = m2.pvalues["kde_is_bimodal"]
    odds_ratio = np.exp(beta1)
    or_ci = (np.exp(beta1 - 1.96 * se1), np.exp(beta1 + 1.96 * se1))

    # Average marginal effect for the bimodal flag
    margeff = m2.get_margeff(at="overall")
    ame = margeff.margeff[list(X2.columns[1:]).index("kde_is_bimodal")]
    ame_p = margeff.pvalues[list(X2.columns[1:]).index("kde_is_bimodal")]

    return {
        "threshold": extreme_threshold,
        "n": len(df),
        "n_extreme": int(df["extreme"].sum()),
        "beta_bimodal": round(beta1, 4),
        "se_type": se_note,
        "p_value": round(p1, 4),
        "odds_ratio": round(odds_ratio, 4),
        "or_ci_low": round(or_ci[0], 4),
        "or_ci_high": round(or_ci[1], 4),
        "ame": round(ame, 4),
        "ame_p": round(ame_p, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kde_h7", required=True)
    ap.add_argument("--kde_h15", required=True)
    ap.add_argument("--kde_h30", required=True)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[5.0, 10.0, 15.0, 20.0])
    ap.add_argument("--cluster_col", default=None,
                     help="Column to cluster SEs on, e.g. a week/episode "
                          "identifier if present in kde_results. Omit to "
                          "use non-clustered SEs.")
    ap.add_argument("--output", default="frl_v4")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    horizons = {"h7": args.kde_h7, "h15": args.kde_h15, "h30": args.kde_h30}
    all_results = []

    for h_label, path in horizons.items():
        df = pd.read_csv(path).dropna(subset=["kde_is_bimodal", "kde_std", "realised_r"])
        print(f"\n{'='*70}\nHORIZON: {h_label}  (N={len(df)})\n{'='*70}")
        for thresh in args.thresholds:
            r = run_logit_with_margins(df, thresh, args.cluster_col)
            r["horizon"] = h_label
            all_results.append(r)
            print(f"  threshold={thresh}%: beta={r['beta_bimodal']}, "
                  f"OR={r['odds_ratio']} [{r['or_ci_low']}, {r['or_ci_high']}], "
                  f"p={r['p_value']}, AME={r['ame']} (p={r['ame_p']}), "
                  f"n_extreme={r['n_extreme']}, SE={r['se_type']}")

    out_df = pd.DataFrame(all_results)
    out_df.to_csv(out_dir / "robustness_table.csv", index=False)
    print(f"\n[saved] {out_dir / 'robustness_table.csv'}")
    print("\nFull robustness table:")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
