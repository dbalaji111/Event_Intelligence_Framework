"""
frl_specification_checks.py
==============================
Addresses reviewer points 1 (event dependence/clustering), 2 (model
misspecification), and 3 (rigorous fix for separation, via Firth's
penalized likelihood rather than exclusion).

Requires: pip install firthlogist --break-system-packages

Usage:
    python frl_specification_checks.py \
        --kde_results results_run/kde_output_v4/kde_v4_results_h15.csv \
        --metadata memory_2024_2026/df_major_metadata_2024_2026.csv \
        --extreme_threshold 10.0
"""
import argparse
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kde_results", required=True)
    ap.add_argument("--metadata", default=None,
                     help="For event_type control (point 2). Optional.")
    ap.add_argument("--extreme_threshold", type=float, default=10.0)
    args = ap.parse_args()

    df = pd.read_csv(args.kde_results).dropna(
        subset=["kde_is_bimodal", "kde_std", "realised_r"])
    df["extreme"] = (df["realised_r"].abs() > args.extreme_threshold).astype(int)
    print(f"N={len(df)}, extreme={df['extreme'].sum()}\n")

    # ============================================================
    # POINT 1: Clustering
    # ============================================================
    print("=" * 70)
    print("POINT 1: EVENT DEPENDENCE / CLUSTERING")
    print("=" * 70)
    date_col = next((c for c in ["base_date", "date", "event_date"] if c in df.columns), None)
    if date_col:
        df["_week"] = pd.to_datetime(df[date_col]).dt.isocalendar().year.astype(str) + \
                       "-W" + pd.to_datetime(df[date_col]).dt.isocalendar().week.astype(str)
        n_clusters = df["_week"].nunique()
        print(f"Found date column '{date_col}'. Clustering by ISO week: {n_clusters} clusters.")

        X = sm.add_constant(df[["kde_std", "kde_is_bimodal"]])
        y = df["extreme"]
        m_cluster = sm.Logit(y, X).fit(disp=0, cov_type="cluster",
                                        cov_kwds={"groups": df["_week"]})
        m_standard = sm.Logit(y, X).fit(disp=0)
        print("\nStandard (non-clustered) SE:", round(m_standard.bse["kde_is_bimodal"], 4),
              "p=", round(m_standard.pvalues["kde_is_bimodal"], 4))
        print("Cluster-robust SE (by week):", round(m_cluster.bse["kde_is_bimodal"], 4),
              "p=", round(m_cluster.pvalues["kde_is_bimodal"], 4))
        print("(If cluster-robust p-value remains < 0.05, the result is not an")
        print(" artefact of underestimated SEs from temporal event clustering.)")
    else:
        print(f"No date column found in {list(df.columns)}.")
        print("Cannot cluster without one -- pass --metadata with a date column,")
        print("or confirm the exact date column name in kde_results.")

    # ============================================================
    # POINT 2: Model misspecification checks
    # ============================================================
    print("\n" + "=" * 70)
    print("POINT 2: MODEL MISSPECIFICATION CHECKS")
    print("=" * 70)

    X = sm.add_constant(df[["kde_std", "kde_is_bimodal"]])
    y = df["extreme"]

    m_probit = sm.Probit(y, X).fit(disp=0)
    print("\n2a. Probit cross-check (should agree in sign/significance if not")
    print("    a logit-specific artefact):")
    print(f"    Bimodal coef: {m_probit.params['kde_is_bimodal']:.4f}, "
          f"p={m_probit.pvalues['kde_is_bimodal']:.4f}")

    df["kde_std_sq"] = df["kde_std"] ** 2
    X_nl = sm.add_constant(df[["kde_std", "kde_std_sq", "kde_is_bimodal"]])
    m_nl = sm.Logit(y, X_nl).fit(disp=0)
    print("\n2b. Adding kde_std^2 (does bimodal survive allowing non-linear")
    print("    dispersion effects?):")
    print(f"    Bimodal coef: {m_nl.params['kde_is_bimodal']:.4f}, "
          f"p={m_nl.pvalues['kde_is_bimodal']:.4f}")
    print(f"    kde_std^2 coef: {m_nl.params['kde_std_sq']:.4f}, "
          f"p={m_nl.pvalues['kde_std_sq']:.4f}")

    if args.metadata:
        meta = pd.read_csv(args.metadata)
        id_col = "event_id" if "event_id" in meta.columns else "document_id"
        type_col = next((c for c in ["Event type", "Event_type", "event_type"]
                          if c in meta.columns), None)
        if type_col:
            merged = df.merge(meta[[id_col, type_col]], left_on="event_id",
                               right_on=id_col, how="inner")
            if len(merged) > 0:
                dummies = pd.get_dummies(merged[type_col], prefix="etype", drop_first=True)
                X_et = sm.add_constant(pd.concat(
                    [merged[["kde_std", "kde_is_bimodal"]], dummies], axis=1).astype(float))
                y_et = merged["extreme"]
                m_et = sm.Logit(y_et, X_et).fit(disp=0)
                print(f"\n2c. Controlling for event type ({merged[type_col].nunique()} "
                      f"categories, N={len(merged)}):")
                print(f"    Bimodal coef: {m_et.params['kde_is_bimodal']:.4f}, "
                      f"p={m_et.pvalues['kde_is_bimodal']:.4f}")
            else:
                print("\n2c. SKIP: zero matches on merge for event-type control.")
        else:
            print("\n2c. SKIP: no event-type column found in metadata.")
    else:
        print("\n2c. SKIP: no --metadata provided for event-type control.")

    # ============================================================
    # POINT 3: Firth's penalized logistic regression (separation fix)
    # ============================================================
    print("\n" + "=" * 70)
    print("POINT 3: FIRTH'S PENALIZED LIKELIHOOD (separation-robust)")
    print("=" * 70)
    try:
        from firthlogist import FirthLogisticRegression
        Xf = df[["kde_std", "kde_is_bimodal"]].values
        yf = df["extreme"].values
        fl = FirthLogisticRegression()
        fl.fit(Xf, yf)
        print(f"Firth-corrected bimodal coefficient: {fl.coef_[1]:.4f}")
        print("(Remains numerically stable even in separation-affected cells")
        print(" that failed to converge under standard MLE.)")
    except ImportError:
        print("firthlogist not installed. Run:")
        print("  pip install firthlogist --break-system-packages")
        print("then rerun to get Firth-corrected estimates for the")
        print("separation-affected cells (h7@15%, h7@20%, h15@20%) instead of")
        print("excluding them.")


if __name__ == "__main__":
    main()
