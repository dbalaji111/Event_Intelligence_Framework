"""
frl_master_analysis.py
========================
Runs every computation FRL's paper needs, in one pass, using the
REAL schema (cause/effect/topic), not the aspirational Modality/
Causal_links fields that don't exist in the actual corpus yet.

Produces, printed and saved to CSV:
  1. H1: logistic regression, extreme move ~ bimodal_flag + kde_std
     (does bimodality predict beyond dispersion alone?)
  2. Naive baseline comparison: bimodal flag vs an OVX-threshold rule
     for predicting extreme moves
  3. Text-based validation: does cause/effect field PRESENCE and
     LENGTH predict bimodal rate? (the real, available proxy for
     the original modality/causal-chain hypothesis)

Usage:
    python frl_master_analysis.py \
        --kde_results kde_output_v4/kde_v4_results_h15.csv \
        --metadata memory_2001_2023/df_major_metadata.csv \
        --output frl_v4/
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kde_results", required=True)
    ap.add_argument("--metadata", required=True,
                     help="df_major_metadata.csv or equivalent, must "
                          "contain cause/effect/topic columns")
    ap.add_argument("--extreme_threshold", type=float, default=10.0,
                     help="Absolute return %% defining an 'extreme move'")
    ap.add_argument("--ovx_col", default="base_ovx")
    ap.add_argument("--output", default="frl_v4")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    kde = pd.read_csv(args.kde_results)
    meta = pd.read_csv(args.metadata)
    print(f"KDE results: {len(kde)} rows")
    print(f"Metadata: {len(meta)} rows, columns: {meta.columns.tolist()}\n")

    required_kde = ["event_id", "kde_is_bimodal", "kde_std", "realised_r"]
    missing = [c for c in required_kde if c not in kde.columns]
    if missing:
        print(f"ERROR: missing columns in kde_results: {missing}")
        return

    df = kde.dropna(subset=required_kde).copy()
    df["extreme"] = (df["realised_r"].abs() > args.extreme_threshold).astype(int)
    print(f"{len(df)} events with complete data. "
          f"{df['extreme'].sum()} extreme moves (|r|>{args.extreme_threshold}%).\n")

    # ============================================================
    # H1: Does bimodality predict extreme moves beyond dispersion?
    # ============================================================
    print("=" * 70)
    print("H1: BEYOND-DISPERSION TEST")
    print("=" * 70)
    try:
        import statsmodels.api as sm

        X1 = sm.add_constant(df[["kde_std"]])
        m1 = sm.Logit(df["extreme"], X1).fit(disp=0)

        X2 = sm.add_constant(df[["kde_std", "kde_is_bimodal"]])
        m2 = sm.Logit(df["extreme"], X2).fit(disp=0)

        print("\nModel 1 (dispersion only):")
        print(m1.summary().tables[1])
        print("\nModel 2 (+ bimodal flag):")
        print(m2.summary().tables[1])

        lr_stat = 2 * (m2.llf - m1.llf)
        lr_p = stats.chi2.sf(lr_stat, df=1)
        print(f"\nLikelihood-ratio test (does bimodal flag add value?): "
              f"LR={lr_stat:.3f}, p={lr_p:.4f}")
        print(f"{'BIMODAL FLAG ADDS SIGNIFICANT VALUE' if lr_p < 0.05 else 'NOT significant'} at alpha=0.05")

        h1_results = pd.DataFrame({
            "term": ["const", "kde_std", "bimodal_flag"],
            "model2_coef": [m2.params.get(k, np.nan) for k in ["const", "kde_std", "kde_is_bimodal"]],
            "model2_pvalue": [m2.pvalues.get(k, np.nan) for k in ["const", "kde_std", "kde_is_bimodal"]],
        })
        h1_results.to_csv(out_dir / "h1_logistic_regression.csv", index=False)
        print(f"\n[saved] {out_dir / 'h1_logistic_regression.csv'}")
    except ImportError:
        print("statsmodels not installed - run: pip install statsmodels --break-system-packages")
    except Exception as e:
        print(f"H1 regression failed: {e}")

    # ============================================================
    # Naive baseline: bimodal flag vs OVX threshold
    # ============================================================
    print("\n" + "=" * 70)
    print("NAIVE BASELINE COMPARISON")
    print("=" * 70)
    if args.ovx_col in df.columns:
        sub = df.dropna(subset=[args.ovx_col])
        ovx_threshold = sub[args.ovx_col].quantile(0.75)
        sub = sub.copy()
        sub["ovx_flag"] = (sub[args.ovx_col] >= ovx_threshold).astype(int)

        def prf(flag_col, label):
            tp = ((sub[flag_col] == 1) & (sub["extreme"] == 1)).sum()
            fp = ((sub[flag_col] == 1) & (sub["extreme"] == 0)).sum()
            fn = ((sub[flag_col] == 0) & (sub["extreme"] == 1)).sum()
            precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
            recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            fpr = fp / ((sub[flag_col] == 1).sum() + 1e-9)
            print(f"{label}: N_flagged={sub[flag_col].sum()}, "
                  f"precision={precision:.3f}, recall={recall:.3f}")
            return {"signal": label, "n_flagged": int(sub[flag_col].sum()),
                    "precision": round(precision, 3) if not np.isnan(precision) else None,
                    "recall": round(recall, 3) if not np.isnan(recall) else None}

        r1 = prf("kde_is_bimodal", "Bimodal flag")
        r2 = prf("ovx_flag", f"OVX top-quartile (>={ovx_threshold:.1f})")
        pd.DataFrame([r1, r2]).to_csv(out_dir / "baseline_comparison.csv", index=False)
        print(f"[saved] {out_dir / 'baseline_comparison.csv'}")
    else:
        print(f"SKIP: '{args.ovx_col}' not found in kde_results columns: {df.columns.tolist()}")

    # ============================================================
    # Text validation: cause/effect presence & length vs bimodal
    # ============================================================
    print("\n" + "=" * 70)
    print("TEXT-LEVEL VALIDATION (cause/effect fields)")
    print("=" * 70)
    id_col_meta = "document_id" if "document_id" in meta.columns else "event_id"
    text_cols = [c for c in ["cause", "effect", "cause_effect_summary"] if c in meta.columns]
    if id_col_meta not in meta.columns or not text_cols:
        print(f"SKIP: need an id column and cause/effect columns. "
              f"Found id={id_col_meta in meta.columns}, text_cols={text_cols}")
    else:
        merged = df.merge(meta[[id_col_meta] + text_cols], left_on="event_id",
                           right_on=id_col_meta, how="inner")
        print(f"Merged: {len(merged)} events matched (of {len(df)} kde rows, {len(meta)} metadata rows)")

        if len(merged) == 0:
            print("ERROR: zero matches - check event_id/document_id format alignment")
        else:
            for col in text_cols:
                merged[f"{col}_len"] = merged[col].fillna("").astype(str).str.len()
                merged[f"{col}_present"] = (merged[f"{col}_len"] > 0).astype(int)

            rows = []
            for col in text_cols:
                for present_val, label in [(1, "present"), (0, "absent")]:
                    sub = merged[merged[f"{col}_present"] == present_val]
                    if len(sub) > 0:
                        rows.append({
                            "field": col, "status": label, "n": len(sub),
                            "bimodal_pct": round(sub["kde_is_bimodal"].mean() * 100, 1)
                        })
            result_df = pd.DataFrame(rows)
            print(result_df.to_string(index=False))
            result_df.to_csv(out_dir / "text_validation.csv", index=False)
            print(f"[saved] {out_dir / 'text_validation.csv'}")

            # correlation: total cause+effect length vs bimodal
            merged["total_text_len"] = sum(merged[f"{c}_len"] for c in text_cols)
            if merged["total_text_len"].std() > 0:
                corr, p = stats.pointbiserialr(merged["kde_is_bimodal"], merged["total_text_len"])
                print(f"\nPoint-biserial correlation (bimodal vs total cause+effect length): "
                      f"r={corr:.3f}, p={p:.4f}")

    print("\n" + "=" * 70)
    print("DONE. Three CSVs saved to", out_dir)
    print("=" * 70)


if __name__ == "__main__":
    main()
