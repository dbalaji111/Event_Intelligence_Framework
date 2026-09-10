"""
frl_text_validation_v2.py
============================
Corrected text-level validation using the REAL columns confirmed
in df_major_metadata.csv (Modality, Causal Links, Polarity), not
the cause/effect schema wrongly assumed from a single gold-corpus
sample in the thesis. Also adds a diagnostic for the exact-zero
precision/recall result from the first baseline run.

Usage:
    python frl_text_validation_v2.py \
        --kde_results results_run/kde_output_v4/kde_v4_results_h15.csv \
        --metadata memory_2001_2023/df_major_metadata.csv \
        --output frl_v4
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kde_results", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--ovx_col", default="base_ovx")
    ap.add_argument("--extreme_threshold", type=float, default=10.0)
    ap.add_argument("--output", default="frl_v4")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    kde = pd.read_csv(args.kde_results)
    meta = pd.read_csv(args.metadata)
    df = kde.dropna(subset=["event_id", "kde_is_bimodal", "kde_std", "realised_r"]).copy()
    df["extreme"] = (df["realised_r"].abs() > args.extreme_threshold).astype(int)

    # ============================================================
    # DIAGNOSTIC: confirm the exact-zero baseline result is real,
    # not a bug -- print raw counts directly.
    # ============================================================
    print("=" * 70)
    print("DIAGNOSTIC: raw overlap counts (confirming baseline result)")
    print("=" * 70)
    print(f"Total events: {len(df)}, extreme events: {df['extreme'].sum()}")
    print(f"Bimodal-flagged: {df['kde_is_bimodal'].sum()}")
    bimodal_and_extreme = ((df["kde_is_bimodal"] == 1) & (df["extreme"] == 1)).sum()
    print(f"Bimodal AND extreme (raw overlap count): {bimodal_and_extreme}")

    if args.ovx_col in df.columns:
        sub = df.dropna(subset=[args.ovx_col])
        thresh = sub[args.ovx_col].quantile(0.75)
        ovx_flag = (sub[args.ovx_col] >= thresh)
        print(f"OVX top-quartile threshold: {thresh:.2f}, N flagged: {ovx_flag.sum()}")
        ovx_and_extreme = (ovx_flag & (sub['extreme'] == 1)).sum()
        print(f"OVX-flagged AND extreme (raw overlap count): {ovx_and_extreme}")
    print()

    # ============================================================
    # TEXT VALIDATION -- using REAL confirmed columns
    # ============================================================
    print("=" * 70)
    print("TEXT-LEVEL VALIDATION (Modality / Causal Links / Polarity)")
    print("=" * 70)

    id_col = "event_id" if "event_id" in meta.columns else "document_id"
    real_cols = [c for c in ["Modality", "Causal Links", "Polarity", "Event type"]
                 if c in meta.columns]
    print(f"Using columns: {real_cols}")

    merged = df.merge(meta[[id_col] + real_cols], left_on="event_id",
                       right_on=id_col, how="inner")
    print(f"Merged: {len(merged)} of {len(df)} kde rows matched\n")

    if len(merged) == 0:
        print("ERROR: zero matches -- check event_id vs document_id format "
              "(e.g. 'EVT_2024.0_0010.0' vs a raw integer/date-based id). "
              "Print a few examples from each side to compare formats:")
        print("kde event_id sample:", df["event_id"].head(3).tolist())
        print("metadata id sample:", meta[id_col].head(3).tolist())
        return

    # Bimodal rate by Modality category
    if "Modality" in merged.columns:
        merged["Modality"] = merged["Modality"].astype(str).str.strip().str.lower()
        print("Modality value counts:")
        print(merged["Modality"].value_counts())
        print()
        rows = []
        for val, sub in merged.groupby("Modality"):
            rows.append({"category": f"Modality: {val}", "n": len(sub),
                         "bimodal_pct": round(sub["kde_is_bimodal"].mean() * 100, 1)})
        result = pd.DataFrame(rows)
        print(result.to_string(index=False))

        contingency = pd.crosstab(merged["Modality"], merged["kde_is_bimodal"])
        if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
            chi2, p, dof, _ = stats.chi2_contingency(contingency)
            print(f"\nChi-square (modality vs bimodal): chi2={chi2:.3f}, p={p:.4f}")
        result.to_csv(out_dir / "text_validation_modality.csv", index=False)
        print(f"[saved] {out_dir / 'text_validation_modality.csv'}\n")

    # Bimodal rate by causal chain length proxy
    if "Causal Links" in merged.columns:
        def chain_length(text):
            if pd.isna(text) or str(text).strip() == "":
                return 0
            text = str(text)
            for delim in ["->", "\u2192", ";", "|"]:
                if delim in text:
                    return text.count(delim) + 1
            return 1
        merged["chain_length"] = merged["Causal Links"].apply(chain_length)
        merged["chain_group"] = np.where(merged["chain_length"] >= 2,
                                          "Multi-step (>=2)", "Single-step/none")
        rows = []
        for val, sub in merged.groupby("chain_group"):
            rows.append({"category": val, "n": len(sub),
                         "bimodal_pct": round(sub["kde_is_bimodal"].mean() * 100, 1)})
        result2 = pd.DataFrame(rows)
        print(result2.to_string(index=False))
        result2.to_csv(out_dir / "text_validation_causal_links.csv", index=False)
        print(f"[saved] {out_dir / 'text_validation_causal_links.csv'}")


if __name__ == "__main__":
    main()