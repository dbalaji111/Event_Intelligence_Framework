"""
calibration_table.py
======================
Validates the risk measure empirically: does the model's STATED
probability of a downside extreme move (kde_p_down10) match the
REALIZED frequency of that outcome, across many events? This is
the standard calibration/reliability check for any probabilistic
risk measure (the same logic underlying VaR backtesting).

Also produces a bimodal-specific version: for bimodal events only,
does the tail-scenario weight (kde_tail_weight) track how often the
realized outcome actually lands in the tail scenario's territory?

Usage:
    python calibration_table.py \
        --kde_results results_run/kde_output_v4/kde_v4_results_h15.csv \
        --n_bins 5
"""
import argparse
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kde_results", required=True)
    ap.add_argument("--n_bins", type=int, default=5)
    ap.add_argument("--down_threshold", type=float, default=-10.0)
    args = ap.parse_args()

    df = pd.read_csv(args.kde_results)
    required = ["kde_p_down10", "realised_r"]
    df = df.dropna(subset=required)
    df["actual_down_extreme"] = (df["realised_r"] < args.down_threshold).astype(int)

    print("=" * 70)
    print("CALIBRATION TABLE: stated P(down>10%) vs realized frequency")
    print("=" * 70)
    print(f"N={len(df)}\n")

    df["bin"] = pd.qcut(df["kde_p_down10"], q=args.n_bins, duplicates="drop")
    calib = df.groupby("bin").agg(
        n=("actual_down_extreme", "size"),
        mean_stated_prob=("kde_p_down10", "mean"),
        realized_freq=("actual_down_extreme", "mean"),
    ).reset_index()
    calib["gap"] = calib["realized_freq"] - calib["mean_stated_prob"]

    print(f"{'Bin (stated P range)':<25}{'N':>6}{'Mean stated P':>16}{'Realized freq':>16}{'Gap':>10}")
    for _, row in calib.iterrows():
        print(f"{str(row['bin']):<25}{row['n']:>6}{row['mean_stated_prob']*100:>15.1f}%"
              f"{row['realized_freq']*100:>15.1f}%{row['gap']*100:>9.1f}pp")

    corr = df["kde_p_down10"].corr(df["actual_down_extreme"])
    print(f"\nCorrelation (stated probability, realized outcome): {corr:.3f}")
    print("(A well-calibrated measure shows realized freq tracking mean stated")
    print(" probability closely across bins, and a positive, meaningful correlation.)")

    calib.to_csv("calibration_table_full.csv", index=False)

    # Bimodal-specific check
    print("\n" + "=" * 70)
    print("BIMODAL-SPECIFIC: does tail_weight track realized tail outcomes?")
    print("=" * 70)
    bimodal_df = df[df["kde_is_bimodal"] == 1].copy() if "kde_is_bimodal" in df.columns else None
    if bimodal_df is not None and len(bimodal_df) > 10 and "kde_tail_mu" in bimodal_df.columns:
        # Direction-aware: the tail scenario isn't always on the downside.
        # Check whether the realized outcome moved in the SAME DIRECTION as
        # this specific event's tail scenario, and by a comparably large
        # magnitude -- not just "was there a down move" universally.
        bimodal_df["tail_direction"] = np.sign(bimodal_df["kde_tail_mu"])
        bimodal_df["realized_matches_tail_direction"] = (
            (np.sign(bimodal_df["realised_r"]) == bimodal_df["tail_direction"]) &
            (bimodal_df["realised_r"].abs() > 10.0)
        ).astype(int)

        n_bins_bimodal = min(args.n_bins, 4)
        bimodal_df["tw_bin"] = pd.qcut(bimodal_df["kde_tail_weight"], q=n_bins_bimodal, duplicates="drop")
        calib_bimodal = bimodal_df.groupby("tw_bin", observed=True).agg(
            n=("realized_matches_tail_direction", "size"),
            mean_tail_weight=("kde_tail_weight", "mean"),
            realized_freq=("realized_matches_tail_direction", "mean"),
        ).reset_index()
        print(f"N bimodal events with data: {len(bimodal_df)}")
        print(f"(Direction-aware: checks whether realized outcome moved in the")
        print(f" SAME direction as that event's own tail scenario, |r|>10%)\n")
        print(f"{'Tail weight bin':<25}{'N':>6}{'Mean tail weight':>18}{'Realized freq':>16}")
        for _, row in calib_bimodal.iterrows():
            print(f"{str(row['tw_bin']):<25}{row['n']:>6}{row['mean_tail_weight']*100:>17.1f}%"
                  f"{row['realized_freq']*100:>15.1f}%")
        calib_bimodal.to_csv("calibration_table_bimodal.csv", index=False)
    else:
        print("Insufficient bimodal events with required columns for this check.")

    print(f"\n[saved] calibration_table_full.csv, calibration_table_bimodal.csv")


if __name__ == "__main__":
    main()