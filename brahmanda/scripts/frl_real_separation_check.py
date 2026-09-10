"""
frl_real_separation_check.py
===============================
Extracts the ACTUAL mode separation observed in real bimodal-flagged
events, using the same find_peaks logic as bimodal detection itself.
This replaces the illustrative separations in
frl_mixture_artifact_check.py with the real distribution, so the
mechanical-artefact comparison is grounded in actual data rather
than assumption.

Usage:
    python frl_real_separation_check.py \
        --per_event_dir results_run/kde_output_v4/per_event \
        --kde_results results_run/kde_output_v4/kde_v4_results_h15.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_event_dir", required=True)
    ap.add_argument("--kde_results", required=True)
    ap.add_argument("--prominence_frac", type=float, default=0.10)
    args = ap.parse_args()

    df = pd.read_csv(args.kde_results)
    bimodal_ids = df[df["kde_is_bimodal"] == 1]["event_id"].tolist()
    print(f"{len(bimodal_ids)} bimodal-flagged events to check\n")

    separations = []
    for eid in bimodal_ids:
        # adjust filename pattern if yours differs
        candidates = list(Path(args.per_event_dir).glob(f"*{eid}*kde_v4.json"))
        if not candidates:
            continue
        data = json.load(open(candidates[0]))
        x = np.array(data.get("x_grid", []))
        d = np.array(data.get("density", []))
        if len(x) < 10 or d.max() <= 0:
            continue
        peaks, props = find_peaks(d, prominence=d.max() * args.prominence_frac)
        if len(peaks) < 2:
            continue
        order = np.argsort(props["prominences"])[::-1]
        top2 = sorted(peaks[order[:2]])
        sep = abs(x[top2[1]] - x[top2[0]])
        separations.append(sep)

    if not separations:
        print("No separations extracted -- check --per_event_dir path and "
              "filename pattern matches your actual per-event JSON files.")
        return

    sep_arr = np.array(separations)
    print(f"Matched {len(sep_arr)} of {len(bimodal_ids)} bimodal events to real posterior files")
    print(f"\nReal observed mode separation distribution:")
    print(f"  Mean:   {sep_arr.mean():.2f}")
    print(f"  Median: {np.median(sep_arr):.2f}")
    print(f"  Min:    {sep_arr.min():.2f}")
    print(f"  Max:    {sep_arr.max():.2f}")
    print(f"  25th/75th pctile: {np.percentile(sep_arr, 25):.2f} / {np.percentile(sep_arr, 75):.2f}")

    pd.DataFrame({"separation": sep_arr}).to_csv("real_separations.csv", index=False)
    print(f"\n[saved] real_separations.csv")
    print(f"\nCompare this to the mechanical simulation's well-behaved range")
    print(f"(roughly 0 to {2*7.666:.1f}, i.e. up to 2x mean kde_std, before the")
    print(f"parameterization degenerates). If most real separations fall in")
    print(f"that well-behaved range, the mechanical-OR-at-that-separation is")
    print(f"the correct number to compare against the empirical OR=0.169 --")
    print(f"not the illustrative full range originally reported.")


if __name__ == "__main__":
    main()
