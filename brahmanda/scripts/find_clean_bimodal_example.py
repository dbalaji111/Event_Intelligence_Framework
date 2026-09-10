"""
find_clean_bimodal_example.py
===============================
Scans all v4 per-event posterior JSONs and finds genuinely clean,
well-separated two-mode bimodal examples, using scipy's find_peaks
with a real prominence threshold -- the same detector path_kde.py
uses, NOT the noisy gradient-sign-change counter still active inside
kde_engine_v4.py's run_batch(), which is what likely produced the
four-bump figure (small wiggles counted as separate "modes").

This does not change kde_engine_v4.py or any reported statistic --
it is purely a selection tool for picking a good illustrative example
for Figure 3 (fig:bimodal).

Usage:
    python find_clean_bimodal_example.py --kde_dir kde_output_v4/per_event
"""
import argparse, json
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks


def evaluate_event(json_path: Path, min_prominence_frac=0.10, min_separation_pct=8.0):
    """
    Returns a dict describing the event's peak structure using a proper
    prominence-based detector, or None if the file can't be scored.
    """
    try:
        data = json.load(json_path.open())
    except Exception:
        return None

    x = np.array(data.get("x_grid", []))
    d = np.array(data.get("density", []))
    if len(x) < 10 or len(d) < 10 or d.max() <= 0:
        return None

    peaks, props = find_peaks(d, prominence=d.max() * min_prominence_frac)
    if len(peaks) < 2:
        return None

    # keep the two most prominent peaks
    order = np.argsort(props["prominences"])[::-1]
    top2_idx = sorted(order[:2])
    p1, p2 = peaks[top2_idx[0]], peaks[top2_idx[1]]
    x1, x2 = x[p1], x[p2]
    separation = abs(x2 - x1)

    if separation < min_separation_pct:
        return None  # peaks too close together to be a meaningful split

    prom1 = props["prominences"][top2_idx[0]]
    prom2 = props["prominences"][top2_idx[1]]
    balance = min(prom1, prom2) / max(prom1, prom2)  # 1.0 = perfectly balanced modes

    n_total_peaks = len(peaks)  # how many bumps total, including minor ones

    return {
        "event_id":       data.get("event_id", json_path.stem),
        "path":           str(json_path),
        "n_total_peaks":  int(n_total_peaks),
        "mode1_x":        round(float(x1), 2),
        "mode2_x":        round(float(x2), 2),
        "separation":     round(float(separation), 2),
        "balance":        round(float(balance), 3),
        "stats":          data.get("posterior_stats", {}),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kde_dir", required=True)
    ap.add_argument("--min_prominence_frac", type=float, default=0.10,
                     help="Minimum peak prominence as a fraction of max "
                          "density. Higher = stricter, fewer false modes.")
    ap.add_argument("--min_separation_pct", type=float, default=8.0,
                     help="Minimum distance (in %% return) between the two "
                          "modes to count as genuinely separated.")
    ap.add_argument("--top_n", type=int, default=10)
    args = ap.parse_args()

    json_files = sorted(Path(args.kde_dir).glob("*_kde_v4.json"))
    print(f"Scanning {len(json_files)} events...")

    candidates = []
    for jf in json_files:
        result = evaluate_event(jf, args.min_prominence_frac, args.min_separation_pct)
        if result is not None:
            candidates.append(result)

    print(f"\n{len(candidates)} events have >=2 genuinely prominent, "
          f"well-separated peaks (prominence >= {args.min_prominence_frac*100:.0f}% "
          f"of max density, separation >= {args.min_separation_pct}%)\n")

    # Rank by: fewest total peaks (cleanest, least noisy) first,
    # then by best balance between the two main modes (most visually
    # convincing bimodality), then by largest separation.
    candidates.sort(key=lambda c: (c["n_total_peaks"], -c["balance"], -c["separation"]))

    print(f"{'Rank':<5}{'Event ID':<28}{'#peaks':<8}{'Mode1':<8}{'Mode2':<8}"
          f"{'Sep':<8}{'Balance':<9}")
    print("-" * 80)
    for i, c in enumerate(candidates[:args.top_n], 1):
        print(f"{i:<5}{c['event_id']:<28}{c['n_total_peaks']:<8}"
              f"{c['mode1_x']:<8}{c['mode2_x']:<8}{c['separation']:<8}{c['balance']:<9}")

    if candidates:
        best = candidates[0]
        print(f"\nBest candidate: {best['event_id']}")
        print(f"  Path: {best['path']}")
        print(f"  {best['n_total_peaks']} total peaks detected (fewer = cleaner), "
              f"modes at {best['mode1_x']}% and {best['mode2_x']}%, "
              f"separation {best['separation']}pp, balance {best['balance']}")
        print(f"\nUse this for generate_figures.py:")
        print(f"  --kde_json {best['path']}")
    else:
        print("\nNo clean two-mode candidates found with these thresholds. "
              "Try lowering --min_prominence_frac or --min_separation_pct.")


if __name__ == "__main__":
    main()
