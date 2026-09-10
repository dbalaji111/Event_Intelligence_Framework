"""
diagnose_full_pipeline_toggle.py
==================================
We've confirmed compute_three_rank_weight() DOES respond correctly to
W_DTW (0.586 vs 0.786 for one real analogue -- a genuine, substantial
difference). So the bug is not in that function. This script runs the
FULL run_event_v3 pipeline for one event under both W_DTW=0.30 and
W_DTW=0.00 and diffs the actual output dict field-by-field, to find
exactly where that real per-analogue difference gets lost before it
reaches kde_p5/median/p95/covers_90.

Usage:
    python scripts/diagnose_full_pipeline_toggle.py \
        --analogues retrieval_output_v3/per_event \
        --prices    "data/oil_prices_full.xlsx" \
        --horizon   15
"""
import argparse, json, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import Kde_engine_v3 as eng


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analogues", required=True)
    ap.add_argument("--prices", required=True)
    ap.add_argument("--horizon", type=int, default=15)
    args = ap.parse_args()

    prices = eng.PriceData(Path(args.prices), None)
    json_files = sorted(Path(args.analogues).glob("*_analogues.json"))
    jf = json_files[0]
    data = json.load(jf.open())
    base_date = pd.to_datetime(data["base_date"])
    analogues = data["analogues"]
    event_id = data.get("base_event_id", jf.stem.replace("_analogues", ""))

    print(f"Event: {event_id}, base_date={base_date.date()}, "
          f"{len(analogues)} analogues, horizon=+{args.horizon}d\n")

    results = {}
    for w_dtw_test, label in [(0.30, "FULL"), (0.00, "NO_DTW")]:
        eng.W_SIM, eng.W_QUANTILE, eng.W_DTW = 0.50, 0.20, w_dtw_test
        print(f"=== {label}: W_SIM={eng.W_SIM} W_QUANTILE={eng.W_QUANTILE} "
              f"W_DTW={eng.W_DTW} ===")

        out = eng.run_event_v3(
            event_id=event_id, base_date=base_date, analogues=analogues,
            prices=prices, horizon=args.horizon, min_analogues=eng.MIN_ANALOGUES,
        )
        result = out[0] if isinstance(out, tuple) else out
        details = out[1] if isinstance(out, tuple) and len(out) > 1 else None

        results[label] = result

        for k in ["n_analogues", "tau_effective", "kde_p5", "kde_median",
                  "kde_p95", "kde_std", "covers_90", "kde_is_bimodal",
                  "kde_tail_weight"]:
            print(f"  {k}: {result.get(k)}")

        if details:
            print(f"  first 5 analogue weights: "
                  f"{[round(d['weight'], 6) for d in details[:5]]}")
        print()

    print("=" * 60)
    print("DIFF: FULL vs NO_DTW")
    print("=" * 60)
    keys = set(results["FULL"].keys()) | set(results["NO_DTW"].keys())
    any_diff = False
    for k in sorted(keys):
        v1, v2 = results["FULL"].get(k), results["NO_DTW"].get(k)
        if v1 != v2:
            any_diff = True
            print(f"  DIFFERS  {k}: FULL={v1}  NO_DTW={v2}")
    if not any_diff:
        print("  NO DIFFERENCE in any field of the returned result dict, "
              "despite individual analogue weights genuinely differing.")
        print("  -> the bug is inside run_event_v3's aggregation step, "
              "between building the weights array and calling "
              "compute_mixture_posterior (or inside that function itself "
              "when given THIS event's actual mus/sigmas/weights).")


if __name__ == "__main__":
    main()
