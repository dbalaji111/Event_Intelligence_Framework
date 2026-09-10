"""
build_rolling_window_files.py  (FIXED)
=========================================
Builds the train/test file pairs needed for a true expanding-window
rolling analysis: for each test year Y, train = all events strictly
before Jan 1 of Y; test = only events within year Y.

Fix vs. previous version: dates are kept in a separate parallel list,
never added to or deleted from the actual event dicts. The earlier
version mutated shared dict objects across loop iterations, causing
a KeyError once a dict's temporary date field had been deleted in
one iteration but was needed again in the next.

Usage:
    python build_rolling_window_files.py \
        --corpus_file memory_2001_2023/price_event_memory.json \
        --start_year 2013 --end_year 2023 \
        --output_dir rolling_window_files/
"""
import argparse
import json
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus_file", required=True)
    ap.add_argument("--start_year", type=int, required=True)
    ap.add_argument("--end_year", type=int, required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.corpus_file) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} total events")

    valid = [d for d in data if isinstance(d.get("date"), str)]
    skipped = len(data) - len(valid)
    if skipped:
        print(f"WARNING: {skipped} events skipped (non-string/missing date)")

    # Dates kept SEPARATELY, parallel to `valid` -- never written into
    # the event dicts themselves, so no mutation/restoration needed.
    parsed_dates = [pd.to_datetime(d["date"]) for d in valid]

    summary = []
    for year in range(args.start_year, args.end_year + 1):
        cutoff = pd.Timestamp(f"{year}-01-01")
        next_cutoff = pd.Timestamp(f"{year+1}-01-01")

        train = [d for d, dt in zip(valid, parsed_dates) if dt < cutoff]
        test_year = [d for d, dt in zip(valid, parsed_dates) if cutoff <= dt < next_cutoff]

        train_path = out_dir / f"train_before_{year}.json"
        test_path = out_dir / f"test_only_{year}.json"
        with open(train_path, "w") as f:
            json.dump(train, f)
        with open(test_path, "w") as f:
            json.dump(test_year, f)

        print(f"Year {year}: train={len(train)} events (< {cutoff.date()}), "
              f"test={len(test_year)} events (within {year})")
        summary.append({"year": year, "n_train": len(train), "n_test": len(test_year)})

    pd.DataFrame(summary).to_csv(out_dir / "rolling_window_summary.csv", index=False)
    print(f"\n[saved] all train/test file pairs to {out_dir}/")
    print(f"[saved] {out_dir}/rolling_window_summary.csv")
    print(f"\nTotal test events across all years: {sum(s['n_test'] for s in summary)}")


if __name__ == "__main__":
    main()