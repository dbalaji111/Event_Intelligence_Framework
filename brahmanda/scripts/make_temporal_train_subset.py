"""
make_temporal_train_subset.py
================================
Creates a date-filtered subset of the training corpus, containing
ONLY events strictly before a given cutoff date. Use this to build
a safe --train file for retrieval_engine_v2.py when testing an
event from INSIDE the training corpus itself (e.g. a 2012 event),
rather than trusting the script to filter dates internally.

Handles both JSON list-of-dicts and JSON dict-of-dicts formats,
auto-detecting which applies.

Usage:
    python make_temporal_train_subset.py \
        --train_file "memory_2001_2023/price_event_memory.json" \
        --cutoff_date 2012-01-01 \
        --output train_subset_before_2012.json
"""
import argparse
import json
import pandas as pd


def parse_date(d):
    return pd.to_datetime(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--cutoff_date", required=True,
                     help="YYYY-MM-DD; keep events strictly before this date")
    ap.add_argument("--date_key", default=None,
                     help="Field name holding the date. Auto-detected if omitted.")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    with open(args.train_file) as f:
        data = json.load(f)

    cutoff = parse_date(args.cutoff_date)
    print(f"Cutoff date: {cutoff.date()}")

    date_key_candidates = [args.date_key] if args.date_key else \
        ["date", "base_date", "event_date", "Date"]

    if isinstance(data, list):
        print(f"Detected list format, {len(data)} entries")
        sample = data[0] if data else {}
        date_key = next((k for k in date_key_candidates if k and k in sample), None)
        if date_key is None:
            print(f"ERROR: no date field found. Sample entry keys: {list(sample.keys())}")
            print("Pass --date_key explicitly.")
            return
        print(f"Using date field: '{date_key}'")

        filtered = []
        skipped_unparseable = 0
        for entry in data:
            try:
                entry_date = parse_date(entry[date_key])
                if entry_date < cutoff:
                    filtered.append(entry)
            except Exception:
                skipped_unparseable += 1
        print(f"Kept {len(filtered)} of {len(data)} entries (strictly before cutoff)")
        if skipped_unparseable:
            print(f"WARNING: {skipped_unparseable} entries had unparseable dates, excluded")

        with open(args.output, "w") as f:
            json.dump(filtered, f)

    elif isinstance(data, dict):
        print(f"Detected dict format, {len(data)} entries")
        sample_key = next(iter(data))
        sample = data[sample_key]
        date_key = next((k for k in date_key_candidates if k and k in sample), None)
        if date_key is None:
            print(f"ERROR: no date field found. Sample entry keys: {list(sample.keys())}")
            print("Pass --date_key explicitly.")
            return
        print(f"Using date field: '{date_key}'")

        filtered = {}
        skipped_unparseable = 0
        for key, entry in data.items():
            try:
                entry_date = parse_date(entry[date_key])
                if entry_date < cutoff:
                    filtered[key] = entry
            except Exception:
                skipped_unparseable += 1
        print(f"Kept {len(filtered)} of {len(data)} entries (strictly before cutoff)")
        if skipped_unparseable:
            print(f"WARNING: {skipped_unparseable} entries had unparseable dates, excluded")

        with open(args.output, "w") as f:
            json.dump(filtered, f)
    else:
        print(f"ERROR: unexpected top-level JSON type: {type(data)}")
        return

    print(f"\n[saved] {args.output}")
    print(f"Safe to use as --train for events on or after {args.cutoff_date} -- "
          f"guaranteed no entries from that date onward are included.")


if __name__ == "__main__":
    main()
