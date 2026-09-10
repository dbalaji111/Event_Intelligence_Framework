"""
verify_batch_no_leakage.py
=============================
Checks EVERY event in a completed batch retrieval output (not just
one diagnostic example) to confirm zero analogues have a date on or
after their base event's own date. This is the real pre-flight check
before trusting a full multi-year rolling-window run.

Usage:
    python verify_batch_no_leakage.py \
        --base_events rolling_window_files/test_only_2013.json \
        --analogue_dir rolling_output/2013
"""
import argparse
import json
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_events", required=True,
                     help="The test_only_YYYY.json file used as input")
    ap.add_argument("--analogue_dir", required=True,
                     help="Directory containing per-event analogue JSON output")
    args = ap.parse_args()

    with open(args.base_events) as f:
        base_events = json.load(f)
    print(f"Base events file: {len(base_events)} events")

    analogue_files = list(Path(args.analogue_dir).glob("*.json"))
    print(f"Analogue output files found: {len(analogue_files)}\n")

    if len(analogue_files) == 0:
        print("ERROR: no analogue JSON files found. Check --analogue_dir path "
              "and that this batch run actually completed.")
        return

    # Peek at one file's structure so we know the real key names before
    # checking all of them -- avoid guessing blind.
    sample = json.load(open(analogue_files[0]))
    print(f"Sample file structure ({analogue_files[0].name}): {list(sample.keys())}\n")

    checked = 0
    total_analogues_checked = 0
    violations = []
    missing_analogue_key = 0

    base_dates = {}
    for e in base_events:
        if isinstance(e.get("date"), str):
            base_dates[e["event_id"]] = pd.to_datetime(e["date"])

    for af in analogue_files:
        data = json.load(open(af))
        event_id = data.get("event_id") or data.get("base_event_id") or af.stem.replace("_analogues", "")
        if event_id not in base_dates:
            continue
        base_date = base_dates[event_id]

        # Try common key names for the analogue list
        analogues = (data.get("analogues") or data.get("top_analogues") or
                     data.get("results") or data.get("neighbors"))
        if analogues is None:
            missing_analogue_key += 1
            continue

        checked += 1
        for a in analogues:
            a_date_raw = a.get("date") if isinstance(a, dict) else None
            if a_date_raw is None:
                continue
            a_date = pd.to_datetime(a_date_raw)
            total_analogues_checked += 1
            if a_date >= base_date:
                violations.append({
                    "base_event": event_id, "base_date": str(base_date.date()),
                    "analogue_date": str(a_date.date())
                })

    print(f"Events checked: {checked}")
    print(f"Total analogue-pairs checked: {total_analogues_checked}")
    if missing_analogue_key:
        print(f"WARNING: {missing_analogue_key} files did not match expected "
              f"analogue-list key names -- structure may differ, adjust the "
              f"script's key-name guesses above.")

    print()
    if violations:
        print(f"LEAKAGE FOUND: {len(violations)} violations across "
              f"{len(set(v['base_event'] for v in violations))} events")
        for v in violations[:15]:
            print(f"  {v['base_event']} (base={v['base_date']}): "
                  f"analogue dated {v['analogue_date']} (NOT strictly prior)")
        pd.DataFrame(violations).to_csv("leakage_violations.csv", index=False)
        print("\n[saved] leakage_violations.csv -- DO NOT proceed with the "
              "remaining years until this is understood and fixed.")
    else:
        print(f"CLEAN: zero violations across {checked} events and "
              f"{total_analogues_checked} analogue-pairs checked.")
        print("Safe to proceed with the remaining years.")


if __name__ == "__main__":
    main()
