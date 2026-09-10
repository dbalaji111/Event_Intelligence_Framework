"""
verify_price_series.py
=======================
Cross-checks price_series.py against known, independently-verifiable
historical WTI facts, and audits compute_return() for silent date-snapping
errors. Run this before trusting any return numbers for the thesis.

    PYTHONPATH=. python scripts/verify_price_series.py
"""
import sys
sys.path.insert(0, "scripts")
import pandas as pd
import price_series

def check(label, actual, expected_range, note=""):
    lo, hi = expected_range
    ok = actual is not None and lo <= actual <= hi
    status = "OK  " if ok else "FAIL"
    print(f"[{status}] {label}: got {actual}  (expected {lo} to {hi})  {note}")
    return ok


def main():
    series = price_series.load_wti_series(".")
    print(f"Series loaded: {len(series)} points, "
          f"{series.index.min().date()} to {series.index.max().date()}\n")

    # ─────────────────────────────────────────────────────────────────
    # PART 1 — spot checks against publicly known historical WTI prices
    # ─────────────────────────────────────────────────────────────────
    print("=" * 70)
    print("PART 1: known historical price checks")
    print("=" * 70)

    def price_on(date_str):
        d = pd.Timestamp(date_str)
        avail = series.index[series.index <= d]
        if len(avail) == 0:
            return None, None
        actual_date = avail[-1]
        return float(series.loc[actual_date]), actual_date

    checks = [
        # (date, expected low, expected high, what this date is known for)
        ("2020-04-20", -50, -30, "WTI went negative (~-$37) for the first time ever"),
        ("2020-04-17", 15, 25, "just before the April 20 crash, still positive/normal"),
        ("2022-02-24", 90, 100, "Russia invades Ukraine, WTI spiked"),
        ("2020-03-09", 25, 35, "OPEC+ price war / COVID crash begins"),
        ("2008-07-11", 140, 150, "WTI all-time high near $147"),
        ("2016-02-11", 25, 32, "multi-year low during 2014-16 oil glut"),
    ]

    for date_str, lo, hi, note in checks:
        p, actual_date = price_on(date_str)
        drift = "" if actual_date == pd.Timestamp(date_str) else f" [snapped from {date_str} to {actual_date.date() if actual_date is not None else 'N/A'}]"
        check(date_str, p, (lo, hi), note + drift)

    # ─────────────────────────────────────────────────────────────────
    # PART 2 — manual return recomputation for a few analogues,
    # cross-checked against what /analyze actually returned
    # ─────────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("PART 2: manual return recomputation (should match /analyze output)")
    print("=" * 70)

    # (event_date, reported r1, r7, r15 from the last /analyze response)
    reported = [
        ("2012-06-13", 1.4645, -0.9925, -5.9671),
        ("2020-01-09", -0.9063, -1.7791, -9.1641),
        ("2020-04-23", 4.129, 21.5484, 59.6129),
    ]

    for date_str, exp_r1, exp_r7, exp_r15 in reported:
        p0, snapped0 = price_on(date_str)
        print(f"\nEvent date {date_str} -> snapped to {snapped0.date() if snapped0 is not None else 'N/A'}, price=${p0}")
        for label, days, expected in [("r1", 1, exp_r1), ("r7", 7, exp_r7), ("r15", 15, exp_r15)]:
            r = price_series.compute_return(series, date_str, days)
            target = snapped0 + pd.Timedelta(days=days) if snapped0 is not None else None
            future_avail = series.index[series.index >= target] if target is not None else []
            actual_target_date = future_avail[0] if len(future_avail) else None
            match = "MATCH" if r is not None and abs(r - expected) < 0.01 else "MISMATCH"
            gap = (actual_target_date - target).days if actual_target_date is not None and target is not None else "N/A"
            print(f"  {label}: computed={r}  reported={expected}  [{match}]  "
                  f"target_date={target.date() if target is not None else 'N/A'} "
                  f"-> snapped_to={actual_target_date.date() if actual_target_date is not None else 'N/A'} "
                  f"(gap={gap} days)")

    # ─────────────────────────────────────────────────────────────────
    # PART 3 — investigate the r10=null anomaly for the 2020-04-08 event
    # ─────────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("PART 3: why did r10 come back null for the 2020-04-08 analogue?")
    print("=" * 70)

    date_str = "2020-04-08"
    p0, snapped0 = price_on(date_str)
    print(f"Event date {date_str} -> snapped to {snapped0.date() if snapped0 is not None else 'N/A'}, price=${p0}")
    for label, days in [("r7", 7), ("r10", 10), ("r15", 15)]:
        target = snapped0 + pd.Timedelta(days=days)
        future_avail = series.index[series.index >= target]
        r = price_series.compute_return(series, date_str, days)
        if len(future_avail) == 0:
            print(f"  {label} (target {target.date()}): NO DATA AT OR AFTER THIS DATE -> null "
                  f"(series ends {series.index.max().date()})")
        else:
            nearest = future_avail[0]
            gap = (nearest - target).days
            print(f"  {label} (target {target.date()}): nearest available date = {nearest.date()} "
                  f"(gap {gap}d) -> r={r}")

    # ─────────────────────────────────────────────────────────────────
    # PART 4 — audit for silent large date-snapping gaps across a sample
    # ─────────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("PART 4: date-snapping gap audit (flags any snap > 3 calendar days)")
    print("=" * 70)

    sample_dates = pd.date_range("2001-01-01", "2026-08-01", freq="90D")
    large_gaps = 0
    for d in sample_dates:
        avail = series.index[series.index <= d]
        if len(avail) == 0:
            continue
        snapped = avail[-1]
        gap = (d - snapped).days
        if gap > 3:
            large_gaps += 1
            print(f"  {d.date()} snapped back to {snapped.date()} -- gap of {gap} days")

    print(f"\n{large_gaps} large gaps (>3 days) found out of {len(sample_dates)} sampled dates.")
    if large_gaps == 0:
        print("Clean -- no unexpected multi-day data gaps in the base series.")


if __name__ == "__main__":
    main()
