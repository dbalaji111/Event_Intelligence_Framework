"""
combine_rolling_kde_results.py
=================================
Combines all 11 years of rolling-window KDE v4 results
(rolling_kde_output/{year}/kde_v4_results_h15.csv) into one dataset,
ready for the FRL analysis scripts. 2020 excluded by default given
its confirmed, extreme COVID-driven breakdown (74.4% coverage vs.
90% target, FEI=6.4, std=29.81% vs 4-9% every other year) --
consistent with the companion IJF paper's own precedent of
excluding the acute COVID period from the analogue pool.

Usage:
    python combine_rolling_kde_results.py \
        --input_dir rolling_kde_output \
        --years 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 \
        --exclude_years 2020 \
        --output rolling_kde_combined_h15.csv
"""
import argparse
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="rolling_kde_output")
    ap.add_argument("--years", type=int, nargs="+",
                     default=[2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023])
    ap.add_argument("--exclude_years", type=int, nargs="*", default=[2020],
                     help="Years to load but flag/exclude from the primary combined file")
    ap.add_argument("--output", default="rolling_kde_combined_h15.csv")
    args = ap.parse_args()

    all_dfs = []
    for year in args.years:
        path = Path(args.input_dir) / str(year) / "kde_v4_results_h15.csv"
        if not path.exists():
            print(f"WARNING: {path} not found, skipping year {year}")
            continue
        df = pd.read_csv(path)
        df["source_year"] = year
        df["excluded_from_primary"] = year in args.exclude_years
        all_dfs.append(df)
        print(f"{year}: loaded {len(df)} events "
              f"({'EXCLUDED from primary' if year in args.exclude_years else 'included'})")

    if not all_dfs:
        print("ERROR: no files loaded, nothing to combine.")
        return

    # Column consistency check before concatenating
    col_sets = [frozenset(df.columns) for df in all_dfs]
    if len(set(col_sets)) > 1:
        print("WARNING: column mismatch across years detected. Using intersection.")
        common_cols = set.intersection(*[set(c) for c in col_sets])
        all_dfs = [df[list(common_cols)] for df in all_dfs]

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(args.output, index=False)

    primary = combined[~combined["excluded_from_primary"]]
    print(f"\nTotal combined: {len(combined)} events across {len(args.years)} years")
    print(f"Primary analysis set (excluding {args.exclude_years}): {len(primary)} events")
    print(f"[saved] {args.output}")

    primary_path = args.output.replace(".csv", "_primary.csv")
    primary.to_csv(primary_path, index=False)
    print(f"[saved] {primary_path}  <- use THIS for the FRL analysis scripts")


if __name__ == "__main__":
    main()
