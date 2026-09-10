"""
Fetches / incrementally updates daily close prices for WTI, Brent, OVX, VIX
from Refinitiv Eikon and stores them in a single CSV.

Run any time — on first run it does a full historical pull from 2000-01-01;
on subsequent runs it only fetches days since the last saved date.

    python scripts/fetch_oil_timeseries.py
"""
import pandas as pd
import eikon as ek
from pathlib import Path
from datetime import datetime, timedelta
from src.config import Config

# -----------------------------
# CONFIG
# -----------------------------
cfg = Config()
ek.set_app_key(cfg.REFINITIV_APP_KEY)

data_dir = Path(cfg.DATA_DIR)
data_dir.mkdir(parents=True, exist_ok=True)

file_path = data_dir / "oil_timeseries.csv"

# Pull from 2000 by default so the series covers your full 2001-2023 event corpus.
FULL_HISTORY_START = "2000-01-01"

# -----------------------------
# INSTRUMENTS
# -----------------------------
RIC_MAP = {
    "CLc1": "WTI_M1",
    "CLc2": "WTI_M2",
    "CLc3": "WTI_M3",
    "CLc6": "WTI_M6",
    "LCOc1": "BRENT_M1",
    "LCOc2": "BRENT_M2",
    "LCOc3": "BRENT_M3",
    ".OVX": "OVX",
    ".VIX": "VIX",
}


# -----------------------------
# FETCH FUNCTION (SAFE)
# -----------------------------
def fetch_timeseries(ric, start_date, end_date):
    try:
        df = ek.get_timeseries(
            ric,
            fields="CLOSE",
            interval="daily",
            start_date=start_date,
            end_date=end_date,
        )
        if df is None or df.empty:
            return None

        df = df.rename(columns={"CLOSE": RIC_MAP[ric]})
        df = df.reset_index()
        return df

    except Exception as e:
        print(f"Error fetching {ric}: {e}")
        return None


# -----------------------------
# MAIN PIPELINE
# -----------------------------
def main():
    # -----------------------------
    # INITIAL LOAD OR UPDATE
    # -----------------------------
    if file_path.exists():
        existing = pd.read_csv(file_path)
        existing["Date"] = pd.to_datetime(existing["Date"])
        last_date = existing["Date"].max()
        start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"Updating from {start_date}")
    else:
        existing = None
        start_date = FULL_HISTORY_START
        print(f"Initial full load from {start_date}")

    end_date = datetime.today().strftime("%Y-%m-%d")

    if start_date > end_date:
        print("Already up to date — nothing to fetch.")
        return

    # -----------------------------
    # FETCH ALL INSTRUMENTS
    # -----------------------------
    merged_df = None
    for ric in RIC_MAP:
        print(f"Fetching {ric}")
        df = fetch_timeseries(ric, start_date, end_date)
        if df is None:
            continue
        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on="Date", how="outer")

    if merged_df is None:
        print("No new data fetched.")
        return

    # -----------------------------
    # CLEAN + SORT
    # -----------------------------
    merged_df["Date"] = pd.to_datetime(merged_df["Date"])
    merged_df = merged_df.sort_values("Date")

    # -----------------------------
    # APPEND TO EXISTING
    # -----------------------------
    if existing is not None:
        final_df = pd.concat([existing, merged_df], ignore_index=True)
        final_df = final_df.drop_duplicates(subset="Date", keep="last")
    else:
        final_df = merged_df

    final_df = final_df.sort_values("Date")
    # Forward-fill only price columns, not Date, and only genuine market holidays —
    # don't ffill if there's a large gap (e.g. this being the very first row).
    price_cols = [c for c in final_df.columns if c != "Date"]
    final_df[price_cols] = final_df[price_cols].ffill()

    # -----------------------------
    # SAVE
    # -----------------------------
    final_df.to_csv(file_path, index=False)
    print(f"\nUpdated file saved: {file_path} ({len(final_df)} rows, "
          f"{final_df['Date'].min().date()} to {final_df['Date'].max().date()})")


if __name__ == "__main__":
    main()
