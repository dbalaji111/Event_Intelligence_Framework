"""
Incrementally updates OVX (CBOE Crude Oil Volatility Index) data from
Yahoo Finance, appending only the gap since the file's last saved date.

Fixes yfinance's flattened multi-index column export (the raw download has
"Ticker"/"Date" as junk data rows and puts dates under a column literally
named "Price" - this script cleans that up on the way in and keeps the
saved CSV in a normal Date/Close/High/Low/Open/Volume shape).

Run any time you want to refresh OVX data:

    python scripts/update_ovx_yahoo.py
"""
import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

OUT_PATH = Path("ovx_data.csv")
TICKER = "^OVX"


def clean_yf_download(df):
    """
    Handles two shapes yfinance can return:
      1. MultiIndex columns, e.g. ('Close', '^OVX') - flatten to 'Close'.
      2. The flattened-export shape some yfinance versions/CSVs produce,
         where the first two rows are junk ("Ticker"/"Date" labels) and
         the date column is literally named "Price".
    Normalizes either into a plain Date/Close/High/Low/Open/Volume frame.
    """
    df = df.copy()

    # Case 1: MultiIndex columns from a live yf.download() call
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]  # drop the ticker level

    # Case 2: flattened junk-row export (from a previously mis-saved CSV)
    if "Price" in df.columns and df["Price"].iloc[0] in ("Ticker", "Date"):
        df = df[~df["Price"].isin(["Ticker", "Date"])]
        df = df.rename(columns={"Price": "Date"})

    if "Date" not in df.columns:
        df = df.reset_index().rename(columns={"index": "Date"})

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    for col in ["Close", "High", "Low", "Open", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values("Date").reset_index(drop=True)


def main():
    if OUT_PATH.exists():
        existing = pd.read_csv(OUT_PATH)
        existing = clean_yf_download(existing)
        last_date = existing["Date"].max()
        start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"Existing file last date: {last_date.date()}, updating from {start_date}")
    else:
        existing = None
        start_date = "2020-01-01"
        print(f"No existing file - initial load from {start_date}")

    end_date = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")  # yfinance end is exclusive

    if start_date >= end_date:
        print("Already up to date.")
        return

    raw = yf.download(TICKER, start=start_date, end=end_date)
    if raw is None or raw.empty:
        print("No new data returned.")
        return

    new_data = clean_yf_download(raw.reset_index())

    if existing is not None:
        combined = pd.concat([existing, new_data], ignore_index=True)
        combined = combined.drop_duplicates(subset="Date", keep="last")
    else:
        combined = new_data

    combined = combined.sort_values("Date").reset_index(drop=True)
    combined.to_csv(OUT_PATH, index=False)
    print(f"Saved: {OUT_PATH} ({len(combined)} rows, "
          f"{combined['Date'].min().date()} to {combined['Date'].max().date()})")


if __name__ == "__main__":
    main()