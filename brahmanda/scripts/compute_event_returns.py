"""
Computes real forward returns (1d, 5d, 7d, 15d, 30d) for each event in the
metadata corpus, using the price series fetched by fetch_oil_timeseries.py.

This replaces the r7/r15 fields that were duplicated for avg_return_1d and
avg_return_5d in the scorer's outcome_stats. Instead of aliasing one window
onto two labels, every return field here is computed from its own actual
forward-looking price window.

Assumption: WTI_M1 (front-month WTI) is used as the reference price series
for event returns, matching the "Crude Oil-WTI Spot Cushing" convention in
your existing metadata. Change PRICE_COL below to "BRENT_M1" if you'd
rather benchmark against Brent.

Run after fetch_oil_timeseries.py, any time you want to refresh event
returns against the latest price data:

    python scripts/compute_event_returns.py
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from src.config import Config

cfg = Config()

PRICE_COL = "WTI_M1"          # change to "BRENT_M1" to benchmark against Brent
PRICE_FILE = Path(cfg.DATA_DIR) / "oil_timeseries.csv"

META_PATH = "memory_2001_2023/df_major_metadata.csv"
EMB_PATH  = "memory_2001_2023/df_major_embeddings_RECOMPUTED.npz"
OUT_PATH  = "scorer_corpus.jsonl"

RETURN_WINDOWS = {"r1": 1, "r5": 5, "r7": 7, "r15": 15, "r30": 30}


def load_price_series():
    prices = pd.read_csv(PRICE_FILE)
    prices["Date"] = pd.to_datetime(prices["Date"])
    prices = prices.sort_values("Date").set_index("Date")
    if PRICE_COL not in prices.columns:
        raise ValueError(
            f"'{PRICE_COL}' not found in {PRICE_FILE}. "
            f"Available columns: {list(prices.columns)}"
        )
    series = prices[PRICE_COL].dropna()
    return series


def pct_return(series, event_date, window_days):
    """
    % return from event_date to event_date + window_days trading days
    (approximated as calendar days, then snapped to nearest available price).
    Returns None if either side of the window falls outside the price series.
    """
    if event_date not in series.index:
        # snap to nearest prior trading day
        prior = series.index[series.index <= event_date]
        if len(prior) == 0:
            return None
        event_date = prior[-1]

    target_date = event_date + pd.Timedelta(days=window_days)
    future = series.index[series.index >= target_date]
    if len(future) == 0:
        return None  # window extends beyond available price data

    p0 = series.loc[event_date]
    p1 = series.loc[future[0]]
    if p0 == 0 or pd.isna(p0) or pd.isna(p1):
        return None
    return round(float((p1 - p0) / p0 * 100), 4)


def main():
    print(f"Loading price series ({PRICE_COL}) from {PRICE_FILE} ...")
    series = load_price_series()
    print(f"  {len(series)} price points, {series.index.min().date()} to {series.index.max().date()}")

    print(f"Loading metadata from {META_PATH} ...")
    meta = pd.read_csv(META_PATH)
    meta["date"] = pd.to_datetime(meta["date"], errors="coerce")
    print(f"  {len(meta)} events")

    print(f"Loading embeddings from {EMB_PATH} ...")
    npz = np.load(EMB_PATH, allow_pickle=True)
    embs, ids = npz["embeddings"], npz["ids"]
    id_to_emb = {str(i): e for i, e in zip(ids, embs)}

    n_written = n_skipped_no_id = n_skipped_no_emb = n_skipped_no_date = 0
    n_return_gaps = 0

    with open(OUT_PATH, "w") as f:
        for _, row in meta.iterrows():
            raw_id = row.get("document_id")
            if pd.isna(raw_id):
                n_skipped_no_id += 1
                continue
            doc_id = str(raw_id)

            emb = id_to_emb.get(doc_id)
            if emb is None:
                n_skipped_no_emb += 1
                continue

            event_date = row.get("date")
            if pd.isna(event_date):
                n_skipped_no_date += 1
                continue

            returns = {}
            any_gap = False
            for label, days in RETURN_WINDOWS.items():
                r = pct_return(series, event_date, days)
                returns[label] = r
                if r is None:
                    any_gap = True
            if any_gap:
                n_return_gaps += 1

            record = {
                "document_id": doc_id,
                "date": str(event_date.date()) if pd.notna(event_date) else "",
                "text": str(row.get("text", ""))[:800],
                "event_type": str(row.get("Event type", row.get("Event_type", ""))),
                "r1": returns["r1"],
                "r5": returns["r5"],
                "r7": returns["r7"],
                "r15": returns["r15"],
                "r30": returns["r30"],
                "embedding": emb.tolist(),
            }
            f.write(json.dumps(record) + "\n")
            n_written += 1

    print()
    print(f"written              = {n_written}")
    print(f"skipped (no id)      = {n_skipped_no_id}")
    print(f"skipped (no emb)     = {n_skipped_no_emb}")
    print(f"skipped (no date)    = {n_skipped_no_date}")
    print(f"events w/ any gap    = {n_return_gaps}  (price window missing for at least one horizon — usually recent events near the end of your price series)")
    print(f"output file          = {Path(OUT_PATH).resolve()}")


if __name__ == "__main__":
    main()
