"""
price_series.py
================
Single source of truth for WTI price data and event returns.

All numerical return values (r1, r5, r7, r15, r30) are computed live from
this module, from actual price CSVs on disk — never baked into the corpus
JSONL, never duplicated across endpoints. Both /analyze and /retrieve
import this module so they can never disagree with each other again.

Price sources combined into one continuous series:
  1. spot_and_spreads_data.xlsx  (1983-01-10 -> 2024-11-19, WTI Cushing spot)
  2. oil_timeseries.csv          (Eikon CLc1 continuous futures, 2014 -> today,
                                   used only to extend past the spot file's
                                   last date)
Where both overlap, the spot file wins (it's the authoritative series your
metadata / thesis returns were originally built against).
"""
from __future__ import annotations
import pandas as pd
from pathlib import Path
from functools import lru_cache

SPOT_XLSX_PATH = "crude_oil_data/oil_price_data/spot_and_spreads_data.xlsx"
SPOT_XLSX_DATE_COL = "Name"
SPOT_XLSX_PRICE_COL = "Crude Oil-WTI Spot Cushing U$/BBL"

FUTURES_CSV_PATH = "data/oil_timeseries.csv"
FUTURES_PRICE_COL = "WTI_M1"

RETURN_WINDOWS = {"r1": 1, "r3": 3, "r5": 5, "r7": 7, "r10": 10, "r15": 15}


@lru_cache(maxsize=1)
def load_wti_series(base_path: str = ".") -> pd.Series:
    """
    Loads and combines the spot + futures price sources into one
    continuous daily WTI series, indexed by date. Cached after first call
    within a process — restart the server to pick up freshly updated
    price files.
    """
    base = Path(base_path)

    spot_path = base / SPOT_XLSX_PATH
    if not spot_path.exists():
        raise FileNotFoundError(f"Spot price file not found: {spot_path}")

    spot = pd.read_excel(spot_path)
    spot = spot.rename(columns={SPOT_XLSX_DATE_COL: "Date",
                                 SPOT_XLSX_PRICE_COL: "price"})
    spot["Date"] = pd.to_datetime(spot["Date"])
    spot = spot[["Date", "price"]].dropna().sort_values("Date")
    combined = spot.set_index("Date")["price"]

    futures_path = base / FUTURES_CSV_PATH
    if futures_path.exists():
        futures = pd.read_csv(futures_path)
        futures["Date"] = pd.to_datetime(futures["Date"])
        futures = futures.sort_values("Date").set_index("Date")
        if FUTURES_PRICE_COL in futures.columns:
            fut_series = futures[FUTURES_PRICE_COL].dropna()
            tail = fut_series[fut_series.index > combined.index.max()]
            combined = pd.concat([combined, tail]).sort_index()

    combined = combined[~combined.index.duplicated(keep="first")]
    return combined


FUTURES_LEG_COLS = {
    "M1": "WTI_M1", "M2": "WTI_M2", "M3": "WTI_M3",
    # WTI_M4/M5 are not in your current RIC_MAP (only M1,M2,M3,M6 are pulled
    # by fetch_oil_timeseries.py) -- add CLc4/CLc5 to RIC_MAP there if you
    # want the full curve; this loader just skips legs it can't find rather
    # than fabricating a gap-filled series.
    "M6": "WTI_M6",
}


@lru_cache(maxsize=1)
def load_futures_curve(base_path: str = ".") -> dict:
    """
    Loads each individual M1-M6 futures leg as its own series, from the
    same oil_timeseries.csv that fetch_oil_timeseries.py maintains.

    Coverage note: this data starts 2014 (Eikon CLc1-CLc6 entitlement
    limit, confirmed during tonight's build). Analogues dated before 2014
    will simply be absent from any dict this function returns for -- do
    NOT silently backfill from the spot series to plug that gap; a
    reconstructed M1 is not the same instrument as the real one and would
    quietly corrupt the calendar spread math downstream.
    """
    base = Path(base_path)
    futures_path = base / FUTURES_CSV_PATH
    if not futures_path.exists():
        log_msg = f"Futures curve file not found: {futures_path}"
        raise FileNotFoundError(log_msg)

    futures = pd.read_csv(futures_path)
    futures["Date"] = pd.to_datetime(futures["Date"])
    futures = futures.sort_values("Date").set_index("Date")

    legs = {}
    for leg_name, col in FUTURES_LEG_COLS.items():
        if col in futures.columns:
            legs[leg_name] = futures[col].dropna()
    return legs


@lru_cache(maxsize=1)
def load_all_price_sources(base_path: str = ".") -> dict:
    """
    Single entry point for path_kde.py: returns
        {"spot": series, "M1": series, "M2": series, "M3": series, "M6": series}
    "spot" is the same combined spot+futures-tail series used everywhere
    else (load_wti_series) -- kept separate from the raw M1 leg since spot
    and front-month futures are related but distinct instruments.
    """
    sources = {"spot": load_wti_series(base_path)}
    sources.update(load_futures_curve(base_path))
    return sources


def price_percentile(series: pd.Series, date, lookback_years: int = 5) -> float | None:
    """
    Ported from kde_engine_v3.PriceData.price_percentile.
    Percentile rank of the price at `date` relative to the trailing
    `lookback_years` of the same series. Used as the "quantile regime"
    rank in three-rank analogue weighting.
    """
    date = pd.Timestamp(date)
    avail = series.index[series.index <= date]
    if len(avail) == 0:
        return None
    t = avail[-1]
    p_now = series.loc[t]
    start = t - pd.Timedelta(days=lookback_years * 365)
    hist = series.loc[start:t].dropna()
    if len(hist) < 50:
        return None
    return float((hist < p_now).mean() * 100)


COVID_START = pd.Timestamp("2020-02-15")
COVID_END   = pd.Timestamp("2020-07-01")


def is_covid_period(date) -> bool:
    """
    Ported from kde_engine_v3.py's exclude_covid logic. WTI briefly went
    negative on 2020-04-20 -- a contract-mechanics artifact (storage
    capacity exhaustion at Cushing forcing near-expiry longs to pay to
    exit), not a supply/demand shock comparable to the geopolitical
    events this corpus otherwise represents. Callers retrieving analogues
    should exclude this window rather than let percentage-return math
    (which breaks down once price crosses zero) silently corrupt the
    mixture posterior with a -250%-style non-return.
    """
    d = pd.Timestamp(date)
    return COVID_START <= d <= COVID_END


def compute_return(series: pd.Series, event_date, window_days: int) -> float | None:
    """
    % return from event_date to event_date + window_days calendar days,
    snapped to the nearest available prior trading day on each end.
    Returns None if either side of the window falls outside the series,
    OR if the price crosses zero within the window (percentage return is
    undefined once price is non-positive on either end -- this is a
    principled exclusion, not a magnitude-based outlier clip. See
    is_covid_period() for excluding the surrounding window at the
    analogue-retrieval level, which is the correct fix for the WTI
    negative-price event specifically.
    """
    event_date = pd.Timestamp(event_date)

    if event_date not in series.index:
        prior = series.index[series.index <= event_date]
        if len(prior) == 0:
            return None
        event_date = prior[-1]

    target_date = event_date + pd.Timedelta(days=window_days)
    future = series.index[series.index >= target_date]
    if len(future) == 0:
        return None

    p0 = series.loc[event_date]
    p1 = series.loc[future[0]]
    if pd.isna(p0) or pd.isna(p1):
        return None
    if p0 <= 0 or p1 <= 0:
        # percentage return is undefined once either endpoint is
        # non-positive -- this is what actually made the April 2020
        # window produce a nonsense -250%, not the magnitude itself.
        return None

    return round(float((p1 - p0) / p0 * 100), 4)


def compute_all_returns(series: pd.Series, event_date) -> dict:
    """Returns {"r1": ..., "r3": ..., "r5": ..., "r7": ..., "r10": ..., "r15": ...}"""
    return {label: compute_return(series, event_date, days)
            for label, days in RETURN_WINDOWS.items()}


def build_analogue_table(series: pd.Series, analogues: list) -> list:
    """
    Step 2 deliverable: event x similarity x multi-horizon return table.
    `analogues` is a list of dicts each with at least "date" and "similarity"
    (the output of retrieve_analogues / /retrieve). Returns are computed
    live from `series` -- never read from a precomputed CSV.
    """
    rows = []
    for a in analogues:
        returns = compute_all_returns(series, a["date"]) if a.get("date") else {}
        rows.append({
            "event_id":   a.get("event_id") or a.get("analogue_id"),
            "date":       a.get("date"),
            "similarity": a.get("similarity"),
            **returns,
        })
    return rows