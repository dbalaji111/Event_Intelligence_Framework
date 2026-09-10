"""
path_kde.py
===========
Time-indexed KDE posteriors across retrieved analogues (Steps 3 & 4 of the
redesigned pipeline), built by porting kde_engine_v3.py's mixture-model
logic and generalising it to run at EVERY relative day (t-15..t+15)
instead of once at a single fixed horizon.

What was ported from kde_engine_v3.py, and what changed:

  KEPT AS-IS:
    - dtw_increments(): DTW on path first-differences with Sakoe-Chiba
      band. No bug here, reused verbatim.
    - compute_three_rank_weight(): similarity^0.5 x quantile^0.2 x dtw^0.3
      combined weighting. Reused verbatim (constants exposed as module
      level W_SIM/W_QUANTILE/W_DTW so they can be tuned from one place).
    - analogue_prior_params(): mu = realised return, h = std of path
      increments, floor 1.0. Reused verbatim.
    - compute_mixture_posterior(): weighted Gaussian mixture, quantiles,
      tail probabilities, scenario separation. Reused verbatim except the
      bimodal detector (see FIXED below).

  FIXED:
    - Base-path fabrication: kde_v3 built the query event's own path for
      DTW matching by taking forward_return(date, N) and NEGATING it to
      stand in for what happened N days BEFORE the event -- that assumes
      symmetric pre/post-event drift, which is not generally true. Now
      that price_series.py gives continuous daily prices, the real t-15
      to t0 trajectory is used directly for both the query event and
      every analogue. No fabrication anywhere in this file.
    - Bimodal detection: kde_v3 counted density sign-changes, which is
      noisy (any small wiggle counts as a "mode"). Standardised on
      scipy.signal.find_peaks with a prominence floor, matching the
      detector already used in redflag_api_v5.py's compute_kde_posterior,
      so "bimodal" means the same thing everywhere in the codebase.
    - Single price source: kde_v3 had its own PriceData/Excel loader,
      a third independent price-loading implementation alongside the one
      in redflag_api_v5.py and price_series.py itself. This file takes
      pre-loaded pd.Series from price_series.py exclusively.

  GENERALISED (this is Steps 3 & 4 -- the actual new capability):
    - kde_v3 computed ONE mixture posterior at a single fixed horizon.
      kde_timeseries() sweeps every day from t-15 to t+15 (or t+1 to
      t+15 for a live forward-only forecast), so the posterior's shape
      is tracked continuously rather than evaluated once at the end.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm as scipy_norm
from scipy.signal import find_peaks

import price_series

REL_DAYS   = list(range(-15, 16))   # t-15 ... t0 ... t+15
FWD_DAYS   = list(range(1, 16))     # t+1 ... t+15  (live forecast window)
SHAPE_DAYS = list(range(-15, 1))    # t-15 ... t0    (DTW pre-event window)

TAIL_THRESHOLD = 10.0   # |r| > 10% = tail scenario, matches kde_v3
W_SIM      = 0.50
W_QUANTILE = 0.20
W_DTW      = 0.30
SIGMA_Q    = 20.0        # quantile-match Gaussian width, matches kde_v3


# ─────────────────────────────────────────────────────────────────────────
# PATH EXTRACTION (real data, no fabrication)
# ─────────────────────────────────────────────────────────────────────────

def extract_path(series: pd.Series, event_date, rel_days: list = REL_DAYS) -> dict:
    """{relative_day: price_or_None}. Missing days come back as None --
    never interpolated, never fabricated."""
    event_date = pd.Timestamp(event_date)
    path = {}
    for d in rel_days:
        target = event_date + pd.Timedelta(days=d)
        avail = series.index[series.index <= target]
        path[d] = None if len(avail) == 0 else float(series.loc[avail[-1]])
    return path


def path_to_returns(path: dict, base_day: int = 0) -> dict:
    """Converts a price path into % returns relative to base_day."""
    base = path.get(base_day)
    if base is None or base == 0:
        return {d: None for d in path}
    return {d: (None if v is None else round((v - base) / base * 100, 4))
            for d, v in path.items()}


def path_array(returns: dict, days: list):
    """Ordered array of returns over `days`, or None if any day is missing
    (DTW needs a complete, gap-free array)."""
    vals = [returns.get(d) for d in days]
    if any(v is None for v in vals):
        return None
    return np.array(vals, dtype=float)


# ─────────────────────────────────────────────────────────────────────────
# THREE-RANK WEIGHTING — ported from kde_engine_v3
# ─────────────────────────────────────────────────────────────────────────

def dtw_increments(path_a: np.ndarray, path_b: np.ndarray, radius: int = 2) -> float:
    """DTW on first-differences, Sakoe-Chiba banded. Verbatim port."""
    a = np.diff(path_a.astype(float))
    b = np.diff(path_b.astype(float))
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if abs(i - j) > radius:
                continue
            cost = (a[i - 1] - b[j - 1]) ** 2
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(np.sqrt(D[n, m]))


def compute_three_rank_weight(similarity: float,
                               base_quantile, analogue_quantile,
                               base_shape: np.ndarray, analogue_shape: np.ndarray,
                               all_dtw: list) -> float:
    """w = sim^0.5 x quantile_match^0.2 x dtw_match^0.3. Verbatim port,
    quantile inputs now real percentiles (0-100) from
    price_series.price_percentile rather than a bucketed category string."""
    sim_score = float(similarity)

    if base_quantile is None or analogue_quantile is None:
        quantile_score = 0.5  # neutral if percentile unavailable
    else:
        quantile_score = float(np.exp(-0.5 * ((base_quantile - analogue_quantile) / SIGMA_Q) ** 2))

    dtw_val = dtw_increments(base_shape, analogue_shape)
    med_dtw = np.median(all_dtw) if all_dtw else 1.0
    if med_dtw < 1e-6:
        med_dtw = 1.0
    dtw_score = float(np.exp(-dtw_val / med_dtw))

    return float((sim_score ** W_SIM) * (quantile_score ** W_QUANTILE) * (dtw_score ** W_DTW))


def analogue_prior_params(realised_r: float, path: np.ndarray):
    """mu = realised return, h = std of path increments, floor 1.0. Verbatim port."""
    increments = np.diff(path.astype(float))
    h = max(float(np.std(increments)), 1.0) if len(increments) else 1.0
    return float(realised_r), h


# ─────────────────────────────────────────────────────────────────────────
# MIXTURE POSTERIOR — ported from kde_engine_v3, bimodal detector fixed
# ─────────────────────────────────────────────────────────────────────────

def compute_mixture_posterior(mus: np.ndarray, sigmas: np.ndarray,
                               weights: np.ndarray, n_grid: int = 1000) -> dict:
    """f(r) = sum_i w_i * N(r | mu_i, sigma_i^2). Returns posterior stats
    only (not the full grid) -- callers needing the grid for plotting can
    reconstruct it cheaply from mus/sigmas/weights."""
    w_norm = weights / weights.sum()

    lo = float(np.min(mus - 3 * sigmas)) - 5
    hi = float(np.max(mus + 3 * sigmas)) + 5
    x = np.linspace(lo, hi, n_grid)
    dx = x[1] - x[0]

    density = np.zeros(n_grid)
    for mu, sigma, w in zip(mus, sigmas, w_norm):
        density += w * scipy_norm.pdf(x, mu, sigma)

    cdf = np.cumsum(density) * dx
    cdf = np.clip(cdf / max(cdf[-1], 1e-10), 0, 1)

    def q(p):
        idx = np.searchsorted(cdf, p)
        return float(x[min(idx, len(x) - 1)])

    mask_down10 = x <= -TAIL_THRESHOLD
    mask_up10 = x >= TAIL_THRESHOLD
    p_down10 = float(np.trapz(density[mask_down10], x[mask_down10])) if mask_down10.any() else 0.0
    p_up10 = float(np.trapz(density[mask_up10], x[mask_up10])) if mask_up10.any() else 0.0

    tail_mask = np.abs(mus) > TAIL_THRESHOLD
    mid_mask = ~tail_mask
    tail_w = float(w_norm[tail_mask].sum()) if tail_mask.any() else 0.0
    medium_w = float(w_norm[mid_mask].sum()) if mid_mask.any() else 0.0

    # FIXED: find_peaks with a prominence floor, not raw sign-change
    # counting -- matches the detector used elsewhere in the codebase.
    peaks, _ = find_peaks(density, prominence=density.max() * 0.1)
    is_bimodal = len(peaks) >= 2

    return {
        "p5": round(q(0.05), 3), "p25": round(q(0.25), 3),
        "median": round(q(0.50), 3), "p75": round(q(0.75), 3),
        "p95": round(q(0.95), 3),
        "mean": round(float(np.average(mus, weights=w_norm)), 3),
        "std": round(float(np.sqrt(np.average(
            (mus - np.average(mus, weights=w_norm)) ** 2 + sigmas ** 2, weights=w_norm))), 3),
        "p_down10": round(max(0, min(1, p_down10)), 4),
        "p_up10": round(max(0, min(1, p_up10)), 4),
        "n_analogues": len(mus),
        "tail_weight": round(tail_w, 4),
        "medium_weight": round(medium_w, 4),
        "is_bimodal": is_bimodal,
    }


# ─────────────────────────────────────────────────────────────────────────
# GENERALISED: mixture swept across every relative day (Steps 3 & 4)
# ─────────────────────────────────────────────────────────────────────────

def compute_analogue_weights(price_source: pd.Series, query_date, analogues: list) -> dict:
    """
    Computes the three-rank weight for each analogue ONCE, using a single
    reference series (spot, by convention). Returns {event_id: weight}.

    This weight answers "how relevant is this historical analogue to
    today's regime" -- a property of the overall market state, not of
    any single futures leg. Reusing one weight vector across spot/M1..M6
    keeps the six series directly comparable (same effective analogue
    set, same emphasis), rather than each leg silently retrieving a
    different weighted mix via its own DTW pass.
    """
    base_path = extract_path(price_source, query_date, SHAPE_DAYS)
    base_returns = path_to_returns(base_path, base_day=0)
    base_shape = path_array(base_returns, SHAPE_DAYS)
    query_q = price_series.price_percentile(price_source, query_date)

    if base_shape is None:
        return {}

    dtw_cache = []
    prepared = []
    for a in analogues:
        a_date = a.get("date")
        eid = a.get("event_id") or a_date
        if not a_date:
            continue
        a_path_raw = extract_path(price_source, a_date, SHAPE_DAYS)
        a_returns = path_to_returns(a_path_raw, base_day=0)
        a_shape = path_array(a_returns, SHAPE_DAYS)
        if a_shape is None:
            continue
        dtw_val = dtw_increments(base_shape, a_shape)
        dtw_cache.append(dtw_val)
        prepared.append((eid, a, a_shape))

    weights = {}
    for eid, a, a_shape in prepared:
        sim = float(a.get("similarity", 0.5))
        a_q = price_series.price_percentile(price_source, a["date"])
        w = compute_three_rank_weight(
            similarity=sim, base_quantile=query_q, analogue_quantile=a_q,
            base_shape=base_shape, analogue_shape=a_shape, all_dtw=dtw_cache,
        )
        if w > 0:
            weights[eid] = w
    return weights


def kde_timeseries_multi(price_sources: dict, query_date, analogues: list,
                          days: list = None, weight_source: str = "spot") -> dict:
    """
    Same evolving day-by-day mixture as kde_timeseries(), but run across
    every price source in `price_sources` (e.g. spot, M1, M2, M3, M6),
    sharing ONE weight vector computed from `weight_source` (spot by
    default) so every leg is weighted by the same effective analogue set.

    Returns {leg_name: {relative_day: posterior_stats_or_None}}
    Time-series structure is preserved exactly as in kde_timeseries() --
    every leg gets its own full day-by-day evolving posterior, not a
    single collapsed number.
    """
    if days is None:
        days = REL_DAYS

    ref_series = price_sources.get(weight_source)
    if ref_series is None:
        return {leg: {d: None for d in days} for leg in price_sources}

    weights = compute_analogue_weights(ref_series, query_date, analogues)
    if not weights:
        return {leg: {d: None for d in days} for leg in price_sources}

    results = {}
    for leg_name, series in price_sources.items():
        leg_results = {}
        for d in days:
            mus, sigmas, wts = [], [], []
            for a in analogues:
                eid = a.get("event_id") or a.get("date")
                w = weights.get(eid)
                if w is None or not a.get("date"):
                    continue
                a_path_raw = extract_path(series, a["date"], SHAPE_DAYS + [d])
                a_returns = path_to_returns(a_path_raw, base_day=0)
                a_shape = path_array(a_returns, SHAPE_DAYS)
                target_r = a_returns.get(d)
                if a_shape is None or target_r is None:
                    continue
                mu, h = analogue_prior_params(target_r, a_shape)
                mus.append(mu); sigmas.append(h); wts.append(w)

            if len(mus) < 3:
                leg_results[d] = None
                continue
            leg_results[d] = compute_mixture_posterior(
                np.array(mus), np.array(sigmas), np.array(wts))
        results[leg_name] = leg_results
    return results


def kde_timeseries(price_source: pd.Series, query_date, analogues: list,
                    days: list = None) -> dict:
    """
    Step 3/4 entry point: sweeps the three-rank-weighted mixture posterior
    across every relative day in `days` (default: full t-15..t+15 for
    historical backtesting; pass FWD_DAYS for a live forward-only
    forecast cone).

    price_source: a single pd.Series (spot, or one futures leg) from
                  price_series.load_all_price_sources()
    query_date: the event being analysed (live query or a backtest point)
    analogues: list of dicts with "date" and "similarity"

    Returns {relative_day: posterior_stats_dict_or_None}. None means fewer
    than 3 analogues had complete data for that day -- not zero-filled.
    """
    if days is None:
        days = REL_DAYS

    base_path = extract_path(price_source, query_date, SHAPE_DAYS)
    base_returns = path_to_returns(base_path, base_day=0)
    base_shape = path_array(base_returns, SHAPE_DAYS)
    query_q = price_series.price_percentile(price_source, query_date)

    results = {}
    if base_shape is None:
        # query event itself doesn't have a full pre-event window on this
        # price source -- every day comes back None rather than guessing.
        return {d: None for d in days}

    for d in days:
        mus, sigmas, weights, dtw_cache = [], [], [], []
        prepared = []
        for a in analogues:
            a_date = a.get("date")
            if not a_date:
                continue
            a_path_raw = extract_path(price_source, a_date, SHAPE_DAYS + [d])
            a_returns = path_to_returns(a_path_raw, base_day=0)
            a_shape = path_array(a_returns, SHAPE_DAYS)
            target_r = a_returns.get(d)
            if a_shape is None or target_r is None:
                continue
            dtw_val = dtw_increments(base_shape, a_shape)
            dtw_cache.append(dtw_val)
            prepared.append((a, a_shape, target_r))

        if len(prepared) < 3:
            results[d] = None
            continue

        for a, a_shape, target_r in prepared:
            sim = float(a.get("similarity", 0.5))
            a_q = price_series.price_percentile(price_source, a["date"])
            w = compute_three_rank_weight(
                similarity=sim, base_quantile=query_q, analogue_quantile=a_q,
                base_shape=base_shape, analogue_shape=a_shape, all_dtw=dtw_cache,
            )
            if w <= 0:
                continue
            mu, h = analogue_prior_params(target_r, a_shape)
            mus.append(mu); sigmas.append(h); weights.append(w)

        if len(mus) < 3:
            results[d] = None
            continue

        results[d] = compute_mixture_posterior(np.array(mus), np.array(sigmas), np.array(weights))

    return results


def build_calendar_spreads(price_sources: dict) -> dict:
    """{"M1": s, "M2": s, ...} -> {"M1-M2": spread_series, ...}"""
    spreads = {}
    legs = sorted([k for k in price_sources if k.startswith("M")],
                  key=lambda k: int(k[1:]))
    for a, b in zip(legs, legs[1:]):
        common = price_sources[a].index.intersection(price_sources[b].index)
        spreads[f"{a}-{b}"] = price_sources[a].loc[common] - price_sources[b].loc[common]
    return spreads