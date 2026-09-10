"""
kde_engine_v4.py  —  Bayesian Mixture Model with Scenario Separation
=====================================================================
FIXED VERSION -- see "FIX APPLIED" comments below for the exact change.

Core idea:
  Each of the 25 analogues contributes an INDIVIDUAL Gaussian prior
  centred on its own realised return with its own bandwidth.
  The posterior is a weighted mixture of these 25 priors.

  f_posterior(r) = Σᵢ w̃ᵢ × N(r | μᵢ, hᵢ²)

  where:
    μᵢ = realised forward return of analogue i (from real daily prices)
    hᵢ = path uncertainty of analogue i (std of path increments)
    w̃ᵢ = normalised weight from 3-rank scoring

Three-rank weighting (time-series preserving):
  w_i = sim_score^0.5 × quantile_score^0.2 × dtw_score^0.3

  Rank 1 — BERT cosine similarity (semantic context match)
  Rank 2 — Quantile regime match (same price level regime)
  Rank 3 — DTW on path INCREMENTS (trajectory shape, not cumulative)
            with Sakoe-Chiba band to preserve temporal structure

Scenario separation:
  The mixture naturally separates into:
    TAIL scenarios   — analogues where |r| > 10%  (crises/shocks)
    MEDIUM scenarios — analogues where |r| ≤ 10%  (normal moves)

  This produces BIMODAL posteriors for geopolitical events
  that GARCH cannot replicate.

Minimum 25 analogues enforced via adaptive tau relaxation.

Usage
-----
    python kde_engine_v4.py \
        --analogues retrieval_output_v3/per_event \
        --prices    "data/oil_prices_full.xlsx" \
        --ovx       "crude_oil_data/oil_price_data/CBOE Crude Oil Volatility Historical Data.csv" \
        --output    kde_output_v4 \
        --horizon   15

Dependencies
------------
    pip install numpy pandas scipy scikit-learn tqdm openpyxl arch
"""

from __future__ import annotations
import argparse, json, logging, sys, warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import norm as scipy_norm, pearsonr, spearmanr
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

MIN_ANALOGUES  = 25       # minimum analogues — hard requirement
TAU_FALLBACKS  = [0.90, 0.87, 0.85, 0.82, 0.80, 0.77, 0.75, 0.70, 0.65, 0.0]
TAU_LEVELS     = [0.0, 0.75, 0.80, 0.85, 0.90]
PRICE_NODES    = [-30, -15, -7, 0, 7, 15, 30]
HORIZONS       = [7, 15, 30]

TAIL_THRESHOLD = 10.0     # |r| > 10% = tail scenario
COVID_START    = pd.Timestamp("2020-02-15")
COVID_END      = pd.Timestamp("2020-07-01")

# Weight exponents for 3-rank scoring
W_SIM      = 0.50   # semantic similarity
W_QUANTILE = 0.20   # quantile regime match
W_DTW      = 0.30   # DTW path shape

# Quantile category → percentile midpoint (for distance calculation)
QUANTILE_MIDPOINTS = {
    "5%":       2.5,  "5%-20%":  12.5, "20%-30%": 25.0,
    "30%-40%":  35.0, "40%-60%": 50.0, "60%-80%": 70.0,
    "80%-90%":  85.0, "90%+":    95.0,
    "0%-20%":   10.0, "20%-40%": 30.0, "80%-100%": 90.0,
}


def bucket_for_percentile(pctile: float) -> str:
    """
    FIX APPLIED: this function is new. Converts a real numeric price
    percentile (0-100) into the same bucket scheme QUANTILE_MIDPOINTS
    and the analogue corpus's quantile_cat field already use.

    This is what base_quantile inside run_event_v3() should have been
    computed with all along -- previously it was hardcoded to the
    literal string "40%-60%" for every single event regardless of the
    real price regime, which meant the quantile-regime weighting term
    was always comparing analogues against a fictional "always normal"
    reference point instead of today's actual conditions.
    """
    if pctile < 5:   return "5%"
    if pctile < 20:  return "5%-20%"
    if pctile < 30:  return "20%-30%"
    if pctile < 40:  return "30%-40%"
    if pctile < 60:  return "40%-60%"
    if pctile < 80:  return "60%-80%"
    if pctile < 90:  return "80%-90%"
    return "90%+"


# ─────────────────────────────────────────────────────────────────────────────
# PRICE DATA
# ─────────────────────────────────────────────────────────────────────────────

class PriceData:
    def __init__(self, prices_path: Path,
                 ovx_path: Optional[Path] = None):
        log.info(f"[price] loading {prices_path.name}")
        df = pd.read_excel(prices_path)
        df = df.rename(columns={df.columns[0]: "Date"})
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).set_index("Date").sort_index()

        rename = {
            "Crude Oil-WTI Spot Cushing U$/BBL":  "WTI",
            "Crude Oil WTI NYMEX Close M U$/BBL":  "WTI_M1",
            "NYMEX Crude Oil WTI M1-M2 Spread":    "M1M2",
            "Crude Oil Brent US M1 U$/BBL":         "BM1",
            "Crude Oil Brent US M2 U$/BBL":         "BM2",
            "Crude Oil Brent US M3 U$/BBL":         "BM3",
        }
        df = df.rename(columns={k:v for k,v in rename.items() if k in df.columns})
        if "BM1" in df.columns and "BM2" in df.columns:
            df["BM1M2"] = df["BM1"] - df["BM2"]
        df = df.ffill(limit=5)

        if ovx_path and ovx_path.exists():
            ovx = pd.read_csv(ovx_path)
            ovx["Date"] = pd.to_datetime(
                ovx["Date"], format="%m/%d/%Y", errors="coerce")
            ovx = ovx.dropna(subset=["Date"]).set_index("Date").sort_index()
            ovx = ovx.rename(columns={"Price": "OVX"})
            df   = df.join(ovx[["OVX"]], how="left")

        self.df = df.ffill(limit=5)
        log.info(f"[price]   {len(self.df)} rows  "
                 f"{self.df.index[0].date()} → {self.df.index[-1].date()}")
        log.info(f"[price]   cols: "
                 f"{[c for c in ['WTI','M1M2','OVX','BM1','BM2'] if c in self.df.columns]}")

    def _nearest(self, date: pd.Timestamp,
                 max_offset: int = 5) -> Optional[pd.Timestamp]:
        if date in self.df.index:
            return date
        cands = self.df.index[
            abs(self.df.index - date) <= pd.Timedelta(days=max_offset)]
        return cands[abs(cands - date).argmin()] if len(cands) else None

    def forward_return(self, date: pd.Timestamp,
                       horizon: int, col: str = "WTI") -> Optional[float]:
        col = col if col in self.df.columns else "WTI"
        t0  = self._nearest(date)
        if t0 is None: return None
        p0  = self.df.loc[t0, col]
        if not np.isfinite(p0) or p0 == 0: return None
        t1  = self._nearest(date + pd.Timedelta(days=horizon), max_offset=10)
        if t1 is None: return None
        p1  = self.df.loc[t1, col]
        return float((p1-p0)/abs(p0)*100) if np.isfinite(p1) else None

    def get_val(self, date: pd.Timestamp,
                col: str) -> Optional[float]:
        if col not in self.df.columns: return None
        t = self._nearest(date)
        if t is None: return None
        v = self.df.loc[t, col]
        return float(v) if np.isfinite(v) else None

    def spread_at(self, date): return self.get_val(date, "M1M2")
    def ovx_at(self, date):    return self.get_val(date, "OVX")
    def wti_at(self, date):
        for c in ["WTI", "WTI_M1"]:
            v = self.get_val(date, c)
            if v: return v
        return None

    def price_percentile(self, date: pd.Timestamp,
                          lookback_years: int = 5) -> Optional[float]:
        """Return percentile of WTI price relative to past N years."""
        col = "WTI" if "WTI" in self.df.columns else "WTI_M1"
        if col not in self.df.columns: return None
        t = self._nearest(date)
        if t is None: return None
        p_now  = self.df.loc[t, col]
        start  = t - pd.Timedelta(days=lookback_years*365)
        hist   = self.df.loc[start:t, col].dropna()
        if len(hist) < 50: return None
        return float((hist < p_now).mean() * 100)


# ─────────────────────────────────────────────────────────────────────────────
# DTW  (increments + Sakoe-Chiba)
# ─────────────────────────────────────────────────────────────────────────────

def dtw_increments(path_a: np.ndarray, path_b: np.ndarray,
                   radius: int = 2) -> float:
    """
    DTW on first-differences of price paths.
    Preserves autocorrelation / time-series structure.
    Sakoe-Chiba band prevents unrealistic temporal warping.
    Squared distance penalises large regime mismatches.
    """
    a = np.diff(path_a.astype(float))
    b = np.diff(path_b.astype(float))
    n, m = len(a), len(b)
    D = np.full((n+1, m+1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n+1):
        for j in range(1, m+1):
            if abs(i-j) > radius:
                continue
            cost = (a[i-1] - b[j-1]) ** 2   # squared distance
            D[i,j] = cost + min(D[i-1,j], D[i,j-1], D[i-1,j-1])
    return float(np.sqrt(D[n, m]))


# ─────────────────────────────────────────────────────────────────────────────
# THREE-RANK WEIGHTING
# ─────────────────────────────────────────────────────────────────────────────

def compute_three_rank_weight(
    similarity:    float,
    base_quantile: str,
    analogue_quantile: str,
    base_path:     np.ndarray,
    analogue_path: np.ndarray,
    all_dtw:       List[float],
    sigma_q:       float = 20.0,
) -> float:
    """
    Combined weight from three ranks — time-series preserving.

    Rank 1 (50%): BERT cosine similarity
    Rank 2 (20%): Quantile regime match — Gaussian on percentile distance
    Rank 3 (30%): DTW on path increments — exponential decay
    """
    # ── Rank 1: semantic similarity (already 0-1) ─────────────────────────
    sim_score = float(similarity)

    # ── Rank 2: quantile regime match ─────────────────────────────────────
    q_base = QUANTILE_MIDPOINTS.get(str(base_quantile).strip(), 50.0)
    q_anal = QUANTILE_MIDPOINTS.get(str(analogue_quantile).strip(), 50.0)
    quantile_score = float(np.exp(-0.5 * ((q_base - q_anal) / sigma_q) ** 2))

    # ── Rank 3: DTW score — normalise by median DTW of all analogues ──────
    dtw_val = dtw_increments(base_path, analogue_path)
    med_dtw = np.median(all_dtw) if all_dtw else 1.0
    if med_dtw < 1e-6: med_dtw = 1.0
    dtw_score = float(np.exp(-dtw_val / med_dtw))

    # ── Combined: power-weighted product ──────────────────────────────────
    weight = (sim_score ** W_SIM) * (quantile_score ** W_QUANTILE) * (dtw_score ** W_DTW)
    return float(weight)


# ─────────────────────────────────────────────────────────────────────────────
# INDIVIDUAL ANALOGUE PRIOR
# ─────────────────────────────────────────────────────────────────────────────

def analogue_prior_params(
    realised_r: float,
    path:       np.ndarray,
) -> Tuple[float, float]:
    """
    Each analogue contributes N(μ, h²) where:
      μ = realised forward return
      h = path uncertainty = std of path INCREMENTS
          (volatile analogues → wider prior → appropriate uncertainty)
    Minimum bandwidth = 1.0% to avoid degenerate priors.
    """
    increments = np.diff(path.astype(float))
    h = max(float(np.std(increments)), 1.0)
    return float(realised_r), h


# ─────────────────────────────────────────────────────────────────────────────
# MIXTURE POSTERIOR
# ─────────────────────────────────────────────────────────────────────────────

def compute_mixture_posterior(
    mus:     np.ndarray,   # shape (N,) — mean of each prior
    sigmas:  np.ndarray,   # shape (N,) — std of each prior
    weights: np.ndarray,   # shape (N,) — unnormalised weights
    n_grid:  int = 1000,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Compute the weighted Gaussian mixture posterior:
      f(r) = Σᵢ w̃ᵢ × N(r | μᵢ, σᵢ²)

    Returns:
      x_grid   — evaluation points
      density  — mixture density values
      stats    — posterior statistics
    """
    w_norm = weights / weights.sum()

    # Grid covering 99% of the mixture support
    lo = min(mus - 3*sigmas) - 5
    hi = max(mus + 3*sigmas) + 5
    x  = np.linspace(lo, hi, n_grid)
    dx = x[1] - x[0]

    # Mixture density
    density = np.zeros(n_grid)
    for mu, sigma, w in zip(mus, sigmas, w_norm):
        density += w * scipy_norm.pdf(x, mu, sigma)

    # CDF
    cdf = np.cumsum(density) * dx
    cdf = np.clip(cdf / max(cdf[-1], 1e-10), 0, 1)

    def q(p):
        idx = np.searchsorted(cdf, p)
        return float(x[min(idx, len(x)-1)])

    # Tail probabilities
    mask_down10 = x <= -TAIL_THRESHOLD
    mask_up10   = x >= TAIL_THRESHOLD
    p_down10 = float(np.trapz(density[mask_down10], x[mask_down10])) if mask_down10.any() else 0.0
    p_up10   = float(np.trapz(density[mask_up10],   x[mask_up10]))   if mask_up10.any()   else 0.0
    mask_down5 = x <= -5.0
    p_down5  = float(np.trapz(density[mask_down5], x[mask_down5]))   if mask_down5.any()  else 0.0

    # Scenario separation
    tail_mask  = np.abs(mus) > TAIL_THRESHOLD
    mid_mask   = ~tail_mask

    tail_w     = float(w_norm[tail_mask].sum()) if tail_mask.any()  else 0.0
    medium_w   = float(w_norm[mid_mask].sum())  if mid_mask.any()   else 0.0
    tail_mu    = float(np.average(mus[tail_mask],  weights=w_norm[tail_mask]))  if tail_mask.any()  else None
    medium_mu  = float(np.average(mus[mid_mask],   weights=w_norm[mid_mask]))   if mid_mask.any()   else None

    # Is posterior bimodal? (simple check: two local maxima)
    d2 = np.gradient(np.gradient(density))
    n_modes = int(np.sum((np.diff(np.sign(np.gradient(density))) < 0)))

    stats = {
        "p5":           round(q(0.05), 3),
        "p25":          round(q(0.25), 3),
        "median":       round(q(0.50), 3),
        "p75":          round(q(0.75), 3),
        "p95":          round(q(0.95), 3),
        "mean":         round(float(np.average(mus, weights=w_norm)), 3),
        "std":          round(float(np.sqrt(np.average(
                            (mus - np.average(mus, weights=w_norm))**2 + sigmas**2,
                            weights=w_norm))), 3),
        "p_down10":     round(max(0, min(1, p_down10)), 4),
        "p_up10":       round(max(0, min(1, p_up10)),   4),
        "p_down5":      round(max(0, min(1, p_down5)),  4),
        "n_analogues":  len(mus),
        "n_tail_analogues":   int(tail_mask.sum()),
        "n_medium_analogues": int(mid_mask.sum()),
        "tail_weight":        round(tail_w,   4),
        "medium_weight":      round(medium_w, 4),
        "tail_mu":            round(tail_mu,   3) if tail_mu    else None,
        "medium_mu":          round(medium_mu, 3) if medium_mu  else None,
        "is_bimodal":         int(n_modes >= 2),
        "band_width":         round(q(0.95) - q(0.05), 3),
    }
    return x, density, stats


# ─────────────────────────────────────────────────────────────────────────────
# GARCH BENCHMARK
# ─────────────────────────────────────────────────────────────────────────────

def garch_forecast(prices: PriceData, date: pd.Timestamp,
                   horizon: int) -> Optional[Dict]:
    try:
        from arch import arch_model
    except ImportError:
        return None
    start = date - pd.Timedelta(days=400)
    col   = "WTI" if "WTI" in prices.df.columns else prices.df.columns[0]
    series = prices.df.loc[start:date, col].dropna()
    if len(series) < 100: return None
    rets = series.pct_change().dropna() * 100
    try:
        res   = arch_model(rets, vol="Garch", p=1, q=1).fit(
            disp="off", show_warning=False)
        fc    = res.forecast(horizon=horizon, reindex=False)
        sigma = float(np.sqrt(fc.variance.values[-1, -1] * horizon))
        return {
            "garch_sigma":    round(sigma, 4),
            "garch_p5":       round(float(scipy_norm.ppf(0.05, 0, sigma)), 3),
            "garch_p95":      round(float(scipy_norm.ppf(0.95, 0, sigma)), 3),
            "garch_p_down10": round(float(scipy_norm.cdf(-10, 0, sigma)),  4),
            "garch_p_up10":   round(float(1 - scipy_norm.cdf(10, 0, sigma)), 4),
        }
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# CORE EVENT RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_event_v3(
    event_id:    str,
    base_date:   pd.Timestamp,
    analogues:   List[Dict],
    prices:      PriceData,
    horizon:     int   = 15,
    min_analogues: int = MIN_ANALOGUES,
    exclude_covid: bool = True,
) -> Dict:
    """
    Run Bayesian Mixture Model for one base event.

    Enforces minimum 25 analogues via adaptive tau.
    Three-rank weighting: similarity × quantile × DTW.
    Individual Gaussian prior per analogue.
    Scenario separation: tail vs medium.
    """
    base_spread   = prices.spread_at(base_date)
    base_ovx      = prices.ovx_at(base_date)
    base_wti      = prices.wti_at(base_date)
    base_pctile   = prices.price_percentile(base_date)
    realised      = prices.forward_return(base_date, horizon)

    # ── Adaptive tau to ensure min_analogues ──────────────────────────────
    valid_pool = []
    effective_tau = None

    for tau in TAU_FALLBACKS:
        pool = []
        for a in analogues:
            if float(a.get("similarity", 0)) < tau:
                continue
            a_date = pd.to_datetime(a.get("date",""), errors="coerce")
            if pd.isna(a_date) or a_date >= base_date:
                continue
            if exclude_covid and COVID_START <= a_date <= COVID_END:
                continue
            pool.append(a)
        if len(pool) >= min_analogues:
            valid_pool   = pool
            effective_tau = tau
            break

    if not valid_pool:
        valid_pool    = [a for a in analogues
                         if pd.to_datetime(a.get("date",""), errors="coerce") < base_date]
        effective_tau = 0.0

    log.debug(f"  {event_id}: {len(valid_pool)} analogues at tau={effective_tau}")

    if len(valid_pool) < 5:
        return {
            "event_id": event_id, "base_date": str(base_date.date()),
            "horizon": horizon, "n_analogues": len(valid_pool),
            "realised_r": realised, "tau_effective": effective_tau,
            "error": "insufficient_analogues",
        }

    # ── Extract base event price path ────────────────────────────────────
    def get_path(rec: Dict) -> np.ndarray:
        p0 = float(rec.get("price", rec.get("price_usd", 1)) or 1)
        path = []
        for node in PRICE_NODES:
            if node == 0:
                path.append(0.0)
            else:
                pp = rec.get("price_path", {})
                if pp:
                    path.append(float(pp.get(str(node), pp.get(node, 0)) or 0))
                else:
                    path.append(0.0)
        return np.array(path, dtype=float)

    # For base event, use price path from JSON
    base_path_raw = np.zeros(len(PRICE_NODES))
    if analogues:
        # Reconstruct from realised prices
        for node_idx, node in enumerate(PRICE_NODES):
            r = prices.forward_return(base_date,
                                       abs(node) if node != 0 else 1)
            if r is not None:
                base_path_raw[node_idx] = r if node > 0 else -r if node < 0 else 0.0

    # ── Compute DTW for all analogues first (needed for normalisation) ────
    # FIX APPLIED: base_quantile was previously hardcoded to the literal
    # string "40%-60%" for every event regardless of actual conditions
    # (see comment history / patch notes). It now calls
    # prices.price_percentile(base_date), which was already implemented
    # in the PriceData class but never wired up here, and converts the
    # real numeric percentile into the same bucket scheme the rest of
    # the scoring already uses via bucket_for_percentile(). The fallback
    # to "40%-60%" is retained ONLY for the genuine edge case where
    # price_percentile() returns None (fewer than 50 trading days of
    # history exist before this date -- i.e. very early in the corpus).
    base_quantile = "40%-60%"   # fallback if insufficient price history
    if analogues:
        base_pctile_for_bucket = prices.price_percentile(base_date)
        if base_pctile_for_bucket is not None:
            base_quantile = bucket_for_percentile(base_pctile_for_bucket)

    all_dtw_vals = []
    for a in valid_pool:
        a_path = get_path(a)
        d = dtw_increments(base_path_raw, a_path)
        all_dtw_vals.append(d)

    # ── Compute weights and priors ────────────────────────────────────────
    mus, sigmas, weights = [], [], []
    analogue_details = []

    for idx, a in enumerate(valid_pool):
        a_date = pd.to_datetime(a.get("date",""), errors="coerce")

        # Get realised return from real daily prices
        r = prices.forward_return(a_date, horizon)
        if r is None:
            fk = {7:"r7", 15:"r15", 30:"r30"}.get(horizon)
            if fk:
                try: r = float(a.get(fk, 0) or 0)
                except: r = 0.0
        if r is None or not np.isfinite(r):
            continue

        a_path     = get_path(a)
        a_quantile = str(a.get("quantile_cat", a.get("quantile",""))).strip()
        sim        = float(a.get("similarity", 0.5))

        # Three-rank weight
        w = compute_three_rank_weight(
            similarity        = sim,
            base_quantile     = base_quantile,
            analogue_quantile = a_quantile,
            base_path         = base_path_raw,
            analogue_path     = a_path,
            all_dtw           = all_dtw_vals,
        )
        if w <= 0: continue

        # Individual prior parameters
        mu, h = analogue_prior_params(r, a_path)

        mus.append(mu)
        sigmas.append(h)
        weights.append(w)

        analogue_details.append({
            "event_id":    a.get("event_id",""),
            "date":        str(a_date.date()),
            "similarity":  round(sim, 4),
            "quantile":    a_quantile,
            "realised_r":  round(r, 4),
            "prior_mu":    round(mu, 4),
            "prior_sigma": round(h, 4),
            "weight":      round(w, 6),
            "is_tail":     int(abs(r) > TAIL_THRESHOLD),
            "scenario":    "tail" if abs(r) > TAIL_THRESHOLD else "medium",
        })

    if len(mus) < 5:
        return {
            "event_id": event_id, "base_date": str(base_date.date()),
            "horizon": horizon, "n_analogues": len(mus),
            "realised_r": realised, "tau_effective": effective_tau,
            "error": "insufficient_valid_analogues",
        }

    mus_arr  = np.array(mus)
    sigs_arr = np.array(sigmas)
    w_arr    = np.array(weights)

    # ── Compute mixture posterior ─────────────────────────────────────────
    x_grid, density, stats = compute_mixture_posterior(mus_arr, sigs_arr, w_arr)

    # ── Coverage test ─────────────────────────────────────────────────────
    covers_90 = covers_50 = None
    if realised is not None and np.isfinite(realised):
        covers_90 = int(stats["p5"]  <= realised <= stats["p95"])
        covers_50 = int(stats["p25"] <= realised <= stats["p75"])

    red_flag = int(stats["p_down10"] > 0.20)

    if   stats["median"] > 2.0 and stats["p5"]  > -5.0: signal = "LONG"
    elif stats["median"] < -2.0 and stats["p95"] <  5.0: signal = "SHORT"
    else:                                                  signal = "NEUTRAL"

    # ── GARCH benchmark ───────────────────────────────────────────────────
    garch = garch_forecast(prices, base_date, horizon)

    result = {
        "event_id":         event_id,
        "base_date":        str(base_date.date()),
        "horizon":          horizon,
        "tau_effective":    effective_tau,
        "tau_relaxed":      int(effective_tau < 0.90) if effective_tau is not None else 1,
        "n_analogues":      len(mus),
        "realised_r":       round(realised, 4) if realised else None,
        "covers_90":        covers_90,
        "covers_50":        covers_50,
        "red_flag":         red_flag,
        "signal":           signal,
        "base_wti":         round(base_wti,  2) if base_wti  else None,
        "base_spread_M1M2": round(base_spread,4) if base_spread else None,
        "base_ovx":         round(base_ovx,  2) if base_ovx  else None,
        "base_pctile":      round(base_pctile,1) if base_pctile else None,
        **{f"kde_{k}": v for k, v in stats.items()},
    }

    if garch:


        
        result.update(garch)
        if stats["std"] > 0 and garch.get("garch_sigma", 0) > 0:
            result["FEI"] = round(garch["garch_sigma"] / stats["std"], 4)
        if realised is not None and np.isfinite(realised):
            result["garch_covers_90"] = int(
                garch["garch_p5"] <= realised <= garch["garch_p95"])

    return result, analogue_details, (x_grid, density)


# ─────────────────────────────────────────────────────────────────────────────
# BATCH RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_batch(analogues_dir: Path, prices: PriceData,
              output_dir: Path, horizon: int):

    output_dir.mkdir(parents=True, exist_ok=True)
    per_dir = output_dir / "per_event"
    per_dir.mkdir(exist_ok=True)

    json_files = sorted(analogues_dir.glob("*_analogues.json"))
    if not json_files:
        log.error(f"No analogue JSONs in {analogues_dir}")
        sys.exit(1)

    log.info(f"\n{'='*60}")
    log.info(f"BATCH KDE v4  (Bayesian Mixture + Scenario Separation)")
    log.info(f"  {len(json_files)} events  horizon=+{horizon}d  "
             f"min_analogues={MIN_ANALOGUES}")
    log.info(f"  Weights: sim^{W_SIM} × quantile^{W_QUANTILE} × dtw^{W_DTW}")
    log.info(f"  [FIX] base_quantile now computed from real price_percentile(), "
             f"not hardcoded")
    log.info(f"{'='*60}")

    main_rows = []

    for jf in tqdm(json_files, desc="KDE-v4", unit=" ev"):
        try:
            with jf.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            log.warning(f"SKIP {jf.name}: {e}")
            continue

        event_id  = data.get("base_event_id", jf.stem.replace("_analogues",""))
        base_date = pd.to_datetime(data.get("base_date",""), errors="coerce")
        analogues = data.get("analogues", [])
        if pd.isna(base_date) or not analogues: continue

        out = run_event_v3(event_id, base_date, analogues,
                            prices, horizon=horizon)
        if isinstance(out, tuple):
            result, details, (x_grid, density) = out
        else:
            result = out
            details, x_grid, density = [], None, None

        if "error" in result:
            log.debug(f"  {event_id}: {result['error']}")
            main_rows.append(result)
            continue

        main_rows.append(result)

        # Save per-event JSON with full posterior
        out_file = per_dir / f"{event_id}_kde_v4.json"
        with out_file.open("w", encoding="utf-8") as f:
            json.dump({
                "event_id":         event_id,
                "base_date":        result["base_date"],
                "horizon":          horizon,
                "posterior_stats":  {k:v for k,v in result.items()
                                     if k.startswith("kde_")},
                "garch":            {k:v for k,v in result.items()
                                     if k.startswith("garch_")},
                "analogue_details": details,
                "x_grid":           x_grid.tolist() if x_grid is not None else [],
                "density":          density.tolist() if density is not None else [],
            }, f, indent=2, default=str)

    df = pd.DataFrame(main_rows)
    csv_path = output_dir / f"kde_v4_results_h{horizon}.csv"
    df.to_csv(csv_path, index=False)
    log.info(f"\n[save] {csv_path}  ({len(df)} rows)")

    _print_summary(df, horizon)
    _ovx_validation(df, horizon)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary(df: pd.DataFrame, horizon: int):
    valid = df.dropna(subset=["kde_p5"]) if "kde_p5" in df.columns else df
    n = len(valid)

    print(f"\n{'='*65}")
    print(f"  KDE v4 RESULTS  (Bayesian Mixture + Scenario Separation)")
    print(f"  horizon=+{horizon}d   {n} events")
    print(f"{'='*65}")

    if "realised_r" in valid.columns:
        r = valid["realised_r"].dropna()
        print(f"  Realised r(+{horizon}): "
              f"mean={r.mean():.2f}%  std={r.std():.2f}%  "
              f"min={r.min():.1f}%  max={r.max():.1f}%")

    if "covers_90" in valid.columns:
        cov = valid["covers_90"].dropna()
        print(f"\n  KDE v4 90% coverage    : {cov.mean()*100:.1f}%")
    if "garch_covers_90" in valid.columns:
        gcov = valid["garch_covers_90"].dropna()
        print(f"  GARCH   90% coverage   : {gcov.mean()*100:.1f}%")

    if "FEI" in valid.columns:
        fei = valid["FEI"].dropna()
        print(f"\n  FEI mean               : {fei.mean():.3f}")
        print(f"  FEI > 1 (KDE wins)     : {(fei>1).sum()}/{len(fei)}")

    if "kde_is_bimodal" in valid.columns:
        bm = valid["kde_is_bimodal"].sum()
        print(f"\n  Bimodal posteriors     : {bm}/{n} ({bm/n*100:.1f}%)")

    if "kde_tail_weight" in valid.columns:
        tw = valid["kde_tail_weight"].mean()
        mw = valid["kde_medium_weight"].mean()
        print(f"  Mean tail weight       : {tw:.3f}")
        print(f"  Mean medium weight     : {mw:.3f}")

    if "red_flag" in valid.columns:
        print(f"\n  Red flags              : {valid['red_flag'].sum()}/{n}")

    if "signal" in valid.columns:
        print(f"  Signals : {dict(valid['signal'].value_counts())}")

    if "tau_relaxed" in valid.columns:
        tr = valid["tau_relaxed"].sum()
        print(f"\n  Tau relaxed (rare ev.) : {tr}/{n} ({tr/n*100:.1f}%)")

    if "kde_band_width" in valid.columns:
        bw = valid["kde_band_width"].dropna()
        print(f"  Mean KDE band width    : {bw.mean():.2f}%")

    print(f"{'='*65}\n")


def _ovx_validation(df: pd.DataFrame, horizon: int):
    if "base_ovx" not in df.columns or "kde_std" not in df.columns:
        return
    sub = df.dropna(subset=["base_ovx", "kde_std"])
    if len(sub) < 10: return

    print(f"\n{'─'*65}")
    print(f"  OVX VALIDATION (N={len(sub)})")
    print(f"{'─'*65}")
    for col, label in [
        ("kde_std",          "KDE dispersion (std)"),
        ("kde_p_down10",     "P(r < −10%)"),
        ("kde_is_bimodal",   "Bimodal posterior"),
        ("kde_tail_weight",  "Tail analogue weight"),
    ]:
        if col not in sub.columns: continue
        try:
            r_p, p_p = pearsonr(sub["base_ovx"], sub[col].fillna(0))
            r_s, p_s = spearmanr(sub["base_ovx"], sub[col].fillna(0))
            print(f"  OVX vs {label:<30} "
                  f"r={r_p:.3f}(p={p_p:.3f})  ρ={r_s:.3f}(p={p_s:.3f})")
        except Exception:
            pass

    med = sub["base_ovx"].median()
    hi  = sub[sub["base_ovx"] >= med]
    lo  = sub[sub["base_ovx"] <  med]
    print(f"\n  High OVX (≥{med:.0f}): N={len(hi)}  "
          f"coverage={hi['covers_90'].mean()*100:.1f}%  "
          f"P(down10%)={hi['kde_p_down10'].mean()*100:.1f}%  "
          f"bimodal={hi['kde_is_bimodal'].mean()*100:.0f}%")
    print(f"  Low  OVX (< {med:.0f}): N={len(lo)}  "
          f"coverage={lo['covers_90'].mean()*100:.1f}%  "
          f"P(down10%)={lo['kde_p_down10'].mean()*100:.1f}%  "
          f"bimodal={lo['kde_is_bimodal'].mean()*100:.0f}%")
    print(f"{'─'*65}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analogues",  required=True)
    ap.add_argument("--prices",     required=True)
    ap.add_argument("--ovx",        default=None)
    ap.add_argument("--output",     default="kde_output_v4")
    ap.add_argument("--horizon",    type=int, default=15)
    ap.add_argument("--min_analogues", type=int, default=25)
    args = ap.parse_args()

    prices = PriceData(Path(args.prices),
                       Path(args.ovx) if args.ovx else None)

    global MIN_ANALOGUES
    MIN_ANALOGUES = args.min_analogues

    run_batch(
        analogues_dir = Path(args.analogues),
        prices        = prices,
        output_dir    = Path(args.output),
        horizon       = args.horizon,
    )

if __name__ == "__main__":
    main()
