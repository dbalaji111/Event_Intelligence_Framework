"""
kde_engine.py  —  Event-Conditioned Bayesian KDE Forecasting Engine
====================================================================

Reads:
  1. Retrieval output  (retrieval_output_v3/per_event/*.json)
  2. oil_prices_full.xlsx  — daily WTI spot, M1-M2 spread, Brent curve

For each base event and each analogue:
  - Extracts real ±15 day daily price/spread windows from the Excel file
  - Weights analogues by: similarity × recency × regime_match
    (regime_match uses M1-M2 spread at event date — contango vs backwardation)
  - Fits weighted Gaussian KDE on forward returns at t+7 and t+15
  - Computes: P5/P25/median/P75/P95, P(r<-10%), P(r>+10%), red flag
  - Runs tau sensitivity across 5 levels
  - Compares to GARCH(1,1) benchmark

Outputs:
  kde_output/kde_results.csv          — one row per event (main results)
  kde_output/tau_sensitivity.csv      — Table 3 of paper
  kde_output/per_event/*_kde.json     — full posteriors

Usage
-----
    python kde_engine.py \
        --analogues retrieval_output_v3/per_event \
        --prices    "data/oil_prices_full.xlsx" \
        --output    kde_output

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
from scipy.stats import gaussian_kde
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

WINDOW      = 15          # ±15 days around event
HORIZONS    = [7, 15]     # forward forecast horizons
TAU_LEVELS  = [0.0, 0.75, 0.80, 0.85, 0.90]
MIN_OBS_KDE = 5           # minimum analogues needed to fit KDE

# COVID exclusion window — extreme outlier period
COVID_START = pd.Timestamp("2020-02-15")
COVID_END   = pd.Timestamp("2020-07-01")

# Column names from oil_prices_full.xlsx
COL_DATE       = "Date"
COL_WTI_SPOT   = "Crude Oil-WTI Spot Cushing U$/BBL"
COL_WTI_NYMEX  = "Crude Oil WTI NYMEX Close M U$/BBL"
COL_M1M2_SPREAD= "NYMEX Crude Oil WTI M1-M2 Spread"
COL_BRENT_M1   = "Crude Oil Brent US M1 U$/BBL"
COL_BRENT_M2   = "Crude Oil Brent US M2 U$/BBL"
COL_BRENT_M3   = "Crude Oil Brent US M3 U$/BBL"


# ─────────────────────────────────────────────────────────────────────────────
# PRICE DATA LOADER
# ─────────────────────────────────────────────────────────────────────────────

class PriceData:
    """
    Loads oil_prices_full.xlsx and provides:
      - forward_return(date, horizon)   → % return from date to date+horizon
      - spread_at(date)                 → M1/M2 spread value
      - regime_at(date)                 → 'backwardation' or 'contango'
      - window(date, col)               → daily series ±WINDOW days
    """

    def __init__(self, path: Path):
        log.info(f"[price] loading {path.name} ...")
        df = pd.read_excel(path)

        # Rename date column
        df = df.rename(columns={df.columns[0]: "Date"})
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).set_index("Date").sort_index()

        # Rename key columns to short names
        rename = {
            COL_WTI_SPOT:    "WTI",
            COL_WTI_NYMEX:   "WTI_M1",
            COL_M1M2_SPREAD: "M1M2",
            COL_BRENT_M1:    "BM1",
            COL_BRENT_M2:    "BM2",
            COL_BRENT_M3:    "BM3",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

        # Compute Brent M1/M2 spread where available
        if "BM1" in df.columns and "BM2" in df.columns:
            df["BM1M2"] = df["BM1"] - df["BM2"]

        # Forward-fill gaps up to 5 days (weekends/holidays)
        df = df.ffill(limit=5)
        self.df = df

        # Report coverage
        log.info(f"[price]   {len(df)} rows  "
                 f"{df.index[0].date()} → {df.index[-1].date()}")
        avail = [c for c in ["WTI","WTI_M1","M1M2","BM1","BM2","BM3","BM1M2"]
                 if c in df.columns]
        log.info(f"[price]   columns: {avail}")

    def _nearest(self, date: pd.Timestamp, max_offset: int = 5
                 ) -> Optional[pd.Timestamp]:
        """Find nearest trading day within max_offset days."""
        if date in self.df.index:
            return date
        candidates = self.df.index[
            abs(self.df.index - date) <= pd.Timedelta(days=max_offset)
        ]
        if len(candidates) == 0:
            return None
        return candidates[abs(candidates - date).argmin()]

    def forward_return(self, date: pd.Timestamp, horizon: int,
                       col: str = "WTI") -> Optional[float]:
        """% return from date to date+horizon trading days."""
        if col not in self.df.columns:
            col = "WTI" if "WTI" in self.df.columns else self.df.columns[0]
        t0 = self._nearest(date)
        if t0 is None:
            return None
        p0 = self.df.loc[t0, col]
        if not np.isfinite(p0) or p0 == 0:
            return None

        target = date + pd.Timedelta(days=horizon)
        t1 = self._nearest(target, max_offset=10)
        if t1 is None:
            return None
        p1 = self.df.loc[t1, col]
        if not np.isfinite(p1):
            return None

        return float((p1 - p0) / abs(p0) * 100.0)

    def spread_at(self, date: pd.Timestamp) -> Optional[float]:
        """M1/M2 spread at date. Uses WTI M1M2 first, Brent BM1M2 as fallback."""
        t = self._nearest(date)
        if t is None:
            return None
        for col in ["M1M2", "BM1M2"]:
            if col in self.df.columns:
                v = self.df.loc[t, col]
                if np.isfinite(v):
                    return float(v)
        return None

    def regime_at(self, date: pd.Timestamp) -> str:
        """'backwardation' if M1>M2, 'contango' if M1<M2, 'unknown' otherwise."""
        s = self.spread_at(date)
        if s is None:
            return "unknown"
        if s > 0.10:
            return "backwardation"
        if s < -0.10:
            return "contango"
        return "flat"

    def wti_at(self, date: pd.Timestamp) -> Optional[float]:
        t = self._nearest(date)
        if t is None:
            return None
        for col in ["WTI", "WTI_M1"]:
            if col in self.df.columns:
                v = self.df.loc[t, col]
                if np.isfinite(v):
                    return float(v)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# KDE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def fit_weighted_kde(obs: np.ndarray,
                     weights: np.ndarray) -> Optional[gaussian_kde]:
    obs = np.asarray(obs, dtype=float)
    weights = np.asarray(weights, dtype=float)

    mask = np.isfinite(obs) & np.isfinite(weights) & (weights > 0)
    obs, weights = obs[mask], weights[mask]

    if len(obs) < MIN_OBS_KDE:
        return None
    if np.std(obs) < 1e-6:
        return None

    weights = weights / weights.sum()
    try:
        return gaussian_kde(obs, weights=weights)
    except Exception:
        return None


def kde_statistics(kde: gaussian_kde, obs: np.ndarray) -> Dict:
    obs_clean = obs[np.isfinite(obs)]
    lo = obs_clean.min() - 10
    hi = obs_clean.max() + 10
    x  = np.linspace(lo, hi, 1000)
    dx = x[1] - x[0]

    density = kde(x)
    cdf = np.cumsum(density) * dx
    cdf = np.clip(cdf / cdf[-1], 0, 1)

    def q(p):
        idx = np.searchsorted(cdf, p)
        return float(x[min(idx, len(x)-1)])

    p_down10 = float(np.clip(kde.integrate_box_1d(-np.inf, -10.0), 0, 1))
    p_up10   = float(np.clip(kde.integrate_box_1d(10.0,   np.inf), 0, 1))
    p_down5  = float(np.clip(kde.integrate_box_1d(-np.inf,  -5.0), 0, 1))
    p_up5    = float(np.clip(kde.integrate_box_1d( 5.0,   np.inf), 0, 1))

    return {
        "n_obs":    int(len(obs_clean)),
        "p5":       round(q(0.05), 3),
        "p25":      round(q(0.25), 3),
        "median":   round(q(0.50), 3),
        "p75":      round(q(0.75), 3),
        "p95":      round(q(0.95), 3),
        "mean":     round(float(np.mean(obs_clean)), 3),
        "std":      round(float(np.std(obs_clean)),  3),
        "p_down10": round(p_down10, 4),
        "p_up10":   round(p_up10,   4),
        "p_down5":  round(p_down5,  4),
        "p_up5":    round(p_up5,    4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# GARCH BENCHMARK
# ─────────────────────────────────────────────────────────────────────────────

def garch_forecast(prices: PriceData, date: pd.Timestamp,
                   horizon: int = 15) -> Optional[Dict]:
    try:
        from arch import arch_model
        from scipy.stats import norm
    except ImportError:
        return None

    start = date - pd.Timedelta(days=400)
    col   = "WTI" if "WTI" in prices.df.columns else prices.df.columns[0]
    series = prices.df.loc[start:date, col].dropna()
    if len(series) < 100:
        return None

    rets = series.pct_change().dropna() * 100
    try:
        model  = arch_model(rets, vol="Garch", p=1, q=1, dist="normal")
        result = model.fit(disp="off", show_warning=False)
        fc     = result.forecast(horizon=horizon, reindex=False)
        var_h  = fc.variance.values[-1, -1]
        sigma  = float(np.sqrt(var_h * horizon))
        return {
            "garch_sigma":    round(sigma, 4),
            "garch_p5":       round(float(norm.ppf(0.05, 0, sigma)), 3),
            "garch_p95":      round(float(norm.ppf(0.95, 0, sigma)), 3),
            "garch_p_down10": round(float(norm.cdf(-10, 0, sigma)),  4),
            "garch_p_up10":   round(float(1 - norm.cdf(10, 0, sigma)), 4),
        }
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# ANALOGUE WEIGHTING
# ─────────────────────────────────────────────────────────────────────────────

def compute_weight(
    similarity:     float,
    analogue_date:  pd.Timestamp,
    base_date:      pd.Timestamp,
    base_spread:    Optional[float],
    analogue_spread: Optional[float],
    recency_lambda: float = 0.05,
    sigma_spread:   float = 1.0,
    exclude_covid:  bool  = True,
) -> float:
    """
    Combined analogue weight:
        w = similarity × recency × regime_match

    similarity  : BERT cosine score
    recency     : exp(-λ × age_years)  — recent events weighted more
    regime_match: Gaussian similarity on M1/M2 spread
                  — analogues in same spread regime weighted more
    """
    # Exclude COVID crash period
    if exclude_covid and COVID_START <= analogue_date <= COVID_END:
        return 0.0

    # Recency weight
    age_years = max((base_date - analogue_date).days / 365.25, 0.0)
    recency_w = float(np.exp(-recency_lambda * age_years))

    # Regime match weight
    regime_w = 1.0
    if base_spread is not None and analogue_spread is not None:
        if np.isfinite(base_spread) and np.isfinite(analogue_spread):
            regime_w = float(
                np.exp(-0.5 * ((base_spread - analogue_spread) / sigma_spread) ** 2)
            )

    return float(similarity) * recency_w * regime_w


# ─────────────────────────────────────────────────────────────────────────────
# CORE KDE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def run_event_kde(
    event_id:   str,
    base_date:  pd.Timestamp,
    analogues:  List[Dict],
    prices:     PriceData,
    tau_filter: float = 0.0,
    horizon:    int   = 15,
    exclude_covid: bool = True,
) -> Dict:
    """
    Run KDE for one base event at one tau level and one horizon.
    """
    # Base event market state
    base_spread = prices.spread_at(base_date)
    base_regime = prices.regime_at(base_date)
    base_wti    = prices.wti_at(base_date)

    # Realised forward return (for coverage test)
    realised = prices.forward_return(base_date, horizon)

    # Filter analogues by tau
    pool = [a for a in analogues
            if float(a.get("similarity", 0.0)) >= tau_filter]

    if not pool:
        return {
            "event_id":    event_id,
            "base_date":   str(base_date.date()),
            "tau":         tau_filter,
            "horizon":     horizon,
            "n_analogues": 0,
            "realised_r":  realised,
            "base_spread": base_spread,
            "base_regime": base_regime,
        }

    # Collect observations and weights
    obs, weights, dates_used = [], [], []

    for a in pool:
        a_date = pd.to_datetime(a.get("date",""), errors="coerce")
        if pd.isna(a_date):
            continue

        # Get forward return from REAL daily price data
        r = prices.forward_return(a_date, horizon)

        # Fallback to pre-computed nodes if daily data unavailable
        if r is None:
            fallback_key = {7: "r7", 15: "r15", 30: "r30"}.get(horizon)
            if fallback_key:
                r_raw = a.get(fallback_key)
                if r_raw is not None:
                    try: r = float(r_raw)
                    except: pass

        if r is None or not np.isfinite(float(r)):
            continue

        # Get analogue's spread regime from real data
        a_spread = prices.spread_at(a_date)

        # Compute combined weight
        w = compute_weight(
            similarity      = float(a.get("similarity", 0.5)),
            analogue_date   = a_date,
            base_date       = base_date,
            base_spread     = base_spread,
            analogue_spread = a_spread,
            exclude_covid   = exclude_covid,
        )

        if w <= 0:
            continue

        obs.append(float(r))
        weights.append(w)
        dates_used.append(str(a_date.date()))

    if len(obs) < MIN_OBS_KDE:
        return {
            "event_id":    event_id,
            "base_date":   str(base_date.date()),
            "tau":         tau_filter,
            "horizon":     horizon,
            "n_analogues": len(obs),
            "realised_r":  realised,
            "base_spread": base_spread,
            "base_regime": base_regime,
            "note":        "insufficient_observations",
        }

    obs_arr = np.array(obs)
    w_arr   = np.array(weights)

    # Fit weighted KDE
    kde = fit_weighted_kde(obs_arr, w_arr)
    if kde is None:
        return {"event_id": event_id, "error": "kde_fit_failed",
                "n_analogues": len(obs)}

    stats = kde_statistics(kde, obs_arr)

    # Coverage test
    covers_90 = covers_50 = None
    if realised is not None and np.isfinite(realised):
        covers_90 = int(stats["p5"]  <= realised <= stats["p95"])
        covers_50 = int(stats["p25"] <= realised <= stats["p75"])

    # Red flag: P(r < -10%) > 20%
    red_flag = int(stats["p_down10"] > 0.20)

    # Trade signal
    if stats["median"] > 2.0 and stats["p5"] > -5.0:
        signal = "LONG"
    elif stats["median"] < -2.0 and stats["p95"] < 5.0:
        signal = "SHORT"
    else:
        signal = "NEUTRAL"

    # GARCH benchmark
    garch = garch_forecast(prices, base_date, horizon)

    result = {
        "event_id":         event_id,
        "base_date":        str(base_date.date()),
        "tau":              tau_filter,
        "horizon":          horizon,
        "n_analogues":      len(obs),
        "realised_r":       round(realised, 4) if realised else None,
        "covers_90":        covers_90,
        "covers_50":        covers_50,
        "red_flag":         red_flag,
        "signal":           signal,
        "base_wti":         round(base_wti,    2) if base_wti    else None,
        "base_spread_M1M2": round(base_spread, 4) if base_spread else None,
        "base_regime":      base_regime,
        **{f"kde_{k}": v for k, v in stats.items()},
    }

    if garch:
        result.update(garch)
        if stats["std"] > 0 and garch.get("garch_sigma", 0) > 0:
            result["FEI"] = round(garch["garch_sigma"] / stats["std"], 4)
        if realised is not None and np.isfinite(realised):
            result["garch_covers_90"] = int(
                garch["garch_p5"] <= realised <= garch["garch_p95"]
            )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# BATCH RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_batch(analogues_dir: Path, prices: PriceData,
              output_dir: Path, horizon: int = 15):

    output_dir.mkdir(parents=True, exist_ok=True)
    per_dir = output_dir / "per_event"
    per_dir.mkdir(exist_ok=True)

    json_files = sorted(analogues_dir.glob("*_analogues.json"))
    if not json_files:
        log.error(f"No analogue JSONs in {analogues_dir}")
        log.error("Run retrieval_engine.py first.")
        sys.exit(1)

    log.info(f"\n{'='*60}")
    log.info(f"BATCH KDE: {len(json_files)} events  horizon=+{horizon}d")
    log.info(f"{'='*60}")

    main_rows, tau_rows = [], []

    for jf in tqdm(json_files, desc="KDE", unit=" ev"):
        try:
            with jf.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            log.warning(f"SKIP {jf.name}: {e}")
            continue

        event_id  = data.get("base_event_id", jf.stem.replace("_analogues",""))
        base_date = pd.to_datetime(data.get("base_date",""), errors="coerce")
        analogues = data.get("analogues", [])

        if pd.isna(base_date) or not analogues:
            continue

        # Main result at tau=0.0
        result = run_event_kde(
            event_id, base_date, analogues, prices,
            tau_filter=0.0, horizon=horizon
        )
        main_rows.append(result)

        # Tau sensitivity
        tau_results = [
            run_event_kde(event_id, base_date, analogues, prices,
                          tau_filter=tau, horizon=horizon)
            for tau in TAU_LEVELS
        ]
        tau_rows.extend(tau_results)

        # Save per-event
        out = per_dir / f"{event_id}_kde.json"
        with out.open("w", encoding="utf-8") as f:
            json.dump({
                "event_id":       event_id,
                "base_date":      str(base_date.date()),
                "main_result":    result,
                "tau_sensitivity": tau_results,
            }, f, indent=2, default=str)

    # Save CSVs
    df_main = pd.DataFrame(main_rows)
    df_tau  = pd.DataFrame(tau_rows)

    main_csv = output_dir / f"kde_results_h{horizon}.csv"
    tau_csv  = output_dir / f"tau_sensitivity_h{horizon}.csv"
    df_main.to_csv(main_csv, index=False)
    df_tau.to_csv(tau_csv,   index=False)

    log.info(f"\n[save] {main_csv}  ({len(df_main)} rows)")
    log.info(f"[save] {tau_csv}")

    _print_summary(df_main, df_tau, horizon)
    return df_main, df_tau


def _print_summary(df: pd.DataFrame, df_tau: pd.DataFrame, horizon: int):
    print(f"\n{'='*65}")
    print(f"  KDE RESULTS  (horizon=+{horizon}d,  {len(df)} events)")
    print(f"{'='*65}")

    if "realised_r" in df.columns:
        r = df["realised_r"].dropna()
        print(f"  Realised r(+{horizon}): mean={r.mean():.2f}%  "
              f"std={r.std():.2f}%  min={r.min():.1f}%  max={r.max():.1f}%")

    if "covers_90" in df.columns:
        cov = df["covers_90"].dropna()
        print(f"\n  KDE 90% coverage  : {cov.mean()*100:.1f}%  (target=90%)")
    if "garch_covers_90" in df.columns:
        gcov = df["garch_covers_90"].dropna()
        print(f"  GARCH 90% coverage: {gcov.mean()*100:.1f}%")

    if "FEI" in df.columns:
        fei = df["FEI"].dropna()
        print(f"\n  FEI mean          : {fei.mean():.3f}  "
              f"(KDE wins if FEI>1, i.e. GARCH wider)")
        print(f"  FEI > 1 (KDE wins): {(fei>1).sum()}/{len(fei)} events")

    if "red_flag" in df.columns:
        print(f"\n  Red flags         : {df['red_flag'].sum()}/{len(df)}")

    if "signal" in df.columns:
        print(f"  Signals           : {dict(df['signal'].value_counts())}")

    if "base_regime" in df.columns:
        print(f"\n  Regime breakdown:")
        for reg, grp in df.groupby("base_regime"):
            cov = grp["covers_90"].mean()*100 if "covers_90" in grp else float("nan")
            print(f"    {reg:<15} N={len(grp):>3}  coverage={cov:.1f}%")

    # Tau sensitivity table
    if not df_tau.empty and "tau" in df_tau.columns:
        print(f"\n{'─'*65}")
        print(f"  TAU SENSITIVITY  (horizon=+{horizon}d)")
        print(f"{'─'*65}")
        print(f"  {'tau':<6} {'N_mean':>7} {'P5':>7} {'median':>8} "
              f"{'P95':>7} {'Coverage':>10} {'P(down10%)':>12}")
        print(f"  {'─'*60}")
        for tau in TAU_LEVELS:
            sub = df_tau[df_tau["tau"] == tau]
            if sub.empty: continue
            n   = sub["n_analogues"].mean()
            p5  = sub["kde_p5"].mean()   if "kde_p5"    in sub else float("nan")
            med = sub["kde_median"].mean()if "kde_median" in sub else float("nan")
            p95 = sub["kde_p95"].mean()  if "kde_p95"   in sub else float("nan")
            cov = sub["covers_90"].mean()*100 if "covers_90" in sub else float("nan")
            pd10= sub["kde_p_down10"].mean()*100 if "kde_p_down10" in sub else float("nan")
            print(f"  {tau:<6.2f} {n:>7.1f} {p5:>7.2f} {med:>8.2f} "
                  f"{p95:>7.2f} {cov:>9.1f}% {pd10:>11.1f}%")
    print(f"{'='*65}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analogues", required=True,
                    help="path to per_event/ directory from retrieval_engine.py")
    ap.add_argument("--prices",    required=True,
                    help="path to oil_prices_full.xlsx")
    ap.add_argument("--output",    default="kde_output")
    ap.add_argument("--horizon",   type=int, default=15,
                    help="forward horizon in days (default: 15)")
    ap.add_argument("--no_covid_filter", action="store_true",
                    help="include COVID crash period analogues")
    args = ap.parse_args()

    prices = PriceData(Path(args.prices))

    run_batch(
        analogues_dir = Path(args.analogues),
        prices        = prices,
        output_dir    = Path(args.output),
        horizon       = args.horizon,
    )

if __name__ == "__main__":
    main()