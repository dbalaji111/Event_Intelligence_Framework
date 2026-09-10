"""
redflag_backtest_layer.py
=========================
Backtesting router for Brahmanda v5 — quant-grade rewrite.

Fixes applied over the original engine (see PR notes at bottom of file):
  1. Conviction-weighted continuous sizing, not binary long/short/flat.
  2. Fractional-Kelly base size, estimated walk-forward (expanding window,
     no lookahead) from the strategy's own realised trade history.
  3. Tail risk scales position size down continuously; it no longer
     hard-vetoes the trade to zero.
  4. One directional position at a time (enforced via a cooldown until the
     prior trade's horizon elapses) — eliminates the overlapping-exposure
     bug that let 349 trades in 840 days on a 15-day hold silently stack
     leverage nobody sized for.
  5. Bimodal posteriors route to a separate, parallel "straddle book"
     instead of being forced into a directional long/short decision they
     don't actually describe.
  6. Cumulative return compounds properly (product of (1+r)) instead of
     naively summing per-trade percentage returns.
  7. The Sharpe annualisation factor is derived from the strategy's own
     realised trade frequency, not assumed from the horizon parameter.

Endpoints (unchanged):
  POST /backtest/run      — start async backtest
  GET  /backtest/status   — progress polling
  GET  /backtest/results  — final results + metrics
"""

import os, uuid, logging, threading
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

log = logging.getLogger("brahmanda.backtest")
backtest_router = APIRouter()

BASE = Path(os.environ.get("BASE_PATH", "."))

# ── State ─────────────────────────────────────────────────────────────────────
_state = {
    "status":   "idle",
    "progress": 0,
    "total":    0,
    "message":  "",
    "results":  None,
}
_lock = threading.Lock()


def _load_kde_results(horizon: int) -> pd.DataFrame:
    """Load KDE results for the given horizon."""
    candidates = [
        BASE / f"kde_output_v3_h{horizon}" / f"kde_v3_results_h{horizon}.csv",
        BASE / "kde_output_v3"              / f"kde_v3_results_h{horizon}.csv",
        Path(f"kde_output_v3_h{horizon}")   / f"kde_v3_results_h{horizon}.csv",
        Path("kde_output_v3")               / f"kde_v3_results_h{horizon}.csv",
        BASE / "kde_output_v3" / "kde_v3_results_h15.csv",
        Path("kde_output_v3")  / "kde_v3_results_h15.csv",
    ]
    for p in candidates:
        if Path(p).exists():
            log.info(f"[backtest] loading results from {p}")
            df = pd.read_csv(p)
            df["base_date"] = pd.to_datetime(df["base_date"], errors="coerce")
            df = df.sort_values("base_date").reset_index(drop=True)
            return df
    raise FileNotFoundError(f"No KDE results found for h={horizon}")


# ── Sizing helpers ──────────────────────────────────────────────────────────

def _conviction(row) -> float:
    """
    Signal-to-noise of the posterior: how much of the distribution's spread
    the median actually represents. A median of -5% inside a tight [-7,-3]
    interval is high conviction; the same -5% median inside a [-25,+15]
    interval is mostly noise and should be sized down hard, independent of
    the tail-risk gate.
    Returns a value in [0, 1].
    """
    p5  = row.get("kde_p5")
    p95 = row.get("kde_p95")
    med = row.get("kde_median", 0)
    if p5 is None or p95 is None or pd.isna(p5) or pd.isna(p95):
        return 0.5  # no width info available — neutral conviction
    half_width = max((p95 - p5) / 2.0, 1e-6)
    return float(np.clip(abs(med) / half_width, 0.0, 1.0))


def _expanding_kelly_fraction(trade_history: list, fractional: float = 0.25,
                               min_trades: int = 20, default_frac: float = 0.02) -> float:
    """
    Estimates the Kelly fraction from the strategy's own realised trades
    SO FAR in the walk (expanding window, no lookahead into future trades).
    Falls back to a small flat default until there's enough of a track
    record to estimate edge reliably (professional practice: don't trust
    Kelly estimates from <20 trades).
    """
    if len(trade_history) < min_trades:
        return default_frac

    rets = np.array(trade_history)
    wins  = rets[rets > 0]
    losses = rets[rets < 0]
    if len(wins) == 0 or len(losses) == 0:
        return default_frac

    p = len(wins) / len(rets)
    q = 1 - p
    b = np.mean(wins) / abs(np.mean(losses))  # payoff ratio
    if b <= 0:
        return default_frac

    kelly = (b * p - q) / b
    kelly = max(0.0, kelly)  # never bet negative edge
    return float(kelly * fractional)


# ── Core backtest ────────────────────────────────────────────────────────────

def _run(cfg: dict):
    global _state
    try:
        horizon    = int(cfg.get("horizon", 15))
        start_date = pd.Timestamp(cfg.get("start_date", "2024-01-01"))
        end_date   = pd.Timestamp(cfg.get("end_date",   "2026-04-28"))
        band       = float(cfg.get("no_trade_band", 0.5))     # % — min |median| to act at all
        tail_ref   = float(cfg.get("tail_veto", 0.25))         # now a SCALING reference, not a hard cutoff
        cost_bps   = float(cfg.get("cost_bps", 5))
        cost_pct   = cost_bps / 10000 * 100
        max_pos    = float(cfg.get("max_position_pct", 15.0))  # hard risk cap regardless of Kelly output
        min_sim    = float(cfg.get("min_similarity", 0.0))     # confidence gate; 0 = disabled if col absent

        df = _load_kde_results(horizon)
        mask = (df["base_date"] >= start_date) & (df["base_date"] <= end_date)
        df   = df[mask].dropna(subset=["realised_r", "kde_median"]).reset_index(drop=True)
        total = len(df)
        if total == 0:
            raise ValueError("No events in date range")

        with _lock:
            _state.update({"status":"running","progress":0,"total":total,
                           "message":"Initialising..."})

        # ── two parallel books ──
        # Directional book: one position at a time, sized by conviction + Kelly.
        # Straddle book: bimodal events with no clear direction — profits from
        # move MAGNITUDE regardless of sign, so it is not exposed to the same
        # directional risk as the main book and does not share its cooldown.
        dir_cooldown_until = None
        straddle_cooldown_until = None
        dir_trade_history = []      # net-of-cost realised returns, for expanding Kelly

        events = []       # per-event record for the response payload
        pnl_series = []   # (date, pnl_pct, book) chronological
        n_directional = n_straddle = n_flat_gated = n_flat_cooldown = n_flat_band = 0

        for i, row in df.iterrows():
            with _lock:
                _state["progress"] = i + 1
                if i % 20 == 0:
                    _state["message"] = f"Processing {i+1}/{total}"

            date       = row["base_date"]
            r          = float(row["realised_r"])
            med        = float(row.get("kde_median", 0))
            pd10       = float(row.get("kde_p_down10", 0))
            tail_w     = float(row.get("tail_weight", pd10))  # fall back to pd10 if tail_weight absent
            is_bimodal = bool(row.get("is_bimodal", False))
            similarity = row.get("similarity_mean", row.get("similarity", None))

            record = {"date": str(date)[:10], "kde_median": med, "realised_r": r,
                      "book": None, "size_pct": 0.0, "pnl_pct": 0.0}

            # confidence gate — skip low-evidence events entirely rather than
            # sizing them small; a thin analogue set shouldn't enter the book.
            if min_sim > 0 and similarity is not None and not pd.isna(similarity) and similarity < min_sim:
                n_flat_gated += 1
                events.append(record)
                continue

            # ── STRADDLE BOOK: bimodal, no clear directional edge ──
            if is_bimodal and abs(med) <= band:
                if straddle_cooldown_until is not None and date <= straddle_cooldown_until:
                    n_flat_cooldown += 1
                    events.append(record)
                    continue
                conv = _conviction(row)
                size = min(max_pos, 5.0 + 10.0 * conv)  # base straddle allocation, richer when tails are wide+clear
                # Straddle payoff proxy: profits from |move|, pays premium ~ cost_pct*2 (both legs)
                premium = cost_pct * 2
                pnl = (abs(r) - premium) * (size / 100)
                straddle_cooldown_until = date + pd.Timedelta(days=horizon)
                record.update(book="straddle", size_pct=round(size, 2), pnl_pct=round(pnl, 4))
                pnl_series.append((date, pnl, "straddle"))
                n_straddle += 1
                events.append(record)
                continue

            # ── DIRECTIONAL BOOK ──
            if abs(med) <= band:
                n_flat_band += 1
                events.append(record)
                continue
            if dir_cooldown_until is not None and date <= dir_cooldown_until:
                n_flat_cooldown += 1
                events.append(record)
                continue

            sig = 1 if med > 0 else -1
            conv = _conviction(row)
            tail_scale = max(0.0, 1.0 - min(tail_w / max(tail_ref, 1e-6), 1.0) * 0.7)
            # ^ tail risk shrinks size continuously; even at very high tail risk
            #   we retain 30% of base size rather than zeroing the trade —
            #   a hard veto discards well-evidenced signals for no gain.

            kelly_frac = _expanding_kelly_fraction(dir_trade_history)
            size = min(max_pos, 100 * kelly_frac * conv * tail_scale)

            gross = sig * r
            pnl   = (gross - cost_pct) * (size / 100)

            dir_cooldown_until = date + pd.Timedelta(days=horizon)
            dir_trade_history.append(gross - cost_pct)  # feed forward for next Kelly estimate

            record.update(book="directional", size_pct=round(size, 2),
                          pnl_pct=round(pnl, 4), direction=sig)
            pnl_series.append((date, pnl, "directional"))
            n_directional += 1
            events.append(record)

        # ── aggregate ──
        pnl_series.sort(key=lambda x: x[0])
        dates_sorted = [str(d)[:10] for d, _, _ in pnl_series]
        rets_sorted  = np.array([p for _, p, _ in pnl_series])

        n_trades = len(rets_sorted)
        if n_trades == 0:
            raise ValueError("No trades generated — thresholds too strict for this date range")

        # honest compounding, not naive sum
        growth_curve = np.cumprod(1 + rets_sorted / 100)
        cum_ret_pct  = float((growth_curve[-1] - 1) * 100) if len(growth_curve) else 0.0
        equity_curve = [round(float((g - 1) * 100), 3) for g in growth_curve]

        win_rate = float(np.mean(rets_sorted > 0)) if n_trades > 0 else 0
        mean_ret = float(np.mean(rets_sorted))
        std_ret  = float(np.std(rets_sorted)) if n_trades > 1 else 0

        # annualise from ACTUAL trade frequency, not an assumed horizon cadence
        span_days = max((df["base_date"].max() - df["base_date"].min()).days, 1)
        trades_per_year = n_trades * 365.0 / span_days
        ann_factor = np.sqrt(trades_per_year) if trades_per_year > 0 else 0
        sharpe = (mean_ret / std_ret * ann_factor) if std_ret > 0 else 0

        # max drawdown off the SAME compounded equity curve used for return
        peak = np.maximum.accumulate(growth_curve)
        drawdowns = (growth_curve - peak) / peak
        max_dd = float(np.min(drawdowns) * 100)

        covers = df["covers_90"].dropna() if "covers_90" in df.columns else pd.Series(dtype=float)
        coverage = float(covers.mean() * 100) if len(covers) > 0 else 0

        rf_col = "red_flag" if "red_flag" in df.columns else None
        if rf_col:
            flagged   = df[df[rf_col] == 1]["realised_r"]
            unflagged = df[df[rf_col] == 0]["realised_r"]
            rf_flagged_mean   = float(flagged.mean())   if len(flagged)   > 0 else 0
            rf_unflagged_mean = float(unflagged.mean()) if len(unflagged) > 0 else 0
            n_flagged = int(len(flagged))
        else:
            rf_flagged_mean = rf_unflagged_mean = 0
            n_flagged = 0

        results = {
            "run_id":  str(uuid.uuid4())[:8],
            "config":  cfg,
            "n_events": total,
            "metrics": {
                "calibration": {"coverage_90pct": round(coverage, 1)},
                "book_composition": {
                    "n_directional_trades": n_directional,
                    "n_straddle_trades":    n_straddle,
                    "n_flat_no_signal":     n_flat_band,
                    "n_flat_cooldown":      n_flat_cooldown,
                    "n_flat_low_confidence": n_flat_gated,
                },
                "trading": {
                    "cumulative_return_pct":  round(cum_ret_pct, 2),
                    "mean_trade_return_pct":  round(mean_ret, 3),
                    "sharpe_ratio":           round(sharpe, 3),
                    "max_drawdown_pct":       round(max_dd, 2),
                    "win_rate_pct":           round(win_rate * 100, 1),
                    "n_trades":               n_trades,
                    "trades_per_year":        round(trades_per_year, 1),
                    "annualisation_factor":   round(ann_factor, 3),
                },
                "red_flag_utility": {
                    "n_flagged":        n_flagged,
                    "flagged_mean_r":   round(rf_flagged_mean, 2),
                    "unflagged_mean_r": round(rf_unflagged_mean, 2),
                    "differential_pp":  round(rf_unflagged_mean - rf_flagged_mean, 2),
                },
            },
            "equity_curve": equity_curve,
            "dates":        dates_sorted,
            "events":       events,
        }

        with _lock:
            _state.update({"status":"done","progress":total,"total":total,
                           "message":"Complete","results":results})
        log.info(f"[backtest] done — {n_trades} trades ({n_directional} directional, "
                 f"{n_straddle} straddle), Sharpe {sharpe:.3f}, cum_ret {cum_ret_pct:.2f}%, "
                 f"max_dd {max_dd:.2f}%")

    except Exception as e:
        log.error(f"[backtest] error: {e}")
        with _lock:
            _state.update({"status":"error","message":str(e)})


# ── Endpoints ─────────────────────────────────────────────────────────────────

@backtest_router.post("/backtest/run")
def backtest_run(payload: dict):
    with _lock:
        if _state["status"] == "running":
            raise HTTPException(status_code=409, detail="A backtest is already running")

    cfg = {
        "horizon":         int(payload.get("horizon", 15)),
        "start_date":      str(payload.get("start_date", "2024-01-01")),
        "end_date":        str(payload.get("end_date",   "2026-04-28")),
        "no_trade_band":   float(payload.get("no_trade_band", 0.5)),
        "tail_veto":       float(payload.get("tail_veto", 0.25)),
        "cost_bps":        float(payload.get("cost_bps", 5)),
        "max_position_pct": float(payload.get("max_position_pct", 15.0)),
        "min_similarity":  float(payload.get("min_similarity", 0.0)),
    }

    t = threading.Thread(target=_run, args=(cfg,), daemon=True)
    t.start()
    log.info(f"[backtest] started: {cfg}")
    return {"status": "started", "config": cfg}


@backtest_router.get("/backtest/status")
def backtest_status():
    with _lock:
        return {k: v for k, v in _state.items() if k != "results"}


@backtest_router.get("/backtest/results")
def backtest_results():
    with _lock:
        if _state["results"] is None:
            raise HTTPException(status_code=404,
                detail="No results — run a backtest first")
        return _state["results"]


# ─────────────────────────────────────────────────────────────────────────────
# PR NOTES — what changed vs the original engine, and why
# ─────────────────────────────────────────────────────────────────────────────
# 1. Binary sig in {-1,0,1} at fixed size -> continuous size = f(conviction,
#    Kelly, tail risk). A median of +0.6% and +8% no longer get identical
#    exposure.
# 2. Hard tail veto (pd10 > 0.25 -> flat) -> continuous tail_scale multiplier,
#    floor 30% of base size. Well-evidenced trades are shrunk, not discarded.
# 3. No overlap control -> one directional position at a time, enforced via
#    cooldown_until = trade_date + horizon. This is the fix for the
#    Sharpe/drawdown contradiction: trades can no longer silently stack
#    leverage the sizing math never accounted for.
# 4. Bimodal events forced into a directional decision they don't describe
#    -> routed to a parallel straddle book that profits from move magnitude,
#    uncorrelated to the directional book's cooldown.
# 5. cum_ret = sum(returns) -> cum_ret = compound(returns). The naive sum
#    is what produced the impossible +1524%/-90.2% combination.
# 6. ann_factor assumed from horizon (252/horizon) -> derived from the
#    strategy's OWN realised trade frequency (trades_per_year). This is
#    what makes Sharpe internally consistent with the drawdown and return
#    figures again, since all three now come from the same trade stream.
# 7. Kelly sizing uses an EXPANDING window of the strategy's own past
#    trades (no lookahead) rather than full-sample stats, avoiding the
#    classic "sized on the same data the signal was optimised on" overfit.
# ─────────────────────────────────────────────────────────────────────────────