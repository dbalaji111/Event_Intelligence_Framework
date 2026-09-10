"""
retrieval_engine.py
====================
Retrieval Engine for Event-Conditioned KDE Forecasting.

Data sources
------------
  TRAIN (2001-2023):
    memory_2001_2023/price_event_memory.json   — 11,983 events with
    bert_embeddings inline, all price/spread/technical fields

  TEST (2024-2026):
    memory_2024_2026/df_major_memory_2024_2026.jsonl — 406 events,
    same schema

Pipeline
--------
  1. Load both corpora → parse inline BERT embeddings → build index
  2. Given a base event (test corpus):
       Stage 1: cosine similarity > tau
       Stage 2: quantile group match
       Stage 3: temporal exclusion (analogue must predate base)
     Rank by similarity × recency_weight → return top-K
  3. Extract normalised price paths per analogue
  4. DTW vs random baseline → Table 1 / RQ1

Usage — list event IDs
-----------------------
    python retrieval_engine.py --train ... --test ... --list_events

Usage — single event
---------------------
    python retrieval_engine.py \
        --train "memory_2001_2023/price_event_memory.json" \
        --test  "memory_2024_2026/df_major_memory_2024_2026.jsonl" \
        --event EVT_2025.0_0042.0 \
        --tau 0.75 --top_k 50 --output retrieval_output/

Usage — full batch
-------------------
    python retrieval_engine.py \
        --train "memory_2001_2023/price_event_memory.json" \
        --test  "memory_2024_2026/df_major_memory_2024_2026.jsonl" \
        --tau 0.75 --output retrieval_output/

Usage — Table 1 (paper RQ1)
-----------------------------
    python retrieval_engine.py \
        --train "memory_2001_2023/price_event_memory.json" \
        --test  "memory_2024_2026/df_major_memory_2024_2026.jsonl" \
        --table1 --output retrieval_output/

Dependencies
------------
    pip install numpy pandas scikit-learn scipy tqdm
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

EMB_DIM = 768

# Quantile group mapping — covers both train (2001-2023) and test (2024-2026)
# schemas which use different bucket label conventions
QUANTILE_GROUPS: Dict[str, str] = {
    # Test corpus labels (2024-2026)
    "0%-20%":   "low",
    "20%-40%":  "low",
    "40%-60%":  "mid",
    "60%-80%":  "high",
    "80%-100%": "high",
    # Train corpus labels (2001-2023)
    "5%":       "low",      # bottom 5%
    "5%-20%":   "low",
    "20%-30%":  "low",
    "30%-40%":  "mid",
    "40%-60%":  "mid",      # same as test mid
    "60%-80%":  "high",     # same as test high
    "80%-90%":  "high",
    "90%+":     "high",
}

# Event types to EXCLUDE from analogue pool
# Company earnings/production reports don't cause price moves
EXCLUDE_EVENT_TYPES = {
    "position-high", "position-low", "position_high", "position_low",
    "grow-strong", "grow_strong", "movement-flat", "movement_flat",
    "nan", "", "none",
}

# Price nodes — day offset → field name (confirmed from schema)
PRICE_NODES = [-30, -15, -7, 0, 7, 15, 30]
PRICE_COLS: Dict[int, str] = {
    -30: "price_t_minus_30",
    -15: "price_t_minus_15",
     -7: "price_t_minus_7",
      0: "price",
      7: "price_t_plus_7",
     15: "price_t_plus_15",
     30: "price_t_plus_30",
}

# Spread — only event-date value confirmed; ± nodes fallback to event-date
SPREAD_COLS: Dict[int, str] = {
     0: "Spread_M1_M2",
     7: "Spread_M1_M2_t_plus_7",
    15: "Spread_M1_M2_t_plus_15",
    30: "Spread_M1_M2_t_plus_30",
    -7: "Spread_M1_M2_t_minus_7",
   -15: "Spread_M1_M2_t_minus_15",
   -30: "Spread_M1_M2_t_minus_30",
}

# Technical indicators (stored per event, used as retrieval features)
TECH_COLS = ["RSI", "SMA_7", "EMA_7", "Bollinger_Upper", "Bollinger_Lower"]


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EventRecord:
    event_id:        str
    date:            pd.Timestamp
    year:            float
    polarity:        str
    quantile_cat:    str
    quantile_group:  str
    event_type:      str
    influence_score: float
    rsi:             float
    embedding:       np.ndarray
    price_nodes:     Dict[int, float]
    spread_nodes:    Dict[int, float]


@dataclass
class RetrievalResult:
    event_id:        str
    date:            str
    similarity:      float
    combined_score:  float
    recency_weight:  float
    quantile_cat:    str
    event_type:      str
    polarity:        str
    influence_score: float
    rsi:             float
    price_path:      Dict[int, float]   # % return vs event-date price
    spread_path:     Dict[int, float]   # spread delta vs event-date
    r7:              float
    r15:             float
    r30:             float
    spread_chg_7:    float
    spread_chg_30:   float


@dataclass
class RetrievalOutput:
    base_event_id:   str
    base_date:       str
    tau:             float
    top_k:           int
    n_candidates:    int
    n_analogues:     int
    analogues:       List[RetrievalResult]
    bert_dtw_mean:   float
    random_dtw_mean: float
    dtw_pvalue:      float
    realised_path:   Dict[int, float]
    realised_r30:    float


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", " ", s)


def _parse_embedding(raw) -> Optional[np.ndarray]:
    """Parse bert_embeddings — handles list, ndarray, or string."""
    if raw is None:
        return None
    if isinstance(raw, (list, np.ndarray)):
        arr = np.array(raw, dtype=np.float32).ravel()
        return arr if len(arr) >= 100 else None
    if isinstance(raw, str):
        clean = _strip_ansi(raw).replace("[","").replace("]","").replace("\n"," ")
        nums = []
        for t in clean.split():
            try:
                nums.append(float(t.rstrip(",")))
            except ValueError:
                pass
        if len(nums) >= 100:
            return np.array(nums, dtype=np.float32)
    return None


def _safe_float(v, default: float = 0.0) -> float:
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _quantile_group(cat: str) -> str:
    return QUANTILE_GROUPS.get(str(cat).strip(), "mid")


def _norm_return(price_t: float, price_0: float) -> float:
    if not price_0 or not np.isfinite(price_0):
        return 0.0
    return (price_t - price_0) / abs(price_0) * 100.0


def _path_array(path: Dict[int, float],
                use_increments: bool = True) -> np.ndarray:
    """
    Convert price path dict to numpy array for DTW.

    use_increments=True (default, per supervisor feedback):
        Day-over-day return increments — preserves shock timing,
        prevents cumulative smoothing from hiding local reactions.

        Cumulative: [-4.3, -0.3, -0.8,  0.0, +4.4, +5.9, +1.5]
        Increments: [+4.0, -0.5, +0.8, +4.4, +1.5, -4.4]  (np.diff)

    use_increments=False:
        Raw cumulative normalised returns.
    """
    cumulative = np.array([path.get(d, 0.0) for d in PRICE_NODES],
                           dtype=np.float64)
    if use_increments:
        return np.diff(cumulative)
    return cumulative


def _row_to_record(row: dict, idx: int) -> Optional[EventRecord]:
    """Convert one JSON row to EventRecord. Returns None if unusable."""

    # ── Embedding (inline in JSON — confirmed) ────────────────────────────────
    emb = _parse_embedding(row.get("bert_embeddings"))
    if emb is None:
        return None
    if len(emb) < EMB_DIM:
        emb = np.pad(emb, (0, EMB_DIM - len(emb)))
    else:
        emb = emb[:EMB_DIM]

    # ── Date ──────────────────────────────────────────────────────────────────
    raw_date = row.get("date") or row.get("Date") or ""
    try:
        date = pd.to_datetime(str(raw_date), errors="raise")
    except Exception:
        return None

    # ── Event ID ──────────────────────────────────────────────────────────────
    eid = str(
        row.get("event_id")
        or row.get("Name")
        or f"EVT_{date.year:.0f}_{idx:05d}.0"
    )

    # ── Price nodes ───────────────────────────────────────────────────────────
    p0 = _safe_float(
        row.get("price")
        or row.get("Crude Oil-WTI Spot Cushing U$/BBL")
        or 0.0
    )
    price_nodes: Dict[int, float] = {}
    for day, col in PRICE_COLS.items():
        val = _safe_float(row.get(col, None))
        price_nodes[day] = val if val != 0.0 else p0

    # Ensure day-0 is set correctly
    price_nodes[0] = p0 if p0 else price_nodes.get(0, 0.0)

    # ── Spread nodes ──────────────────────────────────────────────────────────
    s0 = _safe_float(row.get("Spread_M1_M2", 0.0))
    spread_nodes: Dict[int, float] = {}
    for day, col in SPREAD_COLS.items():
        spread_nodes[day] = _safe_float(row.get(col, s0))
    spread_nodes[0] = s0

    # ── Event type (multiple possible field names) ─────────────────────────────
    etype = str(
        row.get("Event type")
        or row.get("Event_type")
        or row.get("Event Types")
        or row.get("category")
        or ""
    ).strip()

    qcat = str(row.get("Quantile_Category", "")).strip()

    return EventRecord(
        event_id        = eid,
        date            = date,
        year            = _safe_float(row.get("year") or row.get("Year") or date.year),
        polarity        = str(row.get("Polarity", "Neutral")).strip(),
        quantile_cat    = qcat,
        quantile_group  = _quantile_group(qcat),
        event_type      = etype,
        influence_score = _safe_float(row.get("Influence_Score", 0.0)),
        rsi             = _safe_float(row.get("RSI", 50.0)),
        embedding       = emb.astype(np.float32),
        price_nodes     = price_nodes,
        spread_nodes    = spread_nodes,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CORPUS LOADER  — handles both .json (list) and .jsonl (line-delimited)
# ─────────────────────────────────────────────────────────────────────────────

def load_corpus(path: Path, label: str = "corpus") -> List[EventRecord]:
    if not path.exists():
        log.error(f"FILE NOT FOUND: {path}")
        sys.exit(1)

    size_mb = path.stat().st_size / 1e6
    log.info(f"[load] {label}  ({path.name}, {size_mb:.0f} MB)")

    records: List[EventRecord] = []
    bad = 0

    # ── Auto-detect format ───────────────────────────────────────────────────
    # Peek at first non-empty line to decide format
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        first_line = ""
        for line in fh:
            first_line = line.strip()
            if first_line:
                break

    is_json_array = first_line.startswith("[")

    if is_json_array:
        # Pretty-printed or compact JSON array  e.g. price_event_memory.json
        log.info("[load] format: JSON array (pretty-printed)")
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            raw = json.load(fh)
        if not isinstance(raw, list):
            # Could be a dict-of-events
            raw = list(raw.values())
        for idx, row in enumerate(tqdm(raw, desc=f"  {label}", unit=" events")):
            rec = _row_to_record(row, idx)
            if rec is None:
                bad += 1
            else:
                records.append(rec)
    else:
        # Line-delimited JSON  e.g. df_major_memory_2024_2026.jsonl
        log.info("[load] format: JSONL (line-delimited)")
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for idx, line in enumerate(
                tqdm(fh, desc=f"  {label}", unit=" events")
            ):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
                    continue
                rec = _row_to_record(row, idx)
                if rec is None:
                    bad += 1
                else:
                    records.append(rec)

    log.info(f"[load]   {len(records):>6} valid events")
    log.info(f"[load]   {bad:>6} skipped (bad embedding / date)")

    if not records:
        log.error("No valid records — check file format.")
        sys.exit(1)

    # Year distribution
    from collections import Counter
    yc = Counter(int(r.year) for r in records)
    yr_range = f"{min(yc)}-{max(yc)}"
    log.info(f"[load]   year range: {yr_range}")

    return records


# ─────────────────────────────────────────────────────────────────────────────
# DTW  (pure numpy, no external dep, O(n*m) fine for 7-point paths)
# ─────────────────────────────────────────────────────────────────────────────

def dtw_distance(a: np.ndarray, b: np.ndarray,
                  sakoe_chiba_radius: int = 2) -> float:
    """
    Constrained DTW with:
      - Squared distance (penalises large regime mismatches more)
      - Sakoe-Chiba band (prevents unrealistic temporal warping)
        radius=2 means each timepoint can only align ±2 steps away

    With 7-point paths (nodes at -30,-15,-7,0,+7,+15,+30):
      radius=2 allows ±2 node alignment — reasonable given uneven spacing.

    NOTE: Supervisor feedback acknowledged — 7-point paths limit DTW power.
    This metric is used as a trajectory-shape heuristic, not high-resolution
    temporal alignment. Full 61-day constrained DTW deferred to next version
    pending daily price data availability.
    """
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Sakoe-Chiba band constraint
            if abs(i - j) > sakoe_chiba_radius:
                continue
            # Squared distance — penalises large deviations more
            cost = (float(a[i-1]) - float(b[j-1])) ** 2
            D[i, j] = cost + min(D[i-1,j], D[i,j-1], D[i-1,j-1])
    return float(np.sqrt(D[n, m]))   # sqrt to restore interpretable scale


def mean_dtw(paths: np.ndarray, target: np.ndarray) -> float:
    if len(paths) == 0:
        return float("nan")
    return float(np.mean([dtw_distance(p, target) for p in paths]))


# ─────────────────────────────────────────────────────────────────────────────
# RETRIEVAL ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class RetrievalEngine:
    """
    Event analogue retrieval engine.

    Parameters
    ----------
    train_path      : path to price_event_memory.json (2001-2023)
    test_path       : path to 2024-2026 JSONL
    recency_lambda  : decay rate for recency weighting (default 0.05)
    """

    def __init__(
        self,
        train_path:     Path,
        test_path:      Path,
        recency_lambda: float = 0.05,
    ):
        self.lam = recency_lambda

        log.info("=" * 60)
        log.info("RetrievalEngine — initialising")
        log.info("=" * 60)

        self.train_corpus = load_corpus(train_path, "TRAIN 2001-2023")
        self.test_corpus  = load_corpus(test_path,  "TEST  2024-2026")

        # L2-normalised embedding matrix for vectorised cosine similarity
        log.info("[index] building embedding index ...")
        train_embs = np.stack([r.embedding for r in self.train_corpus])
        self._train_norm = normalize(train_embs, norm="l2").astype(np.float32)
        log.info(f"[index] shape: {self._train_norm.shape}")

        # Fast lookups
        self._test_lookup: Dict[str, int] = {
            r.event_id: i for i, r in enumerate(self.test_corpus)
        }
        self._train_lookup: Dict[str, int] = {
            r.event_id: i for i, r in enumerate(self.train_corpus)
        }

        train_yrs = sorted(set(int(r.year) for r in self.train_corpus))
        test_yrs  = sorted(set(int(r.year) for r in self.test_corpus))
        log.info(f"[index] train: {train_yrs[0]}-{train_yrs[-1]}"
                 f"  test: {test_yrs[0]}-{test_yrs[-1]}")
        log.info("RetrievalEngine ready.\n")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _recency_weight(self, analogue_date: pd.Timestamp,
                        base_date: pd.Timestamp) -> float:
        age = max((base_date - analogue_date).days / 365.25, 0.0)
        return float(np.exp(-self.lam * age))

    def _extract_paths(
        self, rec: EventRecord
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        p0 = rec.price_nodes.get(0, 1.0) or 1.0
        s0 = rec.spread_nodes.get(0, 0.0)
        price_path  = {d: _norm_return(rec.price_nodes.get(d, p0), p0)
                       for d in PRICE_NODES}
        spread_path = {d: float(rec.spread_nodes.get(d, s0) - s0)
                       for d in PRICE_NODES}
        return price_path, spread_path

    def _find_event(self, event_id: str) -> EventRecord:
        if event_id in self._test_lookup:
            return self.test_corpus[self._test_lookup[event_id]]
        if event_id in self._train_lookup:
            return self.train_corpus[self._train_lookup[event_id]]
        # Partial match fallback
        for r in self.test_corpus:
            if event_id.lower() in r.event_id.lower():
                log.warning(f"Partial match: '{event_id}' → '{r.event_id}'")
                return r
        sample = [r.event_id for r in self.test_corpus[:8]]
        raise KeyError(
            f"event_id '{event_id}' not found.\n"
            f"Sample test IDs: {sample}\n"
            f"Tip: run --list_events to see all available IDs."
        )

    # ── Main retrieval ────────────────────────────────────────────────────────

    def retrieve(
        self,
        base_event_id: str,
        tau:           float = 0.75,
        top_k:         int   = 50,
        random_seed:   int   = 42,
    ) -> RetrievalOutput:
        """
        Four-step retrieval pipeline:

        STEP 1 — BERT cosine similarity (only filter: tau + temporal exclusion)
            Quantile is NOT a filter here — it is analysed as an output.

        STEP 2 — Quantile analysis of retrieved set
            Describe quantile distribution of candidates.
            Used for KDE stratification downstream.

        STEP 3 — DTW ranking
            Rank candidates by DTW distance between their price path
            and the base event price path. Lower = more similar trajectory.

        STEP 4 — Persistence scoring + final rerank
            Persistence = how many days (7/15/30) the event-day price
            direction was sustained. Higher persistence = stronger analogue.
            Final rank: persistence desc, then DTW asc.
        """
        base      = self._find_event(base_event_id)
        base_date = base.date

        log.info(f"{'─'*60}")
        log.info(f"RETRIEVE: {base_event_id}  ({base_date.date()})")
        log.info(f"  quantile={base.quantile_cat}  polarity={base.polarity}"
                 f"  rsi={base.rsi:.1f}  tau={tau}  top_k={top_k}")

        # ── STEP 1: BERT cosine similarity ────────────────────────────────────
        # Only filters: similarity > tau AND analogue predates base event
        # No quantile filter — quantile is an output characteristic
        base_norm = normalize(
            base.embedding.reshape(1, -1), norm="l2"
        ).astype(np.float32)
        sims = (self._train_norm @ base_norm.T).ravel()

        step1: List[Tuple[int, float]] = []
        for i, sim in enumerate(sims):
            if self.train_corpus[i].date >= base_date:
                continue
            # Skip earnings/company reports — not market-moving events
            etype = self.train_corpus[i].event_type.lower().strip()
            if etype in EXCLUDE_EVENT_TYPES:
                continue
            step1.append((i, float(sim)))

        all_sims = [s for _, s in step1]

        # ── Adaptive tau ──────────────────────────────────────────────────────
        # Start at requested tau. If fewer than MIN_ANALOGUES candidates,
        # automatically relax tau in steps until we have enough.
        # This handles rare/extreme events (e.g. military buildups, novel crises)
        # that have few close historical parallels.
        MIN_ANALOGUES  = 10
        TAU_FALLBACKS  = [tau, 0.87, 0.85, 0.80, 0.75, 0.0]

        effective_tau = tau
        filtered = [(i, s) for i, s in step1 if s >= tau]

        if len(filtered) < MIN_ANALOGUES:
            for fallback in TAU_FALLBACKS:
                if fallback >= tau:
                    continue
                filtered = [(i, s) for i, s in step1 if s >= fallback]
                if len(filtered) >= MIN_ANALOGUES:
                    effective_tau = fallback
                    log.info(f"  ADAPTIVE TAU: relaxed {tau} → {fallback} "
                             f"(got {len(filtered)} candidates)")
                    break

        step1 = filtered
        if effective_tau != tau:
            log.info(f"  NOTE: tau relaxed from {tau} to {effective_tau} "
                     f"— event may be rare/unprecedented (see paper §4.3)")
        else:
            log.info(f"  STEP 1 — tau={tau}: {len(step1)} candidates")

        # Log similarity distribution for diagnostics
        if all_sims:
            for thresh in [0.90, 0.87, 0.85, 0.80, 0.75]:
                cnt = sum(1 for s in all_sims if s >= thresh)
                log.debug(f"    > {thresh}: {cnt:>5} events")

        # ── STEP 2: Quantile analysis of retrieved set ────────────────────────
        from collections import Counter
        qcounts = Counter(self.train_corpus[i].quantile_cat for i, _ in step1)
        log.info(f"  STEP 2 — Quantile distribution of candidates:")
        for qcat, cnt in sorted(qcounts.items(), key=lambda x: -x[1]):
            pct = cnt / max(len(step1), 1) * 100
            log.info(f"    {qcat:<15} {cnt:>4}  ({pct:.0f}%)")

        # ── STEP 3: DTW ranking ───────────────────────────────────────────────
        base_pp, base_sp = self._extract_paths(base)
        realised_arr = _path_array(base_pp)

        step3 = []
        for i, sim in step1:
            pp, sp = self._extract_paths(self.train_corpus[i])
            dtw    = dtw_distance(_path_array(pp), realised_arr)
            step3.append((i, sim, dtw, pp, sp))

        step3.sort(key=lambda x: x[2])   # sort by DTW ascending
        log.info(f"  STEP 3 — DTW range: "
                 f"[{step3[0][2]:.2f}, {step3[-1][2]:.2f}]"
                 if step3 else "  STEP 3 — no candidates")

        # ── STEP 4: Persistence scoring + final rerank ────────────────────────
        def persistence(pp: Dict[int, float]) -> int:
            """Days (out of 7/15/30) that price sustained event-day direction."""
            r7  = pp.get(7,  0.0)
            r15 = pp.get(15, 0.0)
            r30 = pp.get(30, 0.0)
            direction = 1 if r7 >= 0 else -1
            score = 0
            if r7  * direction > 0: score += 1
            if r15 * direction > 0: score += 1
            if r30 * direction > 0: score += 1
            return score

        # Take top 2×top_k by DTW, rerank by persistence then DTW
        pool = step3[:top_k * 2]
        pool_scored = []
        for i, sim, dtw, pp, sp in pool:
            rec  = self.train_corpus[i]
            pers = persistence(pp)
            rw   = self._recency_weight(rec.date, base_date)
            pool_scored.append((i, sim, dtw, pers, rw, pp, sp))

        pool_scored.sort(key=lambda x: (-x[3], x[2]))   # persist desc, DTW asc
        final = pool_scored[:top_k]

        log.info(f"  STEP 4 — Persistence rerank: {len(final)} analogues")

        # ── Build output ──────────────────────────────────────────────────────
        analogues:     List[RetrievalResult] = []
        analogue_arrs: List[np.ndarray]      = []

        for i, sim, dtw, pers, rw, pp, sp in final:
            rec = self.train_corpus[i]
            arr = _path_array(pp)
            analogue_arrs.append(arr)
            analogues.append(RetrievalResult(
                event_id        = rec.event_id,
                date            = str(rec.date.date()),
                similarity      = round(sim,      6),
                combined_score  = round(sim * rw, 6),
                recency_weight  = round(rw,       6),
                quantile_cat    = rec.quantile_cat,
                event_type      = rec.event_type,
                polarity        = rec.polarity,
                influence_score = rec.influence_score,
                rsi             = rec.rsi,
                price_path      = pp,
                spread_path     = sp,
                r7              = round(pp.get(7,  0.0), 4),
                r15             = round(pp.get(15, 0.0), 4),
                r30             = round(pp.get(30, 0.0), 4),
                spread_chg_7    = round(sp.get(7,  0.0), 4),
                spread_chg_30   = round(sp.get(30, 0.0), 4),
            ))

        if analogues:
            pers_vals = [pool_scored[j][3] for j in range(min(5, len(final)))]
            log.info(f"  Persistence scores (top 5): {pers_vals}")
            sims_vals = [a.similarity for a in analogues]
            log.info(f"  Similarity range: "
                     f"[{min(sims_vals):.3f}, {max(sims_vals):.3f}]")

        # ── DTW vs random baseline (RQ1) ──────────────────────────────────────
        rng    = np.random.default_rng(random_seed)
        n_rand = min(50, len(self.train_corpus))
        rand_arrs = []
        for ri in rng.choice(len(self.train_corpus), size=n_rand, replace=False):
            rpp, _ = self._extract_paths(self.train_corpus[ri])
            rand_arrs.append(_path_array(rpp))

        bert_paths = np.stack(analogue_arrs) if analogue_arrs else np.array([])
        rand_paths = np.stack(rand_arrs)
        bert_dtw   = mean_dtw(bert_paths, realised_arr) if len(bert_paths) else float("nan")
        random_dtw = mean_dtw(rand_paths, realised_arr)

        dtw_pvalue = float("nan")
        if len(analogue_arrs) >= 10:
            try:
                from scipy.stats import wilcoxon
                bd = [dtw_distance(p, realised_arr) for p in bert_paths]
                rd = [dtw_distance(p, realised_arr) for p in rand_paths[:len(bd)]]
                _, dtw_pvalue = wilcoxon(bd, rd)
            except Exception:
                pass

        log.info(f"  BERT DTW={bert_dtw:.4f}  Random DTW={random_dtw:.4f}"
                 + (f"  p={dtw_pvalue:.4f}" if np.isfinite(dtw_pvalue) else ""))

        return RetrievalOutput(
            base_event_id   = base_event_id,
            base_date       = str(base_date.date()),
            tau             = tau,
            top_k           = top_k,
            n_candidates    = len(step1),
            n_analogues     = len(analogues),
            analogues       = analogues,
            bert_dtw_mean   = round(bert_dtw,   4) if np.isfinite(bert_dtw)   else -1.0,
            random_dtw_mean = round(random_dtw, 4) if np.isfinite(random_dtw) else -1.0,
            dtw_pvalue      = round(dtw_pvalue, 6) if np.isfinite(dtw_pvalue) else -1.0,
            realised_path   = base_pp,
            realised_r30    = round(base_pp.get(30, 0.0), 4),
        )

    # ── Batch retrieve ─────────────────────────────────────────────────────────

    def batch_retrieve(
        self,
        tau:        float = 0.75,
        top_k:      int   = 50,
        output_dir: Optional[Path] = None,
    ) -> pd.DataFrame:
        """Run retrieve() over all test events. Saves per-event JSON."""
        if output_dir:
            per_dir = Path(output_dir) / "per_event"
            per_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"\n{'='*60}")
        log.info(f"BATCH RETRIEVE: {len(self.test_corpus)} test events")
        log.info(f"  tau={tau}  top_k={top_k}")
        log.info(f"{'='*60}")

        rows = []
        for rec in tqdm(self.test_corpus, desc="batch", unit=" events"):
            try:
                res = self.retrieve(
                    base_event_id = rec.event_id,
                    tau           = tau,
                    top_k         = top_k,
                )
            except Exception as e:
                log.warning(f"SKIP {rec.event_id}: {e}")
                continue

            r30s = [a.r30 for a in res.analogues]
            r15s = [a.r15 for a in res.analogues]
            r7s  = [a.r7  for a in res.analogues]

            rows.append(dict(
                event_id          = rec.event_id,
                date              = str(rec.date.date()),
                year              = int(rec.year),
                quantile_cat      = rec.quantile_cat,
                polarity          = rec.polarity,
                event_type        = rec.event_type,
                influence_score   = rec.influence_score,
                rsi               = rec.rsi,
                n_candidates      = res.n_candidates,
                n_analogues       = res.n_analogues,
                tau_requested     = tau,
                tau_effective     = res.tau,
                tau_relaxed       = int(abs(res.tau - tau) > 0.001),
                bert_dtw          = res.bert_dtw_mean,
                random_dtw        = res.random_dtw_mean,
                dtw_pvalue        = res.dtw_pvalue,
                realised_r30      = res.realised_r30,
                # Analogue distribution → fed into KDE engine
                analogue_r7_mean  = np.nanmean(r7s)  if r7s else np.nan,
                analogue_r15_mean = np.nanmean(r15s) if r15s else np.nan,
                analogue_r30_mean = np.nanmean(r30s) if r30s else np.nan,
                analogue_r30_std  = np.nanstd(r30s)  if r30s else np.nan,
                analogue_r30_p5   = np.nanpercentile(r30s,  5) if r30s else np.nan,
                analogue_r30_p25  = np.nanpercentile(r30s, 25) if r30s else np.nan,
                analogue_r30_p75  = np.nanpercentile(r30s, 75) if r30s else np.nan,
                analogue_r30_p95  = np.nanpercentile(r30s, 95) if r30s else np.nan,
            ))

            if output_dir:
                out_file = per_dir / f"{rec.event_id}_analogues.json"
                with out_file.open("w", encoding="utf-8") as f:
                    json.dump({
                        "base_event_id":   res.base_event_id,
                        "base_date":       res.base_date,
                        "n_analogues":     res.n_analogues,
                        "bert_dtw_mean":   res.bert_dtw_mean,
                        "random_dtw_mean": res.random_dtw_mean,
                        "dtw_pvalue":      res.dtw_pvalue,
                        "realised_path":   res.realised_path,
                        "realised_r30":    res.realised_r30,
                        "analogues":       [asdict(a) for a in res.analogues],
                    }, f, ensure_ascii=False, indent=2, default=str)

        df = pd.DataFrame(rows)
        if len(df):
            log.info(f"\n[batch] {len(df)} events processed")
            log.info(f"[batch] mean analogues/event : {df['n_analogues'].mean():.1f}")
            log.info(f"[batch] zero-analogue events : {(df['n_analogues']==0).sum()}")
            log.info(f"[batch] BERT < random DTW    : "
                     f"{(df['bert_dtw']<df['random_dtw']).sum()}/{len(df)}")
        return df

    # ── Table 1 ───────────────────────────────────────────────────────────────

    def summary_table1(
        self,
        thresholds: Tuple = (0.70, 0.75, 0.80),
    ) -> pd.DataFrame:
        """Generate Table 1 of the paper: DTW by similarity threshold."""
        from scipy.stats import wilcoxon
        rows = []
        for tau in thresholds:
            log.info(f"\n[table1] tau={tau} ...")
            df = self.batch_retrieve(tau=tau)
            if df.empty:
                continue
            valid = df.dropna(subset=["bert_dtw", "random_dtw"])
            try:
                _, pval = wilcoxon(valid["bert_dtw"], valid["random_dtw"])
            except Exception:
                pval = float("nan")
            improv = (
                (valid["random_dtw"].mean() - valid["bert_dtw"].mean())
                / valid["random_dtw"].mean() * 100
            )
            rows.append({
                "tau":               tau,
                "n_analogues_mean":  round(valid["n_analogues"].mean(), 1),
                "bert_dtw_mean":     round(valid["bert_dtw"].mean(), 4),
                "random_dtw_mean":   round(valid["random_dtw"].mean(), 4),
                "dtw_improvement_%": round(improv, 1),
                "wilcoxon_p":        round(pval, 6) if np.isfinite(pval) else "n/a",
                "n_events":          len(valid),
            })
        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--train",  required=True,
                    help="path to price_event_memory.json (2001-2023)")
    ap.add_argument("--test",   required=True,
                    help="path to 2024-2026 JSONL")
    ap.add_argument("--event",  default=None,
                    help="event_id for single-event mode")
    ap.add_argument("--tau",    type=float, default=0.75)
    ap.add_argument("--top_k",  type=int,   default=50)
    ap.add_argument("--output", default="retrieval_output")
    ap.add_argument("--table1", action="store_true",
                    help="generate Table 1 (three tau thresholds)")
    ap.add_argument("--list_events", action="store_true",
                    help="print test event_ids and exit")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    engine = RetrievalEngine(
        train_path = Path(args.train),
        test_path  = Path(args.test),
    )

    # ── List events ───────────────────────────────────────────────────────────
    if args.list_events:
        print(f"\nFirst 20 TEST event_ids:")
        for r in engine.test_corpus[:20]:
            print(f"  {r.event_id:<30}  {r.date.date()}  "
                  f"{r.quantile_cat:<12}  {r.event_type[:35]}")
        return

    # ── Table 1 ───────────────────────────────────────────────────────────────
    if args.table1:
        t1 = engine.summary_table1()
        p  = out_dir / "table1_dtw_by_threshold.csv"
        t1.to_csv(p, index=False)
        print(f"\n{'='*55}")
        print(t1.to_string(index=False))
        print(f"\nSaved -> {p}")
        return

    # ── Single event ──────────────────────────────────────────────────────────
    if args.event:
        res = engine.retrieve(
            base_event_id = args.event,
            tau           = args.tau,
            top_k         = args.top_k,
        )
        out_file = out_dir / f"{args.event}_analogues.json"
        with out_file.open("w", encoding="utf-8") as f:
            json.dump({
                "base_event_id":   res.base_event_id,
                "base_date":       res.base_date,
                "n_analogues":     res.n_analogues,
                "bert_dtw_mean":   res.bert_dtw_mean,
                "random_dtw_mean": res.random_dtw_mean,
                "dtw_pvalue":      res.dtw_pvalue,
                "realised_r30":    res.realised_r30,
                "realised_path":   res.realised_path,
                "analogues":       [asdict(a) for a in res.analogues],
            }, f, ensure_ascii=False, indent=2, default=str)

        print(f"\n{'='*55}")
        print(f"  Base event   : {res.base_event_id}")
        print(f"  Date         : {res.base_date}")
        print(f"  Analogues    : {res.n_analogues} "
              f"(from {res.n_candidates} candidates)")
        print(f"  BERT DTW     : {res.bert_dtw_mean}")
        print(f"  Random DTW   : {res.random_dtw_mean}")
        print(f"  p-value      : {res.dtw_pvalue}")
        print(f"  Realised r30 : {res.realised_r30}%")
        if res.analogues:
            print(f"\n  Top 5 analogues:")
            for a in res.analogues[:5]:
                print(f"    {a.date}  sim={a.similarity:.3f}  "
                      f"r30={a.r30:+.1f}%  {a.event_type[:35]}")
        print(f"\n  Saved -> {out_file}")
        print(f"{'='*55}")
        return

    # ── Full batch ────────────────────────────────────────────────────────────
    df = engine.batch_retrieve(
        tau        = args.tau,
        top_k      = args.top_k,
        output_dir = out_dir,
    )
    csv_path = out_dir / f"batch_tau{args.tau}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n{'='*55}")
    print(f"  Events processed : {len(df)}")
    print(f"  Mean analogues   : {df['n_analogues'].mean():.1f}")
    print(f"  BERT beats random: "
          f"{(df['bert_dtw'] < df['random_dtw']).sum()}/{len(df)}")
    print(f"  Saved -> {csv_path}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()