"""
retrieval_engine_v2.py
=======================
Side-by-side comparison of three retrieval methods:

  METHOD A — BERT cosine similarity (v1 baseline)
  METHOD B — FAISS flat index (exact inner product, same math as cosine
              but indexed — shows speed difference)
  METHOD C — Hybrid score
              0.60 * cosine_similarity
            + 0.25 * entity_jaccard_overlap
            + 0.15 * event_type_match

For each base event, all three methods run and their results are compared:
  - Top-K analogues (who appears in each method's list)
  - DTW of retrieved set vs realised path
  - Overlap between methods (Jaccard of top-50 sets)
  - Speed (seconds per query)

Usage
-----
    python retrieval_engine_v2.py \
        --train "memory_2001_2023/price_event_memory.json" \
        --test  "memory_2024_2026/df_major_memory_2024_2026.jsonl" \
        --event EVT_2026.0_0001.0 \
        --tau 0.25 --top_k 50 \
        --output retrieval_v2_output/

    # Full comparison across all test events:
    python retrieval_engine_v2.py \
        --train "memory_2001_2023/price_event_memory.json" \
        --test  "memory_2024_2026/df_major_memory_2024_2026.jsonl" \
        --tau 0.25 --top_k 50 \
        --output retrieval_v2_output/

Dependencies
------------
    pip install numpy pandas scikit-learn scipy tqdm faiss-cpu
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
import warnings
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from tqdm import tqdm

try:
    import faiss
except ImportError:
    sys.exit("ERROR: pip install faiss-cpu")

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS  (same as v1)
# ─────────────────────────────────────────────────────────────────────────────

EMB_DIM    = 768
PRICE_NODES = [-30, -15, -7, 0, 7, 15, 30]
PRICE_COLS: Dict[int, str] = {
    -30: "price_t_minus_30", -15: "price_t_minus_15",
     -7: "price_t_minus_7",    0: "price",
      7: "price_t_plus_7",    15: "price_t_plus_15",
     30: "price_t_plus_30",
}
SPREAD_COLS: Dict[int, str] = {
     0: "Spread_M1_M2",         7: "Spread_M1_M2_t_plus_7",
    15: "Spread_M1_M2_t_plus_15", 30: "Spread_M1_M2_t_plus_30",
    -7: "Spread_M1_M2_t_minus_7",-15: "Spread_M1_M2_t_minus_15",
   -30: "Spread_M1_M2_t_minus_30",
}

TOKEN_SPLIT = re.compile(r"[,;\n/|]+")


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
    event_type:      str
    entity_tokens:   Set[str]          # tokenised Entity field
    influence_score: float
    rsi:             float
    embedding:       np.ndarray
    price_nodes:     Dict[int, float]
    spread_nodes:    Dict[int, float]


@dataclass
class AnalogueResult:
    event_id:       str
    date:           str
    method:         str              # "cosine" | "faiss" | "hybrid"
    score:          float            # method-specific score
    cosine_sim:     float            # always stored for comparison
    entity_jaccard: float
    type_match:     int
    dtw:            float
    persistence:    int
    quantile_cat:   str
    event_type:     str
    r7:             float
    r15:            float
    r30:            float


@dataclass
class ComparisonOutput:
    base_event_id:    str
    base_date:        str
    tau:              float
    # Per-method results
    cosine_analogues: List[AnalogueResult]
    faiss_analogues:  List[AnalogueResult]
    hybrid_analogues: List[AnalogueResult]
    # DTW metrics per method
    cosine_dtw_mean:  float
    faiss_dtw_mean:   float
    hybrid_dtw_mean:  float
    random_dtw_mean:  float
    # Speed (seconds)
    cosine_time:      float
    faiss_time:       float
    hybrid_time:      float
    # Overlap between methods
    cosine_faiss_jaccard:  float     # how similar are v1 and FAISS top-50?
    cosine_hybrid_jaccard: float     # how much does hybrid change the list?
    faiss_hybrid_jaccard:  float
    # Realised
    realised_r30:     float


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", " ", s)

def _parse_embedding(raw) -> Optional[np.ndarray]:
    if raw is None: return None
    if isinstance(raw, (list, np.ndarray)):
        arr = np.array(raw, dtype=np.float32).ravel()
        return arr if len(arr) >= 100 else None
    if isinstance(raw, str):
        clean = _strip_ansi(raw).replace("[","").replace("]","").replace("\n"," ")
        nums = []
        for t in clean.split():
            try: nums.append(float(t.rstrip(",")))
            except ValueError: pass
        if len(nums) >= 100:
            return np.array(nums, dtype=np.float32)
    return None

def _safe_float(v, default=0.0) -> float:
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except: return default

def _tokenise_entities(entity_str: str) -> Set[str]:
    tokens = set()
    for tok in TOKEN_SPLIT.split(str(entity_str)):
        t = tok.strip().lower()
        if t and len(t) > 2 and t not in ("nan","none","n/a"):
            tokens.add(t)
    return tokens

def _entity_jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b: return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

def _norm_return(pt: float, p0: float) -> float:
    if not p0 or not np.isfinite(p0): return 0.0
    return (pt - p0) / abs(p0) * 100.0

def _path_array(nodes: Dict[int, float]) -> np.ndarray:
    return np.array([nodes.get(d, 0.0) for d in PRICE_NODES], dtype=np.float64)

def _persistence(pp: Dict[int, float]) -> int:
    r7, r15, r30 = pp.get(7,0.0), pp.get(15,0.0), pp.get(30,0.0)
    direction = 1 if r7 >= 0 else -1
    score = 0
    if r7  * direction > 0: score += 1
    if r15 * direction > 0: score += 1
    if r30 * direction > 0: score += 1
    return score

def _set_jaccard(a: Set, b: Set) -> float:
    if not a and not b: return 1.0
    return len(a & b) / len(a | b) if (a | b) else 0.0

def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    n, m = len(a), len(b)
    D = np.full((n+1, m+1), np.inf)
    D[0,0] = 0.0
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(float(a[i-1]) - float(b[j-1]))
            D[i,j] = cost + min(D[i-1,j], D[i,j-1], D[i-1,j-1])
    return float(D[n,m])

def mean_dtw(paths: np.ndarray, target: np.ndarray) -> float:
    if len(paths) == 0: return float("nan")
    return float(np.mean([dtw_distance(p, target) for p in paths]))


# ─────────────────────────────────────────────────────────────────────────────
# CORPUS LOADER
# ─────────────────────────────────────────────────────────────────────────────

def _row_to_record(row: dict, idx: int) -> Optional[EventRecord]:
    emb = _parse_embedding(row.get("bert_embeddings"))
    if emb is None: return None
    if len(emb) < EMB_DIM: emb = np.pad(emb, (0, EMB_DIM - len(emb)))
    else: emb = emb[:EMB_DIM]

    raw_date = row.get("date") or ""
    try: date = pd.to_datetime(str(raw_date), errors="raise")
    except: return None

    eid = str(row.get("event_id") or f"EVT_{date.year:.0f}_{idx:05d}.0")

    p0 = _safe_float(row.get("price") or row.get("Crude Oil-WTI Spot Cushing U$/BBL") or 0.0)
    price_nodes = {d: _safe_float(row.get(c, p0)) for d, c in PRICE_COLS.items()}
    price_nodes[0] = p0 or price_nodes.get(0, 0.0)

    s0 = _safe_float(row.get("Spread_M1_M2", 0.0))
    spread_nodes = {d: _safe_float(row.get(c, s0)) for d, c in SPREAD_COLS.items()}
    spread_nodes[0] = s0

    etype = str(
        row.get("Event type") or row.get("Event_type") or
        row.get("Event Types") or row.get("category") or ""
    ).strip()

    return EventRecord(
        event_id        = eid,
        date            = date,
        year            = _safe_float(row.get("year") or date.year),
        polarity        = str(row.get("Polarity","Neutral")).strip(),
        quantile_cat    = str(row.get("Quantile_Category","")).strip(),
        event_type      = etype,
        entity_tokens   = _tokenise_entities(row.get("Entity","")),
        influence_score = _safe_float(row.get("Influence_Score",0.0)),
        rsi             = _safe_float(row.get("RSI",50.0)),
        embedding       = emb.astype(np.float32),
        price_nodes     = price_nodes,
        spread_nodes    = spread_nodes,
    )


def load_corpus(path: Path, label: str = "corpus") -> List[EventRecord]:
    if not path.exists():
        log.error(f"NOT FOUND: {path}"); sys.exit(1)

    size_mb = path.stat().st_size / 1e6
    log.info(f"[load] {label}  ({path.name}, {size_mb:.0f} MB)")
    records, bad = [], 0

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        first = ""
        for line in fh:
            first = line.strip()
            if first: break

    if first.startswith("["):
        log.info("[load] format: JSON array")
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            raw = json.load(fh)
        for idx, row in enumerate(tqdm(raw, desc=f"  {label}", unit=" ev")):
            rec = _row_to_record(row, idx)
            if rec: records.append(rec)
            else:   bad += 1
    else:
        log.info("[load] format: JSONL")
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for idx, line in enumerate(tqdm(fh, desc=f"  {label}", unit=" ev")):
                line = line.strip()
                if not line: continue
                try: row = json.loads(line)
                except: bad += 1; continue
                rec = _row_to_record(row, idx)
                if rec: records.append(rec)
                else:   bad += 1

    log.info(f"[load]   {len(records)} valid  |  {bad} skipped")
    if not records:
        log.error("No valid records."); sys.exit(1)
    return records


# ─────────────────────────────────────────────────────────────────────────────
# RETRIEVAL ENGINE V2
# ─────────────────────────────────────────────────────────────────────────────

class RetrievalEngineV2:
    """
    Three retrieval methods for direct comparison:
      A. Cosine similarity (v1 baseline)
      B. FAISS flat index  (same math, indexed)
      C. Hybrid score      (cosine + entity Jaccard + event type match)
    """

    def __init__(self, train_path: Path, test_path: Path,
                 recency_lambda: float = 0.05):
        self.lam = recency_lambda

        log.info("=" * 60)
        log.info("RetrievalEngineV2 — initialising")
        log.info("=" * 60)

        self.train_corpus = load_corpus(train_path, "TRAIN 2001-2023")
        self.test_corpus  = load_corpus(test_path,  "TEST  2024-2026")

        # ── Build indices ──────────────────────────────────────────────────────
        log.info("[index] building cosine + FAISS indices ...")
        train_embs = np.stack([r.embedding for r in self.train_corpus])

        # Cosine: L2-normalised matrix for dot-product similarity
        self._train_norm = normalize(train_embs, norm="l2").astype(np.float32)

        # FAISS: inner product on L2-normalised vectors = cosine similarity
        self._faiss_index = faiss.IndexFlatIP(EMB_DIM)
        self._faiss_index.add(self._train_norm)
        log.info(f"[index] FAISS index: {self._faiss_index.ntotal} vectors")

        # Lookups
        self._test_lookup  = {r.event_id: i for i, r in enumerate(self.test_corpus)}
        self._train_lookup = {r.event_id: i for i, r in enumerate(self.train_corpus)}

        log.info("RetrievalEngineV2 ready.\n")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _recency_weight(self, adate: pd.Timestamp, bdate: pd.Timestamp) -> float:
        age = max((bdate - adate).days / 365.25, 0.0)
        return float(np.exp(-self.lam * age))

    def _extract_paths(self, rec: EventRecord):
        p0 = rec.price_nodes.get(0, 1.0) or 1.0
        s0 = rec.spread_nodes.get(0, 0.0)
        pp = {d: _norm_return(rec.price_nodes.get(d, p0), p0) for d in PRICE_NODES}
        sp = {d: float(rec.spread_nodes.get(d, s0) - s0)      for d in PRICE_NODES}
        return pp, sp

    def _find_event(self, eid: str) -> EventRecord:
        if eid in self._test_lookup:
            return self.test_corpus[self._test_lookup[eid]]
        if eid in self._train_lookup:
            return self.train_corpus[self._train_lookup[eid]]
        for r in self.test_corpus:
            if eid.lower() in r.event_id.lower():
                return r
        raise KeyError(f"'{eid}' not found. "
                       f"Sample IDs: {[r.event_id for r in self.test_corpus[:5]]}")

    def _build_analogue(self, i: int, method: str, score: float,
                        cosine_sim: float, entity_jac: float,
                        type_match: int, realised_arr: np.ndarray,
                        base_date: pd.Timestamp) -> AnalogueResult:
        rec = self.train_corpus[i]
        pp, sp = self._extract_paths(rec)
        arr = _path_array(pp)
        dtw = dtw_distance(arr, realised_arr)
        return AnalogueResult(
            event_id       = rec.event_id,
            date           = str(rec.date.date()),
            method         = method,
            score          = round(score,        6),
            cosine_sim     = round(cosine_sim,   6),
            entity_jaccard = round(entity_jac,   6),
            type_match     = type_match,
            dtw            = round(dtw,          4),
            persistence    = _persistence(pp),
            quantile_cat   = rec.quantile_cat,
            event_type     = rec.event_type,
            r7             = round(pp.get(7,  0.0), 4),
            r15            = round(pp.get(15, 0.0), 4),
            r30            = round(pp.get(30, 0.0), 4),
        )

    # ── Core comparison ───────────────────────────────────────────────────────

    def compare(
        self,
        base_event_id: str,
        tau:           float = 0.25,
        top_k:         int   = 50,
        random_seed:   int   = 42,
    ) -> ComparisonOutput:

        base      = self._find_event(base_event_id)
        base_date = base.date
        base_pp, _ = self._extract_paths(base)
        realised_arr = _path_array(base_pp)

        log.info(f"{'─'*60}")
        log.info(f"COMPARE: {base_event_id}  ({base_date.date()})")
        log.info(f"  quantile={base.quantile_cat}  type={base.event_type}")
        log.info(f"  tau={tau}  top_k={top_k}")

        base_norm = normalize(
            base.embedding.reshape(1, -1), norm="l2"
        ).astype(np.float32)

        # Pre-compute cosine similarities for all train events
        all_cosine = (self._train_norm @ base_norm.T).ravel()

        # Temporal mask — only events before base_date
        temporal_mask = np.array([
            rec.date < base_date for rec in self.train_corpus
        ], dtype=bool)

        # ── METHOD A: Cosine similarity ───────────────────────────────────────
        t0 = time.perf_counter()
        cosine_candidates = [
            (i, float(all_cosine[i]))
            for i in range(len(self.train_corpus))
            if temporal_mask[i] and float(all_cosine[i]) >= tau
        ]
        cosine_candidates.sort(key=lambda x: -x[1])
        cosine_top = cosine_candidates[:top_k]
        cosine_time = time.perf_counter() - t0

        cosine_analogues = [
            self._build_analogue(
                i, "cosine", sim, sim,
                _entity_jaccard(base.entity_tokens,
                                self.train_corpus[i].entity_tokens),
                int(self.train_corpus[i].event_type == base.event_type),
                realised_arr, base_date
            )
            for i, sim in cosine_top
        ]

        log.info(f"\n  METHOD A — Cosine similarity:")
        log.info(f"    candidates (tau={tau}): {len(cosine_candidates)}")
        log.info(f"    top-{top_k} returned  : {len(cosine_analogues)}")
        log.info(f"    time                  : {cosine_time*1000:.1f} ms")
        if cosine_analogues:
            log.info(f"    sim range: [{min(a.cosine_sim for a in cosine_analogues):.3f}, "
                     f"{max(a.cosine_sim for a in cosine_analogues):.3f}]")

        # ── METHOD B: FAISS flat index ─────────────────────────────────────────
        t0 = time.perf_counter()
        # Query more than top_k to account for temporal filtering
        n_query = min(len(self.train_corpus), top_k * 10)
        scores_faiss, indices_faiss = self._faiss_index.search(base_norm, n_query)
        scores_faiss = scores_faiss.ravel()
        indices_faiss = indices_faiss.ravel()

        faiss_candidates = [
            (int(idx), float(sc))
            for idx, sc in zip(indices_faiss, scores_faiss)
            if idx >= 0
            and temporal_mask[int(idx)]
            and float(sc) >= tau
        ]
        faiss_candidates.sort(key=lambda x: -x[1])
        faiss_top = faiss_candidates[:top_k]
        faiss_time = time.perf_counter() - t0

        faiss_analogues = [
            self._build_analogue(
                i, "faiss", sc, sc,
                _entity_jaccard(base.entity_tokens,
                                self.train_corpus[i].entity_tokens),
                int(self.train_corpus[i].event_type == base.event_type),
                realised_arr, base_date
            )
            for i, sc in faiss_top
        ]

        log.info(f"\n  METHOD B — FAISS flat index:")
        log.info(f"    candidates (tau={tau}): {len(faiss_candidates)}")
        log.info(f"    top-{top_k} returned  : {len(faiss_analogues)}")
        log.info(f"    time                  : {faiss_time*1000:.1f} ms")

        # ── METHOD C: Hybrid score ─────────────────────────────────────────────
        t0 = time.perf_counter()
        hybrid_candidates = []
        for i in range(len(self.train_corpus)):
            if not temporal_mask[i]:
                continue
            cosine = float(all_cosine[i])
            if cosine < tau:
                continue
            rec         = self.train_corpus[i]
            entity_jac  = _entity_jaccard(base.entity_tokens, rec.entity_tokens)
            type_match  = int(rec.event_type == base.event_type)
            hybrid      = 0.60 * cosine + 0.25 * entity_jac + 0.15 * type_match
            hybrid_candidates.append((i, hybrid, cosine, entity_jac, type_match))

        hybrid_candidates.sort(key=lambda x: -x[1])
        hybrid_top = hybrid_candidates[:top_k]
        hybrid_time = time.perf_counter() - t0

        hybrid_analogues = [
            self._build_analogue(
                i, "hybrid", hybrid, cosine, entity_jac, type_match,
                realised_arr, base_date
            )
            for i, hybrid, cosine, entity_jac, type_match in hybrid_top
        ]

        log.info(f"\n  METHOD C — Hybrid (cosine + entity + type):")
        log.info(f"    candidates (tau={tau}): {len(hybrid_candidates)}")
        log.info(f"    top-{top_k} returned  : {len(hybrid_analogues)}")
        log.info(f"    time                  : {hybrid_time*1000:.1f} ms")
        if hybrid_analogues:
            log.info(f"    score range: [{min(a.score for a in hybrid_analogues):.3f}, "
                     f"{max(a.score for a in hybrid_analogues):.3f}]")

        # ── Random baseline ───────────────────────────────────────────────────
        rng = np.random.default_rng(random_seed)
        rand_arrs = []
        for ri in rng.choice(len(self.train_corpus),
                              size=min(50, len(self.train_corpus)),
                              replace=False):
            rpp, _ = self._extract_paths(self.train_corpus[ri])
            rand_arrs.append(_path_array(rpp))
        rand_paths = np.stack(rand_arrs)
        random_dtw = mean_dtw(rand_paths, realised_arr)

        # ── DTW per method ────────────────────────────────────────────────────
        def _dtw_mean(analogues):
            if not analogues: return float("nan")
            paths = np.stack([
                _path_array(self._extract_paths(
                    self.train_corpus[self._train_lookup[a.event_id]]
                )[0])
                for a in analogues
                if a.event_id in self._train_lookup
            ])
            return mean_dtw(paths, realised_arr) if len(paths) else float("nan")

        cosine_dtw = _dtw_mean(cosine_analogues)
        faiss_dtw  = _dtw_mean(faiss_analogues)
        hybrid_dtw = _dtw_mean(hybrid_analogues)

        # ── Overlap between methods ───────────────────────────────────────────
        cosine_ids = set(a.event_id for a in cosine_analogues)
        faiss_ids  = set(a.event_id for a in faiss_analogues)
        hybrid_ids = set(a.event_id for a in hybrid_analogues)

        cf_jac  = _set_jaccard(cosine_ids, faiss_ids)
        ch_jac  = _set_jaccard(cosine_ids, hybrid_ids)
        fh_jac  = _set_jaccard(faiss_ids,  hybrid_ids)

        # ── Print comparison table ────────────────────────────────────────────
        log.info(f"\n{'─'*60}")
        log.info(f"COMPARISON SUMMARY")
        log.info(f"{'─'*60}")
        log.info(f"  {'Method':<20} {'N':>5} {'DTW':>8} {'vs Random':>10} {'Time(ms)':>10}")
        log.info(f"  {'─'*55}")
        for name, n, dtw, t in [
            ("A. Cosine",  len(cosine_analogues), cosine_dtw, cosine_time*1000),
            ("B. FAISS",   len(faiss_analogues),  faiss_dtw,  faiss_time*1000),
            ("C. Hybrid",  len(hybrid_analogues), hybrid_dtw, hybrid_time*1000),
            ("Random",     50,                    random_dtw, 0),
        ]:
            improv = f"{(random_dtw-dtw)/random_dtw*100:+.1f}%" if np.isfinite(dtw) and random_dtw else "n/a"
            log.info(f"  {name:<20} {n:>5} {dtw:>8.2f} {improv:>10} {t:>10.1f}")

        log.info(f"\n  Overlap (Jaccard of top-{top_k} sets):")
        log.info(f"    Cosine ∩ FAISS  : {cf_jac:.3f}  "
                 f"({'identical' if cf_jac==1.0 else 'different'})")
        log.info(f"    Cosine ∩ Hybrid : {ch_jac:.3f}  "
                 f"(hybrid changed {int((1-ch_jac)*top_k)} events)")
        log.info(f"    FAISS  ∩ Hybrid : {fh_jac:.3f}")

        # ── Top 5 per method ──────────────────────────────────────────────────
        log.info(f"\n  Top 5 analogues per method:")
        for name, analogues in [
            ("Cosine", cosine_analogues),
            ("FAISS",  faiss_analogues),
            ("Hybrid", hybrid_analogues),
        ]:
            log.info(f"\n  [{name}]")
            for a in analogues[:5]:
                log.info(f"    {a.date}  score={a.score:.3f}  "
                         f"cos={a.cosine_sim:.3f}  "
                         f"jac={a.entity_jaccard:.2f}  "
                         f"r30={a.r30:+.1f}%  "
                         f"{a.event_type[:25]}")

        return ComparisonOutput(
            base_event_id         = base_event_id,
            base_date             = str(base_date.date()),
            tau                   = tau,
            cosine_analogues      = cosine_analogues,
            faiss_analogues       = faiss_analogues,
            hybrid_analogues      = hybrid_analogues,
            cosine_dtw_mean       = round(cosine_dtw, 4) if np.isfinite(cosine_dtw) else -1,
            faiss_dtw_mean        = round(faiss_dtw,  4) if np.isfinite(faiss_dtw)  else -1,
            hybrid_dtw_mean       = round(hybrid_dtw, 4) if np.isfinite(hybrid_dtw) else -1,
            random_dtw_mean       = round(random_dtw, 4),
            cosine_time           = round(cosine_time, 4),
            faiss_time            = round(faiss_time,  4),
            hybrid_time           = round(hybrid_time, 4),
            cosine_faiss_jaccard  = round(cf_jac, 4),
            cosine_hybrid_jaccard = round(ch_jac, 4),
            faiss_hybrid_jaccard  = round(fh_jac, 4),
            realised_r30          = round(base_pp.get(30, 0.0), 4),
        )

    # ── Batch comparison ──────────────────────────────────────────────────────

    def batch_compare(
        self,
        tau:       float = 0.25,
        top_k:     int   = 50,
        output_dir: Optional[Path] = None,
    ) -> pd.DataFrame:
        """Run compare() over all test events, return summary DataFrame."""
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"\n{'='*60}")
        log.info(f"BATCH COMPARE: {len(self.test_corpus)} test events")
        log.info(f"{'='*60}")

        rows = []
        for rec in tqdm(self.test_corpus, desc="batch compare", unit=" ev"):
            try:
                result = self.compare(rec.event_id, tau=tau, top_k=top_k)
            except Exception as e:
                log.warning(f"SKIP {rec.event_id}: {e}")
                continue

            rows.append({
                "event_id":              rec.event_id,
                "date":                  str(rec.date.date()),
                "year":                  int(rec.year),
                "quantile_cat":          rec.quantile_cat,
                "event_type":            rec.event_type,
                "realised_r30":          result.realised_r30,
                # Analogue counts
                "n_cosine":              len(result.cosine_analogues),
                "n_faiss":               len(result.faiss_analogues),
                "n_hybrid":              len(result.hybrid_analogues),
                # DTW
                "cosine_dtw":            result.cosine_dtw_mean,
                "faiss_dtw":             result.faiss_dtw_mean,
                "hybrid_dtw":            result.hybrid_dtw_mean,
                "random_dtw":            result.random_dtw_mean,
                # Speed
                "cosine_ms":             round(result.cosine_time*1000, 2),
                "faiss_ms":              round(result.faiss_time*1000,  2),
                "hybrid_ms":             round(result.hybrid_time*1000, 2),
                # Overlap
                "cosine_faiss_jaccard":  result.cosine_faiss_jaccard,
                "cosine_hybrid_jaccard": result.cosine_hybrid_jaccard,
                # Which method wins on DTW?
                "dtw_winner":            min(
                    [("cosine", result.cosine_dtw_mean),
                     ("faiss",  result.faiss_dtw_mean),
                     ("hybrid", result.hybrid_dtw_mean)],
                    key=lambda x: x[1] if x[1] > 0 else float("inf")
                )[0],
            })

            if output_dir:
                out = output_dir / f"{rec.event_id}_comparison.json"
                with out.open("w", encoding="utf-8") as f:
                    json.dump({
                        "base_event_id":   result.base_event_id,
                        "base_date":       result.base_date,
                        "cosine_top5":     [asdict(a) for a in result.cosine_analogues[:5]],
                        "faiss_top5":      [asdict(a) for a in result.faiss_analogues[:5]],
                        "hybrid_top5":     [asdict(a) for a in result.hybrid_analogues[:5]],
                        "dtw":             {
                            "cosine": result.cosine_dtw_mean,
                            "faiss":  result.faiss_dtw_mean,
                            "hybrid": result.hybrid_dtw_mean,
                            "random": result.random_dtw_mean,
                        },
                        "overlap": {
                            "cosine_faiss":  result.cosine_faiss_jaccard,
                            "cosine_hybrid": result.cosine_hybrid_jaccard,
                        },
                    }, f, indent=2, default=str)

        df = pd.DataFrame(rows)
        if len(df):
            log.info(f"\n{'='*60}")
            log.info(f"BATCH RESULTS ({len(df)} events)")
            log.info(f"{'─'*60}")
            log.info(f"  Mean DTW:")
            log.info(f"    Cosine : {df['cosine_dtw'].mean():.4f}")
            log.info(f"    FAISS  : {df['faiss_dtw'].mean():.4f}")
            log.info(f"    Hybrid : {df['hybrid_dtw'].mean():.4f}")
            log.info(f"    Random : {df['random_dtw'].mean():.4f}")
            log.info(f"  Mean speed (ms):")
            log.info(f"    Cosine : {df['cosine_ms'].mean():.1f}")
            log.info(f"    FAISS  : {df['faiss_ms'].mean():.1f}")
            log.info(f"    Hybrid : {df['hybrid_ms'].mean():.1f}")
            log.info(f"  DTW winner counts:")
            log.info(f"{df['dtw_winner'].value_counts().to_string()}")
            log.info(f"  Mean Cosine∩FAISS overlap  : "
                     f"{df['cosine_faiss_jaccard'].mean():.3f}")
            log.info(f"  Mean Cosine∩Hybrid overlap : "
                     f"{df['cosine_hybrid_jaccard'].mean():.3f}")
            log.info(f"{'='*60}")
        return df


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train",  required=True)
    ap.add_argument("--test",   required=True)
    ap.add_argument("--event",  default=None,
                    help="single event_id to compare (omit for full batch)")
    ap.add_argument("--tau",    type=float, default=0.25)
    ap.add_argument("--top_k",  type=int,   default=50)
    ap.add_argument("--output", default="retrieval_v2_output")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    engine = RetrievalEngineV2(
        train_path = Path(args.train),
        test_path  = Path(args.test),
    )

    if args.event:
        # Single event comparison
        result = engine.compare(
            base_event_id = args.event,
            tau           = args.tau,
            top_k         = args.top_k,
        )
        out_file = out_dir / f"{args.event}_comparison.json"
        with out_file.open("w", encoding="utf-8") as f:
            json.dump({
                "cosine_top10": [asdict(a) for a in result.cosine_analogues[:10]],
                "faiss_top10":  [asdict(a) for a in result.faiss_analogues[:10]],
                "hybrid_top10": [asdict(a) for a in result.hybrid_analogues[:10]],
                "dtw": {
                    "cosine": result.cosine_dtw_mean,
                    "faiss":  result.faiss_dtw_mean,
                    "hybrid": result.hybrid_dtw_mean,
                    "random": result.random_dtw_mean,
                },
                "speed_ms": {
                    "cosine": round(result.cosine_time*1000, 2),
                    "faiss":  round(result.faiss_time*1000,  2),
                    "hybrid": round(result.hybrid_time*1000, 2),
                },
                "overlap": {
                    "cosine_faiss_jaccard":  result.cosine_faiss_jaccard,
                    "cosine_hybrid_jaccard": result.cosine_hybrid_jaccard,
                },
            }, f, indent=2, default=str)
        log.info(f"\nSaved -> {out_file}")
    else:
        # Full batch comparison
        df = engine.batch_compare(
            tau        = args.tau,
            top_k      = args.top_k,
            output_dir = out_dir / "per_event",
        )
        csv_path = out_dir / f"comparison_tau{args.tau}.csv"
        df.to_csv(csv_path, index=False)
        log.info(f"Saved -> {csv_path}")


if __name__ == "__main__":
    main()