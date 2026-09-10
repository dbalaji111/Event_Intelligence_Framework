"""
redflag_api.py
==============
Red Flag Detection API — FastAPI backend

Wires together:
  retrieval_engine.py  → semantic similarity search
  kde_engine_v3.py     → KDE posterior + red flag score
  llm_event_assessor.py → Claude classification + severity

Usage
-----
    pip install fastapi uvicorn anthropic sentence-transformers
    uvicorn redflag_api:app --reload --port 8000

Then POST to http://localhost:8000/analyze
"""

from __future__ import annotations
import json, logging, os, time, uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd

log = logging.getLogger("redflag")

# ─────────────────────────────────────────────────────────────────────────────
# PYDANTIC MODELS
# ─────────────────────────────────────────────────────────────────────────────

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
except ImportError:
    raise ImportError("pip install fastapi uvicorn pydantic")

class NewsInput(BaseModel):
    headline:  str   = Field(..., description="News headline text")
    timestamp: str   = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source:    str   = Field(default="Unknown")
    asset:     str   = Field(default="crude_oil",
                             description="crude_oil | gold | copper | macro")

class SimilarEvent(BaseModel):
    event_id:   str
    date:       str
    similarity: float
    event_type: str
    summary:    str
    r7:         Optional[float]
    r15:        Optional[float]
    r30:        Optional[float]

class OutcomeStats(BaseModel):
    avg_return_1d:      Optional[float]
    avg_return_5d:      Optional[float]
    avg_return_15d:     Optional[float]
    probability_up:     Optional[float]
    probability_down10: Optional[float]
    avg_volatility:     Optional[float]
    n_analogues:        int

class RedFlagResponse(BaseModel):
    request_id:          str
    headline:            str
    timestamp:           str
    source:              str
    event_type:          str
    severity:            int
    severity_label:      str
    red_flag_score:      int
    red_flag_label:      str
    recommendation:      str
    similar_events:      List[SimilarEvent]
    similar_events_found: int
    outcome_stats:       OutcomeStats
    kde_p5:              Optional[float]
    kde_median:          Optional[float]
    kde_p95:             Optional[float]
    is_bimodal:          bool
    tail_weight:         float
    processing_time_ms:  int


# ─────────────────────────────────────────────────────────────────────────────
# CORE SCORER
# ─────────────────────────────────────────────────────────────────────────────

class RedFlagScorer:
    """
    Wires together retrieval + KDE + LLM into one scoring pipeline.
    Can run without LLM (faster) or with LLM (richer classification).
    """

    # Event type → severity prior
    SEVERITY_PRIORS = {
        "supply shock":            3,
        "geopolitical risk":       3,
        "infrastructure disruption": 3,
        "monetary policy":         2,
        "demand shock":            2,
        "weather event":           2,
        "regulatory event":        2,
        "economic data release":   1,
        "corporate event":         1,
        "other":                   1,
    }

    # Keywords → event type (fast classification without LLM)
    KEYWORD_RULES = {
        "supply shock":   ["production cut","barrel","opec","output cut","supply",
                           "shut","pipeline","tanker","refinery","sanction"],
        "geopolitical risk": ["war","military","strike","attack","conflict","tension",
                               "iran","russia","ukraine","israel","hamas","strait",
                               "hormuz","blockade","invasion"],
        "demand shock":   ["demand","recession","gdp","china","slowdown","pmi",
                           "manufacturing","consumption","inventory build"],
        "monetary policy": ["fed","rate","interest","inflation","powell","ecb",
                            "hike","cut rate","quantitative"],
        "weather event":  ["hurricane","storm","flood","freeze","cold snap",
                           "tropical","wildfire","drought"],
        "infrastructure disruption": ["pipeline","terminal","port","outage",
                                       "explosion","fire","leak","spill"],
        "regulatory event": ["ban","regulation","law","court","ruling","penalty",
                              "fine","license"],
    }

    def __init__(self,
                 corpus_path:  Optional[str] = None,
                 prices_path:  Optional[str] = None,
                 ovx_path:     Optional[str] = None,
                 use_llm:      bool = True):

        self.use_llm      = use_llm
        self.corpus_path  = corpus_path
        self.prices_path  = prices_path
        self.ovx_path     = ovx_path
        self._retriever   = None
        self._prices      = None
        self._embedder    = None

        log.info("[scorer] RedFlagScorer initialised")
        log.info(f"[scorer]   corpus  : {corpus_path}")
        log.info(f"[scorer]   prices  : {prices_path}")
        log.info(f"[scorer]   LLM     : {use_llm}")

    def _load_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("bert-base-uncased")
            log.info("[scorer] BERT embedder loaded")

    def _load_corpus(self):
        if self._retriever is not None:
            return
        if not self.corpus_path:
            log.warning("[scorer] No corpus path — similarity search disabled")
            return
        try:
            import torch
            from sentence_transformers import SentenceTransformer

            # Load corpus
            p = Path(self.corpus_path)
            if p.suffix == ".json":
                with p.open("r", encoding="utf-8") as f:
                    corpus = json.load(f)
            else:
                with p.open("r", encoding="utf-8") as f:
                    corpus = [json.loads(l) for l in f if l.strip()]

            self._corpus = corpus
            log.info(f"[scorer] corpus loaded: {len(corpus)} events")

            # Load embeddings
            embs = []
            for ev in corpus:
                emb = ev.get("embedding", ev.get("bert_embedding", []))
                if emb:
                    embs.append(emb)
                else:
                    embs.append([0.0]*768)

            self._corpus_embs = np.array(embs, dtype=np.float32)
            # L2 normalise
            norms = np.linalg.norm(self._corpus_embs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            self._corpus_embs = self._corpus_embs / norms
            self._retriever = True
            log.info(f"[scorer] embedding matrix: {self._corpus_embs.shape}")

        except Exception as e:
            log.error(f"[scorer] corpus load failed: {e}")

    def _load_prices(self):
        if self._prices is not None:
            return
        if not self.prices_path:
            return
        try:
            df = pd.read_excel(self.prices_path)
            df = df.rename(columns={df.columns[0]: "Date"})
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
            rename = {
                "Crude Oil-WTI Spot Cushing U$/BBL": "WTI",
                "NYMEX Crude Oil WTI M1-M2 Spread":  "M1M2",
            }
            df = df.rename(columns={k:v for k,v in rename.items() if k in df.columns})
            df = df.ffill(limit=5)

            if self.ovx_path and Path(self.ovx_path).exists():
                ovx = pd.read_csv(self.ovx_path)
                ovx["Date"] = pd.to_datetime(ovx["Date"],
                    format="%m/%d/%Y", errors="coerce")
                ovx = ovx.dropna(subset=["Date"]).set_index("Date")
                df = df.join(ovx[["Price"]].rename(columns={"Price":"OVX"}),
                             how="left")

            self._prices = df
            log.info(f"[scorer] prices loaded: {len(df)} rows")
        except Exception as e:
            log.error(f"[scorer] prices load failed: {e}")

    def classify_event(self, headline: str) -> tuple[str, int]:
        """
        Fast keyword-based classification.
        Returns (event_type, severity).
        """
        hl_lower = headline.lower()

        matched_type = "other"
        for etype, keywords in self.KEYWORD_RULES.items():
            if any(kw in hl_lower for kw in keywords):
                matched_type = etype
                break

        severity = self.SEVERITY_PRIORS.get(matched_type, 1)

        # Boost severity for extreme keywords
        extreme_words = ["unprecedented","emergency","war","invasion","attack",
                         "major","significant","shock","crisis","collapse",
                         "explosion","sanctions","military"]
        if any(w in hl_lower for w in extreme_words):
            severity = min(4, severity + 1)

        return matched_type, severity

    def embed_headline(self, headline: str) -> np.ndarray:
        """Embed headline using BERT."""
        self._load_embedder()
        emb = self._embedder.encode([headline], normalize_embeddings=True)[0]
        return emb.astype(np.float32)

    def retrieve_analogues(self, emb: np.ndarray,
                            top_k: int = 20) -> List[Dict]:
        """
        Retrieve top-K analogues from corpus using cosine similarity.
        """
        self._load_corpus()
        if self._retriever is None:
            return []

        sims = self._corpus_embs @ emb
        top_indices = np.argsort(sims)[::-1][:top_k*3]

        analogues = []
        for idx in top_indices:
            if len(analogues) >= top_k:
                break
            ev  = self._corpus[idx]
            sim = float(sims[idx])
            if sim < 0.75:
                continue
            analogues.append({
                "event_id":   ev.get("event_id", f"EVT_{idx}"),
                "date":       str(ev.get("date", "")),
                "similarity": round(sim, 4),
                "event_type": ev.get("event_type", ""),
                "summary":    str(ev.get("primary_event",
                              ev.get("full_text",""))[:120]),
                "r7":         ev.get("r7_pct", ev.get("r7")),
                "r15":        ev.get("r15_pct", ev.get("r15")),
                "r30":        ev.get("r30_pct", ev.get("r30")),
            })
        return analogues

    def compute_outcome_stats(self, analogues: List[Dict]) -> Dict:
        """Compute outcome statistics from retrieved analogues."""
        if not analogues:
            return {
                "avg_return_1d": None, "avg_return_5d": None,
                "avg_return_15d": None, "probability_up": None,
                "probability_down10": None, "avg_volatility": None,
                "n_analogues": 0,
            }

        r15s = [float(a["r15"]) for a in analogues
                if a.get("r15") is not None]
        r7s  = [float(a["r7"])  for a in analogues
                if a.get("r7")  is not None]

        if not r15s:
            return {"n_analogues": len(analogues),
                    "avg_return_1d": None, "avg_return_5d": None,
                    "avg_return_15d": None, "probability_up": None,
                    "probability_down10": None, "avg_volatility": None}

        arr = np.array(r15s)
        return {
            "avg_return_1d":      round(float(np.mean(r7s)), 3) if r7s else None,
            "avg_return_5d":      round(float(np.mean(r7s)), 3) if r7s else None,
            "avg_return_15d":     round(float(np.mean(arr)), 3),
            "probability_up":     round(float((arr > 0).mean()), 3),
            "probability_down10": round(float((arr < -10).mean()), 3),
            "avg_volatility":     round(float(np.std(arr)), 3),
            "n_analogues":        len(analogues),
        }

    def compute_kde_posterior(self, analogues: List[Dict],
                               horizon: int = 15) -> Dict:
        """
        Build Gaussian mixture posterior from analogues.
        Lightweight version of kde_engine_v3.
        """
        from scipy.stats import norm as scipy_norm
        from scipy.signal import find_peaks

        rkey = {7:"r7", 15:"r15", 30:"r30"}.get(horizon, "r15")
        returns = [float(a[rkey]) for a in analogues
                   if a.get(rkey) is not None]
        sims    = [float(a["similarity"]) for a in analogues
                   if a.get(rkey) is not None]

        if len(returns) < 3:
            return {"kde_p5": None, "kde_median": None, "kde_p95": None,
                    "is_bimodal": False, "tail_weight": 0.0,
                    "p_down10": 0.0}

        mus     = np.array(returns)
        weights = np.array(sims) ** 0.5
        weights = weights / weights.sum()
        sigmas  = np.full(len(mus), max(np.std(mus) * 0.5, 2.0))

        # Build mixture on grid
        lo = mus.min() - 10
        hi = mus.max() + 10
        x  = np.linspace(lo, hi, 800)
        dx = x[1] - x[0]

        density = sum(w * scipy_norm.pdf(x, m, s)
                      for m, s, w in zip(mus, sigmas, weights))

        cdf = np.cumsum(density) * dx
        cdf = np.clip(cdf / max(cdf[-1], 1e-10), 0, 1)

        def q(p):
            idx = np.searchsorted(cdf, p)
            return float(x[min(idx, len(x)-1)])

        # Bimodal detection
        peaks, _ = find_peaks(density, prominence=density.max()*0.1)
        is_bimodal = len(peaks) >= 2

        # Tail weight
        tail_mask  = np.abs(mus) > 10
        tail_weight = float(weights[tail_mask].sum()) if tail_mask.any() else 0.0

        # P(down 10%)
        mask_d10 = x <= -10
        p_down10 = float(np.trapz(density[mask_d10], x[mask_d10])) \
                   if mask_d10.any() else 0.0

        return {
            "kde_p5":       round(q(0.05), 2),
            "kde_median":   round(q(0.50), 2),
            "kde_p95":      round(q(0.95), 2),
            "is_bimodal":   is_bimodal,
            "tail_weight":  round(tail_weight, 3),
            "p_down10":     round(max(0, min(1, p_down10)), 4),
        }

    def compute_red_flag_score(self,
                                severity:    int,
                                sims:        List[float],
                                outcome:     Dict,
                                kde:         Dict) -> int:
        """
        Red Flag Score = 0-100
        Inputs: severity, similarity confidence, historical consistency,
                KDE tail probability
        """
        # Component 1: Severity (0-30 points)
        sev_score = {1: 5, 2: 12, 3: 22, 4: 30}.get(severity, 5)

        # Component 2: Similarity confidence (0-20 points)
        if sims:
            mean_sim = np.mean(sims[:10])
            sim_score = int(20 * min((mean_sim - 0.75) / 0.20, 1.0))
        else:
            sim_score = 0

        # Component 3: Historical directional consistency (0-25 points)
        prob_down = outcome.get("probability_down10", 0) or 0
        prob_up   = outcome.get("probability_up", 0.5) or 0.5
        # Consistency = how strongly history agrees on direction
        consistency = abs(prob_up - 0.5) * 2  # 0 = random, 1 = unanimous
        hist_score  = int(25 * consistency)

        # Component 4: KDE tail probability (0-25 points)
        p_down10    = kde.get("p_down10", 0) or 0
        tail_weight = kde.get("tail_weight", 0) or 0
        kde_score   = int(15 * min(p_down10 / 0.30, 1.0)) + \
                      int(10 * min(tail_weight / 0.50, 1.0))

        total = sev_score + sim_score + hist_score + kde_score
        return min(100, max(0, total))

    def score_headline(self, headline: str, source: str = "Unknown",
                       timestamp: str = "") -> Dict:
        """
        Full pipeline: classify → retrieve → KDE → score
        """
        t0 = time.time()
        request_id = str(uuid.uuid4())[:8]

        # Step 1: classify
        event_type, severity = self.classify_event(headline)

        # Step 2: embed
        try:
            emb = self.embed_headline(headline)
            analogues = self.retrieve_analogues(emb, top_k=25)
        except Exception as e:
            log.warning(f"[scorer] embedding failed: {e}")
            analogues = []

        # Step 3: outcome stats
        outcome = self.compute_outcome_stats(analogues)

        # Step 4: KDE posterior
        kde = self.compute_kde_posterior(analogues)

        # Step 5: red flag score
        sims      = [a["similarity"] for a in analogues]
        rf_score  = self.compute_red_flag_score(severity, sims,
                                                 outcome, kde)

        # Labels
        sev_labels  = {1:"Low", 2:"Moderate", 3:"High", 4:"Extreme"}
        rf_labels   = {
            (0,  39): ("Ignore",        "No action required"),
            (40, 59): ("Monitor",       "Track closely — no immediate action"),
            (60, 79): ("High Attention","Investigate within 30 minutes"),
            (80,100): ("Critical",      "Investigate Immediately"),
        }
        rf_label = rec = "Monitor"
        for (lo, hi), (lbl, r) in rf_labels.items():
            if lo <= rf_score <= hi:
                rf_label, rec = lbl, r
                break

        ms = int((time.time() - t0) * 1000)

        return {
            "request_id":          request_id,
            "headline":            headline,
            "timestamp":           timestamp or datetime.now(timezone.utc).isoformat(),
            "source":              source,
            "event_type":          event_type.title(),
            "severity":            severity,
            "severity_label":      sev_labels[severity],
            "red_flag_score":      rf_score,
            "red_flag_label":      rf_label,
            "recommendation":      rec,
            "similar_events":      analogues[:10],
            "similar_events_found": len(analogues),
            "outcome_stats":       outcome,
            "kde_p5":              kde.get("kde_p5"),
            "kde_median":          kde.get("kde_median"),
            "kde_p95":             kde.get("kde_p95"),
            "is_bimodal":          kde.get("is_bimodal", False),
            "tail_weight":         kde.get("tail_weight", 0.0),
            "processing_time_ms":  ms,
        }


# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Brahmanda Red Flag Detection API",
    description = "Event-conditioned real-time risk scoring for commodity markets",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# Global scorer (lazy-loaded)
_scorer: Optional[RedFlagScorer] = None

def get_scorer() -> RedFlagScorer:
    global _scorer
    if _scorer is None:
        corpus  = os.environ.get("CORPUS_PATH")
        prices  = os.environ.get("PRICES_PATH")
        ovx     = os.environ.get("OVX_PATH")
        use_llm = os.environ.get("USE_LLM", "true").lower() == "true"
        _scorer = RedFlagScorer(corpus, prices, ovx, use_llm)
    return _scorer


@app.get("/")
def root():
    return {
        "name":    "Brahmanda Red Flag Detection API",
        "version": "1.0.0",
        "status":  "operational",
        "endpoints": ["/analyze", "/health", "/docs"],
    }

@app.get("/health")
def health():
    scorer = get_scorer()
    return {
        "status":       "healthy",
        "corpus_loaded": scorer._retriever is not None,
        "prices_loaded": scorer._prices is not None,
        "timestamp":    datetime.now(timezone.utc).isoformat(),
    }

@app.post("/analyze", response_model=RedFlagResponse)
def analyze(news: NewsInput):
    """
    Analyze a news headline and return a Red Flag score.

    Example:
        POST /analyze
        {
          "headline": "Saudi Arabia unexpectedly cuts oil production by 500,000 bpd",
          "source": "Reuters"
        }
    """
    scorer = get_scorer()
    try:
        result = scorer.score_headline(
            headline  = news.headline,
            source    = news.source,
            timestamp = news.timestamp,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch")
def batch_analyze(headlines: List[NewsInput]):
    """Analyze multiple headlines at once (max 50)."""
    if len(headlines) > 50:
        raise HTTPException(status_code=400,
                            detail="Max 50 headlines per batch")
    scorer  = get_scorer()
    results = [scorer.score_headline(h.headline, h.source, h.timestamp)
               for h in headlines]
    return {
        "count":   len(results),
        "results": results,
        "critical": [r for r in results if r["red_flag_score"] >= 80],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("redflag_api:app", host="0.0.0.0", port=8000,
                reload=True, log_level="info")


# ── V4 ENDPOINTS ─────────────────────────────────────────────────────────────
import glob

@app.get("/events")
def get_events(q: str = "", type: str = "", limit: int = 100):
    """
    Load events from memory_2024_2026/df_major_metadata_2024_2026.csv
    Supports search (q) and type filter.
    """
    import pandas as pd
    from pathlib import Path

    # Try to find metadata file
    base = Path(os.environ.get("BASE_PATH", "."))
    meta_paths = [
        Path("memory_2024_2026") / "df_major_metadata_2024_2026.csv",
        Path(".") / "memory_2024_2026" / "df_major_metadata_2024_2026.csv",
    ]
    meta_file = next((p for p in meta_paths if p.exists()), None)

    if meta_file is None:
        raise HTTPException(status_code=404,
            detail="metadata CSV not found — set BASE_PATH env var")

    df = pd.read_csv(meta_file)
    df["date"] = pd.to_datetime(df.get("date", pd.Series()), errors="coerce")
    df = df.sort_values("date", ascending=False)

    # Filters
    if q:
        mask = df.get("text", df.iloc[:,1]).str.contains(q, case=False, na=False)
        df = df[mask]
    if type and type != "All":
        df = df[df.get("Event_type", pd.Series()).astype(str) == type]

    df = df.head(limit)

    # Check which events have precomputed KDE
    kde_base = base / "kde_output_v3" / "per_event"
    events = []
    for _, row in df.iterrows():
        eid = str(row.get("event_id", ""))
        kde_path = kde_base / f"{eid}_kde_v3.json"
        events.append({
            "event_id":     eid,
            "text":         str(row.get("text", row.get("combined_text", "")))[:300],
            "headline":     str(row.get("text", row.get("combined_text", "")))[:200],
            "date":         row["date"].isoformat() if pd.notna(row.get("date")) else "",
            "source":       str(row.get("source", "Refinitiv")),
            "Event_type":   str(row.get("Event_type", "")),
            "event_type":   str(row.get("Event_type", "")),
            "kde_available": kde_path.exists(),
        })

    return {"events": events, "total": len(events)}


@app.get("/kde/{event_id}")
def get_kde(event_id: str):
    """Return precomputed KDE posterior for a specific event."""
    import json
    from pathlib import Path

    base = Path(os.environ.get("BASE_PATH", "."))
    paths = [
        base / "kde_output_v3" / "per_event" / f"{event_id}_kde_v3.json",
        Path("kde_output_v3") / "per_event" / f"{event_id}_kde_v3.json",
    ]
    kde_file = next((p for p in paths if p.exists()), None)

    if kde_file is None:
        raise HTTPException(status_code=404,
            detail=f"No KDE file found for {event_id}")

    with kde_file.open() as f:
        data = json.load(f)

    # Return stats + truncated grid (200 points for speed)
    stats = data.get("posterior_stats", {})
    x_grid = data.get("x_grid", [])
    density = data.get("density", [])

    # Downsample to 200 points for fast transfer
    if len(x_grid) > 200:
        step = len(x_grid) // 200
        x_grid  = x_grid[::step]
        density = density[::step]

    return {
        "event_id":      event_id,
        "posterior_stats": stats,
        "garch":         data.get("garch", {}),
        "x_grid":        x_grid,
        "density":       density,
        "base_date":     data.get("base_date", ""),
    }


@app.get("/ovx")
def get_ovx(days: int = 90):
    """Return recent OVX data from the CBOE CSV file."""
    import pandas as pd
    from pathlib import Path

    base = Path(os.environ.get("BASE_PATH", "."))
    ovx_path = Path(os.environ.get("OVX_PATH", ""))
    if not ovx_path.exists():
        ovx_path = base / "crude_oil_data" / "oil_price_data" / \
                   "CBOE Crude Oil Volatility Historical Data.csv"

    if not ovx_path.exists():
        raise HTTPException(status_code=404, detail="OVX file not found")

    df = pd.read_csv(ovx_path)
    df.columns = [c.strip() for c in df.columns]
    date_col  = next((c for c in df.columns if "date" in c.lower()), df.columns[0])
    price_col = next((c for c in df.columns if "price" in c.lower()
                      or "close" in c.lower()), df.columns[1])
    df = df[[date_col, price_col]].rename(columns={date_col:"Date", price_col:"OVX"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["OVX"]  = pd.to_numeric(df["OVX"], errors="coerce")
    df = df.dropna().sort_values("Date").tail(days)

    return {
        "dates":  [d.strftime("%Y-%m-%d") for d in df["Date"]],
        "values": [round(float(v), 2) for v in df["OVX"]],
        "latest": round(float(df["OVX"].iloc[-1]), 2) if len(df) else None,
    }
from fastapi.responses import HTMLResponse
from pathlib import Path as _Path

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    html_file = _Path(__file__).parent / "redflag_dashboard_v4.html"
    if html_file.exists():
        return HTMLResponse(content=html_file.read_text())
    return HTMLResponse(content="<h1>Not found</h1>", status_code=404)

@app.post("/retrieve")

@app.post("/retrieve")
def retrieve_analogues(payload: dict):
    """Live retrieval using precomputed embeddings."""
    import numpy as np
    import pandas as pd
    from pathlib import Path

    event_id = payload.get("event_id", "")
    tau      = float(payload.get("tau", 0.85))
    k_min    = int(payload.get("k_min", 25))
    base     = Path(os.environ.get("BASE_PATH", "."))

    npz    = np.load(base / "memory_2024_2026" / "df_major_embeddings_2024_2026_MASTER_COMPAT.npz", allow_pickle=True)
    meta_q = pd.read_csv(base / "memory_2024_2026" / "df_major_metadata_2024_2026.csv")
    row_q  = meta_q[meta_q["event_id"] == event_id]
    if row_q.empty:
        raise HTTPException(status_code=404, detail=f"{event_id} not found")

    doc_id    = row_q.iloc[0]["document_id"]
    ids, embs = npz["ids"], npz["embeddings"].astype(float)
    pos       = np.where(ids == doc_id)[0]
    if len(pos) == 0:
        raise HTTPException(status_code=404, detail="Embedding not found")
    query_emb = embs[pos[0]]

    corp      = np.load(base / "memory_2001_2023" / "df_major_embeddings_RECOMPUTED.npz", allow_pickle=True)
    corp_embs = corp["embeddings"].astype(float)
    corp_ids  = corp["ids"]
    corp_meta = pd.read_csv(base / "memory_2001_2023" / "df_major_metadata.csv")

    q_norm  = query_emb / (np.linalg.norm(query_emb) + 1e-10)
    c_norms = corp_embs / (np.linalg.norm(corp_embs, axis=1, keepdims=True) + 1e-10)
    sims    = c_norms @ q_norm

    tau_used = 0.0
    for t in [tau, 0.82, 0.80, 0.75, 0.70, 0.65, 0.0]:
        if (sims >= t).sum() >= k_min:
            tau_used = t
            break

    idxs  = np.where(sims >= tau_used)[0]
    idxs  = idxs[np.argsort(-sims[idxs])[:50]]

    analogues = []
    for i in idxs:
        r = corp_meta.iloc[i] if i < len(corp_meta) else {}
        analogues.append({
            "analogue_id": str(corp_ids[i]) if i < len(corp_ids) else str(i),
            "similarity":  round(float(sims[i]), 4),
            "date":        str(r.get("date", "")),
            "text":        str(r.get("text", r.get("combined_text", "")))[:200],
            "event_type":  str(r.get("Event_type", "")),
            "r7":          float(r["price_change_t_minus_7_to_t_plus_7"]) if "price_change_t_minus_7_to_t_plus_7" in r and pd.notna(r.get("price_change_t_minus_7_to_t_plus_7")) else None,
            "r15":         float(r["price_change_t_minus_15_to_t_plus_15"]) if "price_change_t_minus_15_to_t_plus_15" in r and pd.notna(r.get("price_change_t_minus_15_to_t_plus_15")) else None,
        })

    return {"event_id": event_id, "tau_used": tau_used, "n_analogues": len(analogues), "analogues": analogues}
