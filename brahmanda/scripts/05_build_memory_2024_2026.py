"""
05_build_memory_2024_2026.py
============================
Build a Brahmanda-style price_event_memory.json for the 2024-2026 oil corpus.

Reads
-----
data/oil_data_2024_2026/_processed/_ALL__kept.json   (from 04b_factiva_batch.py)
data/price/wti_brent_daily_2024_2026.csv             (placeholder — see PRICE_FILE)

Writes
------
memory_2024_2026/price_event_memory_2024_2026.json   (Brahmanda-compatible)
memory_2024_2026/df_major_memory_2024_2026.jsonl     (one record per line)
memory_2024_2026/df_major_embeddings_2024_2026.npz   (raw 768-dim vectors)
memory_2024_2026/df_major_metadata_2024_2026.csv     (everything except bert_embeddings)

Field-by-field compatible with Brahmanda_v3/memory/price_event_memory.json
EXCEPT the Influence_Score family + Trigger_embeddings (skipped per spec).

Embeddings
----------
768-dim. Default model is bert-base-uncased + mean pooling. If your Brahmanda
originals were FinBERT or a different checkpoint, edit EMBEDDING_MODEL_NAME
below — that is the ONE thing to swap if vectors don't match.

Run
---
    python -m scripts.05_build_memory_2024_2026 --limit 5      # smoke test
    python -m scripts.05_build_memory_2024_2026                # full run
    python -m scripts.05_build_memory_2024_2026 --resume       # continue after crash
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Make the existing extractors importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR  = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# ---------------------------------------------------------------------------
# Ollama runner — same shape as 05a_smoke_test_glm.py so we can monkeypatch
# r02_all_llm_functions.run_ollama_prompt(model_name, prompt_text) cleanly.
# Edit LLM_MODEL to swap models without touching anything else.
# ---------------------------------------------------------------------------
import subprocess

LLM_MODEL = "qwen2.5:7b-instruct"   # ← swap to glm-4.7-flash:latest if you upgrade VRAM
LLM_CALL_TIMEOUT = 90               # seconds per Ollama call. Was 300; 90 fails fast on hangs.


def _ollama_runner(*args, **kwargs) -> str:
    """Subprocess call to `ollama run <LLM_MODEL>`. Tolerates both calling conventions.
    Returns 'unknown' on timeout/error so the calling extractor can move on."""
    if len(args) == 2:
        prompt = args[1]
    elif len(args) == 1:
        prompt = args[0]
    else:
        prompt = kwargs.get("prompt_text") or kwargs.get("prompt") or ""
    proc = None
    try:
        proc = subprocess.Popen(
            ["ollama", "run", LLM_MODEL],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
        )
        out, err = proc.communicate(input=prompt, timeout=LLM_CALL_TIMEOUT)
        if proc.returncode != 0:
            return "unknown"
        return out.strip()
    except subprocess.TimeoutExpired:
        if proc is not None:
            try: proc.kill()
            except Exception: pass
        return "unknown"
    except Exception:
        if proc is not None:
            try: proc.kill()
            except Exception: pass
        return "unknown"


# Monkeypatch BEFORE importing the extractor functions so they all route through it.
import r02_all_llm_functions as _llm_module  # noqa: E402
_llm_module.run_ollama_prompt = _ollama_runner

# Brahmanda-style attribute extractors (now routed through LLM_MODEL above)
# `extract_oil_summary` is a Chain-of-Thought summariser defined alongside the
# other extractors — keeps all prompts in one place.
from r02_all_llm_functions import (  # noqa: E402
    extract_category, extract_subcategory, extract_entity,
    extract_primary_event, extract_event_type,
    extract_commodity_attributes, extract_country_attributes,
    extract_duration_attributes, extract_quantity_attributes,
    extract_outcome, extract_financial_attributes,
    extract_forecast_attributes, extract_group_attributes,
    extract_location_attributes, extract_money_attributes,
    extract_production_unit_attributes, extract_state_or_province,
    extract_percent_attributes, extract_person_attributes,
    extract_price_unit_attributes, extract_event_trigger,
    extract_related_entity, extract_second_event, extract_third_event,
    extract_outcome_attribute, extract_temporal_attribute,
    extract_polarity, extract_modality, extract_causal_links,
    extract_influencing_factors,
    extract_oil_summary,
)

# ---------------------------------------------------------------------------
# CONFIG — paths
# ---------------------------------------------------------------------------
KEPT_JSON  = PROJECT_ROOT / "data" / "oil_data_2024_2026" / "_processed" / "_ALL__kept.json"
PRICE_FILE = PROJECT_ROOT / "data" / "oil_timeseries.csv"   # WTI_M1..M6 + BRENT_M1..M3

OUT_DIR    = PROJECT_ROOT / "memory_2024_2026"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_JSON   = OUT_DIR / "price_event_memory_2024_2026.json"
OUT_JSONL  = OUT_DIR / "df_major_memory_2024_2026.jsonl"
OUT_NPZ    = OUT_DIR / "df_major_embeddings_2024_2026.npz"
OUT_CSV    = OUT_DIR / "df_major_metadata_2024_2026.csv"

# ---------------------------------------------------------------------------
# CONFIG — embedding model (THE ONE thing to swap if Brahmanda used a different model)
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "bert-base-uncased"   # ← change to e.g. "ProsusAI/finbert"
EMBEDDING_DIM        = 768                    # Brahmanda original is 768
EMBEDDING_DEVICE     = "cuda"                 # "cuda" | "cpu" | "mps"
EMBEDDING_MAX_TOKENS = 512                    # BERT cap

# ---------------------------------------------------------------------------
# Embedder (lazy-loaded)
# ---------------------------------------------------------------------------
_EMBEDDER: Tuple[Any, Any, str, Any] | None = None


def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        import torch
        from transformers import AutoModel, AutoTokenizer
        dev = EMBEDDING_DEVICE
        if dev == "cuda" and not torch.cuda.is_available():
            dev = "cpu"
        if dev == "mps" and not getattr(torch.backends, "mps", None) or \
           (dev == "mps" and not torch.backends.mps.is_available()):
            dev = "cpu"
        tok = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
        mdl = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME).to(dev).eval()
        _EMBEDDER = (tok, mdl, dev, torch)
        print(f"[embedder] {EMBEDDING_MODEL_NAME} loaded on {dev}")
    return _EMBEDDER


def embed_text(text: str) -> np.ndarray:
    """Mean-pooled BERT embedding. Returns float32, shape (EMBEDDING_DIM,)."""
    if not text or not str(text).strip():
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)
    tok, mdl, dev, torch = _get_embedder()
    enc = tok(str(text), return_tensors="pt", truncation=True,
              max_length=EMBEDDING_MAX_TOKENS, padding=True).to(dev)
    with torch.no_grad():
        out = mdl(**enc)
    last = out.last_hidden_state                            # (1, T, H)
    mask = enc["attention_mask"].unsqueeze(-1).float()      # (1, T, 1)
    pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return pooled.squeeze(0).cpu().numpy().astype(np.float32)


def stringify_embedding(vec: np.ndarray) -> str:
    """Match Brahmanda's stored format: numpy default array repr."""
    return np.array2string(vec, threshold=np.inf)


# ---------------------------------------------------------------------------
# Light cleaning (Brahmanda's cleaned_text is LLM-rewritten — we approximate)
# ---------------------------------------------------------------------------
_FACTIVA_BOILER = re.compile(
    r"(--- Page \d+ ---|Page \d+ of \d+|©\s*\d{4}\s*Factiva|All rights reserved\.)",
    re.IGNORECASE,
)


def light_clean(text: str) -> str:
    if not text:
        return ""
    t = _FACTIVA_BOILER.sub(" ", text)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ---------------------------------------------------------------------------
# Price + technical-indicator + curve join
# ---------------------------------------------------------------------------
def load_price_panel() -> pd.DataFrame:
    """
    Expected columns (case-insensitive — we normalise):
        Date, WTI_M1, WTI_M2, WTI_M3, WTI_M6, BRENT_M1, BRENT_M2, BRENT_M3
        (any subset OK; missing curve months → 0.0)
        WTI_M1 is treated as the WTI spot proxy for all price/quantile/window math.
    """
    if not PRICE_FILE.exists():
        warnings.warn(f"Price file not found at {PRICE_FILE}; "
                      f"price/quantile/curve fields will be 0/NaN.")
        return pd.DataFrame()
    df = pd.read_csv(PRICE_FILE)
    df.columns = [c.strip() for c in df.columns]
    date_col = next((c for c in df.columns if c.lower() in ("date", "datetime")), df.columns[0])
    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()
    df = df.set_index(date_col).sort_index()
    return df


def technical_indicators(prices: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=prices.index)
    out["SMA_7"] = prices.rolling(7).mean()
    out["EMA_7"] = prices.ewm(span=7, adjust=False).mean()
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["RSI"] = 100 - (100 / (1 + rs))
    sma20 = prices.rolling(20).mean()
    sd20  = prices.rolling(20).std()
    out["Bollinger_Upper"] = sma20 + 2 * sd20
    out["Bollinger_Lower"] = sma20 - 2 * sd20
    return out


_QUANTILE_LEVELS = (5, 20, 30, 40, 60, 80, 90)
_QUANTILE_LABELS = ("0%-5%", "5%-20%", "20%-30%", "30%-40%",
                    "40%-60%", "60%-80%", "80%-90%", "90%-100%")


def quantile_bands(prices: pd.Series, year: int) -> Dict[str, float]:
    yr = prices[prices.index.year == year]
    if yr.empty:
        return {f"q{p}%": float("nan") for p in _QUANTILE_LEVELS}
    return {f"q{p}%": float(yr.quantile(p / 100)) for p in _QUANTILE_LEVELS}


def quantile_category(price: float, bands: Dict[str, float]) -> str:
    if pd.isna(price) or any(pd.isna(v) for v in bands.values()):
        return "unknown"
    edges = [bands[f"q{p}%"] for p in _QUANTILE_LEVELS]
    if price < edges[0]:
        return _QUANTILE_LABELS[0]
    for lo, hi, lbl in zip(edges[:-1], edges[1:], _QUANTILE_LABELS[1:-1]):
        if lo <= price < hi:
            return lbl
    return _QUANTILE_LABELS[-1]


def price_window(prices: pd.Series, dt: pd.Timestamp,
                 offsets=(7, 15, 30)) -> Dict[str, float]:
    res: Dict[str, float] = {}
    for off in offsets:
        for sign, key in ((-1, "minus"), (+1, "plus")):
            target = dt + pd.Timedelta(days=sign * off)
            try:
                res[f"price_t_{key}_{off}"] = float(prices.asof(target))
            except (KeyError, ValueError):
                res[f"price_t_{key}_{off}"] = float("nan")
    return res


# ---------------------------------------------------------------------------
# % change buckets (categorical labels for price_change_* fields)
#
# Window-specific thresholds — short-horizon moves that look "big" daily are
# noise on a 30-day basis, so each window has its own scale.
#
#   Window  Strong Decline  Decline       Flat        Rise         Strong Rise
#   daily   < -3%           -3 to -1%     -1 to +1%   +1 to +3%    > +3%
#   ±7d     < -5%           -5 to -2%     -2 to +2%   +2 to +5%    > +5%
#   ±15d    < -7%           -7 to -3%     -3 to +3%   +3 to +7%    > +7%
#   ±30d    < -10%          -10 to -5%    -5 to +5%   +5 to +10%   > +10%
# ---------------------------------------------------------------------------
_BUCKET_THRESHOLDS: Dict[str, Tuple[float, float, float, float]] = {
    "daily": (-3.0, -1.0, 1.0, 3.0),
    "7":     (-5.0, -2.0, 2.0, 5.0),
    "15":    (-7.0, -3.0, 3.0, 7.0),
    "30":    (-10.0, -5.0, 5.0, 10.0),
}
_BUCKET_LABELS = ("Strong Decline", "Decline", "Flat", "Rise", "Strong Rise")


def price_change_bucket(value: float, window: str) -> str:
    """Map a percent change into one of the 5 categorical buckets for the given window.
    `window` ∈ {"daily", "7", "15", "30"}."""
    if value is None or pd.isna(value):
        return "unknown"
    thr = _BUCKET_THRESHOLDS.get(window)
    if thr is None:
        return "unknown"
    sd, d, r, sr = thr
    if value < sd:
        return _BUCKET_LABELS[0]
    if value < d:
        return _BUCKET_LABELS[1]
    if value <= r:
        return _BUCKET_LABELS[2]
    if value <= sr:
        return _BUCKET_LABELS[3]
    return _BUCKET_LABELS[4]


def spread_regime(spread: float) -> str:
    """Brent M1−M2 spread regime. Positive ≈ Backwardation, near-zero ≈ Flat,
    negative ≈ Contango. Threshold ±0.10 USD/bbl."""
    if spread is None or pd.isna(spread):
        return "unknown"
    if spread > 0.10:
        return "Backwardation"
    if spread < -0.10:
        return "Contango"
    return "Flat"


# ---------------------------------------------------------------------------
# Per-article record builder
# ---------------------------------------------------------------------------
def safe_extract(fn, *args, default="unknown"):
    try:
        v = fn(*args)
        return v if v is not None else default
    except Exception as e:
        warnings.warn(f"{fn.__name__} failed: {e}")
        return default


def _fill_price_blanks(rec: Dict[str, Any]) -> None:
    for p in _QUANTILE_LEVELS:
        rec[f"q{p}%"] = float("nan")
    rec["Quantile_Category"] = "unknown"
    rec["price"] = float("nan")
    rec["price_change_daily"] = float("nan")
    rec["price_change_daily_bucket"] = "unknown"
    for off in (7, 15, 30):
        rec[f"price_t_minus_{off}"] = float("nan")
        rec[f"price_t_plus_{off}"]  = float("nan")
        rec[f"price_change_t_minus_{off}_to_t_plus_{off}"] = float("nan")
        rec[f"price_change_t_minus_{off}_to_t_plus_{off}_bucket"] = "unknown"
    for col in ("SMA_7", "EMA_7", "RSI", "Bollinger_Upper", "Bollinger_Lower"):
        rec[col] = float("nan")
    rec["Spread_M1_M2"] = 0.0
    rec["Spread_M1_M2_regime"] = "unknown"
    rec["Spread_M1_M2_change_daily"] = float("nan")
    for off in (7, 15, 30):
        rec[f"Spread_M1_M2_t_minus_{off}"] = float("nan")
        rec[f"Spread_M1_M2_t_plus_{off}"]  = float("nan")
        rec[f"Spread_M1_M2_change_t_minus_{off}_to_t_plus_{off}"] = float("nan")
    rec["Crude Oil-WTI Spot Cushing U$/BBL"] = 0.0
    for m in range(1, 13):
        rec[f"Crude Oil Brent US M{m} U$/BBL"] = 0.0


def build_record(article: Dict[str, Any],
                 price_panel: pd.DataFrame,
                 idx_in_year: int) -> Tuple[Dict[str, Any], np.ndarray]:

    raw = (article.get("body_text") or article.get("body")
           or article.get("text") or article.get("content") or "")
    light_cleaned = light_clean(raw)

    # Oil-focused LLM summary — used as cleaned_text AND as the embedding input.
    # Falls back to light_cleaned if the LLM declares the article irrelevant.
    oil_summary = extract_oil_summary(light_cleaned)
    is_oil = (oil_summary != "" and oil_summary.upper() != "NOT_OIL_RELEVANT")
    cleaned_text_value = oil_summary if is_oil else ""
    embedding_input    = oil_summary if is_oil else light_cleaned

    raw_date = (article.get("date") or article.get("publication_date")
                or article.get("pub_date") or article.get("published"))
    try:
        date = pd.to_datetime(raw_date) if raw_date else pd.NaT
    except Exception:
        date = pd.NaT
    year = float(date.year) if pd.notna(date) else float("nan")

    # ---- LLM attributes (Ollama) — keep using full light-cleaned context ----
    category      = safe_extract(extract_category, light_cleaned)
    subcategory   = safe_extract(extract_subcategory, light_cleaned, category)
    entity        = safe_extract(extract_entity, light_cleaned)
    primary_event = safe_extract(extract_primary_event, light_cleaned)
    event_type    = safe_extract(extract_event_type, light_cleaned)

    # Build record in Brahmanda's exact field order
    rec: Dict[str, Any] = {
        "event_id":       (f"EVT_{int(year)}.0_{idx_in_year:04d}.0"
                           if pd.notna(year) else f"EVT_NA_{idx_in_year:04d}.0"),
        "date":           date.strftime("%Y-%m-%d") if pd.notna(date) else None,
        "document_id":    (article.get("doc_id") or article.get("document_id")
                           or article.get("id") or article.get("an") or ""),
        "text":           raw,
        "cleaned_text":   cleaned_text_value,
        "category":       category,
        "subcategory":    subcategory,
        "Entity":         entity,
        "Primary Event":  primary_event,
        "Event type":     event_type,
        "Commodity Attributes":       safe_extract(extract_commodity_attributes, light_cleaned),
        "Country Attributes":         safe_extract(extract_country_attributes, light_cleaned),
        "Duration Attributes":        safe_extract(extract_duration_attributes, light_cleaned),
        "Quantity Attributes":        safe_extract(extract_quantity_attributes, light_cleaned),
        "Financial Attributes":       safe_extract(extract_financial_attributes, light_cleaned),
        "Forecast Attributes":        safe_extract(extract_forecast_attributes, light_cleaned),
        "Group Attributes":           safe_extract(extract_group_attributes, light_cleaned),
        "Location Attributes":        safe_extract(extract_location_attributes, light_cleaned),
        "Money Attributes":           safe_extract(extract_money_attributes, light_cleaned),
        "Production Unit Attributes": safe_extract(extract_production_unit_attributes, light_cleaned),
        "State or Province":          safe_extract(extract_state_or_province, light_cleaned),
        "Percent Attributes":         safe_extract(extract_percent_attributes, light_cleaned),
        "Person Attributes":          safe_extract(extract_person_attributes, light_cleaned),
        "Price Unit Attributes":      safe_extract(extract_price_unit_attributes, light_cleaned),
        "Event Trigger":              safe_extract(extract_event_trigger, light_cleaned),
        "Related Entity":             safe_extract(extract_related_entity, light_cleaned),
        "Second Event":               safe_extract(extract_second_event, light_cleaned),
        "Third Event":                safe_extract(extract_third_event, light_cleaned),
        "Outcome":                    safe_extract(extract_outcome, light_cleaned),
        "Outcome Attribute":          safe_extract(extract_outcome_attribute, light_cleaned),
        "Temporal Attribute":         safe_extract(extract_temporal_attribute, light_cleaned),
        "Polarity":                   safe_extract(extract_polarity, light_cleaned),
        "Modality":                   safe_extract(extract_modality, light_cleaned),
        "Causal Links":               safe_extract(extract_causal_links, light_cleaned),
        "Influencing Factors":        safe_extract(extract_influencing_factors, light_cleaned),
        "year":                       year,
        "Event_type":                 event_type,    # mirror — matches Brahmanda's duplicate
        "is_oil_relevant":            is_oil,
    }

    # ---- Embedding (over the oil-focused summary, not the raw cleaned text) ----
    vec = embed_text(embedding_input)
    rec["bert_embeddings"] = stringify_embedding(vec)

    # ---- UMAP / HDBSCAN slots (filled in Phase 5 by 06_umap_cluster_assign.py) ----
    rec["dim1"]    = None
    rec["dim2"]    = None
    rec["dim3"]    = None
    rec["cluster"] = None
    rec["Year"]    = year

    # ---- Quantile bands + price + curve + tech indicators ----
    if not price_panel.empty and pd.notna(date):
        wti_col = next((c for c in price_panel.columns
                        if c.lower() in ("wti_m1", "wti", "spot_wti", "wti_spot")), None)
        if wti_col is not None:
            wti = price_panel[wti_col]
            bands = quantile_bands(wti, int(year))
            rec.update(bands)
            try:
                px_today = float(wti.asof(date))
            except Exception:
                px_today = float("nan")
            rec["Quantile_Category"] = quantile_category(px_today, bands)
            rec["price"] = px_today

            try:
                prev = float(wti.asof(date - pd.Timedelta(days=1)))
                rec["price_change_daily"] = ((px_today - prev) / prev * 100) if prev else 0.0
            except Exception:
                rec["price_change_daily"] = float("nan")
            rec["price_change_daily_bucket"] = price_change_bucket(
                rec["price_change_daily"], "daily"
            )

            rec.update(price_window(wti, date))
            for off in (7, 15, 30):
                a = rec[f"price_t_minus_{off}"]
                b = rec[f"price_t_plus_{off}"]
                rec[f"price_change_t_minus_{off}_to_t_plus_{off}"] = (
                    ((b - a) / a * 100) if (a and not pd.isna(a)) else float("nan")
                )
                rec[f"price_change_t_minus_{off}_to_t_plus_{off}_bucket"] = price_change_bucket(
                    rec[f"price_change_t_minus_{off}_to_t_plus_{off}"], str(off)
                )

            ti = technical_indicators(wti)
            for col in ("SMA_7", "EMA_7", "RSI", "Bollinger_Upper", "Bollinger_Lower"):
                try:
                    rec[col] = float(ti[col].asof(date))
                except Exception:
                    rec[col] = float("nan")

            # ---- Brent curve M1-M12 ----
            brent_cols: Dict[int, str] = {}
            for m in range(1, 13):
                col = next((c for c in price_panel.columns
                            if c.lower().replace(" ", "") in
                            (f"brent_m{m}", f"brentm{m}", f"m{m}_brent")), None)
                key = f"Crude Oil Brent US M{m} U$/BBL"
                if col is not None:
                    brent_cols[m] = col
                    try:
                        rec[key] = float(price_panel[col].asof(date))
                    except Exception:
                        rec[key] = 0.0
                else:
                    rec[key] = 0.0

            # ---- Spread M1-M2 time-series + changes + regime ----
            m1_col = brent_cols.get(1)
            m2_col = brent_cols.get(2)
            spread_today = (
                rec["Crude Oil Brent US M1 U$/BBL"] - rec["Crude Oil Brent US M2 U$/BBL"]
            )
            rec["Spread_M1_M2"] = spread_today
            rec["Spread_M1_M2_regime"] = spread_regime(spread_today)

            if m1_col is not None and m2_col is not None:
                m1s = price_panel[m1_col]
                m2s = price_panel[m2_col]
                spread_series = m1s - m2s

                # daily spread change (USD/bbl, not %)
                try:
                    prev_sp = float(spread_series.asof(date - pd.Timedelta(days=1)))
                    rec["Spread_M1_M2_change_daily"] = (spread_today - prev_sp)
                except Exception:
                    rec["Spread_M1_M2_change_daily"] = float("nan")

                # ±7/±15/±30 spread time-series + changes
                for off in (7, 15, 30):
                    try:
                        sp_minus = float(spread_series.asof(date - pd.Timedelta(days=off)))
                    except Exception:
                        sp_minus = float("nan")
                    try:
                        sp_plus  = float(spread_series.asof(date + pd.Timedelta(days=off)))
                    except Exception:
                        sp_plus = float("nan")
                    rec[f"Spread_M1_M2_t_minus_{off}"] = sp_minus
                    rec[f"Spread_M1_M2_t_plus_{off}"]  = sp_plus
                    if pd.notna(sp_minus) and pd.notna(sp_plus):
                        rec[f"Spread_M1_M2_change_t_minus_{off}_to_t_plus_{off}"] = (
                            sp_plus - sp_minus
                        )
                    else:
                        rec[f"Spread_M1_M2_change_t_minus_{off}_to_t_plus_{off}"] = float("nan")
            else:
                rec["Spread_M1_M2_change_daily"] = float("nan")
                for off in (7, 15, 30):
                    rec[f"Spread_M1_M2_t_minus_{off}"] = float("nan")
                    rec[f"Spread_M1_M2_t_plus_{off}"]  = float("nan")
                    rec[f"Spread_M1_M2_change_t_minus_{off}_to_t_plus_{off}"] = float("nan")

            rec["Crude Oil-WTI Spot Cushing U$/BBL"] = rec.get("price", 0.0) or 0.0
        else:
            _fill_price_blanks(rec)
    else:
        _fill_price_blanks(rec)

    # ---- Causal-links mirror + trigger summary fields ----
    rec["causal_links_1"] = rec["Causal Links"]
    trig_words_list = re.findall(r"[A-Za-z]+", str(rec["Event Trigger"]))[:5]
    rec["Trigger Words"] = ", ".join(trig_words_list)
    rec["Event Types"]   = rec["Event_type"]
    rec["combined_text"] = f"{rec['Trigger Words']} {rec['Event Types']}".strip()

    # NOTE: Trigger_embeddings + Influence_Score family are SKIPPED per spec.

    return rec, vec


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit",  type=int, default=None,
                    help="Process only the first N kept articles (smoke test).")
    ap.add_argument("--resume", action="store_true",
                    help="Skip articles whose document_id is already in OUT_JSON.")
    ap.add_argument("--checkpoint", type=int, default=25,
                    help="Save every N records (default 25).")
    args = ap.parse_args()

    if not KEPT_JSON.exists():
        sys.exit(f"❌ Kept articles not found: {KEPT_JSON}\n"
                 f"   Run scripts/04b_factiva_batch.py first.")

    with KEPT_JSON.open(encoding="utf-8") as f:
        articles = json.load(f)
    print(f"Loaded {len(articles)} kept articles from {KEPT_JSON.name}")

    if args.limit:
        articles = articles[:args.limit]
        print(f"  (limited to first {args.limit})")

    price_panel = load_price_panel()
    print(f"Price panel: {len(price_panel)} rows" if not price_panel.empty
          else "Price panel: EMPTY (price fields will be NaN/0.0)")

    # ---- resume support ----
    existing_records: List[Dict[str, Any]] = []
    existing_ids: set = set()
    existing_vecs: List[np.ndarray] = []
    if args.resume and OUT_JSON.exists():
        with OUT_JSON.open(encoding="utf-8") as f:
            existing_records = json.load(f)
        existing_ids = {r.get("document_id") for r in existing_records if r.get("document_id")}
        # parse stored embeddings back to vectors
        for r in existing_records:
            emb = r.get("bert_embeddings", "")
            if isinstance(emb, str) and emb.startswith("["):
                v = np.fromstring(emb.strip("[]"), sep=" ", dtype=np.float32)
                existing_vecs.append(v if v.size == EMBEDDING_DIM
                                     else np.zeros(EMBEDDING_DIM, dtype=np.float32))
            else:
                existing_vecs.append(np.zeros(EMBEDDING_DIM, dtype=np.float32))
        print(f"Resume: {len(existing_records)} records already built — skipping their doc_ids")

    # ---- year-counter (for event_id sequencing) ----
    year_counter: Dict[int, int] = {}
    for r in existing_records:
        try:
            y = int(r.get("year")) if r.get("year") else 0
        except Exception:
            y = 0
        year_counter[y] = year_counter.get(y, 0) + 1

    records: List[Dict[str, Any]] = list(existing_records)
    vecs:    List[np.ndarray]     = list(existing_vecs)

    import time
    n_total = len(articles)
    print(f"Starting build loop: {n_total} articles, "
          f"checkpoint every {args.checkpoint}, LLM timeout {LLM_CALL_TIMEOUT}s/call",
          flush=True)
    run_start = time.time()
    for i, art in enumerate(articles):
        doc_id = (art.get("doc_id") or art.get("document_id")
                  or art.get("id") or art.get("an") or f"row_{i}")
        if doc_id in existing_ids:
            continue
        try:
            d = pd.to_datetime(art.get("date") or art.get("publication_date"))
            yr = int(d.year) if pd.notna(d) else 0
        except Exception:
            yr = 0
        year_counter[yr] = year_counter.get(yr, 0) + 1

        t0 = time.time()
        try:
            rec, vec = build_record(art, price_panel, year_counter[yr])
        except Exception as e:
            warnings.warn(f"[{i+1}/{n_total}] doc_id={doc_id} failed: {e}")
            continue
        dt = time.time() - t0

        records.append(rec)
        vecs.append(vec)

        # Per-article timing line — flushes immediately so terminal stays live
        done = len(records) - len(existing_records)
        remaining_articles = (n_total - (i + 1))
        avg = (time.time() - run_start) / max(done, 1)
        eta_min = (remaining_articles * avg) / 60.0
        oil_flag = "oil" if rec.get("is_oil_relevant", True) else "NON-OIL"
        print(f"  [{i+1:>3}/{n_total}] doc={str(doc_id)[:30]:<30} "
              f"{dt:5.1f}s  {oil_flag:<7}  avg={avg:5.1f}s  eta={eta_min:5.1f}m",
              flush=True)

        if (i + 1) % args.checkpoint == 0:
            _save(records, vecs)
            print(f"  + checkpoint saved at {i+1}/{n_total} ({len(records)} total records)",
                  flush=True)

    _save(records, vecs)
    print(f"\nDONE: Wrote {len(records)} records:")
    print(f"   {OUT_JSON}")
    print(f"   {OUT_JSONL}")
    print(f"   {OUT_NPZ}")
    print(f"   {OUT_CSV}")


def _save(records: List[Dict[str, Any]], vecs: List[np.ndarray]) -> None:
    # JSON (canonical, matches price_event_memory.json)
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2, default=str)

    # JSONL (one record per line)
    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    # NPZ - raw vectors aligned to records[].event_id order
    if vecs:
        ids = np.array([r["event_id"] for r in records], dtype=object)
        np.savez_compressed(OUT_NPZ, ids=ids, embeddings=np.vstack(vecs))

    # CSV metadata mirror (everything except the giant embedding string)
    df = pd.DataFrame(records)
    if "bert_embeddings" in df.columns:
        df = df.drop(columns=["bert_embeddings"])
    df.to_csv(OUT_CSV, index=False)


if __name__ == "__main__":
    main()
