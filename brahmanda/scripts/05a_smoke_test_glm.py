"""
05a_smoke_test_glm.py
=====================
Time the full extraction pipeline (30 LLM calls + 1 BERT embedding) per article
using GLM-4.7-Flash via Ollama subprocess.

What it does
------------
1. Monkeypatches r02_all_llm_functions.run_ollama_prompt to use glm-4.7-flash:latest
2. Sends a warm-up prompt so the model is loaded into VRAM (excluded from timing)
3. Runs the first N kept articles through every extractor + the BERT embedder
4. Reports per-extractor median time, per-article total time, and projected
   full-corpus wall-clock time.

Run
---
    python -m scripts.05a_smoke_test_glm                  # 3 articles
    python -m scripts.05a_smoke_test_glm --n 5
    python -m scripts.05a_smoke_test_glm --model qwen2.5:7b-instruct
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List

# ---------------------------------------------------------------------------
# Make existing extractors importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR  = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "glm-4.7-flash:latest"
KEPT_JSON     = PROJECT_ROOT / "data" / "oil_data_2024_2026" / "_processed" / "_ALL__kept.json"


# ---------------------------------------------------------------------------
# GLM subprocess runner — accepts both calling conventions so it can monkeypatch
# r02_all_llm_functions.run_ollama_prompt(model_name, prompt_text) cleanly.
# ---------------------------------------------------------------------------
def make_ollama_runner(model: str) -> Callable[..., str]:
    """Returns a function that calls `ollama run <model>` via subprocess."""
    def run_ollama_prompt(*args, **kwargs) -> str:
        # Detect calling convention:
        #   r02_all_llm_functions style:  (model_name, prompt_text)
        #   user-supplied style:           (prompt, model=...)
        if len(args) == 2:
            prompt = args[1]                # (model_name, prompt_text)
        elif len(args) == 1:
            prompt = args[0]
        else:
            prompt = kwargs.get("prompt_text") or kwargs.get("prompt") or ""
        try:
            proc = subprocess.Popen(
                ["ollama", "run", model],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            stdout, stderr = proc.communicate(input=prompt, timeout=300)
            if proc.returncode != 0:
                print(f"[OLLAMA ERROR] {stderr.strip()[:200]}", file=sys.stderr)
                return "unknown"
            return stdout.strip()
        except subprocess.TimeoutExpired:
            proc.kill()
            print("[OLLAMA TIMEOUT] killed after 300s", file=sys.stderr)
            return "unknown"
        except Exception as e:
            print(f"[OLLAMA EXCEPTION] {e}", file=sys.stderr)
            return "unknown"
    return run_ollama_prompt


def warmup(model: str) -> float:
    """Send a tiny prompt so the model loads into VRAM. Returns seconds taken."""
    print(f"Warming up {model} ...", end=" ", flush=True)
    t0 = time.perf_counter()
    runner = make_ollama_runner(model)
    runner(model, "Reply with the single word: ok")
    dt = time.perf_counter() - t0
    print(f"{dt:.1f}s")
    return dt


# ---------------------------------------------------------------------------
# Patch & import
# ---------------------------------------------------------------------------
def patch_extractors(model: str) -> None:
    import r02_all_llm_functions as _llm
    _llm.run_ollama_prompt = make_ollama_runner(model)


# ---------------------------------------------------------------------------
# Per-extractor timing
# ---------------------------------------------------------------------------
def time_one_article(article: Dict, embedder=None,
                     capture: Dict | None = None) -> Dict[str, float]:
    """Returns {extractor_name: seconds}. Also fills `capture` (if given) with
    {extractor_name: extracted_value} so callers can dump outputs to disk."""
    import r02_all_llm_functions as _llm

    raw = (article.get("body_text") or article.get("body")
           or article.get("text") or article.get("content") or "")
    text = raw[:6000]   # cap at 6k chars to keep prompts reasonable

    # 30 extractor calls in the order from r02_all_llm_functions
    plan = [
        ("category",                    lambda: _llm.extract_category(text)),
        ("subcategory",                 lambda: _llm.extract_subcategory(text, "Geopolitical News")),
        ("entity",                      lambda: _llm.extract_entity(text)),
        ("primary_event",               lambda: _llm.extract_primary_event(text)),
        ("event_type",                  lambda: _llm.extract_event_type(text)),
        ("commodity_attributes",        lambda: _llm.extract_commodity_attributes(text)),
        ("country_attributes",          lambda: _llm.extract_country_attributes(text)),
        ("duration_attributes",         lambda: _llm.extract_duration_attributes(text)),
        ("quantity_attributes",         lambda: _llm.extract_quantity_attributes(text)),
        ("financial_attributes",        lambda: _llm.extract_financial_attributes(text)),
        ("forecast_attributes",         lambda: _llm.extract_forecast_attributes(text)),
        ("group_attributes",            lambda: _llm.extract_group_attributes(text)),
        ("location_attributes",         lambda: _llm.extract_location_attributes(text)),
        ("money_attributes",            lambda: _llm.extract_money_attributes(text)),
        ("production_unit_attributes",  lambda: _llm.extract_production_unit_attributes(text)),
        ("state_or_province",           lambda: _llm.extract_state_or_province(text)),
        ("percent_attributes",          lambda: _llm.extract_percent_attributes(text)),
        ("person_attributes",           lambda: _llm.extract_person_attributes(text)),
        ("price_unit_attributes",       lambda: _llm.extract_price_unit_attributes(text)),
        ("event_trigger",               lambda: _llm.extract_event_trigger(text)),
        ("related_entity",              lambda: _llm.extract_related_entity(text)),
        ("second_event",                lambda: _llm.extract_second_event(text)),
        ("third_event",                 lambda: _llm.extract_third_event(text)),
        ("outcome",                     lambda: _llm.extract_outcome(text)),
        ("outcome_attribute",           lambda: _llm.extract_outcome_attribute(text)),
        ("temporal_attribute",          lambda: _llm.extract_temporal_attribute(text)),
        ("polarity",                    lambda: _llm.extract_polarity(text)),
        ("modality",                    lambda: _llm.extract_modality(text)),
        ("causal_links",                lambda: _llm.extract_causal_links(text)),
        ("influencing_factors",         lambda: _llm.extract_influencing_factors(text)),
    ]

    times: Dict[str, float] = {}
    for name, fn in plan:
        t0 = time.perf_counter()
        try:
            result = fn()
        except Exception as e:
            print(f"  ! {name} crashed: {e}")
            result = f"[ERROR: {e}]"
        times[name] = time.perf_counter() - t0
        if capture is not None:
            capture[name] = result

    # Embedding (only if embedder provided)
    if embedder is not None:
        t0 = time.perf_counter()
        vec = embedder(text)
        times["bert_embedding"] = time.perf_counter() - t0
        if capture is not None:
            capture["bert_embedding_shape"] = list(vec.shape)
            capture["bert_embedding"] = [float(x) for x in vec]   # full 768-dim
            capture["bert_embedding_first5"] = [float(x) for x in vec[:5]]  # sanity preview

    return times


def make_embedder():
    """Lazy BERT embedder so we time it the same way the build pipeline will."""
    import numpy as np
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        print("[warn] transformers/torch not installed — skipping embedding timing")
        return None

    name = "bert-base-uncased"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModel.from_pretrained(name).to(dev).eval()
    print(f"[embedder] {name} loaded on {dev}")

    def embed(text: str):
        if not text.strip():
            return np.zeros(768, dtype=np.float32)
        enc = tok(text, return_tensors="pt", truncation=True,
                  max_length=512, padding=True).to(dev)
        with torch.no_grad():
            out = mdl(**enc)
        last = out.last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return pooled.squeeze(0).cpu().numpy().astype(np.float32)
    return embed


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n",         type=int, default=3, help="Articles to test (default 3)")
    ap.add_argument("--model",     default=DEFAULT_MODEL, help=f"Ollama tag (default {DEFAULT_MODEL})")
    ap.add_argument("--no-embed",  action="store_true", help="Skip BERT embedding timing")
    ap.add_argument("--corpus",    type=int, default=None,
                    help="Project total runtime over this many articles")
    args = ap.parse_args()

    if not KEPT_JSON.exists():
        sys.exit(f"❌ Kept articles not found: {KEPT_JSON}")

    with KEPT_JSON.open(encoding="utf-8") as f:
        articles = json.load(f)
    total_corpus = args.corpus or len(articles)
    print(f"Corpus: {len(articles)} kept articles in {KEPT_JSON.name}")
    articles = articles[: args.n]
    print(f"Smoke test: first {len(articles)}\n")

    # Warm up + patch
    warmup_dt = warmup(args.model)
    patch_extractors(args.model)
    embedder = None if args.no_embed else make_embedder()

    # Output capture
    out_dir  = PROJECT_ROOT / "memory_2024_2026"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"_smoke_test_outputs_{args.model.replace(':', '_').replace('/', '_')}.json"
    captured_articles: List[Dict] = []

    # Run
    all_times: List[Dict[str, float]] = []
    for i, art in enumerate(articles, 1):
        doc_id = (art.get("doc_id") or art.get("document_id") or art.get("id") or f"row_{i}")
        body_chars = len(art.get("body_text") or art.get("body") or art.get("text") or "")
        print(f"\n[{i}/{len(articles)}] {doc_id} — {body_chars} chars")
        capture: Dict = {
            "doc_id":   doc_id,
            "title":    art.get("title", ""),
            "date":     art.get("date", ""),
            "n_chars":  body_chars,
            "model":    args.model,
        }
        t0 = time.perf_counter()
        per_call = time_one_article(art, embedder=embedder, capture=capture)
        article_total = time.perf_counter() - t0
        capture["_total_seconds"] = round(article_total, 2)
        all_times.append({**per_call, "_TOTAL_": article_total})
        captured_articles.append(capture)
        # incremental save so a Ctrl+C still leaves us with what we collected
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(captured_articles, f, ensure_ascii=False, indent=2, default=str)
        print(f"  ⏱  total: {article_total:.1f}s "
              f"(LLM={article_total - per_call.get('bert_embedding', 0):.1f}s, "
              f"BERT={per_call.get('bert_embedding', 0):.2f}s)")
        print(f"  💾 outputs → {out_path.name}")

    # Aggregate
    print("\n" + "=" * 70)
    print(f"PER-EXTRACTOR (median across {len(all_times)} articles)")
    print("=" * 70)
    extractor_names = [k for k in all_times[0].keys() if k != "_TOTAL_"]
    medians = {n: statistics.median(t[n] for t in all_times) for n in extractor_names}
    for n in sorted(medians, key=medians.get, reverse=True):
        bar = "█" * int(medians[n])
        print(f"  {n:30s} {medians[n]:6.2f}s  {bar}")

    article_totals = [t["_TOTAL_"] for t in all_times]
    print("\n" + "=" * 70)
    print("PER-ARTICLE (wall-clock total per article)")
    print("=" * 70)
    print(f"  min    : {min(article_totals):6.1f}s")
    print(f"  median : {statistics.median(article_totals):6.1f}s")
    print(f"  mean   : {statistics.mean(article_totals):6.1f}s")
    print(f"  max    : {max(article_totals):6.1f}s")

    # Projection
    mean = statistics.mean(article_totals)
    print("\n" + "=" * 70)
    print(f"PROJECTION over full corpus = {total_corpus} articles "
          f"(at mean {mean:.1f}s/article)")
    print("=" * 70)
    secs = mean * total_corpus
    print(f"  ≈ {secs/60:.1f} minutes  ({secs/3600:.2f} hours)")
    print(f"  warm-up was {warmup_dt:.1f}s (one-time, not in projection)\n")


if __name__ == "__main__":
    main()
