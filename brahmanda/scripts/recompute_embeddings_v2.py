"""
recompute_embeddings_standardized.py
====================================

Recompute embeddings for 2001–2023 with STRICT standardization.

Guarantees:
- Clean text (ANSI removed)
- Single input (cleaned_text preferred)
- Mean pooling (NOT CLS)
- L2-normalized embeddings
- Stored as numeric arrays

Usage (PowerShell):
python scripts/recompute_embeddings_standardized.py `
  --input  "memory_2001_2023\price_event_memory.json" `
  --output "memory_2001_2023\price_event_memory_reembedded_standardized.json" `
  --batch_size 32
"""

from __future__ import annotations
import argparse, json, re, sys, time
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm

# ---- deps check ----
def check_deps():
    missing = []
    for pkg, imp in [("transformers","transformers"), ("torch","torch")]:
        try: __import__(imp)
        except ImportError: missing.append(pkg)
    if missing:
        sys.exit(f"ERROR: pip install {' '.join(missing)}")

check_deps()

import torch
from transformers import AutoTokenizer, AutoModel

# ---------------- CONFIG ----------------
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 512

# ---------------- CLEANING ----------------
ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = ANSI_PATTERN.sub(" ", text)               # remove ANSI
    text = re.sub(r"[\x00-\x1F\x7F]", " ", text)     # control chars
    text = re.sub(r"\s+", " ", text).strip()         # whitespace
    return text

# ---------------- LOADERS ----------------
def load_json_or_jsonl(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as f:
        first = ""
        for line in f:
            first = line.strip()
            if first:
                break

    if first.startswith("["):
        with path.open("r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
        return data if isinstance(data, list) else list(data.values())
    else:
        rows = []
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except:
                    pass
        return rows

# ---------------- EMBEDDING ----------------
def embed_batch(texts: List[str], tokenizer, model, device: str) -> np.ndarray:
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        output = model(**encoded)

    last_hidden = output.last_hidden_state
    mask = encoded["attention_mask"].unsqueeze(-1).float()

    # ✅ mean pooling
    emb = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    emb = emb.cpu().numpy().astype(np.float32)

    # ✅ L2 normalization
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.clip(norm, 1e-8, None)

    return emb

# ---------------- VALIDATION ----------------
def validate_embeddings(embs: np.ndarray):
    if np.isnan(embs).any():
        raise ValueError("NaN detected in embeddings")

    norms = np.linalg.norm(embs, axis=1)
    if not (0.95 <= norms.mean() <= 1.05):
        raise ValueError(f"Embeddings not normalized. Mean norm={norms.mean():.4f}")

# ---------------- MAIN ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--text_field", default="cleaned_text")
    args = ap.parse_args()

    in_path  = Path(args.input).resolve()
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    print(f"[model] loading {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

    print(f"[load] {in_path.name}")
    events = load_json_or_jsonl(in_path)
    print(f"[load] {len(events)} events")

    # -------- prepare texts --------
    texts = []
    for e in events:
        t = e.get(args.text_field) or e.get("text") or ""
        t = clean_text(str(t))
        texts.append(t[:2000])

    # -------- embed --------
    print(f"[embed] batch_size={args.batch_size}")
    all_embs = []
    t0 = time.time()

    for i in tqdm(range(0, len(texts), args.batch_size), desc="embedding"):
        batch = texts[i:i+args.batch_size]
        embs  = embed_batch(batch, tokenizer, model, device)
        all_embs.append(embs)

    all_embs = np.vstack(all_embs)

    # -------- validate --------
    validate_embeddings(all_embs)

    print(f"[embed] shape: {all_embs.shape}")
    print(f"[embed] norm mean: {np.linalg.norm(all_embs, axis=1).mean():.4f}")

    # -------- write --------
    print(f"[write] {out_path}")
    for i, e in enumerate(events):
        e["bert_embeddings"] = all_embs[i].tolist()

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(events, f, ensure_ascii=False)

    print(f"\n{'='*50}")
    print(f"Done. {len(events)} events re-embedded.")
    print(f"Output: {out_path}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()