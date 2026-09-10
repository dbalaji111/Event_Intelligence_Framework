"""
build_memory_2024_2026_standardized.py
======================================

Rebuilds 2024–2026 memory with STRICT embedding standardization.

Key guarantees:
- Clean text (ANSI removed)
- Single embedding input (oil_summary only)
- Mean pooling (NOT CLS)
- L2 normalized embeddings
- Stored as numeric arrays (NOT strings)

Output:
- JSONL with embeddings
- NPZ file for fast retrieval
"""

import json
import re
import numpy as np
from pathlib import Path
from typing import List

import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# ---------------- CONFIG ----------------
INPUT_FILE = "memory_2024_2026/df_major_memory_2024_2026.jsonl"
OUTPUT_JSONL = "memory_2024_2026/df_major_memory_2024_2026_cleaned.jsonl"
OUTPUT_NPZ = "memory_2024_2026/df_major_embeddings_2024_2026_cleaned.npz"

MODEL_NAME = "bert-base-uncased"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 512

# ---------------- LOAD MODEL ----------------
print(f"[model] Loading {MODEL_NAME} on {DEVICE}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

# ---------------- CLEANING ----------------
ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = ANSI_PATTERN.sub(" ", text)
    text = re.sub(r"[\x00-\x1F\x7F]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------------- SUMMARY (REUSE FIELD) ----------------
def prepare_embedding_text(record):
    # Prefer cleaned_text if already exists
    summary = record.get("cleaned_text", "")
    summary = clean_text(summary)

    if summary == "" or summary.lower() == "nan":
        # fallback to raw text
        raw = record.get("text", "")
        return clean_text(raw)

    return summary

# ---------------- EMBEDDING ----------------
def embed_text(text: str) -> np.ndarray:
    text = text[:2000]

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    ).to(DEVICE)

    with torch.no_grad():
        output = model(**encoded)

    last_hidden = output.last_hidden_state
    mask = encoded["attention_mask"].unsqueeze(-1).float()

    # Mean pooling
    emb = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    emb = emb.cpu().numpy().astype(np.float32)

    # L2 normalize
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.clip(norm, 1e-8, None)

    return emb.squeeze(0)

# ---------------- VALIDATION ----------------
def validate_embedding(vec):
    if np.isnan(vec).any():
        raise ValueError("NaN in embedding")

    norm = np.linalg.norm(vec)
    if not (0.9 <= norm <= 1.1):
        raise ValueError(f"Bad norm: {norm}")

# ---------------- LOAD DATA ----------------
def load_jsonl(path: str) -> List[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

# ---------------- MAIN ----------------
def main():
    data = load_jsonl(INPUT_FILE)
    print(f"[load] {len(data)} records")

    all_embeddings = []

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:
        for rec in tqdm(data, desc="Processing"):
            text = prepare_embedding_text(rec)

            vec = embed_text(text)
            validate_embedding(vec)

            rec["cleaned_text"] = text
            rec["bert_embeddings"] = vec.tolist()

            fout.write(json.dumps(rec) + "\n")
            all_embeddings.append(vec)

    # Save NPZ
    embeddings = np.vstack(all_embeddings)
    np.savez_compressed(OUTPUT_NPZ, embeddings=embeddings)

    # Diagnostics
    norms = np.linalg.norm(embeddings, axis=1)
    print("\n[diagnostic]")
    print("Norm mean:", norms.mean())
    print("Norm std :", norms.std())
    print("Shape    :", embeddings.shape)

    print("\n✅ Done. Clean embeddings generated.")

# ---------------- RUN ----------------
if __name__ == "__main__":
    main()