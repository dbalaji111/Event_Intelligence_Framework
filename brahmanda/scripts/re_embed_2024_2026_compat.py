"""
re_embed_2024_2026_compat.py
=============================
Re-embed the 2024-2026 corpus using the SAME text input strategy as the
2001-2023 master (regex-cleaned article body), so the resulting BERT vectors
live in the same region of embedding space and cross-corpus cosine works.

Why this is needed
------------------
Script 05 set cleaned_text = LLM oil_summary and embedded THAT. The master
2001-2023 corpus was embedded on raw-cleaned article body. The two text
distributions are different enough that BERT vectors land in different
sub-regions. Cosine between any 2026 cluster and any 2001-2023 cluster
saturates at ~0.40 — too low for retrieval and impossible to discriminate.

Fix: re-embed the raw `text` field of the 2024-2026 metadata CSV with the
same light-cleaning + BERT mean-pooling pipeline the master used. New NPZ
gets aligned by document_id (not event_id, which was the previous bug).

Reads
-----
  --metadata_csv    memory_2024_2026/df_major_metadata_2024_2026.csv
  --text_field      'text' (default, matches master) or 'cleaned_text'

Writes
------
  --output_npz      memory_2024_2026/df_major_embeddings_2024_2026_MASTER_COMPAT.npz

Run
---
    python re_embed_2024_2026_compat.py \
        --metadata_csv memory_2024_2026/df_major_metadata_2024_2026.csv \
        --output_npz   memory_2024_2026/df_major_embeddings_2024_2026_MASTER_COMPAT.npz \
        --device cuda

After this:
    1. Re-run script 07 with --embeddings_filename df_major_embeddings_2024_2026_MASTER_COMPAT.npz
    2. Re-run script 08 with the same flag
    3. Re-merge with master via the 'M_' prefix one-liner
    4. Re-run script 09b — cosines should now be 0.7-0.95, real retrievals
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

EMBEDDING_MODEL = "bert-base-uncased"
EMBEDDING_DIM   = 768
MAX_TOKENS      = 512

_FACTIVA_BOILER = re.compile(
    r"(--- Page \d+ ---|Page \d+ of \d+|©\s*\d{4}\s*Factiva|All rights reserved\.)",
    re.IGNORECASE,
)


def light_clean(text: object) -> str:
    """Same regex cleaning script 05 used for light_cleaned, matches master."""
    if not isinstance(text, str):
        return ""
    t = _FACTIVA_BOILER.sub(" ", text)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata_csv", required=True)
    ap.add_argument("--output_npz", required=True)
    ap.add_argument("--text_field", default="text",
                    help="which field to embed: 'text' (raw article, default — "
                         "matches master) or 'cleaned_text' (LLM oil_summary)")
    ap.add_argument("--device", default="cuda",
                    help="cuda | cpu | mps  (falls back to cpu if cuda unavailable)")
    ap.add_argument("--batch_size", type=int, default=8,
                    help="batch size for BERT forward pass (default 8)")
    args = ap.parse_args()

    csv_path = Path(args.metadata_csv).resolve()
    out_npz  = Path(args.output_npz).resolve()
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    print(f"[load] {csv_path.name}")
    df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")
    print(f"[load]   {len(df)} rows")

    if args.text_field not in df.columns:
        sys.exit(f"ERROR: column '{args.text_field}' not in CSV. "
                 f"Available: {list(df.columns)[:30]}")
    if "document_id" not in df.columns:
        sys.exit("ERROR: 'document_id' column required for alignment.")

    print(f"[clean] light-cleaning {len(df)} '{args.text_field}' fields")
    cleaned = df[args.text_field].apply(light_clean).tolist()
    n_empty = sum(1 for t in cleaned if not t)
    print(f"[clean]   {n_empty} empty after cleaning")

    # Load BERT
    print(f"[load] {EMBEDDING_MODEL}")
    import torch
    from transformers import AutoTokenizer, AutoModel
    dev = args.device
    if dev == "cuda" and not torch.cuda.is_available():
        dev = "cpu"
        print(f"[load]   cuda unavailable, using cpu")
    tok = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    mdl = AutoModel.from_pretrained(EMBEDDING_MODEL).to(dev).eval()
    print(f"[load]   on {dev}")

    # Embed (one-by-one for simplicity; 406 docs is fine)
    embs = np.zeros((len(df), EMBEDDING_DIM), dtype=np.float32)
    print(f"[embed] processing {len(df)} events")
    t0 = time.time()
    with torch.no_grad():
        for i, text in enumerate(cleaned):
            if not text:
                continue
            enc = tok(text, return_tensors="pt", truncation=True,
                      max_length=MAX_TOKENS, padding=True).to(dev)
            out = mdl(**enc)
            last = out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            embs[i] = pooled.squeeze(0).cpu().numpy().astype(np.float32)
            if (i + 1) % 50 == 0 or i == 0:
                rate = (i + 1) / (time.time() - t0)
                eta = (len(df) - i - 1) / max(rate, 0.1)
                print(f"  [{i+1:>4}/{len(df)}]  {rate:5.1f} ev/s  ETA {eta:5.1f}s",
                      flush=True)
    print(f"[embed] done in {time.time() - t0:.1f}s")

    # Sanity check
    norms = np.linalg.norm(embs, axis=1)
    n_zero = (norms < 1e-6).sum()
    print(f"[check] norms: min={norms.min():.2f}, mean={norms.mean():.2f}, "
          f"max={norms.max():.2f}; zero-vector rows: {n_zero}")

    ids = df["document_id"].astype(str).tolist()
    np.savez_compressed(out_npz, embeddings=embs,
                        ids=np.array(ids, dtype=object))
    print(f"[save] {out_npz}")
    print(f"[save]   {embs.shape[0]} embeddings, {len(ids)} document_ids")


if __name__ == "__main__":
    main()
