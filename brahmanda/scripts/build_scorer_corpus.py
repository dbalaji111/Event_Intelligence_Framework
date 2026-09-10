"""
Builds scorer_corpus.jsonl from memory_2001_2023/df_major_metadata.csv
and memory_2001_2023/df_major_embeddings_RECOMPUTED.npz.

The scorer (redflag_api_v5.py, _load_corpus) expects a JSON/JSONL file
where each event record carries its own "embedding" field inline.
This script joins the CSV metadata to the NPZ embeddings on document_id
and writes that combined record out, one JSON object per line.

Run from the repo root:
    python scripts/build_scorer_corpus.py
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

META_PATH = "memory_2001_2023/df_major_metadata.csv"
EMB_PATH  = "memory_2001_2023/df_major_embeddings_RECOMPUTED.npz"
OUT_PATH  = "scorer_corpus.jsonl"

def main():
    print(f"Loading metadata from {META_PATH} ...")
    meta = pd.read_csv(META_PATH)
    print(f"  rows: {len(meta)}, columns: {len(meta.columns)}")

    print(f"Loading embeddings from {EMB_PATH} ...")
    npz = np.load(EMB_PATH, allow_pickle=True)
    embs, ids = npz["embeddings"], npz["ids"]
    print(f"  embeddings: {embs.shape}, ids: {ids.shape}")

    id_to_emb = {str(i): e for i, e in zip(ids, embs)}

    def safe_float(v):
        return None if pd.isna(v) else float(v)

    n_written = 0
    n_skipped_no_emb = 0
    n_skipped_no_id = 0

    with open(OUT_PATH, "w") as f:
        for _, row in meta.iterrows():
            raw_id = row.get("document_id")
            if pd.isna(raw_id):
                n_skipped_no_id += 1
                continue
            doc_id = str(raw_id)
            emb = id_to_emb.get(doc_id)
            if emb is None:
                n_skipped_no_emb += 1
                continue

            record = {
                "document_id": doc_id,
                "date": str(row.get("date", "")),
                "text": str(row.get("text", ""))[:800],
                "event_type": str(row.get("Event type", row.get("Event_type", ""))),
                "r7": safe_float(row.get("price_change_t_minus_7_to_t_plus_7")),
                "r15": safe_float(row.get("price_change_t_minus_15_to_t_plus_15")),
                "r30": safe_float(row.get("price_change_t_minus_30_to_t_plus_30")),
                "embedding": emb.tolist(),
            }
            f.write(json.dumps(record) + "\n")
            n_written += 1

    print()
    print(f"written           = {n_written}")
    print(f"skipped (no id)   = {n_skipped_no_id}")
    print(f"skipped (no emb)  = {n_skipped_no_emb}")
    print(f"output file       = {Path(OUT_PATH).resolve()}")
    if n_written == 0:
        print("!! WARNING: zero records written — check id formats between CSV and NPZ")
    elif n_skipped_no_emb > 0.1 * len(meta):
        print("!! WARNING: >10% of rows had no matching embedding — check id format alignment")

if __name__ == "__main__":
    main()