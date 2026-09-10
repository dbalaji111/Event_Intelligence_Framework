"""
recompute_embeddings.py
=======================
Recomputes BERT embeddings for the TRAIN corpus (2001-2023)
using the same model as the TEST corpus (2024-2026).

Reads  : memory_2001_2023/price_event_memory.json
Writes : memory_2001_2023/price_event_memory_reembedded.json
         (same structure, bert_embeddings replaced)

Usage
-----
    python recompute_embeddings.py \
        --input  "memory_2001_2023/price_event_memory.json" \
        --output "memory_2001_2023/price_event_memory_reembedded.json" \
        --model  "bert-base-uncased" \
        --batch_size 32

    # To verify alignment after recomputing:
    python diagnose_embeddings_v2.py \
        --train "memory_2001_2023/price_event_memory_reembedded.json" \
        --test  "memory_2024_2026/df_major_memory_2024_2026.jsonl"

Dependencies
------------
    pip install transformers torch tqdm
"""

from __future__ import annotations
import argparse, json, re, sys, time
from pathlib import Path
from typing import List
import numpy as np
from tqdm import tqdm

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

# ── Embedding function ────────────────────────────────────────────────────────

def get_embeddings(texts: List[str],
                   tokenizer,
                   model,
                   device: str,
                   max_length: int = 512) -> np.ndarray:
    """
    CLS-token embedding from BERT.
    Same method used for 2024-2026 corpus (standard approach).
    """
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        output = model(**encoded)

    # CLS token = first token of last hidden state
    cls_embeddings = output.last_hidden_state[:, 0, :].cpu().numpy()
    return cls_embeddings.astype(np.float32)


# ── Loader ────────────────────────────────────────────────────────────────────

def load_json_or_jsonl(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as f:
        first = ""
        for line in f:
            first = line.strip()
            if first: break

    if first.startswith("["):
        with path.open("r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
        return data if isinstance(data, list) else list(data.values())
    else:
        rows = []
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try: rows.append(json.loads(line))
                except: pass
        return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input",      required=True,
                    help="path to input JSON/JSONL file")
    ap.add_argument("--output",     required=True,
                    help="path to output JSON file")
    ap.add_argument("--model",      default="bert-base-uncased",
                    help="HuggingFace model name (default: bert-base-uncased)")
    ap.add_argument("--batch_size", type=int, default=32,
                    help="batch size for embedding (default: 32)")
    ap.add_argument("--text_field", default="text",
                    help="field to embed (default: text)")
    ap.add_argument("--max_length", type=int, default=512,
                    help="max token length (default: 512)")
    ap.add_argument("--verify",     action="store_true",
                    help="verify output by checking first 5 norms")
    args = ap.parse_args()

    in_path  = Path(args.input).resolve()
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Device ────────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] using: {device}")
    if device == "cpu":
        print(f"[device] WARNING: CPU mode — this will be slow for 12k events.")
        print(f"[device]          Estimate: ~30-60 minutes on CPU.")
        print(f"[device]          If you have GPU, use a machine with CUDA.")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\n[model] loading: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model     = AutoModel.from_pretrained(args.model).to(device)
    model.eval()
    print(f"[model] loaded  hidden_size={model.config.hidden_size}")

    # ── Load events ───────────────────────────────────────────────────────────
    print(f"\n[load] {in_path.name}")
    events = load_json_or_jsonl(in_path)
    print(f"[load] {len(events)} events")

    # ── Extract texts ─────────────────────────────────────────────────────────
    texts = []
    for e in events:
        t = str(e.get(args.text_field) or e.get("cleaned_text") or "")
        t = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", " ", t)  # strip ANSI
        t = re.sub(r"\s+", " ", t).strip()
        texts.append(t[:2000])  # truncate before tokeniser

    # ── Compute embeddings in batches ─────────────────────────────────────────
    print(f"\n[embed] batch_size={args.batch_size}  max_length={args.max_length}")
    all_embeddings = []
    t0 = time.time()

    for start in tqdm(range(0, len(texts), args.batch_size),
                      desc="  embedding", unit="batch"):
        batch = texts[start:start + args.batch_size]
        embs  = get_embeddings(batch, tokenizer, model, device, args.max_length)
        all_embeddings.append(embs)

        # ETA
        done    = min(start + args.batch_size, len(texts))
        elapsed = time.time() - t0
        rate    = done / elapsed
        eta     = (len(texts) - done) / rate / 60
        if (start // args.batch_size) % 20 == 0:
            tqdm.write(f"  {done}/{len(texts)}  "
                       f"({rate:.0f} ev/s  ETA {eta:.1f} min)")

    all_embeddings = np.vstack(all_embeddings)
    print(f"\n[embed] done — shape: {all_embeddings.shape}")
    print(f"[embed] L2 norm mean: {np.linalg.norm(all_embeddings, axis=1).mean():.4f}")
    print(f"[embed] value mean  : {all_embeddings.mean():.6f}")

    # ── Write output ──────────────────────────────────────────────────────────
    print(f"\n[write] {out_path}")
    for i, e in enumerate(events):
        e["bert_embeddings"] = all_embeddings[i].tolist()

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(events, f, ensure_ascii=False,
                  separators=(",", ":"))   # compact — no pretty print

    size_mb = out_path.stat().st_size / 1e6
    print(f"[write] done — {size_mb:.0f} MB")

    # ── Verify ────────────────────────────────────────────────────────────────
    if args.verify:
        print(f"\n[verify] checking first 5 embeddings...")
        for i in range(min(5, len(events))):
            emb = np.array(events[i]["bert_embeddings"])
            print(f"  [{i}] norm={np.linalg.norm(emb):.4f}  "
                  f"first3={emb[:3].round(4)}")

    print(f"\n{'='*55}")
    print(f"  Done. {len(events)} events reembedded.")
    print(f"  Output: {out_path}")
    print(f"\n  Next step — verify alignment:")
    print(f"  python diagnose_embeddings_v2.py \\")
    print(f"      --train \"{out_path}\" \\")
    print(f"      --test  \"memory_2024_2026/df_major_memory_2024_2026.jsonl\"")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()