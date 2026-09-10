"""
verify_retrieval.py
====================
Picks 5 test events and retrieves top-10 analogues for each.
Saves full text of base event + analogues to a readable JSON
so you can manually verify semantic similarity.

Usage
-----
    python verify_retrieval.py \
        --train "memory_2001_2023/price_event_memory_reembedded_standardized.json" \
        --test  "memory_2024_2026/df_major_memory_2024_2026_cleaned.jsonl" \
        --output verification_output/ \
        --tau 0.80 \
        --top_k 10

    # Pick specific events by index (0-based):
    python verify_retrieval.py ... --indices 0 50 100 200 300
"""

import argparse, json, re, sys
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from sklearn.preprocessing import normalize

def strip_ansi(s): 
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", " ", s)

def parse_emb(raw) -> Optional[np.ndarray]:
    if raw is None: return None
    if isinstance(raw, (list, np.ndarray)):
        arr = np.array(raw, dtype=np.float32).ravel()
        return arr if len(arr) >= 100 else None
    if isinstance(raw, str):
        clean = strip_ansi(raw).replace("[","").replace("]","").replace("\n"," ")
        nums = []
        for t in clean.split():
            try: nums.append(float(t.rstrip(",")))
            except: pass
        if len(nums) >= 100:
            return np.array(nums, dtype=np.float32)
    return None

def clean_text(s: str, max_len: int = 600) -> str:
    s = strip_ansi(str(s)).strip()
    s = re.sub(r"\s+", " ", s)
    return s[:max_len] + ("..." if len(s) > max_len else "")

def safe_float(v):
    try:
        f = float(v)
        return round(f, 4) if np.isfinite(f) else None
    except: return None

def load_corpus(path: Path, label: str):
    print(f"[load] {label}: {path.name}")
    records = []

    with path.open("r", encoding="utf-8", errors="replace") as f:
        first = ""
        for line in f:
            first = line.strip()
            if first: break

    if first.startswith("["):
        with path.open("r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
        rows = data if isinstance(data, list) else list(data.values())
    else:
        rows = []
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try: rows.append(json.loads(line))
                except: pass

    bad = 0
    for idx, row in enumerate(rows):
        emb = parse_emb(row.get("bert_embeddings"))
        if emb is None:
            bad += 1
            continue
        if len(emb) < 768: emb = np.pad(emb, (0, 768-len(emb)))
        else: emb = emb[:768]

        raw_date = row.get("date","")
        try:
            import pandas as pd
            date = pd.to_datetime(str(raw_date))
        except: bad += 1; continue

        records.append({
            "idx":        idx,
            "event_id":   str(row.get("event_id", f"EVT_{date.year}_{idx:05d}")),
            "date":       str(date.date()),
            "year":       safe_float(row.get("year", date.year)),
            "event_type": str(row.get("Event type") or row.get("Event_type") or
                              row.get("category") or ""),
            "polarity":   str(row.get("Polarity","")).strip(),
            "quantile":   str(row.get("Quantile_Category","")).strip(),
            "entity":     clean_text(str(row.get("Entity","")), 200),
            "primary":    clean_text(str(row.get("Primary Event","")), 200),
            "text":       clean_text(str(row.get("text","") or row.get("cleaned_text","")), 800),
            "price":      safe_float(row.get("price") or row.get("Crude Oil-WTI Spot Cushing U$/BBL")),
            "r7":         safe_float(row.get("price_change_t_minus_7_to_t_plus_7")),
            "r15":        safe_float(row.get("price_change_t_minus_15_to_t_plus_15")),
            "r30":        safe_float(row.get("price_change_t_minus_30_to_t_plus_30")),
            "influence":  safe_float(row.get("Influence_Score",0)),
            "embedding":  emb.astype(np.float32),
        })

    print(f"[load]   {len(records)} valid  |  {bad} skipped")
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train",   required=True)
    ap.add_argument("--test",    required=True)
    ap.add_argument("--output",  default="verification_output")
    ap.add_argument("--tau",     type=float, default=0.80)
    ap.add_argument("--top_k",   type=int,   default=10)
    ap.add_argument("--indices", type=int, nargs="+", default=None,
                    help="0-based indices of test events to verify (default: auto-pick 5)")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = load_corpus(Path(args.train), "TRAIN 2001-2023")
    test  = load_corpus(Path(args.test),  "TEST  2024-2026")

    # Build normalised embedding matrix for train
    print("[index] building train embedding index...")
    X_train = np.stack([r["embedding"] for r in train])
    X_train_norm = normalize(X_train, norm="l2").astype(np.float32)
    print(f"[index] shape: {X_train_norm.shape}")

    # Pick 5 test events — spread across different event types and years
    if args.indices:
        picks = [test[i] for i in args.indices if i < len(test)]
    else:
        # Auto-pick: diverse event types
        import pandas as pd
        types = {}
        for rec in test:
            t = rec["event_type"][:30]
            if t not in types:
                types[t] = rec
            if len(types) >= 5:
                break
        picks = list(types.values())[:5]
        # If fewer than 5 types, fill with evenly spaced events
        if len(picks) < 5:
            step = max(1, len(test) // 5)
            picks = [test[i*step] for i in range(5) if i*step < len(test)]

    print(f"\n[verify] running retrieval for {len(picks)} test events")
    print(f"[verify] tau={args.tau}  top_k={args.top_k}\n")

    results = []

    for pick_num, base in enumerate(picks):
        import pandas as pd
        base_date = pd.to_datetime(base["date"])
        base_emb  = normalize(base["embedding"].reshape(1,-1), norm="l2").astype(np.float32)
        sims      = (X_train_norm @ base_emb.T).ravel()

        # Filter: tau + temporal
        candidates = [
            (i, float(sims[i]))
            for i in range(len(train))
            if float(sims[i]) >= args.tau
            and pd.to_datetime(train[i]["date"]) < base_date
        ]
        candidates.sort(key=lambda x: -x[1])
        top = candidates[:args.top_k]

        print(f"{'─'*70}")
        print(f"BASE EVENT {pick_num+1}/5: {base['event_id']}")
        print(f"  Date     : {base['date']}")
        print(f"  Type     : {base['event_type']}")
        print(f"  Polarity : {base['polarity']}")
        print(f"  Quantile : {base['quantile']}")
        print(f"  Price    : ${base['price']}/bbl  r(+7)={base['r7']}%  r(+15)={base['r15']}%")
        print(f"  Text     : {base['text'][:200]}")
        print(f"  Candidates (tau={args.tau}): {len(candidates)}")
        print(f"  Top {args.top_k} analogues:")
        for rank, (i, sim) in enumerate(top, 1):
            a = train[i]
            print(f"    [{rank:>2}] sim={sim:.4f}  {a['date']}  "
                  f"r30={a['r30']}%  {a['event_type'][:30]}")
        print()

        # Build rich output record
        result = {
            "base_event": {
                "event_id":   base["event_id"],
                "date":       base["date"],
                "event_type": base["event_type"],
                "polarity":   base["polarity"],
                "quantile":   base["quantile"],
                "entity":     base["entity"],
                "primary_event": base["primary"],
                "price_usd":  base["price"],
                "r7_pct":     base["r7"],
                "r15_pct":    base["r15"],
                "r30_pct":    base["r30"],
                "influence":  base["influence"],
                "full_text":  base["text"],
            },
            "retrieval_stats": {
                "tau":          args.tau,
                "n_candidates": len(candidates),
                "n_returned":   len(top),
                "sim_max":      round(top[0][1], 4) if top else None,
                "sim_min":      round(top[-1][1], 4) if top else None,
            },
            "analogues": []
        }

        for rank, (i, sim) in enumerate(top, 1):
            a = train[i]
            result["analogues"].append({
                "rank":        rank,
                "similarity":  round(sim, 4),
                "event_id":    a["event_id"],
                "date":        a["date"],
                "year":        a["year"],
                "event_type":  a["event_type"],
                "polarity":    a["polarity"],
                "quantile":    a["quantile"],
                "entity":      a["entity"],
                "primary_event": a["primary"],
                "price_usd":   a["price"],
                "r7_pct":      a["r7"],
                "r15_pct":     a["r15"],
                "r30_pct":     a["r30"],
                "influence":   a["influence"],
                "full_text":   a["text"],
                # Semantic match assessment fields (fill manually)
                "MANUAL_CHECK_same_theme":   "?",
                "MANUAL_CHECK_makes_sense":  "?",
                "MANUAL_CHECK_notes":        "",
            })

        results.append(result)

        # Save individual file per base event
        out_file = out_dir / f"verify_{pick_num+1}_{base['event_id'].replace('.','_')}.json"
        with out_file.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        print(f"  Saved -> {out_file}")

    # Save combined file
    combined = out_dir / "all_5_verifications.json"
    with combined.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    # Print summary table
    print(f"\n{'='*70}")
    print(f"VERIFICATION SUMMARY")
    print(f"{'─'*70}")
    print(f"{'Event':<35} {'Type':<25} {'Candidates':>10} {'Top sim':>8}")
    print(f"{'─'*70}")
    for r in results:
        be    = r["base_event"]
        stats = r["retrieval_stats"]
        print(f"{be['event_id']:<35} "
              f"{be['event_type'][:24]:<25} "
              f"{stats['n_candidates']:>10} "
              f"{str(stats['sim_max']):>8}")
    print(f"{'='*70}")
    print(f"\nFiles saved to: {out_dir}")
    print(f"Open all_5_verifications.json to read full texts side by side")
    print(f"\nFor each analogue, check:")
    print(f"  1. Is the EVENT TYPE similar to the base event?")
    print(f"  2. Does the TEXT describe a similar market situation?")
    print(f"  3. Are the ENTITIES relevant (same countries/orgs)?")
    print(f"  4. Does the r30 direction make sense as an analogue?")


if __name__ == "__main__":
    main()