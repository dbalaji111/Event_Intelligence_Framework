"""
06_base_event_clustering.py
============================
Base event engineering — local clustering with temporal context.

For each event in a Brahmanda-style memory CSV:
  1. Pull the temporal slice [t - N, t + N] days.
  2. Build a fused pairwise distance matrix from:
       BERT cosine  +  Entity Jaccard  +  Polarity match  +  Quantile match
  3. Run HDBSCAN inside the slice with metric='precomputed'.
  4. Tag the base event with: cluster_id, cluster_role, member doc_ids,
     cohesion, auto-label (top entities + keywords).

Three nested windows are run in parallel: 3, 15, 30 days.

DESIGNED FOR SUPERCOMPUTE — no LLM calls, no web access. Pure NumPy / scikit /
HDBSCAN. Portable: copy this script + the input CSV to the cluster, run.

Inputs
------
df_<YEAR>_with_json_paths.csv        (Brahmanda v3 schema, BERT 768 inline)

Outputs (alongside the input by default)
----------------------------------------
df_<YEAR>_with_local_clusters.csv    one row per event with new cluster columns
local_clusters_<YEAR>.json           side-table of cluster definitions
local_clusters_<YEAR>_report.txt     diagnostic report (size dist, role dist)

Run
---
    python 06_base_event_clustering.py \
        --input  /path/to/df_2023_with_json_paths.csv \
        --windows 3,15,30 \
        --min_cluster_size 2

CLI flags
---------
    --input            path to Brahmanda-style memory CSV (required)
    --output_dir       where to write outputs (default: same dir as input)
    --windows          comma-separated day offsets (default: 3,15,30)
    --min_cluster_size HDBSCAN min cluster size (default: 2)
    --weights          comma-separated 4 weights for fused distance:
                          bert,entity,polarity,quantile (default: 0.5,0.2,0.1,0.2)
    --n_jobs           parallel workers across windows (default: 1)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import hdbscan
except ImportError:
    sys.exit("ERROR: hdbscan not installed. Run: pip install hdbscan")

# ---------------------------------------------------------------------------
# CONFIG — defaults overridable via CLI
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS = (0.5, 0.2, 0.1, 0.2)   # bert, entity, polarity, quantile
DEFAULT_WINDOWS = (3, 15, 30)
DEFAULT_MIN_CLUSTER = 2
EMBEDDING_DIM = 768

# Columns we may pull entities from
ENTITY_COLS = (
    "Entity", "Country Attributes", "Person Attributes",
    "Group Attributes", "Location Attributes",
)
TOKEN_SPLIT = re.compile(r"[,;\n/|]+")


# ---------------------------------------------------------------------------
# PARSERS
# ---------------------------------------------------------------------------
def parse_embedding(s: Any) -> np.ndarray:
    """Brahmanda stores BERT vectors as `np.array2string(threshold=inf)` text.
    Convert back to a 1-D float32 numpy array of length EMBEDDING_DIM."""
    if not isinstance(s, str) or not s.strip():
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)
    cleaned = s.strip().strip("[]")
    cleaned = re.sub(r"[\[\]]", " ", cleaned)
    try:
        v = np.fromstring(cleaned, sep=" ", dtype=np.float32)
    except Exception:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)
    if v.size != EMBEDDING_DIM:
        # try to right-pad / truncate
        if v.size > EMBEDDING_DIM:
            v = v[:EMBEDDING_DIM]
        else:
            v = np.concatenate([v, np.zeros(EMBEDDING_DIM - v.size, dtype=np.float32)])
    return v


def tokenise_entities(row: pd.Series) -> set:
    """Pull a normalised entity bag from all entity-like columns."""
    bag = set()
    for col in ENTITY_COLS:
        if col not in row.index:
            continue
        v = row[col]
        if not isinstance(v, str) or not v.strip():
            continue
        for tok in TOKEN_SPLIT.split(v):
            t = tok.strip().lower()
            if not t or t in ("nan", "none", "n/a", "no one.", "no entity"):
                continue
            # Drop very long tokens (probably paragraphs of LLM output)
            if len(t) > 60:
                continue
            bag.add(t)
    return bag


# ---------------------------------------------------------------------------
# DISTANCES
# ---------------------------------------------------------------------------
def cosine_distance_matrix(X: np.ndarray) -> np.ndarray:
    """All-pairs cosine distance for a (N, D) matrix. Returns (N, N) in [0, 2]."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = X / norms
    sim = Xn @ Xn.T
    sim = np.clip(sim, -1.0, 1.0)
    return 1.0 - sim


def jaccard_distance_matrix(bags: List[set]) -> np.ndarray:
    n = len(bags)
    D = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = bags[i], bags[j]
            if not a and not b:
                d = 0.0
            else:
                inter = len(a & b)
                uni = len(a | b)
                d = 1.0 - (inter / uni if uni else 0.0)
            D[i, j] = D[j, i] = d
    return D


def categorical_distance_matrix(values: List[str]) -> np.ndarray:
    """0 if equal, 1 if not. NaN treated as a separate category."""
    n = len(values)
    D = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            d = 0.0 if values[i] == values[j] else 1.0
            D[i, j] = D[j, i] = d
    return D


def fused_distance(emb: np.ndarray, bags: List[set],
                   polarity: List[str], quantile: List[str],
                   weights: Tuple[float, float, float, float]) -> np.ndarray:
    """w1·BERT + w2·entity + w3·polarity + w4·quantile, all in [0, 1]."""
    w_b, w_e, w_p, w_q = weights
    D_b = cosine_distance_matrix(emb) / 2.0          # rescale [0,2] → [0,1]
    D_e = jaccard_distance_matrix(bags)
    D_p = categorical_distance_matrix(polarity)
    D_q = categorical_distance_matrix(quantile)
    D = w_b * D_b + w_e * D_e + w_p * D_p + w_q * D_q
    np.fill_diagonal(D, 0.0)
    # HDBSCAN with precomputed needs symmetric, finite, double
    return D.astype(np.float64)


# ---------------------------------------------------------------------------
# CLUSTER ROLE CLASSIFIER
# ---------------------------------------------------------------------------
def classify_role(base_date: pd.Timestamp,
                  member_dates: List[pd.Timestamp],
                  cluster_size: int) -> str:
    """NOVEL / REPEAT / CONTINUATION / REGIME-MARKER, rule-based on cluster
    size and how widely member dates are spread."""
    if cluster_size <= 1:
        return "NOVEL"
    spread_days = (max(member_dates) - min(member_dates)).days
    # If everything is within ±2 days of base → REPEAT (yesterday's blurb again)
    deltas = [abs((d - base_date).days) for d in member_dates]
    if max(deltas) <= 2:
        return "REPEAT"
    # Larger clusters spanning a week+ → REGIME-MARKER
    if cluster_size >= 3 and spread_days >= 7:
        return "REGIME-MARKER"
    # Everything else is an unfolding event arc
    return "CONTINUATION"


# ---------------------------------------------------------------------------
# AUTO-LABEL
# ---------------------------------------------------------------------------
_KEYWORD_BLOCKLIST = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "by",
    "with", "from", "is", "are", "was", "were", "has", "have", "had",
    "be", "been", "as", "at", "this", "that", "it", "its",
}


def auto_label(member_rows: pd.DataFrame, max_terms: int = 3) -> str:
    """Pick the top entities + the top keywords from cleaned_text across cluster
    members. Returns 'top_entity_a · top_entity_b · top_keyword'."""
    # Top entities by frequency across members
    bag_counter: Counter = Counter()
    for _, r in member_rows.iterrows():
        bag_counter.update(tokenise_entities(r))
    top_entities = [w for w, _ in bag_counter.most_common(max_terms)]

    # Top keywords from cleaned_text (very rough; just frequency)
    word_counter: Counter = Counter()
    if "cleaned_text" in member_rows.columns:
        text_blob = " ".join(
            str(t) for t in member_rows["cleaned_text"].fillna("").tolist()
        ).lower()
        words = re.findall(r"[a-z][a-z\-]{3,}", text_blob)
        for w in words:
            if w in _KEYWORD_BLOCKLIST:
                continue
            word_counter[w] += 1
    top_words = [w for w, _ in word_counter.most_common(max_terms)]

    parts = top_entities + top_words
    if not parts:
        return "(unlabeled)"
    return " · ".join(parts[:max_terms])


# ---------------------------------------------------------------------------
# CORE LOOP — one window-size pass over the whole corpus
# ---------------------------------------------------------------------------
def run_window(df: pd.DataFrame, embeddings: np.ndarray,
               entity_bags: List[set],
               window_days: int,
               min_cluster_size: int,
               weights: Tuple[float, float, float, float]) -> Tuple[pd.DataFrame, Dict]:
    """For every event, build a local distance matrix in its ±window slice,
    run HDBSCAN, and record the base event's cluster + role + label."""
    n = len(df)
    out_rows = []
    cluster_table: Dict[str, Dict] = {}      # cluster_uid -> definition
    cluster_uid = 0
    seen_signature: Dict[Tuple, str] = {}    # to dedupe identical clusters

    polarity_vals = df["Polarity"].fillna("UNK").astype(str).tolist() \
        if "Polarity" in df.columns else ["UNK"] * n
    quantile_vals = df["Quantile_Category"].fillna("UNK").astype(str).tolist() \
        if "Quantile_Category" in df.columns else ["UNK"] * n
    dates = pd.to_datetime(df["date"], errors="coerce").tolist()

    for i in range(n):
        base_date = dates[i]
        if pd.isna(base_date):
            out_rows.append({
                "cluster_uid": None, "role": "NOVEL", "size": 1,
                "members": "[]", "cohesion": 0.0, "label": "(no date)"
            })
            continue

        # window slice (inclusive both sides)
        lo = base_date - pd.Timedelta(days=window_days)
        hi = base_date + pd.Timedelta(days=window_days)
        in_window = [
            j for j, d in enumerate(dates)
            if (not pd.isna(d)) and (lo <= d <= hi)
        ]
        # base must be in its own window (it is, by construction)
        local_idx = i  # we'll find this in the window list
        try:
            base_pos = in_window.index(i)
        except ValueError:
            in_window.append(i); base_pos = len(in_window) - 1

        if len(in_window) < min_cluster_size:
            # singleton or below-threshold window → NOVEL by definition
            out_rows.append({
                "cluster_uid": None, "role": "NOVEL", "size": 1,
                "members": json.dumps([str(df.iloc[i].get("document_id", i))]),
                "cohesion": 1.0, "label": "(singleton)"
            })
            continue

        # build fused distance for this window
        D = fused_distance(
            embeddings[in_window],
            [entity_bags[j] for j in in_window],
            [polarity_vals[j] for j in in_window],
            [quantile_vals[j] for j in in_window],
            weights,
        )

        # HDBSCAN — precomputed
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clusterer = hdbscan.HDBSCAN(
                metric="precomputed",
                min_cluster_size=min_cluster_size,
                min_samples=1,
                allow_single_cluster=True,
            )
            labels = clusterer.fit_predict(D)
        base_label = labels[base_pos]
        if base_label == -1:
            # noise → NOVEL
            out_rows.append({
                "cluster_uid": None, "role": "NOVEL", "size": 1,
                "members": json.dumps([str(df.iloc[i].get("document_id", i))]),
                "cohesion": 1.0, "label": "(noise)"
            })
            continue

        # identify cluster members in the window
        member_pos = [k for k, lab in enumerate(labels) if lab == base_label]
        member_global_idx = [in_window[k] for k in member_pos]
        member_doc_ids = [
            str(df.iloc[k].get("document_id", k)) for k in member_global_idx
        ]
        member_dates = [dates[k] for k in member_global_idx]

        # cohesion = 1 - mean intra-cluster fused distance (excluding diagonal)
        sub = D[np.ix_(member_pos, member_pos)]
        if len(member_pos) >= 2:
            triu = sub[np.triu_indices_from(sub, k=1)]
            cohesion = float(1.0 - triu.mean())
        else:
            cohesion = 1.0

        role = classify_role(base_date, member_dates, len(member_pos))
        label_str = auto_label(df.iloc[member_global_idx])

        # dedupe — if this exact cluster was already created for another base
        sig = tuple(sorted(member_doc_ids))
        if sig in seen_signature:
            uid = seen_signature[sig]
        else:
            cluster_uid += 1
            uid = f"c{window_days}d_{cluster_uid:04d}"
            seen_signature[sig] = uid
            cluster_table[uid] = {
                "window_days": window_days,
                "size": len(member_pos),
                "members": member_doc_ids,
                "member_dates": [d.isoformat() if pd.notna(d) else None for d in member_dates],
                "role": role,
                "cohesion": round(cohesion, 4),
                "label": label_str,
                "first_seen": min(member_dates).isoformat(),
                "last_seen": max(member_dates).isoformat(),
                "spread_days": (max(member_dates) - min(member_dates)).days,
            }

        out_rows.append({
            "cluster_uid": uid,
            "role": role,
            "size": len(member_pos),
            "members": json.dumps(member_doc_ids),
            "cohesion": round(cohesion, 4),
            "label": label_str,
        })

    out_df = pd.DataFrame(out_rows).rename(columns={
        "cluster_uid": f"local_cluster_id_{window_days}d",
        "role":        f"local_cluster_role_{window_days}d",
        "size":        f"local_cluster_size_{window_days}d",
        "members":     f"local_cluster_members_{window_days}d",
        "cohesion":    f"local_cluster_cohesion_{window_days}d",
        "label":       f"local_cluster_label_{window_days}d",
    })
    return out_df, cluster_table


# ---------------------------------------------------------------------------
# DRIVER
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input",  required=True, help="Brahmanda-style memory CSV")
    ap.add_argument("--output_dir", default=None,
                    help="defaults to the input file's directory")
    ap.add_argument("--windows", default="3,15,30",
                    help="comma-separated day offsets")
    ap.add_argument("--min_cluster_size", type=int, default=DEFAULT_MIN_CLUSTER)
    ap.add_argument("--weights", default="0.5,0.2,0.1,0.2",
                    help="bert,entity,polarity,quantile")
    args = ap.parse_args()

    in_path = Path(args.input).resolve()
    if not in_path.exists():
        sys.exit(f"ERROR: input not found: {in_path}")
    out_dir = Path(args.output_dir).resolve() if args.output_dir else in_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    year_tag = re.search(r"\d{4}", in_path.stem)
    year_tag = year_tag.group(0) if year_tag else "X"

    windows = tuple(int(w) for w in args.windows.split(","))
    weights = tuple(float(w) for w in args.weights.split(","))
    if len(weights) != 4:
        sys.exit("ERROR: --weights must have exactly 4 numbers")

    print(f"[load] reading {in_path.name}")
    df = pd.read_csv(in_path)
    print(f"[load] {len(df)} rows × {len(df.columns)} columns")

    if "bert_embeddings" not in df.columns:
        sys.exit("ERROR: input has no 'bert_embeddings' column")

    # Parse embeddings
    print(f"[parse] decoding {len(df)} BERT vectors")
    t0 = time.time()
    emb = np.stack([parse_embedding(s) for s in df["bert_embeddings"].tolist()])
    print(f"[parse] embeddings shape={emb.shape}  in {time.time()-t0:.1f}s")

    # Tokenise entity bags
    print(f"[parse] building entity bags")
    bags = [tokenise_entities(r) for _, r in df.iterrows()]
    bag_sizes = pd.Series([len(b) for b in bags])
    print(f"[parse] entity bag size — mean {bag_sizes.mean():.1f}, "
          f"median {bag_sizes.median():.0f}, max {bag_sizes.max()}")

    # Run each window
    pieces = [df.copy()]
    all_clusters: Dict[str, Dict] = {}
    role_summary: Dict[int, Counter] = {}

    for w in windows:
        print(f"\n[cluster] window ±{w}d  (min_size={args.min_cluster_size})")
        t0 = time.time()
        win_df, win_clusters = run_window(
            df, emb, bags, w, args.min_cluster_size, weights
        )
        elapsed = time.time() - t0
        pieces.append(win_df)
        all_clusters[f"{w}d"] = win_clusters
        role_summary[w] = Counter(win_df[f"local_cluster_role_{w}d"])
        print(f"[cluster] window ±{w}d done in {elapsed:.1f}s — "
              f"{len(win_clusters)} unique clusters")

    # Concatenate side-by-side (df + 6 columns per window)
    out_df = pd.concat(pieces, axis=1)

    # Drop the gigantic embedding string before saving CSV
    if "bert_embeddings" in out_df.columns:
        out_df_csv = out_df.drop(columns=["bert_embeddings"])
    else:
        out_df_csv = out_df

    out_csv = out_dir / f"df_{year_tag}_with_local_clusters.csv"
    out_json = out_dir / f"local_clusters_{year_tag}.json"
    out_report = out_dir / f"local_clusters_{year_tag}_report.txt"

    out_df_csv.to_csv(out_csv, index=False)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(all_clusters, f, ensure_ascii=False, indent=2, default=str)

    # ---- Diagnostic report ----
    lines = [
        f"Local cluster report — input: {in_path.name}",
        f"  rows: {len(df)}",
        f"  windows: {list(windows)}",
        f"  min_cluster_size: {args.min_cluster_size}",
        f"  weights (bert/entity/polarity/quantile): {weights}",
        ""
    ]
    for w in windows:
        roles = role_summary[w]
        lines.append(f"--- Window ±{w}d ---")
        lines.append(f"  unique clusters: {len(all_clusters[f'{w}d'])}")
        for role in ("NOVEL", "REPEAT", "CONTINUATION", "REGIME-MARKER"):
            n_role = roles.get(role, 0)
            pct = 100 * n_role / len(df) if len(df) else 0.0
            lines.append(f"    {role:<14}  {n_role:>4}  ({pct:5.1f} %)")
        # cluster size distribution
        if all_clusters[f"{w}d"]:
            sizes = [c["size"] for c in all_clusters[f"{w}d"].values()]
            lines.append(f"  cluster size — mean {np.mean(sizes):.1f}, "
                         f"median {int(np.median(sizes))}, max {max(sizes)}")
            # top-5 clusters by size with their labels
            top = sorted(all_clusters[f"{w}d"].items(),
                         key=lambda kv: -kv[1]["size"])[:5]
            lines.append("  top 5 clusters by size:")
            for uid, c in top:
                lines.append(f"    {uid} (n={c['size']:>2}, "
                             f"{c['spread_days']:>2}d span, role={c['role']:<14}) "
                             f"→ {c['label']}")
        lines.append("")

    report_text = "\n".join(lines)
    out_report.write_text(report_text, encoding="utf-8")

    print("\n" + "=" * 64)
    print(report_text)
    print("=" * 64)
    print(f"\nWrote:\n  {out_csv}\n  {out_json}\n  {out_report}")


if __name__ == "__main__":
    main()
