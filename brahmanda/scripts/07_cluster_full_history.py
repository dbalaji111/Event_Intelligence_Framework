"""
07_cluster_full_history.py
============================
Run local-time clustering across the FULL 23-year master memory in one pass.

Same logic as scripts/a06_base_event_clustering.py (fused distance + HDBSCAN +
role classifier), but reads the master memory directly (with pre-computed
embeddings) instead of per-year CSVs. Windows naturally span year boundaries,
so a story breaking on 2014-12-30 can pull analogues from 2015-01.

Reads
-----
  <memory_dir>/df_major_metadata.csv      ~12k events, 2001-2023, Brahmanda schema
  <memory_dir>/df_major_embeddings.npz    pre-computed BERT 768-dim
                                           keys: 'X' (N,768) + 'ids' (N,) by doc_id

Writes (default: refinitiv-redflags-starter/clusters_full_history/)
-------------------------------------------------------------------
  df_full_2001_2023_with_local_clusters.csv   metadata + cluster columns
  local_clusters_FULL_2001_2023.json          side-table of cluster definitions
  cluster_full_history_report.txt             diagnostic report

NO LLM. NO web. Pure NumPy / pandas / hdbscan. Supercompute-portable.

Run
---
    python 07_cluster_full_history.py \
        --memory_dir "C:\\...\\Brahmanda_v3\\memory" \
        --windows 3,5,10,15,20,30 \
        --min_cluster_size 5

CLI flags
---------
    --memory_dir       absolute path to the memory folder (required)
    --output_dir       defaults to <project>/clusters_full_history/
    --windows          comma-separated day offsets (default: 3,5,10,15,20,30)
    --min_cluster_size HDBSCAN min cluster size (default: 5)
    --weights          fused distance weights bert,entity,polarity,quantile
                       (default: 0.5,0.2,0.1,0.2)
    --limit            optional: process only first N events (for smoke testing)
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
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT_DIR  = PROJECT_ROOT / "clusters_full_history"
DEFAULT_WEIGHTS     = (0.5, 0.2, 0.1, 0.2)   # bert, entity, polarity, quantile
DEFAULT_WINDOWS     = (3, 5, 10, 15, 20, 30)
DEFAULT_MIN_CLUSTER = 5
EMBEDDING_DIM       = 768

ENTITY_COLS = (
    "Entity", "Country Attributes", "Person Attributes",
    "Group Attributes", "Location Attributes",
)
TOKEN_SPLIT = re.compile(r"[,;\n/|]+")


# ---------------------------------------------------------------------------
# PARSERS
# ---------------------------------------------------------------------------
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
            if len(t) > 60:
                continue
            bag.add(t)
    return bag


# ---------------------------------------------------------------------------
# DISTANCES
# ---------------------------------------------------------------------------
def cosine_distance_matrix(X: np.ndarray) -> np.ndarray:
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
    w_b, w_e, w_p, w_q = weights
    D_b = cosine_distance_matrix(emb) / 2.0
    D_e = jaccard_distance_matrix(bags)
    D_p = categorical_distance_matrix(polarity)
    D_q = categorical_distance_matrix(quantile)
    D = w_b * D_b + w_e * D_e + w_p * D_p + w_q * D_q
    np.fill_diagonal(D, 0.0)
    return D.astype(np.float64)


# ---------------------------------------------------------------------------
# CLUSTER ROLE CLASSIFIER
# ---------------------------------------------------------------------------
def classify_role(base_date: pd.Timestamp,
                  member_dates: List[pd.Timestamp],
                  cluster_size: int) -> str:
    if cluster_size <= 1:
        return "NOVEL"
    spread_days = (max(member_dates) - min(member_dates)).days
    deltas = [abs((d - base_date).days) for d in member_dates]
    if max(deltas) <= 2:
        return "REPEAT"
    if cluster_size >= 3 and spread_days >= 7:
        return "REGIME-MARKER"
    return "CONTINUATION"


# ---------------------------------------------------------------------------
# AUTO-LABEL
# ---------------------------------------------------------------------------
_KEYWORD_BLOCKLIST = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "by",
    "with", "from", "is", "are", "was", "were", "has", "have", "had",
    "be", "been", "as", "at", "this", "that", "it", "its", "said",
}


def auto_label(member_rows: pd.DataFrame, max_terms: int = 3) -> str:
    bag_counter: Counter = Counter()
    for _, r in member_rows.iterrows():
        bag_counter.update(tokenise_entities(r))
    top_entities = [w for w, _ in bag_counter.most_common(max_terms)]

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
# MEMORY LOADER — aligns CSV rows with NPZ embeddings via document_id
# ---------------------------------------------------------------------------
def load_memory(memory_dir: Path,
                metadata_filename: str = "df_major_metadata.csv",
                embeddings_filename: str = "df_major_embeddings.npz"
               ) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load metadata CSV + aligned BERT embeddings.
    Returns metadata sorted by date, embeddings array in matching order.

    Auto-detects NPZ key: tries 'X' (Brahmanda master) then 'embeddings'
    (the 2024-2026 corpus convention). Same fallback for ids."""
    csv_path = memory_dir / metadata_filename
    npz_path = memory_dir / embeddings_filename
    if not csv_path.exists():
        sys.exit(f"ERROR: missing {csv_path}")
    if not npz_path.exists():
        sys.exit(f"ERROR: missing {npz_path}")

    print(f"[load] {csv_path.name}  ({csv_path.stat().st_size/1e6:.0f} MB)")
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as e:
        print(f"[load]   C engine failed ({type(e).__name__}); retry with python engine")
        df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")
    print(f"[load]   {len(df)} rows × {len(df.columns)} cols")

    print(f"[load] {npz_path.name}  ({npz_path.stat().st_size/1e6:.0f} MB)")
    z = np.load(npz_path, allow_pickle=True)
    # Auto-detect embedding key: 'X' (master) or 'embeddings' (2024-2026)
    emb_key = next((k for k in ("X", "embeddings", "vectors") if k in z.files), None)
    id_key  = next((k for k in ("ids", "doc_ids", "document_ids") if k in z.files), None)
    if emb_key is None or id_key is None:
        sys.exit(f"ERROR: NPZ keys {z.files} do not contain a recognised "
                 f"embedding+id pair. Expected one of (X|embeddings|vectors) "
                 f"and (ids|doc_ids|document_ids).")
    emb_X = z[emb_key]
    emb_ids = np.array([str(x) for x in z[id_key]])
    print(f"[load]   embeddings: {emb_X.shape}  (key='{emb_key}'),  "
          f"ids: {emb_ids.shape}  (key='{id_key}')")

    # Build lookup: document_id (lowercase) → embedding index
    id_to_idx = {doc_id.lower(): i for i, doc_id in enumerate(emb_ids)}

    # Align embeddings to metadata order (by document_id, case-insensitive)
    aligned = np.zeros((len(df), EMBEDDING_DIM), dtype=np.float32)
    matched, missed = 0, 0
    for ix, doc_id in enumerate(df["document_id"].fillna("").astype(str).tolist()):
        j = id_to_idx.get(doc_id.lower())
        if j is None:
            missed += 1
        else:
            aligned[ix] = emb_X[j]
            matched += 1
    print(f"[load]   embedding alignment: {matched} matched, {missed} missed")

    if missed > matched * 0.10:
        warnings.warn(f"More than 10% of events have no embedding ({missed}/{len(df)})")

    # Sort by date ascending (stable)
    df["_date"] = pd.to_datetime(df["date"], errors="coerce")
    order = df["_date"].argsort(kind="stable").to_numpy()
    df = df.iloc[order].reset_index(drop=True)
    aligned = aligned[order]
    df.drop(columns=["_date"], inplace=True)
    return df, aligned


# ---------------------------------------------------------------------------
# CORE LOOP — one window-size pass over the whole corpus
# ---------------------------------------------------------------------------
def run_window(df: pd.DataFrame, embeddings: np.ndarray,
               entity_bags: List[set],
               window_days: int,
               min_cluster_size: int,
               weights: Tuple[float, float, float, float],
               progress_every: int = 500) -> Tuple[pd.DataFrame, Dict]:
    """For every event, build a local distance matrix in its ±window slice,
    run HDBSCAN, and record the base event's cluster + role + label."""
    n = len(df)
    out_rows: List[Dict] = []
    cluster_table: Dict[str, Dict] = {}
    cluster_uid = 0
    seen_signature: Dict[Tuple, str] = {}

    polarity_vals = (df["Polarity"].fillna("UNK").astype(str).tolist()
                     if "Polarity" in df.columns else ["UNK"] * n)
    quantile_vals = (df["Quantile_Category"].fillna("UNK").astype(str).tolist()
                     if "Quantile_Category" in df.columns else ["UNK"] * n)
    dates = pd.to_datetime(df["date"], errors="coerce").tolist()

    # Pre-build a sorted-date array for fast window slicing via np.searchsorted
    valid_idx = [i for i, d in enumerate(dates) if not pd.isna(d)]
    sorted_idx = sorted(valid_idx, key=lambda i: dates[i])
    sorted_ts  = np.array([dates[i].value for i in sorted_idx], dtype=np.int64)

    t_loop_start = time.time()
    for i in range(n):
        base_date = dates[i]
        if pd.isna(base_date):
            out_rows.append({
                "cluster_uid": None, "role": "NOVEL", "size": 1,
                "members": "[]", "cohesion": 0.0, "label": "(no date)"
            })
            continue

        # Find window indices via binary search on sorted timestamps
        lo_ts = (base_date - pd.Timedelta(days=window_days)).value
        hi_ts = (base_date + pd.Timedelta(days=window_days)).value
        lo = int(np.searchsorted(sorted_ts, lo_ts, side="left"))
        hi = int(np.searchsorted(sorted_ts, hi_ts, side="right"))
        in_window = sorted_idx[lo:hi]
        if i not in in_window:
            in_window.append(i)
        try:
            base_pos = in_window.index(i)
        except ValueError:
            in_window.append(i); base_pos = len(in_window) - 1

        if len(in_window) < min_cluster_size:
            out_rows.append({
                "cluster_uid": None, "role": "NOVEL", "size": 1,
                "members": json.dumps([str(df.iloc[i].get("document_id", i))]),
                "cohesion": 1.0, "label": "(singleton)"
            })
            continue

        # Build fused distance matrix for this window
        D = fused_distance(
            embeddings[in_window],
            [entity_bags[j] for j in in_window],
            [polarity_vals[j] for j in in_window],
            [quantile_vals[j] for j in in_window],
            weights,
        )

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
            out_rows.append({
                "cluster_uid": None, "role": "NOVEL", "size": 1,
                "members": json.dumps([str(df.iloc[i].get("document_id", i))]),
                "cohesion": 1.0, "label": "(noise)"
            })
            continue

        member_pos = [k for k, lab in enumerate(labels) if lab == base_label]
        member_global_idx = [in_window[k] for k in member_pos]
        member_doc_ids = [
            str(df.iloc[k].get("document_id", k)) for k in member_global_idx
        ]
        member_dates = [dates[k] for k in member_global_idx]

        sub = D[np.ix_(member_pos, member_pos)]
        if len(member_pos) >= 2:
            triu = sub[np.triu_indices_from(sub, k=1)]
            cohesion = float(1.0 - triu.mean())
        else:
            cohesion = 1.0

        role = classify_role(base_date, member_dates, len(member_pos))
        label_str = auto_label(df.iloc[member_global_idx])

        sig = tuple(sorted(member_doc_ids))
        if sig in seen_signature:
            uid = seen_signature[sig]
        else:
            cluster_uid += 1
            uid = f"c{window_days}d_{cluster_uid:05d}"
            seen_signature[sig] = uid
            cluster_table[uid] = {
                "window_days": window_days,
                "size": len(member_pos),
                "members": member_doc_ids,
                "member_dates": [d.isoformat() if pd.notna(d) else None
                                 for d in member_dates],
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

        if (i + 1) % progress_every == 0:
            elapsed = time.time() - t_loop_start
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate / 60.0
            print(f"  [w={window_days}d] {i+1:>5}/{n}  "
                  f"({rate:5.0f} ev/s · ETA {eta:5.1f} min · "
                  f"clusters so far: {len(cluster_table):>5})", flush=True)

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
    ap.add_argument("--memory_dir", required=True,
                    help="absolute path to memory folder containing "
                         "the metadata CSV + embeddings NPZ")
    ap.add_argument("--metadata_filename", default="df_major_metadata.csv",
                    help="name of the metadata CSV inside --memory_dir "
                         "(default: df_major_metadata.csv)")
    ap.add_argument("--embeddings_filename", default="df_major_embeddings.npz",
                    help="name of the embeddings NPZ inside --memory_dir "
                         "(default: df_major_embeddings.npz)")
    ap.add_argument("--output_tag", default="FULL_2001_2023",
                    help="tag inserted into output filenames "
                         "(e.g. 2024_2026 -> df_full_2024_2026_with_local_clusters.csv)")
    ap.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--windows", default=",".join(str(w) for w in DEFAULT_WINDOWS))
    ap.add_argument("--min_cluster_size", type=int, default=DEFAULT_MIN_CLUSTER)
    ap.add_argument("--weights", default=",".join(str(w) for w in DEFAULT_WEIGHTS),
                    help="bert,entity,polarity,quantile")
    ap.add_argument("--limit", type=int, default=None,
                    help="optional: process only first N events (smoke test)")
    args = ap.parse_args()

    memory_dir = Path(args.memory_dir).resolve()
    out_dir    = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    windows = tuple(int(w) for w in args.windows.split(","))
    weights = tuple(float(w) for w in args.weights.split(","))
    if len(weights) != 4:
        sys.exit("ERROR: --weights must have exactly 4 numbers")

    df, embeddings = load_memory(memory_dir,
                                 metadata_filename=args.metadata_filename,
                                 embeddings_filename=args.embeddings_filename)
    if args.limit:
        df = df.head(args.limit).copy()
        embeddings = embeddings[:args.limit]
        print(f"[limit] reduced to first {args.limit} rows for smoke test")

    print(f"[parse] building entity bags for {len(df)} events")
    bags = [tokenise_entities(r) for _, r in df.iterrows()]
    bag_sizes = pd.Series([len(b) for b in bags])
    print(f"[parse]   entity bags: mean {bag_sizes.mean():.1f}, "
          f"median {bag_sizes.median():.0f}, max {bag_sizes.max()}")

    pieces = [df.copy()]
    all_clusters: Dict[str, Dict] = {}
    role_summary: Dict[int, Counter] = {}

    for w in windows:
        print(f"\n[cluster] window ±{w}d  (min_size={args.min_cluster_size})")
        t0 = time.time()
        win_df, win_clusters = run_window(
            df, embeddings, bags, w, args.min_cluster_size, weights
        )
        elapsed = time.time() - t0
        pieces.append(win_df)
        all_clusters[f"{w}d"] = win_clusters
        role_summary[w] = Counter(win_df[f"local_cluster_role_{w}d"])
        print(f"[cluster] window ±{w}d done in {elapsed/60:.1f} min — "
              f"{len(win_clusters)} unique clusters")

    out_df = pd.concat(pieces, axis=1)
    if "bert_embeddings" in out_df.columns:
        out_df = out_df.drop(columns=["bert_embeddings"])

    tag = args.output_tag
    out_csv    = out_dir / f"df_full_{tag}_with_local_clusters.csv"
    out_json   = out_dir / f"local_clusters_{tag}.json"
    out_report = out_dir / f"cluster_{tag}_report.txt"

    out_df.to_csv(out_csv, index=False)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(all_clusters, f, ensure_ascii=False, indent=2, default=str)

    # Diagnostic report
    lines = [
        f"Full-history local-cluster report",
        f"  source: {memory_dir}",
        f"  rows: {len(df)}",
        f"  date range: {df['date'].min()} - {df['date'].max()}",
        f"  windows: {list(windows)}",
        f"  min_cluster_size: {args.min_cluster_size}",
        f"  weights (bert/entity/polarity/quantile): {weights}",
        ""
    ]
    for w in windows:
        roles = role_summary[w]
        lines.append(f"--- Window +/-{w}d ---")
        lines.append(f"  unique clusters: {len(all_clusters[f'{w}d'])}")
        for role in ("NOVEL", "REPEAT", "CONTINUATION", "REGIME-MARKER"):
            n_role = roles.get(role, 0)
            pct = 100 * n_role / len(df) if len(df) else 0.0
            lines.append(f"    {role:<14}  {n_role:>5}  ({pct:5.1f} %)")
        if all_clusters[f"{w}d"]:
            sizes = [c["size"] for c in all_clusters[f"{w}d"].values()]
            lines.append(f"  cluster size: mean {np.mean(sizes):.1f}, "
                         f"median {int(np.median(sizes))}, max {max(sizes)}")
            top = sorted(all_clusters[f"{w}d"].items(),
                         key=lambda kv: -kv[1]["size"])[:8]
            lines.append("  top 8 clusters by size:")
            for uid, c in top:
                lines.append(f"    {uid} (n={c['size']:>3}, "
                             f"{c['spread_days']:>3}d span, role={c['role']:<14}) "
                             f"-> {c['label']}")
        lines.append("")

    report_text = "\n".join(lines)
    out_report.write_text(report_text, encoding="utf-8")

    print("\n" + "=" * 72)
    print(report_text)
    print("=" * 72)
    print(f"\nWrote:\n  {out_csv}\n  {out_json}\n  {out_report}")


if __name__ == "__main__":
    main()
