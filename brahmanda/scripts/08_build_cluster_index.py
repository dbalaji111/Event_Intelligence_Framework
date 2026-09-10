"""
08_build_cluster_index.py
==========================
Build the cluster retrieval index from the full-history clusters produced by
07_cluster_full_history.py.

For each cluster, computes the six-part embedding bundle the EventCast PDF
calls for in Stages 1-2:

  1. Centroid BERT embedding   - mean of member event vectors (768-dim)
  2. Aggregated entity bag      - {entity: count} across all members
  3. Polarity / event-type dist - {label: count}
  4. Quantile signature         - {annual: cat, monthly: cat, cum_7d/15d/30d: cat}
                                  computed over the cluster's date span
  5. Propagation profile rho(t) - daily member-event count, normalised to
                                  21-point unit-time grid (event-time t in [0,1])
  6. Realised return path       - daily log-return series from first_seen to
                                  last_seen + post_window_days (default 15)

Reads
-----
  --clusters_json   clusters_full_history/local_clusters_FULL_2001_2023.json
  --memory_dir      memory/ folder with df_major_metadata.csv + df_major_embeddings.npz
  --price_xlsx      oil_price_data/spot_and_spreads_data.xlsx
  --post_days       post-event days for realised path tail (default 15)

Writes (default: clusters_full_history/)
----------------------------------------
  cluster_memory.parquet         one row per cluster, ready for retrieval
  cluster_memory_browse.csv      slim browse-friendly index (no big arrays)

NO LLM. NO web. Pure NumPy / pandas / pyarrow.
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

EMBEDDING_DIM = 768
PROFILE_GRID  = 21    # number of points in the normalised propagation profile
TOKEN_SPLIT   = re.compile(r"[,;\n/|]+")
ENTITY_COLS   = (
    "Entity", "Country Attributes", "Person Attributes",
    "Group Attributes", "Location Attributes",
)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_memory(memory_dir: Path,
                metadata_filename: str = "df_major_metadata.csv",
                embeddings_filename: str = "df_major_embeddings.npz"
               ) -> Tuple[pd.DataFrame, Dict[str, int], np.ndarray]:
    """Returns (metadata_df, doc_id_to_index_lookup, aligned_embeddings).
    Auto-detects NPZ key: 'X' (master) or 'embeddings' (2024-2026)."""
    csv_path = memory_dir / metadata_filename
    npz_path = memory_dir / embeddings_filename
    print(f"[load] {csv_path.name}  ({csv_path.stat().st_size/1e6:.0f} MB)")
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception:
        df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")
    print(f"[load]   {len(df)} rows")
    print(f"[load] {npz_path.name}  ({npz_path.stat().st_size/1e6:.0f} MB)")
    z = np.load(npz_path, allow_pickle=True)
    emb_key = next((k for k in ("X", "embeddings", "vectors") if k in z.files), None)
    id_key  = next((k for k in ("ids", "doc_ids", "document_ids") if k in z.files), None)
    if emb_key is None or id_key is None:
        sys.exit(f"ERROR: NPZ keys {z.files} unrecognised; expected (X|embeddings|vectors) "
                 f"and (ids|doc_ids|document_ids).")
    emb_X   = z[emb_key]
    emb_ids = np.array([str(x).lower() for x in z[id_key]])
    print(f"[load]   embeddings shape={emb_X.shape}  (key='{emb_key}')")

    id_to_idx = {doc: i for i, doc in enumerate(emb_ids)}
    df["_doc_lc"] = df["document_id"].fillna("").astype(str).str.lower()
    return df, id_to_idx, emb_X


def load_price(price_path: Path) -> pd.DataFrame:
    print(f"[load] {price_path.name}")
    if price_path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(price_path)
    else:
        df = pd.read_csv(price_path)
    date_col = next(c for c in df.columns
                    if c.lower() in ("name", "date", "datetime"))
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.rename(columns={
        date_col: "date",
        "Crude Oil-WTI Spot Cushing U$/BBL": "wti",
        "Crude Oil Brent US M1 U$/BBL": "brent",
    })
    df = df[["date", "wti", "brent"]].dropna(subset=["date"]).sort_values("date")
    df = df.set_index("date")
    print(f"[load]   {len(df)} daily rows  "
          f"({df.index.min().date()} -> {df.index.max().date()})")
    return df


# ---------------------------------------------------------------------------
# Per-cluster feature builders
# ---------------------------------------------------------------------------
def tokenise_entities_row(row: pd.Series) -> List[str]:
    out = []
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
            out.append(t)
    return out


def aggregate_entities(member_rows: pd.DataFrame, top_n: int = 30) -> Dict[str, int]:
    counter: Counter = Counter()
    for _, r in member_rows.iterrows():
        counter.update(tokenise_entities_row(r))
    return dict(counter.most_common(top_n))


def distribution(values: pd.Series) -> Dict[str, int]:
    """Counter-style dict, sorted by frequency descending."""
    s = values.fillna("UNK").astype(str)
    return dict(Counter(s.tolist()).most_common())


def quantile_signature(price_index: pd.DataFrame,
                       first: pd.Timestamp,
                       last: pd.Timestamp) -> Dict[str, str]:
    """Three quantile bands during the cluster's date span:
       - annual:  band of mean WTI return over the cluster, vs that calendar year
       - monthly: same vs that calendar month
       - cum_7d:  band of cumulative WTI % change over the cluster, vs all 7d windows that year
       - cum_15d, cum_30d: same for 15/30d
       Each band is one of: 0-5%, 5-20%, 20-40%, 40-60%, 60-80%, 80-95%, 95-100%
    """
    bands = ["0-5%", "5-20%", "20-40%", "40-60%", "60-80%", "80-95%", "95-100%"]
    cuts  = [0.05, 0.20, 0.40, 0.60, 0.80, 0.95]

    def to_band(v: float, ref: pd.Series) -> str:
        if pd.isna(v) or ref.empty:
            return "UNK"
        # Find which quantile bin v falls in within ref's distribution
        try:
            r = (ref < v).mean()
        except Exception:
            return "UNK"
        for c, b in zip(cuts, bands):
            if r < c:
                return b
        return bands[-1]

    sig = {"annual": "UNK", "monthly": "UNK",
           "cum_7d": "UNK", "cum_15d": "UNK", "cum_30d": "UNK"}
    if "wti" not in price_index.columns or pd.isna(first) or pd.isna(last):
        return sig

    year = first.year
    month = first.month
    yr_slice  = price_index[price_index.index.year == year]["wti"].dropna()
    mo_slice  = price_index[(price_index.index.year == year) &
                            (price_index.index.month == month)]["wti"].dropna()
    if yr_slice.empty:
        return sig

    yr_returns = yr_slice.pct_change().dropna()
    span_slice = price_index.loc[first:last]["wti"].dropna()
    if span_slice.empty:
        return sig

    span_mean_ret = span_slice.pct_change().dropna().mean()
    sig["annual"]  = to_band(span_mean_ret, yr_returns)
    if not mo_slice.empty:
        mo_returns = mo_slice.pct_change().dropna()
        sig["monthly"] = to_band(span_mean_ret, mo_returns)

    # Cumulative bands: compute cluster's cumulative move over W days, vs
    # all rolling W-day cumulative moves in the cluster's year.
    for W, key in ((7, "cum_7d"), (15, "cum_15d"), (30, "cum_30d")):
        roll = yr_slice.pct_change().rolling(W).sum().dropna()
        # cluster's actual W-day cumulative move starting at first
        end_target = first + pd.Timedelta(days=W)
        cluster_slice = price_index.loc[first:end_target]["wti"].dropna()
        if len(cluster_slice) >= 2 and not roll.empty:
            cum_change = (cluster_slice.iloc[-1] / cluster_slice.iloc[0]) - 1.0
            sig[key] = to_band(cum_change, roll)
    return sig


def propagation_profile(member_dates: List[pd.Timestamp],
                        first: pd.Timestamp, last: pd.Timestamp,
                        n_bins: int = PROFILE_GRID) -> List[float]:
    """Daily count of member events, normalised to a fixed-length unit-time grid."""
    if pd.isna(first) or pd.isna(last) or len(member_dates) == 0:
        return [0.0] * n_bins
    span_days = max((last - first).days, 1)
    counts = np.zeros(span_days + 1, dtype=np.float32)
    for d in member_dates:
        if pd.isna(d):
            continue
        offset = (d - first).days
        if 0 <= offset < counts.size:
            counts[offset] += 1.0
    # Resample to n_bins via linear interpolation
    if counts.size == n_bins:
        out = counts
    else:
        x_old = np.linspace(0.0, 1.0, counts.size)
        x_new = np.linspace(0.0, 1.0, n_bins)
        out = np.interp(x_new, x_old, counts)
    s = out.sum()
    if s > 0:
        out = out / s
    return [float(x) for x in out]


def realised_return_path(price_index: pd.DataFrame,
                         first: pd.Timestamp,
                         last: pd.Timestamp,
                         post_days: int = 15) -> List[float]:
    """Daily log-returns from first_seen to last_seen + post_days, on WTI."""
    if "wti" not in price_index.columns or pd.isna(first) or pd.isna(last):
        return []
    end = last + pd.Timedelta(days=post_days)
    slc = price_index.loc[first:end]["wti"].dropna()
    if len(slc) < 2:
        return []
    log_ret = np.log(slc / slc.shift(1)).dropna()
    return [float(x) for x in log_ret.tolist()]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clusters_json", required=True,
                    help="local_clusters_FULL_2001_2023.json from script 07")
    ap.add_argument("--memory_dir", required=True,
                    help="folder containing the metadata CSV + embeddings NPZ")
    ap.add_argument("--metadata_filename", default="df_major_metadata.csv")
    ap.add_argument("--embeddings_filename", default="df_major_embeddings.npz")
    ap.add_argument("--output_tag", default=None,
                    help="tag inserted into output filenames; "
                         "defaults to whatever appears between 'local_clusters_' "
                         "and '.json' in --clusters_json")
    ap.add_argument("--price_xlsx", required=True,
                    help="daily oil price file (xlsx with WTI + Brent columns)")
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--post_days", type=int, default=15)
    ap.add_argument("--limit", type=int, default=None,
                    help="optional: only process first N clusters (smoke test)")
    args = ap.parse_args()

    cj_path     = Path(args.clusters_json).resolve()
    memory_dir  = Path(args.memory_dir).resolve()
    price_path  = Path(args.price_xlsx).resolve()
    out_dir     = Path(args.output_dir).resolve() if args.output_dir else cj_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {cj_path.name}")
    with cj_path.open(encoding="utf-8") as f:
        side = json.load(f)
    flat: Dict[str, dict] = {}
    for w_key, w_clusters in side.items():
        for cid, info in w_clusters.items():
            flat[cid] = info
    print(f"[load]   {len(flat)} clusters across all windows")

    df, id_to_idx, embeddings = load_memory(
        memory_dir,
        metadata_filename=args.metadata_filename,
        embeddings_filename=args.embeddings_filename,
    )
    price = load_price(price_path)

    # Build a fast doc_id -> row-index lookup on metadata
    print(f"[index] indexing {len(df)} metadata rows by document_id")
    df_doc_to_row = {doc: i for i, doc in enumerate(df["_doc_lc"].tolist())}

    rows: List[Dict[str, Any]] = []
    cluster_ids = list(flat.keys())
    if args.limit:
        cluster_ids = cluster_ids[: args.limit]
        print(f"[limit] reduced to first {args.limit} clusters")

    print(f"[build] computing 6-part bundle for {len(cluster_ids)} clusters")
    t0 = time.time()
    for n_done, cid in enumerate(cluster_ids):
        info = flat[cid]
        member_doc_ids = [m.lower() for m in info.get("members", [])]
        member_rows_idx = [df_doc_to_row.get(m) for m in member_doc_ids]
        member_rows_idx = [i for i in member_rows_idx if i is not None]
        if not member_rows_idx:
            continue
        member_rows = df.iloc[member_rows_idx]

        # Centroid
        emb_idx = [id_to_idx.get(m) for m in member_doc_ids]
        emb_idx = [i for i in emb_idx if i is not None]
        if emb_idx:
            centroid = embeddings[emb_idx].mean(axis=0).astype(np.float32)
        else:
            centroid = np.zeros(EMBEDDING_DIM, dtype=np.float32)

        # Aggregations
        ent_bag       = aggregate_entities(member_rows)
        polarity_dist = distribution(member_rows.get("Polarity", pd.Series([], dtype=str)))
        evtype_dist   = distribution(member_rows.get("Event_type",
                                     member_rows.get("Event type", pd.Series([], dtype=str))))

        first = pd.to_datetime(info["first_seen"])
        last  = pd.to_datetime(info["last_seen"])

        qtl_sig = quantile_signature(price, first, last)
        member_dates = [pd.to_datetime(d) for d in info.get("member_dates", [])]
        prof = propagation_profile(member_dates, first, last)
        ret_path = realised_return_path(price, first, last, post_days=args.post_days)

        rows.append({
            "cluster_id":        cid,
            "window_days":       info.get("window_days"),
            "size":              info.get("size"),
            "role":              info.get("role"),
            "label":             info.get("label"),
            "first_seen":        first.isoformat(),
            "last_seen":         last.isoformat(),
            "spread_days":       info.get("spread_days"),
            "cohesion":          info.get("cohesion"),
            "n_members":         len(member_doc_ids),
            # The six-part bundle
            "centroid":          centroid.tolist(),
            "entity_bag":        json.dumps(ent_bag, ensure_ascii=False),
            "polarity_dist":     json.dumps(polarity_dist, ensure_ascii=False),
            "evtype_dist":       json.dumps(evtype_dist, ensure_ascii=False),
            "quantile_sig":      json.dumps(qtl_sig, ensure_ascii=False),
            "propagation_prof":  json.dumps(prof),
            "realised_path":     json.dumps(ret_path),
            "realised_path_len": len(ret_path),
            "members":           json.dumps(member_doc_ids, ensure_ascii=False),
        })

        if (n_done + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (n_done + 1) / elapsed
            eta = (len(cluster_ids) - n_done - 1) / rate / 60
            print(f"  [{n_done+1:>5}/{len(cluster_ids)}] "
                  f"{rate:5.0f} clusters/s  ETA {eta:5.1f} min", flush=True)

    print(f"[build] done in {(time.time()-t0)/60:.1f} min, "
          f"{len(rows)} rows built")

    out_df = pd.DataFrame(rows)

    # Determine output tag: explicit --output_tag wins, else parse from filename
    if args.output_tag:
        tag = args.output_tag
    else:
        m = re.search(r"local_clusters_(.+?)\.json$", cj_path.name)
        tag = m.group(1) if m else "FULL"
    pq_name = f"cluster_memory_{tag}.parquet"
    csv_name = f"cluster_memory_{tag}_browse.csv"

    # Parquet (with embeddings as Python lists -> stored efficiently by pyarrow)
    out_pq = out_dir / pq_name
    try:
        out_df.to_parquet(out_pq, index=False, engine="pyarrow",
                          compression="snappy")
    except ImportError:
        out_pq = out_dir / pq_name.replace(".parquet", ".pkl")
        out_df.to_pickle(out_pq)
        print(f"[save] pyarrow not available; saved as pickle instead")

    # Slim CSV for human browsing
    browse_cols = ["cluster_id", "window_days", "size", "role", "label",
                   "first_seen", "last_seen", "spread_days", "cohesion",
                   "n_members", "realised_path_len", "quantile_sig"]
    out_csv = out_dir / csv_name
    out_df[browse_cols].to_csv(out_csv, index=False)

    print(f"\n[save] {out_pq}  ({out_pq.stat().st_size/1e6:.1f} MB)")
    print(f"[save] {out_csv}  ({out_csv.stat().st_size/1e6:.1f} MB)")

    # Brief stats
    print(f"\n--- Cluster index summary ---")
    print(f"  total clusters: {len(out_df)}")
    print(f"  by window:")
    for w, n in out_df["window_days"].value_counts().sort_index().items():
        print(f"    {w}d: {n:>5}")
    print(f"  by role:")
    for r, n in out_df["role"].value_counts().items():
        print(f"    {r:<14} {n:>5}")
    avg_path = out_df[out_df["realised_path_len"] > 0]["realised_path_len"].mean()
    print(f"  realised_path mean length: {avg_path:.1f} days")


if __name__ == "__main__":
    main()
