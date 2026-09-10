"""
13_visualise_clusters_3d.py
============================
Interactive 3D visualisation of cluster centroids — for validating that the
clustering produced sensible structure in BERT embedding space.

What you'll see
---------------
Each point = one cluster's mean BERT centroid, projected from 768-dim into
3D via UMAP (preferred) or PCA (fallback). Default colouring is by year of
the cluster's first_seen date — so if the clustering is healthy, years should
form visible regional bands rather than random mixing. Point size scales with
cluster size (number of member events) so the dominant stories pop visually.

Hover any point: cluster_id + first_seen-last_seen dates + size + role + label.

Reads
-----
  --cluster_index   any cluster_memory*.parquet from script 08

Writes (default: alongside the input parquet)
---------------------------------------------
  clusters_3d_<tag>_w<W>.html

Run
---
    python 13_visualise_clusters_3d.py \
        --cluster_index clusters_full_history/cluster_memory_FULL_2001_2023.parquet \
        --window 15

CLI flags
---------
  --cluster_index   path to cluster_memory*.parquet (required)
  --window          window_days to filter (default 15; use 0 to plot ALL windows)
  --color_by        year (default) | role | size | window
  --reducer         umap (default if installed) | pca | tsne
  --max_points      cap on points to plot (default 8000; UMAP slows for >10k)
  --output_dir      defaults to alongside the input parquet
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
except ImportError:
    sys.exit("ERROR: plotly required.  pip install plotly")


# ---------------------------------------------------------------------------
# Dimensionality reduction
# ---------------------------------------------------------------------------
def reduce_to_3d(centroids: np.ndarray, method: str) -> np.ndarray:
    """Reduce (N,768) → (N,3) with the chosen method. Returns float32."""
    method = method.lower()

    if method == "umap":
        try:
            import umap
        except ImportError:
            print("[reduce] umap-learn not installed, falling back to PCA")
            method = "pca"
        else:
            print(f"[reduce] UMAP n_components=3, n_neighbors=15, min_dist=0.1")
            t0 = time.time()
            reducer = umap.UMAP(
                n_components=3, n_neighbors=15, min_dist=0.1,
                metric="cosine", random_state=42,
            )
            out = reducer.fit_transform(centroids)
            print(f"[reduce] UMAP done in {time.time()-t0:.1f}s")
            return out.astype(np.float32)

    if method == "tsne":
        try:
            from sklearn.manifold import TSNE
            print(f"[reduce] t-SNE n_components=3, perplexity=30")
            t0 = time.time()
            out = TSNE(n_components=3, perplexity=30, init="pca",
                       random_state=42, metric="cosine").fit_transform(centroids)
            print(f"[reduce] t-SNE done in {time.time()-t0:.1f}s")
            return out.astype(np.float32)
        except ImportError:
            print("[reduce] sklearn missing, falling back to PCA")
            method = "pca"

    # PCA fallback
    print(f"[reduce] PCA n_components=3")
    t0 = time.time()
    X = centroids - centroids.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    out = (X @ Vt[:3].T).astype(np.float32)
    print(f"[reduce] PCA done in {time.time()-t0:.1f}s")
    return out


# ---------------------------------------------------------------------------
# Hover text builder
# ---------------------------------------------------------------------------
def hover_text(row: pd.Series) -> str:
    return (
        f"<b>{row['cluster_id']}</b><br>"
        f"{row['first_seen'][:10]} → {row['last_seen'][:10]} "
        f"({row.get('spread_days','?')}d)<br>"
        f"size: <b>{row['size']}</b>  ·  role: <b>{row['role']}</b>  ·  "
        f"window: ±{row['window_days']}d<br>"
        f"<b>topic:</b> <i>{row['label']}</i>"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cluster_index", required=True)
    ap.add_argument("--window", type=int, default=15,
                    help="window_days filter (default 15; 0 = ALL windows)")
    ap.add_argument("--color_by", choices=["year", "role", "size", "window"],
                    default="year")
    ap.add_argument("--reducer", choices=["umap", "pca", "tsne"], default="umap")
    ap.add_argument("--max_points", type=int, default=8000,
                    help="cap on points (UMAP slow above 10k); default 8000")
    ap.add_argument("--output_dir", default=None)
    args = ap.parse_args()

    pq_path = Path(args.cluster_index).resolve()
    out_dir = Path(args.output_dir).resolve() if args.output_dir else pq_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {pq_path.name}")
    df = pd.read_parquet(pq_path,
                         columns=["cluster_id", "window_days", "size", "role",
                                  "label", "first_seen", "last_seen",
                                  "spread_days", "centroid"])
    print(f"[load]   {len(df)} clusters total")

    if args.window > 0:
        df = df[df["window_days"] == args.window].copy()
        print(f"[filter] {len(df)} clusters at ±{args.window}d window")
    if df.empty:
        sys.exit(f"No clusters at window={args.window}")

    # Drop dead-zero centroids if any
    df["_norm"] = df["centroid"].apply(lambda v: float(np.linalg.norm(np.asarray(v, dtype=np.float32))))
    n_zero = (df["_norm"] < 1e-6).sum()
    if n_zero:
        print(f"[filter] dropping {n_zero} zero-vector centroids")
        df = df[df["_norm"] >= 1e-6].copy()

    # Cap rows for tractable UMAP
    if len(df) > args.max_points:
        print(f"[filter] sampling {args.max_points} of {len(df)} for tractable reducer")
        df = df.sample(args.max_points, random_state=42).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    # Stack centroids
    centroids = np.stack([np.asarray(v, dtype=np.float32) for v in df["centroid"].tolist()])
    print(f"[stack]  centroids shape={centroids.shape}")

    # Reduce
    coords = reduce_to_3d(centroids, args.reducer)
    df["x"], df["y"], df["z"] = coords[:, 0], coords[:, 1], coords[:, 2]

    # Year for colouring
    df["year"] = pd.to_datetime(df["first_seen"]).dt.year

    # Build colour values
    if args.color_by == "year":
        color = df["year"]
        cscale = "Viridis"
        ctitle = "year"
        colorbar = dict(title="year", thickness=14)
        showscale = True
    elif args.color_by == "size":
        color = np.log1p(df["size"])
        cscale = "Plasma"
        ctitle = "log(size)"
        colorbar = dict(title="log(size)", thickness=14)
        showscale = True
    elif args.color_by == "window":
        color = df["window_days"]
        cscale = "Turbo"
        ctitle = "window_days"
        colorbar = dict(title="window (d)", thickness=14)
        showscale = True
    else:  # role
        role_palette = {
            "REGIME-MARKER": "#08519c",
            "CONTINUATION":  "#2171b5",
            "REPEAT":        "#d68910",
            "NOVEL":         "#9aa5b1",
        }
        color = df["role"].map(lambda r: role_palette.get(r, "#888"))
        cscale = None; ctitle = "role"; colorbar = None; showscale = False

    # Marker size scales with cluster size
    sizes = np.sqrt(df["size"].astype(float))
    sizes = 4 + (sizes / sizes.max() * 10)   # range ~4-14

    # Build figure
    print(f"[plot]  building 3D scatter ({len(df)} points)")
    hovers = df.apply(hover_text, axis=1)
    marker_cfg = dict(
        size=sizes, opacity=0.78,
        line=dict(color="rgba(30,30,30,0.4)", width=0.4),
    )
    if showscale:
        marker_cfg["color"] = color
        marker_cfg["colorscale"] = cscale
        marker_cfg["colorbar"] = colorbar
    else:
        marker_cfg["color"] = list(color)

    fig = go.Figure(data=[go.Scatter3d(
        x=df["x"], y=df["y"], z=df["z"],
        mode="markers",
        marker=marker_cfg,
        hovertext=hovers, hoverinfo="text",
        name="clusters",
    )])

    pq_stem = pq_path.stem.replace("cluster_memory_", "")
    title_window = "ALL windows" if args.window == 0 else f"±{args.window}d window"
    fig.update_layout(
        title=dict(
            text=(f"<b>3D cluster centroids — {pq_stem} — {title_window}</b><br>"
                  f"<sub>{len(df)} clusters · {args.reducer.upper()} 3D projection · "
                  f"colour = {ctitle} · size = √(member count) · "
                  f"hover for cluster details</sub>"),
            x=0.02, y=0.97, font=dict(size=14, color="#212529"),
        ),
        scene=dict(
            xaxis=dict(title=f"{args.reducer}-1", showgrid=True, gridcolor="#dee2e6",
                       backgroundcolor="#f7f9fc"),
            yaxis=dict(title=f"{args.reducer}-2", showgrid=True, gridcolor="#dee2e6",
                       backgroundcolor="#f7f9fc"),
            zaxis=dict(title=f"{args.reducer}-3", showgrid=True, gridcolor="#dee2e6",
                       backgroundcolor="#f7f9fc"),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.6, y=-1.6, z=1.0)),
        ),
        paper_bgcolor="white",
        height=800,
        margin=dict(l=20, r=20, t=80, b=20),
        hoverlabel=dict(bgcolor="white", bordercolor="#888",
                        font=dict(family="Calibri,Arial", size=11)),
    )

    suffix = f"w{args.window}" if args.window > 0 else "all"
    out_html = out_dir / f"clusters_3d_{pq_stem}_{suffix}_by_{args.color_by}.html"
    fig.write_html(str(out_html), include_plotlyjs="cdn",
                   full_html=True, auto_open=False)
    print(f"[save]  {out_html}  ({out_html.stat().st_size/1e3:.1f} kB)")

    # Quick sanity stats
    print(f"\n--- Summary ---")
    print(f"  clusters plotted: {len(df)}")
    print(f"  year span: {df['year'].min()} → {df['year'].max()}")
    print(f"  role distribution:")
    for r, n in df["role"].value_counts().items():
        print(f"    {r:<14} {n:>5}")


if __name__ == "__main__":
    main()
