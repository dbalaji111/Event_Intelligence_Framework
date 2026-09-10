"""
06b_visualise_local_clusters.py
================================
Interactive 3-layer visualisation of the local clusters produced by
06_base_event_clustering.py.

Layout (one HTML file per window):

  Row 1  Daily WTI + Brent price line for the year, with each event
         plotted as a dot at (date, price). Dot colour = cluster.
         Hover on any dot shows the headline / entities / polarity.

  Row 2  Cluster Gantt — every cluster is a bar from first_seen to
         last_seen, on a vertical slot computed via greedy interval
         scheduling so overlapping clusters never share a row.
         Bar colour = cluster (matches Row 1 dots).

Inputs
------
  --csv      df_YYYY_with_local_clusters.csv          (from script 06)
  --json     local_clusters_YYYY.json                 (from script 06)
  --price    spot_and_spreads_data.xlsx               (daily WTI + Brent)
  --year     YYYY (filter price data to this year)

Outputs
-------
  local_clusters_YYYY_explorer_{W}d.html            one per window

Run
---
    python 06b_visualise_local_clusters.py \
        --csv  /path/df_2023_with_local_clusters.csv \
        --json /path/local_clusters_2023.json \
        --price /path/spot_and_spreads_data.xlsx \
        --year 2023
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    sys.exit("ERROR: plotly not installed.  pip install plotly openpyxl")


# ---------------------------------------------------------------------------
# Palette — distinct colours for many clusters; muted greys for NOVEL/noise
# ---------------------------------------------------------------------------
QUAL_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#17becf", "#bcbd22", "#7f7f7f",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5", "#c49c94",
    "#f7b6d2", "#9edae5", "#dbdb8d", "#c7c7c7",
    "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
    "#5254a3", "#8ca252", "#bd9e39", "#ad494a", "#a55194",
    "#6b6ecf", "#b5cf6b", "#e7ba52", "#d6616b", "#ce6dbd",
    "#e7969c", "#9c9ede", "#cedb9c", "#e7cb94", "#a3a3a3",
]
NOVEL_COLOR  = "#cfd6df"   # very pale grey
NOISE_COLOR  = "#cfd6df"
REPEAT_COLOR = "#e5dada"   # pale brown-grey


def color_for_cluster(cid: str | None, role: str, idx: int) -> str:
    """NOVEL / noise / singleton get muted grey; everything else cycles
    through the qualitative palette by stable index."""
    if cid is None or role in ("NOVEL", "(noise)", "(singleton)"):
        return NOVEL_COLOR
    if role == "REPEAT":
        return REPEAT_COLOR
    return QUAL_PALETTE[idx % len(QUAL_PALETTE)]


# ---------------------------------------------------------------------------
# Greedy interval-coloring for slot assignment  (overlapping clusters → diff slots)
# ---------------------------------------------------------------------------
def assign_slots(intervals: List[Tuple[str, pd.Timestamp, pd.Timestamp]]
                 ) -> Dict[str, int]:
    intervals_sorted = sorted(intervals, key=lambda t: t[1])
    slot_of: Dict[str, int] = {}
    slot_end: List[pd.Timestamp] = []
    for cid, start, end in intervals_sorted:
        placed = False
        for s, e in enumerate(slot_end):
            if start > e:
                slot_end[s] = end
                slot_of[cid] = s
                placed = True
                break
        if not placed:
            slot_of[cid] = len(slot_end)
            slot_end.append(end)
    return slot_of


# ---------------------------------------------------------------------------
# Hover-text
# ---------------------------------------------------------------------------
def short(s: str | None, n: int = 220) -> str:
    if not isinstance(s, str):
        return ""
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= n else s[:n] + "…"


def event_hover(row: pd.Series, cluster_label: str, role: str, cid: str | None) -> str:
    text = short(row.get("cleaned_text") or row.get("text") or "")
    entities = short(row.get("Entity", ""), 100)
    primary = short(row.get("Primary Event", ""), 80)
    polarity = row.get("Polarity", "")
    parts = [
        f"<b>{row.get('document_id', '?')}</b>",
        f"<i>{row.get('date', '')}</i>",
        f"<b>Cluster:</b> {cid or 'NONE'} &nbsp;·&nbsp; <b>Role:</b> {role}",
        f"<b>Label:</b> {cluster_label}",
        f"<b>Primary event:</b> {primary}",
        f"<b>Entities:</b> {entities}",
        f"<b>Polarity:</b> {polarity}",
        f"<br>{text}",
    ]
    return "<br>".join(parts)


def cluster_hover(cid: str, info: dict) -> str:
    return (
        f"<b>{cid}</b> &nbsp;·&nbsp; {info['role']}"
        f"<br><b>Label:</b> {info['label']}"
        f"<br><b>Members:</b> {info['size']}"
        f" &nbsp;·&nbsp; <b>Span:</b> {info['spread_days']}d"
        f" &nbsp;·&nbsp; <b>Cohesion:</b> {info['cohesion']:.2f}"
        f"<br><b>Dates:</b> {info['first_seen'][:10]} → {info['last_seen'][:10]}"
        f"<br><b>Members:</b> {', '.join(info['members'][:6])}"
        f"{' …' if len(info['members']) > 6 else ''}"
    )


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
def load_price(price_path: Path, year: int) -> pd.DataFrame:
    if price_path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(price_path)
    else:
        df = pd.read_csv(price_path)
    # The file uses 'Name' as the date column header
    date_col = next(c for c in df.columns
                    if c.lower() in ("name", "date", "datetime"))
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df[df[date_col].dt.year == year].copy()
    df = df.rename(columns={
        date_col: "date",
        "Crude Oil-WTI Spot Cushing U$/BBL": "wti",
        "Crude Oil Brent US M1 U$/BBL": "brent",
    })
    return df[["date", "wti", "brent"]].sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Build the figure for one window
# ---------------------------------------------------------------------------
def build_figure(events_df: pd.DataFrame, clusters: Dict[str, dict],
                 price_df: pd.DataFrame, window_days: int, year: int) -> go.Figure:
    win_clusters = {cid: info for cid, info in clusters.items()
                    if info["window_days"] == window_days}
    if not win_clusters:
        fig = go.Figure()
        fig.update_layout(title=f"No clusters in window ±{window_days}d")
        return fig

    # Stable ordering: by first_seen then by id, so colour assignment is
    # reproducible across re-runs
    ordered_ids = sorted(win_clusters.keys(),
                         key=lambda cid: (win_clusters[cid]["first_seen"], cid))
    color_of: Dict[str, str] = {}
    palette_idx = 0
    for cid in ordered_ids:
        role = win_clusters[cid]["role"]
        if role in ("NOVEL", "REPEAT", "(noise)", "(singleton)"):
            color_of[cid] = color_for_cluster(cid, role, 0)
        else:
            color_of[cid] = color_for_cluster(cid, role, palette_idx)
            palette_idx += 1

    # Slots only for non-grey clusters (so the Gantt isn't dominated by greys)
    interesting_ids = [cid for cid in ordered_ids
                       if win_clusters[cid]["role"]
                       in ("CONTINUATION", "REGIME-MARKER")]
    intervals = [
        (cid, pd.to_datetime(win_clusters[cid]["first_seen"]),
              pd.to_datetime(win_clusters[cid]["last_seen"]))
        for cid in interesting_ids
    ]
    slot_of = assign_slots(intervals)
    n_slots = (max(slot_of.values()) + 1) if slot_of else 1

    # Map each event's doc_id -> (cluster_id, role, label)
    role_col = f"local_cluster_role_{window_days}d"
    cid_col  = f"local_cluster_id_{window_days}d"
    label_col = f"local_cluster_label_{window_days}d"
    members_lookup: Dict[str, str] = {}
    for cid, info in win_clusters.items():
        for m in info["members"]:
            members_lookup[m] = cid

    # Build figure with 2 rows
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.58, 0.42],
        vertical_spacing=0.06,
        subplot_titles=(
            f"<b>Daily oil price ({year})</b> — events coloured by cluster, hover for headline",
            f"<b>Cluster timeline</b> — overlapping stories stacked into separate rows ({len(interesting_ids)} stories)",
        ),
    )

    # ---- Row 1: price line ----
    fig.add_trace(go.Scatter(
        x=price_df["date"], y=price_df["wti"],
        mode="lines", line=dict(color="#212529", width=1.5),
        name="WTI Cushing spot", hoverinfo="skip", legendgroup="price",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=price_df["date"], y=price_df["brent"],
        mode="lines", line=dict(color="#9AA5B1", width=1, dash="dot"),
        name="Brent M1", hoverinfo="skip", legendgroup="price",
    ), row=1, col=1)

    # ---- Row 1: event dots, grouped by cluster colour ----
    # Group events by cluster id so each cluster gets one trace (clean legend, hover)
    role_legend_added = {"NOVEL": False, "REPEAT": False,
                         "CONTINUATION": False, "REGIME-MARKER": False,
                         "(noise)": False, "(singleton)": False}

    events_df = events_df.copy()
    events_df["_date"] = pd.to_datetime(events_df["date"], errors="coerce")
    events_df = events_df[events_df["_date"].dt.year == year]
    if "price" not in events_df.columns:
        events_df["price"] = events_df.get("Crude Oil-WTI Spot Cushing U$/BBL")
    # Sort so dot draw order is stable
    events_df = events_df.sort_values(["_date"])

    for cid in ordered_ids:
        sub = events_df[events_df[cid_col] == cid]
        if sub.empty:
            continue
        info = win_clusters[cid]
        role = info["role"]
        color = color_of[cid]
        is_grey = role in ("NOVEL", "REPEAT", "(noise)", "(singleton)")
        marker_size = 6 if is_grey else 9
        marker_opacity = 0.55 if is_grey else 0.95
        marker_line = dict(color="#222", width=0.4)

        hovs = [event_hover(r, info["label"], role, cid) for _, r in sub.iterrows()]
        # Show this trace under its role's legend group; only add legend once per role
        show_in_legend = not role_legend_added[role]
        role_legend_added[role] = True

        fig.add_trace(go.Scatter(
            x=sub["_date"],
            y=sub["price"],
            mode="markers",
            marker=dict(size=marker_size, color=color,
                        opacity=marker_opacity, line=marker_line),
            name=role, legendgroup=f"role-{role}",
            showlegend=show_in_legend,
            hoverinfo="text", hovertext=hovs,
        ), row=1, col=1)

    # ---- Row 2: cluster Gantt (only CONTINUATION + REGIME-MARKER) ----
    for cid in interesting_ids:
        info = win_clusters[cid]
        slot = slot_of[cid]
        color = color_of[cid]
        fig.add_trace(go.Scatter(
            x=[pd.to_datetime(info["first_seen"]),
               pd.to_datetime(info["last_seen"])],
            y=[slot, slot],
            mode="lines+markers",
            line=dict(color=color, width=11),
            marker=dict(size=10, color=color, line=dict(color="#222", width=0.6)),
            name=cid, legendgroup=f"cluster-{cid}",
            showlegend=False,
            hoverinfo="text",
            hovertext=[cluster_hover(cid, info)] * 2,
        ), row=2, col=1)

    # ---- Layout polish ----
    fig.update_xaxes(showgrid=True, gridcolor="#e5ebf1", row=1, col=1)
    fig.update_xaxes(showgrid=True, gridcolor="#e5ebf1", row=2, col=1,
                     title="Date")
    fig.update_yaxes(title="Price (USD / bbl)", showgrid=True,
                     gridcolor="#e5ebf1", row=1, col=1)
    fig.update_yaxes(title="Story slot", showgrid=True, gridcolor="#e5ebf1",
                     dtick=1, range=[-0.6, max(0, n_slots - 0.4)],
                     row=2, col=1)

    fig.update_layout(
        title=dict(
            text=(f"<b>Local Cluster Explorer — {year}, window ±{window_days}d</b><br>"
                  f"<sub>{len(events_df)} events · {len(win_clusters)} total clusters · "
                  f"{len(interesting_ids)} CONTINUATION/REGIME stories shown in Gantt · "
                  f"NOVEL + REPEAT in pale grey · click trace name in legend to hide a role</sub>"),
            x=0.02, y=0.98, font=dict(size=15, color="#212529"),
        ),
        height=820,
        margin=dict(l=20, r=20, t=85, b=20),
        legend=dict(orientation="h", x=0.02, y=-0.04,
                    bgcolor="rgba(255,255,255,0.85)", bordercolor="#bbb",
                    borderwidth=1),
        plot_bgcolor="#fafbfd",
        paper_bgcolor="white",
        hoverlabel=dict(bgcolor="white", bordercolor="#aaa",
                        font=dict(family="Calibri, Arial", size=12)),
    )
    return fig


# ---------------------------------------------------------------------------
# DRIVER
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv",   required=True, help="df_YYYY_with_local_clusters.csv")
    ap.add_argument("--json",  required=True, help="local_clusters_YYYY.json")
    ap.add_argument("--price", required=True,
                    help="daily price file (xlsx or csv with WTI + Brent columns)")
    ap.add_argument("--year",  required=True, type=int)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--windows", default="3,15,30",
                    help="comma-separated day offsets to plot (one HTML each)")
    args = ap.parse_args()

    csv_path  = Path(args.csv).resolve()
    json_path = Path(args.json).resolve()
    price_path = Path(args.price).resolve()
    for p in (csv_path, json_path, price_path):
        if not p.exists():
            sys.exit(f"ERROR: missing input {p}")
    out_dir = Path(args.output_dir).resolve() if args.output_dir else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] events: {csv_path.name}")
    # The Brahmanda CSVs sometimes have unescaped newlines inside quoted fields;
    # use the python engine + on_bad_lines=skip to be tolerant.
    events_df = pd.read_csv(
        csv_path, engine="python", on_bad_lines="skip", quoting=0,
    )
    print(f"[load] events: {len(events_df)} rows")

    print(f"[load] clusters: {json_path.name}")
    with json_path.open(encoding="utf-8") as f:
        side = json.load(f)
    flat: Dict[str, dict] = {}
    for w_key, w_clusters in side.items():
        for cid, info in w_clusters.items():
            flat[cid] = info
    print(f"[load] clusters: {len(flat)} total across all windows")

    print(f"[load] price: {price_path.name}  (year={args.year})")
    price_df = load_price(price_path, args.year)
    print(f"[load] price: {len(price_df)} daily rows")

    for w in [int(x) for x in args.windows.split(",")]:
        print(f"[plot] window ±{w}d")
        fig = build_figure(events_df, flat, price_df, w, args.year)
        out = out_dir / f"local_clusters_{args.year}_explorer_{w}d.html"
        fig.write_html(str(out), include_plotlyjs="cdn",
                       full_html=True, auto_open=False)
        print(f"[plot] wrote {out}")

    print("\nDone. Open the HTML files in a browser — drag/zoom the chart, hover for details.")


if __name__ == "__main__":
    main()
