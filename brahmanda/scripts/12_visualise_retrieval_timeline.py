"""
12_visualise_retrieval_timeline.py
====================================
Interactive timeline view of one retrieval JSON for human validation.

Layout (single HTML, two stacked panels with shared X-axis):

  Top panel    Daily WTI Cushing + Brent M1 price line, full history.
               * Base cluster's date window shaded BOLD RED with topic label.
               * Each top-K analogue's date window shaded by rank (deep blue
                 → light blue), label visible on hover.

  Bottom panel Gantt of every cluster (base + analogues) on its own slot,
               sorted by start date. Hover shows doc_ids, scores, and a
               headline preview from the source memory CSVs.

Reads
-----
  --retrieval_json  analogues_<cluster_id>.json from script 09 / 09b
  --price_xlsx      spot_and_spreads_data.xlsx (daily WTI + Brent)
  --master_csv      Brahmanda_v3/memory/df_major_metadata.csv (optional)
  --new_csv         memory_2024_2026/df_major_metadata_2024_2026.csv (optional)

Writes (default: alongside the retrieval JSON)
----------------------------------------------
  timeline_<cluster_id>.html

Run
---
    python 12_visualise_retrieval_timeline.py \
        --retrieval_json clusters_full_history/retrievals/2022_w10/analogues_c10d_02700.json \
        --price_xlsx     "C:\\...\\oil_price_data\\spot_and_spreads_data.xlsx" \
        --master_csv     "C:\\...\\Brahmanda_v3\\memory\\df_major_metadata.csv" \
        --new_csv        ".\\memory_2024_2026\\df_major_metadata_2024_2026.csv"
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    sys.exit("ERROR: plotly required.  pip install plotly openpyxl")


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_price(price_path: Path,
               year_min: int = 2000, year_max: int = 2027) -> pd.DataFrame:
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
        "Crude Oil Brent US M1 U$/BBL":      "brent",
    })
    df = df[["date", "wti", "brent"]].dropna(subset=["date"]).sort_values("date")
    df = df[(df["date"].dt.year >= year_min) & (df["date"].dt.year <= year_max)]
    print(f"[load]   {len(df)} rows  ({df['date'].min().date()} → {df['date'].max().date()})")
    return df


def load_headline_lookup(csv_path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if csv_path is None or not csv_path.exists():
        return {}
    print(f"[load] {csv_path.name} (headline cols only)")
    cols_wanted = ["document_id", "date", "Primary Event",
                   "cleaned_text", "text", "category", "Polarity"]
    try:
        df = pd.read_csv(csv_path, usecols=lambda c: c in cols_wanted,
                         encoding="utf-8", low_memory=False)
    except Exception:
        df = pd.read_csv(csv_path, usecols=lambda c: c in cols_wanted,
                         encoding="utf-8", engine="python", on_bad_lines="skip")
    print(f"[load]   {len(df)} rows")
    out = {}
    for _, r in df.iterrows():
        doc = str(r.get("document_id", "") or "").lower()
        if not doc:
            continue
        body = r.get("cleaned_text") or r.get("text") or ""
        if isinstance(body, str):
            body = re.sub(r"\s+", " ", body).strip()
            if len(body) > 220:
                body = body[:220] + "…"
        prim = r.get("Primary Event", "")
        if isinstance(prim, str):
            prim = re.sub(r"\s+", " ", prim).strip()
            if len(prim) > 100:
                prim = prim[:100] + "…"
        out[doc] = {
            "date":          r.get("date", ""),
            "primary_event": prim,
            "snippet":       body,
            "category":      r.get("category", ""),
            "polarity":      r.get("Polarity", ""),
        }
    return out


# ---------------------------------------------------------------------------
# Greedy interval-coloring for the Gantt slots
# ---------------------------------------------------------------------------
def assign_slots(intervals):
    sorted_iv = sorted(intervals, key=lambda t: t[1])
    slot_of, slot_end = {}, []
    for cid, start, end in sorted_iv:
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
# Hover-text builders
# ---------------------------------------------------------------------------
def short(s, n=200):
    if not isinstance(s, str): return ""
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= n else s[:n] + "…"


def cluster_hover(cluster: Dict[str, Any], lookup: Dict, *, is_base: bool,
                  rank: Optional[int] = None) -> str:
    cid = cluster.get("cluster_id", "?")
    label = cluster.get("label", "")
    fs    = cluster.get("first_seen", "")[:10]
    ls    = cluster.get("last_seen", "")[:10]
    sz    = cluster.get("size", "?")
    role  = cluster.get("role", "")
    spread = cluster.get("spread_days", "?")
    if is_base:
        head = f"<b style='color:#c0392b'>BASE · {cid}</b>"
        scores_line = ""
    else:
        s = cluster.get("scores", {})
        head = f"<b style='color:#2c5fa0'>#{rank}  {cid}</b>"
        scores_line = (
            f"<br><b>w*</b> {s.get('composite_weight','?'):+.3f}"
            f"  ·  <b>sem</b> {s.get('s_sem',0):.2f}"
            f"  ·  <b>jac</b> {s.get('entity_jaccard',0):.2f}"
            f"  ·  <b>d_qtl</b> {s.get('d_qtl',0):.2f}"
            f"  ·  <b>dom</b> {s.get('dominance',0):.2f}"
            f"  ·  <b>age</b> {s.get('age_years',0):.0f}y"
        )
    members = ((cluster.get("factors") or {}).get("members")
               if not is_base else cluster.get("members")) or []
    headlines = []
    for doc in members[:4]:
        info = lookup.get(str(doc).lower())
        if info:
            headlines.append(
                f"<br><i>{info['date']}</i>  · "
                f"<b>{short(info.get('primary_event'), 80) or '—'}</b>"
                f"<br>&nbsp;&nbsp;{short(info.get('snippet'), 200)}"
            )
        else:
            headlines.append(f"<br><i>{doc} (not in metadata)</i>")
    extra = max(0, len(members) - 4)
    return (
        f"{head}"
        f"<br>{fs} → {ls}  ·  <b>{sz}</b> events  ·  spread {spread}d  ·  {role}"
        f"<br><b>Topic:</b> <i>{label}</i>"
        f"{scores_line}"
        f"<br><br><b>Sample members ({min(4, len(members))} of {len(members)}):</b>"
        + "".join(headlines)
        + (f"<br><i>… {extra} more hidden</i>" if extra else "")
    )


# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------
def build_figure(retrieval: Dict[str, Any],
                 price_df: pd.DataFrame,
                 lookup: Dict[str, Dict[str, Any]]) -> go.Figure:
    base = retrieval.get("query", {})
    analogues = retrieval.get("analogues", [])

    # Color ramp for analogues (rank 1 = darkest)
    ramp = ["#08306b", "#08519c", "#2171b5", "#4292c6", "#6baed6",
            "#9ecae1", "#c6dbef", "#deebf7"]
    def color_for_rank(r):
        return ramp[min(r - 1, len(ramp) - 1)]

    base_color = "#c0392b"   # bold red

    # Build slot assignment for the Gantt panel (base + analogues)
    intervals = [(base.get("cluster_id", "BASE"),
                  pd.to_datetime(base.get("first_seen")),
                  pd.to_datetime(base.get("last_seen")))]
    for a in analogues:
        intervals.append((
            a["cluster_id"],
            pd.to_datetime(a["first_seen"]),
            pd.to_datetime(a["last_seen"]),
        ))
    slot_of = assign_slots(intervals)
    n_slots = max(slot_of.values()) + 1 if slot_of else 1

    # Build subplots
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.62, 0.38],
        vertical_spacing=0.04,
        subplot_titles=(
            f"<b>WTI Cushing spot</b> &nbsp;·&nbsp; "
            f"BASE highlighted in red, top-{len(analogues)} analogues in blue ramp",
            "<b>Cluster Gantt</b> &nbsp;·&nbsp; hover for label, members, scores",
        ),
    )

    # ---- Row 1: price lines ----
    fig.add_trace(go.Scatter(
        x=price_df["date"], y=price_df["wti"],
        mode="lines", line=dict(color="#212529", width=1.5),
        name="WTI Cushing", hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=price_df["date"], y=price_df["brent"],
        mode="lines", line=dict(color="#9AA5B1", width=1, dash="dot"),
        name="Brent M1", hoverinfo="skip"), row=1, col=1)

    # ---- Row 1: base shaded band (bold red) ----
    bf = pd.to_datetime(base.get("first_seen"))
    bl = pd.to_datetime(base.get("last_seen"))
    fig.add_vrect(x0=bf, x1=bl, fillcolor=base_color, opacity=0.35,
                  line_width=0, layer="below", row=1, col=1,
                  annotation_text=f"<b>BASE: {base.get('label','')[:50]}</b>",
                  annotation_position="top left",
                  annotation_font=dict(color=base_color, size=11))

    # ---- Row 1: analogue shaded bands (blue ramp by rank) ----
    for a in analogues:
        af = pd.to_datetime(a["first_seen"])
        al = pd.to_datetime(a["last_seen"])
        rank = a.get("rank", 99)
        fig.add_vrect(x0=af, x1=al, fillcolor=color_for_rank(rank),
                      opacity=0.20, line_width=0, layer="below",
                      row=1, col=1)

    # ---- Row 1: invisible markers on the price line above each band, for hover ----
    # Base hover marker
    if pd.notna(bf) and pd.notna(bl):
        mid = bf + (bl - bf) / 2
        try:
            wti_at = float(price_df.loc[price_df["date"].sub(mid).abs().idxmin(), "wti"])
        except Exception:
            wti_at = price_df["wti"].mean()
        fig.add_trace(go.Scatter(
            x=[mid], y=[wti_at], mode="markers",
            marker=dict(size=14, color=base_color, symbol="diamond",
                        line=dict(color="white", width=2)),
            name="BASE", hoverinfo="text",
            hovertext=[cluster_hover(base, lookup, is_base=True)],
        ), row=1, col=1)

    # Analogue hover markers
    for a in analogues:
        af = pd.to_datetime(a["first_seen"])
        al = pd.to_datetime(a["last_seen"])
        if pd.isna(af) or pd.isna(al): continue
        mid = af + (al - af) / 2
        try:
            wti_at = float(price_df.loc[price_df["date"].sub(mid).abs().idxmin(), "wti"])
        except Exception:
            wti_at = price_df["wti"].mean()
        rank = a.get("rank", 99)
        fig.add_trace(go.Scatter(
            x=[mid], y=[wti_at], mode="markers",
            marker=dict(size=10, color=color_for_rank(rank), symbol="circle",
                        line=dict(color="white", width=1.5)),
            name=f"#{rank}",
            legendgroup=f"rank{rank}",
            showlegend=(rank <= 5),
            hoverinfo="text",
            hovertext=[cluster_hover(a, lookup, is_base=False, rank=rank)],
        ), row=1, col=1)

    # ---- Row 2: Gantt ----
    # Base bar
    base_slot = slot_of.get(base.get("cluster_id", "BASE"), 0)
    fig.add_trace(go.Scatter(
        x=[bf, bl], y=[base_slot, base_slot], mode="lines+markers",
        line=dict(color=base_color, width=14),
        marker=dict(size=12, color=base_color, symbol="diamond",
                    line=dict(color="white", width=1)),
        name="BASE (Gantt)", showlegend=False,
        hoverinfo="text",
        hovertext=[cluster_hover(base, lookup, is_base=True)] * 2,
    ), row=2, col=1)
    # Analogue bars
    for a in analogues:
        af = pd.to_datetime(a["first_seen"])
        al = pd.to_datetime(a["last_seen"])
        if pd.isna(af) or pd.isna(al): continue
        slot = slot_of.get(a["cluster_id"], 0)
        rank = a.get("rank", 99)
        col_a = color_for_rank(rank)
        fig.add_trace(go.Scatter(
            x=[af, al], y=[slot, slot], mode="lines+markers",
            line=dict(color=col_a, width=10),
            marker=dict(size=8, color=col_a,
                        line=dict(color="white", width=1)),
            name=f"#{rank}", showlegend=False,
            hoverinfo="text",
            hovertext=[cluster_hover(a, lookup, is_base=False, rank=rank)] * 2,
        ), row=2, col=1)

    # Layout
    base_id = base.get("cluster_id", "?")
    fig.update_layout(
        title=dict(
            text=(f"<b>Retrieval Validation — {base_id}</b><br>"
                  f"<sub>{base.get('first_seen','')[:10]} → {base.get('last_seen','')[:10]}  ·  "
                  f"size {base.get('size','?')}  ·  topic: <i>{base.get('label','')}</i>  ·  "
                  f"K={len(analogues)} analogues</sub>"),
            x=0.02, y=0.98, font=dict(size=15, color="#212529"),
        ),
        height=820,
        hoverlabel=dict(bgcolor="white", bordercolor="#888",
                        font=dict(family="Calibri, Arial", size=12),
                        align="left"),
        plot_bgcolor="#fafbfd", paper_bgcolor="white",
        margin=dict(l=20, r=20, t=85, b=20),
        legend=dict(orientation="h", x=0.02, y=-0.04,
                    bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="#bbb", borderwidth=1),
    )
    fig.update_xaxes(title="Date", showgrid=True, gridcolor="#e5ebf1", row=2, col=1)
    fig.update_yaxes(title="WTI / Brent (USD/bbl)", showgrid=True,
                     gridcolor="#e5ebf1", row=1, col=1)
    fig.update_yaxes(title="Slot", showgrid=True, gridcolor="#e5ebf1",
                     dtick=1, range=[-0.6, max(0, n_slots - 0.4)],
                     row=2, col=1)
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--retrieval_json", required=True)
    ap.add_argument("--price_xlsx", required=True)
    ap.add_argument("--master_csv", default=None,
                    help="optional headline-lookup CSV (2001-2023 master)")
    ap.add_argument("--new_csv", default=None,
                    help="optional headline-lookup CSV (2024-2026)")
    ap.add_argument("--output_dir", default=None)
    args = ap.parse_args()

    rj_path = Path(args.retrieval_json).resolve()
    out_dir = Path(args.output_dir).resolve() if args.output_dir else rj_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {rj_path.name}")
    with rj_path.open(encoding="utf-8") as f:
        retrieval = json.load(f)
    print(f"[load]   base={retrieval['query']['cluster_id']}, "
          f"K={len(retrieval.get('analogues',[]))}")

    price = load_price(Path(args.price_xlsx).resolve())

    lookup = {}
    if args.master_csv:
        lookup.update(load_headline_lookup(Path(args.master_csv).resolve()))
    if args.new_csv:
        lookup.update(load_headline_lookup(Path(args.new_csv).resolve()))
    print(f"[lookup] {len(lookup)} doc_ids indexed")

    print(f"[plot] building timeline figure")
    fig = build_figure(retrieval, price, lookup)
    base_id = retrieval["query"]["cluster_id"]
    safe_id = re.sub(r"[^A-Za-z0-9_\-]", "_", base_id)
    out_html = out_dir / f"timeline_{safe_id}.html"
    fig.write_html(str(out_html), include_plotlyjs="cdn",
                   full_html=True, auto_open=False)
    print(f"[save] {out_html}  ({out_html.stat().st_size/1e3:.1f} kB)")


if __name__ == "__main__":
    main()
