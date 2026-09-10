"""
03_embedding_clusters.py
------------------------
Interactive 3D t-SNE + KMeans cluster explorer for oil-news embeddings.

Source:   data/oil_headlines_processed_all_attributes.csv
          (has `embedding` column from r02_1_text_analyser.py
           plus the 8 LLM attributes from
           r02_text_analysis_with_all_attributes.py)

Run:      streamlit run scripts/03_embedding_clusters.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

# -----------------------------
# Path fix for `src.config`
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import Config  # noqa: E402

cfg = Config()

# =============================
# PAGE
# =============================
st.set_page_config(
    page_title="Oil News Embedding Clusters",
    layout="wide",
)
st.title("Oil News Embedding Clusters")
st.caption(
    "3D t-SNE projection of OpenAI `text-embedding-ada-002` vectors, "
    "coloured by KMeans cluster or any LLM-extracted attribute."
)

# =============================
# SIDEBAR CONTROLS
# =============================
st.sidebar.header("Controls")

k_clusters = st.sidebar.slider("KMeans clusters (k)", 2, 15, 6)
perplexity = st.sidebar.slider("t-SNE perplexity", 5, 50, 30)
n_iter = st.sidebar.slider("t-SNE iterations", 250, 2000, 1000, step=250)
random_state = st.sidebar.number_input("Random seed", value=42, step=1)

cluster_space = st.sidebar.radio(
    "Cluster in",
    ["Original high-dim embeddings", "t-SNE 3D space"],
    index=0,
    help="Clustering on the original embeddings is more robust. "
    "Clustering in t-SNE space matches the visual groupings more literally.",
)

sample_cap = st.sidebar.slider(
    "Max rows (for speed)",
    100, 5000, 1000, step=100,
    help="t-SNE is O(n²). Cap for interactive speed.",
)


# =============================
# LOAD + PARSE EMBEDDINGS
# =============================
@st.cache_data(show_spinner=False)
def load_data(path: str, cap: int) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "embedding" not in df.columns:
        raise ValueError(
            f"No `embedding` column in {path}. "
            "Run r02_1_text_analyser.py first."
        )

    # parse embedding JSON strings -> list[float]
    def parse(x):
        if pd.isna(x):
            return None
        try:
            v = json.loads(x)
            return v if isinstance(v, list) else None
        except Exception:
            return None

    df["embedding_vec"] = df["embedding"].apply(parse)
    df = df[df["embedding_vec"].notna()].reset_index(drop=True)

    if len(df) > cap:
        df = df.sample(cap, random_state=42).reset_index(drop=True)

    return df


@st.cache_data(show_spinner="Running t-SNE…")
def run_tsne(
    vecs: np.ndarray,
    perplexity: int,
    n_iter: int,
    random_state: int,
) -> np.ndarray:
    # sklearn renamed `n_iter` -> `max_iter` in 1.5. Support both.
    kwargs = dict(
        n_components=3,
        perplexity=min(perplexity, max(5, len(vecs) - 1)),
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    )
    import inspect
    if "max_iter" in inspect.signature(TSNE).parameters:
        kwargs["max_iter"] = n_iter
    else:
        kwargs["n_iter"] = n_iter

    tsne = TSNE(**kwargs)
    return tsne.fit_transform(vecs)


@st.cache_data(show_spinner="Running KMeans…")
def run_kmeans(vecs: np.ndarray, k: int, random_state: int) -> np.ndarray:
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    return km.fit_predict(vecs)


# =============================
# PIPELINE
# =============================
try:
    df = load_data(str(cfg.OIL_HEADLINES_PROCESSED_ALL_ATTRIBUTES_FILE), sample_cap)
except Exception as e:
    st.error(str(e))
    st.stop()

if df.empty:
    st.warning("No rows with embeddings found.")
    st.stop()

st.caption(f"Loaded **{len(df):,}** rows with embeddings.")

# Stack embeddings into matrix
X = np.array(df["embedding_vec"].tolist(), dtype=np.float32)

# t-SNE
coords = run_tsne(X, perplexity, n_iter, random_state)
df["tsne_x"] = coords[:, 0]
df["tsne_y"] = coords[:, 1]
df["tsne_z"] = coords[:, 2]

# KMeans (on whichever space)
if cluster_space.startswith("Original"):
    df["cluster"] = run_kmeans(X, k_clusters, random_state)
else:
    df["cluster"] = run_kmeans(coords, k_clusters, random_state)

df["cluster"] = df["cluster"].astype(str)

# =============================
# COLOUR CONTROL + FILTERS
# =============================
attr_candidates = [
    "cluster",
    "event_type",
    "event_severity",
    "supply_shock",
    "demand_shock",
    "geopolitical_risk",
    "impact_horizon",
    "affected_region",
    "sentiment",
    "sourceCode",
]
available = [c for c in attr_candidates if c in df.columns]

colour_by = st.sidebar.selectbox("Colour points by", available, index=0)

# Optional filters (multi-select over any attribute)
with st.sidebar.expander("Filters", expanded=False):
    for col in ["event_type", "sentiment", "affected_region"]:
        if col in df.columns:
            vals = sorted(df[col].dropna().astype(str).unique().tolist())
            if vals:
                picked = st.multiselect(col, vals, default=vals)
                df = df[df[col].astype(str).isin(picked)]

if df.empty:
    st.warning("All rows filtered out. Adjust filters.")
    st.stop()

# =============================
# 3D SCATTER
# =============================
hover_cols = [
    c for c in [
        "event_type",
        "event_severity",
        "sentiment",
        "affected_region",
        "impact_horizon",
        "versionCreated",
        "sourceCode",
    ] if c in df.columns
]

# short headline for hover
df["_hover_text"] = (
    df.get("text", "").astype(str).str.slice(0, 120) + "…"
)

fig = px.scatter_3d(
    df,
    x="tsne_x", y="tsne_y", z="tsne_z",
    color=colour_by,
    hover_name="_hover_text",
    hover_data=hover_cols,
    opacity=0.8,
    height=700,
    template="plotly_dark",
)
fig.update_traces(marker=dict(size=4, line=dict(width=0)))
fig.update_layout(
    scene=dict(
        xaxis_title="t-SNE 1",
        yaxis_title="t-SNE 2",
        zaxis_title="t-SNE 3",
    ),
    legend=dict(itemsizing="constant"),
    margin=dict(l=0, r=0, t=30, b=0),
)

selected = st.plotly_chart(
    fig,
    use_container_width=True,
    on_select="rerun",
    selection_mode=("points", "box", "lasso"),
    key="cluster_scatter",
)

# =============================
# CLUSTER SUMMARY
# =============================
st.markdown("### Cluster summary")

summary_cols = st.columns(2)

with summary_cols[0]:
    st.markdown("**Size per cluster**")
    st.dataframe(
        df["cluster"].value_counts().rename_axis("cluster").reset_index(name="count"),
        use_container_width=True,
        hide_index=True,
    )

with summary_cols[1]:
    if "event_type" in df.columns:
        st.markdown("**Top event_type per cluster**")
        top_types = (
            df.groupby("cluster")["event_type"]
            .agg(lambda s: s.value_counts().head(3).to_dict())
            .reset_index(name="top_event_types")
        )
        st.dataframe(top_types, use_container_width=True, hide_index=True)

# =============================
# SELECTED POINT DETAIL
# =============================
st.markdown("### Selected point")

point_idx = None
if selected and "selection" in selected:
    pts = selected["selection"].get("points", [])
    if pts:
        point_idx = pts[0].get("point_index")

if point_idx is not None and point_idx < len(df):
    row = df.iloc[point_idx]

    st.markdown(f"**{row.get('text', '(no headline)')}**")
    meta_cols = st.columns(4)
    meta_cols[0].metric("Cluster", row.get("cluster", "-"))
    meta_cols[1].metric("Event type", row.get("event_type", "-"))
    meta_cols[2].metric("Sentiment", row.get("sentiment", "-"))
    meta_cols[3].metric("Region", row.get("affected_region", "-"))

    if "clean_text" in df.columns:
        with st.expander("Full article text"):
            st.write(row.get("clean_text", ""))
else:
    st.info("Click any point (or box/lasso select) in the 3D scatter above to inspect it.")


# =============================
# TIMELINE VIEW — prices + cluster overlay
# =============================
st.divider()
st.markdown("## Timeline — prices with cluster overlay")
st.caption(
    "Each headline is plotted at its publication date (`versionCreated`), "
    "on the price of a reference series, and coloured by the same attribute "
    "selected for the 3D view."
)


@st.cache_data(show_spinner=False)
def load_timeseries(path: str) -> pd.DataFrame:
    ts = pd.read_csv(path)
    if "Date" not in ts.columns:
        raise ValueError(f"`Date` column missing from {path}")
    ts["Date"] = pd.to_datetime(ts["Date"], errors="coerce")
    ts = ts.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return ts


try:
    ts = load_timeseries(str(cfg.OIL_TIMESERIES_FILE))
except Exception as e:
    st.warning(f"Timeline skipped: {e}")
    st.stop()

price_cols = [c for c in ts.columns if c != "Date"]
if not price_cols:
    st.warning("No price columns in oil_timeseries.csv.")
    st.stop()

default_pref = [c for c in ["WTI_M1", "BRENT_M1"] if c in price_cols] or price_cols[:1]

tl_cols = st.columns([2, 1])
with tl_cols[0]:
    series_to_plot = st.multiselect(
        "Price series to plot",
        price_cols,
        default=default_pref,
    )
with tl_cols[1]:
    ref_series = st.selectbox(
        "Reference series for dot Y-position",
        price_cols,
        index=price_cols.index(default_pref[0]),
    )

if not series_to_plot:
    st.info("Pick at least one price series.")
    st.stop()

# Parse headline publication dates; strip timezone for alignment to naive ts
if "versionCreated" not in df.columns:
    st.warning("`versionCreated` column missing — can't position dots on timeline.")
    st.stop()

df_tl = df.copy()
dt = pd.to_datetime(df_tl["versionCreated"], errors="coerce", utc=True)
df_tl["_date"] = dt.dt.tz_convert(None) if dt.dt.tz is not None else dt
df_tl = df_tl.dropna(subset=["_date"]).sort_values("_date")

# Merge each headline to the nearest trading day's reference price
ts_ref = ts[["Date", ref_series]].dropna()

wanted_cols = ["_date", "cluster", "text", "event_type", "sentiment",
               "affected_region", "event_severity", "impact_horizon",
               colour_by]
keep_cols = []
for c in wanted_cols:
    if c in df_tl.columns and c not in keep_cols:
        keep_cols.append(c)

merged = pd.merge_asof(
    df_tl[keep_cols].rename(columns={"_date": "Date"}),
    ts_ref,
    on="Date",
    direction="nearest",
    tolerance=pd.Timedelta("7D"),
)
merged = merged.dropna(subset=[ref_series])

st.caption(
    f"Overlaying **{len(merged):,}** headlines onto `{ref_series}` "
    f"(within ±7 days of a trading day)."
)

# Build figure: price lines + cluster-coloured dot overlay
tl_fig = go.Figure()

for col in series_to_plot:
    tl_fig.add_trace(
        go.Scatter(
            x=ts["Date"], y=ts[col],
            mode="lines", name=col,
            line=dict(width=1.5),
            opacity=0.85,
        )
    )

if not merged.empty:
    # Pick whichever of these cols are actually present for hover
    hover_fields = [c for c in
                    ["text", "event_type", "sentiment", "affected_region"]
                    if c in merged.columns]

    hover_lines = [f"{ref_series}: %{{y:.2f}}", "Date: %{x}"]
    for idx, c in enumerate(hover_fields):
        prefix = "<b>%{customdata[0]}</b>" if idx == 0 and c == "text" else f"{c}: %{{customdata[{idx}]}}"
        hover_lines.insert(0 if idx == 0 else len(hover_lines), prefix)
    hover_lines.append("<extra></extra>")
    hovertemplate = "<br>".join(hover_lines)

    # One trace per unique colour value so Plotly gives us a real legend
    merged["_colour"] = merged[colour_by].astype(str)
    for val, sub in merged.groupby("_colour"):
        tl_fig.add_trace(
            go.Scatter(
                x=sub["Date"], y=sub[ref_series],
                mode="markers",
                name=f"{colour_by}={val}",
                marker=dict(size=7, opacity=0.8, line=dict(width=0.5, color="#111")),
                customdata=sub[hover_fields].values if hover_fields else None,
                hovertemplate=hovertemplate if hover_fields else None,
                legendgroup="headlines",
                legendgrouptitle_text="Headlines",
            )
        )

tl_fig.update_layout(
    template="plotly_dark",
    height=550,
    hovermode="closest",
    margin=dict(l=0, r=0, t=30, b=0),
    xaxis=dict(rangeslider=dict(visible=True), type="date", title="Date"),
    yaxis=dict(title="Price"),
    legend=dict(orientation="v", x=1.01, y=1),
)

st.plotly_chart(tl_fig, use_container_width=True)

with st.expander("Cluster volume over time"):
    vol = (
        merged
        .assign(month=merged["Date"].dt.to_period("M").dt.to_timestamp())
        .groupby(["month", "cluster"])
        .size()
        .reset_index(name="count")
    )
    vol_fig = px.bar(
        vol, x="month", y="count", color="cluster",
        template="plotly_dark", height=350,
    )
    vol_fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(vol_fig, use_container_width=True)
