import pandas as pd
import streamlit as st
import sys
from pathlib import Path

# -----------------------------
# FIX IMPORT PATH
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.r04_time_series_visualiser_v_1 import get_full_timeseries_figure
from src.config import Config

cfg = Config()

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(layout="wide")
st.title("Oil Intelligence Dashboard")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_headlines():
    df = pd.read_csv(cfg.OIL_HEADLINES_PROCESSED_ALL_ATTRIBUTES_FILE)

    if "versionCreated" in df.columns:
        df = df.sort_values(by="versionCreated", ascending=False)

    return df

df = load_headlines()
headline_col = "text"

# -----------------------------
# SESSION STATE
# -----------------------------
if "selected_index" not in st.session_state:
    st.session_state.selected_index = None

# -----------------------------
# TOP LAYOUT (HEADLINES + DETAILS)
# -----------------------------
left, right = st.columns([2, 1])

# =============================
# LEFT PANEL → HEADLINES
# =============================
with left:
    st.subheader("Headlines")

    cols_per_row = 6

    for i in range(0, len(df), cols_per_row):
        cols = st.columns(cols_per_row)

        for j in range(cols_per_row):
            if i + j >= len(df):
                break

            row = df.iloc[i + j]

            with cols[j]:
                if st.button(row[headline_col][:80], key=f"{i+j}"):
                    st.session_state.selected_index = i + j

                st.caption(row.get("versionCreated", ""))

# =============================
# RIGHT PANEL → DETAILS
# =============================
with right:
    st.subheader("Article Details")

    if st.session_state.selected_index is None:
        st.info("Select a headline to view details")

    else:
        row = df.iloc[st.session_state.selected_index]

        st.markdown(f"### {row[headline_col]}")
        st.caption(row.get("versionCreated", ""))
        st.caption(row.get("sourceCode", ""))

        st.divider()

        st.markdown("#### Full Text")
        st.write(row.get("clean_text", "No content available"))

        st.divider()

        st.markdown("#### Event Intelligence")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Event Type", row.get("event_type", "N/A"))
            st.metric("Severity", row.get("event_severity", "N/A"))
            st.metric("Region", row.get("affected_region", "N/A"))

        with col2:
            st.metric("Supply Shock", row.get("supply_shock", "N/A"))
            st.metric("Demand Shock", row.get("demand_shock", "N/A"))
            st.metric("Geopolitical Risk", row.get("geopolitical_risk", "N/A"))

        st.divider()

        st.metric("Sentiment", row.get("sentiment", "N/A"))
        st.metric("Impact Horizon", row.get("impact_horizon", "N/A"))

# =============================
# FULL WIDTH TIME SERIES (CLEAN)
# =============================
st.divider()
st.markdown("## Market Time Series")

fig = get_full_timeseries_figure(cfg.OIL_TIMESERIES_FILE)

st.plotly_chart(fig, use_container_width=True)