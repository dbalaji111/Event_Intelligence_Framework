
'''This code is for accessing and displaying oil headlines from a CSV file using Streamlit. It reads the headlines from a specified file, detects the headline column, and displays the headlines along with optional metadata like creation date and source. It also includes an expandable section to view the raw data in a table format. The configuration for file paths is loaded from a Config class that reads from environment variables.'''
import pandas as pd
import streamlit as st
import sys
from pathlib import Path



PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from src.config import Config

# Load configuration
cfg = Config()

st.set_page_config(page_title="Oil Headlines Dashboard", layout="wide")
st.title("Oil Headlines Dashboard")

#csv_path = cfg.OIL_HEADLINES_FILE ## keeping to use everything 
csv_path = cfg.OIL_HEADLINES_PROCESSED_FILE

# Show file location for debugging (optional)
st.caption(f"Source file: {csv_path}")

if not csv_path.exists():
    st.error(f"File not found: {csv_path}")
    st.stop()

# Read data
df = pd.read_csv(csv_path)

# Detect headline column
headline_col = "text" if "text" in df.columns else df.columns[0]

st.subheader("Latest Oil Headlines")

cols_per_row = 6

for i in range(0, len(df), cols_per_row):

    cols = st.columns(cols_per_row)

    for j in range(cols_per_row):

        if i + j >= len(df):
            break

        row = df.iloc[i + j]

        with cols[j]:
            with st.container():
                st.markdown(
                    f"""
                    <div style="
                        border:1px solid #ddd;
                        padding:10px;
                        border-radius:8px;
                        height:140px;
                        overflow:hidden;
                        font-size:14px;
                    ">
                        <b>{row[headline_col]}</b><br>
                    """,
                    unsafe_allow_html=True
                )

                if "versionCreated" in df.columns:
                    st.markdown(
                        f"<small>{row['versionCreated']}</small>",
                        unsafe_allow_html=True
                    )

                if "sourceCode" in df.columns:
                    st.markdown(
                        f"<small>{row['sourceCode']}</small>",
                        unsafe_allow_html=True
                    )

                st.markdown("</div>", unsafe_allow_html=True)