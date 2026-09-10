from math import pi

import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import sys
import plotly.io as pio 
pio.renderers.default = "browser"

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import Config


# -----------------------------
# LOAD DATA
# -----------------------------
def load_timeseries_data(file_path):
    df = pd.read_csv(file_path)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    return df


# -----------------------------
# CREATE PLOT
# -----------------------------
def create_timeseries_plot(df, event_date=None):
    fig = go.Figure()

    # Select columns
    wti_cols = [col for col in df.columns if col.startswith("WTI")]
    brent_cols = [col for col in df.columns if col.startswith("BRENT")]

    series_cols = wti_cols + brent_cols

    # Add lines
    for col in series_cols:
        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df[col],
                mode="lines",
                name=col,
                hovertemplate=f"{col}<br>Date=%{{x}}<br>Price=%{{y}}<extra></extra>"
            )
        )

    # Add event marker if provided
    if event_date is not None:
        fig.add_vline(
            x=event_date,
            line_width=2,
            line_dash="dash",
            line_color="red"
        )

    # Layout
    fig.update_layout(
        title="WTI & Brent Term Structure",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark",
        hovermode="x unified",
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Enable scrolling
    fig.update_xaxes(
        rangeslider_visible=True,
        type="date"
    )

    return fig


# -----------------------------
# EVENT WINDOW (IMPORTANT)
# -----------------------------
def get_event_window(df, event_date, window=15):
    return df[
        (df["Date"] >= event_date - pd.Timedelta(days=window)) &
        (df["Date"] <= event_date + pd.Timedelta(days=window))
    ]


# -----------------------------
# OPTIONAL STANDALONE RUN
# -----------------------------
def main():
    cfg = Config()

    df = load_timeseries_data(cfg.OIL_TIMESERIES_FILE)

    fig = create_timeseries_plot(df)

    # Only for standalone testing
    fig.show()


if __name__ == "__main__":
    main()