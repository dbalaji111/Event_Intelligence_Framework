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
def get_full_timeseries_figure(file_path):
    df = load_timeseries_data(file_path)
    fig = create_timeseries_plot(df)
    return fig

def snap_to_nearest_date(df, event_date):
    """
    Align event_date to nearest available date in df["Date"]
    """

    event_date = pd.to_datetime(event_date, errors="coerce")

    if pd.isna(event_date):
        raise ValueError("Invalid event_date")

    # Find closest date
    nearest_idx = (df["Date"] - event_date).abs().idxmin()
    return df.loc[nearest_idx, "Date"]

def get_event_timeseries_figure(file_path, event_date=None, window=15):

    df = load_timeseries_data(file_path)

    if event_date is not None:
        # Convert safely
        event_date = pd.to_datetime(event_date, errors="coerce")

        if pd.isna(event_date):
            raise ValueError("Invalid event_date")

        # 🔴 KEY FIX: snap to trading date
        event_date = snap_to_nearest_date(df, event_date)

        # Extract window around snapped date
        df = get_event_window(df, event_date, window)

    fig = create_timeseries_plot(df, event_date)

    return fig

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

    # Event marker
    if event_date is not None:
        fig.add_vline(
            x=event_date,
            line_width=2,
            line_dash="dash",
            line_color="red"
        )

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

    fig.update_xaxes(
        rangeslider_visible=True,
        type="date"
    )

    return fig


# -----------------------------
# EVENT WINDOW
# -----------------------------
def get_event_window(df, event_date, window=15):

    event_date = pd.to_datetime(event_date)

    start = event_date - pd.Timedelta(days=window)
    end = event_date + pd.Timedelta(days=window)

    return df[(df["Date"] >= start) & (df["Date"] <= end)]


# -----------------------------
# MAIN WRAPPER (IMPORTANT)
# -----------------------------
def get_event_timeseries_figure(file_path, event_date=None, window=15):
    """
    This is the ONLY function Streamlit should call.
    """

    df = load_timeseries_data(file_path)

    if event_date is not None:
        df = get_event_window(df, event_date, window)

    fig = create_timeseries_plot(df, event_date)

    return fig


# -----------------------------
# OPTIONAL STANDALONE RUN
# -----------------------------
def main():
    cfg = Config()

    df = load_timeseries_data(cfg.OIL_TIMESERIES_FILE)

    fig = create_timeseries_plot(df)

    fig.show()


if __name__ == "__main__":
    main()