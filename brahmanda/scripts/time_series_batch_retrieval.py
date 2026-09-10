import pandas as pd
import eikon as ek
from pathlib import Path
from datetime import datetime
from src.config import Config


# -----------------------------
# CONFIG
# -----------------------------
cfg = Config()
ek.set_app_key(cfg.REFINITIV_APP_KEY)

data_dir = Path(cfg.DATA_DIR)
data_dir.mkdir(parents=True, exist_ok=True)

file_path = data_dir / "spot_prices_base.csv"


# -----------------------------
# SPOT / FRONT MONTH RICS
# -----------------------------
RIC_MAP = {
    "CLc1": "WTI_FRONT",
    "LCOc1": "BRENT_FRONT"
}


# -----------------------------
# FETCH FUNCTION
# -----------------------------
def fetch_data(rics, start_date, end_date):

    df = ek.get_timeseries(
        rics,
        fields="CLOSE",
        start_date=start_date,
        end_date=end_date,
        interval="daily"
    )

    if df is None or df.empty:
        raise ValueError("No data returned.")

    return df


# -----------------------------
# MAIN
# -----------------------------
def main():

    rics = list(RIC_MAP.keys())

    start_date = "2000-01-01"
    end_date = datetime.today().strftime("%Y-%m-%d")

    print(f"Fetching from {start_date} to {end_date}")

    df = fetch_data(rics, start_date, end_date)

    # -----------------------------
    # FLATTEN MULTIINDEX
    # -----------------------------
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # -----------------------------
    # RENAME
    # -----------------------------
    df.rename(columns=RIC_MAP, inplace=True)

    # -----------------------------
    # RESET INDEX
    # -----------------------------
    df.reset_index(inplace=True)
    df.rename(columns={"Date": "DATE"}, inplace=True)

    # -----------------------------
    # SORT
    # -----------------------------
    df = df.sort_values("DATE")

    # -----------------------------
    # SPREAD
    # -----------------------------
    df["WTI_BRENT_SPREAD"] = (
        df["WTI_FRONT"] - df["BRENT_FRONT"]
    )

    # -----------------------------
    # RETURNS
    # -----------------------------
    df["WTI_RET"] = (
        df["WTI_FRONT"]
        .pct_change(fill_method=None)
    )

    df["BRENT_RET"] = (
        df["BRENT_FRONT"]
        .pct_change(fill_method=None)
    )

    # -----------------------------
    # FORWARD FILL
    # -----------------------------
    df = df.ffill()

    # -----------------------------
    # DROP REMAINING NaNs
    # -----------------------------
    df = df.dropna()

    # -----------------------------
    # SAVE
    # -----------------------------
    df.to_csv(file_path, index=False)

    print(f"\nSaved to:\n{file_path}")

    print("\nSTART DATE:")
    print(df["DATE"].min())

    print("\nEND DATE:")
    print(df["DATE"].max())

    print("\nHEAD:")
    print(df.head())

    print("\nTAIL:")
    print(df.tail())


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    main()