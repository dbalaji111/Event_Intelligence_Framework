import yfinance as yf

ovx = yf.download("^OVX", start="2020-01-01", end="2026-08-11")
ovx.to_csv("ovx_data.csv")