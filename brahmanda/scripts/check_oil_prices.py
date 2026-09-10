"""Quick diagnostic for oil_prices_full.xlsx"""
import argparse
import pandas as pd
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--file", required=True)
args = ap.parse_args()

f = Path(args.file)
print(f"FILE: {f.name}  ({f.stat().st_size/1e6:.1f} MB)")

# Try reading
xl = pd.ExcelFile(f)
print(f"Sheets: {xl.sheet_names}")

for sheet in xl.sheet_names[:3]:
    print(f"\n{'─'*55}")
    print(f"SHEET: {sheet}")
    df = pd.read_excel(f, sheet_name=sheet, nrows=5)
    print(f"  Shape (first 5 rows): {df.shape}")
    print(f"  Columns ({len(df.columns)}):")
    for c in df.columns:
        print(f"    {c}")
    print(f"\n  First 3 rows:")
    print(df.head(3).to_string())

    # Full read to check date range
    df_full = pd.read_excel(f, sheet_name=sheet)
    date_col = df_full.columns[0]
    df_full[date_col] = pd.to_datetime(df_full[date_col], errors="coerce")
    df_full = df_full.dropna(subset=[date_col])
    print(f"\n  Full rows: {len(df_full)}")
    print(f"  Date range: {df_full[date_col].min().date()} → {df_full[date_col].max().date()}")
    
    # Check which columns have data from when
    print(f"\n  Column coverage (first non-null date):")
    for c in df_full.columns[1:]:
        first = df_full[df_full[c].notna()][date_col].min()
        last  = df_full[df_full[c].notna()][date_col].max()
        pct   = df_full[c].notna().mean() * 100
        print(f"    {str(c):<55} {str(first.date()) if pd.notna(first) else 'N/A'} → {str(last.date()) if pd.notna(last) else 'N/A'}  ({pct:.0f}% filled)")