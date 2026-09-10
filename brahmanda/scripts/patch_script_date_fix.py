import pandas as pd
from pathlib import Path

# Base directory
base_dir = Path(r"C:\Users\dbala\OneDrive - Swinburne University\OneDrive\context_aware_risk_methodology\Brahmanda_v3")

years = range(2015, 2023)  # 2015–2022

for year in years:
    file_path = base_dir / f"df_{year}_with_json_paths.csv"

    if not file_path.exists():
        print(f"❌ Missing: {file_path}")
        continue

    print(f"\n🔧 Processing {file_path.name}")

    df = pd.read_csv(file_path)

    # Check column
    if "Name" not in df.columns:
        print(f"⚠️ 'Name' not found in {year}, skipping...")
        continue

    # Rename
    df.rename(columns={"Name": "date"}, inplace=True)

    # Convert to datetime
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Validate
    invalid = df["date"].isna().sum()
    total = len(df)
    print(f"   Invalid dates: {invalid}/{total}")

    if invalid > 0:
        print("   ⚠️ Some rows could not be parsed")

    # Overwrite SAME file
    df.to_csv(file_path, index=False)
    print(f"   ✅ Overwritten: {file_path.name}")

print("\n🎯 Done (in-place update)")