import subprocess
from pathlib import Path

base_dir = Path(r"C:\Users\dbala\OneDrive - Swinburne University\OneDrive\context_aware_risk_methodology\Brahmanda_v3")

years = range(2015, 2023)

for year in years:
    file_path = base_dir / f"df_{year}_with_json_paths.csv"

    if not file_path.exists():
        print(f"❌ Missing: {file_path}")
        continue

    print(f"\n🚀 Running clustering for {year}")

    cmd = [
        "python", "06_base_event_clustering.py",
        "--input", str(file_path),
        "--windows", "3,15,30",
        "--min_cluster_size", "5"
    ]

    subprocess.run(cmd)