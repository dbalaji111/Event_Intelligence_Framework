"""
diagnose_corpus.py
==================
Run this on your Windows machine to check:
1. What files exist in the memory folder
2. Exact event_id format
3. Field names present
4. Whether train/test split exists or needs creating

Usage:
    python diagnose_corpus.py --memory_dir "C:\Users\dbala\OneDrive - Swinburne University\OneDrive\context_aware_risk_methodology\Brahmanda_v3\memory"
"""
import argparse, json, os
from pathlib import Path
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--memory_dir", required=True)
args = ap.parse_args()

mem = Path(args.memory_dir)
print(f"\n{'='*60}")
print(f"MEMORY DIR: {mem}")
print(f"{'='*60}\n")

# List all files
print("FILES IN DIRECTORY:")
all_files = list(mem.iterdir()) if mem.exists() else []
for f in sorted(all_files):
    size_mb = f.stat().st_size / 1e6
    print(f"  {f.name:<50} {size_mb:>8.1f} MB")

print()

# Find JSONL files
jsonl_files = [f for f in all_files if f.suffix in ('.jsonl', '.json', '.csv', '.npz')]
print(f"DATA FILES: {len(jsonl_files)}")
for f in jsonl_files:
    print(f"  {f.name}")

print()

# Read first few rows of each JSONL
for f in jsonl_files:
    if f.suffix != '.jsonl':
        continue
    print(f"\n{'─'*50}")
    print(f"FILE: {f.name}")
    print(f"{'─'*50}")
    
    years = defaultdict(int)
    event_ids = []
    fields = None
    n = 0
    
    try:
        with f.open('r', encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except:
                    continue
                
                if fields is None:
                    fields = list(row.keys())
                
                yr = row.get('year', row.get('Year', ''))
                try:
                    years[int(float(yr))] += 1
                except:
                    pass
                
                eid = row.get('event_id', '')
                if eid and len(event_ids) < 5:
                    event_ids.append(eid)
                
                n += 1
    except Exception as e:
        print(f"  ERROR reading: {e}")
        continue
    
    print(f"  Total rows    : {n}")
    print(f"  Year range    : {min(years.keys()) if years else '?'} "
          f"— {max(years.keys()) if years else '?'}")
    print(f"  Events by year (sample):")
    for yr in sorted(years.keys()):
        bar = '█' * min(years[yr] // 10, 40)
        print(f"    {yr}: {bar} {years[yr]}")
    print(f"\n  Sample event_ids:")
    for eid in event_ids:
        print(f"    {eid}")
    print(f"\n  Fields ({len(fields) if fields else 0}):")
    if fields:
        for i in range(0, len(fields), 4):
            chunk = fields[i:i+4]
            print(f"    {', '.join(chunk)}")

print(f"\n{'='*60}")
print("DIAGNOSIS COMPLETE")
print(f"{'='*60}\n")