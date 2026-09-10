"""
04b_factiva_batch.py
====================

Batch driver for 04_factiva_pdf_ingest.py.

Walks a folder of Factiva PDF dossiers, runs the splitter+filter on each,
and aggregates the per-file outputs into combined kept/review/dropped
JSON files plus a master summary.

Usage
-----
    python 04b_factiva_batch.py /path/to/oil_data_2024_2026
    python 04b_factiva_batch.py /path/to/oil_data_2024_2026 --keep-thresh 0.4

Outputs land in <input_dir>/_processed/ by default.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

# Reuse the parsing/scoring functions from the per-file script
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
spec = importlib.util.spec_from_file_location(
    "factiva_ingest", str(Path(__file__).resolve().parent / "04_factiva_pdf_ingest.py")
)
mod = importlib.util.module_from_spec(spec)
# Register before exec_module so @dataclass can resolve the module via sys.modules
sys.modules["factiva_ingest"] = mod
spec.loader.exec_module(mod)  # type: ignore


def process_folder(
    in_dir: Path,
    out_dir: Path,
    keep_thresh: float,
    review_thresh: float,
) -> None:
    pdfs = sorted(p for p in in_dir.iterdir() if p.suffix.lower() == ".pdf")
    if not pdfs:
        print(f"No PDFs found in {in_dir}")
        return

    print(f"[batch] {len(pdfs)} PDFs in {in_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    aggregate = {"kept": [], "review": [], "dropped": []}
    per_file = []

    for pdf in pdfs:
        print(f"\n[batch] --- {pdf.name} ---")
        try:
            summary = mod.process_pdf(
                pdf,
                out_dir=out_dir,
                keep_thresh=keep_thresh,
                review_thresh=review_thresh,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[batch] ERROR processing {pdf.name}: {e}")
            per_file.append({"file": pdf.name, "error": str(e)})
            continue

        # Read back the per-file JSONs and append, tagging source PDF
        stem = pdf.stem
        for bucket in ["kept", "review", "dropped"]:
            p = out_dir / f"{stem}__{bucket}.json"
            if not p.exists():
                continue
            items = json.loads(p.read_text(encoding="utf-8"))
            for it in items:
                it["_source_pdf"] = pdf.name
            aggregate[bucket].extend(items)

        per_file.append(
            {
                "file": pdf.name,
                "total": summary["total_articles"],
                "kept": summary["kept"],
                "review": summary["review"],
                "dropped": summary["dropped"],
            }
        )

    # Write aggregated JSONs
    print(f"\n[batch] writing aggregated outputs to {out_dir}")
    for bucket, items in aggregate.items():
        p = out_dir / f"_ALL__{bucket}.json"
        p.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[batch]   _ALL__{bucket}.json: {len(items):4d} articles")

    # Master summary markdown
    md = ["# Factiva batch ingest summary", ""]
    md.append(f"- Source folder: `{in_dir}`")
    md.append(f"- PDFs processed: **{len(pdfs)}**")
    total = sum(p.get("total", 0) for p in per_file)
    kept_n = sum(p.get("kept", 0) for p in per_file)
    rev_n = sum(p.get("review", 0) for p in per_file)
    drop_n = sum(p.get("dropped", 0) for p in per_file)
    md.append(
        f"- Total articles split: **{total}** "
        f"(kept **{kept_n}**, review **{rev_n}**, dropped **{drop_n}**)"
    )
    md.append("")
    md.append("## Per-file breakdown")
    md.append("")
    md.append("| File | Total | Kept | Review | Dropped |")
    md.append("|------|-------|------|--------|---------|")
    for p in per_file:
        if "error" in p:
            md.append(f"| {p['file']} | ERROR: {p['error']} | | | |")
            continue
        md.append(
            f"| {p['file']} | {p['total']} | {p['kept']} | {p['review']} | {p['dropped']} |"
        )
    md.append("")

    # Year/month distribution of kept articles
    from collections import Counter
    yc: Counter = Counter()
    for it in aggregate["kept"]:
        d = it.get("date") or ""
        # date is like "30 December 2024"
        toks = d.split()
        if len(toks) == 3:
            yc[(toks[2], toks[1])] += 1
    md.append("## Kept articles by year-month")
    md.append("")
    md.append("| Year | Month | Count |")
    md.append("|------|-------|-------|")
    for (y, mo), n in sorted(yc.items()):
        md.append(f"| {y} | {mo} | {n} |")
    md.append("")

    md_path = out_dir / "_BATCH_SUMMARY.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"[batch] summary -> {md_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("in_dir", type=Path)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--keep-thresh", type=float, default=0.5)
    ap.add_argument("--review-thresh", type=float, default=1.5)
    args = ap.parse_args()

    if not args.in_dir.exists():
        print(f"ERROR: {args.in_dir} does not exist", file=sys.stderr)
        return 1

    out_dir = args.out_dir or (args.in_dir / "_processed")
    process_folder(args.in_dir, out_dir, args.keep_thresh, args.review_thresh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
