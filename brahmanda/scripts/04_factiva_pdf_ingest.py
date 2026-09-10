"""
04_factiva_pdf_ingest.py
========================

Ingests Factiva multi-article PDF dossiers (the kind you get when you batch-export
search results) and turns them into structured JSON ready for the LLM enrichment
pipeline.

Pipeline
--------
1. Read PDF and concatenate text across pages, stripping the per-page Factiva
   "Page N of M (c) Year Factiva, Inc." header.
2. Split on Dow Jones document IDs ("Document J000000020241230ekcu0001g") which
   reliably terminate every article.
3. Parse each article's structured header (title, author, word count, date,
   source, page reference) and isolate the body text.
4. Score each article for oil-relevance using a lightweight density metric:
       density = (oil keyword hits per 100 words)  x  title-match boost
5. Triage into three buckets:
       kept    -> high oil density AND oil keyword in title
       review  -> high oil density without title match (manual eyeball)
       dropped -> low oil density (incidental mentions only)

Outputs
-------
By default writes three JSON files plus a summary into the same directory as the
input PDF, named:
    <stem>__kept.json
    <stem>__review.json
    <stem>__dropped.json
    <stem>__summary.md

Usage
-----
    python 04_factiva_pdf_ingest.py /path/to/oil_2025_1.pdf
    python 04_factiva_pdf_ingest.py /path/to/oil_2025_1.pdf --out-dir /some/folder
    python 04_factiva_pdf_ingest.py /path/to/oil_2025_1.pdf --keep-thresh 0.4
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Optional, Tuple

import pypdf


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Factiva inserts a header at the top of every page. Two known formats:
#   2025 style:  "Page12of175 ©2026Factiva,Inc.Allrightsreserved."   (no spaces)
#   2026 style:  "Page 12 of 175 © 2026 Factiva, Inc. All rights reserved."  (spaces)
# So every literal must be followed by `\s*` to absorb optional whitespace.
PAGE_HEADER_RE = re.compile(
    r"Page\s*\d+\s*of\s*\d+\s*(?:©|\(c\))\s*\d{4}\s*"
    r"Factiva\s*,?\s*Inc\s*\.?\s*All\s*rights\s*reserved\s*\.?",
    re.IGNORECASE,
)

# Each article ends with a Dow Jones / Factiva document ID line such as:
#   "Document J000000020241230ekcu0001g"
# The ID is alphanumeric and 20+ chars. We use it as the article delimiter.
DOC_ID_RE = re.compile(r"\bDocument\s+([A-Za-z0-9]{16,})\b")

# Header pieces we look for in each article block (between start and Copyright line)
WORD_COUNT_RE = re.compile(r"^([\d,]+)\s*words?$", re.IGNORECASE)
DATE_RE = re.compile(
    r"^(\d{1,2})\s+("
    r"January|February|March|April|May|June|July|August|September|October|November|December"
    r")\s+(\d{4})$"
)
COPYRIGHT_RE = re.compile(
    r"(?:Copyright|\(c\)|©)\s*\d{4}",
    re.IGNORECASE,
)
BYLINE_RE = re.compile(r"^By\s+(.+)$")


# Oil-relevance vocabulary -- two tiers
OIL_BODY_RE = re.compile(
    r"\b(crude|oil|brent|wti|opec\+?|petroleum|barrel|barrels|"
    r"refinery|refineries|refining|drilling|shale|fracking|"
    r"gasoline|diesel|jet\s*fuel|lng|hydrocarbon|aramco|exxon|"
    r"chevron|conocophillips|pipeline|spr|strategic\s+petroleum|"
    r"upstream|downstream|midstream|wellhead|fracking|frack|"
    r"bpd|bbl|opec\+|saudi\s+arabia|venezuela\s+oil|russian\s+oil|"
    r"iranian\s+oil)\b",
    re.IGNORECASE,
)

OIL_TITLE_RE = re.compile(
    r"\b(crude|oil|brent|wti|opec\+?|petroleum|aramco|gasoline|refinery|"
    r"refining|shale|drilling|fracking|hydrocarbon|spr|exxon|chevron|"
    r"barrel)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Article:
    doc_id: str
    title: Optional[str] = None
    author: Optional[str] = None
    word_count: Optional[int] = None
    date: Optional[str] = None
    source: Optional[str] = None
    page_ref: Optional[str] = None
    body_text: str = ""
    # scoring
    oil_keyword_hits: int = 0
    oil_density_per_100w: float = 0.0
    title_oil_match: bool = False
    score: float = 0.0
    bucket: str = "unscored"  # kept / review / dropped
    # diagnostics
    parse_warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# PDF -> raw text
# ---------------------------------------------------------------------------

def extract_full_text(pdf_path: Path) -> str:
    """
    Read all pages, strip per-page Factiva headers, concatenate.

    Tries pdfplumber.extract_words first (best for Factiva PDFs where adjacent
    glyphs aren't separated by space chars and pypdf would produce
    "U.S.OilProductionStillStrong..."). Falls back to pypdf.extract_text if
    pdfplumber is unavailable.
    """
    try:
        import pdfplumber  # type: ignore

        pieces: List[str] = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                try:
                    words = page.extract_words(
                        x_tolerance=1.5,
                        y_tolerance=3,
                        keep_blank_chars=False,
                    )
                except Exception:
                    words = []
                if words:
                    # Group words by line (rounded y position), then join with spaces
                    lines: dict = {}
                    for w in words:
                        y = round(w["top"])
                        lines.setdefault(y, []).append(w["text"])
                    page_text = "\n".join(
                        " ".join(lines[y]) for y in sorted(lines.keys())
                    )
                else:
                    # Fallback for pages where extract_words returns nothing
                    page_text = page.extract_text() or ""
                page_text = PAGE_HEADER_RE.sub("\n", page_text)
                pieces.append(page_text)
        return "\n".join(pieces)

    except ImportError:
        # Fallback to pypdf
        reader = pypdf.PdfReader(str(pdf_path))
        pieces = []
        for page in reader.pages:
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            t = PAGE_HEADER_RE.sub("\n", t)
            pieces.append(t)
        return "\n".join(pieces)


# ---------------------------------------------------------------------------
# Splitting on Document IDs
# ---------------------------------------------------------------------------

def split_articles(full_text: str) -> List[Tuple[str, str]]:
    """
    Yield (doc_id, article_text) pairs by splitting on "Document ..." markers.
    Article text includes everything from the previous boundary up to and
    including the Document line of the current article.
    """
    matches = list(DOC_ID_RE.finditer(full_text))
    out: List[Tuple[str, str]] = []
    last_end = 0
    for m in matches:
        chunk = full_text[last_end : m.start()].strip()
        # The Document line itself isn't useful inside body text.
        out.append((m.group(1), chunk))
        last_end = m.end()
    # Anything after the final Document ID is unindexed trailing junk -- ignore it.
    return out


# ---------------------------------------------------------------------------
# Per-article header parsing
# ---------------------------------------------------------------------------

def _clean_lines(text: str) -> List[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def _looks_like_concatenated_title(line: str, prev_line: Optional[str]) -> bool:
    """
    Factiva often duplicates the title in a no-spaces concatenated form
    (e.g. "AgingU.S.ShaleImpedesTrump'sOilPlans"). Detect & drop those.
    """
    if not prev_line:
        return False
    # If line has no spaces but previous line does, and they share a lot of chars
    if " " in line:
        return False
    if len(line) < 8:
        return False
    # Compare alphanum-only versions
    a = re.sub(r"[^A-Za-z0-9]", "", line).lower()
    b = re.sub(r"[^A-Za-z0-9]", "", prev_line).lower()
    if not a or not b:
        return False
    # If the concatenated form is almost a substring of the spaced form, it's a dup
    return a[:30] in b or b[:30] in a


def parse_article(doc_id: str, raw: str) -> Article:
    """
    Parse one article chunk into structured fields.

    Layout (very consistent in Factiva PDFs):
        <Title>
        <TitleNoSpaces>     [optional artifact line]
        By <Author>         [optional]
        <N> words
        <DD Month YYYY>
        <Source name>
        <Source code>       [usually 1-6 chars, e.g. "J", "PLEE", "REUTERS"]
        <Page ref>          [optional, e.g. "B1", "A17"]
        English             [optional language line]
        Copyright YYYY ... Reserved.
        <BODY...>
    """
    art = Article(doc_id=doc_id)
    if not raw:
        art.parse_warnings.append("empty_chunk")
        return art

    # Find the copyright marker -- everything after the line containing it is body.
    # Different sources use different formats:
    #   "Copyright 2024 Dow Jones & Company, Inc. All Rights Reserved."     (WSJ)
    #   "(c)2024 copyright Junewarren-Nickle's Energy Group of Companies"   (Daily Oil Bulletin)
    #   "Copyright 2024 Australian Financial Review"                          (AFR)
    cm = COPYRIGHT_RE.search(raw)
    if not cm:
        # Fall back: use the raw chunk wholesale as body, no header
        art.body_text = raw.strip()
        art.parse_warnings.append("no_copyright_marker")
        return art

    # Skip to the end of the line containing the copyright marker so that the
    # rest of the copyright text doesn't bleed into the body.
    line_end = raw.find("\n", cm.end())
    if line_end == -1:
        line_end = len(raw)

    header_block = raw[: cm.start()]
    body = raw[line_end:].strip()
    # If we accidentally captured a Document line trailing the body, strip it
    body = re.sub(r"\s*Document\s+[A-Za-z0-9]{16,}\s*$", "", body)
    # Collapse runs of blank lines
    body = re.sub(r"\n{3,}", "\n\n", body)
    art.body_text = body.strip()

    lines = _clean_lines(header_block)

    # Pass through and pick out structured pieces
    title_candidates: List[str] = []
    prev_for_dup_check: Optional[str] = None
    for ln in lines:
        # word count
        m = WORD_COUNT_RE.match(ln)
        if m and art.word_count is None:
            try:
                art.word_count = int(m.group(1).replace(",", ""))
            except ValueError:
                pass
            continue
        # date
        m = DATE_RE.match(ln)
        if m and art.date is None:
            art.date = ln
            continue
        # byline
        m = BYLINE_RE.match(ln)
        if m and art.author is None:
            art.author = m.group(1).strip()
            continue
        # concatenated-title artifact
        if _looks_like_concatenated_title(ln, prev_for_dup_check):
            continue
        # everything else accumulates as a candidate
        title_candidates.append(ln)
        prev_for_dup_check = ln

    # Title is the first non-empty candidate that isn't a metadata line.
    # Source / page-ref usually come AFTER the date; titles come BEFORE byline.
    # We'll classify candidates by position relative to date/byline:
    if title_candidates:
        # Heuristic: title is the first candidate (the "spacy" version of the title)
        art.title = title_candidates[0]

    # Source: look for known source phrases or take the line right before the
    # short ALL-CAPS code that follows the date.
    for ln in lines:
        if ln in (
            "The Wall Street Journal",
            "The New York Times",
            "Reuters News",
            "Energy Intelligence",
            "Petroleum Intelligence Weekly",
            "International Oil Daily",
            "Oil Daily",
            "Platts Oilgram News",
            "Argus Crude",
            "Bloomberg",
            "Financial Times",
        ):
            art.source = ln
            break

    # Page ref: a short alphanumeric like B1 / A17 / Front Page
    for ln in lines:
        if re.match(r"^[A-Z]\d{1,3}$", ln):
            art.page_ref = ln
            break

    if not art.title:
        art.parse_warnings.append("no_title")
    if not art.date:
        art.parse_warnings.append("no_date")
    if not art.word_count:
        # fall back to len(body.split())
        if art.body_text:
            art.word_count = len(art.body_text.split())
            art.parse_warnings.append("inferred_word_count")
        else:
            art.parse_warnings.append("no_word_count")

    return art


# ---------------------------------------------------------------------------
# Oil-relevance scoring
# ---------------------------------------------------------------------------

def score_article(art: Article) -> None:
    """Compute oil-density score and assign a triage bucket."""
    text = art.body_text or ""
    title = art.title or ""

    body_hits = len(OIL_BODY_RE.findall(text))
    title_match = bool(OIL_TITLE_RE.search(title))

    wc = art.word_count or len(text.split())
    if wc <= 0:
        density = 0.0
    else:
        density = body_hits / (wc / 100.0)

    # Title-match boost: real oil-focused articles almost always say it in the headline.
    score = density * (1.6 if title_match else 1.0)

    art.oil_keyword_hits = body_hits
    art.oil_density_per_100w = round(density, 3)
    art.title_oil_match = title_match
    art.score = round(score, 3)


def triage(art: Article, keep_thresh: float, review_thresh: float) -> None:
    """Assign bucket based on score thresholds."""
    if art.title_oil_match and art.score >= keep_thresh:
        art.bucket = "kept"
    elif art.score >= review_thresh:
        art.bucket = "review"
    else:
        art.bucket = "dropped"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def process_pdf(
    pdf_path: Path,
    out_dir: Path,
    keep_thresh: float = 0.5,
    review_thresh: float = 1.5,
) -> dict:
    """End-to-end ingest. Returns a summary dict."""
    print(f"[ingest] reading {pdf_path}")
    text = extract_full_text(pdf_path)
    print(f"[ingest] extracted {len(text):,} chars")

    pairs = split_articles(text)
    print(f"[ingest] found {len(pairs)} document IDs (article candidates)")

    articles: List[Article] = []
    for doc_id, chunk in pairs:
        art = parse_article(doc_id, chunk)
        score_article(art)
        triage(art, keep_thresh=keep_thresh, review_thresh=review_thresh)
        articles.append(art)

    buckets = {"kept": [], "review": [], "dropped": []}
    for a in articles:
        buckets[a.bucket].append(asdict(a))

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = pdf_path.stem

    paths = {}
    for name, items in buckets.items():
        p = out_dir / f"{stem}__{name}.json"
        p.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
        paths[name] = p
        print(f"[ingest] {name:8s}: {len(items):4d} -> {p.name}")

    summary = {
        "input_pdf": str(pdf_path),
        "total_articles": len(articles),
        "kept": len(buckets["kept"]),
        "review": len(buckets["review"]),
        "dropped": len(buckets["dropped"]),
        "thresholds": {"keep": keep_thresh, "review": review_thresh},
        "output_files": {k: str(v) for k, v in paths.items()},
    }

    # Markdown summary for human eyeball review
    md = []
    md.append(f"# Factiva ingest summary -- {stem}")
    md.append("")
    md.append(f"- Source PDF: `{pdf_path}`")
    md.append(f"- Articles split: **{len(articles)}**")
    md.append(f"- Kept: **{summary['kept']}**, Review: **{summary['review']}**, Dropped: **{summary['dropped']}**")
    md.append(f"- Thresholds: keep>={keep_thresh} (with title match), review>={review_thresh}")
    md.append("")
    md.append("## Top 15 articles by score")
    md.append("")
    md.append("| Score | Bucket | Title | Date | Hits | Density |")
    md.append("|-------|--------|-------|------|------|---------|")
    for a in sorted(articles, key=lambda x: -x.score)[:15]:
        title = (a.title or "")[:80].replace("|", "\\|")
        md.append(
            f"| {a.score:.2f} | {a.bucket} | {title} | {a.date or '-'} | "
            f"{a.oil_keyword_hits} | {a.oil_density_per_100w:.2f} |"
        )
    md.append("")
    md.append("## Bottom 10 dropped (lowest score)")
    md.append("")
    md.append("| Score | Title | Date |")
    md.append("|-------|-------|------|")
    for a in sorted(articles, key=lambda x: x.score)[:10]:
        title = (a.title or "")[:80].replace("|", "\\|")
        md.append(f"| {a.score:.2f} | {title} | {a.date or '-'} |")
    md.append("")
    md.append("## Articles flagged for manual review")
    md.append("")
    md.append("(High oil density but no oil keyword in title -- worth a glance)")
    md.append("")
    for a in articles:
        if a.bucket == "review":
            title = (a.title or "")[:90]
            md.append(f"- **{a.score:.2f}** {title} -- {a.date or '-'}")
    md.append("")
    md_path = out_dir / f"{stem}__summary.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    summary["summary_md"] = str(md_path)
    print(f"[ingest] summary -> {md_path.name}")

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdf", type=Path, help="Input Factiva PDF dossier")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where to write the JSON outputs (default: same dir as PDF)",
    )
    parser.add_argument(
        "--keep-thresh",
        type=float,
        default=0.5,
        help="Min score for an article with a title match to be 'kept' (default 0.5)",
    )
    parser.add_argument(
        "--review-thresh",
        type=float,
        default=1.5,
        help="Min score for an article without a title match to land in 'review' (default 1.5)",
    )
    args = parser.parse_args()

    if not args.pdf.exists():
        print(f"ERROR: {args.pdf} does not exist", file=__import__('sys').stderr)
        return 1

    out_dir = args.out_dir or args.pdf.parent
    summary = process_pdf(
        args.pdf,
        out_dir=out_dir,
        keep_thresh=args.keep_thresh,
        review_thresh=args.review_thresh,
    )

    print("\n" + json.dumps(
        {k: v for k, v in summary.items() if k != "output_files"},
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
