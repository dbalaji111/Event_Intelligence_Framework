"""
browse_events.py
================
Browse all events in a JSON or JSONL memory file.
Shows event text, metadata, and price info in a readable format.

Usage
-----
    # Browse all events (paginated)
    python browse_events.py --file "memory_2001_2023/price_event_memory.json"

    # Search for specific text
    python browse_events.py --file "memory_2001_2023/price_event_memory.json" --search "iran strait"

    # Filter by year
    python browse_events.py --file "memory_2001_2023/price_event_memory.json" --year 2022

    # Filter by event type
    python browse_events.py --file "memory_2001_2023/price_event_memory.json" --type "Geopolitical"

    # Show specific event by ID
    python browse_events.py --file "memory_2001_2023/price_event_memory.json" --event_id EVT_2022.0_0042.0

    # Export filtered results to CSV
    python browse_events.py --file "memory_2001_2023/price_event_memory.json" --search "opec" --export results.csv

    # Show just a summary table (no full text)
    python browse_events.py --file "memory_2001_2023/price_event_memory.json" --summary_only

    # Show N events per page (default 5)
    python browse_events.py --file "memory_2001_2023/price_event_memory.json" --page_size 10
"""

import argparse
import json
import re
import sys
import os
from pathlib import Path
from typing import List, Dict, Optional

# ── ANSI colours for terminal output ─────────────────────────────────────────
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    RED    = "\033[91m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    BLUE   = "\033[94m"
    CYAN   = "\033[96m"
    GREY   = "\033[90m"
    WHITE  = "\033[97m"

def strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", " ", s)

def clean_text(s: str, max_len: int = None) -> str:
    s = strip_ansi(str(s)).strip()
    s = re.sub(r"\s+", " ", s)
    if max_len and len(s) > max_len:
        s = s[:max_len] + "..."
    return s

def safe_float(v, default=None):
    try:
        f = float(v)
        return f if f == f else default  # NaN check
    except:
        return default

# ── Loader ────────────────────────────────────────────────────────────────────

def load_file(path: Path) -> List[Dict]:
    if not path.exists():
        print(f"{C.RED}ERROR: File not found: {path}{C.RESET}")
        sys.exit(1)

    size_mb = path.stat().st_size / 1e6
    print(f"{C.GREY}Loading {path.name} ({size_mb:.0f} MB)...{C.RESET}", end=" ", flush=True)

    with path.open("r", encoding="utf-8", errors="replace") as f:
        first = ""
        for line in f:
            first = line.strip()
            if first: break

    events = []
    if first.startswith("["):
        with path.open("r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
        events = data if isinstance(data, list) else list(data.values())
    else:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    events.append(json.loads(line))
                except:
                    pass

    # Strip bert_embeddings to save memory
    for e in events:
        e.pop("bert_embeddings", None)
        e.pop("Trigger_embeddings", None)

    print(f"{C.GREEN}{len(events)} events loaded{C.RESET}")
    return events

# ── Filtering ─────────────────────────────────────────────────────────────────

def filter_events(events: List[Dict],
                  search: Optional[str] = None,
                  year: Optional[int] = None,
                  event_type: Optional[str] = None,
                  event_id: Optional[str] = None) -> List[Dict]:
    result = events

    if event_id:
        result = [e for e in result
                  if str(e.get("event_id","")).lower() == event_id.lower()]

    if year:
        result = [e for e in result
                  if int(float(e.get("year", e.get("Year", 0)) or 0)) == year]

    if event_type:
        result = [e for e in result
                  if event_type.lower() in str(
                      e.get("Event type", e.get("Event_type",""))
                  ).lower()]

    if search:
        terms = search.lower().split()
        def matches(e):
            blob = " ".join([
                str(e.get("text","")),
                str(e.get("cleaned_text","")),
                str(e.get("Entity","")),
                str(e.get("Primary Event","")),
                str(e.get("event_id","")),
            ]).lower()
            return all(t in blob for t in terms)
        result = [e for e in result if matches(e)]

    return result

# ── Display ───────────────────────────────────────────────────────────────────

def display_event(e: Dict, idx: int, total: int, summary_only: bool = False):
    eid       = e.get("event_id", "N/A")
    date      = e.get("date", "N/A")
    year      = e.get("year", e.get("Year", "N/A"))
    etype     = clean_text(e.get("Event type", e.get("Event_type", "N/A")), 40)
    category  = clean_text(e.get("category", "N/A"), 40)
    polarity  = e.get("Polarity", "N/A")
    quantile  = e.get("Quantile_Category", "N/A")
    entity    = clean_text(e.get("Entity", "N/A"), 80)
    primary   = clean_text(e.get("Primary Event", "N/A"), 100)
    price     = safe_float(e.get("price") or e.get("Crude Oil-WTI Spot Cushing U$/BBL"))
    r7        = safe_float(e.get("price_change_t_minus_7_to_t_plus_7"))
    r30       = safe_float(e.get("price_change_t_minus_30_to_t_plus_30"))
    influence = safe_float(e.get("Influence_Score", 0))

    # Colour polarity
    pol_colour = C.GREEN if "pos" in str(polarity).lower() else \
                 C.RED   if "neg" in str(polarity).lower() else C.YELLOW

    # Price change colour
    def price_str(v, suffix="%"):
        if v is None: return f"{C.GREY}N/A{C.RESET}"
        col = C.GREEN if v > 0 else C.RED if v < 0 else C.GREY
        return f"{col}{v:+.2f}{suffix}{C.RESET}"

    sep = f"{C.GREY}{'─'*72}{C.RESET}"
    print(f"\n{sep}")
    print(f"{C.BOLD}{C.WHITE}[{idx}/{total}] {eid}{C.RESET}  "
          f"{C.CYAN}{date}{C.RESET}  "
          f"{C.GREY}year={year}{C.RESET}")
    print(f"  {C.BOLD}Type:{C.RESET}     {etype}")
    print(f"  {C.BOLD}Category:{C.RESET} {category}")
    print(f"  {C.BOLD}Polarity:{C.RESET} {pol_colour}{polarity}{C.RESET}  "
          f"{C.BOLD}Quantile:{C.RESET} {quantile}  "
          f"{C.BOLD}Influence:{C.RESET} {influence:.1f}")

    if price is not None:
        print(f"  {C.BOLD}Price:{C.RESET}    ${price:.2f}/bbl  "
              f"r(±7d)={price_str(r7)}  r(±30d)={price_str(r30)}")

    print(f"  {C.BOLD}Entities:{C.RESET} {C.CYAN}{entity}{C.RESET}")
    print(f"  {C.BOLD}Primary:{C.RESET}  {primary}")

    if not summary_only:
        # Full text
        text = clean_text(e.get("text", e.get("cleaned_text", "")))
        if text and text != "N/A":
            print(f"\n  {C.BOLD}TEXT:{C.RESET}")
            # Word-wrap at 70 chars
            words = text.split()
            line, lines = [], []
            for w in words:
                line.append(w)
                if len(" ".join(line)) > 70:
                    lines.append("  " + " ".join(line[:-1]))
                    line = [w]
            if line:
                lines.append("  " + " ".join(line))
            for l in lines[:30]:   # max 30 lines
                print(f"  {C.GREY}{l}{C.RESET}")
            if len(lines) > 30:
                print(f"  {C.GREY}  ... [{len(lines)-30} more lines]{C.RESET}")

        # Causal links
        causal = clean_text(e.get("causal_links_1", e.get("Causal Links", "")), 300)
        if causal and causal not in ("N/A", "nan", "#NAME?", ""):
            print(f"\n  {C.BOLD}CAUSAL LINKS:{C.RESET}")
            print(f"  {C.GREY}{causal}{C.RESET}")

# ── Summary table ─────────────────────────────────────────────────────────────

def display_summary_table(events: List[Dict]):
    from collections import Counter

    print(f"\n{C.BOLD}{'─'*80}{C.RESET}")
    print(f"{C.BOLD}{'EVENT ID':<25} {'DATE':<12} {'TYPE':<22} {'POL':<10} {'Q-CAT':<12} {'r30':>7}{C.RESET}")
    print(f"{'─'*80}")

    for e in events:
        eid      = str(e.get("event_id","N/A"))[:24]
        date     = str(e.get("date","N/A"))[:10]
        etype    = clean_text(e.get("Event type", e.get("Event_type","N/A")), 21)
        polarity = str(e.get("Polarity","N/A"))[:9]
        qcat     = str(e.get("Quantile_Category","N/A"))[:11]
        r30      = safe_float(e.get("price_change_t_minus_30_to_t_plus_30"))

        pol_col = C.GREEN if "pos" in polarity.lower() else \
                  C.RED   if "neg" in polarity.lower() else C.YELLOW
        r30_str = f"{r30:+.1f}%" if r30 is not None else "N/A"
        r30_col = C.GREEN if (r30 or 0) > 0 else C.RED if (r30 or 0) < 0 else C.GREY

        print(f"{C.CYAN}{eid:<25}{C.RESET} "
              f"{date:<12} "
              f"{etype:<22} "
              f"{pol_col}{polarity:<10}{C.RESET} "
              f"{qcat:<12} "
              f"{r30_col}{r30_str:>7}{C.RESET}")

    print(f"{'─'*80}")
    print(f"  Total: {len(events)} events")

    # Quick stats
    years  = Counter(int(float(e.get("year",0) or 0)) for e in events)
    etypes = Counter(clean_text(e.get("Event type", e.get("Event_type","")), 30)
                     for e in events)
    pols   = Counter(e.get("Polarity","N/A") for e in events)

    print(f"\n  Years: {dict(sorted(years.items()))}")
    print(f"  Top event types:")
    for k, v in etypes.most_common(5):
        print(f"    {k:<35} {v}")
    print(f"  Polarity: {dict(pols)}")

# ── Export ────────────────────────────────────────────────────────────────────

def export_csv(events: List[Dict], path: str):
    import csv
    cols = ["event_id","date","year","Event type","category","Polarity",
            "Quantile_Category","Entity","Primary Event","price",
            "price_change_t_minus_7_to_t_plus_7",
            "price_change_t_minus_30_to_t_plus_30",
            "Influence_Score","text"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for e in events:
            row = {k: clean_text(str(e.get(k,"")), 500) for k in cols}
            w.writerow(row)
    print(f"{C.GREEN}Exported {len(events)} events → {path}{C.RESET}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file",         required=True,
                    help="path to JSON or JSONL memory file")
    ap.add_argument("--search",       default=None,
                    help="search terms (space-separated, all must match)")
    ap.add_argument("--year",         type=int, default=None,
                    help="filter by year (e.g. 2022)")
    ap.add_argument("--type",         default=None,
                    help="filter by event type (partial match)")
    ap.add_argument("--event_id",     default=None,
                    help="show specific event by ID")
    ap.add_argument("--summary_only", action="store_true",
                    help="show table only, no full text")
    ap.add_argument("--page_size",    type=int, default=5,
                    help="events per page (default 5)")
    ap.add_argument("--export",       default=None,
                    help="export filtered results to CSV")
    ap.add_argument("--no_page",      action="store_true",
                    help="print all results without pagination")
    args = ap.parse_args()

    events = load_file(Path(args.file))

    filtered = filter_events(
        events,
        search     = args.search,
        year       = args.year,
        event_type = args.type,
        event_id   = args.event_id,
    )

    print(f"{C.BOLD}{len(filtered)}{C.RESET} events match your filter "
          f"(out of {len(events)} total)\n")

    if len(filtered) == 0:
        print(f"{C.YELLOW}No events found. Try different filters.{C.RESET}")
        return

    # Export if requested
    if args.export:
        export_csv(filtered, args.export)
        return

    # Summary table always shown first
    display_summary_table(filtered)

    if args.summary_only:
        return

    # Paginated full display
    page_size = args.page_size
    total     = len(filtered)
    idx       = 0

    while idx < total:
        batch = filtered[idx:idx + page_size]
        for i, event in enumerate(batch):
            display_event(event, idx + i + 1, total,
                          summary_only=args.summary_only)

        idx += page_size
        if idx >= total:
            print(f"\n{C.GREEN}── End of results ──{C.RESET}")
            break

        if not args.no_page:
            print(f"\n{C.BOLD}[{idx}/{total} shown]  "
                  f"Press ENTER for next {page_size}  |  "
                  f"'s' to skip to end  |  "
                  f"'q' to quit{C.RESET}  ", end="")
            try:
                choice = input().strip().lower()
            except (EOFError, KeyboardInterrupt):
                break
            if choice == "q":
                break
            if choice == "s":
                args.no_page = True

if __name__ == "__main__":
    main()
    