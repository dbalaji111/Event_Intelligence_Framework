"""
redflag_integration.py
======================
Bridge between your existing scripts and the Red Flag API.

Your existing scripts:          This file:
─────────────────────────────────────────────────────────
r02_text_analyser.py         →  clean_html(), embed_text()
r02_text_analysis_with_all_attributes.py → run_all_reasoners()
r04_fetch_time_series.py     →  get_timeseries_window()
r04_time_series_visualiser.py → get_term_structure_figure()
redflag_api.py               →  score_headline()

Usage
-----
    # Add to your FastAPI app (redflag_api.py):
    from redflag_integration import IntegrationBridge
    bridge = IntegrationBridge(cfg)

    # In your /analyze endpoint:
    llm_attrs = bridge.run_all_reasoners(headline)
    ts_data   = bridge.get_timeseries_window(event_date, window=15)
"""

from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRATION BRIDGE
# ─────────────────────────────────────────────────────────────────────────────

class IntegrationBridge:
    """
    Wires your existing r02 and r04 scripts into the Red Flag API.

    Connects:
      - HTML cleaning              (from r02_text_analyser.py)
      - OpenAI embeddings          (from r02_text_analyser.py)
      - LLM 8-attribute extraction (from r02_text_analysis_with_all_attributes.py)
      - Time series data           (from r04_fetch_time_series.py)
      - Config paths               (from src/config.py)
    """

    # The 8 reasoners from r02_text_analysis_with_all_attributes.py
    REASONERS = [
        ("event_type_reasoner",        "event_type"),
        ("event_severity_reasoner",    "event_severity"),
        ("demand_shock_reasoner",      "demand_shock"),
        ("supply_shock_reasoner",      "supply_shock"),
        ("geopolitical_risk_reasoner", "geopolitical_risk"),
        ("impact_horizon_reasoner",    "impact_horizon"),
        ("affected_region_reasoner",   "affected_region"),
        ("sentiment_reasoner",         "sentiment"),
    ]

    def __init__(self,
                 data_dir:       Optional[str] = None,
                 agents_dir:     str = "data_reasoning_agents",
                 openai_api_key: Optional[str] = None,
                 use_openai:     bool = False,
                 use_config:     bool = True):
        """
        Initialise bridge.

        If use_config=True (default), automatically loads paths from
        your src/config.py Config class:
            Config.DATA_DIR
            Config.OIL_TIMESERIES_FILE
            Config.OIL_HEADLINES_PROCESSED_ALL_ATTRIBUTES_FILE

        Otherwise, data_dir is used directly.
        """
        self._ts_df   = None
        self._openai  = None
        self._cfg     = None

        # ── Load Config from your src/config.py ───────────────────────────
        if use_config:
            try:
                from src.config import Config
                self._cfg        = Config
                self.data_dir    = Path(Config.DATA_DIR)
                self.ts_file     = Path(Config.OIL_TIMESERIES_FILE)
                self.attrs_file  = Path(Config.OIL_HEADLINES_PROCESSED_ALL_ATTRIBUTES_FILE)
                print(f"[bridge] Config loaded — data_dir: {self.data_dir}")
            except ImportError:
                print("[bridge] src.config not found — using fallback paths")
                self.data_dir   = Path(data_dir or "data")
                self.ts_file    = self.data_dir / "oil_timeseries.csv"
                self.attrs_file = self.data_dir / "oil_headlines_processed_all_attributes.csv"
        else:
            self.data_dir   = Path(data_dir or "data")
            self.ts_file    = self.data_dir / "oil_timeseries.csv"
            self.attrs_file = self.data_dir / "oil_headlines_processed_all_attributes.csv"

        self.agents_dir = Path(agents_dir)

        # ── OpenAI client (same as your r02 scripts) ──────────────────────
        if use_openai and openai_api_key:
            try:
                from openai import OpenAI
                self._openai = OpenAI(api_key=openai_api_key)
                print("[bridge] OpenAI client loaded")
            except ImportError:
                print("[bridge] openai not installed — using keyword fallback")

    # ── HTML CLEANING (from r02_text_analyser.py) ─────────────────────────

    def clean_html(self, html_text: str) -> str:
        """
        Exact copy of clean_html_text from r02_text_analyser.py.
        Removes HTML, links, scripts, and formatting.
        """
        if not html_text or not str(html_text).strip():
            return ""

        if HAS_BS4:
            soup = BeautifulSoup(html_text, "html.parser")
            for tag in soup(["script", "style"]):
                tag.decompose()
            text = soup.get_text(separator=" ")
        else:
            text = re.sub(r"<[^>]+>", " ", html_text)

        text = re.sub(r"\s+", " ", text).strip()
        return text

    # ── SAFE JSON PARSER (from r02_text_analyse_with_attributes.py) ───────

    def safe_parse(self, content: str) -> Dict:
        """
        Exact copy of safe_parse from r02_text_analyse_with_attributes.py.
        """
        try:
            return json.loads(content)
        except Exception:
            content = content.replace("```json", "").replace("```", "").strip()
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except Exception:
                    return {}
            return {}

    # ── AGENT LOADER (from r02_text_analysis_with_all_attributes.py) ──────

    def load_agent(self, agent_name: str) -> Optional[str]:
        """
        Load a reasoning agent prompt from data_reasoning_agents/.
        """
        path = self.agents_dir / f"{agent_name}.md"
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    # ── SINGLE REASONER ────────────────────────────────────────────────────

    def run_reasoner(self, text: str,
                     agent_name: str,
                     output_key: str) -> str:
        """
        Run one LLM reasoning agent on the text.
        Mirrors run_reasoner() from r02_text_analysis_with_all_attributes.py.
        Uses OpenAI if available, else returns keyword fallback.
        """
        if not text or not text.strip():
            return "unknown"

        # OpenAI path (from your r02 scripts)
        if self._openai:
            agent_prompt = self.load_agent(agent_name)
            if not agent_prompt:
                return "unknown"

            full_prompt = agent_prompt.replace("{text}", str(text))

            try:
                from time import sleep
                response = self._openai.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=0,
                    messages=[{"role": "user", "content": full_prompt}]
                )
                content = response.choices[0].message.content
                parsed  = self.safe_parse(content)
                sleep(0.3)  # rate limit
                return parsed.get(output_key, "unknown")
            except Exception as e:
                print(f"[bridge] LLM error ({agent_name}): {e}")
                return "unknown"

        # Fallback: keyword-based (no API call)
        return self._keyword_fallback(text, output_key)

    def _keyword_fallback(self, text: str, output_key: str) -> str:
        """
        Fast keyword fallback when OpenAI is not available.
        Mirrors the logic from r02_text_analysis scripts.
        """
        t = text.lower()
        fallbacks = {
            "event_type": (
                "Supply Shock"      if any(w in t for w in
                    ["opec","production cut","barrel","output","supply","pipeline"])
                else "Geopolitical Risk" if any(w in t for w in
                    ["war","military","iran","russia","sanctions","attack","conflict"])
                else "Demand Shock"  if any(w in t for w in
                    ["demand","pmi","china","slowdown","recession","consumption"])
                else "Monetary Policy" if any(w in t for w in
                    ["fed","rate hike","inflation","ecb","central bank"])
                else "Other"
            ),
            "event_severity": (
                "Extreme" if any(w in t for w in
                    ["war","invasion","attack","collapse","unprecedented","crisis"])
                else "High" if any(w in t for w in
                    ["major","significant","shock","sanctions","military"])
                else "Moderate" if any(w in t for w in
                    ["cut","increase","decision","agreement"])
                else "Low"
            ),
            "supply_shock":        "Yes" if any(w in t for w in
                ["opec","production","supply","output","barrel","pipeline","tanker"])
                                   else "No",
            "demand_shock":        "Yes" if any(w in t for w in
                ["demand","pmi","china","recession","consumption","slowdown"])
                                   else "No",
            "geopolitical_risk":   "Yes" if any(w in t for w in
                ["war","military","iran","russia","sanctions","conflict","attack"])
                                   else "No",
            "sentiment":           (
                "Bearish"  if any(w in t for w in
                    ["cut","decline","fall","collapse","sanction","attack","war"])
                else "Bullish" if any(w in t for w in
                    ["increase","rise","resume","ceasefire","agreement","boost"])
                else "Neutral"
            ),
            "impact_horizon":      (
                "Immediate" if any(w in t for w in
                    ["today","immediately","breaking","now","attack","war"])
                else "Short-term" if any(w in t for w in
                    ["week","monthly","quarter","q1","q2","q3","q4"])
                else "Medium-term"
            ),
            "affected_region":     (
                "Middle East" if any(w in t for w in
                    ["iran","saudi","opec","gulf","hormuz","iraq","uae"])
                else "Eastern Europe" if any(w in t for w in
                    ["russia","ukraine","europe"])
                else "Asia" if any(w in t for w in
                    ["china","asia","japan","korea"])
                else "Global"
            ),
        }
        return fallbacks.get(output_key, "unknown")

    # ── ALL 8 REASONERS ────────────────────────────────────────────────────

    def run_all_reasoners(self, text: str) -> Dict[str, str]:
        """
        Run all 8 reasoning agents.
        Mirrors the REASONERS loop in r02_text_analysis_with_all_attributes.py.

        Returns dict with all 8 attributes.
        """
        clean = self.clean_html(text)
        results = {}
        for agent_name, output_key in self.REASONERS:
            results[output_key] = self.run_reasoner(clean, agent_name, output_key)
        return results

    # ── TIME SERIES (from r04_fetch_time_series.py) ────────────────────────

    def load_timeseries(self) -> Optional[pd.DataFrame]:
        """
        Load oil_timeseries.csv built by r04_fetch_time_series.py.
        """
        if self._ts_df is not None:
            return self._ts_df

        candidates = [
            self.data_dir / "oil_timeseries.csv",
            self.data_dir / "oil_prices_full.xlsx",
        ]
        for path in candidates:
            if path.exists():
                try:
                    if path.suffix == ".csv":
                        df = pd.read_csv(path)
                    else:
                        df = pd.read_excel(path)
                    df["Date"] = pd.to_datetime(df["Date"] if "Date" in df.columns
                                                else df.iloc[:, 0], errors="coerce")
                    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
                    df = df.ffill(limit=5)
                    self._ts_df = df
                    print(f"[bridge] timeseries loaded: {len(df)} rows from {path.name}")
                    return df
                except Exception as e:
                    print(f"[bridge] timeseries load failed ({path}): {e}")
        return None

    def get_timeseries_window(self,
                               event_date: str,
                               window: int = 15) -> Optional[Dict]:
        """
        Extract price window around event_date.
        Mirrors get_event_window() from r04_time_series_visualiser.py.
        Returns dict ready for Plotly JSON.
        """
        df = self.load_timeseries()
        if df is None:
            return None

        event_dt = pd.to_datetime(event_date, errors="coerce")
        if pd.isna(event_dt):
            return None

        start = event_dt - pd.Timedelta(days=window)
        end   = event_dt + pd.Timedelta(days=window)
        subset = df.loc[start:end]

        if subset.empty:
            return None

        # Snap event to nearest trading date
        nearest = df.index[abs(df.index - event_dt).argmin()]

        # Build traces (same columns as r04_fetch_time_series.py output)
        wti_cols   = [c for c in subset.columns if "WTI" in c.upper()]
        brent_cols = [c for c in subset.columns if "BRENT" in c.upper() or "Brent" in c]

        traces = []
        colors = ["#2E75B6","#5BA3D9","#88C0E8","#E97132","#F2A96B","#FAD4A2"]

        for i, col in enumerate((wti_cols + brent_cols)[:6]):
            traces.append({
                "x":     [str(d.date()) for d in subset.index],
                "y":     subset[col].ffill().tolist(),
                "name":  col,
                "color": colors[i % len(colors)],
            })

        return {
            "traces":     traces,
            "event_date": str(nearest.date()),
            "start":      str(start.date()),
            "end":        str(end.date()),
            "n_rows":     len(subset),
        }

    # ── FULL PIPELINE (called from /analyze endpoint) ──────────────────────

    def enrich_analysis(self, result: Dict,
                         raw_text: str = "",
                         event_date: Optional[str] = None) -> Dict:
        """
        Enriches the base API result with:
          1. LLM 8 attributes (from your r02 scripts)
          2. Time series window data (from your r04 scripts)

        Call this from the /analyze endpoint after score_headline().
        """
        # 1. LLM attributes
        text = raw_text or result.get("headline", "")
        result["llm_attributes"] = self.run_all_reasoners(text)

        # 2. Time series window
        date_to_use = event_date
        if not date_to_use and result.get("similar_events"):
            date_to_use = result["similar_events"][0].get("date")

        if date_to_use:
            ts = self.get_timeseries_window(date_to_use, window=15)
            result["timeseries_window"] = ts

        return result


# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI ENDPOINT ADDITIONS
# Add these routes to redflag_api.py
# ─────────────────────────────────────────────────────────────────────────────

FASTAPI_ADDITIONS = '''
# ── ADD TO redflag_api.py ─────────────────────────────────────────────────

from redflag_integration import IntegrationBridge

# Initialise bridge with your Config paths
bridge = IntegrationBridge(
    data_dir       = "data",
    agents_dir     = "data_reasoning_agents",
    openai_api_key = os.environ.get("OPENAI_API_KEY"),
    use_openai     = True,  # set False for fast keyword-only mode
)


@app.post("/analyze_full")
def analyze_full(news: NewsInput):
    """
    Full pipeline including LLM 8-attribute extraction
    and time series window from your r04 scripts.
    """
    scorer = get_scorer()
    result = scorer.score_headline(news.headline, news.source, news.timestamp)
    result = bridge.enrich_analysis(result, raw_text=news.headline)
    return result


@app.get("/timeseries")
def get_timeseries(date: str, window: int = 15):
    """
    Returns price window around event date.
    Data from r04_fetch_time_series.py output (oil_timeseries.csv).
    """
    ts = bridge.get_timeseries_window(date, window)
    if ts is None:
        from fastapi import HTTPException
        raise HTTPException(404, "No timeseries data for this date")
    return ts
'''


if __name__ == "__main__":
    # Quick test
    bridge = IntegrationBridge(use_openai=False)

    test_headlines = [
        "Saudi Arabia unexpectedly cuts oil production by 500,000 bpd",
        "US military buildup near Iran threatens Strait of Hormuz",
        "China PMI collapses to 47.8 signaling industrial slowdown",
    ]

    print("=" * 60)
    print("INTEGRATION BRIDGE — KEYWORD FALLBACK TEST")
    print("=" * 60)

    for hl in test_headlines:
        attrs = bridge.run_all_reasoners(hl)
        print(f"\nHeadline: {hl[:50]}...")
        for k, v in attrs.items():
            print(f"  {k:<25}: {v}")