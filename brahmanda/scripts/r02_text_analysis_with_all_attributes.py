# ============================================
# LLM Multi-Reasoner Attribute Extraction Script
# ============================================

import os
import json
import re
import pandas as pd
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from src.config import Config
from time import sleep
from bs4 import BeautifulSoup

# -----------------------------
# Setup
# -----------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

data_dir = Path(Config.DATA_DIR)
data_dir.mkdir(parents=True, exist_ok=True)

input_file = data_dir / "oil_headlines_processed.csv"
output_file = data_dir / "oil_headlines_processed_all_attributes.csv"

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_csv(input_file)
print(f"Loaded {len(df)} rows")

# -----------------------------
# Agent Loader
# -----------------------------
def load_agent(agent_name):
    path = Path(f"data_reasoning_agents/{agent_name}.md")

    if not path.exists():
        raise FileNotFoundError(f"Agent not found: {path}")

    return path.read_text()

# -----------------------------
# Clean HTML
# -----------------------------
def clean_html(text):
    try:
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text(separator=" ", strip=True)
    except:
        return ""

# -----------------------------
# Safe JSON Parser
# -----------------------------
def safe_parse(content):
    try:
        return json.loads(content)
    except:
        content = content.replace("```json", "").replace("```", "").strip()
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                return {}
        return {}

# -----------------------------
# Generic Reasoner
# -----------------------------
def run_reasoner(text, agent_name, output_key):

    if pd.isna(text) or text.strip() == "":
        return "unknown"

    agent_prompt = load_agent(agent_name)
    full_prompt = agent_prompt.replace("{text}", str(text))

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "user", "content": full_prompt}
            ]
        )

        content = response.choices[0].message.content
        parsed = safe_parse(content)

        if output_key not in parsed:
            print(f"⚠️ PARSE FAILED ({agent_name}):", content)
            return "unknown"

        return parsed[output_key]

    except Exception as e:
        print(f"❌ ERROR in {agent_name}:", e)
        return "unknown"

# -----------------------------
# Define All Reasoners
# -----------------------------
REASONERS = [
    ("event_type_reasoner", "event_type"),
    ("event_severity_reasoner", "event_severity"),
    ("demand_shock_reasoner", "demand_shock"),
    ("supply_shock_reasoner", "supply_shock"),
    ("geopolitical_risk_reasoner", "geopolitical_risk"),
    ("impact_horizon_reasoner", "impact_horizon"),
    ("affected_region_reasoner", "affected_region"),
    ("sentiment_reasoner", "sentiment"),  # enable when ready
]

# -----------------------------
# Initialize Results Storage
# -----------------------------
results = {key: [] for _, key in REASONERS}

# -----------------------------
# Main Processing Loop
# -----------------------------
for i, row in df.iterrows():

    raw_text = row.get("full_text", "")

    if pd.isna(raw_text):
        raw_text = ""

    text = clean_html(raw_text)

    # Debug first few rows
    if i < 3:
        print("\n--- CLEAN TEXT ---")
        print(text[:200])

    for agent_name, output_key in REASONERS:

        value = run_reasoner(text, agent_name, output_key)

        print(f"Row {i} → {output_key}: {value}")

        results[output_key].append(value if value else "unknown")

    if i % 10 == 0:
        print(f"Processed {i}/{len(df)}")

    sleep(0.3)  # Avoid rate limits

# -----------------------------
# Assign Columns
# -----------------------------
for key in results:
    if len(results[key]) != len(df):
        print(f"❌ Length mismatch in {key}")
    df[key] = results[key]

# -----------------------------
# Verify Output
# -----------------------------
print("\n--- SAMPLE OUTPUT ---")
print(df[list(results.keys())].head(10))

# -----------------------------
# Save Output
# -----------------------------
if output_file.exists():

    existing = pd.read_csv(output_file)

    combined = pd.concat([existing, df])
    combined = combined.drop_duplicates(subset="storyId")

    combined.to_csv(output_file, index=False)

else:
    df.to_csv(output_file, index=False)

print("\n✅ Processing complete")
print(f"Saved to: {output_file}")