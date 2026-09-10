# ============================================
# LLM Attribute Extraction Script (WORKING)
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
output_file = data_dir / "oil_headlines_processed_attributes.csv"

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
# Event Type Reasoner
# -----------------------------
def run_event_type_reasoner(text):

    if pd.isna(text) or text.strip() == "":
        return "unknown"

    agent_prompt = load_agent("event_type_reasoner")

    # ✅ Correct placeholder replacement
    full_prompt = agent_prompt.replace("{text}", str(text))

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[{"role": "user", "content": full_prompt}]
        )

        content = response.choices[0].message.content

        parsed = safe_parse(content)

        if "event_type" not in parsed:
            print("⚠️ PARSE FAILED:", content)
            return "unknown"

        return parsed["event_type"]

    except Exception as e:
        print("❌ ERROR:", e)
        return "unknown"

# -----------------------------
# Apply Reasoner
# -----------------------------
event_types = []

for i, row in df.iterrows():

    raw_text = row.get("full_text", "")

    if pd.isna(raw_text):
        raw_text = ""

    # ✅ Clean HTML → readable text
    text = clean_html(raw_text)

    # Debug first few rows
    if i < 3:
        print("\n--- CLEAN TEXT ---")
        print(text[:200])

    event_type = run_event_type_reasoner(text)

    print(f"Row {i} → {event_type}")

    event_types.append(event_type if event_type else "unknown")

    if i % 10 == 0:
        print(f"Processed {i}/{len(df)}")

    sleep(0.3)

# -----------------------------
# Assign Column
# -----------------------------
if len(event_types) != len(df):
    print("❌ Length mismatch!")
    print("events:", len(event_types), "df:", len(df))

df["event_type"] = event_types

# -----------------------------
# Verify Output
# -----------------------------
print("\n--- SAMPLE OUTPUT ---")
print(df[["event_type"]].head(10))

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