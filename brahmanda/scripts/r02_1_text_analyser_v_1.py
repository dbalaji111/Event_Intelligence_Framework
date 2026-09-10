import os
import re
import json
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
from openai import OpenAI
from dotenv import load_dotenv
from src.config import Config
# -----------------------------
# Setup
# -----------------------------

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#data_dir = Path("data")
cfg = Config()
    #ATA_DIR = BASE_DIR / os.getenv("DATA_DIR", "data")

data_dir = Path(Config.DATA_DIR)
data_dir.mkdir(parents=True, exist_ok=True)

input_file = data_dir / "oil_headlines_processed.csv"
output_file = data_dir / "oil_headlines_processed_attributes.csv"

"""
Future attributes extraction:

- event_type
- event_severity
- supply_shock
- demand_shock
- geopolitical_risk
- sentiment
- impact_horizon
- affected_region

Use an LLM classification prompt later.
"""



if output_file.exists():

    existing = pd.read_csv(output_file)

    combined = pd.concat([existing, df])

    combined = combined.drop_duplicates(subset="storyId")

    combined.to_csv(output_file, index=False)

else:

    df.to_csv(output_file, index=False)


print("\nProcessing complete")
print(f"Saved to: {output_file}")