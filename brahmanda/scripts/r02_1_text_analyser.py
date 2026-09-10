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

input_file = data_dir / "oil_headlines.csv"
output_file = data_dir / "oil_headlines_processed.csv"

   
    

    # -----------------------------
    # Text Cleaning
    # -----------------------------

def clean_html_text(html_text: str) -> str:
    """
    Remove HTML, links, scripts, and formatting
    """
    soup = BeautifulSoup(html_text, "html.parser")

    # remove scripts/styles
    for tag in soup(["script", "style"]):
        tag.decompose()

    text = soup.get_text(separator=" ")

    # remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# -----------------------------
# Embedding Generation
# -----------------------------

def generate_embedding(text: str):

    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text,
        encoding_format="float"
    )

    return response.data[0].embedding


# -----------------------------
# Load Data

# -----------------------------

df = pd.read_csv(input_file)

print(f"Loaded {len(df)} news articles")

# -----------------------------
# Clean Text
# -----------------------------

df["clean_text"] = df["full_text"].apply(clean_html_text)

# -----------------------------
# Generate Embeddings
# -----------------------------

embeddings = []
for i, row in df.iterrows():
    try:
        text = row["clean_text"]

        # ✅ Guard against empty text
        if not text or not text.strip():
            print(f"Row {i+1}: empty text, skipping")
            embeddings.append(None)
            continue

        emb = generate_embedding(text)

        if emb is not None:
            print(f"Processed {i+1}/{len(df)} | embedding length = {len(emb)}")
        else:
            print(f"Processed {i+1}/{len(df)} | embedding FAILED")

        embeddings.append(emb)

    except Exception as e:
        print(f"Embedding failed at row {i+1}: {e}")
        embeddings.append(None)

# ✅ Attach embeddings to DataFrame (this was the missing step)
df["embedding"] = embeddings

# ✅ Serialize lists to JSON strings so CSV can store them
df["embedding"] = df["embedding"].apply(lambda x: json.dumps(x) if x is not None else None)

# Save
df.to_csv(output_file, index=False)

print(f"\nEmbeddings generated: {df['embedding'].notna().sum()}/{len(df)}")


# -----------------------------
# Placeholder for LLM Attributes
# (to implement later)
# -----------------------------

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


# -----------------------------
# Save Results
# -----------------------------

if output_file.exists():

    existing = pd.read_csv(output_file)

    combined = pd.concat([existing, df])

    combined = combined.drop_duplicates(subset="storyId")

    combined.to_csv(output_file, index=False)

else:

    df.to_csv(output_file, index=False)


print("\nProcessing complete")
print(f"Saved to: {output_file}")

