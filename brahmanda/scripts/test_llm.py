import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# -----------------------------
# Setup
# -----------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------------
# Load Agent
# -----------------------------
def load_agent(agent_name):
    path = Path(f"data_reasoning_agents/{agent_name}.md")

    if not path.exists():
        print("❌ Agent file NOT found:", path)
        return None

    print("✅ Agent file loaded:", path)
    return path.read_text()

# -----------------------------
# Test Function
# -----------------------------
def test_llm():

    # Example input
    test_text = "Oil supply disrupted due to geopolitical tensions"

    agent_prompt = load_agent("event_type_reasoner")

    if agent_prompt is None:
        return

    # Replace placeholder correctly
    full_prompt = agent_prompt.replace("{text}", test_text)

    print("\n==============================")
    print("🔹 FINAL PROMPT SENT TO LLM:")
    print("==============================")
    print(full_prompt[:1000])  # print first part only

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[{"role": "user", "content": full_prompt}]
        )

        content = response.choices[0].message.content

        print("\n==============================")
        print("🔹 RAW LLM OUTPUT:")
        print("==============================")
        print(content)

    except Exception as e:
        print("\n❌ ERROR:", e)

# -----------------------------
# Run Test
# -----------------------------
if __name__ == "__main__":
    test_llm()