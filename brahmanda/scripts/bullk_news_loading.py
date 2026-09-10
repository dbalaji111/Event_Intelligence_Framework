import eikon as ek
import pandas as pd
import datetime
import time
from pathlib import Path

from src.config import Config


def fetch_news(query, start_date, end_date, count=100):
    try:
        return ek.get_news_headlines(
            query,
            date_from=start_date.strftime('%Y-%m-%dT%H:%M:%S'),
            date_to=end_date.strftime('%Y-%m-%dT%H:%M:%S'),
            count=count
        )
    except Exception as e:
        print(f"Fetch error: {e}")
        return pd.DataFrame()


def fetch_full_story(story_id):
    try:
        return ek.get_news_story(story_id)
    except Exception as e:
        print(f"Story fetch failed: {story_id} → {e}")
        return ""


def main():
    # ================= CONFIG =================
    ek.set_app_key(Config.REFINITIV_APP_KEY)

    data_dir = Path(Config.DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)

    output_file = data_dir / "bulk_oil_news.csv"

    query = 'R:RTRS AND (crude oil) AND (supply OR demand OR production OR inventory)'

    start_date = datetime.datetime(2024, 1, 1, 0, 0)
    end_date   = datetime.datetime(2025, 4, 15, 23, 59)

    batch_size = 100

    all_data = []

    current_end = end_date

    # ================= BULK LOOP =================
    while current_end > start_date:

        current_start = current_end - datetime.timedelta(days=1)
        temp_end = current_end

        print(f"\nWindow: {current_start} → {current_end}")

        while True:
            df = fetch_news(query, current_start, temp_end, batch_size)

            if df.empty:
                break

            df = df.copy()
            df["full_text"] = ""

            # ================= FULL STORY FETCH =================
            for i, row in df.iterrows():
                df.at[i, "full_text"] = fetch_full_story(row["storyId"])
                time.sleep(0.2)  # prevent API throttling

            all_data.append(df)

            # pagination
            last_time = df.index[-1]
            temp_end = last_time - datetime.timedelta(seconds=1)

            print(f"Fetched {len(df)} rows")

            if len(df) < batch_size:
                break

        # move window backward
        current_end = current_start

        # ================= CHECKPOINT SAVE =================
        combined = pd.concat(all_data)

        # deduplicate (important)
        combined = combined[~combined.index.duplicated(keep="first")]

        combined.to_csv(output_file, index=True)

        print(f"Checkpoint saved: {len(combined)} rows")

    print("\nBulk retrieval complete.")
    print(f"Saved to: {output_file.resolve()}")


if __name__ == "__main__":
    main()