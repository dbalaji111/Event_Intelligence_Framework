from pathlib import Path

from src.config import Config
import eikon as ek

query = '(crude oil) AND  (production OR supply OR output OR inventory OR stockpile OR demand OR Crisis OR shortage )'

def main():
    cfg = Config()
    #ATA_DIR = BASE_DIR / os.getenv("DATA_DIR", "data")

    data_dir = Path(Config.DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)

    print("Refinitiv key loaded:")
    print(Config.REFINITIV_APP_KEY)
    print("Data storage path:")
    print(data_dir.resolve())

    ek.set_app_key(Config.REFINITIV_APP_KEY)

   

   

    oil_news = ek.get_news_headlines(
        query=query,
        count=100
    )
    oil_news["full_text"] = ""

    # fetch full text using storyId
    for i, row in oil_news.iterrows():
        try:
            story = ek.get_news_story(row["storyId"])
            oil_news.at[i, "full_text"] = story
        except Exception as e:
            print(f"Failed to fetch story {row['storyId']}: {e}")
            oil_news.at[i, "full_text"] = ""
    if data_dir.exists():
        oil_news.to_csv(data_dir / "oil_headlines_test.csv", mode="a", header=False, index=False)
    else:
        oil_news.to_csv(data_dir / "oil_headlines_test.csv", mode="w", header=True, index=False)

    print("\nOil headlines test:")
    print(oil_news)
    print(f"\nSaved files to: {data_dir.resolve()}")



if __name__ == "__main__":
    main()

