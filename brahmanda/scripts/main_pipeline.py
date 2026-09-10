# scripts/run_pipeline.py

def main():
    print("Step 1: API Check")
    #import scripts.r01_check_api.py

    print("Step 2: Fetch Data")
    import scripts.r02_fetch_and_store_timeseries

    print("Step 3: Analyse Text")
    import scripts.r02_1_text_analyser

if __name__ == "__main__":
    main()