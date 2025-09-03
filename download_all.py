import requests
from dotenv import load_dotenv
import os
from sec_api import QueryApi
import time
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
API_KEY = os.getenv("API_KEY")

# queryApi = QueryApi(api_key=API_KEY)
print(API_KEY)

def download_filings_by_strategy(ticker, queryApi, num_10k=5, other_forms_limit=10):
    """
    Downloads SEC filings for a given ticker based on a strategic approach.

    1. Fetches the 'num_10k' most recent 10-K filings to establish a date range.
    2. Fetches ALL 10-Q and 8-K filings within that date range.
    3. Fetches the 'other_forms_limit' most recent DEF 14A, 3, 4, and 5 filings.
    4. Downloads the HTML version of each filing and saves it.
    5. Saves a consolidated JSON file with metadata for all downloaded filings.

    Args:
        ticker (str): The company ticker symbol (e.g., "AAPL").
        queryApi (QueryApi): An initialized instance of the sec_api QueryApi.
        num_10k (int): The number of recent 10-K filings to base the date range on.
        other_forms_limit (int): The max number of recent filings for less critical forms.
    """
    # --- Setup ---
    save_dir = f"{ticker}_sec_filings"
    os.makedirs(save_dir, exist_ok=True)
    headers = {
        "User-Agent": "Gautam Kumar gautam.baranwal2003@gmail.com"
    }
    all_metadata = []
    processed_accession_nos = set()
    count = 0

    # --- Helper function to process and save a single filing ---
    def process_and_save_filing(filing):
        """Inner function to handle downloading and metadata extraction for one filing."""
        nonlocal count
        accession_no = filing.get("accessionNo")
        if not accession_no or accession_no in processed_accession_nos:
            return False

        url = filing.get("linkToFilingDetails")
        if not url:
            print(f"Warning: No URL found for filing {accession_no}. Skipping.")
            return False

        try:
            ticker_from_filing = filing.get("ticker", "UNKNOWN")
            form_type = filing.get("formType", "UNKNOWN").replace("/", "-")
            filed_at = filing.get("filedAt", "")[:10]
            
            filename = f"{ticker_from_filing}_{form_type}_{filed_at}_{accession_no.replace('-', '')}.html"
            filepath = os.path.join(save_dir, filename)

            r = requests.get(url, headers=headers, timeout=20)
            r.raise_for_status()
            with open(filepath, "wb") as f:
                f.write(r.content)
            
            print(f"Saved: {filename}")

            metadata_to_save = {
                "filename": filename,
                "ticker": ticker_from_filing,
                "form_type": form_type,
                "filed_at": filed_at,
                "period_of_report": filing.get("periodOfReport"),
                "items": filing.get("items", []),
                "url": url
            }
            all_metadata.append(metadata_to_save)
            processed_accession_nos.add(accession_no)
            count += 1
            return True
        except Exception as e:
            print(f"Failed to download or process {url}: {e}")
            return False

    # Fetch 10-K filings to establish a date range ---
    print(f"\n--- Step 1: Fetching last {num_10k} 10-K filings for {ticker} ---")
    search_params_10k = {
        "query": f'ticker:{ticker} AND formType:"10-K"',
        "from": "0", "size": str(num_10k), "sort": [{"filedAt": {"order": "desc"}}]
    }
    response_10k = queryApi.get_filings(search_params_10k)
    filings_10k = response_10k.get("filings", [])

    if not filings_10k:
        print(f"Could not find any 10-K filings for {ticker}. Aborting.")
        return

    for f in filings_10k:
        process_and_save_filing(f)
    
    ten_k_dates = [f.get("filedAt") for f in filings_10k]
    start_date = min(ten_k_dates)[:10]
    end_date = max(ten_k_dates)[:10]
    print(f"Date range established: {start_date} to {end_date}")

    # Fetch critical filings (10-Q, 8-K) within the date range ---
    print(f"\n--- Step 2: Fetching ALL 10-Q and 8-K filings from {start_date} to {end_date} ---")
    critical_forms = ["10-Q", "8-K"]
    form_type_query_part = " OR ".join([f'formType:"{t}"' for t in critical_forms])
    
    offset = 0
    batch_size = 200
    while True:
        search_params_critical = {
            "query": f"ticker:{ticker} AND ({form_type_query_part}) AND filedAt:[{start_date} TO {end_date}]",
            "from": str(offset), "size": str(batch_size), "sort": [{"filedAt": {"order": "desc"}}]
        }
        response_critical = queryApi.get_filings(search_params_critical)
        filings_critical = response_critical.get("filings", [])
        if not filings_critical:
            break
        for f in filings_critical:
            process_and_save_filing(f)
        offset += len(filings_critical)
        time.sleep(1)

    # Fetch a limited number of other filings (DEF 14A, 3, 4, 5) ---
    print(f"\n--- Step 3: Fetching RECENT (max {other_forms_limit}) other filings ---")
    limited_forms = ["DEF 14A", "3", "4", "5"]
    for form_type in limited_forms:
        print(f"Fetching recent {form_type} filings...")
        search_params_limited = {
            "query": f"ticker:{ticker} AND formType:\"{form_type}\" AND filedAt:[{start_date} TO {end_date}]",
            "from": "0", "size": str(other_forms_limit), "sort": [{"filedAt": {"order": "desc"}}]
        }
        response_limited = queryApi.get_filings(search_params_limited)
        filings_limited = response_limited.get("filings", [])
        for f in filings_limited:
            process_and_save_filing(f)
        time.sleep(1)

    # Save all collected metadata ---
    print("\n--- Step 4: Saving all collected metadata ---")
    metadata_filepath = os.path.join(save_dir, f"{ticker}_metadata.json")
    with open(metadata_filepath, "w") as f:
        json.dump(all_metadata, f, indent=4)
    
    print(f"Metadata for {len(all_metadata)} filings saved to {metadata_filepath}")
    print(f"\nTotal filings downloaded for {ticker}: {count}")


# Execution block 
if __name__ == "__main__":
    load_dotenv()
    API_KEY = os.getenv("API_KEY")

    if not API_KEY:
        raise ValueError("API_KEY not found. Please create a .env file and add your API_KEY from sec-api.io.")
        
    queryApi = QueryApi(api_key=API_KEY)
    
    tickers = [
        "AAPL", "MSFT", "JPM", "JNJ", "XOM",
        "GOOG", "AMZN", "TSLA", "UNH", "V", 
        "PG", "HD", "CVX", "MRK", "BAC"
    ]
    
    for ticker in tickers:
        try:
            print(f"==================================================")
            print(f"Starting download process for ticker: {ticker}")
            print(f"==================================================")
            download_filings_by_strategy(ticker, queryApi, num_10k=5, other_forms_limit=10)
            print(f"\nSuccessfully completed download for {ticker}.")
        except Exception as e:
            print(f"An error occurred while processing {ticker}: {e}")
        
        time.sleep(2)
