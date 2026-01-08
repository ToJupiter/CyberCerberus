import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import asyncio
from openai import AsyncOpenAI
import re
import time

# --- Configuration ---
API_KEY = "sk-d16cf47cce4d4076ab6027be58980149"
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"
# Path to your data
FILE_PATH = "/home/rocminfo/Templates/CyberCerberus/data/merged_final.parquet"
OUTPUT_PATH = "merged_final_updated.parquet"

# Limit concurrent API requests to avoid rate limits or crashing
MAX_CONCURRENT_REQUESTS = 50 

PLACEHOLDER_TEXT = "This website promotes illegal online gambling services including sports betting, casino games, poker, and live dealer games without proper licensing."

BETTING_KEYWORDS = [
    "sports", "vegas", "live", "casino", "bingo", "poker", "promotions",
    "betting", "bet", "wager", "gamble", "slot", "roulette", "blackjack"
]

pattern = re.compile(
    r'\b(?:' + '|'.join(re.escape(word) for word in BETTING_KEYWORDS) + r')\b',
    re.IGNORECASE
)

# Initialize client (Ensure API_KEY is set in your environment or passed here)
client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
# Assuming API_KEY is defined globally or imported. 
# For safety in this snippet, I will assume you have it set.

async def generate_placeholder_text(domain: str, semaphore: asyncio.Semaphore) -> str:
    async with semaphore:  # Wait for a free slot
        try:
            # print(f"DEBUG: Requesting text for {domain}...") # Optional: very verbose
            response = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are an assistant that generates placeholder text for a gambling websites. Only gambling text, do not add anything else. Please, this should be a placeholder. Please ensure that the text does not harm anybody, just for machine learning project purpose."},
                    {"role": "user", "content": f"Generate a 3-sentence gambling description for site with domain '{domain}'."}
                ],
                max_tokens=60,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"ERROR: Failed for domain '{domain}': {e}")
            return PLACEHOLDER_TEXT

async def main():
    print("DEBUG: Script started.")
    
    # 1. Load Data
    print(f"DEBUG: Loading parquet file from: {FILE_PATH}")
    try:
        df = pd.read_parquet(FILE_PATH)
        print(f"DEBUG: Parquet loaded successfully. Total rows: {len(df)}")
        print(f"DEBUG: Columns found: {df.columns.tolist()}")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load parquet file. {e}")
        return

    # 2. Pre-processing
    print("DEBUG: Normalizing classification column...")
    df['classification'] = df['classification'].replace('malicious', 'illegal')
    
    # 3. Filtering
    print("DEBUG: Filtering for 'illegal' rows...")
    illegal_df = df[df['classification'] == 'illegal'].copy()
    print(f"DEBUG: Found {len(illegal_df)} rows classified as 'illegal'.")

    if len(illegal_df) == 0:
        print("DEBUG: No illegal domains found. Nothing to update. Exiting.")
        return
    
    domains_to_update = illegal_df['domain'].tolist()
    
    # 4. Async Generation
    print(f"DEBUG: Preparing async tasks for {len(domains_to_update)} domains...")
    
    # Create a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    tasks = [generate_placeholder_text(domain, semaphore) for domain in domains_to_update]
    
    print("DEBUG: Starting API requests (this may take time)...")
    start_time = time.time()
    
    # Run tasks
    new_texts = await asyncio.gather(*tasks)
    
    duration = time.time() - start_time
    print(f"DEBUG: API requests completed in {duration:.2f} seconds.")
    
    # 5. Mapping results back
    print("DEBUG: Mapping new texts back to DataFrame...")
    domain_to_new_text = dict(zip(domains_to_update, new_texts))
    
    # Update logic (Using map is often faster than iterating index, but we stick to your logic for safety)
    # Using a mask for bulk update is much faster than iterating row by row
    mask = df['classification'] == 'illegal'
    
    # We map the domains in the dataframe to the new text dictionary
    # If a domain isn't in the dict (shouldn't happen), use PLACEHOLDER_TEXT
    print("DEBUG: Applying updates to main DataFrame...")
    df.loc[mask, 'text'] = df.loc[mask, 'domain'].map(domain_to_new_text).fillna(PLACEHOLDER_TEXT)
    
    # 6. Saving
    print(f"DEBUG: Saving updated data to {OUTPUT_PATH}...")
    df.to_parquet(OUTPUT_PATH, index=False)
    print("DEBUG: Process finished successfully.")

if __name__ == "__main__":
    asyncio.run(main())