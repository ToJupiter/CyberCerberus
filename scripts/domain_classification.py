import json
import csv
import os
import asyncio
from itertools import islice
import pandas as pd
from openai import AsyncOpenAI
from dotenv import load_dotenv
load_dotenv()

client = AsyncOpenAI(
    api_key=os.environ.get("GROQ_API_KEY", ""),
    base_url="https://api.groq.com/openai/v1"
)

MODEL_NAME = "openai/gpt-oss-120b"


print("Loading merged_nodes.parquet...")
try:
    df_parquet = pd.read_parquet("data/edge_nodes.parquet")
    df_parquet['text'] = df_parquet['text'].astype(str).str[:200]
    domain_text_map = dict(zip(df_parquet['domain'], df_parquet['text']))
    print(f"Loaded {len(domain_text_map)} domain entries from Parquet.")
except Exception as e:
    print(f"Error loading parquet file: {e}")
    domain_text_map = {}

system_prompt = """
You are an expert at classifying websites based on their domain names and content text.
Classify the provided websites into one of the following three categories:
1. "good" - Legitimate, safe, and appropriate websites for general audiences (e.g., google.com, wikipedia.org, news sites).
2. "adult" - Websites containing adult content, inappropriate for children (e.g., explicit content, pornography).
3. "illegal" - Websites promoting or facilitating illegal activities (e.g., gambling, drug sales, fraud, piracy).

Analyze the domain and the provided text snippet. Assign the most appropriate label.
If you are uncertain, choose "good" as the default.

INPUT FORMAT:
Each line contains: ID, Domain, TextSnippet

EXAMPLE INPUT:
1,google.com,"Search the world's information..."
2,xxx-videos.com,"Explicit adult videos..."
3,drug-market.net,"Buy illegal substances..."

EXAMPLE OUTPUT:
{
  "1": "good",
  "2": "adult",
  "3": "illegal"
}

Return a JSON object where keys are the numbers from the input and values are "good", "adult", or "illegal".
"""

async def classify_domains_batch(domains_with_numbers):
    domains_str = "\n".join([
        f"{num},{domain},{domain_text_map.get(domain, '')}" 
        for num, domain in domains_with_numbers
    ])
    
    user_prompt = f"Classify the following websites:\n{domains_str}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            response_format={'type': 'json_object'},
            temperature=0.0
        )
        result = json.loads(response.choices[0].message.content)
        return [result.get(str(num), "good") for num, _ in domains_with_numbers]
    except Exception as e:
        print(f"API Error: {e}")
        return ["good" for _ in domains_with_numbers]

async def process_batch(batch_rows):
    domains_with_numbers = [(row['number'], row['domain']) for row in batch_rows]
    return await classify_domains_batch(domains_with_numbers)

def read_csv_in_batches(file_path, batch_size):
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        while True:
            batch = list(islice(reader, batch_size))
            if not batch:
                break
            yield batch

async def main():
    semaphore = asyncio.Semaphore(80)
    
    async def controlled_process_batch(batch_rows):
        async with semaphore:
            return await process_batch(batch_rows)
    
    input_file = "data/edge_domains.csv"
    output_file = "data/classified_edge_domains.csv"
    
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames 
    
    print(f"Starting classification from {input_file}...")
    
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        
        tasks = []
        for batch_rows in read_csv_in_batches(input_file, 100):
            task = controlled_process_batch(batch_rows)
            tasks.append((task, batch_rows))
        
        print(f"Created {len(tasks)} batch tasks. Processing...")
        
        completed_count = 0
        for task, batch_rows in tasks:
            classifications = await task
            for i, row in enumerate(batch_rows):
                row['classification'] = classifications[i]
                writer.writerow(row)
                completed_count += 1
            
            if completed_count % 100 == 0:
                print(f"Progress: {completed_count} domains processed...")

    print(f"Done! Results saved to {output_file}")

if __name__ == "__main__":
    asyncio.run(main())