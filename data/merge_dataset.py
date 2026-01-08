import polars as pl
import re
import os

def count_english_words(text):
    """Count the number of English words in text after cleaning.
    Returns the count of words that match common English patterns."""
    if not text or not isinstance(text, str):
        return 0
    
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    text = re.sub(r'[^\w\s]', '', text)
    
    text = re.sub(r'\d+', '', text)
    
    words = text.lower().split()
    
    common_english_words = {
        'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
        'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
        'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
        'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
        'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me'
    }
    
    english_word_count = sum(1 for word in words if word in common_english_words)
    
    return english_word_count

classified_domains = pl.scan_csv("classified_domains.csv")
classified_edge_domains = pl.scan_csv("classified_edge_domains.csv")

all_classifications = pl.concat([
    classified_domains.select(["domain", "classification"]),
    classified_edge_domains.select(["domain", "classification"])
], how="vertical")  # Vertical concatenation adds rows [[9]]

# Read Parquet files
edges_nodes = pl.scan_parquet("edge_nodes.parquet")
merged_nodes = pl.scan_parquet("merged_nodes.parquet")

# Concatenate Parquet files vertically
all_texts = pl.concat([
    edges_nodes.select(["domain", "text"]),
    merged_nodes.select(["domain", "text"])
], how="vertical")

merged_data = all_classifications.join(
    all_texts,
    on="domain",
    how="inner"
)

final_data = merged_data.filter(
    pl.col("text").map_elements(count_english_words) >= 5
)

result = final_data.select(["domain", "classification", "text"]).collect()

output_file = "merged_final.parquet"
result.write_parquet(output_file)

print(f"Successfully merged files into {output_file}")
print(f"Final row count: {len(result)}")
print(f"Schema: {result.schema}")