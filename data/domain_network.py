import polars as pl

# Load base domains (with number, domain, classification)
base_df = pl.read_csv("merged_domains.csv")

# Extract base domains as a set for fast lookup
base_domains = set(base_df["domain"])

# Load source-target links from Parquet
links_df = pl.read_parquet("merged_edges.parquet")

# Filter targets that are NOT in base domains
new_targets = links_df.filter(
    ~pl.col("target").is_in(base_domains)
)

# Count how many times each target appears (i.e., number of links from base domains)
target_counts = new_targets.group_by("target").agg(
    pl.count().alias("link_count")
)

# Sort by link_count descending (highest first)
target_counts = target_counts.sort("link_count", descending=True)

# Add dummy 'number' and blank 'classification' columns
result = target_counts.with_row_index("number", offset=1).select([
    pl.col("number"),
    pl.col("target").alias("domain"),
    pl.lit("").alias("classification")
])

# Write to CSV
result.write_csv("new_targets_by_link_count.csv")