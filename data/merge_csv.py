import pandas as pd
import numpy as np

# Read the CSV files
df1 = pd.read_csv('nsfw_domains.csv')  # Contains 'status' column
df2 = pd.read_csv('gambling_domains.csv')
df3 = pd.read_csv('classified_domains.csv')

# Keep only the shared columns from all files
shared_cols = ['number', 'domain', 'classification']

# Filter to keep only shared columns
df1 = df1[shared_cols] if all(col in df1.columns for col in shared_cols) else df1
df2 = df2[shared_cols] if all(col in df2.columns for col in shared_cols) else df2
df3 = df3[shared_cols] if all(col in df3.columns for col in shared_cols) else df3

# Keep all domains from df2 and df3 (including shared ones)
df2_domains = df2['domain'].tolist()
df3_domains = df3['domain'].tolist()

# Create a set of domains already included from df2 and df3
already_included_domains = set(df2_domains + df3_domains)

# Filter df1 to exclude domains already in df2 or df3
df1_filtered = df1[~df1['domain'].isin(already_included_domains)].copy()

# Create a priority score for domains in df1
def get_priority_score(domain):
    """Higher score = higher priority"""
    if pd.isna(domain):
        return 0
    
    domain_str = str(domain).lower()
    # Highest priority for .com domains
    if domain_str.endswith('.com'):
        return 100
    # Second priority for .net domains
    elif domain_str.endswith('.net'):
        return 50
    # Third priority for .org domains
    elif domain_str.endswith('.org'):
        return 25
    else:
        return 0

# Add priority score column
df1_filtered['priority_score'] = df1_filtered['domain'].apply(get_priority_score)

# Sort by priority score (highest first)
df1_filtered = df1_filtered.sort_values('priority_score', ascending=False)

# Check how many domains we need to select
n = min(10000, len(df1_filtered))

# Select top n domains from sorted list
df1_selected = df1_filtered.head(n)

# Remove the priority score column before merging
df1_selected = df1_selected.drop(columns=['priority_score'])

# Combine all dataframes
merged_df = pd.concat([df2, df3, df1_selected], ignore_index=True)

# Optional: Add a source column to track where each row came from
# merged_df['source_file'] = 'gambling_domains.csv'
# merged_df.loc[len(df2):len(df2)+len(df3)-1, 'source_file'] = 'classified_domains.csv'
# merged_df.loc[len(df2)+len(df3):, 'source_file'] = 'nsfw_domains.csv'

# Save to new CSV file
merged_df.to_csv('merged_domains.csv', index=False)
print(f"Length of merged df: {len(merged_df)}")
merged_df = merged_df.drop_duplicates(subset=['domain'])
print(f"Length of new merged df: {len(merged_df)}")

# Print summary statistics
print(f"Total domains in merged file: {len(merged_df)}")
print(f"  From gambling_domains.csv: {len(df2)}")
print(f"  From classified_domains.csv: {len(df3)}")
print(f"  From nsfw_domains.csv: {len(df1_selected)}")
print(f"\n.com domains from nsfw selection: {df1_selected['domain'].str.endswith('.com').sum()}")