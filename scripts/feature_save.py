import joblib
import polars as pl
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
import os

# Configuration
DATA_PATH = "../data/merged_final_2.parquet"
OUTPUT_PIPELINE_PATH = "../weights/feature_pipeline.joblib"

def main():
    print(f"Loading data from {DATA_PATH}...")
    df = pl.read_parquet(DATA_PATH)
    
    # 1. Label Encoder
    print("Fitting LabelEncoder...")
    y = df["classification"].to_numpy()
    le = LabelEncoder()
    le.fit(y)

    # 2. Text Preparation
    print("Preparing text data...")
    text_data = df['text'].fill_null('').to_list()
    domain_data = df['domain'].fill_null('').to_list()
    combined_text = [f"{t} {d}" for t, d in zip(text_data, domain_data)]

    # 3. Vectorizers
    print("Fitting Vectorizers (Count & TF-IDF)...")
    count_vec = CountVectorizer(max_features=10000, stop_words='english', min_df=2, max_df=0.95, ngram_range=(1, 2))
    tfidf_vec = TfidfVectorizer(max_features=10000, stop_words='english', min_df=2, max_df=0.95, ngram_range=(1, 2))
    
    # We also initialize HashingVectorizer, though it doesn't strictly need 'fitting' in the same way,
    # we save it to preserve the configuration.
    hash_vec = HashingVectorizer(n_features=30, alternate_sign=False, norm='l2', 
                                token_pattern=r'(?u)\b\w{2,}\b', 
                                stop_words=['com', 'www', 'http', 'https', 'org', 'net'])

    count_matrix = count_vec.fit_transform(combined_text).astype(np.float32)
    tfidf_matrix = tfidf_vec.fit_transform(combined_text).astype(np.float32)

    # 4. SVD Reducers
    print("Fitting SVD reductions...")
    svd_count = TruncatedSVD(n_components=20, random_state=42)
    svd_tfidf = TruncatedSVD(n_components=20, random_state=42)
    svd_neighbor = TruncatedSVD(n_components=20, random_state=42)

    svd_count.fit(count_matrix)
    svd_tfidf.fit(tfidf_matrix)
    svd_neighbor.fit(count_matrix) # Neighbor embeddings use the count matrix space

    # 5. Save everything
    pipeline_data = {
        'le': le,
        'count_vec': count_vec,
        'tfidf_vec': tfidf_vec,
        'hash_vec': hash_vec,
        'svd_count': svd_count,
        'svd_tfidf': svd_tfidf,
        'svd_neighbor': svd_neighbor
    }

    os.makedirs(os.path.dirname(OUTPUT_PIPELINE_PATH), exist_ok=True)
    joblib.dump(pipeline_data, OUTPUT_PIPELINE_PATH)
    print(f"Success! Pipeline resources saved to {OUTPUT_PIPELINE_PATH}")

if __name__ == "__main__":
    main()