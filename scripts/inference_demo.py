import sys
import os
import re
import math
import asyncio
import logging
import warnings
from collections import Counter
from urllib.parse import urlparse
import numpy as np
import polars as pl
import joblib
import aiohttp
import lxml.html
import tldextract

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.ERROR)

# --- CONFIGURATION ---
DATA_PATH = "../data/merged_final_2.parquet"
MODEL_PATH = "../weights/best_model.joblib"
PIPELINE_PATH = "../weights/feature_pipeline.joblib"

ADULT_KEYWORDS = {'porn', 'xxx', 'sex', 'adult', 'nude', 'cams', 'erotic', 'hentai', 'milf', 'boobs', 'pussy', 'ass', 'gay', 'lesbian', 'fuck', 'anal', 'cum', 'cunt', 'dick', 'cock', 'cam'}
ILLEGAL_KEYWORDS = {'bet', 'casino', 'poker', 'slot', 'gamble', 'wager', 'odds', 'bingo', 'roulette', 'blackjack', 'dice', 'lottery', 'jackpot', 'vegas', 'bookie', 'sportsbook', 'betting'}

class SimpleSpider:
    def __init__(self):
        self.headers = {"User-Agent": "Mozilla/5.0 (compatible; WebsiteCategorizer/1.0)"}
        self.re_script = re.compile(r'<(script|style|noscript|iframe)[^>]*>.*?</\1>|', re.IGNORECASE | re.DOTALL)
        self.re_url_text = re.compile(r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9]{1,6}\b(?:[-a-zA-Z0-9@:%_\+.~#?&//=]*)')

    def _clean_parse(self, url, html):
        links = set()
        clean_html = self.re_script.sub('', html)
        try:
            doc = lxml.html.fromstring(clean_html)
            doc.make_links_absolute(url)
            hrefs = doc.xpath('//a/@href | //link/@href')
            for link in hrefs:
                links.add(link)
            text = " ".join(doc.xpath('//body//text()')).strip()
            text = re.sub(r'\s+', ' ', text)
        except:
            text = ""
        
        text_links = self.re_url_text.findall(text)
        links.update(text_links)
        
        normalized_links = set()
        for l in links:
            try:
                ext = tldextract.extract(l)
                if ext.suffix and ext.domain:
                    normalized_links.add(f"{ext.domain}.{ext.suffix}")
            except:
                continue
        return text, list(normalized_links)

    async def crawl(self, url):
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(url, headers=self.headers) as resp:
                    if resp.status == 200:
                        html = await resp.text()
                        return self._clean_parse(url, html)
        except:
            pass
        return "", []

class FeatureProcessor:
    def __init__(self):
        self.df = None
        self.model = None
        self.resources = None

    def load_resources(self):
        print("Loading pipeline resources and model...")
        self.resources = joblib.load(PIPELINE_PATH)
        
        self.model = joblib.load(MODEL_PATH)
        
        self.df = pl.read_parquet(DATA_PATH)
        print("Resources loaded successfully.")

    def _calc_entropy(self, s):
        if not s: return 0.0
        counts = Counter(s)
        length = len(s)
        return -sum((c/length) * math.log2(c/length) for c in counts.values())

    def _extract_url_feats(self, url):
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.split(':')[0]
            path, query, fragment = parsed.path, parsed.query, parsed.fragment
        except:
            domain, path, query, fragment = "", "", "", ""
        
        full_text = f"{domain} {path} {query} {fragment}"
        domain_parts = domain.split('.')
        tld = domain_parts[-1] if domain_parts else ""
        domain_name = domain_parts[-2] if len(domain_parts) > 1 else ""

        feats = [
            sum(1 for k in ADULT_KEYWORDS if k in full_text),
            sum(1 for k in ILLEGAL_KEYWORDS if k in full_text),
            1 if tld in {'xxx', 'sex', 'porn', 'adult', 'cam'} else 0,
            1 if tld in {'bet', 'casino', 'poker', 'bingo'} else 0,
            len([p for p in path.split('/') if p]),
            len([p for p in query.split('&') if p]),
            len(domain_name),
            int('https' in url),
            int('http' in url and 'https' not in url),
            int(any(c.isdigit() for c in domain_name)),
            int('-' in domain_name),
            int('_' in domain_name),
            int('.' in domain_name),
            int(any(k in domain_name for k in ADULT_KEYWORDS)),
            int(any(k in domain_name for k in ILLEGAL_KEYWORDS))
        ]
        
        special_chars = ['.', '-', '@', '?', '&', '=', '_', '%', '+', '#', '$', '!']
        stats = [len(url), (sum(c.isdigit() for c in url)/(len(url)+1))]
        stats.extend([url.count(c) for c in special_chars])
        stats.extend([
            1 if url.startswith("https") else 0,
            1 if re.match(r"^(http://|https://)?\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", url) else 0,
            sum(1 for c in url if c.isalpha()),
            sum(1 for c in url if c.isalpha())/(len(url)+1)
        ])
        
        entropy = self._calc_entropy(url)
        
        kw_ctx = " ".join(["adult" if any(k in domain for k in ADULT_KEYWORDS) else "",
                           "illegal" if any(k in domain for k in ILLEGAL_KEYWORDS) else "",
                           path.replace('/', ' '), query.replace('&', ' '), fragment])
        clean_path = re.sub(r'[^a-zA-Z0-9]', ' ', kw_ctx + ' ' + domain)
        
        hashed = self.resources['hash_vec'].transform([clean_path]).toarray()[0]
        
        return np.hstack([stats, [entropy], feats, hashed])

    def process(self, url, text, neighbors):
        # 1. URL Embeddings
        url_emb = self._extract_url_feats(url)
        
        # 2. Content Embeddings
        ext = tldextract.extract(url)
        norm_domain = f"{ext.domain}.{ext.suffix}"
        combined_text = [f"{text} {norm_domain}"]
        
        c_emb = self.resources['count_vec'].transform(combined_text).astype(np.float32)
        t_emb = self.resources['tfidf_vec'].transform(combined_text).astype(np.float32)
        
        c_red = self.resources['svd_count'].transform(c_emb)[0]
        t_red = self.resources['svd_tfidf'].transform(t_emb)[0]
        
        # 3. Neighbor Embeddings
        valid_neighbors = []
        if neighbors:
            df_filt = self.df.filter(pl.col("domain").is_in(neighbors))
            valid_neighbors = df_filt["domain"].to_list()
            if not df_filt.is_empty():
                n_texts = df_filt["text"].fill_null('').to_list()
                n_doms = df_filt["domain"].fill_null('').to_list()
                n_comb = [f"{t} {d}" for t, d in zip(n_texts, n_doms)]
                
                if n_comb:
                    n_mat = self.resources['count_vec'].transform(n_comb).astype(np.float32)
                    
                    # --- FIX IS HERE ---
                    # np.mean on a sparse matrix returns a matrix object, which sklearn rejects.
                    # We wrap it in np.asarray to ensure it's a standard numpy array.
                    mean_vec = np.asarray(np.mean(n_mat, axis=0))
                    
                    n_red = self.resources['svd_neighbor'].transform(mean_vec)[0]
                else:
                    n_red = np.zeros(20)
            else:
                n_red = np.zeros(20)
        else:
            n_red = np.zeros(20)
            
        final_vec = np.hstack([url_emb, c_red, t_red, n_red])
        return final_vec.reshape(1, -1), valid_neighbors

def print_box(title, content):
    width = 80
    print(f"+{'-' * (width-2)}+")
    print(f"| {title.ljust(width-4)} |")
    print(f"+{'-' * (width-2)}+")
    if isinstance(content, list):
        for line in content:
            print(f"| {str(line)[:width-4].ljust(width-4)} |")
    else:
        wrapped = [content[i:i+width-4] for i in range(0, len(content), width-4)]
        for line in wrapped:
            print(f"| {line.ljust(width-4)} |")
    print(f"+{'-' * (width-2)}+")
    print()

async def main():
    processor = FeatureProcessor()
    processor.load_resources()
    spider = SimpleSpider()

    while True:
        print_box("INPUT", "Type a URL to classify (or 'q' to quit)")
        url = input("> ").strip()
        if url.lower() == 'q': break
        if not url: continue

        print("\nCrawling...")
        text, links = await spider.crawl(url)
        
        print_box("CRAWLED TEXT PREVIEW", text[:500] + "..." if len(text) > 500 else text)
        
        vec, valid_links = processor.process(url, text, links)
        
        print_box("LINK ANALYSIS", [f"Found {len(valid_links)} links known in merged_final_2.parquet:"] + valid_links[:10] + (["..."] if len(valid_links) > 10 else []))
        
        pred_idx = processor.model.predict(vec)[0]
        pred_label = processor.resources['le'].inverse_transform([pred_idx])[0]
        
        print_box("CLASSIFICATION RESULT", f"PREDICTED CATEGORY: {pred_label.upper()}")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())