import asyncio
import aiohttp
import csv
import logging
import os
import re
from typing import List, Dict, Any, Set
from urllib.parse import urljoin, urlparse

import uvloop
import tldextract
import lxml.html
import polars as pl
import tqdm

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

logger = logging.getLogger(__name__)

class Spider:
    def __init__(self, concurency_limit: int = 128, timeout: int = 20, output_dir: str = "output"):
        self.concurency_limit = asyncio.Semaphore(concurency_limit)
        self.timeout = aiohttp.ClientTimeout(timeout)
        self.headers = {
            "User-Agent": "Mozilla/5.0 (compatible; WebsiteCategorizer/1.0; +https://example.com/bot  )"
        }
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.domain_to_id = {}
        self.current_max_id = 0
        self._lock = asyncio.Lock()
        
        self.blocked_domains = set()
        self.keyword_pattern = None
        self.seed_domains = set()
        
        self.re_script = re.compile(r'<(script|style|noscript|iframe)[^>]*>.*?</\1>|<!--.*?-->', re.IGNORECASE | re.DOTALL)
        self.re_cdata = re.compile(r'//<!\[CDATA\[.*?//\]\]>', re.IGNORECASE | re.DOTALL)
        self.re_url_text = re.compile(r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9]{1,6}\b(?:[-a-zA-Z0-9@:%_\+.~#?&//=]*)')
        logger.debug(f"Spider initialized with concurrency_limit={concurency_limit}, timeout={timeout}, output_dir={output_dir}")

    def set_filters(self, blocked_domains: List[str], blocked_keywords: List[str]):
        self.blocked_domains = set(d.lower() for d in blocked_domains if d)
        logger.debug(f"Blocked domains set: {self.blocked_domains}")
        if blocked_keywords:
            pattern_str = '|'.join(re.escape(k) for k in blocked_keywords if k)
            self.keyword_pattern = re.compile(pattern_str, re.IGNORECASE)
            logger.debug(f"Keyword pattern compiled: {pattern_str}")
        else:
            self.keyword_pattern = None
            logger.debug("No keyword filters set")

    def set_seeds(self, seeds: List[str]):
        self.seed_domains = set(self._normalize_domain(s) or s.lower() for s in seeds if s)
        logger.debug(f"Seed domains set: {self.seed_domains}")

    async def _get_domain_id(self, domain: str) -> int:
        async with self._lock:
            if domain not in self.domain_to_id:
                self.domain_to_id[domain] = self.current_max_id
                self.current_max_id += 1
                logger.debug(f"Assigned domain_id {self.domain_to_id[domain]} to domain: {domain}")
            return self.domain_to_id[domain]
    
    def _detect_captcha(self, html_content: str) -> bool:
        if not html_content:
            logger.debug("Empty HTML content, no captcha detected")
            return False
        content_lower = html_content.lower()
        patterns = [
            "/cdn-cgi/challenge-platform",
            "challenges.cloudflare.com/turnstile",
            "checking if the site connection is secure",
            "cf-chl-widget",
            "just a moment...",
            "recaptcha",
            "g-recaptcha",
            "data-sitekey"
        ]
        found = any(p in content_lower for p in patterns)
        logger.debug(f"Captcha detection for content length {len(html_content)}: {'detected' if found else 'not detected'}")
        return found
    
    def _normalize_domain(self, url: str) -> str:
        try:
            extracted = tldextract.extract(url)
            if extracted.suffix and extracted.domain:
                normalized = f"{extracted.domain}.{extracted.suffix}"
                logger.debug(f"Normalized URL {url} to domain: {normalized}")
                return normalized
        except Exception as e:
            logger.debug(f"Failed to normalize URL {url}: {e}")
        return None
    
    def _clean_and_parse(self, url: str, html_content: str) -> Dict[str, Any]:
        links = set()
        clean_html = self.re_cdata.sub('', html_content)
        clean_html = self.re_script.sub('', clean_html)
        logger.debug(f"Cleaned HTML for {url}: removed scripts/styles, new length {len(clean_html)}")
        
        try:
            doc = lxml.html.fromstring(clean_html)
            doc.make_links_absolute(url)
            
            hrefs = doc.xpath('//a/@href | //link/@href | //area/@href')
            srcs = doc.xpath('//img/@src | //source/@src | //iframe/@src')
            
            for link in hrefs + srcs:
                links.add(link)
            
            text_content = " ".join(doc.xpath('//body//text()')).strip()
            text_content = re.sub(r'\s+', ' ', text_content)
            
        except Exception as e:
            logger.debug(f"Failed to parse HTML for {url}: {e}")
            text_content = ""
        
        text_links = self.re_url_text.findall(text_content)
        links.update(text_links)
        
        logger.debug(f"Extracted {len(links)} links and {len(text_content)} characters of text from {url}")
        return {
            "links": sorted(list(links)),
            "text": text_content,
            "is_captcha": self._detect_captcha(html_content)
        }

    async def _fetch_single(self, session: aiohttp.ClientSession, domain: str, classification: str) -> Dict[str, Any]:
        async with self.concurency_limit:
            target_url = domain if domain.startswith(('http://', 'https://')) else f'https://{domain}'
            
            normalized_domain = self._normalize_domain(target_url) or urlparse(target_url).netloc
            logger.debug(f"Fetching {target_url} (normalized: {normalized_domain}) with classification: {classification}")

            result = {
                "domain": normalized_domain,
                "full_url": target_url,
                "classification": classification,
                "status": "success",
                "text": "",
                "error": None,
                "is_captcha": False,
                "edges": [],
                "seed_score": 0
            }

            try:
                async with session.get(target_url, headers=self.headers) as response:
                    logger.debug(f"Received response {response.status} for {target_url}")
                    if response.status == 200:
                        html_content = await response.text()
                        logger.debug(f"Retrieved {len(html_content)} bytes of HTML content from {target_url}")
                        parsed_data = self._clean_and_parse(target_url, html_content)
                        
                        result["text"] = parsed_data["text"]
                        result["is_captcha"] = parsed_data["is_captcha"]
                        
                        valid_edges = set()
                        seed_hits = 0

                        for link in parsed_data["links"]:
                            norm_target = self._normalize_domain(link)
                            
                            if not norm_target or norm_target == normalized_domain:
                                logger.debug(f"Skipping link {link}: invalid or same domain")
                                continue
                            
                            if norm_target in self.blocked_domains:
                                logger.debug(f"Skipping blocked domain: {norm_target}")
                                continue
                            
                            if self.keyword_pattern and self.keyword_pattern.search(link):
                                logger.debug(f"Skipping link due to keyword match: {link}")
                                continue

                            if norm_target in self.seed_domains:
                                seed_hits += 1
                                logger.debug(f"Seed domain hit: {norm_target}")

                            valid_edges.add(norm_target)
                        
                        result["edges"] = list(valid_edges)
                        result["seed_score"] = seed_hits
                        logger.debug(f"Extracted {len(valid_edges)} valid edges and {seed_hits} seed hits from {target_url}")
                    else:
                        result["status"] = f"error_{response.status}"
                        result["error"] = f"HTTP status: {response.status}"
                        logger.debug(f"Non-200 status {response.status} for {target_url}")
                
            except asyncio.TimeoutError:
                result["status"] = "error_timeout"
                logger.debug(f"Timeout fetching {target_url}")
            except Exception as e:
                result["status"] = "error_exception"
                result["error"] = str(e)
                logger.debug(f"Exception fetching {target_url}: {e}")

            return result
        
    def _save_batch(self, results: List[Dict[str, Any]], file_suffix: int):
        if not results:
            logger.debug(f"No results to save for batch {file_suffix}")
            return
        
        node_records = []
        edge_records = []
        
        for r in results:
            if r["status"] == "success":
                node_records.append({
                    "domain": r["domain"],
                    "classification": r["classification"],
                    "text": r["text"],
                    "is_captcha": r["is_captcha"],
                    "crawl_status": r["status"],
                    "seed_score": r["seed_score"]
                })
                
                for target in r["edges"]:
                    edge_records.append({
                        "source": r["domain"],
                        "target": target
                    })
        
        if node_records:
            pl.DataFrame(node_records).write_parquet(f"{self.output_dir}/nodes_batch_{file_suffix}.parquet")
            logger.debug(f"Saved {len(node_records)} node records to nodes_batch_{file_suffix}.parquet")
        
        if edge_records:
            pl.DataFrame(edge_records).write_parquet(f"{self.output_dir}/edges_batch_{file_suffix}.parquet")
            logger.debug(f"Saved {len(edge_records)} edge records to edges_batch_{file_suffix}.parquet")

    async def process_csv(self, csv_file_path: str, filter_csv_path: str = "filters.csv", batch_size: int = 1000):
        domains_to_crawl = []
        try:
            with open(csv_file_path, "r", encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    domains_to_crawl.append({
                        "domain": row["domain"].strip(),
                        "classification": row["classification"].strip()
                    })
            logger.debug(f"Loaded {len(domains_to_crawl)} domains from {csv_file_path}")
        except FileNotFoundError:
            logger.error(f"CSV file not found: {csv_file_path}")
            return

        blocked_domains = []
        blocked_keywords = []
        
        if os.path.exists(filter_csv_path):
            with open(filter_csv_path, "r", encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("domains"): blocked_domains.append(row["domains"].strip())
                    if row.get("keywords"): blocked_keywords.append(row["keywords"].strip())
            logger.debug(f"Loaded {len(blocked_domains)} blocked domains and {len(blocked_keywords)} blocked keywords from {filter_csv_path}")

        self.set_filters(blocked_domains, blocked_keywords)
        self.set_seeds([d["domain"] for d in domains_to_crawl])

        connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300)

        async with aiohttp.ClientSession(connector=connector, timeout=self.timeout) as session:
            total_batches = (len(domains_to_crawl) + batch_size - 1) // batch_size
            for i in tqdm.tqdm(range(0, len(domains_to_crawl), batch_size), total=total_batches, desc="Crawling batches"):
                batch = domains_to_crawl[i:i + batch_size]
                tasks = [self._fetch_single(session, item['domain'], item['classification']) for item in batch]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                clean_results = [res for res in results if not isinstance(res, Exception)]
                logger.debug(f"Batch {i//batch_size + 1}: {len(results)} total, {len(clean_results)} successful, {len(results) - len(clean_results)} exceptions")
                self._save_batch(clean_results, i // batch_size + 1)

        logger.info("Crawling finished.")