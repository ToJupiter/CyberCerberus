import asyncio
import aiohttp
import csv
import logging
import os
import re
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, urlparse
from collections import Counter

import uvloop
import polars as pl
import tldextract
import lxml.html
from lxml.etree import ParserError
import tqdm as tqdm

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

logger = logging.getLogger(__name__)

class Spider:
    def __init__(self, concurency_limit: int = 256, timeout: int = 15, output_dir: str = "output"):
        self.concurency_limit = asyncio.Semaphore(concurency_limit)
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.domain_to_id = {}
        self.current_max_id = 0
        self._lock = asyncio.Lock()
        
        self.blocked_domains = set()
        self.keyword_pattern = None
        self.seed_domains = set()
        self.visited_domains = set()
        self.url_regex = re.compile(
            r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[/\w\.-]*(?:\?[\w=&%]*)?(?:#[\w\-]*)?|'
            r'www\.(?:[-\w.]|(?:%[\da-fA-F]{2}))+[/\w\.-]*(?:\?[\w=&%]*)?(?:#[\w\-]*)?'
        )

    def set_filters(self, blocked_domains: List[str], blocked_keywords: List[str]):
        self.blocked_domains = set(d.lower() for d in blocked_domains if d)
        if blocked_keywords:
            pattern_str = '|'.join(re.escape(k) for k in blocked_keywords if k)
            self.keyword_pattern = re.compile(pattern_str, re.IGNORECASE)
        else:
            self.keyword_pattern = None

    def _detect_captcha(self, html_content: str) -> bool:
        if not html_content:
            return False
        content_lower = html_content.lower()
        patterns = [
            "/cdn-cgi/challenge-platform",
            "challenges.cloudflare.com/turnstile",
            "cf-chl-widget",
            "recaptcha",
            "g-recaptcha",
            "checking if the site connection is secure"
        ]
        return any(p in content_lower for p in patterns)
    
    def _normalize_domain(self, url: str) -> Optional[str]:
        try:
            if not url:
                return None
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            extracted = tldextract.extract(url)
            if extracted.suffix and extracted.domain:
                return f"{extracted.domain}.{extracted.suffix}"
            return None
        except Exception:
            return None
    
    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r'//<!\[CDATA\[.*?//\]\]>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
        text = re.sub(r'[^\x20-\x7E]', ' ', text)
        return " ".join(text.split())

    def _parse_html(self, url: str, html_content: str) -> Dict[str, Any]:
        links = set()
        extracted_text = ""
        is_captcha = self._detect_captcha(html_content)

        if not is_captcha and html_content:
            try:
                doc = lxml.html.fromstring(html_content)
                doc.make_links_absolute(url)

                for elem in doc.xpath('//script | //style | //noscript | //iframe | //meta | //link | //svg | //comment() | //header | //footer | //nav'):
                    elem.drop_tree()

                for elem in doc.iter():
                    for attrib in list(elem.attrib):
                        if attrib.lower().startswith('on'):
                            del elem.attrib[attrib]

                for _, _, link, _ in doc.iterlinks():
                    links.add(link)

                raw_text = doc.text_content()
                extracted_text = self._clean_text(raw_text)

                for match in self.url_regex.finditer(raw_text):
                    potential = match.group(0).rstrip('.,;:?\'"')
                    if potential.startswith('www.'):
                        potential = 'https://' + potential
                    links.add(potential)

            except (ParserError, ValueError, Exception):
                pass

        return {
            "links": sorted(list(links)),
            "text": extracted_text,
            "is_captcha": is_captcha
        }

    async def _fetch_single(self, session: aiohttp.ClientSession, domain: str, classification: str) -> Dict[str, Any]:
        async with self.concurency_limit:
            target_url = domain if domain.startswith(('http://', 'https://')) else f'https://{domain}'
            
            try:
                normalized_domain = self._normalize_domain(target_url) or urlparse(target_url).netloc
            except ValueError:
                normalized_domain = domain

            result = {
                "domain": normalized_domain,
                "full_url": target_url,
                "classification": classification,
                "status": "success",
                "text": "",
                "edges": [],
                "is_captcha": False
            }

            try:
                async with session.get(target_url, headers=self.headers, verify_ssl=False) as response:
                    if response.status == 200:
                        html_content = await response.text(errors='replace')
                        parsed = self._parse_html(target_url, html_content)
                        result["text"] = parsed["text"]
                        result["is_captcha"] = parsed["is_captcha"]
                        
                        valid_edges = set()
                        for link in parsed["links"]:
                            target_dom = self._normalize_domain(link)
                            if target_dom and target_dom != normalized_domain:
                                if target_dom not in self.blocked_domains:
                                    if not (self.keyword_pattern and self.keyword_pattern.search(link)):
                                        valid_edges.add(target_dom)
                        result["edges"] = list(valid_edges)
                    else:
                        result["status"] = f"error_{response.status}"
            except asyncio.TimeoutError:
                result["status"] = "error_timeout"
            except Exception as e:
                result["status"] = "error_exception"
            
            return result

    def _save_batch(self, results: List[Dict[str, Any]], iteration: int, batch_idx: int):
        nodes = []
        edges = []
        
        for r in results:
            if r["status"] == "success" and not r["is_captcha"]:
                nodes.append({
                    "domain": r["domain"],
                    "classification": r["classification"],
                    "text": r["text"],
                    "is_captcha": r["is_captcha"],
                    "status": r["status"]
                })
                for target in r["edges"]:
                    edges.append({"source": r["domain"], "target": target})
            elif r["status"] != "success":
                 nodes.append({
                    "domain": r["domain"],
                    "classification": r["classification"],
                    "text": "",
                    "is_captcha": False,
                    "status": r["status"]
                })

        if nodes:
            pl.DataFrame(nodes).write_parquet(f"{self.output_dir}/nodes_iter{iteration}_batch{batch_idx}.parquet")
        if edges:
            pl.DataFrame(edges).write_parquet(f"{self.output_dir}/edges_iter{iteration}_batch{batch_idx}.parquet")

    async def process_csv(self, csv_file_path: str, filter_csv_path: str = "filters.csv", batch_size: int = 1000, max_iterations: int = 5):
        domains_queue = {}
        
        try:
            with open(csv_file_path, "r", encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    d = row["domain"].strip()
                    domains_queue[d] = row.get("classification", "seed").strip()
        except FileNotFoundError:
            logger.error(f"Input file not found: {csv_file_path}")
            return

        if os.path.exists(filter_csv_path):
            b_dom, b_key = [], []
            try:
                with open(filter_csv_path, "r", encoding='utf-8') as f:
                    for r in csv.DictReader(f):
                        if r.get("domains"): b_dom.append(r["domains"].strip())
                        if r.get("keywords"): b_key.append(r["keywords"].strip())
                self.set_filters(b_dom, b_key)
            except Exception as e:
                logger.error(f"Filter load error: {e}")

        connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300, ssl=False)
        
        async with aiohttp.ClientSession(connector=connector, timeout=self.timeout) as session:
            current_batch_domains = list(domains_queue.keys())
            
            for iteration in range(max_iterations):
                if not current_batch_domains:
                    break
                
                logger.info(f"Iteration {iteration + 1}: Crawling {len(current_batch_domains)} domains")
                
                # Deduplicate and prioritize
                to_crawl = []
                for dom in current_batch_domains:
                    if dom not in self.visited_domains:
                        to_crawl.append(dom)
                        self.visited_domains.add(dom)
                
                if not to_crawl:
                    logger.info("No new domains to crawl in this iteration.")
                    break

                tasks = []
                for dom in to_crawl:
                    cls = domains_queue.get(dom, "discovered")
                    tasks.append(self._fetch_single(session, dom, cls))

                candidate_scores = Counter()
                total_batches = (len(tasks) + batch_size - 1) // batch_size
                
                for i in tqdm(range(0, len(tasks), batch_size), total=total_batches, desc=f"Iter {iteration+1}"):
                    batch_tasks = tasks[i:i + batch_size]
                    batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                    
                    clean_results = []
                    for res in batch_results:
                        if isinstance(res, dict):
                            clean_results.append(res)
                            if res.get("edges"):
                                candidate_scores.update(res["edges"])
                    
                    self._save_batch(clean_results, iteration + 1, (i // batch_size) + 1)

                next_batch_candidates = []
                for domain, _ in candidate_scores.most_common():
                    if domain not in self.visited_domains:
                        next_batch_candidates.append(domain)
                        if len(next_batch_candidates) >= 50000: 
                            break
                
                current_batch_domains = next_batch_candidates
                for d in current_batch_domains:
                    if d not in domains_queue:
                        domains_queue[d] = "discovered"

        logger.info("Crawling finished.")