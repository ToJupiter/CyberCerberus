Materials:
1. A spider.py file, getting the website content and hyperlinks and a csv with domains acting as seed domains. Then, from the seed domains, we start to iterate through all of the domains (the domains given in the hyperlinks) to create a network of graph linking domains, websites together. We chain them altogether, in order to create a network of websites. Find me some ways to detect if the website is protected by captcha. For textual content of the website, we can save it using polars with parquet. For network linking, find me a way to store it effectively because we will be using it later on for graph convolutional network testing.

Goal:
1. Limit the network to around 1M nodes (1M domains). Chain them together using torch_geometric or networkx. 
2. Save the text to polars parquet.
3. Try to solve me the problem of this: this problem is for bad vs good web content classification using web content. But the original domain may not contain the bad content but rather a subdomain of it may contain bad things. Find me solutions to solve this because this means it may not reflect the right picture! But if we craft subdomains into graph, it is too huge. We cannot manage such big thing!

# spider.py
``` py
import asyncio
import aiohttp
import csv
import json
import logging
import os
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, urlparse

from tqdm import tqdm
import polars as pl
import tldextract
import lxml.html
from bs4 import BeautifulSoup
from lxml.etree import ParserError

logger = logging.getLogger(__name__)

class Spider:
    def __init__(self, concurency_limit: int = 128, timeout: int = 5, output_dir: str = "output"):
        """
        Args:
            concurrency_limit: Max number of simultaneous async connections.
            timeout: Seconds to wait for a response before giving up.
        """
        self.concurency_limit = asyncio.Semaphore(concurency_limit)
        self.timeout = aiohttp.ClientTimeout(timeout)
        self.headers = {
            "User-Agent": "Mozilla/5.0 (compatible; WebsiteCategorizer/1.0; +https://example.com/bot)"
        }
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.domain_to_id = {}
        self.current_max_id = 0
        self._lock = asyncio.Lock()

    async def _get_domain_id(self, domain: str) -> int:
        """Thread-safe ID assignment for graph nodes."""
        async with self._lock:
            if domain not in self.domain_to_id:
                self.domain_to_id[domain] = self.current_max_id
                self.current_max_id += 1
            return self.domain_to_id[domain]
    
    def _detect_captcha(self, html_content: str) -> bool:
        if not html_content:
            return False
        
        content_lower = html_content.lower()
        # Cloudflare specific patterns
        cf_patterns = [
            "/cdn-cgi/challenge-platform",
            "challenges.cloudflare.com/turnstile",
            "checking if the site connection is secure",
            "cf-chl-widget",
            "just a moment..."
        ]
        
        # Google reCAPTCHA specific patterns
        google_patterns = [
            "recaptcha",
            "g-recaptcha",
            "data-sitekey",
            "www.google.com/recaptcha/api"
        ]

        if any(p in content_lower for p in cf_patterns):
            return True
        if any(p in content_lower for p in google_patterns):
            return True
    
    def _normalize_domain(self, url: str) -> Optional[str]:
        try:
            extracted = tldextract.extract(url)
            if extracted.suffix and extracted.domain:
                return f"{extracted.domain}.{extracted.suffix}"
            return None
        except Exception:
            return None
    
    async def _parse_html(self, url: str, html_content: str) -> Dict[str, Any]:
        """
        Attempts to parse HTML with lxml first, falls back to BeautifulSoup.
        Extracts all hyperlinks (href) and image sources (src).
        """
        links = set()
        extracted_text = ""
        is_captcha = False

        try:
            if self._detect_captcha(html_content):
                is_captcha = True
                logger.debug(f"Captcha detected at {url}")

            doc = lxml.html.fromstring(html_content)
            doc.make_links_absolute(url)
            for element, attribute, link, pos in doc.iterlinks():
                links.add(link)
            extracted_text = doc.text_content()

        except (ParserError, ValueError) as e:
            logger.debug(f"lxml parsing failed for {url}, falling back to BeautifulSoup: {e}")

            try:
                soup = BeautifulSoup(html_content, 'lxml')
                for tag in soup.find_all(['a', 'link', 'area', 'img', 'script', 'iframe', 'source']):
                    href = tag.get('href')
                    src = tag.get('src')

                    if href:
                        links.add(urljoin(url, href))
                    if src:
                        links.add(urljoin(url, src))

                extracted_text = soup.get_text(separator=" ", strip=True)
            
            except Exception as bs_e:
                logger.error(f"BeautifulSoup also failed for {url}: {bs_e}")
    
        return {
            "links": sorted(list(links)),
            "text": extracted_text,
            "is_captcha": is_captcha
        }

    async def _fetch_single(self, session: aiohttp.ClientSession, domain: str, classification: str) -> Dict[str, Any]:
        async with self.concurency_limit:
            if not domain.startswith(('http://', 'https://')):
                target_url = f'https://{domain}'
            else:
                target_url = domain

            normalized_domain = self._normalize_domain(target_url)
            if not normalized_domain:
                normalized_domain = urlparse(target_url).netloc

            result = {
                "domain": normalized_domain,
                "full_url": target_url,
                "classification": classification,
                "status": "success",
                "text": "",
                "error": None,
                "is_captcha": False,
                "edges": []
            }

            try:
                async with session.get(target_url, headers=self.headers) as response:
                    if response.status == 200:
                        html_content = await response.text()
                        parsed_data = self._parse_html(target_url, html_content)
                        result["text"] = parsed_data["text"]
                        result["is_captcha"] = parsed_data["is_captcha"]

                        unique_targets = set()
                        for link in parsed_data["links"]:
                            norm_target = self._normalize_domain(link)
                            if norm_target and norm_target != normalized_domain:
                                unique_targets.add(norm_target)
                        
                        result["edges"] = list(unique_targets)

                    else:
                        result["status"] = f"error_{response.status}"
                        result["error"] = f"HTTP status: {response.status}"
                
            except asyncio.TimeoutError:
                result["status"] = "error_timeout"
                result["error"] = "Connection timed out"
            except Exception as e:
                result["status"] = "error_exception"
                result["error"] = str(e)
                logger.debug(f"Failed to crawl {domain}: {e}")

            return result
        
    def _save_batch(self, results: List[Dict[str, Any]], file_suffix: int):
        if not results:
            return
        
        node_records = []
        for r in results:
            if r["status"] == "success":
                subdomain_part = ""
                full_domain = urlparse(r["full_url"]).netloc
                if full_domain != r["domain"]:
                    subdomain_part = f"[SUBDOMAIN: {full_domain}] "
                
                node_records.append({
                    "domain": r["domain"],
                    "classification": r["classification"],
                    "text": subdomain_part + r["text"],
                    "is_captcha": r["is_captcha"],
                    "crawl_status": r["status"]
                })

        edge_records = []                
        for r in results:
            if r["status"] == "success" and r["edges"]:
                for target in r["edges"]:
                    edge_records.append({
                        "source": r["domain"],
                        "target": target
                    })
        
        if node_records:
            df_nodes = pl.DataFrame(node_records)
            df_nodes.write_parquet(f"{self.output_dir}/nodes_batch_{file_suffix}.parquet")
            logger.info(f"Saved batch {file_suffix}: {len(df_nodes)} nodes.")

        if edge_records:
            df_edges = pl.DataFrame(edge_records)
            df_edges.write_parquet(f"{self.output_dir}/edges_batch_{file_suffix}.parquet")
            logger.info(f"Saved batch {file_suffix}: {len(df_edges)} edges.")

    async def process_csv(self, csv_file_path: str, batch_size: int = 1000):
        """
        Main entry point. Reads CSV and orchestrates async crawling.
        """
        domains_to_crawl = []

        try:
            with open(csv_file_path, "r", encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    domains_to_crawl.append({
                        "domain": row["domain"].strip(),
                        "classification": row["classification"].strip()
                    })
        except FileNotFoundError:
            logger.error(f"CSV file not found: {csv_file_path}")
            return []
        logger.info(f"Loaded {len(domains_to_crawl)} domains from CSV. Starting crawl...")

        connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300)

        async with aiohttp.ClientSession(connector=connector, timeout=self.timeout) as session:
            total_batches = (len(domains_to_crawl) + batch_size - 1) // batch_size
            for i in tqdm(range(0, len(domains_to_crawl), batch_size), total=total_batches, desc="Crawling batches"):
            # for i in range(0, len(domains_to_crawl), batch_size):
                batch = domains_to_crawl[i:i + batch_size]
                tasks = []

                for item in batch:
                    task = self._fetch_single(session, item['domain'], item['classification'])
                    tasks.append(task)
                
                logger.info(f"Processing batch {i//batch_size + 1} ({len(batch)} domains)...")
                results = await asyncio.gather(*tasks, return_exceptions=True)

                clean_results = []
                for res in results:
                    if isinstance(res, Exception):
                        logger.error(f"Critical Task Failure: {res}")
                    else:
                        clean_results.append(res)
                
                self._save_batch(clean_results, i//batch_size + 1)

        logger.info("Crawling finished.")
```

--- Request 2:
1. For the spider.py code, change the logic of the function to achieve this goal:

2/Edge weight tracking: Domains linked by many seeds are more important. When we iterate through a hyperlink domain, check if it links with multiple domains in our existing base. Prioritize those domains which links with more seed domains.

3/ Domain filtering: Exclude common CDNs, social media, etc.... We will be reading a csv with 2 columns: domains and keywords. Ignore the hyperlinks if they have the domains or keywords in the list. Find some way for this operation to be fast!

-- Request 3:
1, First of all, check if the propagation logic is correct? Did my code really create a graph and really expand. It should expand like a BFS model, of course prioritizing the new nodes which have many links with the nodes in the existing seed nodes.
2, The old code does not parse the html content of the page correctly. I will provide you the results. The parsed result contains hyperlinks, and also javascript code. I want to still parse fast, but only parse english content. The hyperlinks should be added to the edges instead of wasting them because we cannot parse them.
3, Suggest me using uvloop for asyncio event loop in this case, in order to speed the things up. 

``` py
import asyncio
import aiohttp
import csv
import json
import logging
import os
import re
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, urlparse

from tqdm import tqdm
import polars as pl
import tldextract
import lxml.html
from bs4 import BeautifulSoup
from lxml.etree import ParserError

logger = logging.getLogger(__name__)

class Spider:
    def __init__(self, concurency_limit: int = 128, timeout: int = 5, output_dir: str = "output"):
        """
        Args:
            concurrency_limit: Max number of simultaneous async connections.
            timeout: Seconds to wait for a response before giving up.
        """
        self.concurency_limit = asyncio.Semaphore(concurency_limit)
        self.timeout = aiohttp.ClientTimeout(timeout)
        self.headers = {
            "User-Agent": "Mozilla/5.0 (compatible; WebsiteCategorizer/1.0; +https://example.com/bot    )"
        }
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.domain_to_id = {}
        self.current_max_id = 0
        self._lock = asyncio.Lock()
        
        # New attributes
        self.blocked_domains = set()
        self.keyword_pattern = None
        self.seed_domains = set()

    async def _get_domain_id(self, domain: str) -> int:
        """Thread-safe ID assignment for graph nodes."""
        async with self._lock:
            if domain not in self.domain_to_id:
                self.domain_to_id[domain] = self.current_max_id
                self.current_max_id += 1
            return self.domain_to_id[domain]
    
    def _detect_captcha(self, html_content: str) -> bool:
        if not html_content:
            return False
        
        content_lower = html_content.lower()
        # Cloudflare specific patterns
        cf_patterns = [
            "/cdn-cgi/challenge-platform",
            "challenges.cloudflare.com/turnstile",
            "checking if the site connection is secure",
            "cf-chl-widget",
            "just a moment..."
        ]
        
        # Google reCAPTCHA specific patterns
        google_patterns = [
            "recaptcha",
            "g-recaptcha",
            "data-sitekey",
            "www.google.com/recaptcha/api"
        ]

        if any(p in content_lower for p in cf_patterns):
            return True
        if any(p in content_lower for p in google_patterns):
            return True
        return False
    
    def _normalize_domain(self, url: str) -> Optional[str]:
        try:
            extracted = tldextract.extract(url)
            if extracted.suffix and extracted.domain:
                return f"{extracted.domain}.{extracted.suffix}"
            return None
        except Exception:
            return None
    
    async def _parse_html(self, url: str, html_content: str) -> Dict[str, Any]:
        """
        Attempts to parse HTML with lxml first, falls back to BeautifulSoup.
        Extracts all hyperlinks (href) and image sources (src).
        """
        links = set()
        extracted_text = ""
        is_captcha = False

        try:
            if self._detect_captcha(html_content):
                is_captcha = True
                logger.debug(f"Captcha detected at {url}")

            doc = lxml.html.fromstring(html_content)
            doc.make_links_absolute(url)
            for element, attribute, link, pos in doc.iterlinks():
                links.add(link)
            extracted_text = doc.text_content()

        except (ParserError, ValueError) as e:
            logger.debug(f"lxml parsing failed for {url}, falling back to BeautifulSoup: {e}")

            try:
                soup = BeautifulSoup(html_content, 'html.parser')
                for tag in soup.find_all(['a', 'link', 'area', 'img', 'script', 'iframe', 'source']):
                    href = tag.get('href')
                    src = tag.get('src')

                    if href:
                        links.add(urljoin(url, href))
                    if src:
                        links.add(urljoin(url, src))

                extracted_text = soup.get_text(separator=" ", strip=True)
            
            except Exception as bs_e:
                logger.error(f"BeautifulSoup also failed for {url}: {bs_e}")
    
        return {
            "links": sorted(list(links)),
            "text": extracted_text,
            "is_captcha": is_captcha
        }

    async def _fetch_single(self, session: aiohttp.ClientSession, domain: str, classification: str) -> Dict[str, Any]:
        async with self.concurency_limit:
            if not domain.startswith(('http://', 'https://')):
                target_url = f'https://{domain}'
            else:
                target_url = domain

            normalized_domain = self._normalize_domain(target_url)
            if not normalized_domain:
                normalized_domain = urlparse(target_url).netloc

            result = {
                "domain": normalized_domain,
                "full_url": target_url,
                "classification": classification,
                "status": "success",
                "text": "",
                "error": None,
                "is_captcha": False,
                "edges": []
            }

            try:
                async with session.get(target_url, headers=self.headers) as response:
                    if response.status == 200:
                        html_content = await response.text()
                        parsed_data = await self._parse_html(target_url, html_content)
                        result["text"] = parsed_data["text"]
                        result["is_captcha"] = parsed_data["is_captcha"]

                        unique_targets = set()
                        for link in parsed_data["links"]:
                            norm_target = self._normalize_domain(link)
                            
                            if not norm_target or norm_target == normalized_domain:
                                continue
                                
                            # Fast filtering
                            if norm_target in self.blocked_domains:
                                continue
                            
                            if self.keyword_pattern and self.keyword_pattern.search(link):
                                continue

                            unique_targets.add(norm_target)
                        
                        result["edges"] = list(unique_targets)

                    else:
                        result["status"] = f"error_{response.status}"
                        result["error"] = f"HTTP status: {response.status}"
                
            except asyncio.TimeoutError:
                result["status"] = "error_timeout"
                result["error"] = "Connection timed out"
            except Exception as e:
                result["status"] = "error_exception"
                result["error"] = str(e)
                logger.debug(f"Failed to crawl {domain}: {e}")

            return result
        
    def _save_batch(self, results: List[Dict[str, Any]], file_suffix: int):
        if not results:
            return
        
        node_records = []
        for r in results:
            if r["status"] == "success":
                subdomain_part = ""
                full_domain = urlparse(r["full_url"]).netloc
                if full_domain != r["domain"]:
                    subdomain_part = f"[SUBDOMAIN: {full_domain}] "
                
                node_records.append({
                    "domain": r["domain"],
                    "classification": r["classification"],
                    "text": subdomain_part + r["text"],
                    "is_captcha": r["is_captcha"],
                    "crawl_status": r["status"]
                })

        edge_records = []                
        for r in results:
            if r["status"] == "success" and r["edges"]:
                for target in r["edges"]:
                    edge_records.append({
                        "source": r["domain"],
                        "target": target
                    })
        
        if node_records:
            df_nodes = pl.DataFrame(node_records)
            df_nodes.write_parquet(f"{self.output_dir}/nodes_batch_{file_suffix}.parquet")
            logger.info(f"Saved batch {file_suffix}: {len(df_nodes)} nodes.")

        if edge_records:
            df_edges = pl.DataFrame(edge_records)
            df_edges.write_parquet(f"{self.output_dir}/edges_batch_{file_suffix}.parquet")
            logger.info(f"Saved batch {file_suffix}: {len(df_edges)} edges.")

    def set_filters(self, blocked_domains: List[str], blocked_keywords: List[str]):
        self.blocked_domains = set(d.lower() for d in blocked_domains if d)
        if blocked_keywords:
            pattern_str = '|'.join(re.escape(k) for k in blocked_keywords if k)
            self.keyword_pattern = re.compile(pattern_str, re.IGNORECASE)
        else:
            self.keyword_pattern = None

    def set_seeds(self, seeds: List[str]):
        self.seed_domains = set(s.lower() for s in seeds if s)

    async def process_csv(self, csv_file_path: str, filter_csv_path: str = "filters.csv", batch_size: int = 1000):
        """
        Main entry point. Reads CSV and orchestrates async crawling.
        """
        domains_to_crawl = []

        try:
            with open(csv_file_path, "r", encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    domains_to_crawl.append({
                        "domain": row["domain"].strip(),
                        "classification": row["classification"].strip()
                    })
        except FileNotFoundError:
            logger.error(f"CSV file not found: {csv_file_path}")
            return []

        blocked_domains = []
        blocked_keywords = []
        if filter_csv_path and os.path.exists(filter_csv_path):
            try:
                with open(filter_csv_path, "r", encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if "domains" in row and row["domains"].strip():
                            blocked_domains.append(row["domains"].strip())
                        if "keywords" in row and row["keywords"].strip():
                            blocked_keywords.append(row["keywords"].strip())
            except Exception as e:
                logger.error(f"Error loading filters: {e}")

        self.set_filters(blocked_domains, blocked_keywords)
        self.set_seeds([d["domain"] for d in domains_to_crawl])

        logger.info(f"Loaded {len(domains_to_crawl)} domains. Filters: {len(blocked_domains)} domains, {len(blocked_keywords)} keywords.")

        connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300)

        async with aiohttp.ClientSession(connector=connector, timeout=self.timeout) as session:
            total_batches = (len(domains_to_crawl) + batch_size - 1) // batch_size
            for i in tqdm(range(0, len(domains_to_crawl), batch_size), total=total_batches, desc="Crawling batches"):
                batch = domains_to_crawl[i:i + batch_size]
                tasks = []

                for item in batch:
                    task = self._fetch_single(session, item['domain'], item['classification'])
                    tasks.append(task)
                
                results = await asyncio.gather(*tasks, return_exceptions=True)

                clean_results = []
                for res in results:
                    if isinstance(res, Exception):
                        logger.error(f"Critical Task Failure: {res}")
                    else:
                        clean_results.append(res)
                
                self._save_batch(clean_results, i//batch_size + 1)

        logger.info("Crawling finished.")
```
