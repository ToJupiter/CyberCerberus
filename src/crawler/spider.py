import asyncio
import aiohttp
import csv
import json
import logging
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, urlparse

import lxml.html
from bs4 import BeautifulSoup
from lxml.etree import ParserError

logger = logging.getLogger(__name__)

class Spider:
    def __init__(self, concurency_limit: int = 128, timeout: int = 5):
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
    
    async def _parse_html(self, url: str, html_content: str) -> Dict[str, Any]:
        """
        Attempts to parse HTML with lxml first, falls back to BeautifulSoup.
        Extracts all hyperlinks (href) and image sources (src).
        """
        links = set()
        extracted_text = ""

        try:
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
            "text": extracted_text
        }

    async def _fetch_single(self, session: aiohttp.ClientSession, domain: str, classification: str) -> Dict[str, Any]:
        async with self.concurency_limit:
            if not domain.startswith(('http://', 'https://')):
                target_url = f'https://{domain}'
            else:
                target_url = domain

            result = {
                "domain": domain,
                "classification": classification,
                "status": "success",
                "html_content": None,
                "links_json": "[]",
                "error": None
            }

            try:
                async with session.get(target_url, headers=self.headers) as response:
                    if response.status == 200:
                        html_content = await response.text()
                        parsed_data = self._parse_html(target_url, html_content)
                        result["html_content"] = html_content
                        result["links_json"] = json.dumps(parsed_data["links"])

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
    
    

