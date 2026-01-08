import asyncio
import aiohttp
import csv
import re
import uvloop
import logging
from typing import List, Optional, Set, Pattern
from datetime import datetime
from urllib.parse import urlparse
import ssl

# Use uvloop for better async performance
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PhishingURLChecker:
    def __init__(self, 
                 concurrency_limit: int = 100, 
                 timeout: int = 10,
                 output_file: str = "phishing_urls.csv",
                 max_working_urls: int = 8000):
        """
        Args:
            concurrency_limit: Max simultaneous connections
            timeout: Request timeout in seconds
            output_file: Output CSV filename
            max_working_urls: Stop when reaching this many working URLs
        """
        self.concurrency_limit = concurrency_limit
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.output_file = output_file
        self.classification = "malicious"
        self.max_working_urls = max_working_urls
        self.working_urls_count = 0
        self.stop_flag = False
        
        # Headers to mimic browser request
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }
        
        # SSL context for HTTPS
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE
        
        # Initialize CSV file with headers
        self._init_csv()
    
    def _init_csv(self):
        """Initialize CSV file with headers."""
        try:
            with open(self.output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['number', 'domain', 'classification', 'status'])
            logger.info(f"Initialized CSV file: {self.output_file}")
        except Exception as e:
            logger.error(f"Failed to initialize CSV: {e}")
    
    def _append_to_csv(self, number: int, url: str, status: str = "reachable"):
        """Append a single row to CSV with status."""
        try:
            with open(self.output_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([number, url, self.classification, status])
        except Exception as e:
            logger.error(f"Failed to write to CSV: {e}")
    
    def parse_phishing_list(self, file_path: str) -> List[str]:
        """
        Parse phishing URLs from Adblock Plus format file.
        
        Supported formats:
        - ||domain.com/path^
        - ||domain.com^
        - domain.com##selector (ignore)
        - !domain.com (comment)
        
        Returns:
            List of URLs with scheme (http:// or https://)
        """
        urls = set()  # Use set to avoid duplicates
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    # Skip comments and empty lines
                    if not line or line.startswith('!'):
                        continue
                    
                    # Skip CSS selector rules
                    if '##' in line:
                        continue
                    
                    url = None
                    
                    # Format: ||domain.com/path^ or ||domain.com^
                    if line.startswith('||') and line.endswith('^'):
                        # Remove || and ^
                        url_part = line[2:-1]
                        
                        # Extract domain and path
                        if '/' in url_part:
                            # Has path
                            domain_part = url_part.split('/')[0]
                            path_part = '/' + '/'.join(url_part.split('/')[1:])
                        else:
                            # Just domain
                            domain_part = url_part
                            path_part = ''
                        
                        # Validate domain
                        if (re.match(r'^[\w\.\-]+$', domain_part) and 
                            '.' in domain_part and 
                            len(domain_part) > 3):
                            
                            # Try both schemes
                            urls.add(f"https://{domain_part}{path_part}")
                            urls.add(f"http://{domain_part}{path_part}")
                    
                    # Format: Direct URL (not common in adblock but handle anyway)
                    elif line.startswith(('http://', 'https://')):
                        urls.add(line)
            
            logger.info(f"Parsed {len(urls)} unique URLs from {file_path}")
            return sorted(list(urls))  # Sort for consistent ordering
            
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            return []
        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {e}")
            return []
    
    async def check_url(self, session: aiohttp.ClientSession, url: str) -> tuple[Optional[bool], Optional[str]]:
        """
        Check if a URL is reachable via HTTP/HTTPS.
        
        Returns:
            Tuple of (is_reachable, status_code_or_error)
        """
        # Check stop flag first
        if self.stop_flag:
            return False, "stopped"
        
        try:
            async with session.get(
                url, 
                headers=self.headers, 
                timeout=self.timeout,
                allow_redirects=True,
                ssl=self.ssl_context
            ) as response:
                # Accept 2xx (success) and 3xx (redirect) status codes
                if 200 <= response.status < 400:
                    return True, str(response.status)
                else:
                    return False, str(response.status)
                    
        except asyncio.TimeoutError:
            return False, "timeout"
        except aiohttp.ClientConnectorError as e:
            return False, "connection_error"
        except aiohttp.ClientResponseError as e:
            return False, str(e.status)
        except aiohttp.ClientError as e:
            return False, "client_error"
        except Exception as e:
            return False, "unknown_error"
    
    async def process_urls_batch(self, urls: List[str], batch_id: int = 0) -> List[tuple[str, str]]:
        """Process a batch of URLs asynchronously and return results."""
        connector = aiohttp.TCPConnector(
            limit=self.concurrency_limit,
            ttl_dns_cache=300,
            force_close=True
        )
        
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=self.timeout,
            headers=self.headers
        ) as session:
            
            tasks = []
            for url in urls:
                tasks.append(self.check_url(session, url))
            
            # Process tasks with limited concurrency
            semaphore = asyncio.Semaphore(self.concurrency_limit)
            
            async def bounded_task(task, url):
                async with semaphore:
                    return await task
            
            bounded_tasks = [bounded_task(tasks[i], urls[i]) for i in range(len(tasks))]
            
            results = await asyncio.gather(*bounded_tasks, return_exceptions=True)
            
            # Process results
            url_results = []
            for i, (url, result) in enumerate(zip(urls, results)):
                if self.stop_flag:
                    # Skip remaining checks if we've reached the limit
                    url_results.append((url, "skipped_due_to_stop"))
                    continue
                
                if isinstance(result, Exception):
                    logger.debug(f"Error checking {url}: {result}")
                    url_results.append((url, "error"))
                elif isinstance(result, tuple) and len(result) == 2:
                    is_reachable, status = result
                    if is_reachable:
                        url_results.append((url, status))
                        
                        # Increment counter and check if we've reached the limit
                        self.working_urls_count += 1
                        logger.info(f"✓ Working URL #{self.working_urls_count}: {url}")
                        
                        if self.working_urls_count >= self.max_working_urls:
                            logger.info(f"Reached maximum of {self.max_working_urls} working URLs. Stopping...")
                            self.stop_flag = True
                    else:
                        url_results.append((url, status))
                else:
                    url_results.append((url, "unknown_error"))
            
            return url_results
    
    async def process_all_urls(self, urls: List[str], batch_size: int = 200):
        """
        Process all URLs in batches and write results to CSV.
        Stop immediately when reaching max_working_urls.
        """
        logger.info(f"Starting to process {len(urls)} URLs in batches of {batch_size}")
        logger.info(f"Will stop when reaching {self.max_working_urls} working URLs")
        
        total_processed = 0
        
        for batch_num, i in enumerate(range(0, len(urls), batch_size), 1):
            if self.stop_flag:
                logger.info("Stop flag detected. Ending processing.")
                break
                
            batch = urls[i:i + batch_size]
            logger.info(f"Processing batch {batch_num}: {len(batch)} URLs")
            
            try:
                results = await self.process_urls_batch(batch, batch_num)
                total_processed += len(batch)
                
                # Write results to CSV (only working URLs)
                for idx, (url, status) in enumerate(results, start=1):
                    if status not in ["unreachable", "error", "unknown_error", "skipped_due_to_stop", 
                                    "timeout", "connection_error", "client_error"] and status.isdigit():
                        # Only write reachable URLs (status codes are digits)
                        csv_number = self.working_urls_count  # Use working count instead of total
                        self._append_to_csv(csv_number, url, status)
                        logger.debug(f"✓ {url} - Status: {status}")
                    else:
                        logger.debug(f"✗ {url} - Status: {status}")
                
                logger.info(f"Batch {batch_num} complete: {self.working_urls_count}/{self.max_working_urls} working URLs so far")
                
                # Brief pause between batches
                if not self.stop_flag:
                    await asyncio.sleep(0.5)
                else:
                    break
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_num}: {e}")
                if self.stop_flag:
                    break
        
        logger.info(f"Processing complete!")
        logger.info(f"Total URLs checked: {total_processed}")
        logger.info(f"Working URLs found: {self.working_urls_count}")
        logger.info(f"Results saved to: {self.output_file}")
    
    def export_all_urls_to_csv(self, urls: List[str], output_file: Optional[str] = None):
        """
        Export all URLs (without checking) to CSV.
        Useful for creating a comprehensive list.
        """
        if not output_file:
            output_file = f"all_{self.output_file}"
        
        try:
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['number', 'url', 'classification'])
                
                for i, url in enumerate(urls, 1):
                    writer.writerow([i, url, self.classification])
            
            logger.info(f"Exported all {len(urls)} URLs to {output_file}")
        except Exception as e:
            logger.error(f"Failed to export URLs: {e}")


async def main():
    """Main function to run the phishing URL checker."""
    # Configuration
    PHISHING_FILE = "data/phishing.txt"  # Your phishing URLs in Adblock Plus format
    OUTPUT_CSV = "phishing_working_urls.csv"
    CONCURRENCY_LIMIT = 100  # Simultaneous connections
    TIMEOUT = 5  # Seconds per request
    BATCH_SIZE = 200  # URLs per batch
    MAX_WORKING_URLS = 8000  # Stop when reaching this many working URLs
    
    # Create checker instance
    checker = PhishingURLChecker(
        concurrency_limit=CONCURRENCY_LIMIT,
        timeout=TIMEOUT,
        output_file=OUTPUT_CSV,
        max_working_urls=MAX_WORKING_URLS
    )
    
    # Parse phishing URLs file
    logger.info(f"Parsing phishing URLs file: {PHISHING_FILE}")
    all_urls = checker.parse_phishing_list(PHISHING_FILE)
    
    if not all_urls:
        logger.error("No URLs found to process. Exiting.")
        return
    
    # Export all URLs for reference (without checking)
    checker.export_all_urls_to_csv(all_urls, "all_phishing_urls.csv")
    
    # Process URLs
    logger.info(f"Starting URL checking with {CONCURRENCY_LIMIT} concurrent connections")
    logger.info(f"Timeout set to {TIMEOUT} seconds per request")
    logger.info(f"Will stop at {MAX_WORKING_URLS} working URLs")
    
    start_time = datetime.now()
    await checker.process_all_urls(all_urls, BATCH_SIZE)
    end_time = datetime.now()
    
    duration = end_time - start_time
    logger.info(f"Total processing time: {duration}")
    
    # Show summary
    try:
        with open(OUTPUT_CSV, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            row_count = sum(1 for row in reader) - 1  # Subtract header
        logger.info(f"Total working URLs in CSV: {row_count}")
    except Exception as e:
        logger.error(f"Could not read CSV for summary: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Process interrupted by user. Exiting...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")