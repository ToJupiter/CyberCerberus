import asyncio
import aiohttp
import csv
import re
import uvloop
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import logging
from datetime import datetime

# Use uvloop for better async performance
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DomainPinger:
    def __init__(self, 
                 concurrency_limit: int = 100, 
                 timeout: int = 10,
                 output_file: str = "gambling_domains.csv"):
        """
        Args:
            concurrency_limit: Max simultaneous connections
            timeout: Request timeout in seconds
            output_file: Output CSV filename
        """
        self.concurrency_limit = concurrency_limit
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.output_file = output_file
        self.classification = "illegal"
        
        # Headers to mimic browser request
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }
        
        # Initialize CSV file with headers
        self._init_csv()
    
    def _init_csv(self):
        """Initialize CSV file with headers."""
        try:
            with open(self.output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['number', 'domain', 'classification'])
            logger.info(f"Initialized CSV file: {self.output_file}")
        except Exception as e:
            logger.error(f"Failed to initialize CSV: {e}")
    
    def _append_to_csv(self, number: int, domain: str):
        """Append a single row to CSV."""
        try:
            with open(self.output_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([number, domain, self.classification])
        except Exception as e:
            logger.error(f"Failed to write to CSV: {e}")
    
    def parse_hosts_file(self, file_path: str) -> List[str]:
        """
        Parse hosts file and extract domains.
        
        Expected format:
        0.0.0.0 domain.com
        or
        127.0.0.1 domain.com
        
        Returns:
            List of unique domain strings
        """
        domains = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    # Skip comments and empty lines
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # Remove inline comments
                    line = line.split('#')[0].strip()
                    
                    # Match IP address followed by domain
                    match = re.match(r'^(?:0\.0\.0\.0|127\.0\.0\.1|::1)\s+([\w\.\-]+)$', line)
                    if match:
                        domain = match.group(1)
                        
                        # Validate domain format
                        if re.match(r'^[\w\-\.]+$', domain) and '.' in domain:
                            domains.append(domain.lower())
                        else:
                            logger.warning(f"Line {line_num}: Invalid domain format: {domain}")
            
            logger.info(f"Parsed {len(domains)} unique domains from {file_path}")
            return list(set(domains))  # Remove duplicates
            
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            return []
        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {e}")
            return []
    
    async def check_domain(self, session: aiohttp.ClientSession, domain: str) -> Optional[bool]:
        """
        Check if a domain is reachable via HTTP/HTTPS.
        
        Returns:
            True if domain responds with 2xx/3xx status
            False if domain is unreachable
            None if check failed
        """
        # Try HTTPS first, then HTTP
        schemes = ['https://', 'http://']
        
        for scheme in schemes:
            url = f"{scheme}{domain}"
            
            try:
                async with session.get(
                    url, 
                    headers=self.headers, 
                    timeout=self.timeout,
                    allow_redirects=True,
                    ssl=False  # Disable SSL verification for speed
                ) as response:
                    # Accept 2xx (success) and 3xx (redirect) status codes
                    if 200 <= response.status < 400:
                        logger.debug(f"✓ {url} - Status: {response.status}")
                        return True
                    else:
                        logger.debug(f"✗ {url} - Status: {response.status}")
                        # Don't return False yet, try next scheme
                        
            except asyncio.TimeoutError:
                logger.debug(f"✗ {url} - Timeout")
                # Try next scheme
                continue
            except aiohttp.ClientConnectorError:
                logger.debug(f"✗ {url} - Connection failed")
                # Try next scheme
                continue
            except aiohttp.ClientResponseError:
                logger.debug(f"✗ {url} - Response error")
                # Try next scheme
                continue
            except Exception as e:
                logger.debug(f"✗ {url} - Error: {type(e).__name__}")
                # Try next scheme
                continue
        
        # If all schemes failed, domain is unreachable
        return False
    
    async def process_domains_batch(self, domains: List[str], batch_id: int = 0):
        """Process a batch of domains asynchronously."""
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
            for domain in domains:
                tasks.append(self.check_domain(session, domain))
            
            # Process tasks with limited concurrency
            semaphore = asyncio.Semaphore(self.concurrency_limit)
            
            async def bounded_task(task, domain):
                async with semaphore:
                    return await task
            
            bounded_tasks = [bounded_task(tasks[i], domains[i]) for i in range(len(tasks))]
            
            results = await asyncio.gather(*bounded_tasks, return_exceptions=True)
            
            # Process results
            successful_domains = []
            for i, (domain, result) in enumerate(zip(domains, results)):
                if isinstance(result, Exception):
                    logger.warning(f"Error checking {domain}: {result}")
                elif result is True:  # Domain is reachable
                    successful_domains.append(domain)
            
            return successful_domains
    
    async def process_all_domains(self, domains: List[str], batch_size: int = 200):
        """
        Process all domains in batches and write results to CSV.
        
        Args:
            domains: List of domains to check
            batch_size: Number of domains to process in each batch
        """
        logger.info(f"Starting to process {len(domains)} domains in batches of {batch_size}")
        
        total_processed = 0
        successful_count = 0
        
        for batch_num, i in enumerate(range(0, len(domains), batch_size), 1):
            batch = domains[i:i + batch_size]
            logger.info(f"Processing batch {batch_num}: {len(batch)} domains")
            
            try:
                successful = await self.process_domains_batch(batch, batch_num)
                total_processed += len(batch)
                successful_count += len(successful)
                
                # Write successful domains to CSV
                for idx, domain in enumerate(successful, start=1):
                    csv_number = total_processed - len(batch) + idx
                    self._append_to_csv(csv_number, domain)
                
                logger.info(f"Batch {batch_num} complete: {len(successful)}/{len(batch)} domains reachable")
                logger.info(f"Progress: {total_processed}/{len(domains)} domains checked")
                
                # Brief pause between batches to avoid overwhelming
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_num}: {e}")
        
        logger.info(f"Processing complete!")
        logger.info(f"Total domains: {len(domains)}")
        logger.info(f"Reachable domains: {successful_count}")
        logger.info(f"Results saved to: {self.output_file}")


async def main():
    """Main function to run the domain checker."""
    # Configuration
    HOSTS_FILE = "gambling.txt"  # Your hosts file
    OUTPUT_CSV = "gambling_domains.csv"
    CONCURRENCY_LIMIT = 200  # Simultaneous connections
    TIMEOUT = 5  # Seconds per request
    BATCH_SIZE = 1500  # Domains per batch
    
    # Create pinger instance
    pinger = DomainPinger(
        concurrency_limit=CONCURRENCY_LIMIT,
        timeout=TIMEOUT,
        output_file=OUTPUT_CSV
    )
    
    # Parse hosts file
    logger.info(f"Parsing hosts file: {HOSTS_FILE}")
    domains = pinger.parse_hosts_file(HOSTS_FILE)
    
    if not domains:
        logger.error("No domains found to process. Exiting.")
        return
    
    # Process domains
    logger.info(f"Starting domain checking with {CONCURRENCY_LIMIT} concurrent connections")
    logger.info(f"Timeout set to {TIMEOUT} seconds per request")
    
    start_time = datetime.now()
    await pinger.process_all_domains(domains, BATCH_SIZE)
    end_time = datetime.now()
    
    duration = end_time - start_time
    logger.info(f"Total processing time: {duration}")
    
    # Show summary
    try:
        with open(OUTPUT_CSV, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            row_count = sum(1 for row in reader) - 1  # Subtract header
        logger.info(f"Total rows in CSV: {row_count}")
    except Exception as e:
        logger.error(f"Could not read CSV for summary: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Process interrupted by user. Exiting...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")