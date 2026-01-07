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


class AdblockDomainChecker:
    def __init__(self, 
                 concurrency_limit: int = 100, 
                 timeout: int = 10,
                 output_file: str = "adblock_domains.csv",
                 regex_filter: Optional[str] = None):
        """
        Args:
            concurrency_limit: Max simultaneous connections
            timeout: Request timeout in seconds
            output_file: Output CSV filename
            regex_filter: Optional regex pattern to filter domains (e.g., "porn|adult|nsfw")
        """
        self.concurrency_limit = concurrency_limit
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.output_file = output_file
        self.classification = "nsfw_ad"  # Changed classification
        
        # Compile regex filter if provided
        self.regex_filter = None
        if regex_filter:
            try:
                self.regex_filter = re.compile(regex_filter, re.IGNORECASE)
                logger.info(f"Using regex filter: {regex_filter}")
            except re.error as e:
                logger.error(f"Invalid regex pattern '{regex_filter}': {e}")
        
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
    
    def _append_to_csv(self, number: int, domain: str, status: str = "reachable"):
        """Append a single row to CSV with status."""
        try:
            with open(self.output_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([number, domain, self.classification, status])
        except Exception as e:
            logger.error(f"Failed to write to CSV: {e}")
    
    def parse_adblock_file(self, file_path: str) -> List[str]:
        """
        Parse Adblock Plus filter list and extract domains.
        
        Supported formats:
        - ||domain.com^
        - ||sub.domain.com^
        - domain.com##selector
        - !domain.com
        
        Returns:
            List of unique domain strings
        """
        domains = set()  # Use set to avoid duplicates
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    # Skip comments and empty lines
                    if not line or line.startswith('!'):
                        continue
                    
                    # Extract domain from various Adblock Plus formats
                    domain = None
                    
                    # Format 1: ||domain.com^ (most common)
                    match = re.match(r'^\|\|([\w\.\-]+)\^$', line)
                    if match:
                        domain = match.group(1)
                    
                    # Format 2: ||sub.domain.com^ with subdomain
                    elif re.match(r'^\|\|([\w\.\-]+\.[\w\.\-]+)\^', line):
                        # Extract domain without subdomain if possible
                        match = re.match(r'^\|\|([\w\.\-]+)\^', line.split('||')[1])
                        if match:
                            domain = match.group(1)
                    
                    # Format 3: domain.com##selector
                    elif '##' in line:
                        domain_part = line.split('##')[0]
                        if domain_part and not domain_part.startswith(('/', '#', '@')):
                            domain = domain_part
                    
                    # Format 4: Simple domain entries
                    elif re.match(r'^[\w\.\-]+$', line) and '.' in line:
                        domain = line
                    
                    # Format 5: domain.com$... (with parameters)
                    elif '$' in line and not line.startswith('!'):
                        domain_part = line.split('$')[0]
                        if domain_part and re.match(r'^[\w\.\-]+$', domain_part) and '.' in domain_part:
                            domain = domain_part
                    
                    # Clean and validate domain
                    if domain:
                        # Remove any remaining special characters
                        domain = re.sub(r'[^\w\.\-]', '', domain)
                        
                        # Basic domain validation
                        if (re.match(r'^[\w\.\-]+$', domain) and 
                            '.' in domain and 
                            len(domain) > 3 and 
                            not domain.startswith('.') and 
                            not domain.endswith('.')):
                            
                            # Apply regex filter if specified
                            if self.regex_filter:
                                if self.regex_filter.search(domain):
                                    domains.add(domain.lower())
                            else:
                                domains.add(domain.lower())
                        else:
                            logger.debug(f"Line {line_num}: Invalid domain format: {domain}")
            
            logger.info(f"Parsed {len(domains)} unique domains from {file_path}")
            return sorted(list(domains))  # Sort for consistent ordering
            
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            return []
        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {e}")
            return []
    
    def filter_domains_by_pattern(self, domains: List[str], pattern: str) -> List[str]:
        """
        Filter domains using a regex pattern.
        
        Args:
            domains: List of domains to filter
            pattern: Regex pattern to search for in domains
            
        Returns:
            Filtered list of domains
        """
        try:
            compiled_pattern = re.compile(pattern, re.IGNORECASE)
            filtered = [domain for domain in domains if compiled_pattern.search(domain)]
            logger.info(f"Filtered {len(filtered)} domains matching pattern: {pattern}")
            return filtered
        except re.error as e:
            logger.error(f"Invalid regex pattern: {e}")
            return domains
    
    async def check_domain(self, session: aiohttp.ClientSession, domain: str) -> tuple[Optional[bool], Optional[str]]:
        """
        Check if a domain is reachable via HTTP/HTTPS.
        
        Returns:
            Tuple of (is_reachable, status_code_or_error)
        """
        schemes = ['https://', 'http://']
        
        for scheme in schemes:
            url = f"{scheme}{domain}"
            
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
                        # Try next scheme
                        continue
                        
            except asyncio.TimeoutError:
                # Try next scheme
                continue
            except aiohttp.ClientConnectorError as e:
                error_msg = str(e)
                # Try next scheme
                continue
            except aiohttp.ClientResponseError as e:
                error_msg = str(e.status)
                # Try next scheme
                continue
            except Exception as e:
                error_msg = type(e).__name__
                # Try next scheme
                continue
        
        # If all schemes failed
        return False, "unreachable"
    
    async def process_domains_batch(self, domains: List[str], batch_id: int = 0) -> List[tuple[str, str]]:
        """Process a batch of domains asynchronously and return results."""
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
            domain_results = []
            for i, (domain, result) in enumerate(zip(domains, results)):
                if isinstance(result, Exception):
                    logger.debug(f"Error checking {domain}: {result}")
                    domain_results.append((domain, "error"))
                elif isinstance(result, tuple) and len(result) == 2:
                    is_reachable, status = result
                    if is_reachable:
                        domain_results.append((domain, status))
                    else:
                        domain_results.append((domain, "unreachable"))
                else:
                    domain_results.append((domain, "unknown_error"))
            
            return domain_results
    
    async def process_all_domains(self, domains: List[str], batch_size: int = 200):
        """
        Process all domains in batches and write results to CSV.
        """
        logger.info(f"Starting to process {len(domains)} domains in batches of {batch_size}")
        
        total_processed = 0
        reachable_count = 0
        
        for batch_num, i in enumerate(range(0, len(domains), batch_size), 1):
            batch = domains[i:i + batch_size]
            logger.info(f"Processing batch {batch_num}: {len(batch)} domains")
            
            try:
                results = await self.process_domains_batch(batch, batch_num)
                total_processed += len(batch)
                
                # Write results to CSV
                for idx, (domain, status) in enumerate(results, start=1):
                    csv_number = total_processed - len(batch) + idx
                    if status not in ["unreachable", "error", "unknown_error"]:
                        # Only write reachable domains
                        self._append_to_csv(csv_number, domain, status)
                        reachable_count += 1
                        logger.debug(f"✓ {domain} - Status: {status}")
                    else:
                        logger.debug(f"✗ {domain} - Status: {status}")
                
                logger.info(f"Batch {batch_num} complete: {reachable_count}/{total_processed} domains reachable so far")
                
                # Brief pause between batches
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_num}: {e}")
        
        logger.info(f"Processing complete!")
        logger.info(f"Total domains checked: {total_processed}")
        logger.info(f"Reachable domains: {reachable_count}")
        logger.info(f"Results saved to: {self.output_file}")
    
    def export_all_domains_to_csv(self, domains: List[str], output_file: Optional[str] = None):
        """
        Export all domains (without checking) to CSV.
        Useful for creating a comprehensive list.
        """
        if not output_file:
            output_file = f"all_{self.output_file}"
        
        try:
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['number', 'domain', 'classification'])
                
                for i, domain in enumerate(domains, 1):
                    writer.writerow([i, domain, self.classification])
            
            logger.info(f"Exported all {len(domains)} domains to {output_file}")
        except Exception as e:
            logger.error(f"Failed to export domains: {e}")


async def main():
    """Main function to run the domain checker."""
    # Configuration
    ADBLOCK_FILE = "oisd_nsfw.txt"  # Your Adblock Plus file
    OUTPUT_CSV = "adblock_reachable_domains.csv"
    CONCURRENCY_LIMIT = 100  # Simultaneous connections
    TIMEOUT = 5  # Seconds per request
    BATCH_SIZE = 200  # Domains per batch
    
    # Optional regex patterns for filtering
    # Examples:
    #   "porn|adult|sex|xxx" - for adult content domains
    #   "advert|ads|banner" - for ad domains
    #   "track|analytics" - for tracking domains
    #   "hello|test|demo" - for specific keywords
    #   "" or None - to disable filtering
    REGEX_FILTER = None  # Change to your desired pattern
    
    # Create checker instance
    checker = AdblockDomainChecker(
        concurrency_limit=CONCURRENCY_LIMIT,
        timeout=TIMEOUT,
        output_file=OUTPUT_CSV,
        regex_filter=REGEX_FILTER
    )
    
    # Parse Adblock Plus file
    logger.info(f"Parsing Adblock Plus file: {ADBLOCK_FILE}")
    all_domains = checker.parse_adblock_file(ADBLOCK_FILE)
    
    if not all_domains:
        logger.error("No domains found to process. Exiting.")
        return
    
    # Optional: Apply additional filtering (e.g., find domains with "hello")
    additional_filter = "hello"  # Change to your desired substring
    if additional_filter:
        filtered_domains = checker.filter_domains_by_pattern(all_domains, additional_filter)
        logger.info(f"Found {len(filtered_domains)} domains containing '{additional_filter}'")
        
        # Create a separate CSV for filtered domains (without checking)
        checker.export_all_domains_to_csv(
            filtered_domains, 
            f"filtered_{additional_filter}_domains.csv"
        )
        
        # Use filtered domains for checking
        domains_to_check = filtered_domains
    else:
        domains_to_check = all_domains
    
    # Export all domains for reference (without checking)
    checker.export_all_domains_to_csv(all_domains, "all_adblock_domains.csv")
    
    # Process domains
    logger.info(f"Starting domain checking with {CONCURRENCY_LIMIT} concurrent connections")
    logger.info(f"Timeout set to {TIMEOUT} seconds per request")
    
    start_time = datetime.now()
    await checker.process_all_domains(domains_to_check, BATCH_SIZE)
    end_time = datetime.now()
    
    duration = end_time - start_time
    logger.info(f"Total processing time: {duration}")
    
    # Show summary
    try:
        with open(OUTPUT_CSV, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            row_count = sum(1 for row in reader) - 1  # Subtract header
        logger.info(f"Total reachable domains in CSV: {row_count}")
    except Exception as e:
        logger.error(f"Could not read CSV for summary: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Process interrupted by user. Exiting...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")