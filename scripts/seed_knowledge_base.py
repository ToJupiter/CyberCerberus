# seed_knowledge_base.py
import asyncio
import logging
import argparse
import os

from src.crawler.spider import Spider

def setup_logging(log_level=logging.INFO):
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("crawler.log"),
            logging.StreamHandler()
        ]
    )

def main():
    parser = argparse.ArgumentParser(description='Web crawler with filtering capabilities')
    parser.add_argument('--seed_csv', type=str, required=True, help='Path to seed CSV file with domains to crawl')
    parser.add_argument('--filters_csv', type=str, default='filters.csv', help='Path to filters CSV file (optional)')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save output files')
    parser.add_argument('--concurrency', type=int, default=128, help='Maximum concurrent connections')
    parser.add_argument('--timeout', type=int, default=20, help='Request timeout in seconds')
    parser.add_argument('--batch_size', type=int, default=1000, help='Number of domains to process per batch')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()

    if not os.path.exists(args.seed_csv):
        logging.error(f"Seed CSV file not found: {args.seed_csv}")
        return

    if args.filters_csv and not os.path.exists(args.filters_csv):
        logging.warning(f"Filters CSV file not found: {args.filters_csv}. Proceeding without filters.")

    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_level)

    logger = logging.getLogger(__name__)
    logger.info(f"Starting crawler with seed file: {args.seed_csv}")
    logger.info(f"Using filters file: {args.filters_csv}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Concurrency limit: {args.concurrency}, Timeout: {args.timeout}s")

    try:
        spider = Spider(
            concurency_limit=args.concurrency, 
            timeout=args.timeout,
            output_dir=args.output_dir
        )

        asyncio.run(
            spider.process_csv(
                csv_file_path=args.seed_csv,
                filter_csv_path=args.filters_csv,
                batch_size=args.batch_size            
            )
        )

        logger.info("Crawling completed successfully!")

    except Exception as e:
        logger.error(f"Critical error during crawling: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()