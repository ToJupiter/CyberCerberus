# First guideline

To run the graph crawler, place `scripts/seed_knowledge_base.py` along with `src/crawler/spider.py`. The seed_csv for them is `data/classified_domains.csv` and `data/filters.csv` acts as the domain filter.

# Environment setup
Ensure uv is installed
``` bash
uv sync
source .venv/bin/activate
```

# Commands

``` bash
PYTHONPATH=. python scripts/seed_knowledge_base.py --seed_csv data/classified_domains.csv
PYTHONPATH=. python scripts/nsfw_domain_ping.py
PYTHONPATH=. python scripts/illegal_domain_ping.py
PYTHONPATH=. python scripts/phishing_domain_ping.py
PYTHONPATH=. python scripts/domain_classification.py
```