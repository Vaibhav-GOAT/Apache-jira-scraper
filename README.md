# Apache Jira Scraper

This repository contains a scraper that extracts public issue data from Apache's Jira (https://issues.apache.org/jira)
and transforms it into a JSONL corpus suitable for LLM training.

## Contents
- `apache_jira_scraper.py` - main scraper script
- `requirements.txt` - Python dependencies
- `README.md` - this file
- `LICENSE` - MIT license
- `.gitignore` - files to ignore
- `output/` and `state/` directories (created at runtime)

## Quick start
1. Create a python environment:
   ```
   python -m venv venv
   source venv/bin/activate  # on Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Run the scraper:
   ```
   python apache_jira_scraper.py
   ```

Outputs will be written to the `output/` directory and scrape state is stored in `state/`.

## Config
Edit the top of `apache_jira_scraper.py` to change:
- `PROJECTS` - list of Jira project keys to scrape
- `REQUESTS_PER_MINUTE` - throttle rate
- `MAX_RESULTS` - page size

## Notes
- This scraper uses only public Jira endpoints and respects rate limits.
- For very large projects consider incremental scraping using JQL filters on `updated` timestamp.
