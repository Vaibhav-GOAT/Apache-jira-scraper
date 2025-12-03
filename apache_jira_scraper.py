#!/usr/bin/env python3
"""
apache_jira_scraper.py

Usage:
    python apache_jira_scraper.py

Requirements:
    pip install -r requirements.txt

Notes:
 - Targets Apache Jira instance: https://issues.apache.org/jira
 - Uses REST API v2 search endpoint and issue endpoint for comments.
 - Produces raw JSONL and a transformed JSONL suitable for LLM fine-tuning.
"""

import os
import time
import json
import logging
from typing import Dict, Any, List
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from tqdm import tqdm

# ---------- CONFIG ----------
JIRA_BASE = "https://issues.apache.org/jira"
SEARCH_ENDPOINT = "/rest/api/2/search"
ISSUE_ENDPOINT = "/rest/api/2/issue/{issue_id_or_key}"
PROJECTS = ["HADOOP", "SPARK", "KAFKA"]  # change if you want other 3 projects
FIELDS = "summary,description,comment,creator,reporter,assignee,labels,priority,status,created,updated,issuetype"
MAX_RESULTS = 50  # page size (server may cap this)
REQUESTS_PER_MINUTE = 60  # throttle; set lower if you get 429s
OUTPUT_DIR = "output"
STATE_DIR = "state"
LOG_LEVEL = logging.INFO

# ---------- Logging ----------
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(STATE_DIR, exist_ok=True)
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------- HTTP session with retries ----------
session = requests.Session()
retries = Retry(
    total=5,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS"],
)
adapter = HTTPAdapter(max_retries=retries)
session.mount("https://", adapter)
session.mount("http://", adapter)

# small helper to respect RPM
SECONDS_BETWEEN = 60.0 / REQUESTS_PER_MINUTE if REQUESTS_PER_MINUTE > 0 else 0


def throttle_sleep(last_time):
    if SECONDS_BETWEEN <= 0:
        return time.time()
    now = time.time()
    to_wait = SECONDS_BETWEEN - (now - last_time)
    if to_wait > 0:
        time.sleep(to_wait)
    return time.time()


def read_state(project_key: str) -> Dict[str, Any]:
    path = os.path.join(STATE_DIR, f"{project_key}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"startAt": 0, "seen_issue_keys": []}


def save_state(project_key: str, state: Dict[str, Any]):
    path = os.path.join(STATE_DIR, f"{project_key}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def write_jsonl(path: str, records: List[Dict]):
    mode = "a" if os.path.exists(path) else "w"
    with open(path, mode, encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def transform_issue_for_llm(issue_raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Maps Jira issue structure into a training-friendly JSON object.
    Contains metadata, cleaned description, comments concatenated, and derived tasks.
    """
    fields = issue_raw.get("fields", {})
    comments_data = fields.get("comment", {}).get("comments", [])
    comments_texts = []
    for c in comments_data:
        author = c.get("author", {}).get("displayName", "")
        body = c.get("body", "")
        created = c.get("created", "")
        comments_texts.append(f"{author} ({created}): {body}")

    description = fields.get("description") or ""
    title = fields.get("summary") or ""
    metadata = {
        "issue_key": issue_raw.get("key"),
        "project": fields.get("project", {}).get("key"),
        "issuetype": fields.get("issuetype", {}).get("name"),
        "status": fields.get("status", {}).get("name"),
        "priority": (fields.get("priority") or {}).get("name"),
        "reporter": (fields.get("reporter") or {}).get("displayName"),
        "assignee": (fields.get("assignee") or {}).get("displayName"),
        "labels": fields.get("labels", []),
        "created": fields.get("created"),
        "updated": fields.get("updated"),
    }

    summarization_prompt = {
        "task": "summarization",
        "input": f"Title: {title}\n\nDescription:\n{description}\n\nComments:\n" + ("\n".join(comments_texts)),
        "output": "",
    }

    qna_prompt = {
        "task": "qa",
        "context": f"{title}\n\n{description}\n\n" + ("\n".join(comments_texts)),
        "qa_pairs": []
    }

    transformed = {
        "metadata": metadata,
        "title": title,
        "description": description,
        "comments": comments_texts,
        "derived": {
            "summarization": summarization_prompt,
            "qna": qna_prompt,
        },
        "raw_issue_id": issue_raw.get("id"),
    }
    return transformed


def fetch_issue(issue_key: str, last_time: float):
    """Fetch a single issue detail by key (including comments)"""
    last_time = throttle_sleep(last_time)
    url = JIRA_BASE + ISSUE_ENDPOINT.format(issue_id_or_key=issue_key)
    params = {"fields": FIELDS}
    r = session.get(url, params=params, timeout=30)
    if r.status_code == 429:
        ra = r.headers.get("Retry-After")
        wait = int(ra) if ra and ra.isdigit() else 60
        logging.warning("Got 429 for issue %s; sleeping %s seconds", issue_key, wait)
        time.sleep(wait)
        return fetch_issue(issue_key, time.time())
    r.raise_for_status()
    return r.json(), time.time()


def scrape_project(project_key: str):
    state = read_state(project_key)
    startAt = state.get("startAt", 0)
    seen = set(state.get("seen_issue_keys", []))
    raw_out = os.path.join(OUTPUT_DIR, f"{project_key}_raw.jsonl")
    transformed_out = os.path.join(OUTPUT_DIR, f"{project_key}_transformed.jsonl")

    more_to_fetch = True
    last_time = time.time()
    pbar = None

    while more_to_fetch:
        last_time = throttle_sleep(last_time)
        jql = f"project = {project_key} ORDER BY created ASC"
        params = {
            "jql": jql,
            "startAt": startAt,
            "maxResults": MAX_RESULTS,
            "fields": FIELDS
        }
        url = JIRA_BASE + SEARCH_ENDPOINT
        logging.info("Querying %s startAt=%s", project_key, startAt)
        r = session.get(url, params=params, timeout=30)
        if r.status_code == 429:
            ra = r.headers.get("Retry-After")
            wait = int(ra) if ra and ra.isdigit() else 60
            logging.warning("HTTP 429 received. Sleeping %s seconds", wait)
            time.sleep(wait)
            continue
        if r.status_code >= 500:
            logging.warning("Server error %s. Backing off.", r.status_code)
            time.sleep(10)
            continue
        r.raise_for_status()
        resp = r.json()
        issues = resp.get("issues", [])
        total = resp.get("total", 0)
        if pbar is None:
            pbar = tqdm(total=total, desc=f"{project_key} issues", unit="issue")
            pbar.update(startAt)

        if not issues:
            logging.info("No issues returned; finishing for project %s", project_key)
            break

        raw_batch = []
        transformed_batch = []
        for it in issues:
            key = it.get("key")
            if key in seen:
                pbar.update(1)
                continue
            try:
                issue_full, last_time = fetch_issue(key, last_time)
            except Exception as e:
                logging.error("Failed to fetch issue %s: %s", key, e)
                continue

            raw_batch.append(issue_full)
            transformed_batch.append(transform_issue_for_llm(issue_full))
            seen.add(key)
            pbar.update(1)

        if raw_batch:
            write_jsonl(raw_out, raw_batch)
            write_jsonl(transformed_out, transformed_batch)

        startAt = resp.get("startAt", startAt) + len(issues)
        state["startAt"] = startAt
        state["seen_issue_keys"] = list(seen)
        save_state(project_key, state)

        if startAt >= resp.get("total", startAt):
            more_to_fetch = False

    if pbar:
        pbar.close()
    logging.info("Finished scraping project %s. Raw -> %s. Transformed -> %s", project_key, raw_out, transformed_out)


def main():
    logging.info("Starting Apache Jira scraper for projects: %s", PROJECTS)
    for proj in PROJECTS:
        try:
            scrape_project(proj)
        except Exception as e:
            logging.exception("Error scraping project %s: %s", proj, e)


if __name__ == "__main__":
    main()
