"""Scheduled RSS fetch, embed, trending, and clustering jobs."""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import schedule

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.cluster import cluster
from src.dedup import find_duplicate_pairs
from src.embedder import embed
from src.ingest import ingest_rss
from src.paths import ARTICLES_PATH
from src.rss_fetcher import fetch_all_feeds
from src.topics import generate_topics
from src.trending import generate_trending

LOG_PATH = _ROOT / "logs" / "scheduler.log"
FETCH_INTERVAL_MINUTES = int(os.environ.get("FETCH_INTERVAL_MINUTES", "30"))
CLUSTER_INTERVAL_HOURS = int(os.environ.get("CLUSTER_INTERVAL_HOURS", "6"))


def _article_count() -> int:
    if not ARTICLES_PATH.is_file():
        return 0
    return len(pd.read_csv(ARTICLES_PATH))


def _log(job: str, message: str, before: int, after: int) -> None:
    delta = after - before
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"{timestamp} [{job}] articles={after} delta={delta:+d} {message}"
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(line + "\n")
    print(line)


def run_fetch_cycle() -> None:
    """Fetch RSS feeds, save new articles, embed incrementally, and refresh trending."""
    before = _article_count()
    try:
        articles = fetch_all_feeds()
        ingest_rss(
            output_path=ARTICLES_PATH,
            refresh=ARTICLES_PATH.is_file(),
            articles=articles,
        )
        embed()
        generate_trending()
        after = _article_count()
        _log("fetch", "fetch cycle complete", before, after)
    except Exception as exc:
        after = _article_count()
        _log("fetch", f"fetch cycle failed: {exc}", before, after)
        traceback.print_exc()


def run_cluster_cycle() -> None:
    """Run the full clustering pipeline."""
    before = _article_count()
    try:
        cluster()
        generate_topics()
        find_duplicate_pairs()
        generate_trending()
        after = _article_count()
        _log("cluster", "cluster cycle complete", before, after)
    except Exception as exc:
        after = _article_count()
        _log("cluster", f"cluster cycle failed: {exc}", before, after)
        traceback.print_exc()


def _configure_jobs() -> None:
    schedule.clear()
    schedule.every(FETCH_INTERVAL_MINUTES).minutes.do(run_fetch_cycle)
    schedule.every(CLUSTER_INTERVAL_HOURS).hours.do(run_cluster_cycle)


def run_daemon() -> None:
    """Start the scheduler loop."""
    _configure_jobs()
    print(
        f"Scheduler started: fetch every {FETCH_INTERVAL_MINUTES}m, "
        f"cluster every {CLUSTER_INTERVAL_HOURS}h (log: {LOG_PATH})"
    )
    _log("daemon", "scheduler started", _article_count(), _article_count())
    while True:
        schedule.run_pending()
        time.sleep(1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scheduled RSS and clustering jobs.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--once",
        action="store_true",
        help="Run one fetch cycle immediately and exit.",
    )
    group.add_argument(
        "--daemon",
        action="store_true",
        help="Start the scheduler loop.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.once:
        run_fetch_cycle()
        return
    run_daemon()


if __name__ == "__main__":
    main()
