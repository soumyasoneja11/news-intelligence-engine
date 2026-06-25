"""Build script: ingest, embed, and run ML analysis pipeline."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cluster import cluster
from src.dedup import find_duplicate_pairs
from src.embedder import embed, load_articles
from src.ingest import ingest
from src.paths import ARTICLES_PATH, METADATA_PATH
from src.topics import generate_topics
from src.trending import generate_trending

PIPELINE_ERRORS = (OSError, ValueError, FileNotFoundError)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the News Intelligence Engine build pipeline.")
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip article ingestion and use existing data/articles_clean.csv.",
    )
    parser.add_argument(
        "--skip-embed",
        action="store_true",
        help="Skip embedding/index build and use existing index artifacts.",
    )
    return parser.parse_args()


def _run_step(step_name: str, func) -> object:
    print("=" * 60)
    print(step_name)
    print("=" * 60)
    start = time.perf_counter()
    try:
        result = func()
    except PIPELINE_ERRORS as exc:
        raise SystemExit(f"{step_name} failed: {exc}") from exc
    elapsed = time.perf_counter() - start
    print(f"{step_name} elapsed: {elapsed:.1f}s\n")
    return result


def _article_count(df: pd.DataFrame | None = None) -> int:
    if df is not None:
        return len(df)

    if METADATA_PATH.is_file():
        with METADATA_PATH.open(encoding="utf-8") as f:
            metadata = json.load(f)
        if isinstance(metadata, list):
            return len(metadata)

    return len(load_articles(ARTICLES_PATH))


def _print_summary(
    total_articles: int,
    clusters_found: int,
    duplicates_detected: int,
    trending: list[dict],
    total_elapsed: float,
) -> None:
    top_topics = trending[:3]

    print("=" * 60)
    print("Pipeline summary")
    print("=" * 60)
    print(f"{'Metric':<24} {'Value'}")
    print("-" * 60)
    print(f"{'Total articles':<24} {total_articles}")
    print(f"{'Clusters found':<24} {clusters_found}")
    print(f"{'Duplicates detected':<24} {duplicates_detected}")
    print(f"{'Top trending topics':<24}")
    if top_topics:
        for idx, row in enumerate(top_topics, start=1):
            print(
                f"{'':24} {idx}. {row['label']} (score={row['trending_score']})"
            )
    else:
        print(f"{'':24} n/a")
    print(f"{'Total elapsed':<24} {total_elapsed:.1f}s")


def main() -> None:
    args = _parse_args()
    total_start = time.perf_counter()
    df: pd.DataFrame | None = None

    if not args.skip_ingest:
        df = _run_step("Step 1: Ingest", ingest)
    else:
        print("Skipping ingest; using existing articles_clean.csv.\n")

    if not args.skip_embed:
        _run_step("Step 2: Embed", embed)
    else:
        print("Skipping embed; using existing index artifacts.\n")

    _, _, _, clusters_found = _run_step("Step 3: Cluster", cluster)
    _run_step("Step 4: Topics", generate_topics)
    duplicate_pairs = _run_step("Step 5: Dedup", find_duplicate_pairs)
    trending = _run_step("Step 6: Trending", generate_trending)

    total_elapsed = time.perf_counter() - total_start
    _print_summary(
        total_articles=_article_count(df),
        clusters_found=clusters_found,
        duplicates_detected=len(duplicate_pairs),
        trending=trending,
        total_elapsed=total_elapsed,
    )


if __name__ == "__main__":
    main()
