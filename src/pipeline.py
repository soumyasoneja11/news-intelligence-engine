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
    parser.add_argument(
        "--rss-refresh",
        action="store_true",
        help="Append new BBC RSS articles before embedding (implies ingest unless --skip-ingest).",
    )
    parser.add_argument(
        "--skip-cluster",
        action="store_true",
        help="Skip clustering and topic labeling (fast RSS refresh path).",
    )
    parser.add_argument(
        "--full-rebuild",
        action="store_true",
        help="Force full embed rebuild instead of incremental update.",
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


def _run_ingest(args: argparse.Namespace) -> pd.DataFrame | None:
    if args.skip_ingest and not args.rss_refresh:
        print("Skipping ingest; using existing articles_clean.csv.\n")
        return None

    refresh = args.rss_refresh or ARTICLES_PATH.is_file()
    source = "rss" if args.rss_refresh else "hf"
    step_name = "Step 1: Ingest (RSS)" if source == "rss" else "Step 1: Ingest (HuggingFace)"
    return _run_step(step_name, lambda: ingest(source=source, refresh=refresh))


def _run_embed(args: argparse.Namespace) -> None:
    if args.skip_embed:
        print("Skipping embed; using existing index artifacts.\n")
        return

    if args.full_rebuild:
        from src.paths import EMBEDDINGS_PATH, FAISS_PATH, INDEX_DIR, METADATA_PATH

        for path in (FAISS_PATH, METADATA_PATH, EMBEDDINGS_PATH):
            if path.is_file():
                path.unlink()
                print(f"Removed {path} for full rebuild.")

    _run_step("Step 2: Embed", embed)


def main() -> None:
    args = _parse_args()
    total_start = time.perf_counter()
    df: pd.DataFrame | None = None
    clusters_found = 0
    trending: list[dict] = []

    df = _run_ingest(args)
    _run_embed(args)

    if not args.skip_cluster:
        _, _, _, clusters_found = _run_step("Step 3: Cluster", cluster)
        _run_step("Step 4: Topics", generate_topics)
    else:
        print("Skipping cluster and topics (--skip-cluster).\n")

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
