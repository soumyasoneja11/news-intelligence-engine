"""Smoke tests for the IntelligenceEngine facade."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.intelligence import IntelligenceEngine

SEARCH_KEYS = frozenset({"title", "url", "date", "domain", "score"})
SEARCH_QUERY = "climate change"
SIMILAR_ARTICLE_ID = 0


def _report(name: str, passed: bool, detail: str = "") -> bool:
    status = "PASS" if passed else "FAIL"
    suffix = f" — {detail}" if detail and not passed else ""
    print(f"{status}: {name}{suffix}")
    return passed


def test_intelligence() -> None:
    engine = IntelligenceEngine()
    results: list[bool] = []

    search_hits = engine.search(SEARCH_QUERY, k=5)
    results.append(
        _report(
            "search() returns a list of dicts",
            isinstance(search_hits, list)
            and all(isinstance(item, dict) for item in search_hits),
        )
    )
    results.append(
        _report(
            "search() results include required keys",
            bool(search_hits)
            and all(SEARCH_KEYS <= set(item.keys()) for item in search_hits),
        )
    )
    results.append(
        _report(
            "search() similarity scores are floats in [0, 1]",
            bool(search_hits)
            and all(
                isinstance(item["score"], float)
                and 0.0 <= item["score"] <= 1.0
                for item in search_hits
            ),
        )
    )

    similar_hits = engine.find_similar(SIMILAR_ARTICLE_ID, k=10)
    source_url = engine.metadata[SIMILAR_ARTICLE_ID]["url"]
    returned_urls = {item["url"] for item in similar_hits}
    results.append(
        _report(
            "find_similar() excludes the query article",
            source_url not in returned_urls,
            f"query article url appeared in results: {source_url!r}",
        )
    )

    clusters = engine.get_clusters()
    results.append(
        _report(
            "get_clusters() returns at least 5 clusters",
            len(clusters) >= 5,
            f"found {len(clusters)} clusters",
        )
    )
    results.append(
        _report(
            "get_clusters() clusters have labels and article_count > 0",
            len(clusters) >= 5
            and all(
                bool(cluster.get("label")) and int(cluster.get("article_count", 0)) > 0
                for cluster in clusters
            ),
        )
    )

    trending = engine.get_trending(n=10)
    trending_scores = [float(item["trending_score"]) for item in trending]
    results.append(
        _report(
            "get_trending() returns clusters sorted by score descending",
            trending_scores == sorted(trending_scores, reverse=True),
        )
    )

    duplicates = engine.get_duplicates()
    has_self_pairs = any(
        pair["article_a"]["id"] == pair["article_b"]["id"]
        or pair["article_a"]["url"] == pair["article_b"]["url"]
        for pair in duplicates
    )
    results.append(
        _report(
            "get_duplicates() has no self-pairs",
            not has_self_pairs,
            f"checked {len(duplicates)} pairs",
        )
    )

    passed = sum(results)
    total = len(results)
    print(f"\n{passed}/{total} checks passed.")
    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    test_intelligence()
