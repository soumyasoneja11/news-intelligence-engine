"""Smoke tests for semantic retrieval quality and result shape."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retriever import NewsRetriever

QUERIES = [
    "climate change",
    "stock market crash",
    "AI language models",
    "football world cup",
    "covid vaccine",
]
TOP_K = 3


def test_retrieval() -> None:
    retriever = NewsRetriever()

    for query in QUERIES:
        results = retriever.search(query, k=TOP_K)
        print(f"Query: {query!r}")
        print("-" * 60)

        urls: list[str] = []
        for rank, result in enumerate(results, start=1):
            score = result["score"]
            print(f"  {rank}. [{score:.4f}] {result['title']}")
            print(f"     {result['url']}")

            assert 0.0 <= score <= 1.0, f"Score out of range for {query!r}: {score}"
            urls.append(result["url"])

        assert len(urls) == len(set(urls)), f"Duplicate URLs in results for {query!r}"
        print()

    print("All retrieval checks passed.")


if __name__ == "__main__":
    test_retrieval()
