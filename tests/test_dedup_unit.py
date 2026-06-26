"""Unit tests for dedup pair filtering."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def filter_pairs(pairs: list[dict], threshold: float) -> list[dict]:
    return [pair for pair in pairs if float(pair["similarity"]) >= threshold]


def test_dedup_threshold_filter() -> None:
    pairs = [
        {"similarity": 0.95, "article_a": {"id": "0"}, "article_b": {"id": "1"}},
        {"similarity": 0.88, "article_a": {"id": "2"}, "article_b": {"id": "3"}},
    ]
    assert len(filter_pairs(pairs, 0.92)) == 1
    assert len(filter_pairs(pairs, 0.85)) == 2


if __name__ == "__main__":
    test_dedup_threshold_filter()
    print("dedup unit tests passed.")
