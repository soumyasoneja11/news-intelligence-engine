"""Unit tests for trending score logic."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import trending_reference_datetime
from src.trending import _count_recent, _trending_score


def test_trending_score_formula() -> None:
    score = _trending_score(count_24h=2, count_48h=4, count_7d=10, total_articles=20)
    assert score == (2 * 3 + 4 * 1.5 + 10 * 0.5) / 20


def test_trending_score_zero_total() -> None:
    assert _trending_score(1, 1, 1, 0) == 0.0


def test_count_recent_windows() -> None:
    reference = datetime(2018, 6, 1, 12, 0, 0)
    articles = [
        {"date": "2018-06-01 10:00:00"},
        {"date": "2018-05-31 10:00:00"},
        {"date": "2018-05-20 10:00:00"},
    ]
    c24, c48, c7 = _count_recent(articles, reference)
    assert c24 == 1
    assert c48 == 2
    assert c7 == 2


def test_trending_reference_auto_uses_latest_article() -> None:
    metadata = [
        {"date": "2017-01-01 00:00:00"},
        {"date": "2018-03-15 08:00:00"},
    ]
    ref = trending_reference_datetime(metadata)
    assert ref == datetime(2018, 3, 15, 8, 0, 0)


if __name__ == "__main__":
    test_trending_score_formula()
    test_trending_score_zero_total()
    test_count_recent_windows()
    test_trending_reference_auto_uses_latest_article()
    print("trending unit tests passed.")
