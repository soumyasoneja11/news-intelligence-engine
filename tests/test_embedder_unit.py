"""Unit tests for embedder CSV/index alignment."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.embedder import csv_matches_index


def test_csv_matches_index_when_aligned() -> None:
    df = pd.DataFrame(
        {
            "text": ["a", "b"],
            "title": ["t1", "t2"],
            "url": ["https://a.com", "https://b.com"],
            "date": ["2020-01-01", "2020-01-02"],
            "domain": ["a.com", "b.com"],
        }
    )
    metadata = [
        {"id": "0", "url": "https://a.com"},
        {"id": "1", "url": "https://b.com"},
    ]
    assert csv_matches_index(df, metadata) is True


def test_csv_matches_index_detects_drift() -> None:
    df = pd.DataFrame(
        {
            "text": ["a"],
            "title": ["t1"],
            "url": ["https://a.com"],
            "date": ["2020-01-01"],
            "domain": ["a.com"],
        }
    )
    metadata = [
        {"id": "0", "url": "https://a.com"},
        {"id": "1", "url": "https://b.com"},
    ]
    assert csv_matches_index(df, metadata) is False


if __name__ == "__main__":
    test_csv_matches_index_when_aligned()
    test_csv_matches_index_detects_drift()
    print("embedder unit tests passed.")
