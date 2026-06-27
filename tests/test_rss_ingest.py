"""Unit tests for BBC RSS ingestion."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingest import clean_records
from src.rss_ingest import fetch_bbc_rss_from_file

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "bbc_rss_sample.xml"


def test_parse_fixture_entries() -> None:
    records = fetch_bbc_rss_from_file(FIXTURE)
    assert len(records) == 2
    assert records[0]["title"] == "Climate summit reaches new agreement"
    assert records[0]["domain"] == "www.bbc.co.uk"
    assert records[0]["url"].startswith("https://www.bbc.co.uk/")


def test_clean_records_deduplicates_urls() -> None:
    records = fetch_bbc_rss_from_file(FIXTURE)
    duplicated = records + [records[0]]
    cleaned = clean_records(duplicated)
    assert len(cleaned) == 2
    assert all("text" in row and len(row["text"]) >= 50 for row in cleaned)


if __name__ == "__main__":
    test_parse_fixture_entries()
    test_clean_records_deduplicates_urls()
    print("rss ingest unit tests passed.")
