"""Ingest and clean news articles from HuggingFace cc_news or fallbacks."""

from __future__ import annotations

import re
import sys
from html import unescape
from itertools import islice
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from datasets import load_dataset
from faker import Faker

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.paths import ARTICLES_PATH
from src.utils import format_date

OUTPUT_PATH = ARTICLES_PATH
MIN_TEXT_LEN = 50
STREAM_LIMIT = 2000
SYNTHETIC_FALLBACK_COUNT = 500

# Public news CSV used when HuggingFace is unreachable (NewsAPI-compatible field mapping).
NEWSAPI_SAMPLE_CSV_URL = (
    "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/"
    "ag_news_csv/test.csv"
)

HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")

_COLUMN_ALIASES = {
    "title": ("title", "Title", "headline", "Headline"),
    "description": (
        "description",
        "Description",
        "short_description",
        "content",
        "summary",
    ),
    "url": ("url", "URL", "link", "Link", "urlToArticle"),
    "date": ("date", "Date", "publishedAt", "published_at", "pubDate"),
    "domain": ("domain", "Domain", "source", "source_name", "Source"),
}


def strip_html(text: str) -> str:
    """Remove HTML tags and decode entities."""
    if not text:
        return ""
    cleaned = unescape(text)
    cleaned = HTML_TAG_RE.sub(" ", cleaned)
    return WHITESPACE_RE.sub(" ", cleaned).strip()


def combine_text(title: str, description: str) -> str:
    """Merge title and description into one searchable text field."""
    parts = [strip_html(title), strip_html(description)]
    combined = " ".join(part for part in parts if part)
    return WHITESPACE_RE.sub(" ", combined).strip()


def _as_str(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _pick_field(row: pd.Series, aliases: tuple[str, ...]) -> str:
    for alias in aliases:
        if alias in row.index and pd.notna(row[alias]):
            return str(row[alias]).strip()
    return ""


def _domain_from_url(url: str) -> str:
    return urlparse(url).netloc.strip() if url else ""


def _load_from_huggingface(limit: int) -> list[dict]:
    dataset = load_dataset("vblagoje/cc_news", split="train", streaming=True)
    records = []
    for row in islice(dataset, limit):
        records.append(
            {
                "title": row.get("title") or "",
                "description": row.get("description") or "",
                "url": row.get("url") or "",
                "date": row.get("date") or "",
                "domain": row.get("domain") or "",
            }
        )
    return records


def _csv_row_to_record(row: pd.Series, idx: int) -> dict:
    title = _pick_field(row, _COLUMN_ALIASES["title"]) or str(row.get("title", "")).strip()
    description = (
        _pick_field(row, _COLUMN_ALIASES["description"])
        or str(row.get("description", "")).strip()
    )
    article_url = _pick_field(row, _COLUMN_ALIASES["url"])
    date = _pick_field(row, _COLUMN_ALIASES["date"])
    domain = _pick_field(row, _COLUMN_ALIASES["domain"])

    if not article_url:
        slug = re.sub(r"\W+", "-", title.lower())[:60].strip("-") or f"article-{idx}"
        domain = domain or f"news.example-{(idx % 12) + 1}.com"
        article_url = f"https://{domain}/{slug}-{idx}"

    if not date:
        date = f"2020-{(idx % 12) + 1:02d}-{(idx % 28) + 1:02d} 12:00:00"

    if not domain:
        domain = _domain_from_url(article_url)

    return {
        "title": title,
        "description": description,
        "url": article_url,
        "date": date,
        "domain": domain,
    }


def _load_from_newsapi_csv(url: str, limit: int) -> list[dict]:
    preview = pd.read_csv(url, nrows=1)
    has_named_columns = any(
        col in preview.columns
        for col in (
            "title",
            "Title",
            "headline",
            "Headline",
            "description",
            "Description",
            "link",
            "url",
        )
    )

    if has_named_columns:
        df = pd.read_csv(url, nrows=limit)
    else:
        df = pd.read_csv(
            url,
            nrows=limit,
            header=None,
            names=["label", "title", "description"],
        )

    return [_csv_row_to_record(row, idx) for idx, row in df.iterrows()]


def _load_synthetic_records(count: int = SYNTHETIC_FALLBACK_COUNT) -> list[dict]:
    fake = Faker()
    Faker.seed(42)
    fake.seed_instance(42)

    records: list[dict] = []
    for _ in range(count):
        domain = fake.domain_name()
        title = fake.sentence(nb_words=8).rstrip(".")
        description = fake.paragraph(nb_sentences=4)
        article_url = f"https://{domain}/articles/{fake.uuid4()}"
        date = fake.date_time_between(start_date="-2y", end_date="now").strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        records.append(
            {
                "title": title,
                "description": description,
                "url": article_url,
                "date": date,
                "domain": domain,
            }
        )
    return records


def load_raw_records(limit: int = STREAM_LIMIT) -> list[dict]:
    """Load articles from HuggingFace, with CSV and synthetic fallbacks."""
    try:
        return _load_from_huggingface(limit)
    except Exception:
        pass

    try:
        return _load_from_newsapi_csv(NEWSAPI_SAMPLE_CSV_URL, limit)
    except Exception:
        pass

    return _load_synthetic_records(min(limit, SYNTHETIC_FALLBACK_COUNT))


def clean_records(records: list[dict]) -> list[dict]:
    """Build cleaned text, filter short articles, and deduplicate by URL."""
    seen_urls: set[str] = set()
    cleaned: list[dict] = []

    for record in records:
        url = _as_str(record.get("url"))
        if not url or url in seen_urls:
            continue

        text = combine_text(_as_str(record.get("title")), _as_str(record.get("description")))
        if len(text) < MIN_TEXT_LEN:
            continue

        seen_urls.add(url)
        raw_date = _as_str(record.get("date"))
        cleaned.append(
            {
                "title": strip_html(_as_str(record.get("title"))),
                "description": strip_html(_as_str(record.get("description"))),
                "url": url,
                "date": format_date(raw_date),
                "domain": _as_str(record.get("domain")),
                "text": text,
            }
        )

    return cleaned


def ingest(limit: int = STREAM_LIMIT, output_path: Path = OUTPUT_PATH) -> pd.DataFrame:
    """Load, clean, and save articles to CSV."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(f"Failed to create data directory {output_path.parent}") from exc

    print(f"Loading articles (up to {limit} records)...")
    raw = load_raw_records(limit=limit)
    print(f"Loaded {len(raw)} raw records.")

    print("Cleaning text, filtering short articles, deduplicating by URL...")
    cleaned = clean_records(raw)
    print(f"Kept {len(cleaned)} records after cleaning.")

    if not cleaned:
        raise ValueError("No articles remained after cleaning; refusing to write empty CSV.")

    df = pd.DataFrame(cleaned)
    try:
        df.to_csv(output_path, index=False)
    except OSError as exc:
        raise OSError(f"Failed to write articles to {output_path}") from exc
    print(f"Saved to {output_path}")

    return df


if __name__ == "__main__":
    ingest()
