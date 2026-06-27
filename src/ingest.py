"""Ingest and clean news articles from HuggingFace cc_news or RSS feeds."""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from html import unescape
from itertools import islice
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import MAX_ARTICLES
from src.paths import ARTICLES_PATH
from src.utils import format_date

OUTPUT_PATH = ARTICLES_PATH
MIN_TEXT_LEN = 50
STREAM_LIMIT = 2000
SYNTHETIC_FALLBACK_COUNT = 500
RSS_BODY_FETCH_LIMIT = 200

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


def _published_to_str(published: object) -> str:
    if published is None:
        return ""
    if isinstance(published, datetime):
        return format_date(published.strftime("%Y-%m-%d %H:%M:%S"))
    return format_date(str(published))


def _load_from_huggingface(limit: int) -> list[dict]:
    from datasets import load_dataset

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
    from faker import Faker

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


def _normalize_rss_articles(articles: list[dict]) -> list[dict]:
    from src.rss_fetcher import get_article_body

    records: list[dict] = []
    for index, article in enumerate(articles):
        title = _as_str(article.get("title"))
        link = _as_str(article.get("link"))
        summary = _as_str(article.get("summary"))
        if not title or not link:
            continue

        if index < RSS_BODY_FETCH_LIMIT:
            body = get_article_body(link, summary=summary)
        else:
            body = summary

        records.append(
            {
                "title": title,
                "description": body,
                "url": link,
                "date": _published_to_str(article.get("published")),
                "domain": _domain_from_url(link),
            }
        )
    return records


def clean_records(records: list[dict]) -> list[dict]:
    """Build cleaned text, filter short articles, and deduplicate by URL."""
    seen_urls: set[str] = set()
    cleaned: list[dict] = []

    for record in records:
        url = _as_str(record.get("url"))
        if not url or url in seen_urls:
            continue

        title = strip_html(_as_str(record.get("title")))
        body = strip_html(_as_str(record.get("description")))
        text = f"{title} {body}".strip()
        if len(text) < MIN_TEXT_LEN:
            continue

        seen_urls.add(url)
        raw_date = _as_str(record.get("date"))
        domain = _as_str(record.get("domain")) or _domain_from_url(url)
        cleaned.append(
            {
                "title": title,
                "description": body,
                "url": url,
                "date": format_date(raw_date),
                "domain": domain,
                "text": text,
            }
        )

    return cleaned


def _cap_dataframe(df: pd.DataFrame, max_articles: int) -> pd.DataFrame:
    if len(df) <= max_articles:
        return df
    return df.iloc[:max_articles].copy()


def ingest_hf(limit: int = STREAM_LIMIT, output_path: Path = OUTPUT_PATH) -> pd.DataFrame:
    """Load HuggingFace articles and replace the CSV."""
    print(f"Loading articles from HuggingFace (up to {limit} records)...")
    raw = load_raw_records(limit=limit)
    print(f"Loaded {len(raw)} raw records.")

    print("Cleaning text, filtering short articles, deduplicating by URL...")
    cleaned = clean_records(raw)
    print(f"Kept {len(cleaned)} records after cleaning.")

    if not cleaned:
        raise ValueError("No articles remained after cleaning; refusing to write empty CSV.")

    df = pd.DataFrame(cleaned)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} articles to {output_path}")
    return df


def ingest_rss(
    output_path: Path = OUTPUT_PATH,
    refresh: bool = False,
    articles: list[dict] | None = None,
) -> pd.DataFrame:
    """Fetch live RSS feeds, normalize, and append to or create the CSV."""
    from src.rss_fetcher import fetch_all_feeds

    if articles is None:
        print("Fetching articles from RSS feeds...")
        articles = fetch_all_feeds()
        print(f"Fetched {len(articles)} unique feed entries.")
    else:
        print(f"Saving {len(articles)} fetched RSS entries.")

    print(f"Normalizing RSS records (full body fetch for top {RSS_BODY_FETCH_LIMIT})...")
    cleaned = clean_records(_normalize_rss_articles(articles))
    if not cleaned:
        raise ValueError("No articles remained after RSS cleaning; refusing to write empty CSV.")

    existing_df: pd.DataFrame | None = None
    known_urls: set[str] = set()
    if refresh or output_path.is_file():
        existing_df = pd.read_csv(output_path)
        known_urls = set(existing_df["url"].map(_as_str))

    new_rows = [row for row in cleaned if row["url"] not in known_urls]
    duplicate_count = len(cleaned) - len(new_rows)
    new_count = len(new_rows)

    if new_count == 0 and existing_df is not None:
        print(
            f"Added {new_count} new articles, {duplicate_count} duplicates skipped "
            f"({len(existing_df)} total)."
        )
        return existing_df

    if existing_df is not None and not existing_df.empty:
        combined = pd.concat([existing_df, pd.DataFrame(new_rows)], ignore_index=True)
    else:
        combined = pd.DataFrame(new_rows if new_rows else cleaned)

    combined = _cap_dataframe(combined, MAX_ARTICLES)
    combined.to_csv(output_path, index=False)
    from src.rss_fetcher import write_feed_stats

    write_feed_stats()
    print(
        f"Added {new_count} new articles, {duplicate_count} duplicates skipped "
        f"({len(combined)} total)."
    )
    return combined


def ingest(
    limit: int = STREAM_LIMIT,
    output_path: Path = OUTPUT_PATH,
    source: str = "hf",
    refresh: bool = False,
) -> pd.DataFrame:
    """Load, clean, and save articles to CSV."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(f"Failed to create data directory {output_path.parent}") from exc

    if source == "rss":
        return ingest_rss(output_path=output_path, refresh=refresh)
    if source != "hf":
        raise ValueError(f"Unknown ingest source: {source}")

    return ingest_hf(limit=limit, output_path=output_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest news articles into articles_clean.csv.")
    parser.add_argument(
        "--source",
        choices=("hf", "rss"),
        default="hf",
        help="Article source: HuggingFace dataset (hf) or live RSS feeds (rss).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    refresh = args.source == "rss" and ARTICLES_PATH.is_file()
    ingest(source=args.source, refresh=refresh)
