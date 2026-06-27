"""Fetch and append BBC RSS feeds into the articles CSV."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import feedparser
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import MAX_ARTICLES, RSS_FEED_CONFIG_PATH, RSS_POLL_ENABLED
from src.ingest import MIN_TEXT_LEN, _as_str, clean_records, strip_html
from src.paths import ARTICLES_PATH, INDEX_DIR
from src.utils import format_date

RSS_INGEST_STATE_PATH = INDEX_DIR / "rss_ingest_state.json"
DEFAULT_DOMAIN = "bbc.co.uk"


def _load_feed_config(config_path: Path | None = None) -> list[dict]:
    path = Path(config_path or RSS_FEED_CONFIG_PATH)
    if not path.is_file():
        raise FileNotFoundError(f"RSS feed config not found: {path}")
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    feeds = data.get("feeds", [])
    if not isinstance(feeds, list) or not feeds:
        raise ValueError(f"No feeds defined in {path}")
    return feeds


def _entry_date(entry: object) -> str:
    published = getattr(entry, "published", None) or getattr(entry, "updated", None)
    if published:
        return format_date(published)
    parsed = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
    if parsed:
        try:
            dt = datetime(*parsed[:6], tzinfo=timezone.utc)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except (TypeError, ValueError):
            pass
    return format_date("")


def _entry_link(entry: object) -> str:
    link = _as_str(getattr(entry, "link", ""))
    if link:
        return link
    links = getattr(entry, "links", []) or []
    for item in links:
        href = _as_str(getattr(item, "href", "") or (item.get("href") if isinstance(item, dict) else ""))
        if href:
            return href
    return ""


def _entry_description(entry: object) -> str:
    summary = _as_str(getattr(entry, "summary", "")) or _as_str(getattr(entry, "description", ""))
    if summary:
        return strip_html(summary)
    content_list = getattr(entry, "content", None) or []
    if content_list:
        value = content_list[0].get("value") if isinstance(content_list[0], dict) else getattr(
            content_list[0], "value", ""
        )
        return strip_html(_as_str(value))
    return ""


def _entry_to_record(entry: object, feed_name: str) -> dict | None:
    title = strip_html(_as_str(getattr(entry, "title", "")))
    url = _entry_link(entry)
    if not title or not url:
        return None

    domain = urlparse(url).netloc.strip() or DEFAULT_DOMAIN
    description = _entry_description(entry)
    return {
        "title": title,
        "description": description,
        "url": url,
        "date": _entry_date(entry),
        "domain": domain,
        "feed": feed_name,
    }


def fetch_bbc_rss(config_path: Path | None = None) -> tuple[list[dict], list[str]]:
    """Fetch all configured RSS feeds and return raw records plus error messages."""
    if not RSS_POLL_ENABLED:
        raise RuntimeError("RSS polling is disabled (RSS_POLL_ENABLED=false).")

    feeds = _load_feed_config(config_path)
    records: list[dict] = []
    errors: list[str] = []

    for feed in feeds:
        name = _as_str(feed.get("name")) or "RSS feed"
        url = _as_str(feed.get("url"))
        if not url:
            errors.append(f"{name}: missing url")
            continue
        try:
            parsed = feedparser.parse(url)
            if getattr(parsed, "bozo", False) and getattr(parsed, "bozo_exception", None):
                errors.append(f"{name}: parse warning ({parsed.bozo_exception})")
            for entry in parsed.entries:
                record = _entry_to_record(entry, name)
                if record is not None:
                    records.append(record)
        except OSError as exc:
            errors.append(f"{name}: {exc}")

    return records, errors


def fetch_bbc_rss_from_file(path: Path) -> list[dict]:
    """Parse a local RSS/Atom XML file (for tests)."""
    parsed = feedparser.parse(str(path))
    records: list[dict] = []
    for entry in parsed.entries:
        record = _entry_to_record(entry, "fixture")
        if record is not None:
            records.append(record)
    return records


def _cap_dataframe(df: pd.DataFrame, max_articles: int) -> pd.DataFrame:
    if len(df) <= max_articles:
        return df
    return df.iloc[:max_articles].copy()


def _save_state(
    new_count: int,
    total_count: int,
    feeds_polled: int,
    errors: list[str],
    path: Path = RSS_INGEST_STATE_PATH,
) -> None:
    payload = {
        "last_run": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "new_articles": new_count,
        "total_articles": total_count,
        "feeds_polled": feeds_polled,
        "errors": errors,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except OSError as exc:
        raise OSError(f"Failed to write RSS ingest state to {path}") from exc


def append_rss_to_csv(
    output_path: Path = ARTICLES_PATH,
    config_path: Path | None = None,
    max_articles: int = MAX_ARTICLES,
) -> tuple[int, int]:
    """Fetch RSS feeds and prepend new articles to the CSV. Returns (new_count, total_count)."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(f"Failed to create data directory {output_path.parent}") from exc

    feeds = _load_feed_config(config_path)
    raw_records, errors = fetch_bbc_rss(config_path)
    cleaned = clean_records(raw_records)

    known_urls: set[str] = set()
    existing_df: pd.DataFrame | None = None
    if output_path.is_file():
        existing_df = pd.read_csv(output_path)
        known_urls = set(existing_df["url"].map(_as_str))

    new_rows = [row for row in cleaned if _as_str(row.get("url")) not in known_urls]
    new_count = len(new_rows)

    if new_count == 0 and existing_df is not None:
        _save_state(0, len(existing_df), len(feeds), errors)
        print("No new RSS articles to append.")
        return 0, len(existing_df)

    new_df = pd.DataFrame(new_rows)
    if existing_df is not None and not existing_df.empty:
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df

    combined = _cap_dataframe(combined, max_articles)
    combined.to_csv(output_path, index=False)
    total_count = len(combined)

    _save_state(new_count, total_count, len(feeds), errors)
    print(f"Appended {new_count} new RSS articles ({total_count} total, cap={max_articles}).")
    if errors:
        print("RSS warnings/errors:")
        for err in errors:
            print(f"  - {err}")

    return new_count, total_count


def ingest_rss_initial(
    output_path: Path = ARTICLES_PATH,
    config_path: Path | None = None,
    max_articles: int = MAX_ARTICLES,
) -> pd.DataFrame:
    """Initial RSS ingest when no CSV exists yet."""
    raw_records, errors = fetch_bbc_rss(config_path)
    cleaned = clean_records(raw_records)
    if not cleaned:
        raise ValueError("No articles remained after RSS cleaning; refusing to write empty CSV.")

    df = pd.DataFrame(cleaned)
    df = _cap_dataframe(df, max_articles)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    feeds = _load_feed_config(config_path)
    _save_state(len(df), len(df), len(feeds), errors)
    print(f"Saved {len(df)} RSS articles to {output_path}")
    return df


if __name__ == "__main__":
    start = time.perf_counter()
    if ARTICLES_PATH.is_file():
        new, total = append_rss_to_csv()
        print(f"Refresh complete: {new} new, {total} total.")
    else:
        ingest_rss_initial()
    print(f"Elapsed: {time.perf_counter() - start:.1f}s")
