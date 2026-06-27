"""Fetch RSS feeds and normalize entries into article dicts."""

from __future__ import annotations

import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import feedparser
import pandas as pd
import requests
from bs4 import BeautifulSoup
from dateutil import parser as date_parser
from dateutil.parser import ParserError

from src.feeds import CATEGORY_MAP, FEED_SOURCES, SOURCE_SLUGS, domain_to_source
from src.ingest import strip_html
from src.paths import ARTICLES_PATH, FEED_STATS_PATH
from src.utils import parse_date

USER_AGENT = "NewsIntelligenceBot/1.0"
_STRIP_TAGS = ("script", "style", "nav", "header", "footer")
_CONTENT_CLASS_HINTS = ("article", "content", "body")
_MIN_BODY_LEN = 100
_TRACKING_PARAMS = frozenset(
    {"utm_source", "utm_medium", "utm_campaign", "utm_content", "ref", "source"}
)


def _clean_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path
    if path.endswith("/amp/"):
        path = path[:-5] or "/"
    elif path.endswith("/amp"):
        path = path[:-4] or "/"

    cleaned_pairs: list[tuple[str, str]] = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        key_lower = key.lower()
        if key_lower in _TRACKING_PARAMS:
            continue
        if key_lower == "amp" and value.lower() in ("1", "true"):
            continue
        cleaned_pairs.append((key, value))

    query = urlencode(cleaned_pairs, doseq=True)
    return urlunparse((parsed.scheme, parsed.netloc, path, parsed.params, query, parsed.fragment))


def resolve_canonical_url(url: str, timeout: int = 5) -> str:
    """Follow redirects and strip AMP paths, tracking params, and other feed noise."""
    cleaned_input = _clean_url(url.strip()) if url else ""
    if not cleaned_input:
        return url

    try:
        response = requests.head(
            cleaned_input,
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
            allow_redirects=True,
        )
        resolved = response.url or cleaned_input
    except requests.RequestException:
        resolved = cleaned_input
    except Exception:
        resolved = cleaned_input

    return _clean_url(resolved)


def _entry_link(entry: object) -> str:
    link = str(getattr(entry, "link", "") or "").strip()
    if link:
        return link
    for item in getattr(entry, "links", []) or []:
        href = ""
        if isinstance(item, dict):
            href = str(item.get("href", "") or "").strip()
        else:
            href = str(getattr(item, "href", "") or "").strip()
        if href:
            return href
    return ""


def _entry_summary(entry: object) -> str:
    summary = str(getattr(entry, "summary", "") or getattr(entry, "description", "") or "").strip()
    if summary:
        return strip_html(summary)
    content_list = getattr(entry, "content", None) or []
    if content_list:
        first = content_list[0]
        value = first.get("value") if isinstance(first, dict) else getattr(first, "value", "")
        return strip_html(str(value or ""))
    return ""


def _parse_published(entry: object) -> datetime | None:
    raw = getattr(entry, "published", None) or getattr(entry, "updated", None)
    if raw:
        try:
            return date_parser.parse(str(raw))
        except (ParserError, ValueError, TypeError, OverflowError):
            pass

    parsed = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
    if parsed:
        try:
            return datetime(*parsed[:6], tzinfo=timezone.utc)
        except (TypeError, ValueError):
            pass
    return None


def _strip_noise_tags(root) -> None:
    for tag_name in _STRIP_TAGS:
        for tag in root.find_all(tag_name):
            tag.decompose()


def _content_root(soup: BeautifulSoup):
    article = soup.find("article")
    if article is not None:
        return article

    main = soup.find("main")
    if main is not None:
        return main

    for div in soup.find_all("div", class_=True):
        class_names = div.get("class", [])
        class_text = " ".join(str(name) for name in class_names).lower()
        if any(hint in class_text for hint in _CONTENT_CLASS_HINTS):
            return div

    return soup.body or soup


def _text_from_root(root) -> str:
    _strip_noise_tags(root)
    return " ".join(root.stripped_strings)


def get_article_body(url: str, timeout: int = 8, summary: str = "") -> str:
    """Fetch full article text from a URL, falling back to summary on failure or short extract."""
    try:
        response = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
            allow_redirects=True,
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        text = _text_from_root(_content_root(soup))
        if len(text) < _MIN_BODY_LEN:
            return summary
        return text
    except Exception:
        return summary


def fetch_feed(url: str, source_name: str, timeout: int = 10) -> list[dict]:
    """Fetch one RSS feed and return normalized article dicts."""
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    try:
        response = session.get(url, timeout=timeout, allow_redirects=True)
        response.raise_for_status()
        parsed = feedparser.parse(response.content)
    except requests.RequestException:
        return []

    category = CATEGORY_MAP.get(source_name, "")
    articles: list[dict] = []

    for entry in parsed.entries:
        title = strip_html(str(getattr(entry, "title", "") or "").strip())
        link = resolve_canonical_url(_entry_link(entry))
        if not title or not link:
            continue

        articles.append(
            {
                "title": title,
                "link": link,
                "summary": _entry_summary(entry),
                "published": _parse_published(entry),
                "source_name": source_name,
                "category": category,
            }
        )

    return articles


def fetch_all_feeds(sources: dict[str, list[str]] | None = None) -> list[dict]:
    """Fetch all configured feeds concurrently, dedupe by link, and return articles."""
    feed_sources = sources if sources is not None else FEED_SOURCES
    tasks = [(url, source_name) for source_name, urls in feed_sources.items() for url in urls]

    by_source: dict[str, list[dict]] = defaultdict(list)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(fetch_feed, url, source_name): source_name
            for url, source_name in tasks
        }
        for future in as_completed(futures):
            source_name = futures[future]
            by_source[source_name].extend(future.result())

    for source_name in feed_sources:
        count = len(by_source.get(source_name, []))
        print(f"{source_name}: {count} articles")

    seen_links: set[str] = set()
    combined: list[dict] = []
    for source_name in feed_sources:
        for article in by_source.get(source_name, []):
            link = article.get("link", "")
            if not link or link in seen_links:
                continue
            seen_links.add(link)
            combined.append(article)

    write_feed_stats(mark_fetched=True)
    return combined


def write_feed_stats(*, mark_fetched: bool = False) -> None:
    """Write per-source article counts and last article times to feed_stats.json."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    counts: dict[str, int] = {name: 0 for name in FEED_SOURCES}
    last_times: dict[str, datetime | None] = {name: None for name in FEED_SOURCES}
    slug_by_name = {label: slug for slug, label in SOURCE_SLUGS.items()}

    if ARTICLES_PATH.is_file():
        try:
            df = pd.read_csv(ARTICLES_PATH)
        except (OSError, pd.errors.EmptyDataError):
            df = pd.DataFrame()
        for _, row in df.iterrows():
            source_name = domain_to_source(str(row.get("domain", "")))
            if source_name is None or source_name not in counts:
                continue
            counts[source_name] += 1
            published = parse_date(row.get("date"))
            if published is None:
                continue
            current = last_times[source_name]
            if current is None or published > current:
                last_times[source_name] = published

    sources = []
    for name in FEED_SOURCES:
        last_dt = last_times[name]
        sources.append(
            {
                "name": name,
                "slug": slug_by_name.get(name, ""),
                "article_count": counts[name],
                "last_article_time": last_dt.strftime("%Y-%m-%d %H:%M:%S") if last_dt else None,
            }
        )

    payload: dict[str, object] = {
        "updated_at": now,
        "sources": sources,
    }

    last_fetch_at = now if mark_fetched else None
    if not mark_fetched and FEED_STATS_PATH.is_file():
        try:
            with FEED_STATS_PATH.open(encoding="utf-8") as stats_file:
                existing = json.load(stats_file)
            if isinstance(existing, dict):
                last_fetch_at = existing.get("last_fetch_at")
        except (OSError, json.JSONDecodeError):
            pass
    if last_fetch_at:
        payload["last_fetch_at"] = last_fetch_at

    try:
        FEED_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with FEED_STATS_PATH.open("w", encoding="utf-8") as stats_file:
            json.dump(payload, stats_file, indent=2)
    except OSError as exc:
        raise OSError(f"Failed to write feed stats to {FEED_STATS_PATH}") from exc
