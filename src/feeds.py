"""RSS feed sources and category assignments."""

from __future__ import annotations

FEED_SOURCES: dict[str, list[str]] = {
    "BBC News": [
        "https://feeds.bbci.co.uk/news/rss.xml",
        "https://feeds.bbci.co.uk/news/world/rss.xml",
        "https://feeds.bbci.co.uk/news/technology/rss.xml",
    ],
    "Reuters": [
        "https://feeds.reuters.com/reuters/topNews",
        "https://feeds.reuters.com/reuters/technologyNews",
    ],
    "AP News": [
        "https://rsshub.app/apnews/topics/ap-top-news",
    ],
    "Al Jazeera": [
        "https://www.aljazeera.com/xml/rss/all.xml",
    ],
    "The Guardian": [
        "https://www.theguardian.com/world/rss",
    ],
    "TechCrunch": [
        "https://techcrunch.com/feed/",
    ],
    "Ars Technica": [
        "https://feeds.arstechnica.com/arstechnica/index",
    ],
}

CATEGORY_MAP: dict[str, str] = {
    "BBC News": "world",
    "Reuters": "business",
    "AP News": "world",
    "Al Jazeera": "world",
    "The Guardian": "world",
    "TechCrunch": "tech",
    "Ars Technica": "tech",
}

SOURCE_SLUGS: dict[str, str] = {
    "bbc": "BBC News",
    "reuters": "Reuters",
    "ap": "AP News",
    "aljazeera": "Al Jazeera",
    "guardian": "The Guardian",
    "techcrunch": "TechCrunch",
    "arstechnica": "Ars Technica",
}

SOURCE_DOMAIN_PATTERNS: dict[str, tuple[str, ...]] = {
    "BBC News": ("bbc.co.uk", "bbc.com"),
    "Reuters": ("reuters.com",),
    "AP News": ("apnews.com",),
    "Al Jazeera": ("aljazeera.com",),
    "The Guardian": ("theguardian.com",),
    "TechCrunch": ("techcrunch.com",),
    "Ars Technica": ("arstechnica.com",),
}


def domain_to_source(domain: str) -> str | None:
    """Map an article domain to a configured feed source name."""
    domain_lower = str(domain or "").lower()
    if not domain_lower:
        return None
    for source_name, patterns in SOURCE_DOMAIN_PATTERNS.items():
        if any(pattern in domain_lower for pattern in patterns):
            return source_name
    return None


def matches_source_filter(domain: str, selected_slugs: list[str] | None) -> bool:
    """Return True when domain matches one of the selected source slugs."""
    if not selected_slugs:
        return True
    source_name = domain_to_source(domain)
    if source_name is None:
        return False
    allowed = {SOURCE_SLUGS[slug] for slug in selected_slugs if slug in SOURCE_SLUGS}
    return source_name in allowed
