"""Shared utility helpers."""

from __future__ import annotations

from datetime import datetime

DATE_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y",
    "%a, %d %b %Y %H:%M:%S %z",
    "%a, %d %b %Y %H:%M:%S",
    "%B %d, %Y",
    "%b %d, %Y",
)

_EMPTY_VALUES = frozenset({"", "nan", "none", "null", "nat"})


def as_id(value: object, fallback: object = "") -> str:
    """Normalize article/cluster identifiers to strings for JSON round-trips."""
    if value is None:
        return str(fallback)
    text = str(value).strip()
    return text if text else str(fallback)


def _normalize_datetime(value: datetime) -> datetime:
    if value.tzinfo is not None:
        return value.replace(tzinfo=None)
    return value


def parse_date(date_str: object) -> datetime | None:
    """Parse mixed-format date strings; return None when parsing fails."""
    if date_str is None:
        return None

    text = str(date_str).strip()
    if not text or text.lower() in _EMPTY_VALUES:
        return None

    try:
        from dateutil import parser as date_parser
        from dateutil.parser import ParserError

        return _normalize_datetime(date_parser.parse(text))
    except ImportError:
        pass
    except (ParserError, ValueError, TypeError, OverflowError):
        pass

    for fmt in DATE_FORMATS:
        try:
            return _normalize_datetime(datetime.strptime(text, fmt))
        except ValueError:
            continue

    return None


def format_date(date_str: object, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Return a normalized date string, or the original text if parsing fails."""
    if date_str is None:
        return ""

    text = str(date_str).strip()
    if not text:
        return ""

    parsed = parse_date(text)
    return parsed.strftime(fmt) if parsed else text
