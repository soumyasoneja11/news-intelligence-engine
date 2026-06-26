"""Runtime configuration from environment variables."""

from __future__ import annotations

import os
from datetime import datetime

from src.utils import parse_date

# FAISS: "flat" | "hnsw" | "auto" (HNSW when n >= FAISS_HNSW_MIN_VECTORS)
FAISS_INDEX_MODE = os.environ.get("FAISS_INDEX_MODE", "auto").lower()
FAISS_HNSW_MIN_VECTORS = int(os.environ.get("FAISS_HNSW_MIN_VECTORS", "5000"))
FAISS_HNSW_M = int(os.environ.get("FAISS_HNSW_M", "32"))
FAISS_HNSW_EF_CONSTRUCTION = int(os.environ.get("FAISS_HNSW_EF_CONSTRUCTION", "40"))

# Sidebar rebuild button (set false on Streamlit Cloud / public deploy)
ALLOW_REBUILD = os.environ.get("ALLOW_REBUILD", "true").lower() in ("1", "true", "yes")

# Optional app password (APP_PASSWORD env or Streamlit secrets)
APP_PASSWORD = os.environ.get("APP_PASSWORD", "")

# Trending reference: ISO date string, or "auto" to anchor on latest article date
TRENDING_REFERENCE = os.environ.get("TRENDING_REFERENCE", "auto")


def trending_reference_datetime(metadata: list[dict] | None = None) -> datetime:
    """Resolve the 'now' anchor used for trending recency windows."""
    if TRENDING_REFERENCE and TRENDING_REFERENCE.lower() not in ("auto", ""):
        parsed = parse_date(TRENDING_REFERENCE)
        if parsed is not None:
            return parsed

    if metadata:
        latest: datetime | None = None
        for entry in metadata:
            published = parse_date(entry.get("date"))
            if published is not None and (latest is None or published > latest):
                latest = published
        if latest is not None:
            return latest

    return datetime.now()


def rebuild_allowed() -> bool:
    return ALLOW_REBUILD
