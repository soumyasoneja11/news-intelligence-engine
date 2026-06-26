"""Check whether prebuilt index artifacts are present and consistent."""

from __future__ import annotations

from pathlib import Path

from src.paths import (
    ARTICLES_PATH,
    EMBEDDINGS_PATH,
    FAISS_PATH,
    INDEX_DIR,
    METADATA_PATH,
)

CLUSTER_DATA_PATH = INDEX_DIR / "cluster_data.json"
CLUSTER_LABELS_PATH = INDEX_DIR / "cluster_labels.json"
CLUSTER_TOPICS_PATH = INDEX_DIR / "cluster_topics.json"
DUPLICATE_PAIRS_PATH = INDEX_DIR / "duplicate_pairs.json"
TRENDING_PATH = INDEX_DIR / "trending.json"

# Minimum files required for search
SEARCH_ARTIFACTS: tuple[Path, ...] = (
    FAISS_PATH,
    METADATA_PATH,
    EMBEDDINGS_PATH,
)

# Full dashboard experience
DASHBOARD_ARTIFACTS: tuple[Path, ...] = (
    *SEARCH_ARTIFACTS,
    CLUSTER_DATA_PATH,
    CLUSTER_LABELS_PATH,
    CLUSTER_TOPICS_PATH,
    DUPLICATE_PAIRS_PATH,
    TRENDING_PATH,
)


def missing_artifacts(paths: tuple[Path, ...] = DASHBOARD_ARTIFACTS) -> list[Path]:
    return [path for path in paths if not path.is_file()]


def index_is_ready(minimal: bool = False) -> bool:
    paths = SEARCH_ARTIFACTS if minimal else DASHBOARD_ARTIFACTS
    return len(missing_artifacts(paths)) == 0


def index_status_message(missing: list[Path] | None = None) -> str:
    if missing is None:
        missing = missing_artifacts()
    if not missing:
        return "Index ready"
    names = ", ".join(path.name for path in missing[:3])
    suffix = f" (+{len(missing) - 3} more)" if len(missing) > 3 else ""
    return f"Missing index files: {names}{suffix}"


def articles_csv_present() -> bool:
    return ARTICLES_PATH.is_file()
