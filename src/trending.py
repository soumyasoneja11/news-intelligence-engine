"""Compute trending scores for article clusters based on publication recency."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.cluster import CLUSTER_LABELS_PATH
from src.config import trending_reference_datetime
from src.paths import INDEX_DIR, METADATA_PATH
from src.topics import CLUSTER_TOPICS_PATH
from src.utils import as_id, parse_date

TRENDING_PATH = INDEX_DIR / "trending.json"


def _load_json(path: Path) -> object:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except OSError as exc:
        raise OSError(f"Failed to read JSON from {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON at {path}") from exc


def _load_metadata(path: Path = METADATA_PATH) -> list[dict]:
    metadata = _load_json(path)
    if not isinstance(metadata, list):
        raise ValueError(f"Metadata must be a JSON list at {path}")
    return metadata


def _load_cluster_labels(path: Path = CLUSTER_LABELS_PATH) -> dict[str, list[int]]:
    labels = _load_json(path)
    if not isinstance(labels, dict):
        raise ValueError(f"Cluster labels must be a JSON object at {path}")
    return labels


def _load_cluster_labels_map(path: Path = CLUSTER_TOPICS_PATH) -> dict[str, str]:
    if not path.is_file():
        return {}
    topics = _load_json(path)
    if not isinstance(topics, dict):
        return {}
    return {
        cluster_id: str(topic.get("label", f"Cluster {cluster_id}"))
        for cluster_id, topic in topics.items()
        if isinstance(topic, dict)
    }


def _article_summary(meta: dict) -> dict:
    return {
        "id": as_id(meta.get("id")),
        "title": str(meta.get("title", "")),
        "url": str(meta.get("url", "")),
        "date": str(meta.get("date", "")),
    }


def _cluster_articles(metadata: list[dict], indices: list[int]) -> list[dict]:
    articles: list[dict] = []
    for idx in indices:
        if 0 <= idx < len(metadata):
            articles.append(metadata[idx])
    return articles


def _count_recent(
    articles: list[dict],
    reference: datetime,
) -> tuple[int, int, int]:
    cutoff_24h = reference - timedelta(hours=24)
    cutoff_48h = reference - timedelta(hours=48)
    cutoff_7d = reference - timedelta(days=7)

    count_24h = 0
    count_48h = 0
    count_7d = 0

    for article in articles:
        published = parse_date(article.get("date"))
        if published is None:
            continue
        if published >= cutoff_24h:
            count_24h += 1
        if published >= cutoff_48h:
            count_48h += 1
        if published >= cutoff_7d:
            count_7d += 1

    return count_24h, count_48h, count_7d


def _trending_score(
    count_24h: int,
    count_48h: int,
    count_7d: int,
    total_articles: int,
) -> float:
    if total_articles <= 0:
        return 0.0
    raw = count_24h * 3 + count_48h * 1.5 + count_7d * 0.5
    return float(raw / total_articles)


def _top_articles_by_recency(articles: list[dict], limit: int = 3) -> list[dict]:
    dated: list[tuple[datetime, dict]] = []
    undated: list[dict] = []

    for article in articles:
        published = parse_date(article.get("date"))
        if published is None:
            undated.append(article)
        else:
            dated.append((published, article))

    dated.sort(key=lambda item: item[0], reverse=True)
    ordered = [article for _, article in dated] + undated
    return [_article_summary(article) for article in ordered[:limit]]


def build_trending(
    metadata_path: Path = METADATA_PATH,
    cluster_labels_path: Path = CLUSTER_LABELS_PATH,
    cluster_topics_path: Path = CLUSTER_TOPICS_PATH,
    reference: datetime | None = None,
) -> list[dict]:
    """Compute trending cluster rankings from metadata and cluster assignments."""
    metadata = _load_metadata(metadata_path)
    cluster_labels = _load_cluster_labels(cluster_labels_path)
    label_map = _load_cluster_labels_map(cluster_topics_path)
    now = reference or trending_reference_datetime(metadata)

    results: list[dict] = []
    for cluster_id, indices in cluster_labels.items():
        articles = _cluster_articles(metadata, indices)
        total_articles = len(indices)
        count_24h, count_48h, count_7d = _count_recent(articles, now)
        score = _trending_score(count_24h, count_48h, count_7d, total_articles)

        results.append(
            {
                "cluster_id": cluster_id,
                "label": label_map.get(cluster_id, f"Cluster {cluster_id}"),
                "trending_score": round(score, 4),
                "count_24h": count_24h,
                "count_7d": count_7d,
                "top_articles": _top_articles_by_recency(articles),
            }
        )

    results.sort(key=lambda item: item["trending_score"], reverse=True)
    return results


def _save_trending(path: Path, trending: list[dict]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(trending, f, indent=2, ensure_ascii=False)
    except OSError as exc:
        raise OSError(f"Failed to write trending data to {path}") from exc


def generate_trending(
    metadata_path: Path = METADATA_PATH,
    cluster_labels_path: Path = CLUSTER_LABELS_PATH,
    cluster_topics_path: Path = CLUSTER_TOPICS_PATH,
    output_path: Path = TRENDING_PATH,
    reference: datetime | None = None,
) -> list[dict]:
    """Build trending rankings and save them to JSON."""
    trending = build_trending(
        metadata_path=metadata_path,
        cluster_labels_path=cluster_labels_path,
        cluster_topics_path=cluster_topics_path,
        reference=reference,
    )
    _save_trending(output_path, trending)
    print(f"Saved trending rankings to {output_path}")

    print("\nTop trending clusters:")
    for row in trending[:10]:
        print(
            f"  Cluster {row['cluster_id']} ({row['label']}): "
            f"score={row['trending_score']}, "
            f"24h={row['count_24h']}, 7d={row['count_7d']}"
        )

    return trending


if __name__ == "__main__":
    generate_trending()
