"""Unified facade over Day 1 and Day 2 news intelligence modules."""

from __future__ import annotations

import json
import sys
from functools import cached_property
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.cluster import CLUSTER_DATA_PATH, CLUSTER_LABELS_PATH
from src.dedup import DUPLICATE_PAIRS_PATH
from src.embedder import EMBEDDINGS_PATH, FAISS_PATH, METADATA_PATH, MODEL_NAME
from src.index_status import (
    DASHBOARD_ARTIFACTS,
    SEARCH_ARTIFACTS,
    index_is_ready,
    missing_artifacts,
)
from src.paths import PROJECT_ROOT
from src.retriever import NewsRetriever
from src.topics import CLUSTER_TOPICS_PATH
from src.trending import TRENDING_PATH
from src.feeds import matches_source_filter
from src.utils import as_id


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except OSError as exc:
        raise OSError(f"Failed to read JSON from {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON at {path}") from exc


def _as_str(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


class IntelligenceEngine:
    """Lazy-loaded facade for search, clustering, dedup, and trending."""

    def __init__(
        self,
        project_root: Path | None = None,
        faiss_path: Path | None = None,
        metadata_path: Path | None = None,
        embeddings_path: Path | None = None,
        cluster_data_path: Path | None = None,
        cluster_labels_path: Path | None = None,
        cluster_topics_path: Path | None = None,
        duplicate_pairs_path: Path | None = None,
        trending_path: Path | None = None,
        model_name: str = MODEL_NAME,
    ) -> None:
        self.project_root = project_root or PROJECT_ROOT
        self.faiss_path = faiss_path or FAISS_PATH
        self.metadata_path = metadata_path or METADATA_PATH
        self.embeddings_path = embeddings_path or EMBEDDINGS_PATH
        self.cluster_data_path = cluster_data_path or CLUSTER_DATA_PATH
        self.cluster_labels_path = cluster_labels_path or CLUSTER_LABELS_PATH
        self.cluster_topics_path = cluster_topics_path or CLUSTER_TOPICS_PATH
        self.duplicate_pairs_path = duplicate_pairs_path or DUPLICATE_PAIRS_PATH
        self.trending_path = trending_path or TRENDING_PATH
        self.model_name = model_name

    @cached_property
    def retriever(self) -> NewsRetriever:
        return NewsRetriever(
            faiss_path=self.faiss_path,
            metadata_path=self.metadata_path,
            embeddings_path=self.embeddings_path,
            model_name=self.model_name,
        )

    @cached_property
    def metadata(self) -> list[dict]:
        data = _load_json(self.metadata_path)
        if not isinstance(data, list):
            raise ValueError(f"Metadata must be a JSON list at {self.metadata_path}")
        return [
            {**entry, "id": as_id(entry.get("id"), idx)}
            for idx, entry in enumerate(data)
        ]

    @cached_property
    def cluster_data(self) -> list[dict]:
        data = _load_json(self.cluster_data_path)
        if not isinstance(data, list):
            raise ValueError(f"Cluster data must be a JSON list at {self.cluster_data_path}")
        return data

    @cached_property
    def cluster_labels(self) -> dict[str, list[int]]:
        data = _load_json(self.cluster_labels_path)
        if not isinstance(data, dict):
            raise ValueError(f"Cluster labels must be a JSON object at {self.cluster_labels_path}")
        return {str(cluster_id): indices for cluster_id, indices in data.items()}

    @cached_property
    def cluster_topics(self) -> dict[str, dict]:
        if not self.cluster_topics_path.is_file():
            return {}
        data = _load_json(self.cluster_topics_path)
        if not isinstance(data, dict):
            raise ValueError(f"Cluster topics must be a JSON object at {self.cluster_topics_path}")
        return {str(cluster_id): topic for cluster_id, topic in data.items()}

    @cached_property
    def duplicate_pairs(self) -> list[dict]:
        data = _load_json(self.duplicate_pairs_path)
        if not isinstance(data, list):
            raise ValueError(f"Duplicate pairs must be a JSON list at {self.duplicate_pairs_path}")
        return data

    @cached_property
    def trending(self) -> list[dict]:
        data = _load_json(self.trending_path)
        if not isinstance(data, list):
            raise ValueError(f"Trending data must be a JSON list at {self.trending_path}")
        return data

    def index_is_ready(self, minimal: bool = False) -> bool:
        """Return True when required on-disk artifacts exist."""
        return index_is_ready(minimal=minimal)

    def missing_artifacts(self) -> list[Path]:
        return missing_artifacts()

    def require_index(self, minimal: bool = False) -> None:
        paths = SEARCH_ARTIFACTS if minimal else DASHBOARD_ARTIFACTS
        missing = missing_artifacts(paths)
        if missing:
            raise FileNotFoundError(
                f"Required artifact not found: {missing[0]}"
            )

    def search(self, query: str, k: int = 10, sources: list[str] | None = None) -> list[dict]:
        """Semantic search over indexed articles."""
        self.require_index(minimal=True)
        if not sources:
            return self.retriever.search(query, k=k)

        fetch_k = min(len(self.metadata), max(k * 10, 50))
        results = self.retriever.search(query, k=fetch_k)
        filtered = [
            result
            for result in results
            if matches_source_filter(result.get("domain", ""), sources)
        ]
        return filtered[:k]

    def find_similar(self, article_id: str | int, k: int = 10) -> list[dict]:
        """Find articles similar to a given article id."""
        self.require_index(minimal=True)
        return self.retriever.find_similar(article_id, k=k)

    def get_clusters(self, sources: list[str] | None = None) -> list[dict]:
        """Return cluster summaries with labels, counts, and 2D coordinates."""
        self.require_index(minimal=False)
        clusters: list[dict] = []
        for cluster_id in sorted(self.cluster_labels, key=int):
            topic = self.cluster_topics.get(str(cluster_id), {})
            article_indices = self.cluster_labels[str(cluster_id)]
            points: list[dict] = []
            filtered_indices: list[int] = []

            for article_idx in article_indices:
                if not (0 <= article_idx < len(self.cluster_data)):
                    continue
                meta = self.metadata[article_idx] if article_idx < len(self.metadata) else {}
                if sources and not matches_source_filter(meta.get("domain", ""), sources):
                    continue

                filtered_indices.append(article_idx)
                coords = self.cluster_data[article_idx]
                points.append(
                    {
                        "article_idx": article_idx,
                        "cluster_id": as_id(coords.get("cluster_id"), cluster_id),
                        "umap_x": float(coords.get("umap_x", 0.0)),
                        "umap_y": float(coords.get("umap_y", 0.0)),
                        "title": _as_str(meta.get("title")),
                        "url": _as_str(meta.get("url")),
                    }
                )

            if sources and not filtered_indices:
                continue

            clusters.append(
                {
                    "cluster_id": str(cluster_id),
                    "label": _as_str(topic.get("label")) or f"Cluster {cluster_id}",
                    "keywords": list(topic.get("keywords", [])),
                    "article_count": len(filtered_indices) if sources else len(article_indices),
                    "points": points,
                }
            )

        return clusters

    def get_trending(self, n: int = 10) -> list[dict]:
        """Return the top-n trending clusters."""
        self.require_index(minimal=False)
        return self.trending[: max(n, 0)]

    def get_duplicates(self) -> list[dict]:
        """Return detected near-duplicate article pairs."""
        self.require_index(minimal=False)
        return self.duplicate_pairs

    def get_cluster_articles(self, cluster_id: str | int) -> list[dict]:
        """Return full article metadata for a cluster."""
        self.require_index(minimal=False)
        key = str(cluster_id)
        if key not in self.cluster_labels:
            raise KeyError(f"Unknown cluster_id: {cluster_id}")

        articles: list[dict] = []
        for article_idx in self.cluster_labels[key]:
            if not (0 <= article_idx < len(self.metadata)):
                continue

            meta = self.metadata[article_idx]
            article = {
                "id": as_id(meta.get("id"), article_idx),
                "title": _as_str(meta.get("title")),
                "url": _as_str(meta.get("url")),
                "date": _as_str(meta.get("date")),
                "domain": _as_str(meta.get("domain")),
            }

            if 0 <= article_idx < len(self.cluster_data):
                coords = self.cluster_data[article_idx]
                article["cluster_id"] = as_id(coords.get("cluster_id"), cluster_id)
                article["umap_x"] = float(coords.get("umap_x", 0.0))
                article["umap_y"] = float(coords.get("umap_y", 0.0))

            articles.append(article)

        return articles


def load_intelligence_engine() -> IntelligenceEngine:
    """Create an engine instance for use with Streamlit `@st.cache_resource`."""
    return IntelligenceEngine()


if __name__ == "__main__":
    engine = load_intelligence_engine()
    print(f"Articles indexed: {len(engine.metadata)}")
    print(f"Clusters: {len(engine.get_clusters())}")
    print(f"Duplicates: {len(engine.get_duplicates())}")
    print(f"Top trending: {engine.get_trending(3)}")
    print(f"Search sample: {engine.search('climate change', k=2)}")
