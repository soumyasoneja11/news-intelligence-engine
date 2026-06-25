"""Extract TF-IDF topic keywords and labels for each article cluster."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.cluster import CLUSTER_LABELS_PATH
from src.paths import ARTICLES_PATH, INDEX_DIR

CLUSTER_TOPICS_PATH = INDEX_DIR / "cluster_topics.json"

TOP_KEYWORD_COUNT = 5
LABEL_KEYWORD_COUNT = 3
TFIDF_MAX_FEATURES = 1000


def _load_cluster_labels(path: Path = CLUSTER_LABELS_PATH) -> dict[str, list[int]]:
    if not path.is_file():
        raise FileNotFoundError(f"Cluster labels file not found: {path}")
    try:
        with path.open(encoding="utf-8") as f:
            labels = json.load(f)
    except OSError as exc:
        raise OSError(f"Failed to read cluster labels from {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid cluster labels JSON at {path}") from exc

    if not isinstance(labels, dict):
        raise ValueError(f"Cluster labels must be a JSON object at {path}")

    return labels


def _load_article_texts(path: Path = ARTICLES_PATH) -> pd.Series:
    if not path.is_file():
        raise FileNotFoundError(f"Articles file not found: {path}")
    try:
        df = pd.read_csv(path)
    except OSError as exc:
        raise OSError(f"Failed to read articles from {path}") from exc

    if "text" not in df.columns:
        raise ValueError(f"Articles CSV missing required column: text")

    return df["text"].fillna("").astype(str)


def _extract_keywords(cluster_texts: list[str]) -> list[str]:
    """Fit TF-IDF on this cluster's documents only (not the full corpus)."""
    non_empty = [text for text in cluster_texts if text.strip()]
    if not non_empty:
        return []

    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        stop_words="english",
        ngram_range=(1, 2),
    )
    # fit_transform is scoped to non_empty cluster texts only.
    tfidf_matrix = vectorizer.fit_transform(non_empty)
    scores = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
    feature_names = vectorizer.get_feature_names_out()

    if feature_names.size == 0:
        return []

    top_indices = scores.argsort()[-TOP_KEYWORD_COUNT:][::-1]
    return [str(feature_names[idx]) for idx in top_indices if scores[idx] > 0]


def _build_label(keywords: list[str]) -> str:
    label_terms = keywords[:LABEL_KEYWORD_COUNT]
    return ", ".join(label_terms) if label_terms else "unlabeled cluster"


def _cluster_texts_for_indices(texts: pd.Series, indices: list[int]) -> list[str]:
    cluster_texts: list[str] = []
    for idx in indices:
        if 0 <= idx < len(texts):
            cluster_texts.append(texts.iloc[idx])
    return cluster_texts


def build_cluster_topics(
    cluster_labels_path: Path = CLUSTER_LABELS_PATH,
    articles_path: Path = ARTICLES_PATH,
) -> dict[str, dict[str, object]]:
    """Compute TF-IDF keywords and labels for every cluster."""
    cluster_labels = _load_cluster_labels(cluster_labels_path)
    texts = _load_article_texts(articles_path)

    topics: dict[str, dict[str, object]] = {}
    for cluster_id in sorted(cluster_labels, key=int):
        indices = cluster_labels[cluster_id]
        cluster_texts = _cluster_texts_for_indices(texts, indices)
        keywords = _extract_keywords(cluster_texts)

        topics[cluster_id] = {
            "keywords": keywords,
            "label": _build_label(keywords),
            "article_count": len(cluster_texts),
        }

    return topics


def _save_topics(path: Path, topics: dict[str, dict[str, object]]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(topics, f, indent=2, ensure_ascii=False)
    except OSError as exc:
        raise OSError(f"Failed to write cluster topics to {path}") from exc


def generate_topics(
    cluster_labels_path: Path = CLUSTER_LABELS_PATH,
    articles_path: Path = ARTICLES_PATH,
    output_path: Path = CLUSTER_TOPICS_PATH,
) -> dict[str, dict[str, object]]:
    """Build cluster topics and persist them to JSON."""
    topics = build_cluster_topics(cluster_labels_path, articles_path)
    _save_topics(output_path, topics)
    print(f"Saved cluster topics to {output_path}")

    print("\nCluster labels:")
    for cluster_id in sorted(topics, key=int):
        topic = topics[cluster_id]
        print(
            f"  Cluster {cluster_id} ({topic['article_count']} articles): "
            f"{topic['label']}"
        )

    return topics


if __name__ == "__main__":
    generate_topics()
