"""Cluster article embeddings and compute 2D coordinates for visualization."""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Iterable
from pathlib import Path

import faiss
import numpy as np
from sklearn.cluster import KMeans

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.paths import EMBEDDINGS_PATH, INDEX_DIR

CLUSTER_DATA_PATH = INDEX_DIR / "cluster_data.json"
CLUSTER_LABELS_PATH = INDEX_DIR / "cluster_labels.json"

RANDOM_STATE = 42
UMAP_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1


def _load_embeddings(embeddings_path: Path = EMBEDDINGS_PATH) -> np.ndarray:
    if not embeddings_path.is_file():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    try:
        embeddings = np.load(embeddings_path)
    except OSError as exc:
        raise OSError(f"Failed to read embeddings from {embeddings_path}") from exc
    return np.asarray(embeddings, dtype=np.float32)


def _l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize rows to match the FAISS index cosine-similarity geometry."""
    vectors = np.asarray(embeddings, dtype=np.float32).copy()
    faiss.normalize_L2(vectors)
    return vectors


def _elbow_second_derivative(ks: list[int], inertias: list[float]) -> int:
    """Pick the elbow as the k with the largest curvature on the inertia curve."""
    x = np.asarray(ks, dtype=np.float64)
    y = np.asarray(inertias, dtype=np.float64)
    first_derivative = np.gradient(y, x)
    second_derivative = np.gradient(first_derivative, x)

    if len(ks) > 2:
        inner = np.abs(second_derivative[1:-1])
        return int(ks[int(np.argmax(inner)) + 1])

    return int(ks[int(np.argmax(np.abs(second_derivative)))])


def find_optimal_k(
    embeddings: np.ndarray,
    k_range: Iterable[int] = range(5, 30),
    random_state: int = RANDOM_STATE,
) -> int:
    """Estimate the elbow k by scanning KMeans inertia over a range of cluster counts."""
    n_samples = embeddings.shape[0]
    ks = [k for k in k_range if 1 < k < n_samples]
    if not ks:
        return max(1, min(2, n_samples))

    inertias: list[float] = []
    for k in ks:
        model = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        model.fit(embeddings)
        inertias.append(float(model.inertia_))

    try:
        from kneed import KneeLocator

        locator = KneeLocator(ks, inertias, curve="convex", direction="decreasing")
        if locator.elbow is not None:
            return int(locator.elbow)
    except ImportError:
        pass

    return _elbow_second_derivative(ks, inertias)


def _cluster_embeddings(embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
    model = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init="auto")
    return model.fit_predict(embeddings)


def _reduce_to_2d(embeddings: np.ndarray) -> tuple[np.ndarray, str]:
    try:
        import umap

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=UMAP_NEIGHBORS,
            min_dist=UMAP_MIN_DIST,
            random_state=RANDOM_STATE,
        )
        coords = reducer.fit_transform(embeddings)
        return np.asarray(coords, dtype=np.float32), "umap"
    except ImportError:
        from sklearn.decomposition import PCA

        coords = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(embeddings)
        return np.asarray(coords, dtype=np.float32), "pca"


def _build_cluster_data(cluster_ids: np.ndarray, coords: np.ndarray) -> list[dict]:
    records: list[dict] = []
    for cluster_id, (x, y) in zip(cluster_ids, coords):
        records.append(
            {
                "cluster_id": str(int(cluster_id)),
                "umap_x": float(x),
                "umap_y": float(y),
            }
        )
    return records


def _build_cluster_labels(cluster_ids: np.ndarray, n_clusters: int) -> dict[str, list[int]]:
    labels: dict[str, list[int]] = {str(cluster_id): [] for cluster_id in range(n_clusters)}
    for article_idx, cluster_id in enumerate(cluster_ids):
        labels[str(int(cluster_id))].append(article_idx)
    return labels


def _save_json(path: Path, payload: object) -> None:
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except OSError as exc:
        raise OSError(f"Failed to write JSON to {path}") from exc


def cluster(
    embeddings_path: Path = EMBEDDINGS_PATH,
    cluster_data_path: Path = CLUSTER_DATA_PATH,
    cluster_labels_path: Path = CLUSTER_LABELS_PATH,
    k_range: Iterable[int] = range(5, 30),
) -> tuple[list[dict], dict[str, list[int]], str, int]:
    """Cluster embeddings, reduce to 2D, and persist cluster artifacts."""
    try:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(f"Failed to create index directory {INDEX_DIR}") from exc

    embeddings = _load_embeddings(embeddings_path)
    print(f"Loaded embeddings with shape {embeddings.shape}")

    normalized = _l2_normalize(embeddings)
    print("Using L2-normalized embeddings for clustering (consistent with FAISS index).")

    print(f"Finding optimal k over {list(k_range)}...")
    optimal_k = find_optimal_k(normalized, k_range=k_range)
    print(f"Recommended number of clusters (elbow): {optimal_k}")

    print(f"Running KMeans (n_clusters={optimal_k})...")
    cluster_ids = _cluster_embeddings(normalized, n_clusters=optimal_k)

    print("Reducing to 2D for visualization...")
    coords, method = _reduce_to_2d(normalized)
    print(f"2D reduction method: {method}")

    cluster_data = _build_cluster_data(cluster_ids, coords)
    cluster_labels = _build_cluster_labels(cluster_ids, n_clusters=optimal_k)

    _save_json(cluster_data_path, cluster_data)
    print(f"Saved cluster coordinates to {cluster_data_path}")

    _save_json(cluster_labels_path, cluster_labels)
    print(f"Saved cluster label map to {cluster_labels_path}")

    return cluster_data, cluster_labels, method, optimal_k


if __name__ == "__main__":
    start = time.perf_counter()
    cluster()
    elapsed = time.perf_counter() - start
    print(f"Clustering completed in {elapsed:.1f}s")
