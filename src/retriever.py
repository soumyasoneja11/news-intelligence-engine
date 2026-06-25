"""Semantic search over indexed news articles."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.embedder import MODEL_NAME
from src.paths import EMBEDDINGS_PATH, FAISS_PATH, METADATA_PATH
from src.utils import as_id


def _as_str(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value).strip()


class NewsRetriever:
    """Search news articles by text query or by article similarity."""

    def __init__(
        self,
        faiss_path: Path = FAISS_PATH,
        metadata_path: Path = METADATA_PATH,
        embeddings_path: Path = EMBEDDINGS_PATH,
        model_name: str = MODEL_NAME,
    ) -> None:
        self.index = self._load_index(faiss_path)
        self.metadata = self._load_metadata(metadata_path)
        self.embeddings = self._load_embeddings(embeddings_path)
        self._validate_assets()
        self.model = SentenceTransformer(model_name)

    @staticmethod
    def _load_index(faiss_path: Path) -> faiss.IndexFlatIP:
        if not faiss_path.is_file():
            raise FileNotFoundError(f"FAISS index not found: {faiss_path}")
        try:
            return faiss.read_index(str(faiss_path))
        except Exception as exc:
            raise OSError(f"Failed to read FAISS index from {faiss_path}") from exc

    @staticmethod
    def _load_metadata(metadata_path: Path) -> list[dict]:
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        try:
            with metadata_path.open(encoding="utf-8") as f:
                metadata = json.load(f)
        except OSError as exc:
            raise OSError(f"Failed to read metadata from {metadata_path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid metadata JSON at {metadata_path}") from exc

        if not isinstance(metadata, list):
            raise ValueError(f"Metadata must be a JSON list at {metadata_path}")

        return [
            {**entry, "id": as_id(entry.get("id"), idx)}
            for idx, entry in enumerate(metadata)
        ]

    @staticmethod
    def _load_embeddings(embeddings_path: Path) -> np.ndarray:
        if not embeddings_path.is_file():
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
        try:
            embeddings = np.load(embeddings_path)
        except OSError as exc:
            raise OSError(f"Failed to read embeddings from {embeddings_path}") from exc
        return np.asarray(embeddings, dtype=np.float32)

    def _validate_assets(self) -> None:
        if self.embeddings.ndim != 2:
            raise ValueError(
                f"Embeddings must be a 2D matrix; got shape {self.embeddings.shape}."
            )
        if self.embeddings.shape[0] != len(self.metadata):
            raise ValueError(
                f"Embeddings rows ({self.embeddings.shape[0]}) do not match "
                f"metadata entries ({len(self.metadata)})."
            )
        if self.index.ntotal != len(self.metadata):
            raise ValueError(
                f"FAISS index size ({self.index.ntotal}) does not match "
                f"metadata entries ({len(self.metadata)})."
            )

    def _meta_at(self, idx: int) -> dict[str, str]:
        meta = self.metadata[int(idx)]
        return {
            "title": _as_str(meta.get("title")),
            "url": _as_str(meta.get("url")),
            "date": _as_str(meta.get("date")),
            "domain": _as_str(meta.get("domain")),
        }

    def _format_results(self, scores: np.ndarray, indices: np.ndarray) -> list[dict]:
        results: list[dict] = []
        for score, idx in zip(scores, indices):
            if idx < 0:
                continue
            result = self._meta_at(int(idx))
            result["score"] = float(score)
            results.append(result)
        return results

    def _search_vector(self, vector: np.ndarray, k: int) -> list[dict]:
        query = np.asarray(vector, dtype=np.float32).reshape(1, -1).copy()
        faiss.normalize_L2(query)
        scores, indices = self.index.search(query, k)
        return self._format_results(scores[0], indices[0])

    def search(self, query: str, k: int = 10) -> list[dict]:
        """Return the top-k articles most similar to a text query."""
        vector = np.asarray(
            self.model.encode(query, convert_to_numpy=True),
            dtype=np.float32,
        )
        return self._search_vector(vector, k)

    def _resolve_article_index(self, article_id: str | int) -> int:
        target = as_id(article_id)
        for idx, meta in enumerate(self.metadata):
            if as_id(meta.get("id"), idx) == target:
                return idx
        raise IndexError(
            f"article_id {article_id!r} not found among {len(self.metadata)} articles."
        )

    def find_similar(self, article_id: str | int, k: int = 10) -> list[dict]:
        """Return articles most similar to the article at `article_id`."""
        idx = self._resolve_article_index(article_id)
        if idx < 0 or idx >= len(self.metadata):
            raise IndexError(
                f"article_id {article_id!r} out of range for {len(self.metadata)} articles."
            )

        vector = self.embeddings[idx].copy()
        query = vector.reshape(1, -1)
        faiss.normalize_L2(query)
        scores, indices = self.index.search(query, k + 1)

        results: list[dict] = []
        for score, neighbor_idx in zip(scores[0], indices[0]):
            neighbor_idx = int(neighbor_idx)
            if neighbor_idx < 0 or neighbor_idx == idx:
                continue
            result = self._meta_at(neighbor_idx)
            result["score"] = float(score)
            results.append(result)
            if len(results) >= k:
                break

        return results


if __name__ == "__main__":
    retriever = NewsRetriever()
    query = "artificial intelligence chips"
    results = retriever.search(query, k=5)

    print(f"Top results for: {query!r}\n")
    for rank, result in enumerate(results, start=1):
        print(f"{rank}. [{result['score']:.4f}] {result['title']}")
        print(f"   {result['domain']} | {result['date']}")
        print(f"   {result['url']}\n")
