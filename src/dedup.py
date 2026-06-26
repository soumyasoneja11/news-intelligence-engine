"""Find near-duplicate articles using FAISS cosine similarity."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import faiss
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.paths import FAISS_PATH, INDEX_DIR, METADATA_PATH
from src.utils import as_id

DUPLICATE_PAIRS_PATH = INDEX_DIR / "duplicate_pairs.json"

K_NEIGHBORS = 10
SIMILARITY_THRESHOLD = 0.92


def _load_index(path: Path = FAISS_PATH) -> faiss.Index:
    if not path.is_file():
        raise FileNotFoundError(f"FAISS index not found: {path}")
    try:
        return faiss.read_index(str(path))
    except Exception as exc:
        raise OSError(f"Failed to read FAISS index from {path}") from exc


def _load_metadata(path: Path = METADATA_PATH) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    try:
        with path.open(encoding="utf-8") as f:
            metadata = json.load(f)
    except OSError as exc:
        raise OSError(f"Failed to read metadata from {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid metadata JSON at {path}") from exc

    if not isinstance(metadata, list):
        raise ValueError(f"Metadata must be a JSON list at {path}")

    return metadata


def _save_json(path: Path, payload: object) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    except OSError as exc:
        raise OSError(f"Failed to write JSON to {path}") from exc


def _article_brief(meta: dict, index: int) -> dict:
    return {
        "id": as_id(meta.get("id"), index),
        "title": str(meta.get("title", "")),
        "url": str(meta.get("url", "")),
    }


def find_duplicate_pairs(
    k: int = K_NEIGHBORS,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    faiss_path: Path = FAISS_PATH,
    metadata_path: Path = METADATA_PATH,
    output_path: Path = DUPLICATE_PAIRS_PATH,
) -> list[dict]:
    """Scan the index for near-duplicate pairs and write results to JSON."""
    index = _load_index(faiss_path)
    metadata = _load_metadata(metadata_path)

    if index.ntotal != len(metadata):
        raise ValueError(
            f"FAISS index size ({index.ntotal}) does not match "
            f"metadata entries ({len(metadata)})."
        )

    if not isinstance(index, (faiss.IndexFlatIP, faiss.IndexHNSWFlat)):
        raise ValueError(
            "Expected faiss.IndexFlatIP or IndexHNSWFlat for cosine similarity; "
            f"got {type(index).__name__}."
        )

    # IndexFlatIP on L2-normalized vectors returns cosine similarity, not L2 distance.
    d = index.d
    seen_pairs: set[tuple[int, int]] = set()
    duplicates: list[dict] = []

    for i in range(len(metadata)):
        v = np.empty((d,), dtype=np.float32)
        index.reconstruct(i, v)

        q = v.reshape(1, -1).copy()
        faiss.normalize_L2(q)

        scores, indices = index.search(q, k + 1)  # includes self
        for score, j in zip(scores[0], indices[0]):
            j = int(j)
            if j < 0 or j == i:
                continue

            sim = float(score)
            if sim <= similarity_threshold:
                continue

            url_i = str(metadata[i].get("url", ""))
            url_j = str(metadata[j].get("url", ""))
            if not url_i or not url_j or url_i == url_j:
                continue

            a, b = (i, j) if i < j else (j, i)
            if (a, b) in seen_pairs:
                continue
            seen_pairs.add((a, b))

            duplicates.append(
                {
                    "article_a": _article_brief(metadata[a], a),
                    "article_b": _article_brief(metadata[b], b),
                    "similarity": sim,
                }
            )

    # Sort by descending similarity for easier inspection.
    duplicates.sort(key=lambda x: x["similarity"], reverse=True)
    _save_json(output_path, duplicates)

    return duplicates


if __name__ == "__main__":
    start = time.perf_counter()
    pairs = find_duplicate_pairs()
    elapsed = time.perf_counter() - start

    print(f"Near-duplicate pairs found: {len(pairs)}")
    print(f"Wrote: {DUPLICATE_PAIRS_PATH}")
    print(f"Elapsed: {elapsed:.1f}s")

    if pairs:
        print("\nExamples:")
        for ex in pairs[:5]:
            a = ex["article_a"]
            b = ex["article_b"]
            print(f"- {ex['similarity']:.4f}")
            print(f"  A: {a['id']} | {a['title']}")
            print(f"     {a['url']}")
            print(f"  B: {b['id']} | {b['title']}")
            print(f"     {b['url']}")
