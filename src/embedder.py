"""Encode article text and build a FAISS similarity index."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.paths import (
    ARTICLES_PATH,
    EMBEDDINGS_PATH,
    FAISS_PATH,
    INDEX_DIR,
    METADATA_PATH,
)
from src.utils import as_id

MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 64
REQUIRED_COLUMNS = frozenset({"text", "title", "url", "date", "domain"})


def _as_str(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def load_articles(path: Path = ARTICLES_PATH) -> pd.DataFrame:
    """Load cleaned articles from CSV."""
    if not path.is_file():
        raise FileNotFoundError(f"Articles file not found: {path}")

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Articles file is empty: {path}") from exc
    except OSError as exc:
        raise OSError(f"Failed to read articles from {path}") from exc

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Articles CSV missing required columns: {sorted(missing)}")

    return df


def encode_texts(
    texts: list[str],
    model_name: str = MODEL_NAME,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """Encode texts into a float32 embedding matrix."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build an inner-product index on L2-normalized vectors."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    vectors = embeddings.copy()
    faiss.normalize_L2(vectors)
    index.add(vectors)
    return index


def add_to_faiss_index(index: faiss.IndexFlatIP, embeddings: np.ndarray) -> None:
    """Append L2-normalized vectors to an existing FAISS index."""
    vectors = np.asarray(embeddings, dtype=np.float32).copy()
    faiss.normalize_L2(vectors)
    index.add(vectors)


def build_metadata(df: pd.DataFrame, start_id: int = 0) -> list[dict]:
    """Extract searchable article metadata aligned with embedding rows."""
    metadata = []
    for offset, (_, row) in enumerate(df.iterrows()):
        metadata.append(
            {
                "id": as_id(start_id + offset),
                "title": _as_str(row["title"]),
                "url": _as_str(row["url"]),
                "date": _as_str(row["date"]),
                "domain": _as_str(row["domain"]),
            }
        )
    return metadata


def _load_existing_metadata(metadata_path: Path) -> list[dict]:
    try:
        with metadata_path.open(encoding="utf-8") as f:
            metadata = json.load(f)
    except OSError as exc:
        raise OSError(f"Failed to read metadata from {metadata_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid metadata JSON at {metadata_path}") from exc

    if not isinstance(metadata, list):
        raise ValueError(f"Metadata must be a JSON list at {metadata_path}")

    return metadata


def _load_existing_embeddings(embeddings_path: Path) -> np.ndarray:
    if not embeddings_path.is_file():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

    try:
        embeddings = np.load(embeddings_path)
    except OSError as exc:
        raise OSError(f"Failed to read embeddings from {embeddings_path}") from exc

    return np.asarray(embeddings, dtype=np.float32)


def _load_existing_index(faiss_path: Path) -> faiss.IndexFlatIP:
    if not faiss_path.is_file():
        raise FileNotFoundError(f"FAISS index not found: {faiss_path}")

    try:
        return faiss.read_index(str(faiss_path))
    except Exception as exc:
        raise OSError(f"Failed to read FAISS index from {faiss_path}") from exc


def _save_embeddings(embeddings_path: Path, embeddings: np.ndarray) -> None:
    try:
        np.save(embeddings_path, np.asarray(embeddings, dtype=np.float32))
    except OSError as exc:
        raise OSError(f"Failed to save embeddings to {embeddings_path}") from exc


def _save_index(faiss_path: Path, index: faiss.IndexFlatIP) -> None:
    try:
        faiss.write_index(index, str(faiss_path))
    except Exception as exc:
        raise OSError(f"Failed to save FAISS index to {faiss_path}") from exc


def _save_metadata(metadata_path: Path, metadata: list[dict]) -> None:
    try:
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    except OSError as exc:
        raise OSError(f"Failed to save metadata to {metadata_path}") from exc


def _validate_index_consistency(
    embeddings: np.ndarray,
    index: faiss.IndexFlatIP,
    metadata: list[dict],
) -> None:
    if embeddings.shape[0] != len(metadata):
        raise ValueError(
            f"Embeddings rows ({embeddings.shape[0]}) do not match "
            f"metadata entries ({len(metadata)})."
        )
    if index.ntotal != len(metadata):
        raise ValueError(
            f"FAISS index size ({index.ntotal}) does not match "
            f"metadata entries ({len(metadata)})."
        )


def _known_urls(metadata: list[dict]) -> set[str]:
    return {_as_str(entry["url"]) for entry in metadata if entry.get("url")}


def _filter_new_articles(df: pd.DataFrame, known_urls: set[str]) -> pd.DataFrame:
    urls = df["url"].map(_as_str)
    return df[~urls.isin(known_urls)].copy()


def _can_incremental(
    faiss_path: Path,
    metadata_path: Path,
    embeddings_path: Path,
) -> bool:
    return faiss_path.is_file() and metadata_path.is_file() and embeddings_path.is_file()


def _texts_from_df(df: pd.DataFrame) -> list[str]:
    return df["text"].fillna("").map(_as_str).tolist()


def _embed_incremental(
    df: pd.DataFrame,
    embeddings_path: Path,
    faiss_path: Path,
    metadata_path: Path,
) -> tuple[np.ndarray, faiss.IndexFlatIP, list[dict]]:
    index = _load_existing_index(faiss_path)
    metadata = _load_existing_metadata(metadata_path)
    embeddings = _load_existing_embeddings(embeddings_path)
    _validate_index_consistency(embeddings, index, metadata)

    known = _known_urls(metadata)
    new_df = _filter_new_articles(df, known)

    if new_df.empty:
        print(f"No new articles to embed ({len(metadata)} already indexed).")
        return embeddings, index, metadata

    print(f"Incremental update: {len(new_df)} new articles ({len(metadata)} already indexed).")
    print(f"Encoding text with {MODEL_NAME} (batch_size={BATCH_SIZE})...")
    new_embeddings = encode_texts(_texts_from_df(new_df))
    print(f"New embedding matrix shape: {new_embeddings.shape}")

    embeddings = np.vstack([embeddings, new_embeddings])
    _save_embeddings(embeddings_path, embeddings)
    print(f"Saved embeddings to {embeddings_path} (shape {embeddings.shape})")

    print("Adding new vectors to FAISS index...")
    add_to_faiss_index(index, new_embeddings)
    _save_index(faiss_path, index)
    print(f"Saved FAISS index to {faiss_path} (size {index.ntotal})")

    metadata.extend(build_metadata(new_df, start_id=len(metadata)))
    _validate_index_consistency(embeddings, index, metadata)
    _save_metadata(metadata_path, metadata)
    print(f"Saved metadata for {len(metadata)} articles to {metadata_path}")

    return embeddings, index, metadata


def _embed_full(
    df: pd.DataFrame,
    embeddings_path: Path,
    faiss_path: Path,
    metadata_path: Path,
) -> tuple[np.ndarray, faiss.IndexFlatIP, list[dict]]:
    print(f"Encoding text with {MODEL_NAME} (batch_size={BATCH_SIZE})...")
    embeddings = encode_texts(_texts_from_df(df))
    print(f"Embedding matrix shape: {embeddings.shape}")

    _save_embeddings(embeddings_path, embeddings)
    print(f"Saved embeddings to {embeddings_path}")

    print("Building FAISS IndexFlatIP...")
    index = build_faiss_index(embeddings)
    _save_index(faiss_path, index)
    print(f"Saved FAISS index to {faiss_path}")

    metadata = build_metadata(df)
    _validate_index_consistency(embeddings, index, metadata)
    _save_metadata(metadata_path, metadata)
    print(f"Saved metadata for {len(metadata)} articles to {metadata_path}")

    return embeddings, index, metadata


def embed(
    articles_path: Path = ARTICLES_PATH,
    embeddings_path: Path = EMBEDDINGS_PATH,
    faiss_path: Path = FAISS_PATH,
    metadata_path: Path = METADATA_PATH,
) -> tuple[np.ndarray, faiss.IndexFlatIP, list[dict]]:
    """Load articles, encode, persist embeddings, FAISS index, and metadata."""
    try:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(f"Failed to create index directory {INDEX_DIR}") from exc

    print(f"Loading articles from {articles_path}...")
    df = load_articles(articles_path)
    print(f"Loaded {len(df)} articles.")

    if _can_incremental(faiss_path, metadata_path, embeddings_path):
        return _embed_incremental(df, embeddings_path, faiss_path, metadata_path)

    return _embed_full(df, embeddings_path, faiss_path, metadata_path)


if __name__ == "__main__":
    embed()
