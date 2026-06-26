"""Unit tests for index artifact checks."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.index_status import index_is_ready, missing_artifacts
from src.paths import FAISS_PATH, METADATA_PATH


def test_index_is_ready_with_local_artifacts() -> None:
    if FAISS_PATH.is_file() and METADATA_PATH.is_file():
        assert index_is_ready(minimal=True) is True
    else:
        assert index_is_ready(minimal=True) is False


def test_missing_artifacts_lists_paths() -> None:
    missing = missing_artifacts()
    assert isinstance(missing, list)
    for path in missing:
        assert not path.is_file()


if __name__ == "__main__":
    test_index_is_ready_with_local_artifacts()
    test_missing_artifacts_lists_paths()
    print("index_status unit tests passed.")
