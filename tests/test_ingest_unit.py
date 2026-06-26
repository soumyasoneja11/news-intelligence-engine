"""Unit tests for ingest helpers."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingest import combine_text, strip_html


def test_strip_html_removes_tags() -> None:
    assert strip_html("<b>Hello</b> world") == "Hello world"


def test_combine_text_merges_title_and_description() -> None:
    text = combine_text("AI chips", "<p>Semiconductor demand rises.</p>")
    assert "AI chips" in text
    assert "Semiconductor demand rises." in text
    assert "<" not in text


if __name__ == "__main__":
    test_strip_html_removes_tags()
    test_combine_text_merges_title_and_description()
    print("ingest unit tests passed.")
