"""Smoke checks for the Streamlit app module (no Streamlit runtime)."""

from __future__ import annotations

import ast
from pathlib import Path


APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


def _module_function_names(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    ]


def test_app_has_distinct_render_pill_helpers() -> None:
    names = _module_function_names(APP_PATH)
    assert "_render_sidebar_trending_pill" in names
    assert "_render_topic_cloud_pill" in names
    assert "_render_trending_pill" not in names


def test_app_has_no_duplicate_top_level_functions() -> None:
    names = _module_function_names(APP_PATH)
    duplicates = sorted({name for name in names if names.count(name) > 1})
    assert duplicates == []
