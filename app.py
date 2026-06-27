"""News Intelligence Engine — Streamlit entry point."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from html import escape
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.intelligence import IntelligenceEngine, load_intelligence_engine
from src.config import rebuild_allowed
from src.feeds import SOURCE_SLUGS, matches_source_filter
from src.index_status import (
    articles_csv_present,
    index_is_ready,
    index_status_message,
    missing_artifacts,
)
from src.paths import FAISS_PATH, FEED_STATS_PATH, SCHEDULER_LOG_PATH
from src.utils import as_id, parse_date

PIPELINE_SCRIPT = PROJECT_ROOT / "src" / "pipeline.py"
INGEST_SCRIPT = PROJECT_ROOT / "src" / "ingest.py"
STYLES_PATH = PROJECT_ROOT / "src" / "styles.css"
PIPELINE_OUTPUT_LIMIT = 8000


def _inject_app_styles() -> None:
    try:
        css = STYLES_PATH.read_text(encoding="utf-8")
    except OSError:
        return
    st.markdown(
        (
            '<link rel="preconnect" href="https://fonts.googleapis.com">'
            '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
            '<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@1,700&display=swap" '
            'rel="stylesheet">'
            f"<style>{css}</style>"
        ),
        unsafe_allow_html=True,
    )


def _app_password() -> str:
    try:
        secret = st.secrets.get("APP_PASSWORD", "")
        if secret:
            return str(secret)
    except (AttributeError, FileNotFoundError, KeyError):
        pass
    return os.environ.get("APP_PASSWORD", "")


def _check_password() -> bool:
    """Optional gate when APP_PASSWORD is set (env or Streamlit secrets)."""
    password = _app_password()
    if not password:
        return True
    if st.session_state.get("authenticated"):
        return True
    render_heading("News Intelligence Engine", level=1, calligraphic=True)
    st.caption("Authentication required")
    entered = st.text_input("Password", type="password", key="app_password_input")
    if st.button("Sign in", type="primary"):
        if entered == password:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


def _format_pipeline_output(output: str, limit: int = PIPELINE_OUTPUT_LIMIT) -> str:
    if len(output) <= limit:
        return output
    half = limit // 2
    return (
        output[:half]
        + "\n\n... [middle truncated] ...\n\n"
        + output[-half:]
    )


def _display_str(value: object, fallback: str = "") -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text if text else fallback


def _escape_text(value: object, fallback: str = "") -> str:
    return escape(_display_str(value, fallback))


def render_heading(
    text: str,
    level: int = 1,
    calligraphic: bool = False,
    *,
    link: str | None = None,
    sidebar: bool = False,
) -> None:
    """Render a styled HTML heading via st.markdown."""
    target = st.sidebar if sidebar else st
    safe_text = _escape_text(text)
    if link:
        href = escape(link, quote=True)
        safe_text = (
            f'<a href="{href}" style="color:inherit;text-decoration:none;">{safe_text}</a>'
        )

    if level == 1:
        class_attr = ' class="calligraphic"' if calligraphic else ""
        target.markdown(
            f'<h1{class_attr} style="font-size:2.4rem;color:#1a1a1a;margin:0;font-weight:700;">'
            f"{safe_text}</h1>"
            '<hr style="border:none;border-top:1px solid #e0e0e0;width:60px;'
            'margin:0.5rem 0 1rem 0;">',
            unsafe_allow_html=True,
        )
    elif level == 2:
        if sidebar:
            target.markdown(
                f'<h2 class="sidebar-section-title">{safe_text}</h2>',
                unsafe_allow_html=True,
            )
        else:
            target.markdown(
                '<h2 style="font-family:\'SF Pro Display\',-apple-system,BlinkMacSystemFont,sans-serif;'
                f'font-size:1.5rem;font-weight:600;color:#1a1a1a;margin:0 0 0.75rem 0;">{safe_text}</h2>',
                unsafe_allow_html=True,
            )
    else:
        target.markdown(
            '<h3 style="font-family:\'SF Pro Text\',-apple-system,BlinkMacSystemFont,sans-serif;'
            f'font-size:1.1rem;font-weight:500;color:#555;margin:0 0 0.35rem 0;">{safe_text}</h3>',
            unsafe_allow_html=True,
        )


def _safe_caption(domain: object, date: object) -> None:
    st.markdown(
        f'<p style="color:#9ca3af;font-size:0.85rem;margin:0;">'
        f"{_escape_text(domain, 'unknown')} · {_escape_text(date, '—')}</p>",
        unsafe_allow_html=True,
    )


st.set_page_config(
    page_title="News Intelligence Engine",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_engine() -> IntelligenceEngine:
    return load_intelligence_engine()


def _init_perf_state() -> None:
    if "search_cache" not in st.session_state:
        st.session_state.search_cache = {}
    if "similar_cache" not in st.session_state:
        st.session_state.similar_cache = {}


def _clear_perf_caches() -> None:
    st.session_state.search_cache = {}
    st.session_state.similar_cache = {}
    st.session_state.pop("cluster_scatter_fig", None)
    st.session_state.pop("clusters_chart_loaded", None)
    st.session_state.pop("duplicates_cache", None)
    st.session_state.pop("topics_cache", None)


def _index_age_text() -> str:
    if not FAISS_PATH.is_file():
        return "Index last built: not available"
    mtime = datetime.fromtimestamp(FAISS_PATH.stat().st_mtime)
    minutes = int((datetime.now() - mtime).total_seconds() // 60)
    if minutes <= 0:
        return "Index last built: just now"
    if minutes == 1:
        return "Index last built: 1 minute ago"
    return f"Index last built: {minutes} minutes ago"


def _last_fetch_timestamp() -> str | None:
    if not SCHEDULER_LOG_PATH.is_file():
        return None
    try:
        lines = SCHEDULER_LOG_PATH.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        if "[fetch]" in line:
            return line.split(" [fetch]", 1)[0].strip()
    return None


def _load_feed_stats_table() -> pd.DataFrame | None:
    if not FEED_STATS_PATH.is_file():
        return None
    try:
        with FEED_STATS_PATH.open(encoding="utf-8") as stats_file:
            payload = json.load(stats_file)
    except (OSError, json.JSONDecodeError):
        return None

    sources = payload.get("sources", [])
    if not isinstance(sources, list) or not sources:
        return None

    rows = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        rows.append(
            {
                "Source": source.get("name", "—"),
                "Articles": int(source.get("article_count", 0)),
                "Last article": source.get("last_article_time") or "—",
            }
        )
    return pd.DataFrame(rows) if rows else None


def _sync_source_filter_caches() -> None:
    current = tuple(st.session_state.get("source_filter", []))
    previous = st.session_state.get("_source_filter_cache_key")
    if previous != current:
        st.session_state.search_cache = {}
        st.session_state.pop("cluster_scatter_fig", None)
        st.session_state.pop("clusters_chart_loaded", None)
        st.session_state.pop("topics_cache", None)
        st.session_state.pop("duplicates_cache", None)
        st.session_state._source_filter_cache_key = current


def _filter_articles_by_source(
    articles: list[dict],
    selected_sources: list[str],
) -> list[dict]:
    if not selected_sources:
        return articles
    return [
        article
        for article in articles
        if matches_source_filter(article.get("domain", ""), selected_sources)
    ]


def _cached_search(
    engine: IntelligenceEngine,
    query: str,
    k: int = 10,
    sources: list[str] | None = None,
) -> tuple[list[dict], float]:
    cache_key = (query.strip().lower(), tuple(sorted(sources or [])))
    if cache_key in st.session_state.search_cache:
        return st.session_state.search_cache[cache_key]

    with st.spinner("Searching..."):
        start = time.perf_counter()
        try:
            results = engine.search(query.strip(), k=k, sources=sources)
        except FileNotFoundError as exc:
            st.error(f"Search unavailable: {exc}")
            return [], 0.0
        latency_ms = (time.perf_counter() - start) * 1000

    st.session_state.search_cache[cache_key] = (results, latency_ms)
    return results, latency_ms


def _cached_find_similar(
    engine: IntelligenceEngine,
    article_id: str,
    k: int = 10,
) -> tuple[list[dict], float]:
    cache_key = as_id(article_id)
    if cache_key in st.session_state.similar_cache:
        return st.session_state.similar_cache[cache_key]

    with st.spinner("Searching..."):
        start = time.perf_counter()
        try:
            results = engine.find_similar(cache_key, k=k)
        except (FileNotFoundError, IndexError) as exc:
            st.error(f"Similar-articles lookup failed: {exc}")
            return [], 0.0
        latency_ms = (time.perf_counter() - start) * 1000

    st.session_state.similar_cache[cache_key] = (results, latency_ms)
    return results, latency_ms


def _get_cluster_scatter_figure(engine: IntelligenceEngine, clusters: list[dict]):
    if st.session_state.get("cluster_scatter_fig") is not None:
        return st.session_state.cluster_scatter_fig

    scatter_df = _clusters_scatter_dataframe(engine, clusters)
    if scatter_df.empty:
        return None

    with st.spinner("Loading cluster map..."):
        fig = _build_cluster_scatter(scatter_df)

    st.session_state.cluster_scatter_fig = fig
    st.session_state.clusters_chart_loaded = True
    return fig


def _run_pipeline(*, skip_ingest: bool = False) -> tuple[bool, str]:
    """Run the build pipeline as a subprocess."""
    command = [sys.executable, str(PIPELINE_SCRIPT.resolve())]
    if skip_ingest:
        command.append("--skip-ingest")
    result = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    output = (result.stdout or "") + (result.stderr or "")
    return result.returncode == 0, output


def _run_rss_fetch_and_pipeline() -> tuple[bool, str]:
    """Run RSS ingest, then pipeline with --skip-ingest."""
    ingest_result = subprocess.run(
        [sys.executable, str(INGEST_SCRIPT.resolve()), "--source", "rss"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    output = (ingest_result.stdout or "") + (ingest_result.stderr or "")
    if ingest_result.returncode != 0:
        return False, output

    ok, pipeline_output = _run_pipeline(skip_ingest=True)
    output = output + pipeline_output
    return ok, output


def _safe_stats(engine: IntelligenceEngine) -> tuple[int, int, int]:
    try:
        with st.spinner("Loading stats..."):
            total_articles = len(engine.metadata)
            clusters = engine.get_clusters()
            duplicates = engine.get_duplicates()
        return total_articles, len(clusters), len(duplicates)
    except FileNotFoundError as exc:
        st.sidebar.error(f"Index not built yet: {exc}")
        return 0, 0, 0


def _sidebar_metric_cards_html(articles: int, clusters: int, duplicates: int) -> str:
    cards = (
        ("Articles", articles),
        ("Clusters", clusters),
        ("Duplicates", duplicates),
    )
    card_markup = []
    for label, value in cards:
        card_markup.append(
            f"""
            <div class="metric-card">
                <div class="metric-card__value" data-target="{int(value)}">0</div>
                <div class="metric-card__label">{escape(label)}</div>
            </div>
            """
        )
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@1,700&display=swap" rel="stylesheet">
        <style>
            @keyframes countUp {{
                from {{ opacity: 0; transform: translateY(8px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
            body {{
                margin: 0;
                padding: 0;
                background: transparent;
                font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
            }}
            .metric-cards-row {{
                display: flex;
                gap: 8px;
            }}
            .metric-card {{
                flex: 1;
                background: white;
                border: 1px solid #f0f0f0;
                border-radius: 14px;
                padding: 16px 20px;
                min-width: 0;
            }}
            .metric-card__value {{
                font-family: "Playfair Display", Georgia, serif;
                font-size: 2rem;
                font-style: italic;
                font-weight: 700;
                color: #1a1a1a;
                line-height: 1.1;
                animation: countUp 600ms ease forwards;
            }}
            .metric-card__label {{
                font-family: -apple-system, "SF Pro Text", sans-serif;
                font-size: 0.75rem;
                font-weight: 600;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: #999;
                margin-top: 0.35rem;
            }}
        </style>
    </head>
    <body>
        <div class="metric-cards-row">
            {"".join(card_markup)}
        </div>
        <script>
            document.querySelectorAll(".metric-card__value[data-target]").forEach((el) => {{
                const target = Number(el.dataset.target) || 0;
                const duration = 600;
                const start = performance.now();
                const tick = (now) => {{
                    const progress = Math.min((now - start) / duration, 1);
                    el.textContent = Math.round(target * progress);
                    if (progress < 1) requestAnimationFrame(tick);
                }};
                requestAnimationFrame(tick);
            }});
        </script>
    </body>
    </html>
    """


def _render_sidebar_metrics(articles: int, clusters: int, duplicates: int) -> None:
    components.html(
        _sidebar_metric_cards_html(articles, clusters, duplicates),
        height=108,
        scrolling=False,
    )


def _render_sidebar_trending_pill(rank: int, label: str, article_count: object, score: float) -> None:
    st.sidebar.markdown(
        (
            f'<div class="trending-pill-item">'
            f'<span class="trending-pill">{rank}. {_escape_text(label)}</span>'
            f'<span class="trending-pill-meta">'
            f"{_escape_text(article_count)} articles · score {score:.2f}"
            f"</span></div>"
        ),
        unsafe_allow_html=True,
    )


def _render_sidebar(
    engine: IntelligenceEngine | None,
    *,
    ready: bool,
    load_failed: bool = False,
) -> list[str]:
    st.sidebar.markdown("# 📰 News Intelligence")
    st.sidebar.caption("Semantic search · clustering · deduplication")

    index_age_slot = st.sidebar.empty()
    index_age_slot.caption(_index_age_text())

    render_heading("Live feed", level=2, sidebar=True)
    last_fetch = _last_fetch_timestamp()
    st.sidebar.caption(f"Last fetch: {last_fetch or 'not available'}")

    stats_df = _load_feed_stats_table()
    if stats_df is not None:
        st.sidebar.dataframe(stats_df, hide_index=True, use_container_width=True)
    else:
        st.sidebar.caption("No feed stats yet.")

    if rebuild_allowed():
        if st.sidebar.button("Fetch now", use_container_width=True):
            with st.spinner("Fetching RSS articles and updating index…"):
                ok, output = _run_rss_fetch_and_pipeline()
            if output.strip():
                st.sidebar.code(_format_pipeline_output(output), language="text")
            if ok:
                get_engine.clear()
                _clear_perf_caches()
                st.rerun()
            else:
                st.sidebar.error("Fetch or pipeline failed.")

    st.sidebar.divider()

    if "source_filter" not in st.session_state:
        st.session_state.source_filter = []
    selected_sources = st.sidebar.multiselect(
        "Filter by source",
        options=list(SOURCE_SLUGS.keys()),
        format_func=lambda slug: SOURCE_SLUGS[slug],
        key="source_filter",
        help="Limit search results and cluster views to selected publishers.",
    )
    _sync_source_filter_caches()

    if not ready or engine is None:
        if load_failed:
            st.sidebar.caption(
                "Index files are present but the engine failed to load. "
                "Check Streamlit Cloud logs for memory or model errors."
            )
        else:
            st.sidebar.error(index_status_message(missing_artifacts()))
            st.sidebar.caption(
                "Build locally with `python src/pipeline.py`, then push index artifacts."
            )
        if rebuild_allowed() and articles_csv_present() and not load_failed:
            st.sidebar.info("Rebuild is enabled but index files are missing.")
        return selected_sources

    total_articles, cluster_count, duplicate_count = _safe_stats(engine)
    _render_sidebar_metrics(total_articles, cluster_count, duplicate_count)

    st.sidebar.divider()

    if rebuild_allowed():
        if st.sidebar.button("Rebuild Index", type="primary", use_container_width=True):
            with st.sidebar.status("Running pipeline…", expanded=True) as status:
                ok, output = _run_pipeline()
                if output.strip():
                    st.code(_format_pipeline_output(output), language="text")
                if ok:
                    status.update(label="Rebuild complete", state="complete")
                    get_engine.clear()
                    _clear_perf_caches()
                    st.rerun()
                else:
                    status.update(label="Rebuild failed", state="error")
    else:
        st.sidebar.caption("Index rebuild disabled on this deployment.")

    st.sidebar.divider()
    render_heading("Trending Now", level=2, sidebar=True)

    try:
        with st.spinner("Loading stats..."):
            clusters = {str(c["cluster_id"]): c for c in engine.get_clusters()}
            trending = engine.get_trending(n=5)
        if not trending:
            st.sidebar.caption("No trending data yet.")
        for rank, item in enumerate(trending, start=1):
            cluster_id = str(item.get("cluster_id", ""))
            label = item.get("label", f"Cluster {cluster_id}")
            article_count = clusters.get(cluster_id, {}).get("article_count", "—")
            score = float(item.get("trending_score", 0.0))
            _render_sidebar_trending_pill(rank, label, article_count, score)
    except FileNotFoundError:
        st.sidebar.caption("Run **Rebuild Index** to populate trending data.")
    except Exception as exc:
        st.sidebar.caption(f"Trending unavailable: {exc}")

    return selected_sources


def _init_search_state() -> None:
    _init_perf_state()
    defaults = {
        "search_query": "",
        "search_query_input": "",
        "search_results": None,
        "search_latency_ms": None,
        "view_mode": "search",
        "similar_article_id": None,
        "similar_source_title": "",
        "similar_results": None,
        "similar_latency_ms": None,
        "pending_similar_id": None,
        "pending_similar_title": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _article_id_for_url(engine: IntelligenceEngine, url: str) -> str | None:
    for idx, meta in enumerate(engine.metadata):
        if meta.get("url") == url:
            return as_id(meta.get("id"), idx)
    return None


def _score_bar_color(score: float) -> str:
    if score > 0.7:
        return "#22c55e"
    if score > 0.5:
        return "#eab308"
    return "#ef4444"


def _score_bar_html(score: float) -> str:
    color = _score_bar_color(score)
    width = max(0.0, min(100.0, score * 100))
    return f"""
        <div style="margin:0.35rem 0 0.75rem 0;">
            <div style="display:flex;justify-content:space-between;font-size:0.8rem;color:#9ca3af;margin-bottom:0.25rem;">
                <span>Similarity</span><span>{score:.3f}</span>
            </div>
            <div style="background:#374151;border-radius:999px;height:8px;overflow:hidden;">
                <div style="width:{width:.1f}%;background:{color};height:8px;border-radius:999px;"></div>
            </div>
        </div>
        """


def _render_score_bar(score: float) -> None:
    st.markdown(_score_bar_html(score), unsafe_allow_html=True)


def _article_card_html(
    rank: int,
    title: str,
    url: str,
    domain: str,
    date: str,
    score: float,
    card_index: int,
) -> str:
    href = escape(url, quote=True)
    safe_title = _escape_text(title, "Untitled")
    return f"""
        <div class="article-card" style="--card-index: {card_index}">
            <h3><a href="{href}">{rank}. {safe_title}</a></h3>
            <p style="color:#9ca3af;font-size:0.85rem;margin:0;">
                {_escape_text(domain, "unknown")} · {_escape_text(date, "—")}
            </p>
            {_score_bar_html(score)}
        </div>
        """


def _render_result_card(
    engine: IntelligenceEngine,
    rank: int,
    result: dict,
    button_prefix: str,
    card_index: int,
) -> None:
    title = result.get("title", "Untitled")
    url = result.get("url", "")
    domain = result.get("domain", "unknown")
    date = result.get("date", "—")
    score = float(result.get("score", 0.0))
    article_id = _article_id_for_url(engine, url)

    card_html = _article_card_html(rank, title, url, domain, date, score, card_index)
    st.markdown(card_html, unsafe_allow_html=True)

    if article_id is not None:
        if st.button(
            "Find similar articles →",
            key=f"{button_prefix}_similar_{article_id}_{rank}",
        ):
            st.session_state.pending_similar_id = article_id
            st.session_state.pending_similar_title = title


def _render_search_tab(
    engine: IntelligenceEngine,
    *,
    ready: bool,
    selected_sources: list[str],
) -> None:
    _init_search_state()

    if not ready:
        render_heading("Semantic Search", level=2)
        st.warning("Search index is not available on this deployment.")
        st.info(
            "Run `python src/pipeline.py` locally, commit `index/` artifacts, "
            "and redeploy. See DEPLOY.md for details."
        )
        return

    pending_id = st.session_state.pending_similar_id
    if pending_id is not None:
        st.session_state.pending_similar_id = None
        similar, latency_ms = _cached_find_similar(engine, pending_id, k=10)
        st.session_state.view_mode = "similar"
        st.session_state.similar_article_id = pending_id
        st.session_state.similar_source_title = st.session_state.pending_similar_title
        st.session_state.similar_results = similar
        st.session_state.similar_latency_ms = latency_ms
        st.session_state.pending_similar_title = ""

    render_heading("Semantic Search", level=2)
    query = st.text_input(
        "Search articles",
        placeholder="e.g. climate change, AI chips…",
        key="search_query_input",
    )

    search_clicked = st.button("Search", type="primary", key="search_submit_btn")

    if search_clicked:
        if not query.strip():
            st.warning("Enter a search query.")
        else:
            results, latency_ms = _cached_search(
                engine,
                query.strip(),
                k=10,
                sources=selected_sources or None,
            )

            st.session_state.search_query = query.strip()
            st.session_state.search_results = results
            st.session_state.search_latency_ms = latency_ms
            st.session_state.view_mode = "search"
            st.session_state.similar_results = None
            st.session_state.similar_article_id = None
            st.session_state.similar_source_title = ""

    if st.session_state.view_mode == "similar" and st.session_state.similar_results is not None:
        if st.button("← Back to search results", key="back_to_search_results"):
            st.session_state.view_mode = "search"

        st.markdown(
            f"**Similar to:** {st.session_state.similar_source_title}"
        )
        results = st.session_state.similar_results
        latency_ms = st.session_state.similar_latency_ms
        st.caption(f"{len(results)} results · {latency_ms:.0f} ms")
    elif st.session_state.search_results is not None:
        results = st.session_state.search_results
        latency_ms = st.session_state.search_latency_ms
        st.caption(f"{len(results)} results · {latency_ms:.0f} ms")
    else:
        st.info("Enter a query and click **Search** to find articles.")
        return

    if not results:
        st.info("No results found.")
        return

    prefix = "similar" if st.session_state.view_mode == "similar" else "search"
    for i, result in enumerate(results):
        _render_result_card(engine, i + 1, result, button_prefix=prefix, card_index=i)


def _clusters_scatter_dataframe(
    engine: IntelligenceEngine,
    clusters: list[dict],
) -> pd.DataFrame:
    rows: list[dict] = []
    for cluster in clusters:
        cluster_label = cluster.get("label", "")
        for point in cluster.get("points", []):
            article_idx = int(point.get("article_idx", -1))
            domain = ""
            date = ""
            if 0 <= article_idx < len(engine.metadata):
                meta = engine.metadata[article_idx]
                domain = str(meta.get("domain", ""))
                date = str(meta.get("date", ""))
            rows.append(
                {
                    "umap_x": float(point.get("umap_x", 0.0)),
                    "umap_y": float(point.get("umap_y", 0.0)),
                    "cluster_id": str(point.get("cluster_id", cluster.get("cluster_id"))),
                    "cluster_label": cluster_label,
                    "title": str(point.get("title", "")),
                    "domain": domain,
                    "date": date,
                }
            )
    return pd.DataFrame(rows)


def _build_cluster_scatter(df: pd.DataFrame):
    fig = px.scatter(
        df,
        x="umap_x",
        y="umap_y",
        color="cluster_id",
        custom_data=["title", "domain", "date"],
        color_discrete_sequence=px.colors.qualitative.Bold,
        title="Article clusters (UMAP projection)",
        labels={
            "umap_x": "UMAP X",
            "umap_y": "UMAP Y",
            "cluster_id": "Cluster",
        },
    )
    fig.update_traces(
        marker=dict(size=5, opacity=0.75, line=dict(width=0), symbol="circle"),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Source: %{customdata[1]}<br>"
            "Date: %{customdata[2]}"
            "<extra></extra>"
        ),
    )
    fig.update_layout(
        font_family="-apple-system, SF Pro Display, Helvetica Neue, sans-serif",
        title_font_size=18,
        title_font_family="Playfair Display",
        plot_bgcolor="#fafafa",
        paper_bgcolor="white",
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=True,
        legend=dict(
            title_text="Cluster ID",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
        ),
        height=520,
    )
    return fig


def _cluster_article_card_html(article: dict) -> str:
    title = article.get("title", "Untitled")
    url = article.get("url", "")
    domain = article.get("domain", "unknown")
    date = article.get("date", "—")
    href = escape(url, quote=True)
    return f"""
        <div class="cluster-article-card">
            <a href="{href}" style="color:#1a1a1a;text-decoration:none;font-weight:500;">
                {_escape_text(title, "Untitled")}
            </a>
            <p style="color:#9ca3af;font-size:0.85rem;margin:0.35rem 0 0;">
                {_escape_text(domain, "unknown")} · {_escape_text(date, "—")}
            </p>
        </div>
        """


def _render_cluster_articles_morph(articles: list[dict], total: int) -> None:
    cards = "".join(_cluster_article_card_html(article) for article in articles)
    st.markdown(
        f"""
        <div class="cluster-article-list">
            {cards}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"Showing {len(articles)} of {total} articles")


def _recent_cluster_articles(articles: list[dict], limit: int = 10) -> list[dict]:
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
    return ordered[:limit]


def _render_clusters_tab(engine: IntelligenceEngine, selected_sources: list[str]) -> None:
    render_heading("Topic Clusters", level=2)
    try:
        with st.spinner("Loading clusters..."):
            clusters = engine.get_clusters(sources=selected_sources or None)
    except FileNotFoundError as exc:
        st.warning(f"Cluster data unavailable: {exc}")
        return

    if not clusters:
        st.info("No clusters found.")
        return

    fig = _get_cluster_scatter_figure(engine, clusters)
    if fig is None:
        st.warning("No UMAP coordinates available for plotting.")
    else:
        st.plotly_chart(fig, use_container_width=True)

    cluster_options = {
        f"{cluster['label']} ({cluster['article_count']} articles)": cluster
        for cluster in clusters
    }
    selected_label = st.selectbox("Select cluster", list(cluster_options.keys()))
    selected_cluster = cluster_options[selected_label]
    cluster_id = selected_cluster["cluster_id"]
    keywords = list(selected_cluster.get("keywords", []))

    st.markdown(f"**Cluster {cluster_id}:** {selected_cluster['label']}")

    if keywords:
        st.markdown("**Top keywords**")
        badge_cols = st.columns(min(len(keywords), 5))
        for idx, keyword in enumerate(keywords):
            with badge_cols[idx % len(badge_cols)]:
                st.badge(keyword)
    else:
        st.caption("No keywords available for this cluster.")

    with st.spinner("Loading cluster articles..."):
        articles = engine.get_cluster_articles(cluster_id)
    articles = _filter_articles_by_source(articles, selected_sources)
    recent_articles = _recent_cluster_articles(articles, limit=10)

    st.markdown("**Most recent articles**")
    if not recent_articles and selected_sources:
        st.info("No articles from the selected sources in this cluster.")
    else:
        _render_cluster_articles_morph(recent_articles, len(articles))


@st.fragment
def _render_clusters_tab_fragment(
    engine: IntelligenceEngine,
    selected_sources: list[str],
) -> None:
    """Fragment reruns only when the Clusters tab is interacted with."""
    _render_clusters_tab(engine, selected_sources)


def _metadata_by_id(engine: IntelligenceEngine) -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    for idx, meta in enumerate(engine.metadata):
        article_id = as_id(meta.get("id"), idx)
        lookup[article_id] = meta
    return lookup


def _enrich_duplicate_article(article: dict, meta_lookup: dict[str, dict]) -> dict:
    article_id = as_id(article.get("id"))
    meta = meta_lookup.get(article_id, {})
    return {
        "id": article_id,
        "title": _display_str(article.get("title") or meta.get("title"), "Untitled"),
        "url": _display_str(article.get("url") or meta.get("url"), ""),
        "domain": _display_str(article.get("domain") or meta.get("domain"), "unknown"),
        "date": _display_str(article.get("date") or meta.get("date"), "—"),
    }


def _similarity_highlight_color(score: float) -> str:
    if score >= 0.95:
        return "#22c55e"
    if score >= 0.92:
        return "#eab308"
    return "#f97316"


def _render_duplicate_article(article: dict) -> None:
    title = article.get("title", "Untitled")
    url = article.get("url", "")
    domain = article.get("domain", "unknown")
    date = article.get("date", "—")
    st.markdown(f"**[{_escape_text(title, 'Untitled')}]({url})**")
    _safe_caption(domain, date)


def _duplicate_pairs_to_csv(pairs: list[dict]) -> str:
    rows: list[dict] = []
    for pair in pairs:
        article_a = pair["article_a"]
        article_b = pair["article_b"]
        rows.append(
            {
                "similarity": pair["similarity"],
                "article_a_id": article_a["id"],
                "article_a_title": article_a["title"],
                "article_a_url": article_a["url"],
                "article_a_domain": article_a["domain"],
                "article_a_date": article_a["date"],
                "article_b_id": article_b["id"],
                "article_b_title": article_b["title"],
                "article_b_url": article_b["url"],
                "article_b_domain": article_b["domain"],
                "article_b_date": article_b["date"],
            }
        )
    return pd.DataFrame(rows).to_csv(index=False)


def _render_duplicates_tab(engine: IntelligenceEngine) -> None:
    render_heading("Near-Duplicate Articles", level=2)
    try:
        if "duplicates_cache" not in st.session_state:
            with st.spinner("Loading duplicates..."):
                st.session_state.duplicates_cache = engine.get_duplicates()
        duplicates = st.session_state.duplicates_cache
        total_articles = len(engine.metadata)
    except FileNotFoundError as exc:
        st.warning(f"Duplicate data unavailable: {exc}")
        return

    if not duplicates:
        st.info("No near-duplicate pairs detected.")
        return

    meta_lookup = _metadata_by_id(engine)
    enriched_pairs: list[dict] = []
    for pair in duplicates:
        enriched_pairs.append(
            {
                "similarity": float(pair.get("similarity", 0.0)),
                "article_a": _enrich_duplicate_article(pair["article_a"], meta_lookup),
                "article_b": _enrich_duplicate_article(pair["article_b"], meta_lookup),
            }
        )

    total_pairs = len(enriched_pairs)
    duplicate_pct = (total_pairs / total_articles * 100) if total_articles else 0.0
    st.metric(
        "Duplicate pairs",
        f"{total_pairs} / {total_articles}",
        f"{duplicate_pct:.1f}% of indexed articles",
    )

    threshold = st.slider(
        "Similarity threshold",
        min_value=0.85,
        max_value=1.0,
        value=0.92,
        step=0.01,
        help="Only show pairs at or above this cosine similarity score.",
    )

    filtered_pairs = [pair for pair in enriched_pairs if pair["similarity"] >= threshold]
    st.caption(f"Showing {len(filtered_pairs)} of {total_pairs} pairs")

    if filtered_pairs:
        csv_data = _duplicate_pairs_to_csv(filtered_pairs)
        st.download_button(
            label="Export CSV",
            data=csv_data,
            file_name="duplicate_pairs.csv",
            mime="text/csv",
            use_container_width=False,
        )

    if not filtered_pairs:
        st.info("No pairs match the selected similarity threshold.")
        return

    for pair in filtered_pairs:
        score = pair["similarity"]
        color = _similarity_highlight_color(score)

        with st.container(border=True):
            left_col, score_col, right_col = st.columns([5, 2, 5])

            with left_col:
                st.caption("Article A")
                _render_duplicate_article(pair["article_a"])

            with score_col:
                st.markdown(
                    f"""
                    <div style="text-align:center;padding-top:1.5rem;">
                        <div style="font-size:0.75rem;color:#9ca3af;margin-bottom:0.25rem;">
                            Similarity
                        </div>
                        <div style="font-size:1.35rem;font-weight:700;color:{color};">
                            {score:.3f}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with right_col:
                st.caption("Article B")
                _render_duplicate_article(pair["article_b"])


def _pill_font_size(score: float, min_score: float, max_score: float) -> int:
    min_px, max_px = 13, 30
    if max_score <= min_score:
        return (min_px + max_px) // 2
    ratio = (score - min_score) / (max_score - min_score)
    return int(min_px + ratio * (max_px - min_px))


def _render_topic_cloud_pill(label: str, score: float, font_size: int) -> None:
    safe_label = _escape_text(label, "Cluster")
    st.markdown(
        f"""
        <div style="margin-bottom:0.75rem;">
            <span style="display:inline-block;background:#262730;border:1px solid #4F8BF9;
                border-radius:999px;padding:0.4rem 1rem;font-size:{font_size}px;
                font-weight:600;color:#FAFAFA;line-height:1.2;">
                {safe_label}
            </span>
            <div style="color:#9ca3af;font-size:0.75rem;margin-top:0.35rem;">
                trending score {score:.3f}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_topics_trends_tab(engine: IntelligenceEngine, selected_sources: list[str]) -> None:
    render_heading("Topics & Trends", level=2)

    try:
        if "topics_cache" not in st.session_state:
            with st.spinner("Loading topics and trends..."):
                clusters = engine.get_clusters(sources=selected_sources or None)
                trending = engine.get_trending(n=len(clusters) or 50)
                if selected_sources:
                    cluster_ids = {str(c["cluster_id"]) for c in clusters}
                    trending = [
                        item
                        for item in trending
                        if str(item.get("cluster_id", "")) in cluster_ids
                    ]
                st.session_state.topics_cache = (clusters, trending)
        clusters, trending = st.session_state.topics_cache
    except FileNotFoundError as exc:
        st.warning(f"Topics and trends data unavailable: {exc}")
        return

    if not clusters:
        st.info("No cluster data available.")
        return

    st.markdown("**Top clusters by article count**")
    count_df = (
        pd.DataFrame(
            [
                {
                    "label": cluster["label"],
                    "article_count": int(cluster["article_count"]),
                    "cluster_id": cluster["cluster_id"],
                }
                for cluster in clusters
            ]
        )
        .sort_values("article_count", ascending=True)
        .tail(15)
    )

    count_fig = px.bar(
        count_df,
        x="article_count",
        y="label",
        orientation="h",
        color="article_count",
        color_continuous_scale="Blues",
        title="Top 15 clusters by article count",
        labels={"article_count": "Articles", "label": "Cluster"},
    )
    count_fig.update_layout(
        height=520,
        showlegend=False,
        yaxis={"categoryorder": "total ascending"},
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(count_fig, use_container_width=True)

    st.markdown("**Top clusters by trending score**")
    if not trending:
        st.caption("No trending data available.")
    else:
        trend_df = (
            pd.DataFrame(trending)
            .sort_values("trending_score", ascending=True)
            .tail(10)
        )
        trend_fig = px.bar(
            trend_df,
            x="trending_score",
            y="label",
            orientation="h",
            color="count_24h",
            color_continuous_scale="YlOrRd",
            title="Top 10 clusters by trending score",
            labels={
                "trending_score": "Trending score",
                "label": "Cluster",
                "count_24h": "Articles (24h)",
            },
        )
        trend_fig.update_layout(
            height=460,
            yaxis={"categoryorder": "total ascending"},
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(trend_fig, use_container_width=True)

    st.divider()
    st.markdown("**Trending topic cloud**")

    if not trending:
        st.info("No trending clusters to display.")
        return

    scores = [float(item.get("trending_score", 0.0)) for item in trending]
    min_score = min(scores)
    max_score = max(scores)

    pill_columns = st.columns(3)
    for idx, item in enumerate(trending):
        label = str(item.get("label", f"Cluster {item.get('cluster_id', '')}"))
        score = float(item.get("trending_score", 0.0))
        font_size = _pill_font_size(score, min_score, max_score)
        top_articles = list(item.get("top_articles", []))[:3]

        with pill_columns[idx % 3]:
            _render_topic_cloud_pill(label, score, font_size)
            for article in top_articles:
                title = article.get("title", "Untitled")
                url = article.get("url", "")
                date = article.get("date", "—")
                st.markdown(
                    f'<p style="margin:0 0 0.5rem 0;font-size:0.85rem;">'
                    f'<a href="{_escape_text(url)}" target="_blank" style="color:#93c5fd;text-decoration:none;">'
                    f"{_escape_text(title, 'Untitled')}</a><br>"
                    f"<span style=\"color:#9ca3af;\">{_escape_text(date, '—')}</span></p>",
                    unsafe_allow_html=True,
                )


@st.fragment
def _render_duplicates_tab_fragment(engine: IntelligenceEngine) -> None:
    _render_duplicates_tab(engine)


@st.fragment
def _render_topics_trends_tab_fragment(
    engine: IntelligenceEngine,
    selected_sources: list[str],
) -> None:
    _render_topics_trends_tab(engine, selected_sources)


def main() -> None:
    _inject_app_styles()
    _init_perf_state()
    if not _check_password():
        return

    ready = index_is_ready()
    engine = None
    load_failed = False
    if ready:
        try:
            engine = get_engine()
        except Exception as exc:
            st.error(f"Failed to load the intelligence engine: {exc}")
            load_failed = True
            ready = False

    selected_sources = _render_sidebar(engine, ready=ready, load_failed=load_failed)

    render_heading("News Intelligence Engine", level=1, calligraphic=True)
    st.caption("Explore semantic search results, topic clusters, and duplicate coverage.")

    if not ready:
        if not load_failed:
            st.error(index_status_message(missing_artifacts()))
        return

    search_tab, clusters_tab, duplicates_tab, topics_tab = st.tabs(
        ["Search", "Clusters", "Duplicates", "Topics & Trends"]
    )

    with search_tab:
        _render_search_tab(engine, ready=ready, selected_sources=selected_sources)

    with clusters_tab:
        _render_clusters_tab_fragment(engine, selected_sources)

    with duplicates_tab:
        _render_duplicates_tab_fragment(engine)

    with topics_tab:
        _render_topics_trends_tab_fragment(engine, selected_sources)


if __name__ == "__main__":
    main()
