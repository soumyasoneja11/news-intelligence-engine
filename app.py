"""News Intelligence Engine — Streamlit entry point."""

from __future__ import annotations

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

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.intelligence import IntelligenceEngine, load_intelligence_engine
from src.config import rebuild_allowed
from src.index_status import (
    articles_csv_present,
    index_is_ready,
    index_status_message,
    missing_artifacts,
)
from src.paths import FAISS_PATH
from src.utils import as_id, parse_date

PIPELINE_SCRIPT = PROJECT_ROOT / "src" / "pipeline.py"
PIPELINE_OUTPUT_LIMIT = 8000


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
    st.title("News Intelligence Engine")
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


def _cached_search(engine: IntelligenceEngine, query: str, k: int = 10) -> tuple[list[dict], float]:
    cache_key = query.strip().lower()
    if cache_key in st.session_state.search_cache:
        return st.session_state.search_cache[cache_key]

    with st.spinner("Searching..."):
        start = time.perf_counter()
        try:
            results = engine.search(query.strip(), k=k)
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
        fig.update_traces(marker=dict(size=8, opacity=0.8))
        fig.update_layout(
            height=520,
            legend_title_text="Cluster ID",
            margin=dict(l=10, r=10, t=50, b=10),
        )

    st.session_state.cluster_scatter_fig = fig
    st.session_state.clusters_chart_loaded = True
    return fig


def _run_pipeline() -> tuple[bool, str]:
    """Run the full build pipeline as a subprocess."""
    result = subprocess.run(
        [sys.executable, str(PIPELINE_SCRIPT.resolve())],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    output = (result.stdout or "") + (result.stderr or "")
    return result.returncode == 0, output


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


def _render_sidebar(engine: IntelligenceEngine | None, *, ready: bool) -> None:
    st.sidebar.markdown("# 📰 News Intelligence")
    st.sidebar.caption("Semantic search · clustering · deduplication")

    index_age_slot = st.sidebar.empty()
    index_age_slot.caption(_index_age_text())

    if not ready or engine is None:
        st.sidebar.error(index_status_message(missing_artifacts()))
        st.sidebar.caption(
            "Build locally with `python src/pipeline.py`, then push index artifacts."
        )
        if rebuild_allowed() and articles_csv_present():
            st.sidebar.info("Rebuild is enabled but index files are missing.")
        return

    total_articles, cluster_count, duplicate_count = _safe_stats(engine)
    col1, col2, col3 = st.sidebar.columns(3)
    col1.metric("Articles", total_articles)
    col2.metric("Clusters", cluster_count)
    col3.metric("Duplicates", duplicate_count)

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
    st.sidebar.subheader("Trending Now")

    try:
        with st.spinner("Loading stats..."):
            clusters = {c["cluster_id"]: c for c in engine.get_clusters()}
            trending = engine.get_trending(n=5)
        if not trending:
            st.sidebar.caption("No trending data yet.")
        for rank, item in enumerate(trending, start=1):
            cluster_id = str(item.get("cluster_id", ""))
            label = item.get("label", f"Cluster {cluster_id}")
            article_count = clusters.get(cluster_id, {}).get("article_count", "—")
            score = item.get("trending_score", 0.0)
            st.sidebar.markdown(f"**{rank}. {label}**")
            st.sidebar.caption(f"{article_count} articles · score {score}")
    except FileNotFoundError:
        st.sidebar.caption("Run **Rebuild Index** to populate trending data.")


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


def _render_score_bar(score: float) -> None:
    color = _score_bar_color(score)
    width = max(0.0, min(100.0, score * 100))
    st.markdown(
        f"""
        <div style="margin:0.35rem 0 0.75rem 0;">
            <div style="display:flex;justify-content:space-between;font-size:0.8rem;color:#9ca3af;margin-bottom:0.25rem;">
                <span>Similarity</span><span>{score:.3f}</span>
            </div>
            <div style="background:#374151;border-radius:999px;height:8px;overflow:hidden;">
                <div style="width:{width:.1f}%;background:{color};height:8px;border-radius:999px;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_result_card(
    engine: IntelligenceEngine,
    rank: int,
    result: dict,
    button_prefix: str,
) -> None:
    title = result.get("title", "Untitled")
    url = result.get("url", "")
    domain = result.get("domain", "unknown")
    date = result.get("date", "—")
    score = float(result.get("score", 0.0))
    article_id = _article_id_for_url(engine, url)

    with st.container(border=True):
        st.markdown(f"**{rank}.** [{_escape_text(title, 'Untitled')}]({url})")
        _safe_caption(domain, date)
        _render_score_bar(score)

        if article_id is not None:
            if st.button(
                "Find similar articles →",
                key=f"{button_prefix}_similar_{article_id}_{rank}",
            ):
                st.session_state.pending_similar_id = article_id
                st.session_state.pending_similar_title = title


def _render_search_tab(engine: IntelligenceEngine, *, ready: bool) -> None:
    _init_search_state()

    if not ready:
        st.subheader("Semantic Search")
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

    st.subheader("Semantic Search")
    query = st.text_input(
        "Search articles",
        placeholder="e.g. climate change, AI chips…",
        key="search_query_input",
    )

    search_clicked = st.button("Search", type="primary")

    if search_clicked:
        if not query.strip():
            st.warning("Enter a search query.")
        else:
            results, latency_ms = _cached_search(engine, query.strip(), k=10)

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
    for rank, result in enumerate(results, start=1):
        _render_result_card(engine, rank, result, button_prefix=prefix)


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
            if 0 <= article_idx < len(engine.metadata):
                domain = str(engine.metadata[article_idx].get("domain", ""))
            rows.append(
                {
                    "umap_x": float(point.get("umap_x", 0.0)),
                    "umap_y": float(point.get("umap_y", 0.0)),
                    "cluster_id": str(point.get("cluster_id", cluster.get("cluster_id"))),
                    "cluster_label": cluster_label,
                    "title": str(point.get("title", "")),
                    "domain": domain,
                }
            )
    return pd.DataFrame(rows)


def _build_cluster_scatter(df: pd.DataFrame):
    return px.scatter(
        df,
        x="umap_x",
        y="umap_y",
        color="cluster_id",
        hover_data={
            "title": True,
            "domain": True,
            "cluster_label": True,
            "umap_x": False,
            "umap_y": False,
            "cluster_id": False,
        },
        color_discrete_sequence=px.colors.qualitative.Bold,
        title="Article clusters (UMAP projection)",
        labels={
            "umap_x": "UMAP X",
            "umap_y": "UMAP Y",
            "cluster_id": "Cluster",
        },
    )


def _render_cluster_article_card(article: dict) -> None:
    title = article.get("title", "Untitled")
    url = article.get("url", "")
    domain = article.get("domain", "unknown")
    date = article.get("date", "—")

    with st.container(border=True):
        st.markdown(f"[{_escape_text(title, 'Untitled')}]({url})")
        _safe_caption(domain, date)


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


def _render_clusters_tab(engine: IntelligenceEngine) -> None:
    st.subheader("Topic Clusters")
    try:
        with st.spinner("Loading clusters..."):
            clusters = engine.get_clusters()
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
    recent_articles = _recent_cluster_articles(articles, limit=10)

    st.markdown("**Most recent articles**")
    st.caption(f"Showing {len(recent_articles)} of {len(articles)} articles")
    for article in recent_articles:
        _render_cluster_article_card(article)


@st.fragment
def _render_clusters_tab_fragment(engine: IntelligenceEngine) -> None:
    """Fragment reruns only when the Clusters tab is interacted with."""
    _render_clusters_tab(engine)


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
    st.subheader("Near-Duplicate Articles")
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


def _render_trending_pill(label: str, score: float, font_size: int) -> None:
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


def _render_topics_trends_tab(engine: IntelligenceEngine) -> None:
    st.subheader("Topics & Trends")

    try:
        if "topics_cache" not in st.session_state:
            with st.spinner("Loading topics and trends..."):
                clusters = engine.get_clusters()
                trending = engine.get_trending(n=len(clusters))
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
            _render_trending_pill(label, score, font_size)
            for article in top_articles:
                title = article.get("title", "Untitled")
                url = article.get("url", "")
                date = article.get("date", "—")
                st.markdown(
                    f'<p style="margin:0 0 0.5rem 0;font-size:0.85rem;">'
                    f'<a href="{_escape_text(url)}" target="_blank" style="color:#93c5fd;text-decoration:none;">'
                    f"{_escape_text(title, 'Untitled')}</a><br>"
                    f'<span style="color:#9ca3af;">{_escape_text(date, '—')}</span></p>',
                    unsafe_allow_html=True,
                )


@st.fragment
def _render_duplicates_tab_fragment(engine: IntelligenceEngine) -> None:
    _render_duplicates_tab(engine)


@st.fragment
def _render_topics_trends_tab_fragment(engine: IntelligenceEngine) -> None:
    _render_topics_trends_tab(engine)


def main() -> None:
    _init_perf_state()
    if not _check_password():
        return

    ready = index_is_ready()
    engine = get_engine() if ready else None

    _render_sidebar(engine, ready=ready)

    st.title("News Intelligence Engine")
    st.caption("Explore semantic search results, topic clusters, and duplicate coverage.")

    if not ready:
        st.error(index_status_message(missing_artifacts()))
        return

    search_tab, clusters_tab, duplicates_tab, topics_tab = st.tabs(
        ["Search", "Clusters", "Duplicates", "Topics & Trends"]
    )

    with search_tab:
        _render_search_tab(engine, ready=ready)

    with clusters_tab:
        _render_clusters_tab_fragment(engine)

    with duplicates_tab:
        _render_duplicates_tab_fragment(engine)

    with topics_tab:
        _render_topics_trends_tab_fragment(engine)


if __name__ == "__main__":
    main()
