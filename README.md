# News Intelligence Engine

A Python pipeline and Streamlit dashboard that ingests news articles, builds a semantic search index, and surfaces clusters, near-duplicates, and trending topics. Articles are encoded with sentence embeddings and indexed with FAISS so you can search by meaning rather than keywords, explore how stories group together, and spot syndicated or republished content—all from a single local build.

## Architecture

Six processing layers feed a unified `IntelligenceEngine` facade consumed by the Streamlit app:

```
┌─────────────────────────────────────────────────────────────────┐
│                     Presentation Layer                          │
│           app.py  ·  IntelligenceEngine (src/intelligence.py)   │
└───────────────────────────────┬─────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
   Search / Similar        Cluster viz            Dedup & Trends
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                │
┌───────────────────────────────┴─────────────────────────────────┐
│                        index/ artifacts                         │
│  faiss.index · embeddings.npy · metadata.json · *.json          │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────┬───────────┬───────┴───────┬───────────┬─────────────┐
│  Layer 6  │  Layer 5  │    Layer 4    │  Layer 3  │   Layer 2   │  Layer 1
│ Trending  │   Dedup   │    Topics     │  Cluster  │    Embed    │  Ingest
│ trending  │ duplicate │ cluster_      │ cluster_  │ embeddings  │ articles_
│ .json     │ _pairs    │ topics.json   │ data.json │ + FAISS     │ clean.csv
│           │ .json     │               │           │             │
│ recency-  │ FAISS k-  │ TF-IDF per    │ KMeans +  │ MiniLM-L6   │ BBC RSS
│ weighted  │ NN cosine │ cluster       │ UMAP 2D   │ v2 vectors  │ feeds
│ scores    │ ≥ 0.92    │ keywords      │ elbow k   │ IndexFlatIP │ append
└───────────┴───────────┴───────────────┴───────────┴─────────────┘
                                ▲
                                │
                    python src/pipeline.py (orchestrator)
```

| Layer | Module | Output |
|-------|--------|--------|
| 1 — Ingest | `src/ingest.py`, `src/rss_ingest.py` | `data/articles_clean.csv` |
| 2 — Embed | `src/embedder.py` | `index/faiss.index`, `embeddings.npy`, `metadata.json` |
| 3 — Cluster | `src/cluster.py` | `cluster_data.json`, `cluster_labels.json` |
| 4 — Topics | `src/topics.py` | `cluster_topics.json` |
| 5 — Dedup | `src/dedup.py` | `duplicate_pairs.json` |
| 6 — Trending | `src/trending.py` | `trending.json` |

## Quickstart

```bash
git clone <your-repo-url>
cd news-intelligence-engine

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt

# Full pipeline build (ingest, cluster, UMAP):
pip install -r requirements-build.txt
python src/pipeline.py
streamlit run app.py
```

The first pipeline run fetches BBC RSS feeds (see `data/rss_feeds.json`), encodes them, and writes all index artifacts. Scheduled GitHub Actions refresh the index every 30 minutes.

```bash
python src/pipeline.py --rss-refresh --skip-cluster   # fast RSS refresh
python src/pipeline.py --rss-refresh --full-rebuild # daily full rebuild
python src/pipeline.py --skip-ingest --skip-embed   # ML-only rebuild
```

## Features

| Feature | Description |
|---------|-------------|
| **Semantic search** | Find articles by meaning using natural-language queries, ranked by cosine similarity. |
| **Find similar** | From any search result, jump to the ten closest articles in embedding space. |
| **Topic clusters** | Interactive UMAP scatter plot of how articles group into thematic clusters. |
| **Cluster explorer** | Pick a cluster to see its TF-IDF keywords and ten most recent member articles. |
| **Near-duplicate detection** | Surface article pairs above a cosine-similarity threshold, with CSV export. |
| **Topics & trends** | Bar charts of largest clusters and a trending-topic view weighted by publication recency. |
| **Live BBC RSS** | GitHub Actions polls BBC feeds every 30 min; daily full cluster rebuild at 03:00 UTC. |
| **Rebuild index** | One-click sidebar button to rerun the full pipeline (local dev; disabled on cloud). |

## How it works

**Embeddings.** Each article's title and body are passed through `all-MiniLM-L6-v2`, a compact sentence-transformer model that maps text into a 384-dimensional vector. Similar stories end up with vectors that point in nearly the same direction, even when they use different words.

**FAISS.** Facebook AI Similarity Search (FAISS) stores every article vector in an index optimized for fast nearest-neighbor lookup. Vectors are L2-normalized so that inner-product search equals cosine similarity. At query time, your search text is embedded with the same model and FAISS returns the closest vectors in milliseconds—no keyword matching required.

Downstream layers reuse those same vectors: KMeans groups them into topics, TF-IDF names each cluster, neighbor search flags duplicates, and recency-weighted counts drive trending scores.

**Live ingestion.** BBC RSS feeds (Top Stories, World, Technology, Business) are polled every 30 minutes by [`.github/workflows/rss-refresh.yml`](.github/workflows/rss-refresh.yml). New articles append to the CSV, get incrementally embedded, and dedup/trending refresh without re-clustering. A daily full rebuild ([`.github/workflows/rss-full.yml`](.github/workflows/rss-full.yml)) re-runs clustering and topics. Updated `data/` and `index/` artifacts are committed back to the repo so Streamlit Cloud redeploys automatically.

> **BBC content:** This demo indexes publicly available RSS summaries for educational use. For a public deployment, review [BBC Terms of Use](https://www.bbc.co.uk/usingthebbc/terms) and provide appropriate attribution.

## Project layout

```
news-intelligence-engine/
├── app.py                  # Streamlit dashboard
├── requirements.txt
├── data/
│   ├── articles_clean.csv
│   └── rss_feeds.json
├── index/                  # generated artifacts (gitignored contents)
├── src/
│   ├── pipeline.py         # build orchestrator
│   ├── intelligence.py     # unified API facade
│   ├── rss_ingest.py       # BBC RSS fetch + CSV append
│   ├── ingest.py · embedder.py · retriever.py
│   ├── cluster.py · topics.py · dedup.py · trending.py
│   └── paths.py · utils.py
└── tests/
```

## Known limitations

- **RSS summaries only** — BBC feed entries use short descriptions; full article text is not scraped (respect BBC ToS in production demos).
- **Exhaustive index** — `IndexFlatIP` scans every vector on each query, which is fine for thousands of articles but does not scale to millions.
- **Cluster staleness** — fast RSS refreshes skip clustering; UMAP/cluster labels refresh on the daily full rebuild.
- **No generative summaries** — cluster labels come from TF-IDF keywords, not an LLM.

## Ideas for v2

- **LLM-based summaries** — generate human-readable cluster digests and per-article blurbs with a local or API-backed model.
- **Full-text fetch** — optional article-page extraction for richer embeddings (with publisher ToS compliance).
- **API layer** — FastAPI service separate from the Streamlit UI.

## Tests

```bash
python tests/test_rss_ingest.py
python tests/test_retrieval.py
python tests/test_intelligence.py
```

## Requirements

Python **3.11** recommended (see `.python-version`). 

- **App runtime:** `pip install -r requirements.txt`
- **Full pipeline build:** `pip install -r requirements-build.txt`

Key dependencies: `sentence-transformers`, `faiss-cpu`, `scikit-learn`, `streamlit`, `plotly`.

### Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `TRENDING_REFERENCE` | `auto` | Trending anchor (`auto` = latest article date; `""` = wall-clock now for live RSS) |
| `ALLOW_REBUILD` | `true` | Show sidebar Rebuild Index button |
| `APP_PASSWORD` | — | Optional dashboard password |
| `FAISS_INDEX_MODE` | `auto` | `flat`, `hnsw`, or `auto` (HNSW when n ≥ 5000) |
| `MAX_ARTICLES` | `5000` | Cap corpus size after RSS append |
| `INGEST_FALLBACK` | — | Set to `synthetic` for offline dev without network |
