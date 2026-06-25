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
│ recency-  │ FAISS k-  │ TF-IDF per    │ KMeans +  │ MiniLM-L6   │ cc_news /
│ weighted  │ NN cosine │ cluster       │ UMAP 2D   │ v2 vectors  │ CSV /
│ scores    │ ≥ 0.92    │ keywords      │ elbow k   │ IndexFlatIP │ synthetic
└───────────┴───────────┴───────────────┴───────────┴─────────────┘
                                ▲
                                │
                    python src/pipeline.py (orchestrator)
```

| Layer | Module | Output |
|-------|--------|--------|
| 1 — Ingest | `src/ingest.py` | `data/articles_clean.csv` |
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

python src/pipeline.py
streamlit run app.py
```

The first pipeline run downloads ~2,000 articles from HuggingFace (`vblagoje/cc_news`), encodes them, and writes all index artifacts. Subsequent runs can skip steps:

```bash
python src/pipeline.py --skip-ingest          # reuse existing CSV
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
| **Rebuild index** | One-click sidebar button to rerun the full pipeline and refresh all artifacts. |

## How it works

**Embeddings.** Each article's title and body are passed through `all-MiniLM-L6-v2`, a compact sentence-transformer model that maps text into a 384-dimensional vector. Similar stories end up with vectors that point in nearly the same direction, even when they use different words.

**FAISS.** Facebook AI Similarity Search (FAISS) stores every article vector in an index optimized for fast nearest-neighbor lookup. Vectors are L2-normalized so that inner-product search equals cosine similarity. At query time, your search text is embedded with the same model and FAISS returns the closest vectors in milliseconds—no keyword matching required.

Downstream layers reuse those same vectors: KMeans groups them into topics, TF-IDF names each cluster, neighbor search flags duplicates, and recency-weighted counts drive trending scores.

## Project layout

```
news-intelligence-engine/
├── app.py                  # Streamlit dashboard
├── requirements.txt
├── data/
│   └── articles_clean.csv
├── index/                  # generated artifacts (gitignored contents)
├── src/
│   ├── pipeline.py         # build orchestrator
│   ├── intelligence.py     # unified API facade
│   ├── ingest.py · embedder.py · retriever.py
│   ├── cluster.py · topics.py · dedup.py · trending.py
│   └── paths.py · utils.py
└── tests/
```

## Known limitations

- **Batch-only ingestion** — articles are fetched once at build time; there is no live RSS or API polling.
- **Exhaustive index** — `IndexFlatIP` scans every vector on each query, which is fine for thousands of articles but does not scale to millions.
- **Historical demo data** — the default `cc_news` corpus is from 2017–2018, so trending scores are often zero because nothing falls in the last 24 hours.
- **Incremental embed drift** — re-embedding only new URLs can leave the FAISS index slightly out of sync with the CSV row count.
- **No generative summaries** — cluster labels come from TF-IDF keywords, not an LLM.

## Ideas for v2

- **HNSW index** — swap `IndexFlatIP` for `IndexHNSWFlat` to get sub-linear search as the corpus grows.
- **LLM-based summaries** — generate human-readable cluster digests and per-article blurbs with a local or API-backed model.
- **Real-time RSS ingestion** — poll feeds on a schedule, incrementally embed new articles, and refresh trending on a rolling window.

## Tests

```bash
python tests/test_retrieval.py
python tests/test_intelligence.py
```

## Requirements

Python 3.10+ recommended. Key dependencies: `sentence-transformers`, `faiss-cpu`, `scikit-learn`, `umap-learn`, `streamlit`, `plotly`.
