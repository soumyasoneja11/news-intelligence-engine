# Deployment Guide

The app needs ~**1 GB RAM** (sentence-transformers + FAISS) and pre-built files under `data/` and `index/`. Pick one path below.

## Option A — Streamlit Community Cloud (free public URL)

Best if you want a quick shareable link and already use GitHub.

### 1. Build locally

```bash
pip install -r requirements.txt -r requirements-build.txt
python src/pipeline.py --rss-refresh --full-rebuild
python scripts/prepare_cloud_deploy.py
```

### 2. Push to GitHub

`data/` and `index/` are gitignored by default. Force-add the built artifacts:

```bash
git init
git add .
git add -f data/articles_clean.csv index/*
git commit -m "Initial commit with prebuilt index"
git branch -M main
git remote add origin https://github.com/YOUR_USER/news-intelligence-engine.git
git push -u origin main
```

### 3. Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. **New app** → select your repo, branch `main`, main file path **`app.py`**.
3. Advanced settings → Python **3.11** (repo includes `.python-version`).
4. Optional secrets (see `.streamlit/secrets.toml.example`):
   - `ALLOW_REBUILD = "false"`
   - `APP_PASSWORD = "your-password"`
5. Deploy. First load may take 1–2 minutes while the embedding model loads into memory.

### Notes

- **Rebuild Index** is disabled by default in Docker (`ALLOW_REBUILD=false`). On Streamlit Cloud, set the same in secrets — rebuilding in the cloud often exceeds free-tier RAM.
- **Live updates:** Enable GitHub Actions workflow write permission (Settings → Actions → General → Workflow permissions: Read and write). Workflows `RSS Refresh` (every 30 min) and `RSS Full Rebuild` (daily 03:00 UTC) commit updated `data/` and `index/` back to the repo; Streamlit Cloud redeploys on push.
- Set `TRENDING_REFERENCE=""` in workflow env (already configured) so trending scores use real wall-clock recency for live BBC articles.
- Optional: set `HF_TOKEN` in app secrets for faster HuggingFace model downloads on first boot.

---

## Option B — Docker (local, VPS, Railway, Render, Fly.io)

### Build and run locally

```bash
docker build -t news-intelligence-engine .
docker run --rm -p 8501:8501 news-intelligence-engine
```

Open [http://localhost:8501](http://localhost:8501).

The image runs `python src/pipeline.py` during `docker build`, so startup is fast. To skip baking the index (faster build, slower first boot):

```bash
docker build --build-arg BUILD_INDEX=false -t news-intelligence-engine .
```

### Railway

1. Push the repo to GitHub.
2. [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub repo**.
3. Railway detects the `Dockerfile` automatically.
4. Add a public domain under **Settings → Networking**. Railway sets `PORT` for you.

### Render

1. [render.com](https://render.com) → **New Web Service** → connect GitHub repo.
2. Environment: **Docker**.
3. Instance type: at least **1 GB RAM** (Standard).
4. Render injects `PORT`; the entrypoint script reads it.

### Fly.io

```bash
fly launch --no-deploy
fly scale memory 1024
fly deploy
```

---

## Option C — Run on a VPS (no Docker)

```bash
sudo apt update && sudo apt install -y python3.11-venv libgomp1
git clone <your-repo-url> && cd news-intelligence-engine
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/pipeline.py

# foreground
streamlit run app.py --server.address 0.0.0.0 --server.port 8501

# or use systemd / nginx reverse-proxy in production
```

---

## Health check

After deploy, verify:

1. Sidebar shows article/cluster/duplicate counts (not zeros with errors).
2. **Search** tab returns results for `climate change`.
3. **Clusters** tab renders the UMAP scatter plot.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Index not built yet` | Run pipeline locally or set `BUILD_INDEX=true` in Docker build |
| App crashes on startup | Increase RAM to ≥1 GB |
| `ModuleNotFoundError: src` | Ensure `app.py` is at repo root and you deploy from project root |
| HuggingFace rate limits | Set `HF_TOKEN` secret / env var |
| Docker build timeout | Use `BUILD_INDEX=false` and let entrypoint build on first boot |
