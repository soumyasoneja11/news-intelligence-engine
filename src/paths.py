"""Project-root-relative paths shared across pipeline modules."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INDEX_DIR = PROJECT_ROOT / "index"

ARTICLES_PATH = DATA_DIR / "articles_clean.csv"
FEED_STATS_PATH = DATA_DIR / "feed_stats.json"
SCHEDULER_LOG_PATH = PROJECT_ROOT / "logs" / "scheduler.log"
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"
FAISS_PATH = INDEX_DIR / "faiss.index"
METADATA_PATH = INDEX_DIR / "metadata.json"
