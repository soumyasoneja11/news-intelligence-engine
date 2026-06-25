"""Verify local index artifacts before pushing to Streamlit Community Cloud."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQUIRED = [
    ROOT / "index" / "faiss.index",
    ROOT / "index" / "metadata.json",
    ROOT / "index" / "embeddings.npy",
    ROOT / "index" / "cluster_data.json",
    ROOT / "index" / "cluster_labels.json",
    ROOT / "index" / "cluster_topics.json",
    ROOT / "index" / "duplicate_pairs.json",
    ROOT / "index" / "trending.json",
    ROOT / "data" / "articles_clean.csv",
]


def main() -> None:
    missing = [path.relative_to(ROOT) for path in REQUIRED if not path.is_file()]
    if missing:
        print("Missing artifacts — run the pipeline first:")
        print("  python src/pipeline.py")
        print("\nMissing files:")
        for path in missing:
            print(f"  - {path}")
        raise SystemExit(1)

    total_mb = sum(path.stat().st_size for path in REQUIRED) / (1024 * 1024)
    print("All deployment artifacts present.")
    print(f"Total size: {total_mb:.1f} MB")
    print()
    print("Streamlit Community Cloud cannot read gitignored files.")
    print("Force-add artifacts, then push:")
    print()
    print("  git add -f data/articles_clean.csv index/*")
    print("  git commit -m \"Add prebuilt index for cloud deploy\"")
    print("  git push")


if __name__ == "__main__":
    main()
