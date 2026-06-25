"""Container entrypoint: ensure index exists, then start Streamlit."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PIPELINE = ROOT / "src" / "pipeline.py"
FAISS_INDEX = ROOT / "index" / "faiss.index"


def main() -> None:
    os.chdir(ROOT)

    if not FAISS_INDEX.is_file():
        print("Index not found — running build pipeline (first boot may take several minutes)...")
        subprocess.run([sys.executable, str(PIPELINE)], check=True)

    port = os.environ.get("PORT", "8501")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app.py",
            "--server.headless=true",
            "--server.port",
            port,
            "--server.address",
            "0.0.0.0",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
