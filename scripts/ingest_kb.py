#!/usr/bin/env python3
"""Ingest data/kb/*.md into data/kb_index/ (embeddings via OpenAI-compatible API)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kb import ingest  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Ingest local KB docs")
    p.add_argument("--docs", type=Path, default=None, help="docs dir (default data/kb)")
    p.add_argument("--index", type=Path, default=None, help="index dir (default data/kb_index)")
    args = p.parse_args()
    meta = ingest(docs_dir=args.docs, index_dir=args.index)
    print(f"Ingested {meta['n_docs']} docs → {meta['n_chunks']} chunks")
    print(meta)


if __name__ == "__main__":
    main()
