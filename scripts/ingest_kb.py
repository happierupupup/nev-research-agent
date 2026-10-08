#!/usr/bin/env python3
"""Ingest data/kb → data/kb_index/；可选同步 Milvus Lite（无 Docker）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from kb import ingest  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Ingest local KB docs")
    p.add_argument("--docs", type=Path, default=None, help="docs dir (default data/kb)")
    p.add_argument("--index", type=Path, default=None, help="index dir (default data/kb_index)")
    p.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="embedding batch size (DashScope max 20; default 10)",
    )
    p.add_argument(
        "--also-milvus",
        action="store_true",
        help="同时写入 Milvus Lite（data/milvus/nev.db）；也可设 KB_BACKEND=milvus",
    )
    p.add_argument(
        "--no-milvus",
        action="store_true",
        help="强制不写 Milvus（即使 KB_BACKEND=milvus）",
    )
    args = p.parse_args()
    also = True if args.also_milvus else (False if args.no_milvus else None)
    meta = ingest(
        docs_dir=args.docs,
        index_dir=args.index,
        batch_size=args.batch_size,
        also_milvus=also,
    )
    print(f"Ingested {meta['n_docs']} docs → {meta['n_chunks']} chunks")
    print(meta)


if __name__ == "__main__":
    main()
