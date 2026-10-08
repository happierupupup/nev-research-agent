"""Repo root helpers. Package lives under src/agent/ → parents[2] = repo root."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
KB_DOCS = ROOT / "data" / "kb"
KB_INDEX = ROOT / "data" / "kb_index"
KB_STAGING = ROOT / "data" / "kb_staging"  # 爬取待审；审核通过再进 KB_DOCS
CRAWL_QUERIES = ROOT / "data" / "crawl" / "search_queries.yaml"  # 搜索词/参数，不是结果
MEMORY_DIR = ROOT / "data" / "memory" / "users"
