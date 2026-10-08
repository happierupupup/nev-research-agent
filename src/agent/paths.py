"""Repo root helpers. Package lives under src/agent/ → parents[2] = repo root."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
KB_DOCS = ROOT / "data" / "kb"
KB_INDEX = ROOT / "data" / "kb_index"
KB_STAGING = ROOT / "data" / "kb_staging"  # 爬取待审；审核通过再进 KB_DOCS
CRAWL_QUERIES = ROOT / "data" / "crawl" / "search_queries.yaml"  # v1 批量搜索词（保留）
MILVUS_DB = ROOT / "data" / "milvus" / "nev.db"  # Milvus Lite 本地文件（无 Docker）
MEMORY_DIR = ROOT / "data" / "memory" / "users"
SESSIONS_DIR = ROOT / "data" / "sessions"  # Gradio 会话持久化
USERS_FILE = ROOT / "config" / "users.yaml"  # 测试用户 admin / alice
