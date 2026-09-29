"""Repo root helpers. Package lives under src/agent/ → parents[2] = repo root."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
KB_DOCS = ROOT / "data" / "kb"
KB_INDEX = ROOT / "data" / "kb_index"
MEMORY_DIR = ROOT / "data" / "memory" / "users"
