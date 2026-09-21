"""Repo root helpers."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KB_DOCS = ROOT / "data" / "kb"
KB_INDEX = ROOT / "data" / "kb_index"
