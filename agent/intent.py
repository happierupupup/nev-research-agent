"""Lightweight query type heuristic (no LLM)."""

from __future__ import annotations

_REPORT_MARKERS = (
    "分析",
    "梳理",
    "对比",
    "评估",
    "战略",
    "风险",
    "研究报告",
    "研报",
)


def is_report_query(text: str) -> bool:
    t = (text or "").strip()
    return any(m in t for m in _REPORT_MARKERS)
