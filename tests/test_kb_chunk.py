"""Offline tests for KB chunking (no API)."""

from __future__ import annotations

from pathlib import Path

from kb import _split_markdown, build_chunks, format_context
from kb import Hit, Chunk


def test_split_markdown_headings():
    text = "# 标题\n\n前言\n\n## 一\n\n内容甲\n\n## 二\n\n内容乙\n"
    chunks = _split_markdown(text, source="demo.md")
    assert len(chunks) >= 2
    assert any("内容甲" in c.text for c in chunks)


def test_build_chunks_from_sample_kb():
    root = Path(__file__).resolve().parents[1]
    chunks = build_chunks(root / "data" / "kb")
    assert len(chunks) >= 3
    blob = "\n".join(c.text for c in chunks)
    assert "刀片电池" in blob
    assert "渗透率" in blob


def test_format_context():
    h = Hit(
        chunk=Chunk(id="a::0", text="hello", source="a.md", title="t"),
        score=0.9,
    )
    s = format_context([h])
    assert "[1]" in s and "hello" in s
