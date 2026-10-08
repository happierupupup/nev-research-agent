"""Offline tests for KB chunking v2 (no API)."""

from __future__ import annotations

from pathlib import Path

from kb import (
    Chunk,
    Hit,
    _split_markdown,
    build_chunks,
    expand_adjacent_hits,
    format_context,
    iter_doc_files,
    take_hits,
)
from kb.chunking import split_markdown_v2


def test_take_hits_skips_seen_ids_and_sources():
    scored = [
        Hit(Chunk(id="a.md::0", text="a0", source="a.md", title="t"), 0.9),
        Hit(Chunk(id="a.md::1", text="a1", source="a.md", title="t"), 0.8),
        Hit(Chunk(id="b.md::0", text="b0", source="b.md", title="t"), 0.7),
    ]
    out = take_hits(scored, top_k=4, exclude_ids={"a.md::0"}, exclude_sources={"a.md"})
    assert [h.chunk.id for h in out] == ["b.md::0"]


def test_split_markdown_headings():
    text = "# 标题\n\n前言\n\n## 一\n\n内容甲\n\n## 二\n\n内容乙\n"
    chunks = _split_markdown(text, source="demo.md")
    assert len(chunks) >= 2
    assert any("内容甲" in c.text for c in chunks)


def test_split_v2_overlap_and_links():
    # 两段都偏长，强制切成多块并重叠
    para = "新能源汽车渗透率持续提升。" * 40  # ~560
    text = f"# 报告\n\n{para}\n\n{para}\n\n{para}\n"
    chunks = split_markdown_v2(
        text,
        source="long.md",
        url="https://example.com/a",
        crawled_at="2026-10-08T00:00:00Z",
        max_chars=600,
        overlap_chars=120,
    )
    assert len(chunks) >= 2
    assert all(len(c.text) <= 600 + 50 for c in chunks)  # 允许段落边界略超
    assert chunks[0].next_id == chunks[1].id
    assert chunks[1].prev_id == chunks[0].id
    assert chunks[0].url == "https://example.com/a"
    assert chunks[0].crawled_at.startswith("2026")
    # 重叠：后一块开头应能在前一块末尾找到一段公共子串
    if len(chunks) >= 2:
        a, b = chunks[0].text, chunks[1].text
        assert any(a[-i:] and a[-i:] in b for i in (40, 60, 80, 100, 120))


def test_expand_adjacent_pulls_neighbor():
    c0 = Chunk(
        id="d.md::0",
        text="块0",
        source="d.md",
        title="t",
        doc_id="d.md",
        chunk_index=0,
        next_id="d.md::1",
    )
    c1 = Chunk(
        id="d.md::1",
        text="块1",
        source="d.md",
        title="t",
        doc_id="d.md",
        chunk_index=1,
        prev_id="d.md::0",
    )
    by_id = {c0.id: c0, c1.id: c1}
    hits = [Hit(chunk=c0, score=0.95)]
    expanded = expand_adjacent_hits(hits, by_id, radius=1)
    ids = [h.chunk.id for h in expanded]
    assert ids == ["d.md::0", "d.md::1"]


def test_build_chunks_from_sample_kb():
    root = Path(__file__).resolve().parents[1]
    kb_dir = root / "data" / "kb"
    chunks = build_chunks(kb_dir)
    docs = iter_doc_files(kb_dir)
    assert len(docs) >= 10
    assert len(chunks) >= 20
    names = {p.name.lower() for p in docs}
    assert "readme.md" not in names
    blob = "\n".join(c.text for c in chunks)
    assert "刀片电池" in blob
    assert "渗透率" in blob
    assert "37.9" in blob or "海外" in blob
    # 爬取文档应带上 url 元数据
    crawl = [c for c in chunks if c.source.startswith("crawl/")]
    if crawl:
        assert any(c.url.startswith("http") for c in crawl)


def test_qrels_sources_exist():
    import yaml

    root = Path(__file__).resolve().parents[1]
    kb_dir = root / "data" / "kb"
    docs = {p.name for p in kb_dir.glob("*.md") if p.name.lower() != "readme.md"}
    qrels = yaml.safe_load(
        (root / "evals" / "datasets" / "retrieval" / "qrels.yaml").read_text(encoding="utf-8")
    )
    missing: list[str] = []
    for item in qrels["queries"]:
        for src in item["relevant"]:
            if src not in docs:
                missing.append(src)
    assert not missing, f"qrels 指向不存在的语料: {missing}"


def test_format_context():
    h = Hit(
        chunk=Chunk(id="a::0", text="hello", source="a.md", title="t", url="https://x"),
        score=0.9,
    )
    s = format_context([h])
    assert "[1]" in s and "hello" in s and "https://x" in s
