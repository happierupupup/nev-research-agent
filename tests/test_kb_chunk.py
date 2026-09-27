"""Offline tests for KB chunking (no API)."""

from __future__ import annotations

from pathlib import Path

from kb import Chunk, Hit, _split_markdown, build_chunks, format_context, iter_doc_files, take_hits


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


def test_qrels_sources_exist():
    import yaml

    root = Path(__file__).resolve().parents[1]
    kb_dir = root / "data" / "kb"
    docs = {p.name for p in kb_dir.glob("*.md") if p.name.lower() != "readme.md"}
    qrels = yaml.safe_load((root / "evals" / "retrieval" / "qrels.yaml").read_text(encoding="utf-8"))
    missing: list[str] = []
    for item in qrels["queries"]:
        for src in item["relevant"]:
            if src not in docs:
                missing.append(src)
    assert not missing, f"qrels 指向不存在的语料: {missing}"


def test_format_context():
    h = Hit(
        chunk=Chunk(id="a::0", text="hello", source="a.md", title="t"),
        score=0.9,
    )
    s = format_context([h])
    assert "[1]" in s and "hello" in s
