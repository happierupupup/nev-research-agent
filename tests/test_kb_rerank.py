"""重排：lexical（无 API）+ 模式解析。"""

from __future__ import annotations

from kb.chunking import Chunk
from kb.rerank import (
    apply_rerank,
    lexical_score,
    rerank_candidate_count,
    rerank_lexical,
    rerank_mode_from_env,
)


def _c(cid: str, text: str, *, title: str = "") -> Chunk:
    return Chunk(
        id=cid,
        text=text,
        source=f"{cid}.md",
        title=title,
        doc_id=f"{cid}.md",
        chunk_index=0,
    )


def test_mode_from_env(monkeypatch):
    monkeypatch.delenv("KB_RERANK", raising=False)
    assert rerank_mode_from_env() == "off"
    monkeypatch.setenv("KB_RERANK", "gte")
    assert rerank_mode_from_env() == "gte"
    assert rerank_mode_from_env("lexical") == "lexical"


def test_candidate_count():
    assert rerank_candidate_count(8) == 32
    assert rerank_candidate_count(8, 40) == 40


def test_lexical_prefers_title_match():
    a = _c("a", "市场景气度回升", title="宏观综述")
    b = _c("b", "刀片电池结构与安全", title="比亚迪刀片电池")
    # 初检分数故意让 a 更高
    out = rerank_lexical("比亚迪 刀片电池", [(a, 0.95), (b, 0.40)], top_k=2)
    assert out[0][0].id == "b"
    assert lexical_score("比亚迪 刀片电池", b, base_score=0.4) > lexical_score(
        "比亚迪 刀片电池", a, base_score=0.95
    )


def test_apply_rerank_off_truncates():
    hits = [(_c(str(i), f"t{i}"), float(10 - i)) for i in range(5)]
    out = apply_rerank("q", hits, top_k=2, mode="off")
    assert len(out) == 2
    assert out[0][0].id == "0"


def test_kb_search_with_lexical_rerank(tmp_path, monkeypatch):
    import json

    from kb import KnowledgeBase

    monkeypatch.setenv("KB_BACKEND", "json")
    monkeypatch.setenv("KB_RERANK", "lexical")
    chunks = [
        _c("noise", "一般行业评论与展望", title="综述"),
        _c("gold", "特斯拉上海超级工厂产能爬坡", title="特斯拉产能"),
        _c("mid", "新能源乘用车渗透率", title="渗透率"),
    ]
    # 假向量：让 noise 余弦最高，gold 居中 —— 重排应把 gold 抬到前面
    index_dir = tmp_path / "idx"
    index_dir.mkdir()
    with (index_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    (index_dir / "vectors.json").write_text(
        json.dumps([[1.0, 0.0], [0.6, 0.4], [0.2, 0.8]])
    )

    kb = KnowledgeBase(index_dir=index_dir, backend="json")

    def fake_vector(query, top_k, *, exclude_ids=None, exclude_sources=None):
        order = ["noise", "gold", "mid"]
        by_id = {c.id: c for c in chunks}
        scores = {"noise": 0.99, "gold": 0.70, "mid": 0.50}
        return [(by_id[i], scores[i]) for i in order[:top_k]]

    monkeypatch.setattr(kb, "_vector_pairs", fake_vector)
    hits = kb.search(
        "特斯拉 产能",
        top_k=2,
        expand_neighbors=0,
        mode="vector",
        rerank="lexical",
        rerank_candidates=3,
    )
    assert hits[0].chunk.id == "gold"
