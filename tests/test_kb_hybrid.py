"""关键词 BM25 + RRF 混合检索（不调 embedding）。"""

from __future__ import annotations

from kb.chunking import Chunk
from kb.hybrid import candidate_pool, retrieval_mode_from_env, rrf_fuse
from kb.keyword import KeywordIndex, tokenize


def _chunk(cid: str, text: str, *, title: str = "", source: str = "a.md") -> Chunk:
    return Chunk(
        id=cid,
        text=text,
        source=source,
        title=title,
        doc_id=source,
        chunk_index=0,
    )


def test_tokenize_cjk_and_latin():
    toks = tokenize("比亚迪 BYD 刀片电池 2024")
    assert "比" in toks and "亚" in toks
    assert "比亚" in toks  # bigram
    assert "byd" in toks
    assert "2024" in toks


def test_bm25_ranks_exact_entity_higher():
    chunks = [
        _chunk("a::0", "市场景气度回升，销量环比改善", title="宏观"),
        _chunk("b::0", "比亚迪刀片电池安全性与空间优势", title="刀片电池"),
        _chunk("c::0", "宁德时代动力电池装机量", title="CATL"),
    ]
    idx = KeywordIndex(chunks)
    hits = idx.search("比亚迪 刀片电池", top_k=2)
    assert hits
    assert hits[0][0].id == "b::0"
    assert hits[0][1] > 0


def test_bm25_year_and_source():
    chunks = [
        _chunk("x::0", "渗透率持续提升", title="乘联会", source="cpca_2024.md"),
        _chunk("y::0", " unrelated english only", title="other", source="misc.md"),
    ]
    idx = KeywordIndex(chunks)
    hits = idx.search("乘联会 2024", top_k=1)
    assert hits[0][0].id == "x::0"


def test_rrf_prefers_overlap():
    a = _chunk("a", "a")
    b = _chunk("b", "b")
    c = _chunk("c", "c")
    # 向量：a > b > c；关键词：b > a > c → b 应因双路靠前胜出或靠前
    fused = rrf_fuse(
        [[(a, 0.9), (b, 0.8), (c, 0.1)], [(b, 5.0), (a, 4.0), (c, 1.0)]],
        rrf_k=60,
        top_k=3,
    )
    ids = [x[0].id for x in fused]
    assert ids[0] in {"a", "b"}
    assert set(ids) == {"a", "b", "c"}
    # a、b 的 RRF 分应明显高于只排第三的 c
    assert fused[0][1] > fused[2][1]


def test_weighted_rrf_favors_vector():
    a = _chunk("a", "a")
    b = _chunk("b", "b")
    c = _chunk("c", "c")
    # 向量 a>b>c；关键词 b>c>a — 等权 b 常第一；向量权重 5 时 a 应压过 b
    rankings = [
        [(a, 0.9), (b, 0.5), (c, 0.1)],
        [(b, 9.0), (c, 4.0), (a, 1.0)],
    ]
    equal = rrf_fuse(rankings, rrf_k=10, top_k=3, weights=(1.0, 1.0))
    heavy_vec = rrf_fuse(rankings, rrf_k=10, top_k=3, weights=(5.0, 1.0))
    assert equal[0][0].id == "b"
    assert heavy_vec[0][0].id == "a"


def test_convex_fuse_alpha():
    from kb.hybrid import convex_fuse

    a = _chunk("a", "a")
    b = _chunk("b", "b")
    vec = [(a, 1.0), (b, 0.1)]
    kw = [(b, 10.0), (a, 1.0)]
    # α=1 → 纯向量序
    only_v = convex_fuse(vec, kw, alpha=1.0, top_k=2)
    assert only_v[0][0].id == "a"
    # α=0 → 纯关键词序
    only_k = convex_fuse(vec, kw, alpha=0.0, top_k=2)
    assert only_k[0][0].id == "b"


def test_mode_from_env(monkeypatch):
    monkeypatch.delenv("KB_RETRIEVAL_MODE", raising=False)
    assert retrieval_mode_from_env() == "vector"
    monkeypatch.setenv("KB_RETRIEVAL_MODE", "hybrid")
    assert retrieval_mode_from_env() == "hybrid"
    assert retrieval_mode_from_env("keyword") == "keyword"


def test_candidate_pool():
    assert candidate_pool(8) == 24
    assert candidate_pool(8, 40) == 40


def test_kb_search_keyword_mode(tmp_path, monkeypatch):
    import json

    from kb import KnowledgeBase

    monkeypatch.setenv("KB_BACKEND", "json")
    chunks = [
        _chunk("p::0", "特斯拉上海超级工厂产能", title="特斯拉", source="tsla.md"),
        _chunk("q::0", "蔚来换电站网络扩张", title="蔚来", source="nio.md"),
    ]
    index_dir = tmp_path / "idx"
    index_dir.mkdir()
    with (index_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    # 假向量：维度一致即可（keyword 模式不读）
    (index_dir / "vectors.json").write_text(json.dumps([[1.0, 0.0], [0.0, 1.0]]))

    kb = KnowledgeBase(index_dir=index_dir, backend="json")
    hits = kb.search("特斯拉 产能", top_k=1, expand_neighbors=0, mode="keyword")
    assert len(hits) == 1
    assert hits[0].chunk.id == "p::0"


def test_kb_search_hybrid_fuses_without_embed_when_keyword_only_path(tmp_path, monkeypatch):
    """hybrid 会调 embedding；此处用 monkeypatch 假向量通路。"""
    import json

    from kb import KnowledgeBase

    monkeypatch.setenv("KB_BACKEND", "json")
    chunks = [
        _chunk("p::0", "特斯拉上海超级工厂产能 2023", title="特斯拉", source="tsla.md"),
        _chunk("q::0", "蔚来换电站网络扩张", title="蔚来", source="nio.md"),
        _chunk("r::0", "理想增程式销售", title="理想", source="li.md"),
    ]
    index_dir = tmp_path / "idx"
    index_dir.mkdir()
    with (index_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    (index_dir / "vectors.json").write_text(
        json.dumps([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    )

    kb = KnowledgeBase(index_dir=index_dir, backend="json")

    def fake_vector(query, top_k, *, exclude_ids=None, exclude_sources=None):
        # 故意让向量偏好蔚来，关键词应把特斯拉拉回来
        order = ["q::0", "r::0", "p::0"]
        by_id = {c.id: c for c in chunks}
        out = []
        for cid in order[:top_k]:
            out.append((by_id[cid], 0.5))
        return out

    monkeypatch.setattr(kb, "_vector_pairs", fake_vector)
    hits = kb.search(
        "特斯拉 产能 2023",
        top_k=2,
        expand_neighbors=0,
        mode="hybrid",
        candidate_k=3,
    )
    ids = [h.chunk.id for h in hits]
    assert "p::0" in ids
