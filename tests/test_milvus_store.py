"""Milvus Lite 单元测试：无 API Key 时跳过；有 pymilvus 则测本地 .db。"""

from __future__ import annotations

import pytest

from kb.chunking import Chunk


pytest.importorskip("pymilvus")


def test_milvus_lite_replace_and_search(tmp_path):
    from kb.milvus_store import MilvusStore

    db = tmp_path / "t.db"
    store = MilvusStore(uri=str(db))
    chunks = [
        Chunk(
            id="a.md::0",
            text="比亚迪刀片电池安全性高",
            source="a.md",
            title="刀片",
            doc_id="a.md",
            chunk_index=0,
            next_id="a.md::1",
            url="https://example.com/a",
        ),
        Chunk(
            id="a.md::1",
            text="刀片电池续航与空间表现",
            source="a.md",
            title="刀片",
            doc_id="a.md",
            chunk_index=1,
            prev_id="a.md::0",
            url="https://example.com/a",
        ),
    ]
    # 假向量（正交一点即可测通路）
    v0 = [1.0, 0.0, 0.0, 0.0]
    v1 = [0.0, 1.0, 0.0, 0.0]
    n = store.replace_all(chunks, [v0, v1])
    assert n == 2
    hits = store.search(v0, top_k=1)
    assert hits and hits[0][0].id == "a.md::0"
    got = store.get_by_ids(["a.md::1"])
    assert got["a.md::1"].prev_id == "a.md::0"
    assert hits[0][0].text  # 回归：search 必须带回正文，否则 ask_kb 会答「不知道」


def test_hit_fields_prefers_fields_attr():
    from kb.milvus_store import _hit_fields

    class FakeHit:
        distance = 0.9
        id = "x::0"
        fields = {
            "id": "x::0",
            "text": "刀片电池安全性",
            "source": "x.md",
            "title": "t",
        }

    fields, score = _hit_fields(FakeHit())
    assert score == 0.9
    assert fields["text"] == "刀片电池安全性"
    assert fields["source"] == "x.md"
