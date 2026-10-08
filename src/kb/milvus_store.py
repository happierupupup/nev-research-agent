"""Milvus Lite 后端（无 Docker）：本地 .db 文件持久化。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from agent.paths import MILVUS_DB
from kb.chunking import Chunk

COLLECTION = "nev_chunks"
# 正文上限（Milvus VARCHAR）；超长截断写入，检索仍够用
_TEXT_MAX = 6000
_ID_MAX = 512
_SHORT_MAX = 1024


def milvus_uri(path: Path | None = None) -> str:
    p = path or Path(os.getenv("MILVUS_URI") or MILVUS_DB)
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


def _row_from_chunk(c: Chunk, vector: list[float]) -> dict[str, Any]:
    return {
        "id": (c.id or "")[:_ID_MAX],
        "text": (c.text or "")[:_TEXT_MAX],
        "source": (c.source or "")[:_SHORT_MAX],
        "title": (c.title or "")[:_SHORT_MAX],
        "doc_id": (c.doc_id or c.source or "")[:_SHORT_MAX],
        "url": (c.url or "")[:_SHORT_MAX],
        "crawled_at": (c.crawled_at or "")[:128],
        "chunk_index": int(c.chunk_index or 0),
        "prev_id": (c.prev_id or "")[:_ID_MAX],
        "next_id": (c.next_id or "")[:_ID_MAX],
        "overlap_chars": int(c.overlap_chars or 0),
        "embedding": vector,
    }


def _hit_fields(h: Any) -> tuple[dict[str, Any], float]:
    """
    pymilvus Hit：正文在 ``h.fields``（或 to_dict()['entity']）。
    注意：``Hit.to_dict()`` 仍可能返回 Hit（不是 dict），不能 ``isinstance(..., dict)`` 后清空。
    """
    # 首选：Hit.fields + distance/id
    fields = getattr(h, "fields", None)
    if isinstance(fields, dict) and fields:
        try:
            score = float(getattr(h, "distance", getattr(h, "score", 0.0)))
        except (TypeError, ValueError):
            score = 0.0
        entity = dict(fields)
        pk = getattr(h, "id", None)
        if pk:
            entity.setdefault("id", pk)
        return entity, score

    if isinstance(h, dict):
        d: dict[str, Any] = h
    elif hasattr(h, "to_dict"):
        raw = h.to_dict()
        if isinstance(raw, dict):
            d = raw
        else:
            # to_dict 仍是 Hit：再走 fields
            f2 = getattr(raw, "fields", None)
            if isinstance(f2, dict) and f2:
                try:
                    score = float(getattr(raw, "distance", 0.0))
                except (TypeError, ValueError):
                    score = 0.0
                entity = dict(f2)
                pk = getattr(raw, "id", None) or getattr(h, "id", None)
                if pk:
                    entity.setdefault("id", pk)
                return entity, score
            d = {}
    else:
        try:
            d = dict(h)
        except Exception:  # noqa: BLE001
            d = {}

    try:
        score = float(d.get("distance", d.get("score", 0.0)) or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    entity = d.get("entity") if isinstance(d.get("entity"), dict) else {}
    nested = entity.get("entity") if isinstance(entity, dict) else None
    if isinstance(nested, dict) and "text" not in entity and "text" in nested:
        entity = nested
    if not entity and ("text" in d or "source" in d):
        entity = {k: v for k, v in d.items() if k not in {"distance", "score"}}
    pk = d.get("id") or getattr(h, "id", None) or entity.get("id")
    if pk:
        entity.setdefault("id", pk)
    return entity, score


def _chunk_from_row(row: dict[str, Any]) -> Chunk:
    prev = row.get("prev_id") or None
    nxt = row.get("next_id") or None
    if prev == "":
        prev = None
    if nxt == "":
        nxt = None
    return Chunk(
        id=str(row.get("id") or ""),
        text=str(row.get("text") or ""),
        source=str(row.get("source") or ""),
        title=str(row.get("title") or ""),
        doc_id=str(row.get("doc_id") or row.get("source") or ""),
        chunk_index=int(row.get("chunk_index") or 0),
        prev_id=prev,
        next_id=nxt,
        url=str(row.get("url") or ""),
        crawled_at=str(row.get("crawled_at") or ""),
        overlap_chars=int(row.get("overlap_chars") or 0),
    )


class MilvusStore:
    """薄封装：重建 collection、批量写入、向量检索、按 id 取邻块。"""

    def __init__(
        self,
        *,
        uri: str | None = None,
        collection: str = COLLECTION,
    ) -> None:
        from pymilvus import MilvusClient

        self.uri = uri or milvus_uri()
        self.collection = collection
        self.client = MilvusClient(uri=self.uri)

    def drop_collection(self) -> None:
        if self.client.has_collection(self.collection):
            self.client.drop_collection(self.collection)

    def ensure_collection(self, dim: int) -> None:
        from pymilvus import DataType

        if self.client.has_collection(self.collection):
            return
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=_ID_MAX)
        schema.add_field("text", DataType.VARCHAR, max_length=_TEXT_MAX)
        schema.add_field("source", DataType.VARCHAR, max_length=_SHORT_MAX)
        schema.add_field("title", DataType.VARCHAR, max_length=_SHORT_MAX)
        schema.add_field("doc_id", DataType.VARCHAR, max_length=_SHORT_MAX)
        schema.add_field("url", DataType.VARCHAR, max_length=_SHORT_MAX)
        schema.add_field("crawled_at", DataType.VARCHAR, max_length=128)
        schema.add_field("chunk_index", DataType.INT64)
        schema.add_field("prev_id", DataType.VARCHAR, max_length=_ID_MAX)
        schema.add_field("next_id", DataType.VARCHAR, max_length=_ID_MAX)
        schema.add_field("overlap_chars", DataType.INT64)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=dim)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )
        self.client.create_collection(
            collection_name=self.collection,
            schema=schema,
            index_params=index_params,
        )

    def replace_all(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("chunks / vectors length mismatch")
        if not chunks:
            return 0
        dim = len(vectors[0])
        self.drop_collection()
        self.ensure_collection(dim)
        rows = [_row_from_chunk(c, v) for c, v in zip(chunks, vectors)]
        # 分批写入，避免单次过大
        batch = 64
        for i in range(0, len(rows), batch):
            self.client.insert(collection_name=self.collection, data=rows[i : i + batch])
        self.client.flush(self.collection)
        return len(rows)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 4,
        *,
        exclude_ids: set[str] | None = None,
        exclude_sources: set[str] | None = None,
    ) -> list[tuple[Chunk, float]]:
        if not self.client.has_collection(self.collection):
            raise FileNotFoundError(
                f"Milvus collection `{self.collection}` missing under {self.uri}. "
                "Run: python scripts/ingest_kb.py --also-milvus"
            )
        # 多取一些再本地过滤 exclude
        want = max(top_k * 3, top_k)
        raw = self.client.search(
            collection_name=self.collection,
            data=[query_vector],
            limit=want,
            output_fields=[
                "id",
                "text",
                "source",
                "title",
                "doc_id",
                "url",
                "crawled_at",
                "chunk_index",
                "prev_id",
                "next_id",
                "overlap_chars",
            ],
        )
        skip_ids = exclude_ids or set()
        skip_src = exclude_sources or set()
        out: list[tuple[Chunk, float]] = []
        hits = raw[0] if raw else []
        for h in hits:
            entity, score = _hit_fields(h)
            c = _chunk_from_row(entity)
            if c.id in skip_ids or c.source in skip_src:
                continue
            out.append((c, score))
            if len(out) >= top_k:
                break
        return out

    def get_by_ids(self, ids: list[str]) -> dict[str, Chunk]:
        ids = [i for i in ids if i]
        if not ids or not self.client.has_collection(self.collection):
            return {}
        rows = self.client.get(
            collection_name=self.collection,
            ids=ids,
            output_fields=[
                "id",
                "text",
                "source",
                "title",
                "doc_id",
                "url",
                "crawled_at",
                "chunk_index",
                "prev_id",
                "next_id",
                "overlap_chars",
            ],
        )
        out: dict[str, Chunk] = {}
        for row in rows or []:
            if not isinstance(row, dict):
                row = dict(row)
            c = _chunk_from_row(row)
            if c.id:
                out[c.id] = c
        return out
