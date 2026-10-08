"""本地知识库：分块 v2 → 向量索引 → 检索（命中可扩展邻块）。"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from agent.paths import KB_DOCS, KB_INDEX
from kb.chunking import (
    DEFAULT_MAX_CHARS,
    DEFAULT_OVERLAP_CHARS,
    Chunk,
    parse_doc_file,
    split_markdown_v2,
    _split_markdown,  # noqa: F401  — 测试兼容
)


def kb_backend() -> str:
    """json（默认本地余弦）| milvus（Milvus Lite 文件库）。"""
    return (os.getenv("KB_BACKEND") or "json").strip().lower()


META_NAME = "meta.json"
CHUNKS_NAME = "chunks.jsonl"
VECTORS_NAME = "vectors.json"


@dataclass
class Hit:
    chunk: Chunk
    score: float


def iter_doc_files(docs_dir: Path | None = None) -> list[Path]:
    root = docs_dir or KB_DOCS
    if not root.exists():
        return []
    skip_names = {"readme.md", "sources.md"}
    files = sorted(root.rglob("*"))
    return [
        p
        for p in files
        if p.is_file()
        and p.suffix.lower() in {".md", ".txt"}
        and p.name.lower() not in skip_names
    ]


def _strip_frontmatter(text: str) -> str:
    if not text.startswith("---"):
        return text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return text
    return parts[2].lstrip("\n")


def build_chunks(
    docs_dir: Path | None = None,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[Chunk]:
    out: list[Chunk] = []
    base = docs_dir or KB_DOCS
    for path in iter_doc_files(docs_dir):
        doc_meta, body = parse_doc_file(path, base=base)
        out.extend(
            split_markdown_v2(
                body,
                source=doc_meta.source,
                url=doc_meta.url,
                crawled_at=doc_meta.crawled_at,
                title_hint=doc_meta.title_hint,
                max_chars=max_chars,
                overlap_chars=overlap_chars,
            )
        )
    return out


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(y * y for y in b)) or 1e-12
    return dot / (na * nb)


def expand_adjacent_hits(
    hits: list[Hit],
    chunks_by_id: dict[str, Chunk],
    *,
    radius: int = 1,
) -> list[Hit]:
    """
    搜到一块后，用 prev_id / next_id 把前后邻块一起带回。
    例如长文拆成块0、块1（有重叠）——命中块0 时也会带上块1。
    """
    if radius <= 0 or not hits:
        return list(hits)

    seen: set[str] = set()
    out: list[Hit] = []
    doc_order: list[str] = []

    for h in hits:
        primary = h.chunk
        score = float(h.score)
        window: list[str] = [primary.id]

        pid = primary.prev_id
        for _ in range(radius):
            if not pid or pid not in chunks_by_id:
                break
            window.insert(0, pid)
            pid = chunks_by_id[pid].prev_id

        nid = primary.next_id
        for _ in range(radius):
            if not nid or nid not in chunks_by_id:
                break
            window.append(nid)
            nid = chunks_by_id[nid].next_id

        try:
            center = window.index(primary.id)
        except ValueError:
            center = 0

        for j, cid in enumerate(window):
            if cid in seen:
                continue
            c = chunks_by_id.get(cid)
            if not c:
                continue
            seen.add(cid)
            dist = abs(j - center)
            sc = score if dist == 0 else score * max(0.9, 0.99 - 0.01 * dist)
            out.append(Hit(chunk=c, score=sc))
            d = c.doc_id or c.source
            if d not in doc_order:
                doc_order.append(d)

    out.sort(
        key=lambda h: (
            doc_order.index(h.chunk.doc_id or h.chunk.source)
            if (h.chunk.doc_id or h.chunk.source) in doc_order
            else 999,
            h.chunk.chunk_index,
        )
    )
    return out


def ingest(
    *,
    docs_dir: Path | None = None,
    index_dir: Path | None = None,
    batch_size: int = 10,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
    also_milvus: bool | None = None,
) -> dict[str, Any]:
    """分块 v2 → 向量化 → 写入 data/kb_index/；可选同步 Milvus Lite。"""
    from agent.llm import get_embeddings

    docs = docs_dir or KB_DOCS
    out_dir = index_dir or KB_INDEX
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks = build_chunks(docs, max_chars=max_chars, overlap_chars=overlap_chars)
    if not chunks:
        raise FileNotFoundError(f"No .md/.txt docs under {docs}")

    emb = get_embeddings()
    base = (os.getenv("OPENAI_BASE_URL") or "").lower()
    cap = 10 if "dashscope" in base else 32
    batch_size = max(1, min(int(batch_size), cap))
    vectors: list[list[float]] = []
    texts = [c.text for c in chunks]
    n_batches = (len(texts) + batch_size - 1) // batch_size
    print(
        f"Embedding {len(texts)} chunks in {n_batches} batches "
        f"(batch_size={batch_size}, max_chars={max_chars}, overlap={overlap_chars})",
        flush=True,
    )
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        n = i // batch_size + 1
        print(f"  [{n}/{n_batches}] {len(batch)} chunks", flush=True)
        vectors.extend(emb.embed_documents(batch))

    with (out_dir / CHUNKS_NAME).open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")

    (out_dir / VECTORS_NAME).write_text(json.dumps(vectors), encoding="utf-8")

    # 默认：KB_BACKEND=milvus 或显式 --also-milvus 时写入 Lite
    if also_milvus is None:
        also_milvus = kb_backend() == "milvus" or (
            os.getenv("MILVUS_ALSO_INGEST", "").strip().lower() in {"1", "true", "yes"}
        )

    milvus_info: dict[str, Any] | None = None
    if also_milvus:
        try:
            from kb.milvus_store import MilvusStore, milvus_uri

            store = MilvusStore(uri=milvus_uri())
            n = store.replace_all(chunks, vectors)
            milvus_info = {"uri": store.uri, "collection": store.collection, "n_rows": n}
            print(f"Milvus Lite upserted {n} rows → {store.uri}", flush=True)
        except Exception as exc:  # noqa: BLE001
            milvus_info = {"error": str(exc)}
            print(f"Milvus Lite ingest failed: {exc}", flush=True)

    kb_gen = 0
    try:
        from agent.cache.tool_cache import bump_kb_generation

        kb_gen = bump_kb_generation()
    except Exception:  # noqa: BLE001
        kb_gen = 0

    meta = {
        "docs_dir": str(docs),
        "n_chunks": len(chunks),
        "n_docs": len(iter_doc_files(docs)),
        "embedding_model": getattr(emb, "model", None),
        "chunking": {
            "version": 2,
            "max_chars": max_chars,
            "overlap_chars": overlap_chars,
            "neighbor_expand": True,
        },
        "kb_backend_default": kb_backend(),
        "milvus": milvus_info,
        "kb_cache_generation": kb_gen,
    }
    (out_dir / META_NAME).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta


class KnowledgeBase:
    def __init__(
        self,
        index_dir: Path | None = None,
        *,
        backend: str | None = None,
    ) -> None:
        self.index_dir = index_dir or KB_INDEX
        self.backend = (backend or kb_backend()).strip().lower()
        self.chunks: list[Chunk] = []
        self.vectors: list[list[float]] = []
        self.chunks_by_id: dict[str, Chunk] = {}
        self._milvus = None
        self._keyword_index = None
        if self.backend == "milvus":
            from kb.milvus_store import MilvusStore, milvus_uri

            self._milvus = MilvusStore(uri=milvus_uri())
            # 混合 / 关键词检索仍依赖本地 chunks.jsonl（与向量同一次 ingest）
            self._load_chunks_only(required=False)
        else:
            self._load_json()

    def _parse_chunk_row(self, d: dict[str, Any]) -> Chunk:
        return Chunk(
            id=d["id"],
            text=d["text"],
            source=d["source"],
            title=d.get("title") or "",
            doc_id=d.get("doc_id") or d["source"],
            chunk_index=int(d.get("chunk_index") or 0),
            prev_id=d.get("prev_id"),
            next_id=d.get("next_id"),
            url=d.get("url") or "",
            crawled_at=d.get("crawled_at") or "",
            overlap_chars=int(d.get("overlap_chars") or 0),
        )

    def _load_chunks_only(self, *, required: bool = True) -> None:
        chunks_path = self.index_dir / CHUNKS_NAME
        if not chunks_path.exists():
            if required:
                raise FileNotFoundError(
                    f"KB chunks missing under {self.index_dir}. Run: python scripts/ingest_kb.py"
                )
            return
        self.chunks = []
        with chunks_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.chunks.append(self._parse_chunk_row(json.loads(line)))
        self.chunks_by_id = {c.id: c for c in self.chunks}

    def _load_json(self) -> None:
        chunks_path = self.index_dir / CHUNKS_NAME
        vectors_path = self.index_dir / VECTORS_NAME
        if not chunks_path.exists() or not vectors_path.exists():
            raise FileNotFoundError(
                f"KB index missing under {self.index_dir}. Run: python scripts/ingest_kb.py"
            )
        self._load_chunks_only(required=True)
        self.vectors = json.loads(vectors_path.read_text(encoding="utf-8"))
        if len(self.chunks) != len(self.vectors):
            raise ValueError("chunks.jsonl and vectors.json length mismatch; re-ingest")

    def _ensure_keyword_index(self):
        if self._keyword_index is not None:
            return self._keyword_index
        if not self.chunks:
            self._load_chunks_only(required=True)
        from kb.keyword import KeywordIndex

        self._keyword_index = KeywordIndex(self.chunks)
        return self._keyword_index

    def _vector_pairs(
        self,
        query: str,
        top_k: int,
        *,
        exclude_ids: set[str] | None,
        exclude_sources: set[str] | None,
    ) -> list[tuple[Chunk, float]]:
        from agent.llm import get_embeddings

        qv = get_embeddings().embed_query(query)
        if self._milvus is not None:
            return self._milvus.search(
                qv,
                top_k=top_k,
                exclude_ids=exclude_ids,
                exclude_sources=exclude_sources,
            )
        scored = [
            Hit(chunk=c, score=_cosine(qv, v))
            for c, v in zip(self.chunks, self.vectors)
        ]
        scored.sort(key=lambda h: h.score, reverse=True)
        primary = take_hits(
            scored,
            top_k=top_k,
            exclude_ids=exclude_ids,
            exclude_sources=exclude_sources,
        )
        return [(h.chunk, h.score) for h in primary]

    def _keyword_pairs(
        self,
        query: str,
        top_k: int,
        *,
        exclude_ids: set[str] | None,
        exclude_sources: set[str] | None,
    ) -> list[tuple[Chunk, float]]:
        idx = self._ensure_keyword_index()
        # 多取再过滤 exclude
        want = max(top_k * 3, top_k)
        raw = idx.search(query, top_k=want)
        skip_ids = exclude_ids or set()
        skip_src = exclude_sources or set()
        out: list[tuple[Chunk, float]] = []
        for c, s in raw:
            if c.id in skip_ids or c.source in skip_src:
                continue
            out.append((c, s))
            if len(out) >= top_k:
                break
        return out

    def _maybe_expand(self, primary: list[Hit], expand_neighbors: int) -> list[Hit]:
        if expand_neighbors <= 0:
            return primary
        if self._milvus is not None and not self.chunks_by_id:
            need: list[str] = []
            for h in primary:
                if h.chunk.prev_id:
                    need.append(h.chunk.prev_id)
                if h.chunk.next_id:
                    need.append(h.chunk.next_id)
            extra = self._milvus.get_by_ids(need)
            by_id = {h.chunk.id: h.chunk for h in primary}
            by_id.update(extra)
            return expand_adjacent_hits(primary, by_id, radius=expand_neighbors)
        return expand_adjacent_hits(primary, self.chunks_by_id, radius=expand_neighbors)

    def search(
        self,
        query: str,
        top_k: int = 4,
        *,
        exclude_ids: set[str] | None = None,
        exclude_sources: set[str] | None = None,
        expand_neighbors: int = 1,
        mode: str | None = None,
        rrf_k: int | None = None,
        candidate_k: int | None = None,
        rerank: str | None = None,
        rerank_candidates: int | None = None,
    ) -> list[Hit]:
        """
        检索 top_k；默认 expand_neighbors=1，把命中块的前后邻块一并返回。

        mode（KB_RETRIEVAL_MODE，默认 vector）：
          vector  — 纯向量（JSON 余弦 / Milvus）
          keyword — 纯 BM25
          hybrid  — 向量 + BM25，RRF 融合

        rerank（KB_RERANK，默认 off）：
          off | lexical | gte — 初检扩召后重排再截 top_k
        """
        from kb.hybrid import (
            candidate_pool,
            convex_fuse,
            fusion_method_from_env,
            hybrid_alpha_from_env,
            hybrid_weights_from_env,
            retrieval_mode_from_env,
            rrf_fuse,
        )
        from kb.rerank import apply_rerank, rerank_candidate_count, rerank_mode_from_env

        if not query.strip():
            return []

        ret_mode = retrieval_mode_from_env(mode)
        rr_mode = rerank_mode_from_env(rerank)
        # 重排开启时初检取更多候选；hybrid 自身也有 pool
        fetch_k = int(top_k)
        if rr_mode != "off":
            fetch_k = rerank_candidate_count(top_k, rerank_candidates)
        hybrid_pool = candidate_pool(fetch_k if rr_mode != "off" else top_k, candidate_k)
        if rr_mode != "off":
            hybrid_pool = max(hybrid_pool, fetch_k)

        if ret_mode == "keyword":
            pairs = self._keyword_pairs(
                query, fetch_k, exclude_ids=exclude_ids, exclude_sources=exclude_sources
            )
        elif ret_mode == "hybrid":
            vec = self._vector_pairs(
                query, hybrid_pool, exclude_ids=exclude_ids, exclude_sources=exclude_sources
            )
            kw = self._keyword_pairs(
                query, hybrid_pool, exclude_ids=exclude_ids, exclude_sources=exclude_sources
            )
            fusion = fusion_method_from_env()
            cut = fetch_k if rr_mode != "off" else top_k
            if fusion == "convex":
                pairs = convex_fuse(vec, kw, alpha=hybrid_alpha_from_env(), top_k=cut)
            else:
                rk = int(rrf_k) if rrf_k is not None else int(os.getenv("KB_RRF_K") or 60)
                w_vec, w_kw = hybrid_weights_from_env()
                pairs = rrf_fuse([vec, kw], rrf_k=rk, top_k=cut, weights=(w_vec, w_kw))
        else:
            pairs = self._vector_pairs(
                query, fetch_k, exclude_ids=exclude_ids, exclude_sources=exclude_sources
            )

        if rr_mode != "off":
            pairs = apply_rerank(query, pairs, top_k=top_k, mode=rr_mode)
        else:
            pairs = list(pairs)[: max(0, top_k)]

        primary = [Hit(chunk=c, score=s) for c, s in pairs]
        return self._maybe_expand(primary, expand_neighbors)


def take_hits(
    scored: list[Hit],
    top_k: int,
    *,
    exclude_ids: set[str] | None = None,
    exclude_sources: set[str] | None = None,
) -> list[Hit]:
    skip_ids = exclude_ids or set()
    skip_src = exclude_sources or set()
    out: list[Hit] = []
    for h in scored:
        if h.chunk.id in skip_ids:
            continue
        if h.chunk.source in skip_src:
            continue
        out.append(h)
        if len(out) >= top_k:
            break
    return out


def format_context(hits: Iterable[Hit]) -> str:
    blocks: list[str] = []
    for i, h in enumerate(hits, start=1):
        c = h.chunk
        meta_bits = [f"source={c.source}", f"title={c.title}", f"score={h.score:.3f}"]
        if c.url:
            meta_bits.append(f"url={c.url}")
        if c.crawled_at:
            meta_bits.append(f"crawled_at={c.crawled_at}")
        if c.prev_id or c.next_id:
            meta_bits.append(f"chunk={c.chunk_index}")
            if c.prev_id:
                meta_bits.append(f"prev={c.prev_id}")
            if c.next_id:
                meta_bits.append(f"next={c.next_id}")
        blocks.append(f"[{i}] " + " ".join(meta_bits) + f"\n{c.text}")
    return "\n\n".join(blocks)


def ask(question: str, *, top_k: int = 4, index_dir: Path | None = None) -> dict[str, Any]:
    from agent.llm import get_llm
    from kb.rewrite import rewrite_query

    rw = rewrite_query(question)
    search_q = rw.rewritten or question
    kb = KnowledgeBase(index_dir=index_dir)
    hits = kb.search(search_q, top_k=top_k)
    context = format_context(hits)
    llm = get_llm(temperature=0)
    prompt = (
        "你是新能源汽车领域助手。只根据【知识库检索结果】回答用户问题。"
        "若检索结果不足，明确说不知道，不要编造。\n"
        "回答末尾用 [1][2]… 标注依据来自哪条检索结果。\n\n"
        f"【知识库检索结果】\n{context}\n\n"
        f"【用户问题】\n{question}\n"
    )
    resp = llm.invoke(prompt)
    answer = resp.content if hasattr(resp, "content") else str(resp)
    return {
        "question": question,
        "query_used": search_q,
        "query_rewrite": rw.as_dict(),
        "answer": answer,
        "hits": [
            {
                "id": h.chunk.id,
                "source": h.chunk.source,
                "title": h.chunk.title,
                "score": round(h.score, 4),
                "url": h.chunk.url,
                "crawled_at": h.chunk.crawled_at,
                "prev_id": h.chunk.prev_id,
                "next_id": h.chunk.next_id,
                "text": h.chunk.text[:240],
            }
            for h in hits
        ],
    }
