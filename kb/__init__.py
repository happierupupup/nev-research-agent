"""Minimal local knowledge base: chunk → embed → search → LLM answer."""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from agent.paths import KB_DOCS, KB_INDEX

META_NAME = "meta.json"
CHUNKS_NAME = "chunks.jsonl"
VECTORS_NAME = "vectors.json"


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    title: str


@dataclass
class Hit:
    chunk: Chunk
    score: float


def _split_markdown(text: str, source: str, max_chars: int = 600) -> list[Chunk]:
    """Split on ## headings; further split long blocks by paragraphs."""
    text = text.strip()
    if not text:
        return []
    parts = re.split(r"(?m)^##\s+", text)
    chunks: list[Chunk] = []
    title = Path(source).stem
    first = parts[0].strip()
    blocks: list[tuple[str, str]] = []
    if first:
        h1 = re.search(r"(?m)^#\s+(.+)$", first)
        if h1:
            title = h1.group(1).strip()
        blocks.append((title, first))
    for part in parts[1:]:
        lines = part.splitlines()
        heading = lines[0].strip() if lines else title
        body = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""
        blocks.append((heading, f"## {heading}\n{body}".strip()))

    n = 0
    for heading, block in blocks:
        if len(block) <= max_chars:
            chunks.append(Chunk(id=f"{source}::{n}", text=block, source=source, title=heading))
            n += 1
            continue
        paras = [p.strip() for p in re.split(r"\n\s*\n", block) if p.strip()]
        buf = ""
        for p in paras:
            if buf and len(buf) + len(p) + 2 > max_chars:
                chunks.append(Chunk(id=f"{source}::{n}", text=buf, source=source, title=heading))
                n += 1
                buf = p
            else:
                buf = f"{buf}\n\n{p}".strip() if buf else p
        if buf:
            chunks.append(Chunk(id=f"{source}::{n}", text=buf, source=source, title=heading))
            n += 1
    return chunks


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


def build_chunks(docs_dir: Path | None = None) -> list[Chunk]:
    out: list[Chunk] = []
    base = docs_dir or KB_DOCS
    for path in iter_doc_files(docs_dir):
        rel = str(path.relative_to(base))
        text = path.read_text(encoding="utf-8")
        out.extend(_split_markdown(text, source=rel))
    return out


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(y * y for y in b)) or 1e-12
    return dot / (na * nb)


def ingest(
    *,
    docs_dir: Path | None = None,
    index_dir: Path | None = None,
    batch_size: int = 10,
) -> dict[str, Any]:
    """Chunk docs, embed, write index under data/kb_index/."""
    from agent.llm import get_embeddings

    docs = docs_dir or KB_DOCS
    out_dir = index_dir or KB_INDEX
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks = build_chunks(docs)
    if not chunks:
        raise FileNotFoundError(f"No .md/.txt docs under {docs}")

    emb = get_embeddings()
    base = (os.getenv("OPENAI_BASE_URL") or "").lower()
    # DashScope compatible embeddings: batch must be ≤20 (some models ≤10).
    cap = 10 if "dashscope" in base else 32
    batch_size = max(1, min(int(batch_size), cap))
    vectors: list[list[float]] = []
    texts = [c.text for c in chunks]
    n_batches = (len(texts) + batch_size - 1) // batch_size
    print(
        f"Embedding {len(texts)} chunks in {n_batches} batches "
        f"(batch_size={batch_size})",
        flush=True,
    )
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        n = i // batch_size + 1
        print(f"  [{n}/{n_batches}] {len(batch)} chunks", flush=True)
        vectors.extend(emb.embed_documents(batch))

    with (out_dir / CHUNKS_NAME).open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

    (out_dir / VECTORS_NAME).write_text(json.dumps(vectors), encoding="utf-8")

    meta = {
        "docs_dir": str(docs),
        "n_chunks": len(chunks),
        "n_docs": len(iter_doc_files(docs)),
        "embedding_model": getattr(emb, "model", None),
    }
    (out_dir / META_NAME).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta


class KnowledgeBase:
    def __init__(self, index_dir: Path | None = None) -> None:
        self.index_dir = index_dir or KB_INDEX
        self.chunks: list[Chunk] = []
        self.vectors: list[list[float]] = []
        self._load()

    def _load(self) -> None:
        chunks_path = self.index_dir / CHUNKS_NAME
        vectors_path = self.index_dir / VECTORS_NAME
        if not chunks_path.exists() or not vectors_path.exists():
            raise FileNotFoundError(
                f"KB index missing under {self.index_dir}. Run: python scripts/ingest_kb.py"
            )
        self.chunks = []
        with chunks_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                self.chunks.append(Chunk(**d))
        self.vectors = json.loads(vectors_path.read_text(encoding="utf-8"))
        if len(self.chunks) != len(self.vectors):
            raise ValueError("chunks.jsonl and vectors.json length mismatch; re-ingest")

    def search(self, query: str, top_k: int = 4) -> list[Hit]:
        from agent.llm import get_embeddings

        if not query.strip():
            return []
        qv = get_embeddings().embed_query(query)
        scored = [
            Hit(chunk=c, score=_cosine(qv, v))
            for c, v in zip(self.chunks, self.vectors)
        ]
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:top_k]


def format_context(hits: Iterable[Hit]) -> str:
    blocks: list[str] = []
    for i, h in enumerate(hits, start=1):
        blocks.append(
            f"[{i}] source={h.chunk.source} title={h.chunk.title} score={h.score:.3f}\n"
            f"{h.chunk.text}"
        )
    return "\n\n".join(blocks)


def ask(question: str, *, top_k: int = 4, index_dir: Path | None = None) -> dict[str, Any]:
    """Retrieve from KB then ask LLM to answer only from context."""
    from agent.llm import get_llm

    kb = KnowledgeBase(index_dir=index_dir)
    hits = kb.search(question, top_k=top_k)
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
        "answer": answer,
        "hits": [
            {
                "id": h.chunk.id,
                "source": h.chunk.source,
                "title": h.chunk.title,
                "score": round(h.score, 4),
                "text": h.chunk.text[:240],
            }
            for h in hits
        ],
    }
