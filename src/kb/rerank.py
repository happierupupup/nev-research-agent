"""检索重排：初检 topN → 重排取 k。

环境变量 KB_RERANK：
  off（默认）| lexical（本地词面）| gte（DashScope gte-rerank）

KB_RERANK_CANDIDATES：初检条数（默认 max(top_k*4, 32)）
KB_RERANK_MODEL：gte 模型名（默认 gte-rerank-v2）
"""

from __future__ import annotations

import os
import re
from typing import Literal, Sequence

from kb.chunking import Chunk

RerankMode = Literal["off", "lexical", "gte"]

_CJK = re.compile(r"[\u4e00-\u9fff]+")
_LATIN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]*")


def rerank_mode_from_env(override: str | None = None) -> RerankMode:
    raw = (override if override is not None else os.getenv("KB_RERANK") or "off")
    m = str(raw).strip().lower()
    if m in {"off", "lexical", "gte", "api"}:
        if m == "api":
            return "gte"
        return m  # type: ignore[return-value]
    return "off"


def rerank_candidate_count(top_k: int, override: int | None = None) -> int:
    if override is not None and override > 0:
        return max(int(override), int(top_k))
    env = os.getenv("KB_RERANK_CANDIDATES", "").strip()
    if env.isdigit() and int(env) > 0:
        return max(int(env), int(top_k))
    return max(int(top_k) * 4, 32)


def _query_terms(query: str) -> set[str]:
    q = (query or "").lower()
    terms: set[str] = set()
    for m in _LATIN.finditer(q):
        terms.add(m.group(0))
    for m in _CJK.finditer(q):
        s = m.group(0)
        terms.add(s)
        if len(s) >= 2:
            terms.update(s[i : i + 2] for i in range(len(s) - 1))
    return {t for t in terms if t}


def lexical_score(query: str, chunk: Chunk, *, base_score: float = 0.0) -> float:
    """
    轻量重排分：保留初检相对序（base）+ 标题/正文词面命中。
    不调 API，单测与 gte 失败回退用。
    """
    terms = _query_terms(query)
    if not terms:
        return float(base_score)

    title = (chunk.title or "").lower()
    source = (chunk.source or "").lower()
    text = (chunk.text or "").lower()
    blob = f"{title}\n{source}\n{text}"

    hit = 0.0
    title_hit = 0.0
    for t in terms:
        if t in title:
            title_hit += 2.0 if len(t) >= 2 else 1.0
        elif t in source:
            title_hit += 1.0
        if t in blob:
            hit += 1.0 if len(t) >= 2 else 0.4

    # base 归一到 ~[0,1] 量级：向量余弦已在附近；BM25 可能更大，做压缩
    base = float(base_score)
    if base > 1.5:
        base = min(1.0, base / (base + 5.0))
    return base + 0.15 * hit + 0.35 * title_hit


def rerank_lexical(
    query: str,
    hits: Sequence[tuple[Chunk, float]],
    *,
    top_k: int,
) -> list[tuple[Chunk, float]]:
    scored = [(c, lexical_score(query, c, base_score=s)) for c, s in hits]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[: max(0, top_k)]


def _dashscope_api_key() -> str:
    key = (os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("Missing DASHSCOPE_API_KEY / OPENAI_API_KEY for gte rerank")
    return key


def rerank_gte(
    query: str,
    hits: Sequence[tuple[Chunk, float]],
    *,
    top_k: int,
    model: str | None = None,
) -> list[tuple[Chunk, float]]:
    """调用 DashScope 文本排序模型；失败时抛异常由上层回退 lexical。"""
    import httpx

    if not hits:
        return []
    model_name = (model or os.getenv("KB_RERANK_MODEL") or "gte-rerank-v2").strip()
    docs = []
    for c, _ in hits:
        # 标题 + 正文前段，控制请求体
        head = (c.title or "").strip()
        body = (c.text or "").strip()
        if head:
            docs.append(f"{head}\n{body[:1200]}")
        else:
            docs.append(body[:1400])

    url = (
        os.getenv("KB_RERANK_URL")
        or "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
    ).strip()
    payload = {
        "model": model_name,
        "input": {"query": query, "documents": docs},
        "parameters": {
            "return_documents": False,
            "top_n": min(max(1, top_k), len(docs)),
        },
    }
    headers = {
        "Authorization": f"Bearer {_dashscope_api_key()}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    results = (
        (data.get("output") or {}).get("results")
        if isinstance(data, dict)
        else None
    )
    if not isinstance(results, list) or not results:
        raise RuntimeError(f"gte rerank empty response: {str(data)[:200]}")

    out: list[tuple[Chunk, float]] = []
    seen: set[int] = set()
    for item in results:
        if not isinstance(item, dict):
            continue
        idx = int(item.get("index", -1))
        if idx < 0 or idx >= len(hits) or idx in seen:
            continue
        seen.add(idx)
        score = float(item.get("relevance_score") or item.get("score") or 0.0)
        out.append((hits[idx][0], score))
        if len(out) >= top_k:
            break
    if not out:
        raise RuntimeError("gte rerank produced no ranked hits")
    return out


def apply_rerank(
    query: str,
    hits: Sequence[tuple[Chunk, float]],
    *,
    top_k: int,
    mode: RerankMode | str | None = None,
) -> list[tuple[Chunk, float]]:
    """按模式重排并截断 top_k；gte 失败回退 lexical。"""
    m = rerank_mode_from_env(mode if isinstance(mode, str) or mode is None else None)
    if mode in {"off", "lexical", "gte", "api"}:
        m = "gte" if mode == "api" else mode  # type: ignore[assignment]
    if m == "off" or not hits:
        return list(hits)[: max(0, top_k)]
    if m == "lexical":
        return rerank_lexical(query, hits, top_k=top_k)
    try:
        return rerank_gte(query, hits, top_k=top_k)
    except Exception:
        # 网络/配额失败时不丢结果
        return rerank_lexical(query, hits, top_k=top_k)
