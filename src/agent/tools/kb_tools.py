"""Knowledge-base search tool for the S1 single agent."""

from __future__ import annotations

import json

from langchain_core.tools import tool


def _live_kb_search(query: str, top_k: int) -> dict:
    from agent.tools.resilience import run_kb_search
    from kb import KnowledgeBase
    from kb.hybrid import retrieval_mode_from_env
    from kb.rerank import rerank_mode_from_env
    from kb.rewrite import rewrite_query

    rw = rewrite_query(query)
    query_used = rw.rewritten or (query or "").strip()
    retrieval = retrieval_mode_from_env()
    rerank_mode = rerank_mode_from_env()
    k = max(1, min(int(top_k), 8))

    try:
        kb = KnowledgeBase()
    except FileNotFoundError as exc:
        return {
            "error": str(exc),
            "hits": [],
            "degraded": True,
            "tries": 0,
            "fallback": "stop",
            "query_original": rw.original,
            "query_used": query_used,
            "query_rewrite": rw.as_dict(),
            "retrieval_mode": retrieval,
            "rerank": rerank_mode,
        }

    def _search(q: str, kk: int, exclude_ids: set[str], exclude_sources: set[str]):
        return kb.search(
            q,
            top_k=kk,
            exclude_ids=exclude_ids,
            exclude_sources=exclude_sources,
            mode=retrieval,
            rerank=rerank_mode,
        )

    hits, meta = run_kb_search(query_used, k, _search)
    meta = {
        **meta,
        "query_original": rw.original,
        "query_used": query_used,
        "query_rewrite": rw.as_dict(),
        "retrieval_mode": retrieval,
        "rerank": rerank_mode,
    }
    if meta.get("error") and not hits:
        return {"hits": [], **meta}
    return {
        "hits": [
            {
                "id": h.chunk.id,
                "source": h.chunk.source,
                "title": h.chunk.title,
                "score": round(h.score, 4),
                "url": h.chunk.url,
                "crawled_at": h.chunk.crawled_at,
                "chunk_index": h.chunk.chunk_index,
                "prev_id": h.chunk.prev_id,
                "next_id": h.chunk.next_id,
                "text": h.chunk.text,
            }
            for h in hits
        ],
        **meta,
    }


@tool
def kb_search(query: str, top_k: int = 4) -> str:
    """Search the local NEV knowledge base (ingested docs under data/kb).

    Query with topic keywords (company, year, market), not verbs like 分析/梳理.
    Use for facts already in the corpus. Returns JSON hits with source, title, score, text.
    """
    from agent.cache.tool_cache import cache_get, cache_set, make_call_id
    from agent.tools.call_registry import lookup, store

    k = max(1, min(int(top_k), 8))
    call_id = make_call_id("kb_search", query, top_k=k)

    # 1) Same-turn dedup by call_id
    prior = lookup(call_id)
    if prior is not None:
        return json.dumps(prior, ensure_ascii=False)

    # 2) Redis tool cache (cross-turn / hot)
    cached = cache_get("kb_search", call_id, query=query)
    if cached is not None:
        store(call_id, cached)
        return json.dumps(cached, ensure_ascii=False)

    # 3) Live
    payload = _live_kb_search(query, k)
    payload["call_id"] = call_id
    payload["cache_hit"] = False
    payload["dedup_hit"] = False
    store(call_id, payload)
    cache_set("kb_search", call_id, payload, query=query)
    return json.dumps(payload, ensure_ascii=False)
