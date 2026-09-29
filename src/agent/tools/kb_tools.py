"""Knowledge-base search tool for the S1 single agent."""

from __future__ import annotations

import json

from langchain_core.tools import tool


@tool
def kb_search(query: str, top_k: int = 4) -> str:
    """Search the local NEV knowledge base (ingested docs under data/kb).

    Query with topic keywords (company, year, market), not verbs like 分析/梳理.
    Use for facts already in the corpus. Returns JSON hits with source, title, score, text.
    """
    from agent.core.intent import sanitize_search_query
    from agent.tools.resilience import run_kb_search
    from kb import KnowledgeBase

    query = sanitize_search_query(query)

    k = max(1, min(int(top_k), 8))
    try:
        kb = KnowledgeBase()
    except FileNotFoundError as exc:
        return json.dumps(
            {"error": str(exc), "hits": [], "degraded": True, "tries": 0, "fallback": "stop"},
            ensure_ascii=False,
        )

    def _search(q: str, kk: int, exclude_ids: set[str], exclude_sources: set[str]):
        return kb.search(
            q,
            top_k=kk,
            exclude_ids=exclude_ids,
            exclude_sources=exclude_sources,
        )

    hits, meta = run_kb_search(query, k, _search)
    meta = {**meta, "query_used": query}
    if meta.get("error") and not hits:
        return json.dumps({"hits": [], **meta}, ensure_ascii=False)
    payload = {
        "hits": [
            {
                "id": h.chunk.id,
                "source": h.chunk.source,
                "title": h.chunk.title,
                "score": round(h.score, 4),
                "text": h.chunk.text,
            }
            for h in hits
        ],
        **meta,
    }
    return json.dumps(payload, ensure_ascii=False)
