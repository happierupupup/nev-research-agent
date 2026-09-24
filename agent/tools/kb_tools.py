"""Knowledge-base search tool for the S1 single agent."""

from __future__ import annotations

import json

from langchain_core.tools import tool


@tool
def kb_search(query: str, top_k: int = 4) -> str:
    """Search the local NEV knowledge base (ingested docs under data/kb).

    Use for facts already in the corpus (e.g. blade battery demos, sample
    penetration notes). Returns JSON list of hits with source, title, score, text.
    """
    from kb import KnowledgeBase

    try:
        kb = KnowledgeBase()
    except FileNotFoundError as exc:
        return json.dumps({"error": str(exc), "hits": []}, ensure_ascii=False)

    hits = kb.search(query, top_k=max(1, min(int(top_k), 8)))
    payload = {
        "hits": [
            {
                "source": h.chunk.source,
                "title": h.chunk.title,
                "score": round(h.score, 4),
                "text": h.chunk.text,
            }
            for h in hits
        ]
    }
    return json.dumps(payload, ensure_ascii=False)
