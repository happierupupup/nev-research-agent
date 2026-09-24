"""Web search tool — Tavily if configured, else DuckDuckGo (no key)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from langchain_core.tools import tool


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _search_tavily(query: str, max_results: int) -> list[dict]:
    from tavily import TavilyClient

    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    raw = client.search(query, max_results=max_results)
    results = []
    for item in raw.get("results") or []:
        results.append(
            {
                "title": item.get("title") or "",
                "url": item.get("url") or "",
                "snippet": item.get("content") or item.get("snippet") or "",
                "retrieved_at": _now_iso(),
                "provider": "tavily",
            }
        )
    return results


def _search_duckduckgo(query: str, max_results: int) -> list[dict]:
    from duckduckgo_search import DDGS

    results = []
    with DDGS() as ddgs:
        for item in ddgs.text(query, max_results=max_results):
            results.append(
                {
                    "title": item.get("title") or "",
                    "url": item.get("href") or item.get("link") or "",
                    "snippet": item.get("body") or item.get("snippet") or "",
                    "retrieved_at": _now_iso(),
                    "provider": "duckduckgo",
                }
            )
    return results


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the public web for NEV / auto industry information.

    Prefer this for timely facts not in the local KB (news, official stats,
    company announcements). Returns JSON list with title, url, snippet,
    retrieved_at. Do not call again after a budget-exhausted error.
    """
    from agent.tools.budget import allow_web_search

    ok, used, cap = allow_web_search()
    if not ok:
        return json.dumps(
            {
                "error": f"web_search budget exhausted (max={cap})",
                "results": [],
            },
            ensure_ascii=False,
        )

    n = max(1, min(int(max_results), 8))
    try:
        if (os.getenv("TAVILY_API_KEY") or "").strip():
            hits = _search_tavily(query, n)
        else:
            hits = _search_duckduckgo(query, n)
    except Exception as exc:  # noqa: BLE001
        return json.dumps(
            {"error": f"{type(exc).__name__}: {exc}", "results": [], "web_used": used},
            ensure_ascii=False,
        )
    return json.dumps({"results": hits, "web_used": used, "web_max": cap}, ensure_ascii=False)
