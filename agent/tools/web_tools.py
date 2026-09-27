"""Web search tool — Tavily if configured, else DuckDuckGo, else Bing HTML."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from html import unescape
from urllib.parse import quote_plus

from langchain_core.tools import tool

# primp 2.x only accepts generic names; duckduckgo-search 6.3 still samples chrome_131 etc.
_PRIMP_OK = ("chrome", "safari", "edge", "firefox")
_DDG_DISABLED = False

_BING_BLOCK = re.compile(r'<li class="b_algo".*?</li>', re.S)
_BING_HREF = re.compile(r'<h2[^>]*>\s*<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', re.S)
_BING_P = re.compile(r"<p[^>]*>(.*?)</p>", re.S)
_TAG = re.compile(r"<[^>]+>")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _plain(html: str) -> str:
    return unescape(_TAG.sub("", html or "")).strip()


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


def _ddgs_cls():
    from duckduckgo_search.duckduckgo_search import DDGS as Base

    class DDGS(Base):
        _impersonates = _PRIMP_OK

    return DDGS


def _search_duckduckgo(query: str, max_results: int) -> list[dict]:
    DDGS = _ddgs_cls()
    results: list[dict] = []
    last_exc: Exception | None = None
    with DDGS(timeout=6) as ddgs:
        for backend in ("lite", "html", "api"):
            try:
                items = ddgs.text(query, max_results=max_results, backend=backend) or []
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                continue
            for item in items:
                results.append(
                    {
                        "title": item.get("title") or "",
                        "url": item.get("href") or item.get("link") or "",
                        "snippet": item.get("body") or item.get("snippet") or "",
                        "retrieved_at": _now_iso(),
                        "provider": "duckduckgo",
                    }
                )
            if results:
                return results
    if last_exc:
        raise last_exc
    return results


def parse_bing_html(html: str, max_results: int) -> list[dict]:
    """Extract organic Bing results (no network)."""
    out: list[dict] = []
    for block in _BING_BLOCK.findall(html or ""):
        m = _BING_HREF.search(block)
        if not m:
            continue
        url = m.group(1).strip()
        if not url.startswith("http"):
            continue
        title = _plain(m.group(2))
        pm = _BING_P.search(block)
        snippet = _plain(pm.group(1)) if pm else ""
        out.append(
            {
                "title": title,
                "url": url,
                "snippet": snippet,
                "retrieved_at": _now_iso(),
                "provider": "bing",
            }
        )
        if len(out) >= max_results:
            break
    return out


def _search_bing(query: str, max_results: int) -> list[dict]:
    import httpx

    url = f"https://www.bing.com/search?q={quote_plus(query)}"
    resp = httpx.get(
        url,
        timeout=httpx.Timeout(12.0, connect=5.0),
        follow_redirects=True,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        },
    )
    resp.raise_for_status()
    hits = parse_bing_html(resp.text, max_results)
    if not hits:
        raise RuntimeError("bing html parsed 0 results")
    return hits


def _search_public_web(query: str, n: int) -> list[dict]:
    """DDG first; on failure disable it for this process and use Bing."""
    global _DDG_DISABLED
    if not _DDG_DISABLED:
        try:
            hits = _search_duckduckgo(query, n)
            if hits:
                return hits
        except Exception:
            _DDG_DISABLED = True
    return _search_bing(query, n)


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the public web for NEV / auto industry information.

    Prefer this for timely facts not in the local KB (news, official stats,
    company announcements). Returns JSON list with title, url, snippet,
    retrieved_at. Do not call again after a budget-exhausted error.
    """
    from agent.tools.budget import allow_web_search, refund_web_search

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
            hits = _search_public_web(query, n)
    except Exception as exc:  # noqa: BLE001
        refund_web_search()
        return json.dumps(
            {"error": f"{type(exc).__name__}: {exc}", "results": [], "web_used": used - 1},
            ensure_ascii=False,
        )
    if not hits:
        refund_web_search()
        return json.dumps(
            {"error": "web_search returned 0 results", "results": [], "web_used": used - 1},
            ensure_ascii=False,
        )
    return json.dumps({"results": hits, "web_used": used, "web_max": cap}, ensure_ascii=False)
