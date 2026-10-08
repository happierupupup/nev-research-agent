"""Unit tests for web_search providers and budget refund (no live LLM)."""

from __future__ import annotations

import json

from agent.tools import budget, web_tools
from agent.tools.web_tools import parse_bing_html


_SAMPLE_BING = """
<ol id="b_results">
<li class="b_algo">
  <h2><a href="https://www.byd.com/en/news">BYD Thailand plant</a></h2>
  <p>Rayong factory opened in 2024 with 150,000 capacity.</p>
</li>
<li class="b_algo">
  <h2><a href="/relative/path">skip me</a></h2>
  <p>not a real url</p>
</li>
<li class="b_algo">
  <h2><a href="https://example.com/eu-tariff">EU EV tariff</a></h2>
  <p>Provisional duties on Chinese BEVs.</p>
</li>
</ol>
"""


def test_parse_bing_html_organic_only():
    hits = parse_bing_html(_SAMPLE_BING, max_results=5)
    assert len(hits) == 2
    assert hits[0]["url"] == "https://www.byd.com/en/news"
    assert "Rayong" in hits[0]["snippet"]
    assert hits[1]["provider"] == "bing"


def test_ddgs_client_accepts_primp_impersonate():
    DDGS = web_tools._ddgs_cls()
    client = DDGS(timeout=2)
    assert client is not None


def test_web_search_refunds_budget_on_failure(monkeypatch):
    from agent.cache.redis_client import set_redis_override
    from agent.tools.call_registry import reset_call_registry

    budget.reset_web_budget(3)
    reset_call_registry()
    set_redis_override(None)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setattr(
        web_tools,
        "run_web_search",
        lambda *_a, **_k: ([], {"quality": 0.0, "degraded": True, "fallback": "kb", "attempts": []}),
    )
    out = json.loads(web_tools.web_search.invoke({"query": "BYD"}))
    assert out.get("error")
    assert out.get("results") == []
    assert budget.web_used() == 0


def test_web_search_counts_success(monkeypatch):
    from agent.cache.redis_client import set_redis_override
    from agent.tools.call_registry import reset_call_registry

    budget.reset_web_budget(3)
    reset_call_registry()
    set_redis_override(None)  # force fail-open; ignore live REDIS_URL
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setattr(
        web_tools,
        "run_web_search",
        lambda *_a, **_k: (
            [
                {
                    "title": "t",
                    "url": "https://example.com/a",
                    "snippet": "s",
                    "retrieved_at": "t",
                    "provider": "bing",
                }
            ],
            {"quality": 0.7, "degraded": False, "fallback": None, "attempts": []},
        ),
    )
    out = json.loads(web_tools.web_search.invoke({"query": "BYD"}))
    assert out.get("results")
    assert budget.web_used() == 1
