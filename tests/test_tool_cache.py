"""Redis tool cache + per-turn call_id dedup (FakeRedis, no live server)."""

from __future__ import annotations

import json
import time
from typing import Any

from agent.cache.redis_client import reset_redis_client, set_redis_override
from agent.cache.tool_cache import (
    bump_kb_generation,
    cache_get,
    cache_hits,
    cache_set,
    get_kb_generation,
    is_golden_seed,
    make_call_id,
    reset_cache_stats,
)
from agent.tools.call_registry import dedup_hits, lookup, reset_call_registry, store


class FakeRedis:
    """Minimal Redis stand-in: get/setex/incr/expire/delete/ping."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.ttl: dict[str, float] = {}

    def ping(self) -> bool:
        return True

    def _alive(self, key: str) -> bool:
        exp = self.ttl.get(key)
        if exp is not None and exp <= time.time():
            self.store.pop(key, None)
            self.ttl.pop(key, None)
            return False
        return key in self.store

    def get(self, key: str) -> str | None:
        if not self._alive(key):
            return None
        return self.store.get(key)

    def setex(self, key: str, ttl: int, value: str) -> bool:
        self.store[key] = value
        self.ttl[key] = time.time() + int(ttl)
        return True

    def incr(self, key: str) -> int:
        if not self._alive(key):
            self.store[key] = "0"
        n = int(self.store.get(key) or "0") + 1
        self.store[key] = str(n)
        return n

    def expire(self, key: str, ttl: int) -> bool:
        if key not in self.store:
            return False
        self.ttl[key] = time.time() + int(ttl)
        return True

    def delete(self, *keys: str) -> int:
        n = 0
        for k in keys:
            if k in self.store:
                self.store.pop(k, None)
                self.ttl.pop(k, None)
                n += 1
        return n


def setup_function() -> None:
    set_redis_override(None)
    reset_redis_client()
    reset_cache_stats()
    reset_call_registry()


def teardown_function() -> None:
    set_redis_override(None)
    reset_redis_client()
    reset_cache_stats()
    reset_call_registry()


def test_fail_open_without_redis():
    assert cache_get("kb_search", "abc") is None
    assert cache_set("kb_search", "abc", {"hits": [{"id": "1"}]}) is False
    assert bump_kb_generation() == 0


def test_cache_roundtrip_and_hit_flag():
    fake = FakeRedis()
    set_redis_override(fake)
    cid = make_call_id("kb_search", "比亚迪刀片电池", top_k=4)
    payload = {"hits": [{"id": "c1", "text": "刀片"}], "query_used": "比亚迪刀片电池"}
    assert cache_set("kb_search", cid, payload, query="比亚迪刀片电池")
    got = cache_get("kb_search", cid, query="比亚迪刀片电池")
    assert got is not None
    assert got["cache_hit"] is True
    assert got["call_id"] == cid
    assert got["hits"][0]["id"] == "c1"
    assert cache_hits() == 1


def test_does_not_cache_errors_or_empty():
    fake = FakeRedis()
    set_redis_override(fake)
    cid = make_call_id("kb_search", "q", top_k=4)
    assert cache_set("kb_search", cid, {"error": "x", "hits": []}) is False
    assert cache_set("web_search", cid, {"results": []}) is False
    assert cache_set("web_search", cid, {"error": "x", "results": [{"url": "u"}]}) is False


def test_kb_generation_bump_invalidates():
    fake = FakeRedis()
    set_redis_override(fake)
    assert get_kb_generation() == 0
    cid = make_call_id("kb_search", "q", top_k=4)
    assert cache_set("kb_search", cid, {"hits": [{"id": "1"}]})
    assert cache_get("kb_search", cid) is not None
    assert bump_kb_generation() == 1
    assert get_kb_generation() == 1
    # Old key under g0; lookup now uses g1 → miss
    assert cache_get("kb_search", cid) is None


def test_golden_seed_recognized():
    assert is_golden_seed("比亚迪刀片电池的核心卖点是什么？")
    assert is_golden_seed("  比亚迪刀片电池的核心卖点是什么？  ")
    assert not is_golden_seed("随便问一句")


def test_hot_threshold_extends_ttl(monkeypatch):
    fake = FakeRedis()
    set_redis_override(fake)
    monkeypatch.setenv("CACHE_HOT_THRESHOLD", "2")
    monkeypatch.setenv("CACHE_KB_TTL", "100")
    monkeypatch.setenv("CACHE_KB_HOT_TTL", "9999")
    q = "宁德时代供应链"
    cid = make_call_id("kb_search", q, top_k=4)
    payload: dict[str, Any] = {"hits": [{"id": "1"}]}
    cache_set("kb_search", cid, payload, query=q)  # freq=1 → ordinary
    key = next(k for k in fake.store if k.startswith("nev:tool:kb:"))
    ttl1 = fake.ttl[key] - time.time()
    cache_set("kb_search", cid, payload, query=q)  # freq=2 → hot
    ttl2 = fake.ttl[key] - time.time()
    assert ttl1 < 200
    assert ttl2 > 5000


def test_call_id_stable_and_dedup():
    a = make_call_id("kb_search", "比亚迪 刀片", top_k=4)
    b = make_call_id("kb_search", "  比亚迪   刀片 ", top_k=4)
    c = make_call_id("kb_search", "比亚迪 刀片", top_k=5)
    assert a == b
    assert a != c

    reset_call_registry()
    store(a, {"hits": [{"id": "1"}]})
    prior = lookup(a)
    assert prior is not None
    assert prior["dedup_hit"] is True
    assert prior["call_id"] == a
    assert dedup_hits() == 1


def test_kb_search_tool_dedup_and_cache(monkeypatch):
    """Live once → Redis hit (new turn) → same-turn dedup (no second live)."""
    from agent.tools import kb_tools

    fake = FakeRedis()
    set_redis_override(fake)
    reset_call_registry()
    reset_cache_stats()

    calls = {"n": 0}

    def fake_live(query: str, top_k: int) -> dict:
        calls["n"] += 1
        return {
            "hits": [{"id": "h1", "text": "ok", "source": "x.md", "score": 0.9}],
            "query_used": query,
            "tries": 1,
        }

    monkeypatch.setattr(kb_tools, "_live_kb_search", fake_live)

    r1 = json.loads(kb_tools.kb_search.invoke({"query": "刀片电池", "top_k": 4}))
    assert calls["n"] == 1
    assert r1.get("call_id")
    assert r1.get("cache_hit") is False
    assert r1.get("dedup_hit") is False

    # New turn: clear registry, keep Redis → cache hit, no live
    reset_call_registry()
    r2 = json.loads(kb_tools.kb_search.invoke({"query": "刀片电池", "top_k": 4}))
    assert calls["n"] == 1
    assert r2.get("cache_hit") is True
    assert cache_hits() >= 1

    # Same turn again → dedup, still no live
    r3 = json.loads(kb_tools.kb_search.invoke({"query": "刀片电池", "top_k": 4}))
    assert calls["n"] == 1
    assert r3.get("dedup_hit") is True
    assert dedup_hits() >= 1


def test_web_search_dedup_skips_budget(monkeypatch):
    from agent.tools import web_tools
    from agent.tools.budget import reset_web_budget, web_used

    reset_call_registry()
    reset_cache_stats()
    set_redis_override(None)
    reset_web_budget(2)

    calls = {"n": 0}

    def fake_run(query: str, n: int):
        calls["n"] += 1
        return (
            [{"title": "t", "url": "https://example.com", "snippet": "s", "provider": "x"}],
            {"fallback": "ok"},
        )

    monkeypatch.setattr(web_tools, "run_web_search", fake_run)

    r1 = json.loads(web_tools.web_search.invoke({"query": "比亚迪 海外", "max_results": 3}))
    assert calls["n"] == 1
    assert web_used() == 1
    assert r1.get("call_id")

    r2 = json.loads(web_tools.web_search.invoke({"query": "比亚迪 海外", "max_results": 3}))
    assert calls["n"] == 1
    assert web_used() == 1  # dedup: no second budget burn
    assert r2.get("dedup_hit") is True
