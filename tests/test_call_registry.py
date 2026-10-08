"""Per-turn call_id dedup registry."""

from __future__ import annotations

import json

from agent.cache.tool_cache import make_call_id
from agent.tools.call_registry import dedup_hits, lookup, reset_call_registry, store


def setup_function() -> None:
    reset_call_registry()


def test_same_call_id_returns_prior():
    cid = make_call_id("kb_search", "刀片电池", top_k=4)
    store(cid, {"hits": [{"id": "1"}], "call_id": cid})
    got = lookup(cid)
    assert got is not None
    assert got["dedup_hit"] is True
    assert got["hits"][0]["id"] == "1"
    assert dedup_hits() == 1


def test_different_params_miss():
    a = make_call_id("kb_search", "q", top_k=4)
    b = make_call_id("kb_search", "q", top_k=5)
    store(a, {"hits": [{"id": "a"}]})
    assert lookup(b) is None
    assert lookup(a) is not None


def test_reset_clears():
    cid = make_call_id("web_search", "q", max_results=5)
    store(cid, {"results": [{"url": "https://x"}]})
    reset_call_registry()
    assert lookup(cid) is None
    assert dedup_hits() == 0


def test_store_accepts_json_string():
    cid = make_call_id("kb_search", "s", top_k=4)
    store(cid, json.dumps({"hits": [{"id": "z"}]}, ensure_ascii=False))
    got = lookup(cid)
    assert got["hits"][0]["id"] == "z"
