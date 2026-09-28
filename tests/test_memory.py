"""STM/LTM unit tests (no live LLM)."""

from __future__ import annotations

import json

from agent.memory import (
    FOLLOWUP_UNCLEAR,
    INJECT_MAX_CHARS,
    PREFERENCE_ACK,
    SessionMemory,
    UserMemoryStore,
    extract_session_interests,
    format_memory_block,
    make_digest,
    resolve_followup_query,
)
from agent.runtime import run_research


def _nev_out(report: str, source: str = "byd_blade_battery.md") -> dict:
    return {
        "domain": "nev",
        "final_report": report,
        "sources": [{"type": "KB", "ref": f"KB/{source}", "label": source}],
    }


class _EmptyGraph:
    def stream(self, *_a, **_k):
        return []

    def invoke(self, *_a, **_k):
        return {"messages": []}


def test_m1_followup_query_includes_entities():
    session = SessionMemory()
    session.remember_from_result(
        "比亚迪刀片电池的核心卖点是什么？",
        _nev_out("## 回答\n刀片电池主打安全与空间利用率。\n## 信息来源\n1. [KB] x"),
    )
    q = resolve_followup_query("它主要用在哪些车上", session)
    assert "刀片" in q or "比亚迪" in q
    assert not q.startswith("它") or "比亚迪" in q


def test_m2_followup_without_history(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("no graph")

    monkeypatch.setattr("agent.runtime.build_graph", boom)
    monkeypatch.setattr("agent.runtime.run_s2", boom)
    out = run_research("它主要用在哪些车上")
    assert out["domain"] == "followup_unclear"
    assert FOLLOWUP_UNCLEAR in out["final_report"]
    assert out["kb_calls"] == 0
    assert out["total_tokens"] == 0


def test_m3_interests_persist_across_store(tmp_path):
    session = SessionMemory()
    session.remember_from_result(
        "分析比亚迪 2024 年海外扩张战略及风险评估",
        _nev_out("## 回答\n出海以本地化建厂为主。", "byd_overseas_2024.md"),
    )
    terms = [t for t, _ in extract_session_interests(session)]
    assert "比亚迪" in terms
    assert "海外" in terms
    path = tmp_path / "u.json"
    store = UserMemoryStore.load("alice", path=path)
    store.ingest_session(session)
    store.save()
    loaded = UserMemoryStore.load("alice", path=path)
    block = loaded.format_ltm("比亚迪海外")
    assert "比亚迪" in block
    assert "海外" in block or "扩张" in block


def test_m4_explicit_short_preference(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("no graph")

    monkeypatch.setattr("agent.runtime.build_graph", boom)
    monkeypatch.setattr("agent.runtime.run_s2", boom)
    session = SessionMemory()
    out = run_research("以后短答", session=session)
    assert out["domain"] == "preference"
    assert PREFERENCE_ACK in out["final_report"]
    store = UserMemoryStore("bob")
    store.ingest_session(session)
    ltm = store.format_ltm("比亚迪")
    assert "短答" in ltm
    assert session.pending_preferences


def test_m5_window_caps_injection_length():
    session = SessionMemory()
    long_body = "## 回答\n" + ("出海本地化建厂与供应链协同。" * 80)
    for i in range(4):
        session.remember_from_result(
            f"分析比亚迪海外扩张第{i}轮",
            _nev_out(long_body, "byd_overseas_2024.md"),
        )
    assert len(session.turns) <= 3
    assert session.inject_len() <= INJECT_MAX_CHARS
    block = format_memory_block(session, None, "下一问")
    assert len(block) <= INJECT_MAX_CHARS
    assert "出海" in session.running_summary or session.turns


def test_digest_strips_numbers():
    d = make_digest("## 回答\n零售渗透率约47.6%。\n## 信息来源\n1. x", "渗透率")
    assert "47.6" not in d


def test_m1_forced_search_uses_resolved_query(monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "s1")
    captured: dict[str, str] = {}

    class FakeKb:
        def invoke(self, args):
            captured["q"] = (args or {}).get("query") or ""
            return json.dumps({"hits": []}, ensure_ascii=False)

    monkeypatch.setattr("agent.runtime.build_graph", lambda *_a, **_k: _EmptyGraph())
    monkeypatch.setattr("agent.runtime.kb_search", FakeKb())
    session = SessionMemory()
    session.remember_from_result(
        "比亚迪刀片电池的核心卖点是什么？",
        _nev_out("## 回答\n安全与空间。"),
    )
    out = run_research("它主要用在哪些车上", session=session)
    assert out["domain"] == "nev"
    assert "刀片" in (out.get("search_query") or "") or "比亚迪" in (out.get("search_query") or "")
    assert "刀片" in captured.get("q", "") or "比亚迪" in captured.get("q", "")


def test_eval_default_has_no_memory_injection(monkeypatch):
    monkeypatch.setenv("AGENT_MODE", "s1")

    class FakeKb:
        def invoke(self, _args):
            return json.dumps({"hits": []})

    monkeypatch.setattr("agent.runtime.build_graph", lambda *_a, **_k: _EmptyGraph())
    monkeypatch.setattr("agent.runtime.kb_search", FakeKb())
    out = run_research("比亚迪刀片电池的核心卖点是什么？")
    assert out["memory_injected"] is False
    assert out["search_query"].startswith("比亚迪")
