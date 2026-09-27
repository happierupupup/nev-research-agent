"""Domain gate must refuse before building the LangGraph / LLM."""

from __future__ import annotations

from agent.intent import OUT_OF_SCOPE
from agent.runtime import run_research


def test_out_of_scope_skips_graph(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("out-of-scope must not build_graph")

    monkeypatch.setattr("agent.runtime.build_graph", boom)
    out = run_research("今天上海天气怎么样？")
    assert out["domain"] == "out_of_scope"
    assert out["kb_calls"] == 0
    assert out["web_calls"] == 0
    assert out["total_tokens"] == 0
    assert out["llm_calls"] == 0
    assert OUT_OF_SCOPE in out["final_report"]


def test_capability_returns_intro(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("capability must not build_graph")

    monkeypatch.setattr("agent.runtime.build_graph", boom)
    out = run_research("你会做什么")
    assert out["domain"] == "capability"
    assert out["total_tokens"] == 0
    assert "新能源汽车" in out["final_report"]
    assert "短问答" in out["final_report"]
