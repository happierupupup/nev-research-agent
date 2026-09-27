"""CLI intro helpers (no LLM)."""

from __future__ import annotations

from agent.intro import INTRO, is_capability_query


def test_intro_explains_capabilities():
    assert "新能源汽车" in INTRO
    assert "短问答" in INTRO
    assert "研究分析" in INTRO
    assert "信息来源" in INTRO


def test_capability_queries():
    assert is_capability_query("你能做什么？")
    assert is_capability_query("你会做什么")
    assert is_capability_query("你会做什么？")
    assert is_capability_query("你能帮我做什么")
    assert is_capability_query("你是谁")
    assert is_capability_query("Help")
    assert is_capability_query("帮助")
    assert not is_capability_query("比亚迪刀片电池的核心卖点是什么？")
    assert not is_capability_query("今天上海天气怎么样？")
