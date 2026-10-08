"""查询改写（规则路径，不调 LLM）。"""

from __future__ import annotations

from kb.rewrite import rewrite_mode_from_env, rewrite_query, rewrite_rules


def test_rewrite_strips_task_and_qa():
    r = rewrite_rules("请问分析一下比亚迪刀片电池的核心卖点是什么？")
    assert r.method == "rules"
    assert "分析" not in r.rewritten
    assert "请问" not in r.rewritten
    assert "是什么" not in r.rewritten
    assert "比亚迪" in r.rewritten
    assert "刀片" in r.rewritten


def test_rewrite_expands_alias():
    r = rewrite_rules("宁德电池供应链风险")
    assert "宁德时代" in r.rewritten
    assert any("alias" in n for n in r.notes)


def test_rewrite_byd_english():
    r = rewrite_rules("BYD blade battery advantages")
    assert "比亚迪" in r.rewritten or "刀片电池" in r.rewritten


def test_rewrite_off_passthrough():
    q = "分析比亚迪海外扩张"
    r = rewrite_query(q, mode="off")
    assert r.rewritten == q
    assert r.method == "off"


def test_mode_from_env(monkeypatch):
    monkeypatch.setenv("KB_QUERY_REWRITE", "off")
    assert rewrite_mode_from_env() == "off"
    monkeypatch.setenv("KB_QUERY_REWRITE", "llm")
    assert rewrite_mode_from_env() == "llm"
    monkeypatch.delenv("KB_QUERY_REWRITE", raising=False)
    assert rewrite_mode_from_env() == "rules"


def test_empty_query():
    r = rewrite_rules("  ")
    assert r.rewritten == ""
