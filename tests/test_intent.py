"""Query-type heuristic (no LLM)."""

from __future__ import annotations

from agent.intent import focus_web_query, is_nev_query, is_report_query, sanitize_search_query


def test_is_report_query():
    assert is_report_query("分析比亚迪 2024 年海外扩张战略及风险评估")
    assert is_report_query("梳理宁德时代电池供应链")
    assert is_report_query("对比特斯拉 vs 比亚迪")
    assert not is_report_query("比亚迪刀片电池的核心卖点是什么？")
    assert not is_report_query("2024 年中国新能源乘用车渗透率大概多少？")


def test_is_nev_query_golden():
    assert is_nev_query("比亚迪刀片电池的核心卖点是什么？")
    assert is_nev_query("2024 年中国新能源乘用车渗透率大概多少？")
    assert is_nev_query("分析比亚迪 2024 年海外扩张战略及风险评估")
    assert is_nev_query("梳理宁德时代电池供应链的集中度与潜在风险")
    assert is_nev_query("对比特斯拉 vs 比亚迪：成本结构、技术路线、市场份额变化")
    assert is_nev_query("What is BYD blade battery?")


def test_is_nev_query_rejects_off_domain():
    assert not is_nev_query("今天上海天气怎么样？")
    assert not is_nev_query("用 Python 写一个快速排序")
    assert not is_nev_query("红烧肉怎么做")
    assert not is_nev_query("What is a beverage?")


def test_sanitize_search_query_drops_analyze_verb():
    q = sanitize_search_query("分析比亚迪 2024 年海外扩张战略及风险评估")
    assert not q.startswith("分析")
    assert "比亚迪" in q
    assert "海外扩张" in q
    assert "风险评估" in q
    assert sanitize_search_query("请分析比亚迪刀片电池") == "比亚迪刀片电池"
    assert sanitize_search_query("比亚迪刀片电池的核心卖点是什么？").startswith("比亚迪")


def test_focus_web_query_moves_leading_year():
    q = focus_web_query("2024年中国新能源乘用车渗透率数据")
    assert q.startswith("中国新能源")
    assert q.endswith("2024年")
    assert focus_web_query("比亚迪刀片电池") == "比亚迪刀片电池"
