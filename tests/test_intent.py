"""Query-type heuristic (no LLM)."""

from __future__ import annotations

from agent.intent import is_report_query


def test_is_report_query():
    assert is_report_query("分析比亚迪 2024 年海外扩张战略及风险评估")
    assert is_report_query("梳理宁德时代电池供应链")
    assert is_report_query("对比特斯拉 vs 比亚迪")
    assert not is_report_query("比亚迪刀片电池的核心卖点是什么？")
    assert not is_report_query("2024 年中国新能源乘用车渗透率大概多少？")
