"""Lightweight query heuristics (no LLM)."""

from __future__ import annotations

import re

_REPORT_MARKERS = (
    "分析",
    "梳理",
    "对比",
    "评估",
    "战略",
    "风险",
    "研究报告",
    "研报",
)

# Substring markers (Chinese + longer English). Short "ev" uses a word boundary below.
_NEV_MARKERS = (
    "新能源",
    "电动车",
    "电车",
    "电动汽车",
    "混动",
    "插电",
    "增程",
    "刀片电池",
    "动力电池",
    "电池",
    "电芯",
    "磷酸铁锂",
    "三元锂",
    "充电桩",
    "换电",
    "补能",
    "超充",
    "充电",
    "乘用车",
    "商用车",
    "整车",
    "车企",
    "智驾",
    "自动驾驶",
    "反补贴",
    "比亚迪",
    "特斯拉",
    "宁德",
    "理想汽车",
    "蔚来",
    "小鹏",
    "问界",
    "极氪",
    "深蓝",
    "埃安",
    "哪吒",
    "零跑",
    "赛力斯",
    "小米汽车",
)

_NEV_EN_WORD = re.compile(
    r"\b(nev|bev|phev|hev|byd|tesla|catl|lfp|nmc|evs?|oem)\b",
    re.I,
)

OUT_OF_SCOPE = (
    "这个问题不在新能源汽车研究范围内。"
    "我只能回答整车、电池、供应链、政策、竞品相关问题。"
    "可以问例如：比亚迪刀片电池的核心卖点是什么？"
)


def is_report_query(text: str) -> bool:
    t = (text or "").strip()
    return any(m in t for m in _REPORT_MARKERS)


def is_nev_query(text: str) -> bool:
    """True if the question looks like NEV industry research (no LLM)."""
    t = (text or "").strip()
    if not t:
        return False
    low = t.lower()
    if any(m in low for m in _NEV_MARKERS):
        return True
    return bool(_NEV_EN_WORD.search(t))
