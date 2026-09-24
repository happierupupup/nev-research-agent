"""CLI self-intro: who the agent is and what it can do."""

from __future__ import annotations

INTRO = """你好，我是新能源汽车行业研究助手。

我可以帮你：
- 短问答：电池技术、车企事实、渗透率、政策口径等
- 研究分析：海外扩张、供应链、竞品对比等，输出带引用的文档
- 检索本地知识库与公开网页；回答会展示思考过程，并标出信息来源

目前只覆盖新能源汽车（整车 / 电池 / 供应链 / 政策 / 竞品）。

可以这样问：
  比亚迪刀片电池的核心卖点是什么？
  2024 年中国新能源乘用车渗透率大概多少？
  分析比亚迪 2024 年海外扩张战略及风险评估

退出：exit / quit / q / 退出
"""

_CAPABILITY_QUERIES = frozenset(
    {
        "你能做什么",
        "你是谁",
        "你会什么",
        "你可以做什么",
        "能做什么",
        "你会干什么",
        "介绍一下",
        "介绍下自己",
        "自我介绍",
        "help",
        "帮助",
        "功能",
    }
)


def is_capability_query(text: str) -> bool:
    t = text.strip().lower().rstrip("。.！!？?")
    return t in _CAPABILITY_QUERIES
