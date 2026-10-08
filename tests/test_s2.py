"""S2 agent package unit tests (no live LLM)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from agent.s2.supervisor import decide_node, fallback_plan, plan_node
from agent.s2.verifier import heuristic_verify, run_node as verifier_run
from agent.core.schemas import critical_issues, parse_issues


def test_parse_issues_fenced():
    raw = """```json
{"issues": [{"severity": "critical", "type": "citation", "message": "缺来源", "citation": ""}]}
```"""
    issues = parse_issues(raw)
    assert len(issues) == 1
    assert issues[0]["severity"] == "critical"
    assert issues[0]["type"] == "citation"


def test_parse_issues_empty():
    assert parse_issues('{"issues": []}') == []
    assert parse_issues("not json") == []


def test_fallback_plan_qa_vs_report():
    qa = fallback_plan("比亚迪刀片电池的核心卖点是什么？")
    assert qa["mode"] == "qa"
    assert qa["need_web"] is False
    rep = fallback_plan("分析比亚迪 2024 年海外扩张战略及风险评估")
    assert rep["mode"] == "report"
    assert rep["need_web"] is True
    assert len(rep["subqueries"]) >= 2


def test_plan_node_qa_skips_supervisor_llm():
    """短问答不调 Supervisor LLM，且强制 need_web=false。"""
    with patch("agent.s2.supervisor.node.get_llm") as mock_llm:
        out = plan_node({"query": "比亚迪刀片电池的核心卖点是什么？"})
    mock_llm.assert_not_called()
    assert out["report_mode"] is False
    assert out["research_plan"]["need_web"] is False
    assert out["research_plan"]["mode"] == "qa"
    assert out["usage_messages"] == []
    assert out["route"] == "research"


def test_heuristic_verify_report_needs_cites():
    issues = heuristic_verify(
        "没有引用的长文" + ("字" * 100),
        report_mode=True,
        workspace=[{"type": "KB", "source": "a.md", "text": "x"}],
    )
    msgs = [i["message"] for i in issues]
    assert any("引用" in m for m in msgs)


def test_verifier_qa_skips_llm():
    """短问答 Verifier 仅启发式，不调 LLM。"""
    with patch("agent.s2.verifier.node.get_llm") as mock_llm:
        out = verifier_run(
            {
                "query": "刀片电池卖点？",
                "final_report": "刀片电池能量密度高 (来源: byd.md)",
                "research_workspace": [
                    {"type": "KB", "source": "byd.md", "text": "刀片"}
                ],
                "report_mode": False,
                "usage_messages": [],
            }
        )
    mock_llm.assert_not_called()
    assert out["route"] == "supervisor_decide"
    assert out["usage_messages"] == []


def test_verifier_report_calls_llm():
    fake = MagicMock()
    fake.invoke.return_value = MagicMock(content='{"issues": []}')
    with patch("agent.s2.verifier.node.get_llm", return_value=fake):
        out = verifier_run(
            {
                "query": "分析比亚迪海外战略",
                "final_report": "报告 " + ("字" * 80) + " (来源: a) (来源: b) (来源: c)",
                "research_workspace": [{"type": "KB", "source": "a.md", "text": "x"}],
                "report_mode": True,
                "usage_messages": [],
            }
        )
    fake.invoke.assert_called_once()
    assert len(out["usage_messages"]) == 1


def test_supervisor_decide_reworks_on_critical():
    out = decide_node(
        {
            "verification_issues": [
                {"severity": "critical", "type": "citation", "message": "缺来源"}
            ],
            "iteration_count": 1,
        }
    )
    assert out["route"] == "analyst"
    done = decide_node(
        {
            "verification_issues": [
                {"severity": "critical", "type": "citation", "message": "缺来源"}
            ],
            "iteration_count": 2,
        }
    )
    assert done["route"] == "end"
    assert len(critical_issues([{"severity": "critical", "type": "citation", "message": "x"}])) == 1
