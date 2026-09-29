"""S2 agent package unit tests (no live LLM)."""

from __future__ import annotations

from agent.s2.supervisor import decide_node, fallback_plan
from agent.s2.verifier import heuristic_verify
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


def test_heuristic_verify_report_needs_cites():
    issues = heuristic_verify(
        "没有引用的长文" + ("字" * 100),
        report_mode=True,
        workspace=[{"type": "KB", "source": "a.md", "text": "x"}],
    )
    msgs = [i["message"] for i in issues]
    assert any("引用" in m for m in msgs)


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
