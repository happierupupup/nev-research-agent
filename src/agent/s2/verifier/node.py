"""Verifier node: heuristic gates + LLM Issue JSON."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agent.s2.common import ProgressCallback, emit
from agent.s2.research import format_workspace
from agent.s2.verifier.prompts import SYSTEM
from agent.llm import get_llm
from agent.core.schemas import Issue, critical_issues, parse_issues
from evals.metrics import count_cited_facts


def heuristic_verify(
    report: str,
    *,
    report_mode: bool,
    workspace: list[dict[str, Any]],
) -> list[Issue]:
    issues: list[Issue] = []
    cites = count_cited_facts(report or "")
    kb_ok = any(m.get("type") == "KB" and not m.get("error") for m in workspace)
    if not kb_ok:
        issues.append(
            {
                "severity": "critical",
                "type": "citation",
                "message": "研究素材中无有效 KB 命中，报告可能缺少可回溯本地依据",
            }
        )
    if report_mode and cites < 3:
        issues.append(
            {
                "severity": "critical",
                "type": "citation",
                "message": f"研报文内引用标注不足（{cites} < 3）",
            }
        )
    if not report_mode and cites < 1 and kb_ok:
        issues.append(
            {
                "severity": "minor",
                "type": "citation",
                "message": "短问答建议至少 1 处 (来源: …) 标注",
            }
        )
    bad = ("来源: 会话", "来源: 记忆", "来源: 本会话")
    if any(b in (report or "") for b in bad):
        issues.append(
            {
                "severity": "critical",
                "type": "citation",
                "message": "把会话/记忆写成了来源标注",
            }
        )
    return issues


def run_node(state: dict[str, Any]) -> dict[str, Any]:
    on_event: ProgressCallback | None = state.get("_on_event")
    report = state.get("final_report") or ""
    workspace = state.get("research_workspace") or []
    report_mode = bool(state.get("report_mode"))
    usage_msgs: list[Any] = list(state.get("usage_messages") or [])

    emit(on_event, {"kind": "tool_call", "step": "verifier", "tool": "verify", "query": "issue list"})
    issues = heuristic_verify(report, report_mode=report_mode, workspace=workspace)
    try:
        llm = get_llm(temperature=0)
        human = (
            f"## 命题\n{state['query']}\n\n## 报告\n{report[:6000]}\n\n"
            f"## 素材索引\n{format_workspace(workspace)[:4000]}\n"
        )
        msg = llm.invoke([SystemMessage(content=SYSTEM), HumanMessage(content=human)])
        usage_msgs.append(msg)
        llm_issues = parse_issues(str(getattr(msg, "content", "") or ""))
        seen = {i.get("message") for i in issues}
        for i in llm_issues:
            if i.get("message") not in seen:
                issues.append(i)
                seen.add(i.get("message"))
    except Exception:  # noqa: BLE001
        pass

    emit(
        on_event,
        {
            "kind": "tool_result",
            "tool": "verify",
            "preview": f"issues={len(issues)} critical={len(critical_issues(issues))}",
        },
    )
    return {
        "verification_issues": issues,
        "usage_messages": usage_msgs,
        "current_phase": "supervising",
        "route": "supervisor_decide",
    }
