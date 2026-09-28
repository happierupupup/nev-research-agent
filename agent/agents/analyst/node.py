"""Analyst node: synthesize Markdown from research_workspace."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agent.agents.analyst.prompts import SYSTEM
from agent.agents.common import ProgressCallback, emit
from agent.agents.research import format_workspace
from agent.llm import get_llm
from agent.schemas import issues_to_prompt
from evals.metrics import pad_inline_citations


def run_node(state: dict[str, Any]) -> dict[str, Any]:
    on_event: ProgressCallback | None = state.get("_on_event")
    query = state["query"]
    workspace = state.get("research_workspace") or []
    issues = state.get("verification_issues") or []
    memory_block = state.get("memory_block") or ""
    usage_msgs: list[Any] = list(state.get("usage_messages") or [])
    iteration = int(state.get("iteration_count") or 0) + 1

    emit(on_event, {"kind": "tool_call", "step": "analyst", "tool": "synthesize", "query": query})
    materials = format_workspace(workspace)
    fix = ""
    if issues:
        fix = (
            "\n\n## Verifier Issue List（必须逐条处理 critical）\n"
            + issues_to_prompt(issues)
            + "\n只根据已有素材修正；不要编造新来源。"
        )
    mem = ""
    if memory_block:
        mem = (
            "\n\n## 会话记忆（非证据）\n"
            f"{memory_block}\n禁止把记忆写成 (来源: …)。\n"
        )
    human = (
        f"## 命题\n{query}\n\n## 研究素材（Research 产出，勿编造）\n{materials}"
        f"{mem}{fix}\n\n请输出规定 Markdown。"
    )
    llm = get_llm(temperature=0)
    msg = llm.invoke([SystemMessage(content=SYSTEM), HumanMessage(content=human)])
    usage_msgs.append(msg)
    text = str(getattr(msg, "content", "") or "").strip()
    if bool(state.get("report_mode")):
        text = pad_inline_citations(text, state.get("sources") or [], min_n=3)
    emit(on_event, {"kind": "answer_ready"})
    return {
        "final_report": text,
        "analysis_workspace": {"draft": text, "iteration": iteration},
        "iteration_count": iteration,
        "usage_messages": usage_msgs,
        "current_phase": "verifying",
        "route": "verifier",
    }
