"""Supervisor nodes: plan then decide."""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agent.agents.common import ProgressCallback, emit
from agent.agents.supervisor.prompts import MAX_ANALYST_PASSES, SYSTEM
from agent.intent import is_report_query, sanitize_search_query
from agent.llm import get_llm
from agent.schemas import ResearchPlan, critical_issues
from agent.tools.budget import reset_web_budget


def parse_plan(raw: str, query: str) -> ResearchPlan:
    text = (raw or "").strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.I)
    if m:
        text = m.group(1).strip()
    data: dict[str, Any] = {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                data = {}
    report = is_report_query(query)
    mode = data.get("mode") if data.get("mode") in ("qa", "report") else (
        "report" if report else "qa"
    )
    need_web = bool(data["need_web"]) if "need_web" in data else (mode == "report")
    # 短问答铁律：禁止联网，避免 Supervisor LLM 乱开 web
    if mode == "qa":
        need_web = False
    subs = data.get("subqueries") if isinstance(data.get("subqueries"), list) else []
    cleaned = [sanitize_search_query(str(s)) for s in subs if str(s).strip()]
    cleaned = [s for s in cleaned if s]
    seed = sanitize_search_query(query)
    if not cleaned:
        cleaned = [seed]
        if mode == "report":
            cleaned.append(f"{seed} 风险 关税")
    if mode == "qa":
        cleaned = cleaned[:1]
    focus = [str(x) for x in (data.get("focus") or []) if str(x).strip()]
    return {
        "mode": mode,  # type: ignore[typeddict-item]
        "need_web": need_web,
        "subqueries": cleaned[:4],
        "focus": focus,
    }


def fallback_plan(query: str) -> ResearchPlan:
    report = is_report_query(query)
    seed = sanitize_search_query(query)
    subs = [seed]
    if report:
        subs.append(f"{seed} 风险")
    return {
        "mode": "report" if report else "qa",
        "need_web": report,
        "subqueries": subs,
        "focus": [],
    }


def plan_node(state: dict[str, Any]) -> dict[str, Any]:
    query = state["query"]
    on_event: ProgressCallback | None = state.get("_on_event")
    emit(on_event, {"kind": "tool_call", "step": "supervisor", "tool": "plan", "query": query})
    usage_msgs: list[Any] = list(state.get("usage_messages") or [])
    plan = fallback_plan(query)
    try:
        llm = get_llm(temperature=0)
        msg = llm.invoke(
            [SystemMessage(content=SYSTEM), HumanMessage(content=f"用户命题：{query}")]
        )
        usage_msgs.append(msg)
        plan = parse_plan(str(getattr(msg, "content", "") or ""), query)
    except Exception:  # noqa: BLE001
        plan = fallback_plan(query)

    report_mode = plan.get("mode") == "report"
    reset_web_budget(3 if report_mode else 1)
    emit(
        on_event,
        {
            "kind": "tool_result",
            "tool": "plan",
            "preview": (
                f"mode={plan.get('mode')} web={plan.get('need_web')} "
                f"q={plan.get('subqueries')}"
            ),
        },
    )
    return {
        "research_plan": plan,
        "report_mode": report_mode,
        "current_phase": "researching",
        "iteration_count": int(state.get("iteration_count") or 0),
        "usage_messages": usage_msgs,
        "route": "research",
    }


def decide_node(state: dict[str, Any]) -> dict[str, Any]:
    issues = state.get("verification_issues") or []
    crit = critical_issues(issues)
    iteration = int(state.get("iteration_count") or 0)
    on_event: ProgressCallback | None = state.get("_on_event")
    if crit and iteration < MAX_ANALYST_PASSES:
        emit(
            on_event,
            {
                "kind": "tool_call",
                "step": "supervisor",
                "tool": "rework",
                "query": f"critical={len(crit)} → analyst",
            },
        )
        return {"current_phase": "analyzing", "route": "analyst"}
    emit(
        on_event,
        {
            "kind": "tool_result",
            "tool": "supervisor",
            "preview": f"done iteration={iteration} critical_left={len(crit)}",
        },
    )
    return {"current_phase": "finalizing", "route": "end"}
