"""Research node: tool calls only → research_workspace."""

from __future__ import annotations

from typing import Any

from agent.s2.common import ProgressCallback, emit
from agent.s2.research.materials import materials_from_tool, tool_pair
from agent.s2.supervisor import fallback_plan
from agent.s1.graph import collect_sources, summarize_tool_result
from agent.core.intent import focus_web_query, sanitize_search_query
from agent.core.schemas import ResearchPlan
from agent.tools.budget import web_used
from agent.tools.kb_tools import kb_search
from agent.tools.web_tools import web_search


def run_node(state: dict[str, Any]) -> dict[str, Any]:
    plan: ResearchPlan = state.get("research_plan") or fallback_plan(state["query"])
    on_event: ProgressCallback | None = state.get("_on_event")
    report_mode = bool(state.get("report_mode"))
    need_web = bool(plan.get("need_web"))
    subs = list(plan.get("subqueries") or [sanitize_search_query(state["query"])])

    workspace: list[dict[str, Any]] = list(state.get("research_workspace") or [])
    message_trace: list[Any] = list(state.get("tool_messages") or [])
    steps_meta: list[dict[str, Any]] = list(state.get("tool_steps") or [])

    kb_q = subs[0]
    emit(on_event, {"kind": "tool_call", "step": "research", "tool": "kb_search", "query": kb_q})
    raw_kb = kb_search.invoke({"query": kb_q})
    ai, tm = tool_pair("kb_search", kb_q, raw_kb)
    message_trace.extend([ai, tm])
    workspace.extend(materials_from_tool("kb_search", kb_q, raw_kb))
    preview, sources = summarize_tool_result("kb_search", raw_kb)
    steps_meta.append(
        {
            "step": len(steps_meta) + 1,
            "tool": "kb_search",
            "args": {"query": kb_q},
            "result_preview": preview,
            "sources": sources,
        }
    )
    emit(on_event, {"kind": "tool_result", "tool": "kb_search", "preview": preview, "sources": sources})

    if need_web or report_mode:
        web_q = focus_web_query(subs[1] if len(subs) > 1 else subs[0])
        emit(on_event, {"kind": "tool_call", "step": "research", "tool": "web_search", "query": web_q})
        raw_web = web_search.invoke({"query": web_q})
        ai, tm = tool_pair("web_search", web_q, raw_web)
        message_trace.extend([ai, tm])
        workspace.extend(materials_from_tool("web_search", web_q, raw_web))
        preview, sources = summarize_tool_result("web_search", raw_web)
        steps_meta.append(
            {
                "step": len(steps_meta) + 1,
                "tool": "web_search",
                "args": {"query": web_q},
                "result_preview": preview,
                "sources": sources,
            }
        )
        emit(
            on_event,
            {"kind": "tool_result", "tool": "web_search", "preview": preview, "sources": sources},
        )

    return {
        "research_workspace": workspace,
        "tool_messages": message_trace,
        "tool_steps": steps_meta,
        "sources": collect_sources(steps_meta),
        "kb_calls": sum(1 for s in steps_meta if s.get("tool") == "kb_search"),
        "web_calls": web_used(),
        "current_phase": "analyzing",
        "route": "analyst",
    }
