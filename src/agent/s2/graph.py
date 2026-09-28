"""S2 LangGraph wiring only — role logic lives under agent/s2/{role}/."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from agent.s2.analyst import run_node as analyst_node
from agent.s2.research import run_node as research_node
from agent.s2.supervisor import decide_node as supervisor_decide_node
from agent.s2.supervisor import plan_node as supervisor_plan_node
from agent.s2.verifier import run_node as verifier_node


class S2State(TypedDict, total=False):
    query: str
    search_query: str
    memory_block: str
    report_mode: bool
    research_plan: dict[str, Any]
    research_workspace: list[dict[str, Any]]
    analysis_workspace: dict[str, Any]
    verification_issues: list[dict[str, Any]]
    current_phase: str
    iteration_count: int
    final_report: str
    tool_steps: list[dict[str, Any]]
    tool_messages: list[Any]
    sources: list[dict[str, str]]
    kb_calls: int
    web_calls: int
    usage_messages: list[Any]
    route: str
    _on_event: Any


_ROUTES = {
    "research": "research",
    "analyst": "analyst",
    "verifier": "verifier",
    "supervisor_decide": "supervisor_decide",
    "__end__": END,
}


def _route_after(
    state: S2State,
) -> Literal["research", "analyst", "verifier", "supervisor_decide", "__end__"]:
    route = state.get("route") or "end"
    if route in ("research", "analyst", "verifier", "supervisor_decide"):
        return route  # type: ignore[return-value]
    return "__end__"


def build_s2_graph():
    g = StateGraph(S2State)
    g.add_node("supervisor_plan", supervisor_plan_node)
    g.add_node("research", research_node)
    g.add_node("analyst", analyst_node)
    g.add_node("verifier", verifier_node)
    g.add_node("supervisor_decide", supervisor_decide_node)

    g.add_edge(START, "supervisor_plan")
    for node in (
        "supervisor_plan",
        "research",
        "analyst",
        "verifier",
        "supervisor_decide",
    ):
        g.add_conditional_edges(node, _route_after, _ROUTES)
    return g.compile()


def run_s2(
    query: str,
    *,
    search_query: str = "",
    memory_block: str = "",
    on_event: Any = None,
) -> dict[str, Any]:
    graph = build_s2_graph()
    init: S2State = {
        "query": query,
        "search_query": search_query or query,
        "memory_block": memory_block or "",
        "research_workspace": [],
        "verification_issues": [],
        "tool_steps": [],
        "tool_messages": [],
        "usage_messages": [],
        "sources": [],
        "iteration_count": 0,
        "kb_calls": 0,
        "web_calls": 0,
        "_on_event": on_event,
    }
    return graph.invoke(init)
