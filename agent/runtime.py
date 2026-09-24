"""
Runtime entry used by main.py and evals/run_eval.py.

S1: single LangGraph ReAct agent (kb_search + web_search).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from agent.graph import (
    build_graph,
    collect_sources,
    extract_tool_steps,
    extract_tool_trace,
    final_text,
    payload_messages,
    progress_events_from_message,
)

_DEFAULT_RECURSION = 25

ProgressCallback = Callable[[dict[str, Any]], None]


def run_research(
    query: str,
    *,
    recursion_limit: int | None = None,
    on_event: ProgressCallback | None = None,
) -> dict[str, Any]:
    """
    Run end-to-end research. Returns at least:
      final_report, retrieval_rounds, iteration_count, tool_trace,
      tool_steps, sources

    on_event: optional callback for live CLI progress
      {kind: tool_call|tool_result|answer_ready, ...}
    """
    q = (query or "").strip()
    if not q:
        raise ValueError("query is empty")

    limit = recursion_limit or int(os.getenv("AGENT_RECURSION_LIMIT", _DEFAULT_RECURSION))
    graph = build_graph()
    config = {"recursion_limit": limit}
    messages: list[Any] = []
    seq = 0

    for update in graph.stream(
        {"messages": [("user", q)]},
        config=config,
        stream_mode="updates",
    ):
        if not isinstance(update, dict):
            continue
        for _node, payload in update.items():
            for msg in payload_messages(payload):
                messages.append(msg)
                if on_event is None:
                    continue
                events, seq = progress_events_from_message(msg, seq)
                for ev in events:
                    on_event(ev)

    # Fallback if stream yielded nothing (shouldn't happen on a healthy graph)
    if not messages:
        result = graph.invoke({"messages": [("user", q)]}, config=config)
        messages = list(result.get("messages") or [])

    trace = extract_tool_trace(messages)
    steps = extract_tool_steps(messages)
    sources = collect_sources(steps)
    report = final_text(messages)
    return {
        "final_report": report,
        "retrieval_rounds": trace["retrieval_rounds"],
        "iteration_count": 1,
        "tool_trace": trace,
        "tool_steps": steps,
        "sources": sources,
        "kb_calls": trace["kb_calls"],
        "web_calls": trace["web_calls"],
    }
