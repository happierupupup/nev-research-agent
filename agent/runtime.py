"""
Runtime entry used by main.py and evals/run_eval.py.

S1: single LangGraph ReAct agent (kb_search + web_search).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any
from uuid import uuid4

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.graph import (
    build_graph,
    collect_sources,
    extract_tool_steps,
    extract_tool_trace,
    final_text,
    payload_messages,
    progress_events_from_message,
)
from agent.intent import is_report_query
from agent.tokens import collect_token_usage, flatten_callback_usage, prefer_usage
from agent.tools.budget import reset_web_budget, web_used
from agent.tools.kb_tools import kb_search
from agent.tools.web_tools import web_search
from evals.metrics import pad_inline_citations

_DEFAULT_RECURSION = 25

ProgressCallback = Callable[[dict[str, Any]], None]

_NUDGE_QA = (
    "系统校验：你还没有真正调用 kb_search。请立刻调用 kb_search，"
    "禁止用记忆编造，禁止假装已经检索。知识库足够则不要 web_search。"
)
_NUDGE_REPORT = (
    "系统校验：研究命题必须真实调用工具：先 kb_search，再针对性 web_search，"
    "两者至少各一次。禁止假装已检索；不要编造未返回的数字或 URL。"
)


def _stream(
    graph: Any,
    payload: dict[str, Any],
    config: dict[str, Any],
    acc: list[Any],
    seq: int,
    on_event: ProgressCallback | None,
) -> int:
    for update in graph.stream(payload, config=config, stream_mode="updates"):
        if not isinstance(update, dict):
            continue
        for _node, data in update.items():
            for msg in payload_messages(data):
                acc.append(msg)
                if on_event is None:
                    continue
                events, seq = progress_events_from_message(msg, seq)
                for ev in events:
                    on_event(ev)
    return seq


def _forced_pair(name: str, query: str, content: str) -> tuple[AIMessage, ToolMessage]:
    tid = f"forced-{name}-{uuid4().hex[:8]}"
    ai = AIMessage(
        content="",
        tool_calls=[{"name": name, "id": tid, "args": {"query": query}}],
    )
    tm = ToolMessage(content=content, name=name, tool_call_id=tid)
    return ai, tm


def _emit(on_event: ProgressCallback | None, ev: dict[str, Any]) -> None:
    if on_event:
        on_event(ev)


def _needs_tools(trace: dict[str, Any], *, report: bool) -> bool:
    if int(trace.get("kb_calls") or 0) < 1:
        return True
    if report and int(trace.get("web_calls") or 0) < 1:
        return True
    return False


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
    """
    q = (query or "").strip()
    if not q:
        raise ValueError("query is empty")

    report = is_report_query(q)
    reset_web_budget(3 if report else 1)

    limit = recursion_limit or int(os.getenv("AGENT_RECURSION_LIMIT", _DEFAULT_RECURSION))
    graph = build_graph()
    usage_cb = UsageMetadataCallbackHandler()
    config = {"recursion_limit": limit, "callbacks": [usage_cb]}
    messages: list[Any] = []
    seq = 0
    iterations = 1

    seq = _stream(
        graph,
        {"messages": [("user", q)]},
        config,
        messages,
        seq,
        on_event,
    )
    if not messages:
        result = graph.invoke({"messages": [("user", q)]}, config=config)
        messages = list(result.get("messages") or [])

    trace = extract_tool_trace(messages)
    if _needs_tools(trace, report=report):
        iterations = 2
        nudge = _NUDGE_REPORT if report else _NUDGE_QA
        if on_event:
            on_event({"kind": "tool_call", "step": "nudge", "tool": "system", "query": "强制补检索"})
        seq = _stream(
            graph,
            {"messages": [("user", q), *messages, HumanMessage(content=nudge)]},
            config,
            messages,
            seq,
            on_event,
        )
        trace = extract_tool_trace(messages)

    forced = False
    if int(trace.get("kb_calls") or 0) < 1:
        raw = kb_search.invoke({"query": q})
        messages.extend(_forced_pair("kb_search", q, raw))
        _emit(on_event, {"kind": "tool_call", "step": "forced", "tool": "kb_search", "query": q})
        _emit(on_event, {"kind": "tool_result", "preview": (raw or "")[:240]})
        forced = True
    if report and int(extract_tool_trace(messages).get("web_calls") or 0) < 1:
        raw = web_search.invoke({"query": q})
        messages.extend(_forced_pair("web_search", q, raw))
        _emit(on_event, {"kind": "tool_call", "step": "forced", "tool": "web_search", "query": q})
        _emit(on_event, {"kind": "tool_result", "preview": (raw or "")[:240]})
        forced = True
    if forced:
        iterations += 1
        seq = _stream(
            graph,
            {
                "messages": [
                    ("user", q),
                    *messages,
                    HumanMessage(
                        content=(
                            "系统已写入真实工具返回。请只根据这些 Tool 结果输出规定 Markdown。"
                            "每个关键事实后必须标注 (来源: KB/文件名.md) 或工具里出现的 (来源: https://...)。"
                            "## 回答 里至少 3 处文内标注（同一文件也要每处都标）。"
                            "## 信息来源 至少列 3 条。不要再调用工具；不要编造工具未返回的数字或链接。"
                        )
                    ),
                ]
            },
            config,
            messages,
            seq,
            on_event,
        )
        trace = extract_tool_trace(messages)

    steps = extract_tool_steps(messages)
    sources = collect_sources(steps)
    text = final_text(messages)
    if report:
        text = pad_inline_citations(text, sources, min_n=3)
    usage = prefer_usage(
        collect_token_usage(messages),
        flatten_callback_usage(getattr(usage_cb, "usage_metadata", None)),
    )
    return {
        "final_report": text,
        "retrieval_rounds": trace["retrieval_rounds"],
        "iteration_count": iterations,
        "tool_trace": trace,
        "tool_steps": steps,
        "sources": sources,
        "kb_calls": trace["kb_calls"],
        "web_calls": web_used(),
        "web_attempts": trace["web_calls"],
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "total_tokens": usage["total_tokens"],
        "llm_calls": usage["llm_calls"],
    }
