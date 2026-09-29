"""
Runtime entry used by main.py and evals/run_eval.py.

S1: single LangGraph ReAct agent (kb_search + web_search).
S2: Supervisor → Research → Analyst → Verifier（默认；AGENT_MODE=s1 回退）.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any
from uuid import uuid4

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.s1.graph import (
    build_graph,
    collect_sources,
    extract_tool_steps,
    extract_tool_trace,
    final_text,
    payload_messages,
    progress_events_from_message,
)
from agent.s2.graph import run_s2
from agent.core.tokens import collect_token_usage, flatten_callback_usage, prefer_usage as _prefer
from agent.core.intent import (
    OUT_OF_SCOPE,
    focus_web_query,
    is_nev_query,
    is_report_query,
    sanitize_search_query,
)
from agent.core.intro import INTRO, is_capability_query
from agent.memory import (
    FOLLOWUP_UNCLEAR,
    PREFERENCE_ACK,
    SessionMemory,
    UserMemoryStore,
    extract_preferences_from_text,
    format_memory_block,
    looks_like_followup,
    resolve_followup_query,
)
from agent.tools.budget import reset_web_budget, web_used
from agent.tools.kb_tools import kb_search
from agent.tools.web_tools import web_search
from evals.metrics import pad_inline_citations

prefer_usage = _prefer

_DEFAULT_RECURSION = 25


def _agent_mode() -> str:
    return (os.getenv("AGENT_MODE") or "s2").strip().lower()

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


def _empty_trace() -> dict[str, Any]:
    return {"tool_calls": [], "kb_calls": 0, "web_calls": 0, "retrieval_rounds": 0}


def _short_circuit(text: str, domain: str) -> dict[str, Any]:
    return {
        "final_report": text,
        "retrieval_rounds": 0,
        "iteration_count": 0,
        "tool_trace": _empty_trace(),
        "tool_steps": [],
        "sources": [],
        "kb_calls": 0,
        "web_calls": 0,
        "web_attempts": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "llm_calls": 0,
        "domain": domain,
    }


def _refused(text: str) -> dict[str, Any]:
    return _short_circuit(text, "out_of_scope")


def _compose_user_message(q: str, memory_block: str) -> str:
    if not memory_block:
        return q
    return (
        "【会话记忆，不是证据。禁止把其中数字或结论写成 (来源: …)。"
        "本轮事实必须来自工具返回。】\n"
        f"{memory_block}\n\n"
        f"## 本轮问题\n{q}"
    )


def run_research(
    query: str,
    *,
    recursion_limit: int | None = None,
    on_event: ProgressCallback | None = None,
    session: SessionMemory | None = None,
    user_memory: UserMemoryStore | None = None,
) -> dict[str, Any]:
    """
    Run end-to-end research. Returns at least:
      final_report, retrieval_rounds, iteration_count, tool_trace,
      tool_steps, sources
    """
    q = (query or "").strip()
    if not q:
        raise ValueError("query is empty")

    if is_capability_query(q):
        return _short_circuit(INTRO.strip(), "capability")

    prefs = extract_preferences_from_text(q)
    if prefs and session is not None:
        session.add_preferences(prefs)
    if prefs and not is_nev_query(q) and not looks_like_followup(q):
        return _short_circuit(PREFERENCE_ACK, "preference")

    if not is_nev_query(q):
        if looks_like_followup(q) and session and session.recent_entities():
            pass
        elif looks_like_followup(q):
            return _short_circuit(FOLLOWUP_UNCLEAR, "followup_unclear")
        else:
            return _refused(OUT_OF_SCOPE)

    report = is_report_query(q)
    reset_web_budget(3 if report else 1)

    search_seed = resolve_followup_query(q, session)
    kb_q = sanitize_search_query(search_seed)
    web_q = focus_web_query(search_seed)
    memory_block = format_memory_block(session, user_memory, q)

    if _agent_mode() != "s1":
        out = _run_s2(
            q,
            kb_q=kb_q,
            memory_block=memory_block,
            on_event=on_event,
        )
        if session is not None:
            session.remember_from_result(q, out)
        return out

    return _run_s1(
        q,
        report=report,
        kb_q=kb_q,
        web_q=web_q,
        memory_block=memory_block,
        recursion_limit=recursion_limit,
        on_event=on_event,
        session=session,
    )


def _run_s2(
    q: str,
    *,
    kb_q: str,
    memory_block: str,
    on_event: ProgressCallback | None,
) -> dict[str, Any]:
    final = run_s2(
        q,
        search_query=kb_q,
        memory_block=memory_block,
        on_event=on_event,
    )
    steps = list(final.get("tool_steps") or [])
    sources = list(final.get("sources") or [])
    usage = collect_token_usage(final.get("usage_messages") or [])
    kb_calls = int(final.get("kb_calls") or 0)
    web_calls = int(final.get("web_calls") or web_used())
    trace = {
        "tool_calls": [s.get("tool") for s in steps if s.get("tool")],
        "kb_calls": kb_calls,
        "web_calls": sum(1 for s in steps if s.get("tool") == "web_search"),
        "retrieval_rounds": len(steps),
    }
    return {
        "final_report": final.get("final_report") or "",
        "retrieval_rounds": trace["retrieval_rounds"],
        "iteration_count": int(final.get("iteration_count") or 1),
        "tool_trace": trace,
        "tool_steps": steps,
        "sources": sources,
        "kb_calls": kb_calls,
        "web_calls": web_calls,
        "web_attempts": trace["web_calls"],
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "total_tokens": usage["total_tokens"],
        "llm_calls": usage["llm_calls"],
        "domain": "nev",
        "search_query": kb_q,
        "memory_injected": bool(memory_block),
        "agent_mode": "s2",
        "research_plan": final.get("research_plan"),
        "verification_issues": final.get("verification_issues") or [],
        "current_phase": final.get("current_phase"),
    }


def _run_s1(
    q: str,
    *,
    report: bool,
    kb_q: str,
    web_q: str,
    memory_block: str,
    recursion_limit: int | None,
    on_event: ProgressCallback | None,
    session: SessionMemory | None,
) -> dict[str, Any]:
    user_text = _compose_user_message(q, memory_block)
    limit = recursion_limit or int(os.getenv("AGENT_RECURSION_LIMIT", _DEFAULT_RECURSION))
    graph = build_graph()
    usage_cb = UsageMetadataCallbackHandler()
    config = {"recursion_limit": limit, "callbacks": [usage_cb]}
    messages: list[Any] = []
    seq = 0
    iterations = 1

    seq = _stream(
        graph,
        {"messages": [("user", user_text)]},
        config,
        messages,
        seq,
        on_event,
    )
    if not messages:
        result = graph.invoke({"messages": [("user", user_text)]}, config=config)
        messages = list(result.get("messages") or [])

    trace = extract_tool_trace(messages)
    if _needs_tools(trace, report=report):
        iterations = 2
        nudge = _NUDGE_REPORT if report else _NUDGE_QA
        if on_event:
            on_event({"kind": "tool_call", "step": "nudge", "tool": "system", "query": "强制补检索"})
        seq = _stream(
            graph,
            {"messages": [("user", user_text), *messages, HumanMessage(content=nudge)]},
            config,
            messages,
            seq,
            on_event,
        )
        trace = extract_tool_trace(messages)

    forced = False
    if int(trace.get("kb_calls") or 0) < 1:
        raw = kb_search.invoke({"query": kb_q})
        messages.extend(_forced_pair("kb_search", kb_q, raw))
        _emit(on_event, {"kind": "tool_call", "step": "forced", "tool": "kb_search", "query": kb_q})
        _emit(on_event, {"kind": "tool_result", "tool": "kb_search", "preview": (raw or "")[:240]})
        forced = True
    if report and int(extract_tool_trace(messages).get("web_calls") or 0) < 1:
        raw = web_search.invoke({"query": web_q})
        messages.extend(_forced_pair("web_search", web_q, raw))
        _emit(on_event, {"kind": "tool_call", "step": "forced", "tool": "web_search", "query": web_q})
        _emit(on_event, {"kind": "tool_result", "tool": "web_search", "preview": (raw or "")[:240]})
        forced = True
    if forced:
        iterations += 1
        seq = _stream(
            graph,
            {
                "messages": [
                    ("user", user_text),
                    *messages,
                    HumanMessage(
                        content=(
                            "系统已写入真实工具返回。请只根据这些 Tool 结果输出规定 Markdown。"
                            "每个关键事实后必须标注 (来源: KB/文件名.md) 或工具里出现的 (来源: https://...)。"
                            "## 回答 里至少 3 处文内标注（同一文件也要每处都标）。"
                            "## 信息来源 至少列 3 条。不要再调用工具；不要编造工具未返回的数字或链接。"
                            "不要引用会话记忆作为来源。"
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
    out = {
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
        "domain": "nev",
        "search_query": kb_q,
        "memory_injected": bool(memory_block),
        "agent_mode": "s1",
    }
    if session is not None:
        session.remember_from_result(q, out)
    return out
