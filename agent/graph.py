"""S1 LangGraph: single ReAct agent with kb_search + web_search."""

from __future__ import annotations

import json
from typing import Any

from agent.llm import get_llm
from agent.prompts import SYSTEM_PROMPT
from agent.tools import ALL_TOOLS


def build_graph(temperature: float = 0):
    """Build a compiled LangGraph ReAct agent (single node + tool loop)."""
    from langgraph.prebuilt import create_react_agent

    llm = get_llm(temperature=temperature)
    return create_react_agent(
        model=llm,
        tools=ALL_TOOLS,
        prompt=SYSTEM_PROMPT,
    )


def extract_tool_trace(messages: list[Any]) -> dict[str, Any]:
    """Count tool calls from AIMessage tool_calls / ToolMessage names."""
    names: list[str] = []
    for msg in messages:
        tool_calls = getattr(msg, "tool_calls", None) or []
        for tc in tool_calls:
            if isinstance(tc, dict):
                names.append(tc.get("name") or "unknown")
            else:
                names.append(getattr(tc, "name", None) or "unknown")
    return {
        "tool_calls": names,
        "kb_calls": sum(1 for n in names if n == "kb_search"),
        "web_calls": sum(1 for n in names if n == "web_search"),
        "retrieval_rounds": len(names),
    }


def _tc_field(tc: Any, key: str, default: Any = None) -> Any:
    if isinstance(tc, dict):
        return tc.get(key, default)
    return getattr(tc, key, default)


def payload_messages(payload: Any) -> list[Any]:
    """Pull message list out of a LangGraph stream update payload."""
    if payload is None:
        return []
    if hasattr(payload, "model_dump") and not isinstance(payload, dict):
        try:
            payload = payload.model_dump()
        except Exception:  # noqa: BLE001
            pass
    msgs: Any = None
    if isinstance(payload, dict):
        msgs = payload.get("messages")
    else:
        msgs = getattr(payload, "messages", None)
    if msgs is None:
        if isinstance(payload, list):
            return payload
        return []
    if isinstance(msgs, list):
        return msgs
    return [msgs]


def progress_events_from_message(
    msg: Any, seq: int
) -> tuple[list[dict[str, Any]], int]:
    """Turn one new message into CLI progress events (tool_call / tool_result)."""
    events: list[dict[str, Any]] = []
    for tc in getattr(msg, "tool_calls", None) or []:
        seq += 1
        args = _tc_field(tc, "args") or _tc_field(tc, "arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"raw": args}
        query: Any = args
        if isinstance(args, dict):
            query = args.get("query") or args.get("q") or args
        events.append(
            {
                "kind": "tool_call",
                "step": seq,
                "tool": _tc_field(tc, "name") or "unknown",
                "query": query,
                "call_id": str(_tc_field(tc, "id") or f"call-{seq}"),
            }
        )

    cls_name = msg.__class__.__name__
    if cls_name == "ToolMessage":
        content = getattr(msg, "content", "") or ""
        name = getattr(msg, "name", None) or ""
        preview, sources = _summarize_tool_result(name, content)
        events.append(
            {
                "kind": "tool_result",
                "tool": name or "unknown",
                "preview": preview,
                "sources": sources,
                "call_id": str(getattr(msg, "tool_call_id", "") or ""),
            }
        )
    elif not events and cls_name in {"AIMessage", "AIMessageChunk"}:
        content = getattr(msg, "content", None)
        tool_calls = getattr(msg, "tool_calls", None) or []
        if content and not tool_calls:
            events.append({"kind": "answer_ready"})

    return events, seq


def extract_tool_steps(messages: list[Any]) -> list[dict[str, Any]]:
    """Ordered tool call + result pairs for showing the thinking/retrieval trail."""
    pending: dict[str, dict[str, Any]] = {}
    steps: list[dict[str, Any]] = []
    seq = 0

    for msg in messages:
        for tc in getattr(msg, "tool_calls", None) or []:
            seq += 1
            call_id = str(_tc_field(tc, "id") or f"call-{seq}")
            name = _tc_field(tc, "name") or "unknown"
            args = _tc_field(tc, "args") or _tc_field(tc, "arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"raw": args}
            step = {
                "step": seq,
                "tool": name,
                "args": args,
                "result_preview": "",
                "sources": [],
            }
            pending[call_id] = step
            steps.append(step)

        if msg.__class__.__name__ != "ToolMessage":
            continue
        content = getattr(msg, "content", "") or ""
        call_id = str(getattr(msg, "tool_call_id", "") or "")
        name = getattr(msg, "name", None) or ""
        step = pending.get(call_id)
        if step is None:
            seq += 1
            step = {
                "step": seq,
                "tool": name or "unknown",
                "args": {},
                "result_preview": "",
                "sources": [],
            }
            steps.append(step)
        preview, sources = _summarize_tool_result(step["tool"] or name, content)
        step["result_preview"] = preview
        step["sources"] = sources

    return steps


def _summarize_tool_result(tool: str, content: str) -> tuple[str, list[dict[str, str]]]:
    sources: list[dict[str, str]] = []
    text = content if isinstance(content, str) else str(content)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return (text[:240] + ("…" if len(text) > 240 else ""), sources)

    if tool == "kb_search":
        hits = data.get("hits") or []
        lines = []
        for h in hits[:5]:
            src = h.get("source") or ""
            title = h.get("title") or ""
            score = h.get("score")
            lines.append(f"{src} · {title} (score={score})")
            sources.append(
                {
                    "type": "KB",
                    "label": f"{src}" + (f" / {title}" if title else ""),
                    "ref": f"KB/{src}" if src else "KB",
                }
            )
        if data.get("error"):
            lines.insert(0, f"error: {data['error']}")
        return ("；".join(lines) or "(无命中)", sources)

    if tool == "web_search":
        results = data.get("results") or []
        lines = []
        for r in results[:5]:
            title = r.get("title") or ""
            url = r.get("url") or ""
            lines.append(f"{title} | {url}")
            sources.append(
                {
                    "type": "Web",
                    "label": title or url,
                    "ref": url,
                }
            )
        if data.get("error"):
            lines.insert(0, f"error: {data['error']}")
        return ("；".join(lines) or "(无结果)", sources)

    return (text[:240] + ("…" if len(text) > 240 else ""), sources)


def collect_sources(steps: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Deduplicate sources across tool steps (stable order)."""
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for step in steps:
        for s in step.get("sources") or []:
            key = f"{s.get('type')}|{s.get('ref')}"
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
    return out


def final_text(messages: list[Any]) -> str:
    for msg in reversed(messages):
        content = getattr(msg, "content", None)
        if not content:
            continue
        if msg.__class__.__name__ == "ToolMessage":
            continue
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text") or "")
                elif isinstance(block, str):
                    parts.append(block)
            text = "".join(parts).strip()
            if text:
                return text
    return ""
