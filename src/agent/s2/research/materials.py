"""Parse tool JSON into research_workspace material cards."""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, ToolMessage


def tool_pair(name: str, query: str, content: str) -> tuple[AIMessage, ToolMessage]:
    tid = f"s2-{name}-{uuid4().hex[:8]}"
    ai = AIMessage(
        content="",
        tool_calls=[{"name": name, "id": tid, "args": {"query": query}}],
    )
    tm = ToolMessage(content=content, name=name, tool_call_id=tid)
    return ai, tm


def materials_from_tool(name: str, query: str, raw: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return [
            {
                "type": "KB" if name == "kb_search" else "Web",
                "query": query,
                "text": raw[:1200],
                "source": "",
                "title": "",
                "error": True,
            }
        ]
    if name == "kb_search":
        for h in data.get("hits") or []:
            items.append(
                {
                    "type": "KB",
                    "query": query,
                    "source": h.get("source") or "",
                    "title": h.get("title") or "",
                    "text": (h.get("text") or "")[:1200],
                    "score": h.get("score"),
                    "ref": f"KB/{h.get('source')}" if h.get("source") else "KB",
                }
            )
        if data.get("error"):
            items.append(
                {
                    "type": "KB",
                    "query": query,
                    "error": True,
                    "text": str(data.get("error")),
                    "source": "",
                    "title": "",
                }
            )
        return items
    for r in data.get("results") or []:
        url = r.get("url") or ""
        items.append(
            {
                "type": "Web",
                "query": query,
                "source": url,
                "title": r.get("title") or "",
                "text": (r.get("content") or r.get("snippet") or "")[:1200],
                "provider": r.get("provider") or data.get("provider"),
                "ref": url,
            }
        )
    if data.get("error"):
        items.append(
            {
                "type": "Web",
                "query": query,
                "error": True,
                "text": str(data.get("error")),
                "source": "",
                "title": "",
            }
        )
    return items


def format_workspace(workspace: list[dict[str, Any]], *, max_body: int = 900) -> str:
    if not workspace:
        return "(无素材)"
    blocks: list[str] = []
    for i, m in enumerate(workspace, 1):
        typ = m.get("type") or "?"
        ref = m.get("ref") or m.get("source") or ""
        title = m.get("title") or ""
        head = f"[{i}] {typ} | {title} | {ref}".strip(" |")
        body = (m.get("text") or "")[:max_body]
        err = " ERROR" if m.get("error") else ""
        blocks.append(f"{head}{err}\n{body}")
    return "\n\n".join(blocks)
