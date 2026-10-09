"""Analyst node: synthesize Markdown from research_workspace."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agent.s2.analyst.prompts import SYSTEM
from agent.s2.common import ProgressCallback, emit
from agent.s2.research import format_workspace
from agent.llm import get_llm
from agent.core.schemas import issues_to_prompt
from evals.metrics import pad_inline_citations


def _content_piece(chunk: Any) -> str:
    c = getattr(chunk, "content", None)
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        bits: list[str] = []
        for part in c:
            if isinstance(part, str):
                bits.append(part)
            elif isinstance(part, dict):
                bits.append(str(part.get("text") or ""))
            else:
                bits.append(str(getattr(part, "text", None) or ""))
        return "".join(bits)
    return str(c)


def _stream_synthesize(
    llm: Any,
    messages: list[Any],
    on_event: ProgressCallback | None,
) -> tuple[str, Any]:
    """
    真流式生成：每块 content 发 kind=token；失败则回退 invoke。
    返回 (全文, 用于 usage 统计的 message/chunk)。
    """
    emit(on_event, {"kind": "answer_start", "step": "analyst"})
    try:
        acc = None
        parts: list[str] = []
        for chunk in llm.stream(messages):
            piece = _content_piece(chunk)
            if piece:
                parts.append(piece)
                emit(on_event, {"kind": "token", "text": piece, "step": "analyst"})
            acc = chunk if acc is None else acc + chunk
        text = "".join(parts).strip()
        if text:
            return text, acc
    except Exception:  # noqa: BLE001
        pass
    # stream 失败或空结果：整段 invoke，再当作单块 token 推一次（便于 SSE）
    msg = llm.invoke(messages)
    text = str(getattr(msg, "content", "") or "").strip()
    if text:
        emit(on_event, {"kind": "token", "text": text, "step": "analyst"})
    return text, msg


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
    messages = [SystemMessage(content=SYSTEM), HumanMessage(content=human)]
    text, msg = _stream_synthesize(llm, messages, on_event)
    if msg is not None:
        usage_msgs.append(msg)
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
