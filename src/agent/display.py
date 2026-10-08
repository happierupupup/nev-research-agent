"""角色分流：管理员看完整工具过程，普通用户只看简要思考阶段。"""

from __future__ import annotations

from typing import Any

from agent.auth import User, is_admin

# 普通用户：工具细节折叠成简短阶段提示
USER_THINKING: dict[tuple[str, str | None], str] = {
    ("tool_call", "plan"): "正在理解问题…",
    ("tool_call", "kb_search"): "正在检索知识库…",
    ("tool_call", "web_search"): "正在检索公开信息…",
    ("tool_call", "synthesize"): "正在整理分析…",
    ("tool_call", "verify"): "正在核对要点…",
    ("answer_ready", None): "即将输出回答…",
}


def _normalize_user_key(ev: dict[str, Any]) -> tuple[str, str | None]:
    kind = str(ev.get("kind") or "")
    tool = str(ev.get("tool") or "") if kind == "tool_call" else None
    key: tuple[str, str | None] = (kind, tool if kind == "tool_call" else None)
    if kind == "tool_call" and key not in USER_THINKING:
        step = str(ev.get("step") or "")
        if step == "supervisor" or tool == "plan":
            key = ("tool_call", "plan")
        elif tool in {"kb_search", "web_search", "synthesize", "verify"}:
            key = ("tool_call", tool)
        elif step in {"analyst", "verifier", "research"}:
            key = (
                "tool_call",
                {"analyst": "synthesize", "verifier": "verify", "research": "kb_search"}.get(
                    step, "plan"
                ),
            )
    return key


def format_event_admin(ev: dict[str, Any]) -> str | None:
    kind = ev.get("kind")
    if kind == "tool_call":
        return f"[{ev.get('step')}] {ev.get('tool')} ← {ev.get('query')}"
    if kind == "tool_result":
        preview = ev.get("preview") or ""
        if not preview:
            return None
        tool = ev.get("tool") or ""
        tag = f"[{tool}] " if tool else ""
        return f"    → {tag}{preview}"
    if kind == "answer_ready":
        return "    ✓ 检索结束，正在输出回答"
    return None


def format_event_user(ev: dict[str, Any], seen_labels: set[str]) -> str | None:
    kind = ev.get("kind")
    if kind == "tool_result":
        return None
    key = _normalize_user_key(ev)
    label = USER_THINKING.get(key) or USER_THINKING.get((str(kind), None))
    if not label or label in seen_labels:
        return None
    seen_labels.add(label)
    return f"· {label}"


def format_event_line(
    ev: dict[str, Any],
    *,
    user: User | str | None,
    seen_labels: set[str] | None = None,
) -> str | None:
    if is_admin(user):
        return format_event_admin(ev)
    labels = seen_labels if seen_labels is not None else set()
    return format_event_user(ev, labels)


def _stage_label(lines: list[str], fallback: str = "思考中…") -> str:
    if not lines:
        return fallback
    last = (lines[-1] or "").strip().lstrip("· ").strip()
    return last or fallback


def format_collapsible_thinking(
    lines: list[str],
    *,
    admin: bool,
    stage: str | None = None,
    open_: bool = False,
) -> str:
    """豆包/千问风格：折叠只看当前阶段，展开看详细过程。"""
    if not lines and not stage:
        return ""
    label = (stage or _stage_label(lines)).strip()
    prefix = "思考 / 工具" if admin else "思考过程"
    summary = f"{prefix} · {label}"
    body = "\n".join(lines) if lines else label
    # 转义 details 内反引号冲突：用缩进代码块
    safe = body.replace("```", "'''")
    open_attr = " open" if open_ else ""
    return (
        f"<details{open_attr}>"
        f"<summary style='cursor:pointer;color:#5b6570;user-select:none'>"
        f"💭 {summary}</summary>\n\n"
        f"<pre style='white-space:pre-wrap;font-size:0.85em;opacity:0.9;"
        f"background:#f6f7f9;padding:10px;border-radius:8px;margin:8px 0 4px'>"
        f"{safe}</pre>\n"
        f"</details>\n"
    )


def format_thinking_block(lines: list[str], *, admin: bool) -> str:
    """兼容旧调用：默认折叠。"""
    return format_collapsible_thinking(lines, admin=admin, open_=False)


def format_sources_block(sources: list[dict[str, Any]] | None) -> str:
    if not sources:
        return ""
    parts = ["\n---\n**信息来源**"]
    for i, s in enumerate(sources, start=1):
        typ = s.get("type") or "?"
        label = s.get("label") or ""
        ref = s.get("ref") or ""
        if typ == "KB":
            parts.append(f"{i}. [KB] {label}")
        else:
            line = f"{i}. [Web] {label}"
            if ref and ref != label:
                line += f"  \n   {ref}"
            parts.append(line)
    return "\n".join(parts)


def format_assistant_message(
    report: str,
    *,
    thinking_lines: list[str],
    user: User | str | None,
    sources: list[dict[str, Any]] | None = None,
    meta: dict[str, Any] | None = None,
    thinking_open: bool = False,
    stage: str | None = None,
) -> str:
    admin = is_admin(user)
    parts: list[str] = []
    block = format_collapsible_thinking(
        thinking_lines,
        admin=admin,
        stage=stage or (_stage_label(thinking_lines, "已完成") if thinking_lines else None),
        open_=thinking_open,
    )
    if block:
        parts.append(block)
    parts.append((report or "").strip() or "(无输出)")
    src = format_sources_block(sources)
    if src:
        parts.append(src)
    if admin and meta:
        parts.append(
            f"\n`tools: kb={meta.get('kb_calls', 0)} web={meta.get('web_calls', 0)} "
            f"tokens={meta.get('total_tokens', 0)}`"
        )
    return "\n".join(parts).strip()


def compose_streaming_bubble(
    *,
    thinking_lines: list[str],
    stage: str,
    report: str = "",
    admin: bool,
    thinking_open: bool = True,
) -> str:
    """流式过程中的气泡内容。"""
    think = format_collapsible_thinking(
        thinking_lines, admin=admin, stage=stage, open_=thinking_open
    )
    report = (report or "").strip()
    if think and report:
        return f"{think}\n\n{report}"
    return think or report or "…"
