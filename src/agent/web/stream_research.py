"""Shared research stream: yields SSE-shaped events; persists session state."""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator
from typing import Any

from agent.auth import User, is_admin
from agent.display import format_assistant_message, format_event_line
from agent.memory import UserMemoryStore
from agent.runtime import run_research
from agent.web.session_store import STORE, ChatMessage, ChatSession


def _flush_ltm(user_id: str, sess: ChatSession) -> None:
    store = UserMemoryStore.load(user_id)
    sm = sess.load_memory()
    if not sm.queries and not sm.pending_preferences and not sm.pending_episodes:
        return
    store.ingest_session(sm)
    store.save()
    # 已合并的 pending 清空，避免重复 ingest
    sm.pending_episodes.clear()
    sm.pending_preferences.clear()
    sess.dump_memory(sm)


def iter_research_events(
    *,
    user: User,
    session_id: str,
    query: str,
) -> Iterator[dict[str, Any]]:
    """
    Yield dict events:
      kind=status|thinking|delta|done|error
    Persists context / tool events / task progress on the chat session.
    """
    q = (query or "").strip()
    if not q:
        yield {"kind": "error", "message": "问题不能为空"}
        return

    sess = STORE.get(user.id, session_id)
    if sess is None:
        yield {"kind": "error", "message": "会话不存在"}
        return

    admin = is_admin(user)
    seen_labels: set[str] = set()
    thinking_lines: list[str] = []
    raw_events: list[dict[str, Any]] = []
    event_q: queue.Queue[dict[str, Any] | None] = queue.Queue()
    result_box: dict[str, Any] = {}
    error_box: dict[str, BaseException] = {}

    sess.status = "running"
    sess.error = ""
    sess.task_progress = "开始研究…"
    STORE.set_title_from_query(sess, q)
    sess.messages.append(ChatMessage(role="user", content=q))
    STORE.save(sess)

    yield {"kind": "status", "status": "running", "progress": sess.task_progress, "title": sess.title}

    def on_event(ev: dict[str, Any]) -> None:
        event_q.put(ev)

    def worker() -> None:
        try:
            sm = sess.load_memory()
            um = UserMemoryStore.load(user.id)
            out = run_research(q, on_event=on_event, session=sm, user_memory=um)
            sess.dump_memory(sm)
            result_box["out"] = out
        except BaseException as exc:  # noqa: BLE001
            error_box["exc"] = exc
        finally:
            event_q.put(None)

    th = threading.Thread(target=worker, name=f"research-{session_id}", daemon=True)
    th.start()

    while True:
        ev = event_q.get()
        if ev is None:
            break
        raw_events.append(ev)
        sess.events.append({**ev, "ts": ev.get("ts")})
        line = format_event_line(ev, user=user, seen_labels=seen_labels)
        if line:
            thinking_lines.append(line)
            sess.task_progress = line.lstrip("· ").strip()
            STORE.save(sess)
            yield {
                "kind": "thinking",
                "line": line,
                "progress": sess.task_progress,
                "admin": admin,
                "event": ev if admin else {"kind": ev.get("kind"), "tool": ev.get("tool")},
            }
        else:
            # 管理员才透传 tool_result 等无展示行的原始事件
            if admin:
                yield {"kind": "event", "event": ev, "progress": sess.task_progress}

    th.join(timeout=1)

    if "exc" in error_box:
        exc = error_box["exc"]
        sess.status = "error"
        sess.error = str(exc)
        sess.task_progress = f"出错：{exc}"
        STORE.save(sess)
        yield {"kind": "error", "message": str(exc)}
        return

    out = result_box.get("out") or {}
    report = str(out.get("final_report") or "").strip()
    sources = list(out.get("sources") or [])
    meta = {
        "kb_calls": out.get("kb_calls", 0),
        "web_calls": out.get("web_calls", 0),
        "total_tokens": out.get("total_tokens", 0),
        "prompt_tokens": out.get("prompt_tokens", 0),
        "completion_tokens": out.get("completion_tokens", 0),
        "llm_calls": out.get("llm_calls", 0),
        "tool_trace": out.get("tool_trace") or {},
        "agent_mode": out.get("agent_mode"),
    }
    if not thinking_lines:
        thinking_lines = ["(未调用工具)" if admin else "· 正在生成回答…"]

    content = format_assistant_message(
        report,
        thinking_lines=thinking_lines,
        user=user,
        sources=sources,
        meta=meta,
        thinking_open=False,
        stage="已完成",
    )
    sess.messages.append(
        ChatMessage(
            role="assistant",
            content=content,
            thinking_lines=list(thinking_lines),
            events=list(raw_events),
            sources=sources,
            meta=meta,
        )
    )
    sess.status = "done"
    sess.task_progress = "完成"
    sess.error = ""
    STORE.save(sess)
    try:
        _flush_ltm(user.id, sess)
        STORE.save(sess)
    except Exception:  # noqa: BLE001
        pass

    # 仅流式输出正文（思考块由前端 / content 完整消息带折叠）
    chunk_size = 48
    for i in range(0, len(report), chunk_size):
        yield {"kind": "delta", "text": report[i : i + chunk_size]}

    yield {
        "kind": "done",
        "content": content,
        "report": report,
        "sources": sources,
        "meta": meta if admin else {"kb_calls": meta.get("kb_calls"), "web_calls": meta.get("web_calls")},
        "thinking_lines": thinking_lines,
        "status": "done",
        "title": sess.title,
    }


def history_for_gradio(sess: ChatSession | None) -> list[list[str | None]]:
    """Gradio 3 Chatbot 格式：[[user, assistant], ...]。"""
    if sess is None:
        return []
    pairs: list[list[str | None]] = []
    pending_user: str | None = None
    for m in sess.messages:
        if m.role == "user":
            if pending_user is not None:
                pairs.append([pending_user, None])
            pending_user = m.content
        else:
            if pending_user is None:
                pairs.append([None, m.content])
            else:
                pairs.append([pending_user, m.content])
                pending_user = None
    if pending_user is not None:
        pairs.append([pending_user, None])
    return pairs
