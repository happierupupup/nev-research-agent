"""Web session store + role-based display helpers."""

from __future__ import annotations

from pathlib import Path

from agent.auth import resolve_user
from agent.display import format_assistant_message, format_event_line
from agent.web.session_store import SessionStore
from agent.web.users_admin import get_password, verify_login


def test_session_persist_roundtrip(tmp_path: Path):
    store = SessionStore(root=tmp_path / "sessions")
    sess = store.create("alice", title="新会话")
    sess.messages  # noqa: B018 — touch
    sm = sess.load_memory()
    sm.queries.append("比亚迪刀片电池")
    sess.dump_memory(sm)
    sess.task_progress = "正在检索知识库…"
    sess.status = "running"
    store.save(sess)

    loaded = store.get("alice", sess.id)
    assert loaded is not None
    assert loaded.task_progress == "正在检索知识库…"
    assert loaded.load_memory().queries == ["比亚迪刀片电池"]
    rows = store.list("alice")
    assert any(r["id"] == sess.id for r in rows)


def test_user_thinking_hides_detail():
    alice = resolve_user("alice")
    seen: set[str] = set()
    line = format_event_line(
        {"kind": "tool_call", "tool": "kb_search", "query": "secret"},
        user=alice,
        seen_labels=seen,
    )
    assert line and "正在检索知识库" in line
    assert "secret" not in line
    hidden = format_event_line(
        {"kind": "tool_result", "tool": "kb_search", "preview": "不该出现"},
        user=alice,
        seen_labels=seen,
    )
    assert hidden is None


def test_admin_sees_tool_detail():
    admin = resolve_user("admin")
    line = format_event_line(
        {"kind": "tool_call", "step": "research", "tool": "kb_search", "query": "刀片"},
        user=admin,
    )
    assert line and "刀片" in line and "kb_search" in line


def test_login_defaults():
    assert verify_login("admin", "admin") is not None
    assert verify_login("admin", "wrong") is None
    pw = get_password("alice")
    assert pw  # yaml 或默认 = id
    assert verify_login("alice", pw) is not None


def test_assistant_message_role_split():
    alice = resolve_user("alice")
    admin = resolve_user("admin")
    user_msg = format_assistant_message(
        "## 回答\nok",
        thinking_lines=["· 正在检索知识库…"],
        user=alice,
        sources=[{"type": "KB", "label": "a.md", "ref": "KB/a.md"}],
        meta={"kb_calls": 1, "web_calls": 0, "total_tokens": 9},
    )
    assert "思考过程" in user_msg
    assert "<details" in user_msg
    assert "tools:" not in user_msg
    admin_msg = format_assistant_message(
        "## 回答\nok",
        thinking_lines=["[research] kb_search ← q"],
        user=admin,
        sources=[],
        meta={"kb_calls": 1, "web_calls": 0, "total_tokens": 9},
    )
    assert "思考 / 工具" in admin_msg
    assert "<details" in admin_msg
    assert "tools:" in admin_msg
