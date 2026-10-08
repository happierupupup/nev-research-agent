"""角色与 CLI 展示分流。"""

from __future__ import annotations

from agent.auth import is_admin, load_users, resolve_user
from agent.cli import _print_event_user
from kb.crawl import augment_query_with_dates


def test_two_seed_users():
    users = load_users()
    assert users["admin"].role == "admin"
    assert users["alice"].role == "user"
    assert is_admin("admin")
    assert not is_admin("alice")
    assert resolve_user("unknown_x").role == "user"


def test_user_thinking_hides_tool_detail(capsys):
    seen: set[str] = set()
    _print_event_user({"kind": "tool_call", "tool": "kb_search", "query": "secret"}, seen)
    _print_event_user(
        {"kind": "tool_result", "tool": "kb_search", "preview": "不该出现"}, seen
    )
    out = capsys.readouterr().out
    assert "正在检索知识库" in out
    assert "secret" not in out
    assert "不该出现" not in out


def test_augment_query_dates():
    assert augment_query_with_dates("比亚迪", date_from="2026-01-01", date_to="2026-06-30") == (
        "比亚迪 2026-01-01..2026-06-30"
    )
    assert augment_query_with_dates("刀片", date_from="2026-01-01") == "刀片 2026-01-01"
