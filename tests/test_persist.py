"""CLI run JSONL persistence (no live LLM)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agent.persist import RunRecorder, run_log_enabled


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NEV_RUN_LOG", raising=False)
    monkeypatch.delenv("NEV_RUN_LOG_TOKENS", raising=False)


def _read_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def test_run_log_enabled_default():
    assert run_log_enabled() is True


def test_run_log_disabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("NEV_RUN_LOG", "0")
    rec = RunRecorder(user_id="alice", query="q", root=tmp_path)
    assert rec.enabled is False
    assert rec.path is None
    rec.record({"kind": "tool_call", "tool": "kb_search"})
    assert list(tmp_path.rglob("*.jsonl")) == []


def test_jsonl_start_events_finish(tmp_path: Path):
    rec = RunRecorder(user_id="alice", query="比亚迪刀片电池？", agent_mode="s2", root=tmp_path)
    assert rec.enabled and rec.path is not None
    rec.record({"kind": "tool_call", "step": "research", "tool": "kb_search", "query": "比亚迪"})
    rec.record(
        {
            "kind": "tool_result",
            "tool": "kb_search",
            "preview": "x" * 800,
        }
    )
    rec.record({"kind": "token", "text": "不应落盘", "step": "analyst"})
    rec.finish(
        {
            "total_tokens": 100,
            "kb_calls": 1,
            "web_calls": 0,
            "sources": [{"type": "KB"}],
            "agent_mode": "s2",
        }
    )
    rows = _read_rows(rec.path)
    kinds = [r["kind"] for r in rows]
    assert kinds[0] == "run_start"
    assert "token" not in kinds
    assert "tool_call" in kinds
    assert "tool_result" in kinds
    assert kinds[-1] == "run_done"
    assert rows[0]["query"] == "比亚迪刀片电池？"
    assert rows[0]["user_id"] == "alice"
    # preview truncated
    tr = next(r for r in rows if r["kind"] == "tool_result")
    assert len(tr["preview"]) == 500
    done = rows[-1]
    assert done["total_tokens"] == 100
    assert done["kb_calls"] == 1
    assert done["n_sources"] == 1
    assert "wall_seconds" in done
    assert "seq" in rows[0] and rows[1]["seq"] == 1


def test_log_tokens_when_enabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("NEV_RUN_LOG_TOKENS", "1")
    rec = RunRecorder(user_id="alice", query="q", root=tmp_path)
    rec.record({"kind": "token", "text": "hi", "step": "analyst"})
    rec.finish({})
    kinds = [r["kind"] for r in _read_rows(rec.path)]
    assert "token" in kinds


def test_fail_writes_run_error(tmp_path: Path):
    rec = RunRecorder(user_id="alice", query="q", root=tmp_path)
    rec.fail(RuntimeError("boom"))
    rows = _read_rows(rec.path)
    assert rows[-1]["kind"] == "run_error"
    assert "boom" in rows[-1]["error"]


def test_fail_open_unwritable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    bad = tmp_path / "blocked"
    bad.mkdir()
    # Make root a file so mkdir under it fails
    file_as_root = tmp_path / "not_a_dir"
    file_as_root.write_text("x", encoding="utf-8")
    rec = RunRecorder(user_id="alice", query="q", root=file_as_root)
    assert rec.enabled is False
    # Must not raise
    rec.record({"kind": "tool_call", "tool": "kb_search"})
    rec.finish({"total_tokens": 1})
    rec.fail("x")


def test_user_id_path_safe(tmp_path: Path):
    rec = RunRecorder(user_id="../evil/name", query="q", root=tmp_path)
    assert ".." not in str(rec.path)
    assert rec.path.parent.name == "___evil_name"
