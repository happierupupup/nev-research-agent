"""Chat session persistence: messages, tool events, task progress, STM."""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from agent.memory import SessionMemory, Turn, user_id_from_env
from agent.paths import SESSIONS_DIR

_LOCK = threading.RLock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _title_from_query(query: str) -> str:
    t = (query or "").strip().replace("\n", " ")
    return (t[:36] + "…") if len(t) > 36 else (t or "新会话")


@dataclass
class ChatMessage:
    role: str  # user | assistant
    content: str
    thinking_lines: list[str] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ChatMessage:
        return cls(
            role=str(raw.get("role") or "assistant"),
            content=str(raw.get("content") or ""),
            thinking_lines=list(raw.get("thinking_lines") or []),
            events=list(raw.get("events") or []),
            sources=list(raw.get("sources") or []),
            meta=dict(raw.get("meta") or {}),
            created_at=str(raw.get("created_at") or _now_iso()),
        )


@dataclass
class ChatSession:
    id: str
    user_id: str
    title: str = "新会话"
    status: str = "idle"  # idle | running | done | error
    task_progress: str = ""
    messages: list[ChatMessage] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    session_memory: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "title": self.title,
            "status": self.status,
            "task_progress": self.task_progress,
            "messages": [m.to_dict() for m in self.messages],
            "events": self.events,
            "session_memory": self.session_memory,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ChatSession:
        return cls(
            id=str(raw.get("id") or uuid4().hex),
            user_id=str(raw.get("user_id") or "local"),
            title=str(raw.get("title") or "新会话"),
            status=str(raw.get("status") or "idle"),
            task_progress=str(raw.get("task_progress") or ""),
            messages=[ChatMessage.from_dict(m) for m in (raw.get("messages") or []) if isinstance(m, dict)],
            events=list(raw.get("events") or []),
            session_memory=dict(raw.get("session_memory") or {}),
            created_at=str(raw.get("created_at") or _now_iso()),
            updated_at=str(raw.get("updated_at") or _now_iso()),
            error=str(raw.get("error") or ""),
        )

    def load_memory(self) -> SessionMemory:
        raw = self.session_memory or {}
        sm = SessionMemory(
            running_summary=str(raw.get("running_summary") or ""),
            queries=list(raw.get("queries") or []),
            pending_episodes=list(raw.get("pending_episodes") or []),
            pending_preferences=list(raw.get("pending_preferences") or []),
        )
        for t in raw.get("turns") or []:
            if not isinstance(t, dict):
                continue
            sm.turns.append(
                Turn(
                    query=str(t.get("query") or ""),
                    digest=str(t.get("digest") or ""),
                    entities=list(t.get("entities") or []),
                    cited_refs=list(t.get("cited_refs") or []),
                    kind=str(t.get("kind") or "qa"),
                )
            )
        return sm

    def dump_memory(self, sm: SessionMemory) -> None:
        self.session_memory = {
            "running_summary": sm.running_summary,
            "queries": list(sm.queries),
            "pending_episodes": list(sm.pending_episodes),
            "pending_preferences": list(sm.pending_preferences),
            "turns": [asdict(t) for t in sm.turns],
        }


class SessionStore:
    """File-backed per-user chat sessions under data/sessions/{user_id}/."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or SESSIONS_DIR

    def _user_dir(self, user_id: str) -> Path:
        uid = user_id_from_env(user_id)
        d = self.root / uid
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _path(self, user_id: str, session_id: str) -> Path:
        return self._user_dir(user_id) / f"{session_id}.json"

    def create(self, user_id: str, title: str = "新会话") -> ChatSession:
        with _LOCK:
            sid = uuid4().hex[:12]
            sess = ChatSession(id=sid, user_id=user_id_from_env(user_id), title=title or "新会话")
            self.save(sess)
            return sess

    def save(self, sess: ChatSession) -> None:
        with _LOCK:
            sess.updated_at = _now_iso()
            path = self._path(sess.user_id, sess.id)
            path.write_text(json.dumps(sess.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    def get(self, user_id: str, session_id: str) -> ChatSession | None:
        with _LOCK:
            path = self._path(user_id, session_id)
            if not path.exists():
                return None
            raw = json.loads(path.read_text(encoding="utf-8"))
            return ChatSession.from_dict(raw)

    def list(self, user_id: str) -> list[dict[str, Any]]:
        with _LOCK:
            d = self._user_dir(user_id)
            rows: list[dict[str, Any]] = []
            for path in d.glob("*.json"):
                try:
                    raw = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                rows.append(
                    {
                        "id": raw.get("id") or path.stem,
                        "title": raw.get("title") or "新会话",
                        "status": raw.get("status") or "idle",
                        "task_progress": raw.get("task_progress") or "",
                        "updated_at": raw.get("updated_at") or "",
                        "created_at": raw.get("created_at") or "",
                        "message_count": len(raw.get("messages") or []),
                    }
                )
            rows.sort(key=lambda r: r.get("updated_at") or "", reverse=True)
            return rows

    def delete(self, user_id: str, session_id: str) -> bool:
        with _LOCK:
            path = self._path(user_id, session_id)
            if not path.exists():
                return False
            path.unlink()
            return True

    def set_title_from_query(self, sess: ChatSession, query: str) -> None:
        if sess.title in {"", "新会话"} or sess.title.startswith("新会话"):
            sess.title = _title_from_query(query)


STORE = SessionStore()
