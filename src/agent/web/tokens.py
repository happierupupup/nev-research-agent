"""简易内存 token（本地 Gradio 登录态，非生产）。"""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass

from agent.auth import User, resolve_user

_LOCK = threading.RLock()
_TTL_SEC = 60 * 60 * 12  # 12h


@dataclass
class AuthSession:
    token: str
    user_id: str
    created_at: float
    last_seen: float


_SESSIONS: dict[str, AuthSession] = {}


def issue_token(user: User) -> str:
    token = secrets.token_urlsafe(24)
    now = time.time()
    with _LOCK:
        _SESSIONS[token] = AuthSession(
            token=token, user_id=user.id, created_at=now, last_seen=now
        )
    return token


def revoke_token(token: str | None) -> None:
    if not token:
        return
    with _LOCK:
        _SESSIONS.pop(token, None)


def resolve_token(token: str | None) -> User | None:
    if not token:
        return None
    with _LOCK:
        sess = _SESSIONS.get(token)
        if sess is None:
            return None
        if time.time() - sess.last_seen > _TTL_SEC:
            _SESSIONS.pop(token, None)
            return None
        sess.last_seen = time.time()
        uid = sess.user_id
    return resolve_user(uid)
