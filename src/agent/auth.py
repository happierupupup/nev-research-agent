"""本地测试用户 / 角色（admin | user）。非生产鉴权。"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml

from agent.paths import USERS_FILE

Role = Literal["admin", "user"]


@dataclass(frozen=True)
class User:
    id: str
    name: str
    role: Role
    note: str = ""


_BUILTIN: tuple[User, ...] = (
    User(id="admin", name="管理员", role="admin", note="测试管理员"),
    User(id="alice", name="普通用户", role="user", note="测试普通用户"),
)


def _normalize_role(raw: str | None) -> Role:
    r = (raw or "user").strip().lower()
    return "admin" if r in {"admin", "administrator", "root"} else "user"


@lru_cache(maxsize=4)
def load_users(path: str | None = None) -> dict[str, User]:
    """从 config/users.yaml 加载；缺文件则用内置两名测试用户。"""
    p = Path(path) if path else USERS_FILE
    users: dict[str, User] = {u.id: u for u in _BUILTIN}
    if p.exists():
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        for item in data.get("users") or []:
            if not isinstance(item, dict):
                continue
            uid = str(item.get("id") or "").strip()
            if not uid:
                continue
            users[uid] = User(
                id=uid,
                name=str(item.get("name") or uid),
                role=_normalize_role(str(item.get("role") or "user")),
                note=str(item.get("note") or ""),
            )
    return users


def resolve_user(user_id: str | None) -> User:
    """未知 id → 按普通用户处理（安全默认）。"""
    uid = (user_id or "alice").strip() or "alice"
    users = load_users()
    if uid in users:
        return users[uid]
    return User(id=uid, name=uid, role="user", note="未登记用户，按普通用户")


def is_admin(user: User | str | None) -> bool:
    if isinstance(user, User):
        return user.role == "admin"
    return resolve_user(user).role == "admin"
