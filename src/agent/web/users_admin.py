"""Admin CRUD for config/users.yaml（本地测试鉴权，非生产）。"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import yaml

from agent.auth import Role, User, load_users, resolve_user
from agent.paths import USERS_FILE

_LOCK = threading.RLock()


def _normalize_role(raw: str | None) -> Role:
    r = (raw or "user").strip().lower()
    return "admin" if r in {"admin", "administrator", "root"} else "user"


def _read_raw(path: Path | None = None) -> dict[str, Any]:
    p = path or USERS_FILE
    if not p.exists():
        return {"users": []}
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        return {"users": []}
    users = data.get("users") or []
    data["users"] = [u for u in users if isinstance(u, dict)]
    return data


def _write_raw(data: dict[str, Any], path: Path | None = None) -> None:
    p = path or USERS_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
    p.write_text(text, encoding="utf-8")
    load_users.cache_clear()


def list_users_public(*, include_builtin: bool = True) -> list[dict[str, str]]:
    """不含密码。管理页用 include_builtin=False 只展示 yaml 中的用户。"""
    with _LOCK:
        data = _read_raw()
        out: list[dict[str, str]] = []
        seen: set[str] = set()
        for item in data.get("users") or []:
            uid = str(item.get("id") or "").strip()
            if not uid:
                continue
            seen.add(uid)
            out.append(
                {
                    "id": uid,
                    "name": str(item.get("name") or uid),
                    "role": _normalize_role(str(item.get("role") or "user")),
                    "note": str(item.get("note") or ""),
                    "has_password": bool(str(item.get("password") or "").strip()),
                }
            )
        if include_builtin:
            for uid, u in load_users().items():
                if uid in seen:
                    continue
                out.append(
                    {
                        "id": u.id,
                        "name": u.name,
                        "role": u.role,
                        "note": u.note,
                        "has_password": False,
                    }
                )
        out.sort(key=lambda r: (0 if r["role"] == "admin" else 1, r["id"]))
        return out


def get_user_public(user_id: str) -> dict[str, str] | None:
    uid = (user_id or "").strip()
    if not uid:
        return None
    for u in list_users_public(include_builtin=True):
        if u["id"] == uid:
            return u
    return None


def get_password(user_id: str) -> str | None:
    """返回配置的密码；未配置则默认等于 user_id（本地测试）。"""
    uid = (user_id or "").strip()
    if not uid:
        return None
    with _LOCK:
        data = _read_raw()
        for item in data.get("users") or []:
            if str(item.get("id") or "").strip() == uid:
                pw = str(item.get("password") or "").strip()
                return pw if pw else uid
        # yaml 无此用户但 builtin 有 → 默认密码 = id
        users = load_users()
        if uid in users:
            return uid
        return None


def verify_login(user_id: str, password: str) -> User | None:
    uid = (user_id or "").strip()
    if not uid:
        return None
    expected = get_password(uid)
    if expected is None:
        return None
    if (password or "") != expected:
        return None
    return resolve_user(uid)


def upsert_user(
    *,
    user_id: str,
    name: str = "",
    role: str = "user",
    note: str = "",
    password: str | None = None,
) -> dict[str, str]:
    uid = (user_id or "").strip()
    if not uid:
        raise ValueError("用户 id 不能为空")
    with _LOCK:
        data = _read_raw()
        users = list(data.get("users") or [])
        found = False
        for item in users:
            if str(item.get("id") or "").strip() != uid:
                continue
            found = True
            item["name"] = (name or item.get("name") or uid).strip()
            item["role"] = _normalize_role(role)
            if note is not None:
                item["note"] = str(note)
            if password is not None and str(password).strip():
                item["password"] = str(password).strip()
            break
        if not found:
            row: dict[str, Any] = {
                "id": uid,
                "name": (name or uid).strip(),
                "role": _normalize_role(role),
                "note": str(note or ""),
            }
            if password is not None and str(password).strip():
                row["password"] = str(password).strip()
            else:
                row["password"] = uid
            users.append(row)
        data["users"] = users
        # 保留文件头注释风格：只写结构化内容
        _write_raw(data)
        return next(u for u in list_users_public() if u["id"] == uid)


def delete_user(user_id: str) -> bool:
    uid = (user_id or "").strip()
    if not uid:
        return False
    if uid == "admin":
        raise ValueError("不能删除内置管理员 admin")
    with _LOCK:
        data = _read_raw()
        before = len(data.get("users") or [])
        data["users"] = [
            u for u in (data.get("users") or []) if str(u.get("id") or "").strip() != uid
        ]
        if len(data["users"]) == before:
            return False
        _write_raw(data)
        return True
