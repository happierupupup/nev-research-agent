"""Redis client with fail-open behaviour (no URL / down → None)."""

from __future__ import annotations

import os
from typing import Any

_client: Any | None = None
_tried = False
# Sentinel: force None (tests). Distinct from "no override" (fall through to env).
_FORCE_NONE = object()
_override: Any = None


def set_redis_override(client: Any | None) -> None:
    """Test hook: inject FakeRedis, or pass None to force fail-open (ignore REDIS_URL)."""
    global _override, _client, _tried
    _override = _FORCE_NONE if client is None else client
    _client = None
    _tried = False


def clear_redis_override() -> None:
    """Remove test override; next get_redis() uses env REDIS_URL again."""
    global _override, _client, _tried
    _override = None
    _client = None
    _tried = False


def reset_redis_client() -> None:
    """Drop cached connection (next get_redis reconnects)."""
    global _client, _tried
    _client = None
    _tried = False


def get_redis() -> Any | None:
    """Return a live Redis client, or None if unavailable (fail-open)."""
    global _client, _tried
    if _override is _FORCE_NONE:
        return None
    if _override is not None:
        return _override
    if _tried:
        return _client
    _tried = True
    url = (os.getenv("REDIS_URL") or "").strip()
    if not url:
        _client = None
        return None
    try:
        import redis  # type: ignore

        r = redis.Redis.from_url(
            url,
            decode_responses=True,
            socket_connect_timeout=0.5,
            socket_timeout=1.0,
        )
        r.ping()
        _client = r
        return _client
    except Exception:  # noqa: BLE001
        _client = None
        return None
