"""Per-turn tool call registry: same call_id returns prior result (no re-invoke)."""

from __future__ import annotations

import json
from typing import Any

from agent.cache.tool_cache import make_call_id

_REGISTRY: dict[str, str] = {}
_STATS = {"hits": 0, "stores": 0}


def reset_call_registry() -> None:
    _REGISTRY.clear()
    _STATS["hits"] = 0
    _STATS["stores"] = 0


def dedup_hits() -> int:
    return int(_STATS["hits"])


def lookup(call_id: str) -> dict[str, Any] | None:
    """Return prior payload for this turn's call_id, or None."""
    raw = _REGISTRY.get(call_id)
    if raw is None:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    _STATS["hits"] += 1
    out = dict(data)
    out["dedup_hit"] = True
    out["call_id"] = call_id
    return out


def store(call_id: str, payload: dict[str, Any] | str) -> None:
    """Remember successful (or any) JSON result for this turn."""
    if isinstance(payload, str):
        raw = payload
        try:
            data = json.loads(payload)
            if not isinstance(data, dict):
                return
        except json.JSONDecodeError:
            return
    else:
        data = dict(payload)
        raw = json.dumps(data, ensure_ascii=False)
    # Don't store hard failures that would poison later retries with different strategies
    # — but identical call_id means identical args, so storing errors is fine to avoid loops.
    _REGISTRY[call_id] = raw
    _STATS["stores"] += 1


def call_id_for(tool: str, query: str, **params: Any) -> str:
    return make_call_id(tool, query, **params)
