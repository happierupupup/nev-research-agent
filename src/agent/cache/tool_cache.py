"""Tool JSON cache: key / TTL / hot (≥3 in 24h or golden seed) / KB generation."""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any

from agent.cache.redis_client import get_redis

# Defaults match docs/usage.md
_DEFAULT_KB_TTL = 21600
_DEFAULT_WEB_TTL = 1800
_DEFAULT_KB_HOT_TTL = 86400
_DEFAULT_WEB_HOT_TTL = 7200
_DEFAULT_HOT_THRESHOLD = 3
_FREQ_WINDOW_SEC = 86400

_PREFIX = "nev"
_GEN_KEY = f"{_PREFIX}:kb_gen"
_STATS = {"hits": 0, "misses": 0, "sets": 0}

# Golden-set seeds always treated as hot (normalized).
_GOLDEN_SEEDS: frozenset[str] = frozenset(
    {
        "比亚迪刀片电池的核心卖点是什么？",
        "2024 年中国新能源乘用车渗透率大概多少？",
        "分析比亚迪 2024 年海外扩张战略及风险评估",
        "梳理宁德时代电池供应链的集中度与潜在风险",
        "对比特斯拉 vs 比亚迪：成本结构、技术路线、市场份额变化",
    }
)

_WS = re.compile(r"\s+")


def reset_cache_stats() -> None:
    _STATS["hits"] = 0
    _STATS["misses"] = 0
    _STATS["sets"] = 0


def cache_hits() -> int:
    return int(_STATS["hits"])


def cache_misses() -> int:
    return int(_STATS["misses"])


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        return default


def normalize_query(query: str) -> str:
    return _WS.sub(" ", (query or "").strip().lower())


def make_call_id(tool: str, query: str, **params: Any) -> str:
    """Stable short id for (tool, query, params) — used by cache keys and call registry."""
    payload = {
        "tool": (tool or "").strip(),
        "q": normalize_query(query),
        "p": {k: params[k] for k in sorted(params)},
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def is_golden_seed(query: str) -> bool:
    n = normalize_query(query)
    return any(normalize_query(s) == n for s in _GOLDEN_SEEDS)


def _ttl(tool: str, *, hot: bool) -> int:
    if tool == "kb_search":
        return _env_int("CACHE_KB_HOT_TTL" if hot else "CACHE_KB_TTL", _DEFAULT_KB_HOT_TTL if hot else _DEFAULT_KB_TTL)
    return _env_int("CACHE_WEB_HOT_TTL" if hot else "CACHE_WEB_TTL", _DEFAULT_WEB_HOT_TTL if hot else _DEFAULT_WEB_TTL)


def _hot_threshold() -> int:
    return _env_int("CACHE_HOT_THRESHOLD", _DEFAULT_HOT_THRESHOLD)


def get_kb_generation(r: Any | None = None) -> int:
    client = r if r is not None else get_redis()
    if client is None:
        return 0
    try:
        raw = client.get(_GEN_KEY)
        return int(raw or 0)
    except Exception:  # noqa: BLE001
        return 0


def bump_kb_generation() -> int:
    """Invalidate all KB tool caches after ingest. No-op without Redis."""
    r = get_redis()
    if r is None:
        return 0
    try:
        return int(r.incr(_GEN_KEY))
    except Exception:  # noqa: BLE001
        return 0


def _value_key(tool: str, call_id: str, *, gen: int | None = None) -> str:
    if tool == "kb_search":
        g = get_kb_generation() if gen is None else int(gen)
        return f"{_PREFIX}:tool:kb:{call_id}:g{g}"
    return f"{_PREFIX}:tool:web:{call_id}"


def _freq_key(tool: str, call_id: str) -> str:
    return f"{_PREFIX}:freq:{tool}:{call_id}"


def _touch_freq(r: Any, tool: str, call_id: str) -> int:
    """Increment 24h frequency counter; returns count after incr (0 on failure)."""
    key = _freq_key(tool, call_id)
    try:
        n = int(r.incr(key))
        if n == 1:
            r.expire(key, _FREQ_WINDOW_SEC)
        return n
    except Exception:  # noqa: BLE001
        return 0


def _is_hot(query: str, freq: int) -> bool:
    return is_golden_seed(query) or freq >= _hot_threshold()


def _payload_ok(tool: str, payload: dict[str, Any]) -> bool:
    """Only cache successful tool JSON (non-empty hits/results, no error field)."""
    if not isinstance(payload, dict):
        return False
    if payload.get("error"):
        return False
    if tool == "kb_search":
        return bool(payload.get("hits"))
    return bool(payload.get("results"))


def cache_get(tool: str, call_id: str, *, query: str = "") -> dict[str, Any] | None:
    r = get_redis()
    if r is None:
        return None
    key = _value_key(tool, call_id)
    try:
        raw = r.get(key)
        if not raw:
            _STATS["misses"] += 1
            return None
        data = json.loads(raw)
        if not isinstance(data, dict):
            _STATS["misses"] += 1
            return None
        freq = _touch_freq(r, tool, call_id)
        _STATS["hits"] += 1
        out = dict(data)
        out["cache_hit"] = True
        out["call_id"] = call_id
        out["cache_hot"] = _is_hot(query, freq)
        return out
    except Exception:  # noqa: BLE001
        return None


def cache_set(tool: str, call_id: str, payload: dict[str, Any], *, query: str = "") -> bool:
    if not _payload_ok(tool, payload):
        return False
    r = get_redis()
    if r is None:
        return False
    key = _value_key(tool, call_id)
    try:
        freq = _touch_freq(r, tool, call_id)
        hot = _is_hot(query, freq)
        ttl = _ttl(tool, hot=hot)
        if ttl <= 0:
            return False
        # Drop turn-local / hit flags; keep evidence JSON only.
        skip = {"cache_hit", "dedup_hit", "cache_hot", "web_used", "web_max"}
        body = {k: v for k, v in payload.items() if k not in skip}
        body["call_id"] = call_id
        r.setex(key, ttl, json.dumps(body, ensure_ascii=False))
        _STATS["sets"] += 1
        return True
    except Exception:  # noqa: BLE001
        return False
