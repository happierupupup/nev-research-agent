"""Per-turn tool budgets. Module-level so LangChain tool threads share the counter."""

from __future__ import annotations

_STATE = {"used": 0, "max": 8}


def reset_web_budget(max_calls: int) -> None:
    _STATE["max"] = max(0, int(max_calls))
    _STATE["used"] = 0


def allow_web_search() -> tuple[bool, int, int]:
    """Returns (allowed, used_after_if_ok, max). Increments used when allowed."""
    used = _STATE["used"]
    cap = _STATE["max"]
    if used >= cap:
        return False, used, cap
    _STATE["used"] = used + 1
    return True, _STATE["used"], cap


def web_used() -> int:
    return int(_STATE["used"])
