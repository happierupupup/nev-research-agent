"""Shared helpers for S2 agents (no role logic)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

ProgressCallback = Callable[[dict[str, Any]], None]


def emit(on_event: ProgressCallback | None, ev: dict[str, Any]) -> None:
    if on_event:
        on_event(ev)
