"""CLI run event persistence (JSONL, fail-open).

Writes under data/runs/{user_id}/{run_id}.jsonl — one event per line.
Does not depend on the web/Gradio package (main branch).
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from agent.paths import RUNS_DIR

# Events never written unless NEV_RUN_LOG_TOKENS=1
_SKIP_KINDS_DEFAULT = frozenset({"token"})


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_log_enabled() -> bool:
    raw = (os.getenv("NEV_RUN_LOG") or "1").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def log_tokens_enabled() -> bool:
    raw = (os.getenv("NEV_RUN_LOG_TOKENS") or "0").strip().lower()
    return raw in {"1", "true", "on", "yes"}


def _safe_user_id(user_id: str) -> str:
    uid = (user_id or "local").strip() or "local"
    # path-safe: keep alnum / _ / -
    cleaned = "".join(c if c.isalnum() or c in "_-" else "_" for c in uid)
    return cleaned[:64] or "local"


class RunRecorder:
    """Append-only JSONL recorder for one research run. Fail-open on IO errors."""

    def __init__(
        self,
        *,
        user_id: str,
        query: str,
        agent_mode: str = "s2",
        root: Path | None = None,
        enabled: bool | None = None,
    ) -> None:
        self.enabled = run_log_enabled() if enabled is None else bool(enabled)
        self.user_id = _safe_user_id(user_id)
        self.query = (query or "").strip()
        self.agent_mode = (agent_mode or "s2").strip().lower() or "s2"
        self.root = root or RUNS_DIR
        self.run_id = uuid4().hex[:12]
        self.path: Path | None = None
        self._seq = 0
        self._t0 = time.monotonic()
        self._log_tokens = log_tokens_enabled()
        if self.enabled:
            self._open()

    def _open(self) -> None:
        try:
            d = self.root / self.user_id
            d.mkdir(parents=True, exist_ok=True)
            self.path = d / f"{self.run_id}.jsonl"
            self._write(
                {
                    "kind": "run_start",
                    "query": self.query[:500],
                    "user_id": self.user_id,
                    "agent_mode": self.agent_mode,
                    "run_id": self.run_id,
                }
            )
        except Exception:  # noqa: BLE001
            self.enabled = False
            self.path = None

    def _write(self, payload: dict[str, Any]) -> None:
        if not self.enabled or self.path is None:
            return
        row = {
            "ts": _now_iso(),
            "seq": self._seq,
            **payload,
        }
        self._seq += 1
        try:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception:  # noqa: BLE001
            self.enabled = False

    def record(self, ev: dict[str, Any] | None) -> None:
        if not self.enabled or not isinstance(ev, dict):
            return
        kind = str(ev.get("kind") or "")
        if kind in _SKIP_KINDS_DEFAULT and not self._log_tokens:
            return
        # Shallow copy; drop oversized free-form blobs if present
        body = {k: v for k, v in ev.items() if k not in {"ts", "seq"}}
        preview = body.get("preview")
        if isinstance(preview, str) and len(preview) > 500:
            body["preview"] = preview[:500]
        text = body.get("text")
        if isinstance(text, str) and len(text) > 500:
            body["text"] = text[:500]
        self._write(body)

    def finish(self, out: dict[str, Any] | None = None) -> None:
        if not self.enabled:
            return
        out = out or {}
        wall = round(time.monotonic() - self._t0, 3)
        sources = out.get("sources") or []
        self._write(
            {
                "kind": "run_done",
                "run_id": self.run_id,
                "wall_seconds": wall,
                "total_tokens": out.get("total_tokens", 0),
                "prompt_tokens": out.get("prompt_tokens", 0),
                "completion_tokens": out.get("completion_tokens", 0),
                "llm_calls": out.get("llm_calls", 0),
                "kb_calls": out.get("kb_calls", 0),
                "web_calls": out.get("web_calls", 0),
                "cache_hits": out.get("cache_hits", 0),
                "n_sources": len(sources) if isinstance(sources, list) else 0,
                "agent_mode": out.get("agent_mode") or self.agent_mode,
                "domain": out.get("domain"),
            }
        )

    def fail(self, exc: BaseException | str) -> None:
        if not self.enabled:
            return
        wall = round(time.monotonic() - self._t0, 3)
        self._write(
            {
                "kind": "run_error",
                "run_id": self.run_id,
                "wall_seconds": wall,
                "error": str(exc)[:500],
            }
        )
