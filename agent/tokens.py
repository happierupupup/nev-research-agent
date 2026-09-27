"""Chat-LLM token usage from AIMessage metadata (S1: record only, no gates)."""

from __future__ import annotations

from typing import Any


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def usage_from_message(msg: Any) -> dict[str, int] | None:
    """One model call: prompt / completion / total. None if no usage on the message."""
    um = getattr(msg, "usage_metadata", None)
    if isinstance(um, dict) and any(
        _as_int(um.get(k)) for k in ("input_tokens", "output_tokens", "total_tokens")
    ):
        prompt = _as_int(um.get("input_tokens"))
        completion = _as_int(um.get("output_tokens"))
        total = _as_int(um.get("total_tokens")) or (prompt + completion)
        return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}

    meta = getattr(msg, "response_metadata", None) or {}
    if not isinstance(meta, dict):
        return None
    tu = meta.get("token_usage") or meta.get("usage") or {}
    if not isinstance(tu, dict):
        return None
    prompt = _as_int(tu.get("prompt_tokens") or tu.get("input_tokens"))
    completion = _as_int(tu.get("completion_tokens") or tu.get("output_tokens"))
    total = _as_int(tu.get("total_tokens")) or (prompt + completion)
    if prompt or completion or total:
        return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}
    return None


def empty_usage() -> dict[str, int]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "llm_calls": 0,
    }


def collect_token_usage(messages: list[Any] | None) -> dict[str, int]:
    """Sum chat-model usage on AI messages. Skips ToolMessage. Dedupes by message.id."""
    out = empty_usage()
    seen: set[str] = set()
    for msg in messages or []:
        name = type(msg).__name__
        if name == "ToolMessage" or getattr(msg, "type", None) == "tool":
            continue
        uid = getattr(msg, "id", None)
        if uid and str(uid) in seen:
            continue
        piece = usage_from_message(msg)
        if not piece:
            continue
        if uid:
            seen.add(str(uid))
        out["prompt_tokens"] += piece["prompt_tokens"]
        out["completion_tokens"] += piece["completion_tokens"]
        out["total_tokens"] += piece["total_tokens"]
        out["llm_calls"] += 1
    if out["total_tokens"] == 0:
        out["total_tokens"] = out["prompt_tokens"] + out["completion_tokens"]
    return out


def flatten_callback_usage(by_model: dict[str, Any] | None) -> dict[str, int]:
    """Collapse UsageMetadataCallbackHandler.usage_metadata into one dict."""
    out = empty_usage()
    for um in (by_model or {}).values():
        if not isinstance(um, dict):
            continue
        prompt = _as_int(um.get("input_tokens"))
        completion = _as_int(um.get("output_tokens"))
        total = _as_int(um.get("total_tokens")) or (prompt + completion)
        if not (prompt or completion or total):
            continue
        out["prompt_tokens"] += prompt
        out["completion_tokens"] += completion
        out["total_tokens"] += total
        out["llm_calls"] += 1
    return out


def prefer_usage(primary: dict[str, int], fallback: dict[str, int]) -> dict[str, int]:
    """Keep the harvest with more total_tokens (messages vs callback)."""
    a = primary or empty_usage()
    b = fallback or empty_usage()
    return a if int(a.get("total_tokens") or 0) >= int(b.get("total_tokens") or 0) else b
