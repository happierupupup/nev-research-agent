"""Unit tests for chat-LLM token harvest (no live model)."""

from __future__ import annotations

from types import SimpleNamespace

from agent.tokens import collect_token_usage, flatten_callback_usage, prefer_usage, usage_from_message
from evals.metrics import compare_runs


def test_usage_from_langchain_metadata():
    msg = SimpleNamespace(
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        response_metadata={},
        id="a1",
    )
    assert usage_from_message(msg) == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }


def test_usage_from_openai_token_usage_fallback():
    msg = SimpleNamespace(
        usage_metadata=None,
        response_metadata={"token_usage": {"prompt_tokens": 20, "completion_tokens": 7, "total_tokens": 27}},
        id="b1",
    )
    assert usage_from_message(msg)["total_tokens"] == 27


def test_collect_skips_tool_and_dedupes():
    ai = SimpleNamespace(
        usage_metadata={"input_tokens": 8, "output_tokens": 2, "total_tokens": 10},
        response_metadata={},
        id="same",
    )
    tool = SimpleNamespace(content="hit", type="tool")
    out = collect_token_usage([ai, tool, ai])
    assert out["llm_calls"] == 1
    assert out["total_tokens"] == 10


def test_prefer_usage_picks_richer():
    a = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "llm_calls": 1}
    b = {"prompt_tokens": 9, "completion_tokens": 1, "total_tokens": 10, "llm_calls": 2}
    assert prefer_usage(a, b)["total_tokens"] == 10
    assert flatten_callback_usage(
        {"m": {"input_tokens": 3, "output_tokens": 1, "total_tokens": 4}}
    )["total_tokens"] == 4


def test_compare_runs_includes_tokens():
    baseline = {
        "results": [
            {"id": "Q1", "rubric_total": None, "wall_seconds": 9, "kb_calls": 0, "total_tokens": None},
        ]
    }
    current = {
        "results": [
            {
                "id": "Q1",
                "rubric_total": None,
                "wall_seconds": 12,
                "kb_calls": 1,
                "prompt_tokens": 100,
                "completion_tokens": 40,
                "total_tokens": 140,
                "llm_calls": 2,
            },
        ]
    }
    diff = compare_runs(baseline, current)
    assert diff["per_query_tools"][0]["total_tokens"]["current"] == 140
    assert diff["mean_total_tokens"]["current"] == 140
    assert diff["sum_total_tokens"]["current"] == 140
