"""Unit tests for S1 graph helpers (no LLM)."""

from __future__ import annotations

import json
from types import SimpleNamespace

from agent.graph import (
    collect_sources,
    extract_tool_steps,
    extract_tool_trace,
    final_text,
    payload_messages,
    progress_events_from_message,
)


def test_extract_tool_trace():
    msgs = [
        SimpleNamespace(
            tool_calls=[{"name": "kb_search"}, {"name": "web_search"}],
            content="",
        ),
        SimpleNamespace(tool_calls=[{"name": "kb_search"}], content=""),
    ]
    trace = extract_tool_trace(msgs)
    assert trace["kb_calls"] == 2
    assert trace["web_calls"] == 1
    assert trace["retrieval_rounds"] == 3


def test_final_text_skips_tool_message():
    class ToolMessage:
        content = "tool raw"

    class AIMessage:
        def __init__(self, content):
            self.content = content

    msgs = [AIMessage("draft"), ToolMessage(), AIMessage("最终答案")]
    assert final_text(msgs) == "最终答案"


def test_extract_tool_steps_and_sources():
    kb_payload = json.dumps(
        {
            "hits": [
                {
                    "source": "byd_blade_battery.md",
                    "title": "核心卖点",
                    "score": 0.82,
                    "text": "安全",
                }
            ]
        },
        ensure_ascii=False,
    )
    web_payload = json.dumps(
        {
            "results": [
                {
                    "title": "乘联会渗透率",
                    "url": "https://example.com/cpca",
                    "snippet": "40%",
                }
            ]
        },
        ensure_ascii=False,
    )

    class AIMessage:
        def __init__(self, tool_calls):
            self.tool_calls = tool_calls
            self.content = ""

    class ToolMessage:
        def __init__(self, name, content, tool_call_id):
            self.name = name
            self.content = content
            self.tool_call_id = tool_call_id

    msgs = [
        AIMessage(
            [
                {"id": "c1", "name": "kb_search", "args": {"query": "刀片电池"}},
                {"id": "c2", "name": "web_search", "args": {"query": "渗透率"}},
            ]
        ),
        ToolMessage("kb_search", kb_payload, "c1"),
        ToolMessage("web_search", web_payload, "c2"),
    ]
    steps = extract_tool_steps(msgs)
    assert len(steps) == 2
    assert steps[0]["tool"] == "kb_search"
    assert "byd_blade_battery.md" in steps[0]["result_preview"]
    sources = collect_sources(steps)
    assert [s["type"] for s in sources] == ["KB", "Web"]
    assert sources[0]["ref"] == "KB/byd_blade_battery.md"
    assert sources[1]["ref"] == "https://example.com/cpca"


def test_progress_events_from_message():
    class AIMessage:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class ToolMessage:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    evs, seq = progress_events_from_message(
        AIMessage(
            tool_calls=[{"id": "c1", "name": "kb_search", "args": {"query": "刀片"}}],
            content="",
        ),
        0,
    )
    assert seq == 1
    assert evs[0]["kind"] == "tool_call"
    assert evs[0]["query"] == "刀片"

    evs2, seq2 = progress_events_from_message(
        ToolMessage(
            name="kb_search",
            tool_call_id="c1",
            content=json.dumps({"hits": [{"source": "a.md", "title": "t", "score": 1}]}),
        ),
        seq,
    )
    assert seq2 == 1
    assert evs2[0]["kind"] == "tool_result"
    assert "a.md" in evs2[0]["preview"]

    evs3, seq3 = progress_events_from_message(
        AIMessage(tool_calls=[], content="正式答案"),
        seq2,
    )
    assert seq3 == 1
    assert evs3 == [{"kind": "answer_ready"}]


def test_payload_messages():
    assert payload_messages(None) == []
    assert payload_messages({"messages": ["a", "b"]}) == ["a", "b"]
    assert payload_messages({"messages": "one"}) == ["one"]
    assert payload_messages(SimpleNamespace(messages=["x"])) == ["x"]
