"""Analyst 真流式：mock llm.stream 发出 token 事件。"""

from __future__ import annotations

from types import SimpleNamespace

from agent.s2.analyst.node import _content_piece, _stream_synthesize, run_node


class _Chunk:
    def __init__(self, text: str) -> None:
        self.content = text

    def __add__(self, other: "_Chunk") -> "_Chunk":
        return _Chunk(str(self.content) + str(other.content))


class _FakeLLM:
    def __init__(self, pieces: list[str], *, fail_stream: bool = False) -> None:
        self.pieces = pieces
        self.fail_stream = fail_stream
        self.invoked = False

    def stream(self, _messages):
        if self.fail_stream:
            raise RuntimeError("stream down")
        for p in self.pieces:
            yield _Chunk(p)

    def invoke(self, _messages):
        self.invoked = True
        return _Chunk("".join(self.pieces) or "fallback")


def test_content_piece_str_and_list():
    assert _content_piece(_Chunk("刀片")) == "刀片"
    assert _content_piece(SimpleNamespace(content=[{"type": "text", "text": "电池"}])) == "电池"


def test_stream_emits_answer_start_and_tokens():
    events: list[dict] = []
    llm = _FakeLLM(["比亚迪", "刀片", "电池"])
    text, msg = _stream_synthesize(llm, [], events.append)
    assert text == "比亚迪刀片电池"
    assert msg is not None
    kinds = [e["kind"] for e in events]
    assert kinds[0] == "answer_start"
    assert kinds[1:] == ["token", "token", "token"]
    assert "".join(e["text"] for e in events if e["kind"] == "token") == text


def test_stream_fallback_to_invoke():
    events: list[dict] = []
    llm = _FakeLLM(["整段回答"], fail_stream=True)
    text, _msg = _stream_synthesize(llm, [], events.append)
    assert llm.invoked
    assert text == "整段回答"
    assert any(e.get("kind") == "token" and e.get("text") == "整段回答" for e in events)


def test_run_node_streams(monkeypatch):
    events: list[dict] = []
    llm = _FakeLLM(["# 结论\n", "安全。"])

    monkeypatch.setattr("agent.s2.analyst.node.get_llm", lambda temperature=0: llm)
    monkeypatch.setattr(
        "agent.s2.analyst.node.format_workspace",
        lambda _w: "素材",
    )

    out = run_node(
        {
            "query": "刀片电池卖点",
            "research_workspace": [{"type": "KB", "text": "x"}],
            "_on_event": events.append,
            "report_mode": False,
        }
    )
    assert "安全" in out["final_report"]
    assert out["route"] == "verifier"
    assert any(e.get("kind") == "token" for e in events)
    assert any(e.get("kind") == "answer_ready" for e in events)
