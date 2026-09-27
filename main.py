"""
CLI for nev-research-agent (S1 single agent).

Usage:
    python main.py
    python main.py "比亚迪刀片电池的核心卖点是什么？"
"""

from __future__ import annotations

import argparse
import json
import sys

from agent.intro import INTRO, is_capability_query

DIVIDER = "─" * 56

EXIT_COMMANDS = {
    "exit",
    "quit",
    "q",
    "bye",
    "stop",
    "end",
    "退出",
    "结束",
    "再见",
    "不聊了",
}


def _is_exit(text: str) -> bool:
    t = text.strip().lower().rstrip("。.！!？?")
    return t in EXIT_COMMANDS


def _print_event(ev: dict) -> None:
    kind = ev.get("kind")
    if kind == "tool_call":
        print(f"[{ev.get('step')}] {ev.get('tool')} ← {ev.get('query')}", flush=True)
        return
    if kind == "tool_result":
        preview = ev.get("preview") or ""
        if preview:
            tool = ev.get("tool") or ""
            tag = f"[{tool}] " if tool else ""
            print(f"    → {tag}{preview}", flush=True)
        return
    if kind == "answer_ready":
        print("    ✓ 检索结束，正在输出回答", flush=True)


def _print_sources(out: dict) -> None:
    sources = out.get("sources") or []
    print(f"\n{DIVIDER}")
    print("信息来源")
    print(DIVIDER)
    if not sources:
        print("(无)")
        return
    for i, s in enumerate(sources, start=1):
        typ = s.get("type") or "?"
        label = s.get("label") or ""
        ref = s.get("ref") or ""
        if typ == "KB":
            print(f"{i}. [KB] {label}")
        else:
            print(f"{i}. [Web] {label}")
            if ref and ref != label:
                print(f"   {ref}")


def _print_result(out: dict, *, streamed: bool) -> None:
    if not streamed:
        steps = out.get("tool_steps") or []
        print(f"\n{DIVIDER}")
        print("思考 / 检索过程")
        print(DIVIDER)
        if not steps:
            print("(未调用工具)")
        else:
            for step in steps:
                args = step.get("args") or {}
                query = args.get("query") or args.get("q") or args
                print(f"[{step.get('step')}] {step.get('tool')} ← {query}")
                preview = step.get("result_preview") or ""
                if preview:
                    print(f"    → {preview}")

    report = (out.get("final_report") or "").strip()
    print(f"\n{DIVIDER}")
    print("Agent:")
    print(DIVIDER)
    print(report or "(无输出)")
    _print_sources(out)
    trace = out.get("tool_trace") or {}
    print(
        f"\ntools: kb={out.get('kb_calls', 0)} web={out.get('web_calls', 0)} "
        f"tokens={out.get('total_tokens', 0)} "
        f"(p={out.get('prompt_tokens', 0)}/c={out.get('completion_tokens', 0)}"
        f"/llm={out.get('llm_calls', 0)}) "
        f"calls={trace.get('tool_calls')}"
    )
    print()


def _print_intro() -> None:
    print(f"\n{DIVIDER}")
    print("  NEV Research Agent  |  新能源汽车研究助手")
    print(DIVIDER)
    print(INTRO, end="" if INTRO.endswith("\n") else "\n")


def _run_one(query: str, *, as_json: bool) -> None:
    if is_capability_query(query):
        if as_json:
            print(json.dumps({"final_report": INTRO.strip(), "sources": []}, ensure_ascii=False, indent=2))
            return
        print(f"\n{DIVIDER}")
        print("Agent:")
        print(DIVIDER)
        print(INTRO)
        return

    from agent.runtime import run_research

    if as_json:
        out = run_research(query)
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    print(f"\n{DIVIDER}")
    print("思考 / 检索过程")
    print(DIVIDER)
    seen = {"n": 0}

    def on_event(ev: dict) -> None:
        seen["n"] += 1
        _print_event(ev)

    out = run_research(query, on_event=on_event)
    if seen["n"] == 0:
        print("(未调用工具)")
    _print_result(out, streamed=True)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="NEV Research Agent (S1)")
    p.add_argument("question", nargs="?", help="one-shot question (skip REPL)")
    p.add_argument("--json", action="store_true", help="print full JSON result")
    args = p.parse_args(argv)

    if args.question:
        try:
            _run_one(args.question, as_json=args.json)
        except Exception as exc:  # noqa: BLE001
            print(f"Error: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        return

    _print_intro()

    while True:
        try:
            query = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break
        if not query:
            continue
        if _is_exit(query):
            print("Bye.")
            break
        try:
            _run_one(query, as_json=args.json)
        except Exception as exc:  # noqa: BLE001
            print(f"\nError: {exc}\n")


if __name__ == "__main__":
    main()
