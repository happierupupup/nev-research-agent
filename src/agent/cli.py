"""
CLI for nev-research-agent（默认 S2 四智能体）.

Usage:
    python main.py --user alice
    python main.py --user admin "比亚迪刀片电池的核心卖点是什么？"
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from agent.auth import User, is_admin, resolve_user
from agent.core.intro import INTRO, is_capability_query
from agent.display import format_event_admin, format_event_user
from agent.memory import SessionMemory, UserMemoryStore, user_id_from_env

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


def _print_event_admin(ev: dict) -> None:
    line = format_event_admin(ev)
    if line:
        print(line, flush=True)


def _print_event_user(ev: dict, seen_labels: set[str]) -> None:
    line = format_event_user(ev, seen_labels)
    if line:
        print(line, flush=True)


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


def _print_result(out: dict, *, streamed: bool, user: User) -> None:
    admin = is_admin(user)
    if admin and not streamed:
        steps = out.get("tool_steps") or []
        print(f"\n{DIVIDER}")
        print("思考 / 工具调用过程（管理员）")
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
    if admin:
        trace = out.get("tool_trace") or {}
        print(
            f"\ntools: kb={out.get('kb_calls', 0)} web={out.get('web_calls', 0)} "
            f"tokens={out.get('total_tokens', 0)} "
            f"(p={out.get('prompt_tokens', 0)}/c={out.get('completion_tokens', 0)}"
            f"/llm={out.get('llm_calls', 0)}) "
            f"calls={trace.get('tool_calls')}"
        )
    print()


def _print_intro(user: User) -> None:
    role_cn = "管理员" if is_admin(user) else "普通用户"
    print(f"\n{DIVIDER}")
    print(f"  NEV Research Agent  |  {user.name}（{role_cn}）")
    print(DIVIDER)
    print(INTRO, end="" if INTRO.endswith("\n") else "\n")
    if is_admin(user):
        print("提示: 管理员可见完整思考与工具调用；扩库用 scripts/crawl_to_staging.py --admin\n")
    else:
        print("提示: 普通用户仅显示简要思考过程。\n")


def _run_one(
    query: str,
    *,
    as_json: bool,
    user: User,
    session: SessionMemory | None = None,
    user_memory: UserMemoryStore | None = None,
) -> None:
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
        out = run_research(query, session=session, user_memory=user_memory)
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    admin = is_admin(user)
    print(f"\n{DIVIDER}")
    print("思考 / 工具调用过程（管理员）" if admin else "思考过程")
    print(DIVIDER)
    seen = {"n": 0, "answer_streamed": False}
    user_labels: set[str] = set()

    def on_event(ev: dict) -> None:
        seen["n"] += 1
        kind = ev.get("kind")
        if kind == "answer_start":
            if seen["answer_streamed"]:
                print("\n", flush=True)
            print(f"\n{DIVIDER}")
            print("Agent（流式）:")
            print(DIVIDER)
            seen["answer_streamed"] = True
            return
        if kind == "token":
            print(str(ev.get("text") or ""), end="", flush=True)
            seen["answer_streamed"] = True
            return
        if admin:
            _print_event_admin(ev)
        else:
            _print_event_user(ev, user_labels)

    out = run_research(query, on_event=on_event, session=session, user_memory=user_memory)
    if seen["n"] == 0:
        print("(未调用工具)" if admin else "· 正在生成回答…")
    if seen["answer_streamed"]:
        print(flush=True)
        # 正文已流式打印；只补来源 / token 统计
        _print_sources(out)
        if admin:
            trace = out.get("tool_trace") or {}
            print(
                f"\n[meta] mode={out.get('agent_mode')} "
                f"kb={out.get('kb_calls')} web={out.get('web_calls')} "
                f"tok={out.get('total_tokens')} llm={out.get('llm_calls')} "
                f"calls={trace.get('tool_calls')}"
            )
        return
    _print_result(out, streamed=True, user=user)


def _flush_memory(store: UserMemoryStore | None, session: SessionMemory | None) -> None:
    if store is None or session is None:
        return
    if not session.queries and not session.pending_preferences and not session.pending_episodes:
        return
    store.ingest_session(session)
    store.save()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="NEV Research Agent")
    p.add_argument("question", nargs="?", help="one-shot question (skip REPL)")
    p.add_argument("--json", action="store_true", help="print full JSON result")
    p.add_argument(
        "--memory",
        action="store_true",
        help="one-shot 也读写长期记忆（REPL 默认开启）",
    )
    p.add_argument(
        "--user",
        default=None,
        help="用户 id：admin（管理员）/ alice（普通用户）；默认 NEV_USER_ID 或 alice",
    )
    args = p.parse_args(argv)

    raw_uid = args.user or None
    # 未指定时默认 alice（普通用户），避免误用 local 看不到角色差异
    if raw_uid is None and not (os.getenv("NEV_USER_ID") or "").strip():
        raw_uid = "alice"
    uid = user_id_from_env(raw_uid)
    user = resolve_user(uid)

    use_memory = bool(args.memory or not args.question)
    store = UserMemoryStore.load(uid) if use_memory else None
    session = SessionMemory() if use_memory else None

    if args.question:
        try:
            _run_one(
                args.question,
                as_json=args.json,
                user=user,
                session=session,
                user_memory=store,
            )
            _flush_memory(store, session)
        except Exception as exc:  # noqa: BLE001
            print(f"Error: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        return

    _print_intro(user)

    while True:
        try:
            query = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            _flush_memory(store, session)
            print("\nBye.")
            break
        if not query:
            continue
        if _is_exit(query):
            _flush_memory(store, session)
            print("Bye.")
            break
        try:
            _run_one(query, as_json=args.json, user=user, session=session, user_memory=store)
        except Exception as exc:  # noqa: BLE001
            print(f"\nError: {exc}\n")


if __name__ == "__main__":
    main()
