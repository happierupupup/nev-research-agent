#!/usr/bin/env python3
"""Ask the local knowledge base (retrieve + LLM). Not the full research agent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from kb import ask  # noqa: E402

EXIT = {"exit", "quit", "q", "退出", "结束"}


def main() -> None:
    p = argparse.ArgumentParser(description="KB Q&A (retrieve + LLM)")
    p.add_argument("question", nargs="?", help="one-shot question; omit for REPL")
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--json", action="store_true", help="print full JSON")
    args = p.parse_args()

    if args.question:
        out = ask(args.question, top_k=args.top_k)
        if args.json:
            print(json.dumps(out, ensure_ascii=False, indent=2))
        else:
            _print_out(out)
        return

    print("KB Q&A  |  仅知识库检索 + LLM")
    print("示例: 比亚迪刀片电池的核心卖点是什么？")
    print("退出: exit / quit / q\n")
    while True:
        try:
            q = input("问: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break
        if not q:
            continue
        if q.lower().rstrip("。.！!") in EXIT:
            print("Bye.")
            break
        try:
            out = ask(q, top_k=args.top_k)
        except Exception as exc:  # noqa: BLE001
            print(f"错误: {exc}\n")
            continue
        _print_out(out)


def _print_out(out: dict) -> None:
    print("\n答:", out["answer"])
    print("\n命中:")
    for h in out["hits"]:
        print(f"  [{h['score']:.3f}] {h['source']} · {h['title']}")
    print()


if __name__ == "__main__":
    main()
