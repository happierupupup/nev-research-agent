"""
CLI stub for nev-research-agent.

Phase 1: wire agent.runtime.run_research here.
Usage:
    python main.py
"""

from __future__ import annotations

EXIT = {"exit", "quit", "q", "退出", "结束"}


def main() -> None:
    print("NEV Research Agent  |  Phase 1 stub")
    print("评估: python evals/run_eval.py --tag baseline_phase1 --dry-run")
    print("退出: exit / quit / q\n")
    while True:
        try:
            q = input("研究命题: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break
        if not q:
            continue
        if q.lower().rstrip("。.！!") in EXIT:
            print("Bye.")
            break
        try:
            from agent.runtime import run_research
        except ImportError:
            print("agent.runtime 尚未实现。请先按 plan.md Phase 1 搭建，或先跑 evals dry-run。\n")
            continue
        out = run_research(q)
        report = out["final_report"] if isinstance(out, dict) else str(out)
        print("\n" + report + "\n")


if __name__ == "__main__":
    main()
