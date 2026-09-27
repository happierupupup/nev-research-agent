#!/usr/bin/env python3
"""Memory suite: STM follow-up, LTM interests, preference. Not the golden set.

Usage:
  PYTHONPATH=. python evals/run_memory_eval.py --tag after_memory
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.memory import (  # noqa: E402
    INJECT_MAX_CHARS,
    SessionMemory,
    UserMemoryStore,
    extract_session_interests,
    format_memory_block,
)
from agent.runtime import run_research  # noqa: E402
from evals.metrics import attach_run_meta, count_cited_facts, write_baseline

BASELINES = Path(__file__).resolve().parent / "baselines"

Q1 = "比亚迪刀片电池的核心卖点是什么？"
FOLLOWUP = "它主要用在哪些车上？"
OVERSEAS = "分析比亚迪 2024 年海外扩张战略及风险评估"


def _row(case_id: str, **kwargs: Any) -> dict[str, Any]:
    return {"id": case_id, **kwargs}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="after_memory")
    p.add_argument(
        "--method",
        default=(
            "S1.5 混合短期记忆（近窗结构化回合 + 溢出摘要）"
            " + 按用户长期画像（兴趣关键词 / 偏好 / 情景笔记）。"
            "黄金集 run_research 仍不传 session。"
        ),
        help="这次测的记忆方法，写入 JSON",
    )
    p.add_argument("--notes", default="", help="补充说明，可选")
    args = p.parse_args()

    results: list[dict[str, Any]] = []
    gates: dict[str, bool] = {}

    # M2 first (no LLM): follow-up without history must not invent 它.
    t0 = time.perf_counter()
    m2 = run_research(FOLLOWUP)
    m2_wall = round(time.perf_counter() - t0, 2)
    m2_ok = m2["domain"] == "followup_unclear" and int(m2.get("total_tokens") or 0) == 0
    gates["M2_no_history"] = m2_ok
    results.append(
        _row(
            "M2",
            query=FOLLOWUP,
            wall_seconds=m2_wall,
            domain=m2["domain"],
            total_tokens=m2.get("total_tokens"),
            kb_calls=m2.get("kb_calls"),
            passed=m2_ok,
            note="无短期记忆时不得把「它」编成刀片电池",
        )
    )

    # M4 (no LLM): explicit preference.
    session_pref = SessionMemory()
    t0 = time.perf_counter()
    m4 = run_research("以后短答", session=session_pref)
    m4_wall = round(time.perf_counter() - t0, 2)
    store_pref = UserMemoryStore("eval_pref")
    store_pref.ingest_session(session_pref)
    m4_ok = m4["domain"] == "preference" and store_pref.pref_value("length") == "short"
    gates["M4_preference"] = m4_ok
    results.append(
        _row(
            "M4",
            query="以后短答",
            wall_seconds=m4_wall,
            domain=m4["domain"],
            total_tokens=m4.get("total_tokens"),
            pref=store_pref.pref_value("length"),
            passed=m4_ok,
        )
    )

    # M1 live: Q1 then 它… with the same SessionMemory.
    session = SessionMemory()
    t0 = time.perf_counter()
    turn1 = run_research(Q1, session=session)
    wall1 = round(time.perf_counter() - t0, 2)
    inject = len(format_memory_block(session, None, FOLLOWUP))
    t0 = time.perf_counter()
    turn2 = run_research(FOLLOWUP, session=session)
    wall2 = round(time.perf_counter() - t0, 2)

    sq = turn2.get("search_query") or ""
    resolved = any(x in sq for x in ("刀片", "比亚迪", "BYD"))
    still_kb = int(turn2.get("kb_calls") or 0) >= 1
    cites = count_cited_facts(turn2.get("final_report") or "")
    report2 = turn2.get("final_report") or ""
    cites_memory = any(x in report2 for x in ("来源: 会话", "来源: 记忆", "来源: 本会话"))
    vehicle_hit = any(x in report2 for x in ("汉", "唐", "海豹", "秦", "宋", "车型", "车"))
    m1_ok = (
        turn2.get("domain") == "nev"
        and resolved
        and still_kb
        and cites >= 1
        and not cites_memory
        and vehicle_hit
        and turn2.get("memory_injected") is True
    )
    gates["M1_followup"] = m1_ok
    results.append(
        _row(
            "M1a",
            query=Q1,
            wall_seconds=wall1,
            domain=turn1.get("domain"),
            kb_calls=turn1.get("kb_calls"),
            web_calls=turn1.get("web_calls"),
            prompt_tokens=turn1.get("prompt_tokens"),
            completion_tokens=turn1.get("completion_tokens"),
            total_tokens=turn1.get("total_tokens"),
            llm_calls=turn1.get("llm_calls"),
            cited_facts=count_cited_facts(turn1.get("final_report") or ""),
            search_query=turn1.get("search_query"),
            memory_injected=turn1.get("memory_injected"),
            passed=turn1.get("domain") == "nev" and int(turn1.get("kb_calls") or 0) >= 1,
        )
    )
    results.append(
        _row(
            "M1b",
            query=FOLLOWUP,
            wall_seconds=wall2,
            domain=turn2.get("domain"),
            kb_calls=turn2.get("kb_calls"),
            web_calls=turn2.get("web_calls"),
            prompt_tokens=turn2.get("prompt_tokens"),
            completion_tokens=turn2.get("completion_tokens"),
            total_tokens=turn2.get("total_tokens"),
            llm_calls=turn2.get("llm_calls"),
            cited_facts=cites,
            search_query=sq,
            memory_injected=turn2.get("memory_injected"),
            followup_resolved=resolved,
            retrieval_not_skipped=still_kb,
            cites_memory_as_source=cites_memory,
            vehicle_mention=vehicle_hit,
            inject_chars=inject,
            inject_cap=INJECT_MAX_CHARS,
            passed=m1_ok,
            report_preview=report2[:400],
        )
    )

    # M3: flush LTM, new process-equivalent store, related prompt contains interests.
    path = BASELINES.parent / "._memory_eval_user.json"
    store = UserMemoryStore("eval_user", path=path)
    store.ingest_session(session)
    store.save()
    loaded = UserMemoryStore.load("eval_user", path=path)
    terms = [t for t, _ in extract_session_interests(session)]
    ltm_block = loaded.format_ltm(OVERSEAS)
    interest_ok = ("比亚迪" in terms or "刀片" in terms) and (
        "比亚迪" in ltm_block or "刀片" in ltm_block
    )
    gates["M3_ltm"] = interest_ok
    results.append(
        _row(
            "M3",
            query="(session-end extract)",
            interests=loaded.top_interests(),
            episode_n=len(loaded.episodes),
            ltm_preview=ltm_block[:300],
            passed=interest_ok,
        )
    )
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass

    tok1 = int(turn1.get("total_tokens") or 0)
    tok2 = int(turn2.get("total_tokens") or 0)
    summary = {
        "followup_resolved": resolved,
        "retrieval_not_skipped": still_kb,
        "memory_not_used_as_source": not cites_memory,
        "m2_refuses_bare_deixis": m2_ok,
        "ltm_interest_hit": interest_ok,
        "preference_recorded": m4_ok,
        "inject_chars": inject,
        "inject_cap": INJECT_MAX_CHARS,
        "tokens_turn1": tok1,
        "tokens_turn2": tok2,
        "token_delta_followup_vs_q1": tok2 - tok1,
        "gates_passed": sum(1 for v in gates.values() if v),
        "gates_total": len(gates),
        "all_passed": all(gates.values()),
    }

    payload = attach_run_meta(
        {
            "tag": args.tag,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "suite": "memory",
            "gates": gates,
            "summary": summary,
            "results": results,
        },
        method=args.method,
        notes=args.notes,
        code=["agent/memory.py", "agent/runtime.py", "main.py"],
        compare_to=None,
    )
    BASELINES.mkdir(parents=True, exist_ok=True)
    hist_path = write_baseline(BASELINES, args.tag, payload)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {hist_path}")
    print(f"Latest alias {BASELINES / f'{args.tag}.json'}")
    print(f"Method: {args.method}")
    if not summary["all_passed"]:
        failed = [k for k, v in gates.items() if not v]
        raise SystemExit(f"memory gates failed: {failed}")


if __name__ == "__main__":
    main()
