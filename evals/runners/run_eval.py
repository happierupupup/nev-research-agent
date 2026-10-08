#!/usr/bin/env python3
"""
统一评测入口：跑黄金集、落盘 baseline、前后对比。

用法:
  python evals/run_eval.py --suite golden --tag s1_baseline --ids Q1,Q2
  python evals/run_eval.py --suite golden --tag s1_baseline --ids Q1,Q2,G1
  python evals/run_eval.py --suite golden --tag after_xxx --compare s1_baseline
  python evals/run_eval.py --suite golden --tag dry --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for _p in (str(ROOT), str(SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import yaml  # type: ignore

from evals.metrics import (
    attach_run_meta,
    check_hard_gates,
    compare_runs,
    count_chars,
    count_cited_facts,
    write_baseline,
)

EVALS = ROOT / "evals"
BASELINES = EVALS / "baselines"
GOLDEN = EVALS / "datasets" / "golden" / "queries.yaml"
RUBRIC = EVALS / "rubric.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _stub_report(query: str) -> str:
    """占位报告：仅用于打通评测管道，不代表真实质量。"""
    body = (
        f"# 占位研究报告\n\n命题：{query}\n\n"
        "## 摘要\n本报告为 eval dry-run 占位文本，用于验证指标抽取与 baseline 落盘。"
        "正式跑评测请去掉 --dry-run 并接入 Agent。\n\n"
    )
    # pad past Phase1 min_chars so dry-run validates the harness, not content quality
    pad = ("（待生成占位段落，正式评测请接入 Agent 后去掉 --dry-run。）\n") * 130
    cites = "\n".join(f"- 事实点 {i} [来源占位][{i}]" for i in range(1, 6))
    return body + pad + "\n\n## 引用\n" + cites + "\n"


def _run_one(query: str, *, dry_run: bool) -> dict[str, Any]:
    t0 = time.perf_counter()
    if dry_run:
        report = _stub_report(query)
        meta = {"mode": "dry_run", "retrieval_rounds": 0}
    else:
        # Phase 1 接入点：from agent.runtime import run_research
        try:
            from agent.runtime import run_research  # type: ignore
        except ImportError as exc:
            raise SystemExit(
                "Agent 未就绪。请先实现 agent.runtime.run_research，或使用 --dry-run。"
            ) from exc
        out = run_research(query)  # 黄金集不传 session / user_memory（S1.5 记忆默认关）
        report = out["final_report"] if isinstance(out, dict) else str(out)
        meta = out if isinstance(out, dict) else {}
    wall = time.perf_counter() - t0
    return {
        "report": report,
        "wall_seconds": round(wall, 2),
        "retrieval_rounds": meta.get("retrieval_rounds"),
        "iteration_count": meta.get("iteration_count"),
        "kb_calls": meta.get("kb_calls"),
        "web_calls": meta.get("web_calls"),
        "cache_hits": meta.get("cache_hits"),
        "tool_trace": meta.get("tool_trace"),
        "prompt_tokens": meta.get("prompt_tokens"),
        "completion_tokens": meta.get("completion_tokens"),
        "total_tokens": meta.get("total_tokens"),
        "llm_calls": meta.get("llm_calls"),
        "domain": meta.get("domain"),
    }


def run_suite(
    *,
    tag: str,
    dry_run: bool,
    compare_tag: str | None,
    ids: set[str] | None = None,
    method: str,
    notes: str = "",
) -> Path:
    suite = _load_yaml(GOLDEN)
    rubric = _load_yaml(RUBRIC)
    weights = {d["id"]: float(d["weight"]) for d in rubric["dimensions"]}

    items = suite["queries"]
    if ids:
        items = [q for q in items if q["id"] in ids]
        missing = ids - {q["id"] for q in items}
        if missing:
            raise SystemExit(f"unknown query ids: {sorted(missing)}")

    results: list[dict[str, Any]] = []
    for item in items:
        print(f"\nRunning {item['id']}: {item['query'][:80]}", flush=True)
        run = _run_one(item["query"], dry_run=dry_run)
        report = run["report"]
        chars = count_chars(report)
        cited = count_cited_facts(report)
        gates = item.get("hard_gates") or rubric.get("hard_gates", {}).get("phase1", {})
        gate_result = check_hard_gates(
            chars=chars,
            cited_facts=cited,
            wall_seconds=run["wall_seconds"],
            gates=gates,
            kb_calls=run.get("kb_calls"),
            web_calls=run.get("web_calls"),
        )
        dimension_scores = {d: None for d in weights}
        status = "PASS" if gate_result["passed"] else "FAIL"
        print(
            f"  [{status}] {item['id']}: chars={chars} cites={cited} "
            f"wall={run['wall_seconds']}s kb={run.get('kb_calls')} web={run.get('web_calls')} "
            f"cache={run.get('cache_hits')} "
            f"tok={run.get('total_tokens')} (p={run.get('prompt_tokens')}/c={run.get('completion_tokens')}"
            f"/calls={run.get('llm_calls')}) failures={gate_result['failures']}",
            flush=True,
        )
        results.append(
            {
                "id": item["id"],
                "query": item["query"],
                "wall_seconds": run["wall_seconds"],
                "chars": chars,
                "cited_facts": cited,
                "retrieval_rounds": run.get("retrieval_rounds"),
                "kb_calls": run.get("kb_calls"),
                "web_calls": run.get("web_calls"),
                "cache_hits": run.get("cache_hits"),
                "prompt_tokens": run.get("prompt_tokens"),
                "completion_tokens": run.get("completion_tokens"),
                "total_tokens": run.get("total_tokens"),
                "llm_calls": run.get("llm_calls"),
                "domain": run.get("domain"),
                "tool_trace": run.get("tool_trace"),
                "hard_gates": gate_result,
                "dimension_scores": dimension_scores,
                "rubric_total": None,
                "agent_mode": run.get("agent_mode"),
                "iteration_count": run.get("iteration_count"),
                "verification_issues": run.get("verification_issues"),
                "report_preview": report[:400],
                "report": report,
            }
        )

    payload = attach_run_meta(
        {
            "tag": tag,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dry_run": dry_run,
            "suite": "golden",
            "rubric_version": rubric.get("version"),
            "results": results,
        },
        method=method,
        notes=notes,
        code=["src/agent/runtime.py", "src/agent/s2/graph.py", "src/agent/s2", "src/agent/s1/graph.py"],
        compare_to=compare_tag,
    )

    BASELINES.mkdir(parents=True, exist_ok=True)
    hist_path = write_baseline(BASELINES, tag, payload)
    latest = BASELINES / f"{tag}.json"
    print(f"Wrote {hist_path}")
    print(f"Latest alias {latest}")
    print(f"Method: {method}")

    for r in results:
        status = "PASS" if r["hard_gates"]["passed"] else "FAIL"
        print(
            f"  [{status}] {r['id']}: chars={r['chars']} cites={r['cited_facts']} "
            f"wall={r['wall_seconds']}s kb={r.get('kb_calls')} web={r.get('web_calls')} "
            f"tok={r.get('total_tokens')} failures={r['hard_gates']['failures']}"
        )

    if compare_tag:
        base_path = BASELINES / f"{compare_tag}.json"
        if not base_path.exists():
            raise SystemExit(f"baseline not found: {base_path}")
        baseline = json.loads(base_path.read_text(encoding="utf-8"))
        diff = compare_runs(baseline, payload)
        print("\n=== Compare vs", compare_tag, "===")
        print(json.dumps(diff, ensure_ascii=False, indent=2))

    return hist_path


def main() -> None:
    p = argparse.ArgumentParser(description="NEV research-agent eval runner")
    p.add_argument("--suite", default="golden", choices=["golden"])
    p.add_argument("--tag", required=True, help="snapshot name, e.g. baseline_phase1")
    p.add_argument("--compare", default=None, help="previous tag to diff against")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="use stub report (no LLM); for wiring metrics before Agent exists",
    )
    p.add_argument(
        "--ids",
        default=None,
        help="comma-separated query ids, e.g. Q1,Q2,G1 (default: all)",
    )
    p.add_argument(
        "--method",
        default=None,
        help="这次测的方法/改动（必填，除非 --dry-run），写入 JSON 供日后对照",
    )
    p.add_argument("--notes", default="", help="补充说明，可选")
    args = p.parse_args()
    dry = args.dry_run
    if not dry:
        try:
            from agent.runtime import run_research  # noqa: F401
            # implemented if callable and not raising NotImplemented by import alone
        except ImportError:
            print("agent.runtime missing → falling back to --dry-run")
            dry = True
    method = (args.method or "").strip()
    if not method:
        if dry:
            method = "dry-run stub reports；不跑 LLM / 工具"
        else:
            raise SystemExit(
                "请用 --method 写明本次测的是什么方法，例如：\n"
                "  --method 'S1 ReAct + 领域门闩；黄金集关记忆'"
            )
    id_set = {x.strip() for x in args.ids.split(",") if x.strip()} if args.ids else None
    run_suite(
        tag=args.tag,
        dry_run=dry,
        compare_tag=args.compare,
        ids=id_set,
        method=method,
        notes=args.notes,
    )


if __name__ == "__main__":
    main()
