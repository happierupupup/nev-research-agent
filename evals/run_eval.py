#!/usr/bin/env python3
"""
统一评测入口：跑黄金集、落盘 baseline、前后对比。

用法:
  python evals/run_eval.py --suite golden --tag baseline_phase1
  python evals/run_eval.py --suite golden --tag after_p1 --compare baseline_phase1
  python evals/run_eval.py --suite golden --tag dry --dry-run

Agent 未就绪时默认 --dry-run：用占位报告写快照，验证指标管道。
接上 main/agent 后去掉 --dry-run，由 runner 调用真实流水线。
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

import yaml  # type: ignore

from evals.metrics import check_hard_gates, compare_runs, count_chars, count_cited_facts

EVALS = Path(__file__).resolve().parent
BASELINES = EVALS / "baselines"
GOLDEN = EVALS / "golden" / "queries.yaml"
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
        out = run_research(query)
        report = out["final_report"] if isinstance(out, dict) else str(out)
        meta = out if isinstance(out, dict) else {}
    wall = time.perf_counter() - t0
    return {
        "report": report,
        "wall_seconds": round(wall, 2),
        "retrieval_rounds": meta.get("retrieval_rounds"),
        "iteration_count": meta.get("iteration_count"),
    }


def run_suite(*, tag: str, dry_run: bool, compare_tag: str | None) -> Path:
    suite = _load_yaml(GOLDEN)
    rubric = _load_yaml(RUBRIC)
    weights = {d["id"]: float(d["weight"]) for d in rubric["dimensions"]}

    results: list[dict[str, Any]] = []
    for item in suite["queries"]:
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
        )
        # 人工 / Verifier 分数：dry-run 不填；正式跑后可写入 dimension_scores
        dimension_scores = {d: None for d in weights}
        results.append(
            {
                "id": item["id"],
                "query": item["query"],
                "wall_seconds": run["wall_seconds"],
                "chars": chars,
                "cited_facts": cited,
                "retrieval_rounds": run.get("retrieval_rounds"),
                "hard_gates": gate_result,
                "dimension_scores": dimension_scores,
                "rubric_total": None,  # 填完 dimension_scores 后用 metrics.weighted_rubric_score
                "report_preview": report[:400],
            }
        )

    payload = {
        "tag": tag,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "suite": "golden",
        "rubric_version": rubric.get("version"),
        "results": results,
    }

    BASELINES.mkdir(parents=True, exist_ok=True)
    out_path = BASELINES / f"{tag}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")

    for r in results:
        status = "PASS" if r["hard_gates"]["passed"] else "FAIL"
        print(
            f"  [{status}] {r['id']}: chars={r['chars']} cites={r['cited_facts']} "
            f"wall={r['wall_seconds']}s failures={r['hard_gates']['failures']}"
        )

    if compare_tag:
        base_path = BASELINES / f"{compare_tag}.json"
        if not base_path.exists():
            raise SystemExit(f"baseline not found: {base_path}")
        baseline = json.loads(base_path.read_text(encoding="utf-8"))
        diff = compare_runs(baseline, payload)
        print("\n=== Compare vs", compare_tag, "===")
        print(json.dumps(diff, ensure_ascii=False, indent=2))

    return out_path


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
    args = p.parse_args()
    # 无 Agent 时默认建议 dry-run；若未指定且 runtime 不存在则强制 dry-run
    dry = args.dry_run
    if not dry:
        try:
            import agent.runtime  # noqa: F401
        except ImportError:
            print("agent.runtime missing → falling back to --dry-run")
            dry = True
    run_suite(tag=args.tag, dry_run=dry, compare_tag=args.compare)


if __name__ == "__main__":
    main()
