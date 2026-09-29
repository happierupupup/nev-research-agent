#!/usr/bin/env python3
"""
KB 检索评测：P@k / R@k / Hit@k（不跑对话 LLM，只需 embedding）。

用法:
  python evals/run_retrieval.py --tag kb_expand
  python evals/run_retrieval.py --tag after_chunk --compare kb_expand
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

from evals.metrics import attach_run_meta, hit_at_k, precision_at_k, recall_at_k, write_baseline
from kb import KnowledgeBase

EVALS = ROOT / "evals"
BASELINES = EVALS / "baselines"
QRELS = EVALS / "datasets" / "retrieval" / "qrels.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def run_suite(
    *,
    tag: str,
    k: int | None,
    compare_tag: str | None,
    method: str,
    notes: str = "",
) -> Path:
    spec = _load_yaml(QRELS)
    top_k = int(k or spec.get("k") or 4)
    kb = KnowledgeBase()

    results: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for item in spec["queries"]:
        relevant = set(item["relevant"])
        hits = kb.search(item["query"], top_k=top_k)
        ranked_sources = [h.chunk.source for h in hits]
        unique_docs = _unique(ranked_sources)
        row = {
            "id": item["id"],
            "query": item["query"],
            "relevant": sorted(relevant),
            "retrieved_sources": ranked_sources,
            "retrieved_docs": unique_docs,
            "p_at_k": precision_at_k(ranked_sources, relevant, top_k),
            "r_at_k": recall_at_k(unique_docs, relevant, top_k),
            "hit_at_k": hit_at_k(ranked_sources, relevant, top_k),
            "top_titles": [h.chunk.title for h in hits],
        }
        results.append(row)

    wall = round(time.perf_counter() - t0, 2)
    n = len(results) or 1
    mean_p = round(sum(r["p_at_k"] for r in results) / n, 4)
    mean_r = round(sum(r["r_at_k"] for r in results) / n, 4)
    mean_hit = round(sum(r["hit_at_k"] for r in results) / n, 4)

    payload = attach_run_meta(
        {
            "tag": tag,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "suite": "retrieval",
            "k": top_k,
            "n_queries": len(results),
            "wall_seconds": wall,
            "mean_p_at_k": mean_p,
            "mean_r_at_k": mean_r,
            "mean_hit_at_k": mean_hit,
            "results": results,
        },
        method=method,
        notes=notes,
        code=["src/kb/__init__.py", "scripts/ingest_kb.py", "evals/datasets/retrieval/qrels.yaml"],
        compare_to=compare_tag,
    )

    BASELINES.mkdir(parents=True, exist_ok=True)
    hist_path = write_baseline(BASELINES, f"retrieval_{tag}", payload)
    print(f"Wrote {hist_path}")
    print(f"Latest alias {BASELINES / f'retrieval_{tag}.json'}")
    print(f"Method: {method}")
    print(f"  mean P@{top_k}={mean_p}  R@{top_k}={mean_r}  Hit@{top_k}={mean_hit}  n={len(results)}")
    for r in results:
        print(
            f"  [{r['id']}] P={r['p_at_k']} R={r['r_at_k']} Hit={r['hit_at_k']} "
            f"docs={r['retrieved_docs']}"
        )

    if compare_tag:
        base_path = BASELINES / f"retrieval_{compare_tag}.json"
        if not base_path.exists():
            raise SystemExit(f"baseline not found: {base_path}")
        baseline = json.loads(base_path.read_text(encoding="utf-8"))
        print(f"\n=== Compare vs retrieval_{compare_tag} ===")
        print(
            json.dumps(
                {
                    "delta_mean_p_at_k": round(mean_p - float(baseline.get("mean_p_at_k") or 0), 4),
                    "delta_mean_r_at_k": round(mean_r - float(baseline.get("mean_r_at_k") or 0), 4),
                    "delta_mean_hit_at_k": round(
                        mean_hit - float(baseline.get("mean_hit_at_k") or 0), 4
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return hist_path


def main() -> None:
    p = argparse.ArgumentParser(description="NEV KB retrieval eval (P@k / R@k / Hit@k)")
    p.add_argument("--tag", required=True, help="snapshot name, e.g. kb_expand")
    p.add_argument("--k", type=int, default=None, help="override qrels k (default 4)")
    p.add_argument("--compare", default=None, help="previous retrieval tag")
    p.add_argument(
        "--method",
        default="本地 JSON 余弦检索（DashScope embed）P@k / R@k / Hit@k，不跑对话 LLM",
        help="这次测的检索方法，写入 JSON",
    )
    p.add_argument("--notes", default="", help="补充说明，可选")
    args = p.parse_args()
    try:
        run_suite(
            tag=args.tag,
            k=args.k,
            compare_tag=args.compare,
            method=args.method,
            notes=args.notes,
        )
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
