"""
KB 检索评测：P@k / R@k / Hit@k（不跑对话 LLM，只需 embedding）。

用法:
  # 文件级（旧）
  python evals/run_retrieval.py --tag kb_expand

  # chunk 级（加固评测主用）
  python evals/run_retrieval.py --tag rag_chunk_baseline_v3 \\
    --qrels evals/datasets/retrieval/qrels_v3.yaml --level chunk
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
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from evals.metrics import (
    attach_run_meta,
    hit_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    write_baseline,
)
from kb import KnowledgeBase
from kb.hybrid import retrieval_mode_from_env
from kb.rerank import rerank_mode_from_env
from kb.rewrite import rewrite_mode_from_env, rewrite_query

EVALS = ROOT / "evals"
BASELINES = EVALS / "baselines"
QRELS_DEFAULT = EVALS / "datasets" / "retrieval" / "qrels.yaml"


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


def _gold_ids(item: dict[str, Any], *, level: str) -> set[str]:
    if level == "chunk":
        raw = item.get("relevant_chunks") or item.get("relevant") or []
    else:
        raw = item.get("relevant_docs") or item.get("relevant") or []
    return {str(x) for x in raw}


def _validate_gold(spec: dict[str, Any], *, level: str, kb: KnowledgeBase) -> list[str]:
    missing: list[str] = []
    known_chunks = set(kb.chunks_by_id) if level == "chunk" and kb.chunks_by_id else set()
    docs_root = ROOT / "data" / "kb"
    seen: set[str] = set()
    for item in spec.get("queries") or []:
        for gid in _gold_ids(item, level=level):
            if gid in seen:
                continue
            seen.add(gid)
            if level == "chunk":
                if known_chunks and gid not in known_chunks:
                    missing.append(gid)
            else:
                if not (docs_root / gid).exists():
                    missing.append(gid)
    return missing


def run_suite(
    *,
    tag: str,
    k: int | None,
    compare_tag: str | None,
    method: str,
    notes: str = "",
    qrels_path: Path | None = None,
    level: str = "doc",
    expand_neighbors: int | None = None,
    rewrite: str | None = None,
    retrieval: str | None = None,
    rerank: str | None = None,
    rerank_candidates: int | None = None,
) -> Path:
    qrels = qrels_path or QRELS_DEFAULT
    if not qrels.is_absolute():
        qrels = (ROOT / qrels).resolve()
    if not qrels.exists():
        raise FileNotFoundError(f"qrels not found: {qrels}")

    level = (level or "doc").strip().lower()
    if level not in {"doc", "chunk"}:
        raise SystemExit("--level 须为 doc 或 chunk")

    spec = _load_yaml(qrels)
    # chunk 级默认不扩邻块，避免同文档重复槽位干扰准确率
    if expand_neighbors is None:
        expand_neighbors = 0 if level == "chunk" else 1
    top_k = int(k or spec.get("k") or (8 if level == "chunk" else 4))

    kb = KnowledgeBase()
    if level == "chunk" and kb.backend == "milvus" and not kb.chunks_by_id:
        # milvus 模式未预载 chunks；校验跳过，仅靠运行期 id
        pass

    missing = _validate_gold(spec, level=level, kb=kb)
    if missing:
        print(f"WARN: 金标缺失 {len(missing)} 个: {missing[:8]}…", flush=True)

    rewrite_mode = rewrite_mode_from_env(rewrite)
    retrieval_mode = retrieval_mode_from_env(retrieval)
    rerank_mode = rerank_mode_from_env(rerank)
    results: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for item in spec["queries"]:
        relevant = _gold_ids(item, level=level)
        if not relevant:
            raise SystemExit(f"query {item.get('id')} 无金标（level={level}）")
        raw_q = str(item["query"] or "")
        rw = rewrite_query(raw_q, mode=rewrite_mode)
        search_q = rw.rewritten or raw_q
        hits = kb.search(
            search_q,
            top_k=top_k,
            expand_neighbors=int(expand_neighbors),
            mode=retrieval_mode,
            rerank=rerank_mode,
            rerank_candidates=rerank_candidates,
        )
        if level == "chunk":
            ranked = [h.chunk.id for h in hits]
        else:
            ranked = [h.chunk.source for h in hits]
        unique_ranked = _unique(ranked)
        row = {
            "id": item["id"],
            "query": item["query"],
            "query_rewritten": search_q,
            "query_rewrite": rw.as_dict(),
            "level": level,
            "relevant": sorted(relevant),
            "retrieved": ranked,
            "retrieved_unique": unique_ranked,
            "retrieved_docs": _unique([h.chunk.source for h in hits]),
            "p_at_k": precision_at_k(ranked, relevant, top_k),
            "r_at_k": recall_at_k(ranked, relevant, top_k),
            "hit_at_k": hit_at_k(ranked, relevant, top_k),
            "ndcg_at_k": ndcg_at_k(ranked, relevant, top_k),
            "mrr": mrr(ranked, relevant, top_k),
            "top_titles": [h.chunk.title for h in hits[:top_k]],
            "tags": item.get("tags") or [],
        }
        # 兼容旧字段名
        if level == "doc":
            row["retrieved_sources"] = ranked
            row["retrieved_docs"] = unique_ranked
        results.append(row)

    wall = round(time.perf_counter() - t0, 2)
    n = len(results) or 1
    mean_p = round(sum(r["p_at_k"] for r in results) / n, 4)
    mean_r = round(sum(r["r_at_k"] for r in results) / n, 4)
    mean_hit = round(sum(r["hit_at_k"] for r in results) / n, 4)
    mean_ndcg = round(sum(r["ndcg_at_k"] for r in results) / n, 4)
    mean_mrr = round(sum(r["mrr"] for r in results) / n, 4)

    try:
        qrels_rel = str(qrels.relative_to(ROOT))
    except ValueError:
        qrels_rel = str(qrels)

    payload = attach_run_meta(
        {
            "tag": tag,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "suite": "retrieval",
            "level": level,
            "expand_neighbors": int(expand_neighbors),
            "qrels": qrels_rel,
            "qrels_version": spec.get("version"),
            "k": top_k,
            "rewrite": rewrite_mode,
            "retrieval_mode": retrieval_mode,
            "rerank": rerank_mode,
            "rerank_candidates": rerank_candidates,
            "n_queries": len(results),
            "wall_seconds": wall,
            "mean_p_at_k": mean_p,
            "mean_r_at_k": mean_r,
            "mean_hit_at_k": mean_hit,
            "mean_ndcg_at_k": mean_ndcg,
            "mean_mrr": mean_mrr,
            "results": results,
        },
        method=method,
        notes=notes,
        code=[
            "src/kb/__init__.py",
            "src/kb/rewrite.py",
            "src/kb/keyword.py",
            "src/kb/hybrid.py",
            "src/kb/rerank.py",
            "scripts/ingest_kb.py",
            qrels_rel,
        ],
        compare_to=compare_tag,
    )

    BASELINES.mkdir(parents=True, exist_ok=True)
    hist_path = write_baseline(BASELINES, f"retrieval_{tag}", payload)
    print(f"Wrote {hist_path}")
    print(f"Latest alias {BASELINES / f'retrieval_{tag}.json'}")
    print(f"Method: {method}")
    print(
        f"  level={level} rewrite={rewrite_mode} retrieval={retrieval_mode} "
        f"rerank={rerank_mode} expand_neighbors={expand_neighbors}  "
        f"mean P@{top_k}={mean_p}  R@{top_k}={mean_r}  Hit@{top_k}={mean_hit}  "
        f"nDCG@{top_k}={mean_ndcg}  MRR={mean_mrr}  n={len(results)}"
    )
    for r in results:
        shown = r.get("retrieved_unique") or r.get("retrieved_docs") or []
        print(
            f"  [{r['id']}] P={r['p_at_k']} R={r['r_at_k']} Hit={r['hit_at_k']} "
            f"nDCG={r['ndcg_at_k']} MRR={r['mrr']} top={shown[:5]}"
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
                    "delta_mean_ndcg_at_k": round(
                        mean_ndcg - float(baseline.get("mean_ndcg_at_k") or 0), 4
                    ),
                    "delta_mean_mrr": round(mean_mrr - float(baseline.get("mean_mrr") or 0), 4),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return hist_path


def main() -> None:
    p = argparse.ArgumentParser(description="NEV KB retrieval eval (P@k / R@k / Hit@k)")
    p.add_argument("--tag", required=True, help="snapshot name")
    p.add_argument("--k", type=int, default=None, help="override qrels k")
    p.add_argument("--compare", default=None, help="previous retrieval tag")
    p.add_argument("--qrels", type=Path, default=None, help="金标路径")
    p.add_argument(
        "--level",
        choices=["doc", "chunk"],
        default=None,
        help="doc=文件级（默认旧行为）；chunk=chunk id 级。可被 qrels.level 覆盖",
    )
    p.add_argument(
        "--expand-neighbors",
        type=int,
        default=None,
        help="邻块扩展半径；chunk 级默认 0，doc 级默认 1",
    )
    p.add_argument(
        "--method",
        default="本地 JSON 余弦检索 P@k / R@k / Hit@k，不跑对话 LLM",
        help="这次测的检索方法",
    )
    p.add_argument("--notes", default="", help="补充说明")
    p.add_argument(
        "--rewrite",
        default=None,
        choices=["rules", "llm", "off"],
        help="查询改写：rules（默认，可用环境变量 KB_QUERY_REWRITE）| llm | off",
    )
    p.add_argument(
        "--retrieval",
        default=None,
        choices=["vector", "keyword", "hybrid"],
        help="检索模式：vector（默认）| keyword（BM25）| hybrid（向量+BM25 RRF）；"
        "也可用环境变量 KB_RETRIEVAL_MODE",
    )
    p.add_argument(
        "--rerank",
        default=None,
        choices=["off", "lexical", "gte"],
        help="重排：off（默认）| lexical（本地）| gte（DashScope）；KB_RERANK",
    )
    p.add_argument(
        "--rerank-candidates",
        type=int,
        default=None,
        help="重排前初检条数（默认 max(k*4, 32)；KB_RERANK_CANDIDATES）",
    )
    args = p.parse_args()

    qrels = args.qrels
    level = args.level
    if qrels is not None:
        qp = qrels if qrels.is_absolute() else (ROOT / qrels)
        if qp.exists() and level is None:
            spec = _load_yaml(qp)
            level = str(spec.get("level") or "doc")
    level = level or "doc"

    try:
        run_suite(
            tag=args.tag,
            k=args.k,
            compare_tag=args.compare,
            method=args.method,
            notes=args.notes,
            qrels_path=args.qrels,
            level=level,
            expand_neighbors=args.expand_neighbors,
            rewrite=args.rewrite,
            retrieval=args.retrieval,
            rerank=args.rerank,
            rerank_candidates=args.rerank_candidates,
        )
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
