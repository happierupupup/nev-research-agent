"""Eval metrics: extract report stats and aggregate rubric scores."""

from __future__ import annotations

import re
from typing import Any

# Markdown-ish citation patterns: [1], (来源: …), 【引】…
_CITATION_PATTERNS = [
    re.compile(r"\[\d+\]"),
    re.compile(r"[（(]来源[:：][^）)]+[）)]"),
    re.compile(r"【\s*引\s*】"),
    re.compile(r"https?://\S+"),
]


def count_chars(text: str) -> int:
    return len(text or "")


def count_cited_facts(text: str) -> int:
    """Heuristic: unique citation markers in report (Phase 1 proxy)."""
    if not text:
        return 0
    found: set[str] = set()
    for pat in _CITATION_PATTERNS:
        for m in pat.finditer(text):
            found.add(m.group(0))
    return len(found)


def weighted_rubric_score(
    dimension_scores: dict[str, float],
    weights: dict[str, float],
) -> float:
    """dimension_scores: {dim_id: 0-100}; weights: {dim_id: weight}."""
    total_w = sum(weights.values()) or 1.0
    acc = 0.0
    for dim_id, w in weights.items():
        acc += float(dimension_scores.get(dim_id, 0.0)) * w
    return round(acc / total_w * total_w, 2)  # keep 0-100 scale if weights sum to 1


def check_hard_gates(
    *,
    chars: int,
    cited_facts: int,
    wall_seconds: float,
    gates: dict[str, Any],
    kb_calls: int | None = None,
    web_calls: int | None = None,
) -> dict[str, Any]:
    fails: list[str] = []
    if "min_chars" in gates and chars < int(gates["min_chars"]):
        fails.append(f"chars {chars} < {gates['min_chars']}")
    if "min_cited_facts" in gates and cited_facts < int(gates["min_cited_facts"]):
        fails.append(f"cited_facts {cited_facts} < {gates['min_cited_facts']}")
    if "max_wall_seconds" in gates and wall_seconds > float(gates["max_wall_seconds"]):
        fails.append(f"wall_seconds {wall_seconds:.1f} > {gates['max_wall_seconds']}")
    if "min_kb_calls" in gates and int(kb_calls or 0) < int(gates["min_kb_calls"]):
        fails.append(f"kb_calls {kb_calls or 0} < {gates['min_kb_calls']}")
    if "min_web_calls" in gates and int(web_calls or 0) < int(gates["min_web_calls"]):
        fails.append(f"web_calls {web_calls or 0} < {gates['min_web_calls']}")
    if "max_web_calls" in gates and int(web_calls or 0) > int(gates["max_web_calls"]):
        fails.append(f"web_calls {web_calls or 0} > {gates['max_web_calls']}")
    return {"passed": not fails, "failures": fails}


def compare_runs(baseline: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    """Diff suite-level aggregates for before/after."""
    b_scores = {
        r["id"]: r.get("rubric_total")
        for r in baseline.get("results", [])
        if r.get("rubric_total") is not None
    }
    c_scores = {
        r["id"]: r.get("rubric_total")
        for r in current.get("results", [])
        if r.get("rubric_total") is not None
    }
    per_query: list[dict[str, Any]] = []
    lifts: list[float] = []
    for qid in sorted(set(b_scores) | set(c_scores)):
        b, c = b_scores.get(qid), c_scores.get(qid)
        delta = None if b is None or c is None else round(c - b, 2)
        if delta is not None and b:
            lifts.append(delta / b)
        per_query.append({"id": qid, "baseline": b, "current": c, "delta": delta})

    b_wall = _mean([r.get("wall_seconds") for r in baseline.get("results", [])])
    c_wall = _mean([r.get("wall_seconds") for r in current.get("results", [])])

    bmap = {r["id"]: r for r in baseline.get("results", [])}
    cmap = {r["id"]: r for r in current.get("results", [])}
    tool_diff: list[dict[str, Any]] = []
    for qid in sorted(set(bmap) | set(cmap)):
        b, c = bmap.get(qid, {}), cmap.get(qid, {})
        tool_diff.append(
            {
                "id": qid,
                "kb_calls": {"baseline": b.get("kb_calls"), "current": c.get("kb_calls")},
                "web_calls": {"baseline": b.get("web_calls"), "current": c.get("web_calls")},
                "cited_facts": {"baseline": b.get("cited_facts"), "current": c.get("cited_facts")},
                "wall_seconds": {"baseline": b.get("wall_seconds"), "current": c.get("wall_seconds")},
                "hard_pass": {
                    "baseline": (b.get("hard_gates") or {}).get("passed"),
                    "current": (c.get("hard_gates") or {}).get("passed"),
                },
            }
        )

    return {
        "per_query_rubric": per_query,
        "mean_rubric_lift_ratio": round(sum(lifts) / len(lifts), 4) if lifts else None,
        "mean_wall_seconds": {"baseline": b_wall, "current": c_wall},
        "per_query_tools": tool_diff,
    }


def _mean(vals: list[Any]) -> float | None:
    nums = [float(v) for v in vals if v is not None]
    return round(sum(nums) / len(nums), 2) if nums else None


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """retrieved: ranked ids (e.g. chunk sources); relevant: gold set."""
    if k <= 0:
        return 0.0
    top = retrieved[:k]
    hits = sum(1 for item in top if item in relevant)
    return round(hits / k, 4)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = set(retrieved[:k])
    return round(len(top & relevant) / len(relevant), 4)


def hit_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return 1.0 if set(retrieved[:k]) & relevant else 0.0
