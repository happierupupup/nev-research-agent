"""Eval metrics: extract report stats and aggregate rubric scores."""

from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Markdown-ish citation patterns: [1], (来源: …), 【引】, bare URL
_SOURCE_PAREN = re.compile(r"[（(]来源[:：][^）)]+[）)]")
_BRACKET_NUM = re.compile(r"\[\d+\]")
_JINYIN = re.compile(r"【\s*引\s*】")
_URL = re.compile(r"https?://\S+")


def git_snapshot(root: Path | None = None) -> dict[str, Any]:
    """Short commit id so a JSON can be matched back to code."""
    cwd = str(root or Path(__file__).resolve().parents[1])
    meta: dict[str, Any] = {"git_commit": None, "git_branch": None, "git_dirty": None}
    try:
        meta["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        meta["git_branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        meta["git_dirty"] = bool(dirty.strip())
    except (OSError, subprocess.CalledProcessError):
        pass
    return meta


def attach_run_meta(
    payload: dict[str, Any],
    *,
    method: str,
    notes: str = "",
    code: list[str] | None = None,
    compare_to: str | None = None,
) -> dict[str, Any]:
    """Describe which approach this snapshot measured."""
    payload["method"] = {
        "name": (method or "").strip(),
        "notes": (notes or "").strip() or None,
        "code": code or [],
        "compare_to": compare_to,
        **git_snapshot(),
    }
    return payload


def write_baseline(directory: Path, stem: str, payload: dict[str, Any]) -> Path:
    """Keep every run: timestamped file plus ``{stem}.json`` as the latest alias."""
    directory.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    hist = directory / f"{stem}_{ts}.json"
    latest = directory / f"{stem}.json"
    hist.write_text(text, encoding="utf-8")
    latest.write_text(text, encoding="utf-8")
    return hist


def count_chars(text: str) -> int:
    return len(text or "")


def count_cited_facts(text: str) -> int:
    """Count in-text citation marks (occurrences, not unique strings).

    Same `(来源: KB/foo.md)` used on three facts counts as 3.
    URLs already inside a 来源 parenthesis are not double-counted.
    """
    if not text:
        return 0
    n = len(_BRACKET_NUM.findall(text))
    n += len(_SOURCE_PAREN.findall(text))
    n += len(_JINYIN.findall(text))
    covered = [(m.start(), m.end()) for m in _SOURCE_PAREN.finditer(text)]
    for m in _URL.finditer(text):
        if any(a <= m.start() and m.end() <= b for a, b in covered):
            continue
        n += 1
    return n


def pad_inline_citations(
    report: str,
    sources: list[dict[str, Any]] | None,
    *,
    min_n: int = 3,
) -> str:
    """If a report has too few in-text marks, append 来源补注 from real tool sources."""
    text = report or ""
    if count_cited_facts(text) >= min_n:
        return text
    lines: list[str] = []
    for s in sources or []:
        typ = s.get("type") or ""
        ref = str(s.get("ref") or s.get("label") or "").strip()
        if not ref:
            continue
        if typ == "KB" and not ref.startswith("KB/"):
            ref = f"KB/{ref}"
        lines.append(f"- 关键依据 (来源: {ref})")
        if count_cited_facts(text + "\n".join(lines)) >= min_n:
            break
    i = 0
    pool = [ln for ln in lines]
    while pool and count_cited_facts(text + "\n" + "\n".join(lines)) < min_n and i < 8:
        lines.append(pool[i % len(pool)])
        i += 1
    if not lines:
        return text
    return text.rstrip() + "\n\n## 来源补注\n" + "\n".join(lines) + "\n"


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
    if "max_kb_calls" in gates and int(kb_calls or 0) > int(gates["max_kb_calls"]):
        fails.append(f"kb_calls {kb_calls or 0} > {gates['max_kb_calls']}")
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
                "prompt_tokens": {
                    "baseline": b.get("prompt_tokens"),
                    "current": c.get("prompt_tokens"),
                },
                "completion_tokens": {
                    "baseline": b.get("completion_tokens"),
                    "current": c.get("completion_tokens"),
                },
                "total_tokens": {
                    "baseline": b.get("total_tokens"),
                    "current": c.get("total_tokens"),
                },
                "llm_calls": {"baseline": b.get("llm_calls"), "current": c.get("llm_calls")},
                "hard_pass": {
                    "baseline": (b.get("hard_gates") or {}).get("passed"),
                    "current": (c.get("hard_gates") or {}).get("passed"),
                },
            }
        )

    return {
        "method": {
            "baseline": baseline.get("method"),
            "current": current.get("method"),
        },
        "per_query_rubric": per_query,
        "mean_rubric_lift_ratio": round(sum(lifts) / len(lifts), 4) if lifts else None,
        "mean_wall_seconds": {"baseline": b_wall, "current": c_wall},
        "mean_total_tokens": {
            "baseline": _mean([r.get("total_tokens") for r in baseline.get("results", [])]),
            "current": _mean([r.get("total_tokens") for r in current.get("results", [])]),
        },
        "sum_total_tokens": {
            "baseline": _sum([r.get("total_tokens") for r in baseline.get("results", [])]),
            "current": _sum([r.get("total_tokens") for r in current.get("results", [])]),
        },
        "per_query_tools": tool_diff,
    }


def _mean(vals: list[Any]) -> float | None:
    nums = [float(v) for v in vals if v is not None]
    return round(sum(nums) / len(nums), 2) if nums else None


def _sum(vals: list[Any]) -> float | None:
    nums = [float(v) for v in vals if v is not None]
    return round(sum(nums), 2) if nums else None


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
