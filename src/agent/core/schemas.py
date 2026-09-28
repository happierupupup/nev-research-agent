"""Shared schemas for S2 (Issue / Plan). Verifier must emit structured issues."""

from __future__ import annotations

import json
import re
from typing import Any, Literal, TypedDict

Severity = Literal["critical", "minor"]
IssueType = Literal["citation", "coverage", "logic", "data", "other"]


class Issue(TypedDict, total=False):
    severity: Severity
    type: IssueType
    message: str
    citation: str


class ResearchPlan(TypedDict, total=False):
    mode: Literal["qa", "report"]
    need_web: bool
    subqueries: list[str]
    focus: list[str]


_JSON_BLOCK = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.I)


def parse_issues(raw: str) -> list[Issue]:
    """Parse Verifier JSON into Issue list. Tolerates fenced blocks."""
    text = (raw or "").strip()
    if not text:
        return []
    m = _JSON_BLOCK.search(text)
    if m:
        text = m.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return []
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return []
    items = data.get("issues") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return []
    out: list[Issue] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        sev = str(item.get("severity") or "minor").lower()
        if sev not in ("critical", "minor"):
            sev = "minor"
        typ = str(item.get("type") or "other").lower()
        if typ not in ("citation", "coverage", "logic", "data", "other"):
            typ = "other"
        msg = str(item.get("message") or "").strip()
        if not msg:
            continue
        issue: Issue = {
            "severity": sev,  # type: ignore[typeddict-item]
            "type": typ,  # type: ignore[typeddict-item]
            "message": msg,
        }
        cite = str(item.get("citation") or "").strip()
        if cite:
            issue["citation"] = cite
        out.append(issue)
    return out


def critical_issues(issues: list[Issue] | list[dict[str, Any]]) -> list[Issue]:
    return [i for i in issues if str(i.get("severity") or "").lower() == "critical"]  # type: ignore[return-value]


def issues_to_prompt(issues: list[Issue] | list[dict[str, Any]]) -> str:
    if not issues:
        return "(无)"
    lines = []
    for i, issue in enumerate(issues, 1):
        cite = issue.get("citation") or ""
        extra = f" | 引用线索: {cite}" if cite else ""
        lines.append(
            f"{i}. [{issue.get('severity')}/{issue.get('type')}] {issue.get('message')}{extra}"
        )
    return "\n".join(lines)
