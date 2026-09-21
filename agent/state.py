"""
LangGraph state — partitioned workspaces (do not flatten into one bag).

Phase 1 may only fill query / research_workspace / final_report;
Phase 4 fills plan / analysis / verification / iteration.
"""

from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class ResearchState(TypedDict, total=False):
    query: str
    research_plan: Dict[str, Any]
    research_workspace: List[Dict[str, Any]]
    analysis_workspace: Dict[str, Any]
    verification_issues: List[Dict[str, Any]]
    current_phase: str
    iteration_count: int
    final_report: str
