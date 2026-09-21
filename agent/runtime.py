"""
Runtime entry used by main.py and evals/run_eval.py.

Phase 1: implement real graph here. Until then ImportError → eval dry-run.
"""

from __future__ import annotations

from typing import Any


def run_research(query: str) -> dict[str, Any]:
    """
    Run end-to-end research. Must return at least:
      final_report, retrieval_rounds, iteration_count
    """
    raise NotImplementedError(
        "Phase 1: wire LangGraph in agent.graph and call it from run_research"
    )
