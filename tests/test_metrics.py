"""Unit tests for eval metrics (no LLM required)."""

from __future__ import annotations

from evals.metrics import (
    check_hard_gates,
    compare_runs,
    count_chars,
    count_cited_facts,
    hit_at_k,
    precision_at_k,
    recall_at_k,
    weighted_rubric_score,
)


def test_count_cited_facts():
    text = "销量增长[1]。另一说法[2]。来源见 https://example.com/a"
    assert count_cited_facts(text) >= 3


def test_count_cited_facts_source_paren():
    text = "刀片电池针刺不起火 (来源: KB/byd_blade_battery.md)。渗透率约 40% (来源: https://example.com/a)"
    assert count_cited_facts(text) >= 2


def test_hard_gates_pass():
    r = check_hard_gates(
        chars=6000,
        cited_facts=5,
        wall_seconds=100,
        gates={"min_chars": 5000, "min_cited_facts": 5, "max_wall_seconds": 300},
    )
    assert r["passed"] is True


def test_hard_gates_fail_kb_calls():
    r = check_hard_gates(
        chars=200,
        cited_facts=2,
        wall_seconds=10,
        kb_calls=0,
        web_calls=0,
        gates={"min_chars": 80, "min_cited_facts": 1, "min_kb_calls": 1, "max_web_calls": 2},
    )
    assert r["passed"] is False
    assert any("kb_calls" in f for f in r["failures"])


def test_compare_runs_includes_tools():
    baseline = {
        "results": [
            {"id": "Q1", "rubric_total": None, "wall_seconds": 9, "kb_calls": 0, "web_calls": 0},
        ]
    }
    current = {
        "results": [
            {"id": "Q1", "rubric_total": None, "wall_seconds": 12, "kb_calls": 1, "web_calls": 0},
        ]
    }
    diff = compare_runs(baseline, current)
    assert diff["per_query_tools"][0]["kb_calls"]["current"] == 1


def test_hard_gates_fail_chars():
    r = check_hard_gates(
        chars=100,
        cited_facts=5,
        wall_seconds=10,
        gates={"min_chars": 5000, "min_cited_facts": 5, "max_wall_seconds": 300},
    )
    assert r["passed"] is False
    assert any("chars" in f for f in r["failures"])


def test_weighted_rubric():
    weights = {"D1": 0.2, "D2": 0.25, "D3": 0.2, "D4": 0.2, "D5": 0.15}
    scores = {"D1": 80, "D2": 80, "D3": 80, "D4": 80, "D5": 80}
    assert weighted_rubric_score(scores, weights) == 80.0


def test_compare_runs_lift():
    baseline = {
        "results": [
            {"id": "G1", "rubric_total": 50.0, "wall_seconds": 200},
            {"id": "G2", "rubric_total": 50.0, "wall_seconds": 200},
        ]
    }
    current = {
        "results": [
            {"id": "G1", "rubric_total": 60.0, "wall_seconds": 180},
            {"id": "G2", "rubric_total": 60.0, "wall_seconds": 180},
        ]
    }
    diff = compare_runs(baseline, current)
    assert diff["mean_rubric_lift_ratio"] == 0.2


def test_retrieval_metrics():
    retrieved = ["a.md", "b.md", "c.md", "d.md"]
    relevant = {"a.md", "x.md"}
    assert precision_at_k(retrieved, relevant, 4) == 0.25
    assert recall_at_k(retrieved, relevant, 4) == 0.5
    assert hit_at_k(retrieved, relevant, 4) == 1.0
    assert hit_at_k(["z.md"], relevant, 4) == 0.0
    assert precision_at_k([], relevant, 4) == 0.0
