"""爬取待审 / 入库助手的离线单测。"""

from __future__ import annotations

from pathlib import Path

from kb import _strip_frontmatter
from kb.clean import clean_page_text
from kb.crawl import load_config, parse_frontmatter, promote_approved, write_staging_doc


def test_strip_frontmatter():
    raw = "---\nurl: https://x.test\nstatus: approved\n---\n\n# Hi\n\nbody\n"
    assert _strip_frontmatter(raw).startswith("# Hi")


def test_clean_page_removes_nav_keeps_facts():
    noisy = """
首页
党政
党网 · 时政
经济 · 科技

全球电动汽车市场进入加速增长新阶段

国际能源署最新发布的《全球电动汽车展望2026》报告预计，2026年全球电动汽车销量将达到2300万辆，占全球新车销量近30%。

分享
打印
版权所有 人民网
"""
    out = clean_page_text(
        noisy,
        title="全球电动汽车市场进入加速增长新阶段（国际视点） --经济·科技--人民网",
        query="全球电动汽车 销量 展望 2026",
        snippet="国际能源署最新发布的《全球电动汽车展望2026》报告预计",
    )
    assert "首页" not in out
    assert "党政" not in out
    assert "版权所有" not in out
    assert "2300万辆" in out
    assert out.lstrip().startswith("# ")
    assert "经济·科技--人民网" not in out.splitlines()[0]


def test_load_config_is_queries_not_urls(tmp_path: Path):
    cfg = tmp_path / "q.yaml"
    cfg.write_text(
        "defaults:\n  top_k: 3\nqueries:\n"
        "  - id: a\n    query: 比亚迪 海外 2024\n    top_k: 2\n",
        encoding="utf-8",
    )
    defaults, queries = load_config(cfg)
    assert defaults["top_k"] == 3
    assert queries[0]["query"] == "比亚迪 海外 2024"
    assert "url" not in queries[0]


def test_write_and_promote(tmp_path: Path):
    staging = tmp_path / "staging"
    docs = tmp_path / "kb"
    path = write_staging_doc(
        doc_id="demo",
        url="https://example.com/a",
        title="演示标题",
        body="## 小节\n\nhello world " * 20,
        query="比亚迪 海外",
        query_id="byd_overseas",
        staging_dir=staging,
    )
    assert path.exists()
    meta, _ = parse_frontmatter(path.read_text(encoding="utf-8"))
    assert meta["status"] == "pending_review"
    assert meta["query"] == "比亚迪 海外"
    assert meta["url"] == "https://example.com/a"

    assert promote_approved(staging_dir=staging, docs_dir=docs) == []
    written = promote_approved(staging_dir=staging, docs_dir=docs, force_ids=["demo"])
    assert len(written) == 1
    assert written[0].parent.name == "crawl"
