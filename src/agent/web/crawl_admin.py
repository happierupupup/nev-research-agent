"""Admin helpers: list/promote staging, trigger crawl."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from agent.paths import KB_STAGING
from kb.crawl import crawl_by_queries, crawl_manual, parse_frontmatter, promote_approved


def _valid_date(s: str) -> bool:
    if not s:
        return True
    try:
        datetime.strptime(s, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def list_staging(staging_dir: Path | None = None) -> list[dict[str, Any]]:
    staging = staging_dir or KB_STAGING
    if not staging.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(staging.glob("*.md")):
        try:
            meta, _ = parse_frontmatter(path.read_text(encoding="utf-8"))
        except OSError:
            continue
        rows.append(
            {
                "id": str(meta.get("id") or path.stem),
                "status": str(meta.get("status") or "pending"),
                "url": str(meta.get("url") or ""),
                "title": str(meta.get("title") or path.stem),
                "query": str(meta.get("query") or meta.get("search_query") or ""),
                "chars": path.stat().st_size,
                "path": str(path.name),
            }
        )
    return rows


def run_manual_crawl(
    *,
    keyword: str,
    date_from: str = "",
    date_to: str = "",
    top_k: int = 4,
) -> list[dict[str, Any]]:
    kw = (keyword or "").strip()
    if not kw:
        raise ValueError("关键词不能为空")
    df = (date_from or "").strip()
    dt = (date_to or "").strip()
    if not _valid_date(df) or not _valid_date(dt):
        raise ValueError("日期格式须为 YYYY-MM-DD")
    return crawl_manual(
        keyword=kw,
        date_from=df or None,
        date_to=dt or None,
        top_k=max(1, min(int(top_k), 8)),
    )


def run_config_crawl(*, include_optional: bool = False) -> list[dict[str, Any]]:
    return crawl_by_queries(include_optional=include_optional, crawl_mode="config_yaml")


def approve_and_promote(doc_ids: list[str]) -> list[str]:
    ids = [i.strip() for i in doc_ids if (i or "").strip()]
    if not ids:
        raise ValueError("请提供待通过的文档 id")
    staging = KB_STAGING
    if staging.exists():
        for path in staging.glob("*.md"):
            raw = path.read_text(encoding="utf-8")
            meta, body = parse_frontmatter(raw)
            sid = str(meta.get("id") or path.stem)
            if sid in ids or path.stem in ids:
                meta["status"] = "approved"
                fm = yaml.safe_dump(meta, allow_unicode=True, sort_keys=False).strip()
                path.write_text(f"---\n{fm}\n---\n\n{body.lstrip()}", encoding="utf-8")
    written = promote_approved(force_ids=ids)
    return [str(p) for p in written]
