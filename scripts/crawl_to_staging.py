#!/usr/bin/env python3
"""按搜索词检索并抓取候选页 → data/kb_staging/（待审）。不会直接入库。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from kb.crawl import crawl_by_queries  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="按搜索词扩库爬取 → 待审目录")
    p.add_argument("--config", type=Path, default=None, help="search_queries.yaml 路径")
    p.add_argument("--staging", type=Path, default=None, help="待审目录")
    p.add_argument(
        "--include-optional",
        action="store_true",
        help="也跑标记为 optional 的查询",
    )
    args = p.parse_args()
    results = crawl_by_queries(
        config_path=args.config,
        staging_dir=args.staging,
        include_optional=args.include_optional,
    )
    ok = sum(1 for r in results if r.get("ok"))
    print(f"抓取成功 {ok}/{len(results)} 条 → 待审目录")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    if results and ok == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
