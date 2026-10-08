#!/usr/bin/env python3
"""
扩库爬取 → data/kb_staging/（待审）。不会直接入库。

版本：
  v1 批量配置：--from-config（默认）读 data/crawl/search_queries.yaml
  v2 管理员手动：--admin 交互输入关键词 + 时间范围；或 --keyword / --from / --to
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from agent.auth import is_admin, resolve_user  # noqa: E402
from agent.memory import user_id_from_env  # noqa: E402
from kb.crawl import crawl_by_queries, crawl_manual  # noqa: E402

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _valid_date(s: str) -> bool:
    if not s:
        return True
    if not _DATE_RE.match(s):
        return False
    try:
        datetime.strptime(s, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def _require_admin(user_id: str | None) -> None:
    uid = user_id_from_env(user_id)
    user = resolve_user(uid)
    if not is_admin(user):
        print(
            f"拒绝：爬虫为管理员能力。当前用户 `{user.id}` role={user.role}。"
            "请用 --user admin 或设 NEV_USER_ID=admin。",
            file=sys.stderr,
        )
        raise SystemExit(2)


def _prompt_admin() -> tuple[str, str, str, int]:
    print("管理员手动扩库（关键词 + 时间范围）")
    print("提示：时间会写入检索词做偏置，不是严格日历过滤。\n")
    keyword = input("关键词: ").strip()
    if not keyword:
        raise SystemExit("关键词不能为空")
    date_from = input("开始日期 YYYY-MM-DD（可空）: ").strip()
    date_to = input("结束日期 YYYY-MM-DD（可空）: ").strip()
    if not _valid_date(date_from) or not _valid_date(date_to):
        raise SystemExit("日期格式须为 YYYY-MM-DD")
    top_raw = input("每词条数 top_k [4]: ").strip() or "4"
    try:
        top_k = max(1, min(int(top_raw), 8))
    except ValueError:
        top_k = 4
    return keyword, date_from, date_to, top_k


def main() -> None:
    p = argparse.ArgumentParser(
        description="扩库爬取 → 待审目录（v1 配置批量 / v2 管理员手动）"
    )
    p.add_argument(
        "--from-config",
        action="store_true",
        help="v1：按 search_queries.yaml 批量爬（默认，若不给 --admin/--keyword）",
    )
    p.add_argument("--config", type=Path, default=None, help="search_queries.yaml 路径")
    p.add_argument("--staging", type=Path, default=None, help="待审目录")
    p.add_argument(
        "--include-optional",
        action="store_true",
        help="v1：也跑标记为 optional 的查询",
    )
    p.add_argument(
        "--admin",
        action="store_true",
        help="v2：交互输入关键词与时间范围（需管理员）",
    )
    p.add_argument("--keyword", default=None, help="v2：关键词（非交互）")
    p.add_argument("--from", dest="date_from", default=None, help="v2：开始日期 YYYY-MM-DD")
    p.add_argument("--to", dest="date_to", default=None, help="v2：结束日期 YYYY-MM-DD")
    p.add_argument("--top-k", type=int, default=4, help="v2：每词取链条数")
    p.add_argument(
        "--user",
        default=None,
        help="操作用户 id（默认 NEV_USER_ID）；管理员操作用 admin",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="跳过角色检查（仅本地排障）",
    )
    args = p.parse_args()

    manual = bool(args.admin or args.keyword)
    if manual and not args.force:
        _require_admin(args.user)

    if manual:
        if args.admin and not args.keyword:
            keyword, date_from, date_to, top_k = _prompt_admin()
        else:
            keyword = (args.keyword or "").strip()
            if not keyword:
                raise SystemExit("--keyword 不能为空（或改用 --admin 交互）")
            date_from = (args.date_from or "").strip()
            date_to = (args.date_to or "").strip()
            if not _valid_date(date_from) or not _valid_date(date_to):
                raise SystemExit("日期格式须为 YYYY-MM-DD")
            top_k = max(1, min(int(args.top_k), 8))
        results = crawl_manual(
            keyword=keyword,
            date_from=date_from or None,
            date_to=date_to or None,
            top_k=top_k,
            staging_dir=args.staging,
        )
        mode = "admin_manual"
    else:
        # v1 默认：配置文件批量（体现版本迭代，保留）
        results = crawl_by_queries(
            config_path=args.config,
            staging_dir=args.staging,
            include_optional=args.include_optional,
            crawl_mode="config_yaml",
        )
        mode = "config_yaml"

    ok = sum(1 for r in results if r.get("ok"))
    print(f"[{mode}] 抓取成功 {ok}/{len(results)} 条 → 待审目录")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    if results and ok == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
