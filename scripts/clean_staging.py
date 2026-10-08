#!/usr/bin/env python3
"""对 data/kb_staging/ 已有稿件重新清洗并格式化（不重新联网）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from agent.paths import KB_STAGING  # noqa: E402
from kb.clean import rebuild_staging_markdown  # noqa: E402
from kb.crawl import parse_frontmatter  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="清洗待审稿：去噪音、保留关键、Markdown 化")
    p.add_argument("--staging", type=Path, default=None)
    p.add_argument(
        "--only-pending",
        action="store_true",
        help="只处理 status=pending_review",
    )
    args = p.parse_args()
    staging = args.staging or KB_STAGING
    if not staging.exists():
        print("待审目录不存在")
        return
    n = 0
    for path in sorted(staging.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        try:
            meta, body = parse_frontmatter(raw)
        except Exception as exc:  # noqa: BLE001
            print(f"跳过 {path.name}: 元数据损坏 ({exc})")
            continue
        if not meta:
            continue
        if args.only_pending and meta.get("status") != "pending_review":
            continue
        if meta.get("status") == "error":
            continue
        # body 含引用头；清洗时去掉引用行，只留正文
        body_lines = []
        for ln in body.splitlines():
            if ln.startswith("> "):
                continue
            body_lines.append(ln)
        core = "\n".join(body_lines).strip()
        new_text = rebuild_staging_markdown(meta=meta, raw_body=core)
        path.write_text(new_text, encoding="utf-8")
        n += 1
        print(f"已清洗: {path.name} ({len(core)}→{len(new_text)} 字节)")
    print(f"共处理 {n} 篇")


if __name__ == "__main__":
    main()
