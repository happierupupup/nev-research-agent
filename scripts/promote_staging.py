#!/usr/bin/env python3
"""Promote approved staging docs into data/kb/crawl/ then optionally re-ingest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from kb.crawl import parse_frontmatter, promote_approved  # noqa: E402
from agent.paths import KB_STAGING  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(
        description="Promote approved (or --approve ID) staging → data/kb/crawl/"
    )
    p.add_argument(
        "--approve",
        action="append",
        default=[],
        help="force-promote staging id/stem (can repeat); also marks as approved",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="list staging docs and status",
    )
    p.add_argument(
        "--ingest",
        action="store_true",
        help="after promote, run full KB ingest (needs embedding API)",
    )
    args = p.parse_args()

    if args.list or not args.approve:
        staging = KB_STAGING
        if not staging.exists():
            print("No staging dir yet. Run: python scripts/crawl_to_staging.py")
            return
        print("Staging:")
        for path in sorted(staging.glob("*.md")):
            meta, _ = parse_frontmatter(path.read_text(encoding="utf-8"))
            print(
                f"  {meta.get('id') or path.stem}: status={meta.get('status')} "
                f"url={meta.get('url')} chars≈{path.stat().st_size}"
            )
        if args.list and not args.approve:
            return
        if not args.approve:
            print(
                "\nTo promote: edit frontmatter status → approved, then re-run "
                "promote_staging.py --approve <id>\n"
                "Or: python scripts/promote_staging.py --approve <id> [--ingest]"
            )
            return

    # Mark forced IDs approved in staging files first
    for path in KB_STAGING.glob("*.md"):
        raw = path.read_text(encoding="utf-8")
        meta, body = parse_frontmatter(raw)
        sid = str(meta.get("id") or path.stem)
        if sid in args.approve or path.stem in args.approve:
            meta["status"] = "approved"
            import yaml

            fm = yaml.safe_dump(meta, allow_unicode=True, sort_keys=False).strip()
            path.write_text(f"---\n{fm}\n---\n\n{body.lstrip()}", encoding="utf-8")

    written = promote_approved(force_ids=args.approve)
    print(f"Promoted {len(written)} file(s):")
    for w in written:
        print(f"  {w}")

    if args.ingest and written:
        from kb import ingest

        meta = ingest()
        print(f"Ingested {meta['n_docs']} docs → {meta['n_chunks']} chunks")


if __name__ == "__main__":
    main()
