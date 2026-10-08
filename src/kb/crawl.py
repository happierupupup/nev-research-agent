"""管理员扩库：按搜索词检索 → 抓取候选页 → staging 待审。"""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import yaml

from agent.paths import CRAWL_QUERIES, KB_STAGING


class _TextExtractor(HTMLParser):
    """优先抽取 article/main；跳过导航脚本。"""

    def __init__(self) -> None:
        super().__init__()
        self._skip = 0
        self._in_article = 0
        self._article_chunks: list[str] = []
        self._all_chunks: list[str] = []
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        t = tag.lower()
        attrs_d = {k: (v or "") for k, v in attrs}
        role = attrs_d.get("role", "").lower()
        cls = f"{attrs_d.get('class', '')} {attrs_d.get('id', '')}".lower()
        if t in {"script", "style", "noscript", "svg", "nav", "footer", "header", "aside", "form"}:
            self._skip += 1
        if t in {"article", "main"} or role == "main" or "article" in cls:
            self._in_article += 1
        if t == "title":
            self._in_title = True
        if t in {"p", "h1", "h2", "h3", "h4", "li", "br", "tr", "section"} and self._skip == 0:
            (self._article_chunks if self._in_article else self._all_chunks).append("\n")

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in {"script", "style", "noscript", "svg", "nav", "footer", "header", "aside", "form"}:
            self._skip = max(0, self._skip - 1)
        if t in {"article", "main"}:
            self._in_article = max(0, self._in_article - 1)
        # class 含 article 时 start 加过 in_article，无对称 end；靠文本长度回退全页即可
        if t == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        text = data.strip()
        if not text:
            return
        if self._in_title and not self.title:
            self.title = text
        bucket = self._article_chunks if self._in_article else self._all_chunks
        bucket.append(text + " ")

    def text(self) -> str:
        chunks = self._article_chunks if len("".join(self._article_chunks)) > 200 else self._all_chunks
        raw = "".join(chunks)
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def load_config(path: Path | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """读取搜索配置：返回 (defaults, queries)。"""
    p = path or CRAWL_QUERIES
    if not p.exists():
        return {}, []
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    defaults = data.get("defaults") if isinstance(data.get("defaults"), dict) else {}
    queries = data.get("queries") or data.get("seeds") or []
    out: list[dict[str, Any]] = []
    for q in queries:
        if not isinstance(q, dict):
            continue
        # 新格式：query 关键词；旧格式兼容：url 直抓
        if q.get("query") or q.get("url"):
            out.append(q)
    return defaults, out


def _slug(*parts: str) -> str:
    raw = "_".join(p for p in parts if p)
    safe = re.sub(r"[^\w\-]+", "_", raw, flags=re.UNICODE).strip("_")
    return (safe or "page")[:100]


def _host_ok(url: str, allow_domains: list[str]) -> bool:
    if not allow_domains:
        return True
    host = (urlparse(url).netloc or "").lower()
    return any(host == d.lower() or host.endswith("." + d.lower()) for d in allow_domains)


def html_to_text(html: str) -> tuple[str, str]:
    parser = _TextExtractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # noqa: BLE001
        pass
    return parser.title.strip(), parser.text()


def fetch_url(url: str, *, timeout: float = 30.0) -> tuple[int, str, str]:
    headers = {
        # HTTP 头须为 ASCII，勿写中文
        "User-Agent": "nev-research-agent/0.1 (+local admin crawl)",
        "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }
    with httpx.Client(follow_redirects=True, timeout=timeout, headers=headers) as client:
        resp = client.get(url)
        ctype = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
        return resp.status_code, ctype, resp.text


def write_staging_doc(
    *,
    doc_id: str,
    url: str,
    title: str,
    body: str,
    query: str = "",
    query_id: str = "",
    note: str = "",
    tags: list[str] | None = None,
    status_code: int | None = None,
    error: str | None = None,
    snippet: str = "",
    staging_dir: Path | None = None,
) -> Path:
    from kb.clean import rebuild_staging_markdown

    out_dir = staging_dir or KB_STAGING
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{_slug(doc_id)}.md"
    crawled_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    status = "error" if error else "pending_review"
    meta = {
        "id": doc_id,
        "url": url,
        "title": title or doc_id,
        "crawled_at": crawled_at,
        "status": status,
        "source_type": "web_crawl",
        "query_id": query_id,
        "query": query,
        "http_status": status_code,
        "note": note or "",
        "tags": tags or [],
        "snippet": re.sub(r"\s+", " ", snippet or "").strip()[:400],
        "error": re.sub(r"\s+", " ", error or "").strip()[:300],
    }
    if error or not (body or "").strip():
        meta["cleaned"] = False
        fm = yaml.safe_dump(meta, allow_unicode=True, sort_keys=False).strip()
        content = "_（无正文：抓取失败或页面需登录/反爬）_"
        qline = f"> 搜索词: {query}  \n" if query else ""
        text = (
            f"---\n{fm}\n---\n\n# {title or doc_id}\n\n"
            f"{qline}> 来源: {url}  \n> 抓取: {crawled_at}  \n"
            f"> 状态: `{status}`\n\n{content}\n"
        )
    else:
        # 抓取后立刻清洗 + Markdown 格式化
        text = rebuild_staging_markdown(meta=meta, raw_body=body)
    path.write_text(text, encoding="utf-8")
    return path


def _fetch_one_page(
    *,
    doc_id: str,
    url: str,
    query: str,
    query_id: str,
    note: str,
    tags: list[str],
    snippet: str,
    max_chars: int,
    staging_dir: Path | None,
) -> dict[str, Any]:
    try:
        code, ctype, raw = fetch_url(url)
        if code >= 400:
            path = write_staging_doc(
                doc_id=doc_id,
                url=url,
                title=doc_id,
                body="",
                query=query,
                query_id=query_id,
                note=note,
                tags=tags,
                status_code=code,
                error=f"HTTP {code}",
                snippet=snippet,
                staging_dir=staging_dir,
            )
            return {"id": doc_id, "ok": False, "path": str(path), "error": f"HTTP {code}", "url": url}
        title, text = ("", raw)
        if "html" in ctype or raw.lstrip().lower().startswith("<!DOCTYPE") or "<html" in raw[:500].lower():
            title, text = html_to_text(raw)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n…（截断）"
        path = write_staging_doc(
            doc_id=doc_id,
            url=url,
            title=title or doc_id,
            body=text,
            query=query,
            query_id=query_id,
            note=note,
            tags=tags,
            status_code=code,
            snippet=snippet,
            staging_dir=staging_dir,
        )
        return {
            "id": doc_id,
            "ok": True,
            "path": str(path),
            "chars": len(text),
            "title": title or doc_id,
            "url": url,
            "query": query,
        }
    except Exception as exc:  # noqa: BLE001
        path = write_staging_doc(
            doc_id=doc_id,
            url=url,
            title=doc_id,
            body="",
            query=query,
            query_id=query_id,
            note=note,
            tags=tags,
            error=str(exc),
            snippet=snippet,
            staging_dir=staging_dir,
        )
        return {"id": doc_id, "ok": False, "path": str(path), "error": str(exc), "url": url}


def crawl_by_queries(
    *,
    config_path: Path | None = None,
    staging_dir: Path | None = None,
    include_optional: bool = False,
    max_chars: int | None = None,
) -> list[dict[str, Any]]:
    """按搜索词检索并抓取候选页到 staging。"""
    from agent.tools.web_tools import run_web_search

    defaults, queries = load_config(config_path)
    default_top_k = int(defaults.get("top_k") or 5)
    default_max_chars = int(max_chars or defaults.get("max_chars") or 20_000)
    default_domains = [str(d) for d in (defaults.get("allow_domains") or [])]

    results: list[dict[str, Any]] = []
    for item in queries:
        if item.get("optional") and not include_optional:
            continue
        qid = str(item.get("id") or "q")
        note = str(item.get("note") or "")
        tags = [str(t) for t in (item.get("tags") or [])]
        top_k = int(item.get("top_k") or default_top_k)
        allow = [str(d) for d in (item.get("allow_domains") or default_domains)]
        page_max = int(item.get("max_chars") or default_max_chars)

        # 兼容：若仍写了 url，则直抓（不推荐）
        if item.get("url") and not item.get("query"):
            results.append(
                _fetch_one_page(
                    doc_id=qid,
                    url=str(item["url"]),
                    query="",
                    query_id=qid,
                    note=note,
                    tags=tags,
                    snippet="",
                    max_chars=page_max,
                    staging_dir=staging_dir,
                )
            )
            continue

        query = str(item.get("query") or "").strip()
        if not query:
            continue
        # 查询之间稍等，降低检索接口限流概率
        if results:
            time.sleep(1.5)
        try:
            hits, _meta = run_web_search(query, top_k)
        except Exception as exc:  # noqa: BLE001
            results.append({"id": qid, "ok": False, "query": query, "error": f"搜索失败: {exc}"})
            continue

        seen: set[str] = set()
        rank = 0
        for hit in hits:
            url = str(hit.get("url") or "").strip()
            if not url.startswith("http") or url in seen:
                continue
            if not _host_ok(url, allow):
                continue
            seen.add(url)
            rank += 1
            host = urlparse(url).netloc.replace(".", "_")
            doc_id = f"{qid}_{rank}_{host}"[:80]
            results.append(
                _fetch_one_page(
                    doc_id=doc_id,
                    url=url,
                    query=query,
                    query_id=qid,
                    note=note,
                    tags=tags,
                    snippet=str(hit.get("snippet") or ""),
                    max_chars=page_max,
                    staging_dir=staging_dir,
                )
            )
            if rank >= top_k:
                break
        if rank == 0:
            results.append(
                {
                    "id": qid,
                    "ok": False,
                    "query": query,
                    "error": "搜索无可用链接（或均被域名白名单过滤）",
                }
            )
    return results


# 兼容旧名
crawl_seeds = crawl_by_queries


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    meta = yaml.safe_load(parts[1]) or {}
    body = parts[2].lstrip("\n")
    return (meta if isinstance(meta, dict) else {}), body


def promote_approved(
    *,
    staging_dir: Path | None = None,
    docs_dir: Path | None = None,
    force_ids: list[str] | None = None,
) -> list[Path]:
    from agent.paths import KB_DOCS

    staging = staging_dir or KB_STAGING
    base = docs_dir or KB_DOCS
    dest = base / "crawl"
    dest.mkdir(parents=True, exist_ok=True)
    force = set(force_ids or [])
    written: list[Path] = []
    if not staging.exists():
        return written
    for path in sorted(staging.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        meta, body = parse_frontmatter(raw)
        sid = str(meta.get("id") or path.stem)
        status = str(meta.get("status") or "")
        if sid not in force and status != "approved":
            continue
        meta["status"] = "approved"
        meta["promoted_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        fm = yaml.safe_dump(meta, allow_unicode=True, sort_keys=False).strip()
        clean_body = re.sub(r"> 状态:.*\n?", "", body, count=1)
        out = dest / path.name
        out.write_text(f"---\n{fm}\n---\n\n{clean_body.lstrip()}", encoding="utf-8")
        written.append(out)
    return written
