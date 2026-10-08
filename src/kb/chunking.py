"""分块 v2：按标题切分 + 长度上限 + 重叠 + 邻块链接 + 来源元数据。"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any  # DocMeta.extra

import yaml

# 默认：约 600 字一块，重叠约 120 字（前后块共享一段上下文）
DEFAULT_MAX_CHARS = 600
DEFAULT_OVERLAP_CHARS = 120


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    title: str
    # v2 元数据
    doc_id: str = ""
    chunk_index: int = 0
    prev_id: str | None = None
    next_id: str | None = None
    url: str = ""
    crawled_at: str = ""
    overlap_chars: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DocMeta:
    source: str
    url: str = ""
    crawled_at: str = ""
    title_hint: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def parse_doc_file(path: Path, *, base: Path) -> tuple[DocMeta, str]:
    """读文件：抽出 frontmatter / 引用头里的 url·抓取时间，正文去掉元信息行。"""
    raw = path.read_text(encoding="utf-8")
    rel = str(path.relative_to(base))
    meta: dict[str, Any] = {}
    body = raw
    if raw.startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) >= 3:
            loaded = yaml.safe_load(parts[1]) or {}
            if isinstance(loaded, dict):
                meta = loaded
            body = parts[2].lstrip("\n")

    url = str(meta.get("url") or "")
    crawled_at = str(meta.get("crawled_at") or "")
    title_hint = str(meta.get("title") or "")

    cleaned_lines: list[str] = []
    for ln in body.splitlines():
        s = ln.strip()
        if s.startswith("> 搜索词:") or s.startswith("> 状态:"):
            continue
        if s.startswith("> 来源:"):
            if not url:
                url = s.split(":", 1)[-1].strip()
            continue
        if s.startswith("> 抓取:"):
            if not crawled_at:
                crawled_at = s.split(":", 1)[-1].strip()
            continue
        cleaned_lines.append(ln)
    body = "\n".join(cleaned_lines).strip()
    return (
        DocMeta(
            source=rel,
            url=url,
            crawled_at=crawled_at,
            title_hint=title_hint,
            extra={k: v for k, v in meta.items() if k not in {"url", "crawled_at", "title"}},
        ),
        body,
    )


def _section_blocks(text: str, default_title: str) -> list[tuple[str, str]]:
    """按 # / ## 切成 (标题, 正文块)。"""
    text = text.strip()
    if not text:
        return []
    # 统一：先按 ## 切；首段可用 # 标题
    parts = re.split(r"(?m)^##\s+", text)
    blocks: list[tuple[str, str]] = []
    title = default_title
    first = parts[0].strip()
    if first:
        h1 = re.search(r"(?m)^#\s+(.+)$", first)
        if h1:
            title = h1.group(1).strip()
            # 去掉首行 # 标题，避免重复
            rest = re.sub(r"(?m)^#\s+.+\n?", "", first, count=1).strip()
            blocks.append((title, rest if rest else first))
        else:
            blocks.append((title, first))
    for part in parts[1:]:
        lines = part.splitlines()
        heading = lines[0].strip() if lines else title
        body = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""
        content = f"## {heading}\n{body}".strip() if body else f"## {heading}"
        blocks.append((heading, content))
    return blocks


def _split_long_string(s: str, max_chars: int, overlap: int) -> list[str]:
    """无空行的超长串：按字符滑窗切，带重叠。"""
    s = s.strip()
    if not s:
        return []
    if len(s) <= max_chars:
        return [s]
    overlap = max(0, min(overlap, max_chars // 2))
    out: list[str] = []
    start = 0
    n = len(s)
    while start < n:
        end = min(start + max_chars, n)
        out.append(s[start:end])
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return out


def _window_paragraphs(paras: list[str], max_chars: int, overlap: int) -> list[str]:
    """段落装箱 + 重叠：下一块以上一块尾部 overlap 字起头。"""
    if not paras:
        return []
    # 先把超长段切开
    flat: list[str] = []
    for p in paras:
        if len(p) <= max_chars:
            flat.append(p)
        else:
            flat.extend(_split_long_string(p, max_chars, overlap))

    windows: list[str] = []
    buf: list[str] = []
    buf_len = 0

    def flush() -> None:
        nonlocal buf, buf_len
        if not buf:
            return
        windows.append("\n\n".join(buf).strip())
        if overlap <= 0 or not windows[-1]:
            buf, buf_len = [], 0
            return
        # 取上一窗尾部 overlap 字符作为下一块前缀（尽量落在段落边界）
        prev = windows[-1]
        tail = prev[-overlap:]
        # 若尾部含完整段，从最后一个空行后起
        if "\n\n" in tail:
            tail = tail.split("\n\n")[-1]
        buf = [tail] if tail else []
        buf_len = len(tail)

    for p in flat:
        add = len(p) + (2 if buf else 0)
        if buf and buf_len + add > max_chars:
            flush()
            # flush 后 buf 已是 overlap 前缀；若单段仍超长已在 flat 处理
            add = len(p) + (2 if buf else 0)
            if buf and buf_len + add > max_chars:
                # 前缀 + 本段仍爆：先丢掉前缀，单独成块
                buf, buf_len = [], 0
        buf.append(p)
        buf_len += add
    if buf:
        # 最后一块：若与上一块完全相同则跳过（纯 overlap 残留）
        last = "\n\n".join(buf).strip()
        if last and (not windows or last != windows[-1]):
            windows.append(last)
    return windows


def link_adjacent(chunks: list[Chunk]) -> list[Chunk]:
    """同一文档内按 chunk_index 串起 prev_id / next_id。"""
    by_doc: dict[str, list[int]] = {}
    for i, c in enumerate(chunks):
        by_doc.setdefault(c.doc_id or c.source, []).append(i)
    for idxs in by_doc.values():
        idxs.sort(key=lambda i: chunks[i].chunk_index)
        for j, i in enumerate(idxs):
            chunks[i].prev_id = chunks[idxs[j - 1]].id if j > 0 else None
            chunks[i].next_id = chunks[idxs[j + 1]].id if j + 1 < len(idxs) else None
    return chunks


def split_markdown_v2(
    text: str,
    *,
    source: str,
    url: str = "",
    crawled_at: str = "",
    title_hint: str = "",
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> list[Chunk]:
    """分块 v2：标题 → 长度上限 → 重叠 → 邻块待 link_adjacent 填充。"""
    default_title = title_hint or Path(source).stem
    sections = _section_blocks(text, default_title)
    raw_chunks: list[Chunk] = []
    n = 0
    for heading, block in sections:
        block = block.strip()
        if not block:
            continue
        if len(block) <= max_chars:
            pieces = [block]
        else:
            paras = [p.strip() for p in re.split(r"\n\s*\n", block) if p.strip()]
            pieces = _window_paragraphs(paras, max_chars, overlap_chars)
        for piece in pieces:
            cid = f"{source}::{n}"
            raw_chunks.append(
                Chunk(
                    id=cid,
                    text=piece,
                    source=source,
                    title=heading,
                    doc_id=source,
                    chunk_index=n,
                    url=url,
                    crawled_at=crawled_at,
                    overlap_chars=overlap_chars if n > 0 else 0,
                )
            )
            n += 1
    return link_adjacent(raw_chunks)


# 兼容旧测试名
def _split_markdown(text: str, source: str, max_chars: int = DEFAULT_MAX_CHARS) -> list[Chunk]:
    return split_markdown_v2(text, source=source, max_chars=max_chars)
