"""爬取正文清洗：去掉导航/广告噪音，保留关键段落，整理为 Markdown。"""

from __future__ import annotations

import re
from typing import Iterable

# 整行匹配即丢弃（站点导航、按钮、栏目名）
_DROP_LINE = re.compile(
    r"^("
    r"首页|打开|登录|注册|分享|打印|收藏|评论|下载|客户端|手机版|电脑版|"
    r"上一页|下一页|返回|更多|展开|收起|加载中|搜索|订阅|"
    r"党政|党网|时政|人事|反腐|理论|党史|党建|要闻|"
    r"经济|科技|社会|法治|文旅|体育|健康|生活|国际|军事|"
    r"大湾区|台湾|教育|科普|观点|社论|互动|领导留言板|强国论坛|"
    r"人民锐评|人民建议|人民投诉|人民防非|"
    r"本报评论员|任仲平|任平|仲音|人民论坛|人民时评|今日谈|望海楼|"
    r"金社平|寰宇平|钟声|和音|金台随笔|每周经济评论|"
    r"政务公开|政策解读|首都之窗|附件|相关链接|责任编辑|责编|"
    r"版权所有|违法和不良信息|京ICP|京公网安备|"
    r"Image\s+\d+|交易量"
    r")$",
    re.I,
)

_DROP_CONTAINS = (
    "版权所有",
    "京ICP",
    "京公网安备",
    "违法和不良信息",
    "请开启JavaScript",
    "cookie",
    "All rights reserved",
)

# 栏目面包屑一类
_CRUMB = re.compile(r"^[\w\u4e00-\u9fff·\s]{1,20}(>|／|/|>){1,}.{0,40}$")
_SECTION_CN = re.compile(r"^([一二三四五六七八九十百千]+、.+)$")
_SUBSECTION_CN = re.compile(r"^[（(][一二三四五六七八九十]+[）)].+")
_SECTION_NUM = re.compile(r"^(\d+[\.、．]\s*\S.+)$")


def _is_noise_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    if _DROP_LINE.match(s):
        return True
    low = s.lower()
    if any(x.lower() in low for x in _DROP_CONTAINS):
        return True
    # 过短且无数字/实质标点 → 多半是菜单
    if len(s) <= 8 and not re.search(r"[\d%万亿美元€$]|[，。；：、]", s):
        if not _SECTION_CN.match(s) and not _SUBSECTION_CN.match(s):
            return True
    if _CRUMB.match(s) and len(s) < 40:
        return True
    # 纯导航/友情链接：多个短词用空格或·分隔，且几乎无句号
    parts = [p for p in re.split(r"[\s·|｜]+", s) if p]
    if len(parts) >= 4 and all(1 <= len(p) <= 10 for p in parts) and "。" not in s:
        return True
    # 页脚机构名 / 举报网 / 客户端入口
    if any(
        k in s
        for k in (
            "举报网",
            "工作办",
            "体验中心",
            "慕课",
            "写易智能",
            "人民云",
            "来源：",
            "摄",
            "记者",
        )
    ) and len(s) < 40 and "。" not in s:
        # 「新华社记者」配图说明可丢；长句正文里的「记者」不在此（有句号或更长）
        if "。" not in s:
            return True
    # 「2026年07月01日09:20 | 来源：」类元信息
    if re.match(r"^\d{4}年\d{1,2}月\d{1,2}日", s) and ("来源" in s or "|" in s):
        return True
    # 页脚机构 / 本报电头
    if re.search(r"(研究院|委员会|工作办|发布平台|教育课堂|本报.+电)$", s) and "。" not in s:
        return True
    if s.startswith("（本报") and s.endswith("）"):
        return True
    return False


def _looks_header(line: str) -> bool:
    s = line.strip()
    return bool(
        _SECTION_CN.match(s)
        or _SUBSECTION_CN.match(s)
        or _SECTION_NUM.match(s)
        or s.startswith("#")
    )


def _format_line(line: str) -> str:
    s = line.strip()
    if _SECTION_CN.match(s):
        return f"## {s}"
    if _SUBSECTION_CN.match(s):
        return f"### {s}"
    if _SECTION_NUM.match(s) and len(s) < 80:
        return f"## {s}"
    return s


def _dedupe_adjacent(lines: Iterable[str]) -> list[str]:
    out: list[str] = []
    prev = None
    for line in lines:
        if line == prev:
            continue
        out.append(line)
        prev = line
    return out


def _paragraph_score(text: str, *, query: str, snippet: str) -> float:
    """越高越像正文。"""
    if not text:
        return -1.0
    n = len(text)
    score = min(n / 80.0, 8.0)
    if re.search(r"[。！？；]", text):
        score += 2.0
    if re.search(r"\d", text):
        score += 1.0
    # 与搜索词/摘要重叠
    for token in re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9]{3,}", query or ""):
        if token in text:
            score += 0.6
    if snippet:
        head = snippet[:40]
        if head and head in text:
            score += 4.0
    # 惩罚导航残留
    if _is_noise_line(text):
        score -= 5.0
    return score


def clean_page_text(
    raw: str,
    *,
    title: str = "",
    query: str = "",
    snippet: str = "",
    min_keep_chars: int = 200,
) -> str:
    """清洗抓取正文，输出 Markdown 正文（不含 frontmatter）。"""
    text = (raw or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 先按空行切段，再按行过滤
    rough_paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    kept_paras: list[str] = []
    for para in rough_paras:
        lines = [_format_line(x) for x in para.splitlines() if not _is_noise_line(x)]
        lines = _dedupe_adjacent(lines)
        if not lines:
            continue
        block = "\n".join(lines).strip()
        if not block:
            continue
        # 整段仍是噪音
        if len(block) < 20 and not _looks_header(block):
            continue
        kept_paras.append(block)

    # 若仍太吵，按「高分段落」精选
    scored = sorted(
        ((_paragraph_score(p, query=query, snippet=snippet), p) for p in kept_paras),
        key=lambda x: -x[0],
    )
    selected: list[str] = []
    total = 0
    # 先尽量保留标题后的高分段
    for sc, p in scored:
        if sc < 1.5 and total > min_keep_chars:
            continue
        selected.append(p)
        total += len(p)
        if total >= 12_000:
            break

    # 按原文顺序重排
    order = {p: i for i, p in enumerate(kept_paras)}
    selected.sort(key=lambda p: order.get(p, 10**9))

    body = "\n\n".join(selected).strip()
    if not body and snippet:
        body = snippet.strip()
    if not body:
        body = "_（清洗后无有效正文）_"

    # 标题行：去掉站点后缀
    clean_title = (title or "").split("--")[0].split("_")[0].strip()
    clean_title = re.sub(r"^【|】$", "", clean_title).strip("[]【】 ")
    if clean_title and not body.lstrip().startswith("#"):
        body = f"# {clean_title}\n\n{body}"

    # 去掉正文开头重复标题
    lines = body.splitlines()
    if len(lines) >= 2 and lines[0].startswith("# "):
        t = lines[0][2:].strip()
        rest: list[str] = []
        skip_done = False
        for ln in lines[1:]:
            if not skip_done and ln.strip() == t:
                skip_done = True
                continue
            rest.append(ln)
        body = lines[0] + "\n" + "\n".join(rest).lstrip("\n")

    body = re.sub(r"\n{3,}", "\n\n", body).strip() + "\n"
    return body


def rebuild_staging_markdown(
    *,
    meta: dict,
    raw_body: str,
) -> str:
    """用清洗后的正文重写 staging Markdown（保留 frontmatter）。"""
    import yaml

    title = str(meta.get("title") or meta.get("id") or "文档")
    query = str(meta.get("query") or "")
    snippet = str(meta.get("snippet") or "")
    cleaned = clean_page_text(raw_body, title=title, query=query, snippet=snippet)
    meta = dict(meta)
    meta["cleaned"] = True
    # 摘要/备注/错误一律单行，避免 YAML 折行损坏
    for key in ("snippet", "note", "error", "title"):
        if key in meta and meta[key] is not None:
            meta[key] = re.sub(r"\s+", " ", str(meta[key])).strip()[:500]
    fm = yaml.safe_dump(meta, allow_unicode=True, sort_keys=False).strip()
    url = meta.get("url") or ""
    crawled = meta.get("crawled_at") or ""
    status = meta.get("status") or "pending_review"
    qline = f"> 搜索词: {query}  \n" if query else ""
    header = (
        f"{qline}> 来源: {url}  \n> 抓取: {crawled}  \n"
        f"> 状态: `{status}`（审核后改为 `approved` 再入库）\n"
    )
    # cleaned 正文已含 # 标题
    return f"---\n{fm}\n---\n\n{header}\n{cleaned}"
