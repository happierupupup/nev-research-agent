"""RAG 查询改写：规则优先，可选 LLM。

环境变量 KB_QUERY_REWRITE：
  rules（默认）| llm | off

用于把口语/任务型问句收成适合向量检索的短查询，提升召回。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Literal

from agent.core.intent import sanitize_search_query

RewriteMode = Literal["rules", "llm", "off"]

# 实体别名：命中任一侧时，把规范名补进查询（不删原词）
_ALIAS_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("比亚迪", ("BYD", "byd", "比亚迪汽车")),
    ("宁德时代", ("宁德", "CATL", "catl", "当代安培")),
    ("特斯拉", ("Tesla", "tesla", "TSLA")),
    ("刀片电池", ("刀片", "Blade Battery", "blade battery")),
    ("磷酸铁锂", ("LFP", "lfp")),
    ("三元锂", ("NMC", "nmc", "NCM")),
    ("理想汽车", ("理想", "Li Auto", "理想汽车")),
    ("蔚来", ("NIO", "nio")),
    ("小鹏", ("XPeng", "XPENG", "小鹏汽车")),
    ("乘联会", ("CPCA", "全国乘用车市场信息联席会")),
    ("新能源乘用车", ("新能源车", "NEV", "nev", "电动车", "电车")),
    ("渗透率", ("市占率", "份额占比")),
)

# 口语/问答尾巴
# 注意：不要把「风险」等主题词单独当尾巴删掉（须带「是什么」等问句结构）
_QA_TAIL = re.compile(
    r"(?:"
    r"的?(?:核心)?(?:卖点|优势|特点|原因|情况|数据|口径)(?:是什么|怎么样|如何|怎样)?"
    r"|是什么|怎么样|如何|怎样|有哪些|(?:大概|大约)?多少|(?:大概|大约)?多大|对吗|吗|呢|呀|啊"
    r")?[？?！!。.\s]*$"
)

_CHAT_PREFIXES = (
    "请问一下",
    "请问",
    "想问一下",
    "我想问",
    "帮我查一下",
    "帮我查",
    "帮我看看",
    "你知道",
    "能不能告诉我",
    "告诉我",
)

_WS = re.compile(r"\s+")


@dataclass
class RewriteResult:
    original: str
    rewritten: str
    variants: list[str] = field(default_factory=list)
    method: str = "rules"
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "original": self.original,
            "rewritten": self.rewritten,
            "variants": list(self.variants),
            "method": self.method,
            "notes": list(self.notes),
        }


def rewrite_mode_from_env(raw: str | None = None) -> RewriteMode:
    t = (raw if raw is not None else os.getenv("KB_QUERY_REWRITE") or "rules").strip().lower()
    if t in {"off", "0", "false", "no", "none"}:
        return "off"
    if t in {"llm", "model", "ai"}:
        return "llm"
    return "rules"


def _strip_chat_prefix(text: str) -> str:
    t = text.strip()
    for p in sorted(_CHAT_PREFIXES, key=len, reverse=True):
        if t.startswith(p):
            t = t[len(p) :].lstrip(" \t：:，,")
            break
    return t


def _strip_qa_tail(text: str) -> str:
    t = text.strip()
    t2 = _QA_TAIL.sub("", t).strip(" \t：:，,、")
    return t2 or t


def _expand_aliases(text: str) -> tuple[str, list[str]]:
    """补规范实体名；返回 (text, notes)。"""
    notes: list[str] = []
    low = text.lower()
    extras: list[str] = []
    for canon, alts in _ALIAS_GROUPS:
        hit = canon in text or any(
            (a.lower() in low) if a.isascii() else (a in text) for a in alts
        )
        if not hit:
            continue
        if canon not in text:
            extras.append(canon)
            notes.append(f"alias→{canon}")
    if not extras:
        return text, notes
    # 规范名放前面，利于 embedding 对齐语料标题
    merged = " ".join(extras + [text])
    return _WS.sub(" ", merged).strip(), notes


def rewrite_rules(query: str) -> RewriteResult:
    original = (query or "").strip()
    if not original:
        return RewriteResult(original="", rewritten="", method="rules")

    notes: list[str] = []
    # 先去口语前缀，再去任务动词，再去问答尾巴（顺序影响「请问分析一下…」）
    t = _strip_chat_prefix(original)
    if t != original:
        notes.append("strip_chat_prefix")

    t2 = sanitize_search_query(t)
    if t2 != t:
        notes.append("sanitize_task_verb")
        t = t2

    t3 = _strip_qa_tail(t)
    if t3 != t:
        notes.append("strip_qa_tail")
        t = t3

    t, alias_notes = _expand_aliases(t)
    notes.extend(alias_notes)

    t = _WS.sub(" ", t).strip(" \t：:，,、？?！!")
    if not t:
        t = sanitize_search_query(original) or original

    variants: list[str] = []
    # 去掉别名扩充后的「纯实体短查」作可选变体（多路检索预留）
    bare = sanitize_search_query(_strip_qa_tail(_strip_chat_prefix(original)))
    bare = _WS.sub(" ", bare).strip(" \t：:，,、？?！!")
    if bare and bare != t and bare not in variants:
        variants.append(bare)

    return RewriteResult(
        original=original,
        rewritten=t,
        variants=variants,
        method="rules",
        notes=notes,
    )


_LLM_SYSTEM = (
    "你是新能源汽车行业知识库的检索查询改写器。"
    "把用户问题改写成适合向量检索的短查询："
    "保留企业/产品/年份/指标等实体；去掉「分析/对比/请问/是什么」等任务词与口语；"
    "输出一行中文关键词，不超过 40 字；不要解释、不要标点收尾。"
)


def rewrite_llm(query: str) -> RewriteResult:
    """LLM 改写；失败则回退 rules。"""
    base = rewrite_rules(query)
    original = base.original
    if not original:
        return base
    try:
        from agent.llm import get_llm
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = get_llm(temperature=0)
        msg = llm.invoke(
            [
                SystemMessage(content=_LLM_SYSTEM),
                HumanMessage(content=original),
            ]
        )
        text = (getattr(msg, "content", None) or "").strip()
        if isinstance(text, list):
            text = "".join(str(x) for x in text)
        text = _WS.sub(" ", str(text)).strip(" \t\"'`：:，,。")
        if not text or len(text) > 80:
            base.notes.append("llm_empty_or_too_long→rules")
            return base
        # 再走一遍规则清洗，避免模型塞回任务动词
        cleaned = rewrite_rules(text)
        return RewriteResult(
            original=original,
            rewritten=cleaned.rewritten or base.rewritten,
            variants=list(dict.fromkeys([base.rewritten, *cleaned.variants, *base.variants])),
            method="llm",
            notes=base.notes + ["llm"] + cleaned.notes,
        )
    except Exception as exc:  # noqa: BLE001
        base.notes.append(f"llm_error→rules:{type(exc).__name__}")
        return base


def rewrite_query(query: str, *, mode: RewriteMode | str | None = None) -> RewriteResult:
    m = rewrite_mode_from_env(mode if isinstance(mode, str) or mode is None else None)
    if mode in {"rules", "llm", "off"}:
        m = mode  # type: ignore[assignment]
    original = (query or "").strip()
    if m == "off":
        return RewriteResult(original=original, rewritten=original, method="off")
    if m == "llm":
        return rewrite_llm(original)
    return rewrite_rules(original)
