"""关键词检索：轻量 BM25（无 jieba），适合专名 / 年份 / 型号。

中文用字 + 二元组；英文 / 数字保留整词。索引文本含 title + source + body。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

from kb.chunking import Chunk

# 延迟导入 Hit，避免循环：kb.__init__ → keyword → Hit
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]*|[\u4e00-\u9fff]+")
_CJK_RE = re.compile(r"^[\u4e00-\u9fff]+$")


def tokenize(text: str) -> list[str]:
    """查询 / 文档共用分词。"""
    text = (text or "").strip().lower()
    if not text:
        return []
    out: list[str] = []
    for m in _TOKEN_RE.finditer(text):
        tok = m.group(0)
        if _CJK_RE.fullmatch(tok):
            chars = list(tok)
            out.extend(chars)
            if len(chars) >= 2:
                out.extend(tok[i : i + 2] for i in range(len(chars) - 1))
        else:
            out.append(tok)
    return out


def _doc_blob(c: Chunk) -> str:
    return f"{c.title or ''}\n{c.source or ''}\n{c.text or ''}"


class KeywordIndex:
    """内存 BM25 倒排；在 KnowledgeBase 加载 chunks 后构建一次。"""

    def __init__(
        self,
        chunks: Iterable[Chunk],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        self.chunks: list[Chunk] = list(chunks)
        self._tf: list[Counter[str]] = []
        self._dl: list[int] = []
        self._df: Counter[str] = Counter()
        self._avgdl = 1.0
        self._n = 0
        self._build()

    def _build(self) -> None:
        self._tf = []
        self._dl = []
        self._df = Counter()
        for c in self.chunks:
            toks = tokenize(_doc_blob(c))
            tf = Counter(toks)
            self._tf.append(tf)
            self._dl.append(max(1, sum(tf.values())))
            for term in tf:
                self._df[term] += 1
        self._n = len(self.chunks)
        self._avgdl = (sum(self._dl) / self._n) if self._n else 1.0

    def _idf(self, term: str) -> float:
        n = self._n
        df = self._df.get(term, 0)
        # BM25+ 风格平滑，避免零 IDF
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def score_all(self, query: str) -> list[tuple[int, float]]:
        q_terms = tokenize(query)
        if not q_terms or not self.chunks:
            return []
        q_tf = Counter(q_terms)
        scores: list[tuple[int, float]] = []
        for i, tf in enumerate(self._tf):
            s = 0.0
            dl = self._dl[i]
            for term, qf in q_tf.items():
                f = tf.get(term, 0)
                if f <= 0:
                    continue
                idf = self._idf(term)
                denom = f + self.k1 * (1.0 - self.b + self.b * dl / self._avgdl)
                s += idf * (f * (self.k1 + 1.0) / denom) * qf
            if s > 0:
                scores.append((i, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def search(self, query: str, top_k: int = 8) -> list[tuple[Chunk, float]]:
        ranked = self.score_all(query)
        out: list[tuple[Chunk, float]] = []
        for idx, sc in ranked[: max(0, top_k)]:
            out.append((self.chunks[idx], float(sc)))
        return out
