"""混合检索：向量排序 + 关键词排序 → 加权 RRF / 凸组合归一化分。"""

from __future__ import annotations

import os
from typing import Literal, Sequence

from kb.chunking import Chunk

RetrievalMode = Literal["vector", "keyword", "hybrid"]
FusionMethod = Literal["rrf", "convex"]


def retrieval_mode_from_env(override: str | None = None) -> RetrievalMode:
    raw = (override if override is not None else os.getenv("KB_RETRIEVAL_MODE") or "vector")
    m = str(raw).strip().lower()
    if m in {"vector", "keyword", "hybrid"}:
        return m  # type: ignore[return-value]
    return "vector"


def fusion_method_from_env(override: str | None = None) -> FusionMethod:
    raw = (override if override is not None else os.getenv("KB_HYBRID_FUSION") or "rrf")
    m = str(raw).strip().lower()
    if m in {"rrf", "convex"}:
        return m  # type: ignore[return-value]
    return "rrf"


def hybrid_weights_from_env() -> tuple[float, float]:
    """
    向量 / 关键词权重。默认 3:1（等权在 qrels_v3 上会拖累 P/R）。
    KB_HYBRID_WEIGHTS=3,1
    """
    raw = (os.getenv("KB_HYBRID_WEIGHTS") or "3,1").strip()
    try:
        parts = [float(x.strip()) for x in raw.split(",")]
        if len(parts) >= 2 and parts[0] > 0 and parts[1] >= 0:
            return parts[0], parts[1]
    except ValueError:
        pass
    return 3.0, 1.0


def hybrid_alpha_from_env() -> float:
    """convex 融合中向量占比；默认 0.85。KB_HYBRID_ALPHA=0.85"""
    raw = (os.getenv("KB_HYBRID_ALPHA") or "0.85").strip()
    try:
        a = float(raw)
        if 0.0 <= a <= 1.0:
            return a
    except ValueError:
        pass
    return 0.85


def rrf_fuse(
    rankings: Sequence[Sequence[tuple[Chunk, float]]],
    *,
    rrf_k: int = 60,
    top_k: int = 8,
    weights: Sequence[float] | None = None,
) -> list[tuple[Chunk, float]]:
    """
    加权 RRF：score(d) = Σ w_i / (rrf_k + rank_i)。
    默认各路权重 1；混合检索建议向量权重 > 关键词。
    """
    k = max(1, int(rrf_k))
    ws = list(weights) if weights is not None else [1.0] * len(rankings)
    while len(ws) < len(rankings):
        ws.append(1.0)
    fused: dict[str, float] = {}
    best_chunk: dict[str, Chunk] = {}
    for ranking, w in zip(rankings, ws):
        ww = float(w)
        if ww <= 0:
            continue
        for rank, (chunk, _raw) in enumerate(ranking, start=1):
            cid = chunk.id
            if not cid:
                continue
            fused[cid] = fused.get(cid, 0.0) + ww / (k + rank)
            best_chunk.setdefault(cid, chunk)
    ordered = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    out: list[tuple[Chunk, float]] = []
    for cid, sc in ordered[: max(0, top_k)]:
        out.append((best_chunk[cid], float(sc)))
    return out


def _minmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    lo = min(scores.values())
    hi = max(scores.values())
    if hi <= lo:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def convex_fuse(
    vector_pairs: Sequence[tuple[Chunk, float]],
    keyword_pairs: Sequence[tuple[Chunk, float]],
    *,
    alpha: float = 0.85,
    top_k: int = 8,
) -> list[tuple[Chunk, float]]:
    """
    凸组合：score = α·norm(vec) + (1-α)·norm(bm25)；缺一路则该路为 0。
    α 靠近 1 时更信向量（本语料 BM25 弱于向量时推荐）。
    """
    a = min(1.0, max(0.0, float(alpha)))
    vec_raw = {c.id: float(s) for c, s in vector_pairs if c.id}
    kw_raw = {c.id: float(s) for c, s in keyword_pairs if c.id}
    best: dict[str, Chunk] = {}
    for c, _ in list(vector_pairs) + list(keyword_pairs):
        if c.id:
            best.setdefault(c.id, c)
    vn = _minmax(vec_raw)
    kn = _minmax(kw_raw)
    fused: dict[str, float] = {}
    for cid in set(vn) | set(kn):
        fused[cid] = a * vn.get(cid, 0.0) + (1.0 - a) * kn.get(cid, 0.0)
    ordered = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    return [(best[cid], sc) for cid, sc in ordered[: max(0, top_k)] if cid in best]


def candidate_pool(top_k: int, override: int | None = None) -> int:
    """每路初检条数：默认 max(top_k*3, 24)，可用 KB_HYBRID_CANDIDATES 覆盖。"""
    if override is not None and override > 0:
        return int(override)
    env = os.getenv("KB_HYBRID_CANDIDATES", "").strip()
    if env.isdigit() and int(env) > 0:
        return int(env)
    return max(int(top_k) * 3, 24)
