# RAG 方案对照协议

目标：在**冻结金标**上，逐个试检索改动，用 **P@k（准确率）/ R@k（召回率）** 选最优，而不是改数据集去迎合某次结果。

## 金标

| 文件 | 级别 | 用途 |
|------|------|------|
| `qrels.yaml` | doc | 旧 10 题（历史） |
| `qrels_v2.yaml` | doc | 文件级扩题（易虚高） |
| **`qrels_v3.yaml`** | **chunk** | **加固主用**：多品牌语料 + chunk id 金标 |

标注原则：按 **chunk 正文**能否回答该问写入 `relevant_chunks`（可用 `scripts/build_chunk_qrels.py` 按关键词+source 白名单生成候选）；**禁止**按某次检索 top-k 回标。

语料加固：seed 多品牌演示稿 + `data/crawl/diversify_queries.yaml` 爬取入库。

## 流程

1. 冻结 `qrels_v3.yaml`，先跑 **chunk 基线** `rag_chunk_baseline_v3`。
2. 每次只上一种方案（查询改写 / 混合检索 / rerank …），重跑同 qrels，换 `--tag`。
3. 对比 **P@k / R@k / nDCG@k / MRR**（chunk 级默认 `expand_neighbors=0`、`k=8`）。
4. 记入 `docs/eval-log.md`。

## 命令

```bash
# chunk 级基线（推荐）
KB_BACKEND=json python evals/run_retrieval.py \
  --tag rag_chunk_baseline_v3 \
  --qrels evals/datasets/retrieval/qrels_v3.yaml \
  --level chunk \
  --method "baseline: 纯向量；chunk 级"

# 重建 chunk 金标候选（入库变更后）
PYTHONPATH=src python scripts/build_chunk_qrels.py

# 某方案
KB_BACKEND=json python evals/run_retrieval.py \
  --tag rag_rewrite_v3 \
  --qrels evals/datasets/retrieval/qrels_v3.yaml \
  --level chunk \
  --compare rag_chunk_baseline_v3 \
  --method "query rewrite: 检索前 LLM 规范化"

# 混合检索（向量 + BM25 RRF）；对照时 rewrite=off 以免叠方案
KB_BACKEND=json python evals/run_retrieval.py \
  --tag hybrid_rrf_v3 \
  --qrels evals/datasets/retrieval/qrels_v3.yaml \
  --level chunk \
  --rewrite off \
  --retrieval hybrid \
  --compare rag_chunk_baseline_v3 \
  --method "hybrid: vector + BM25 RRF"

# 重排（初检扩召 → gte / lexical）；对照时其它开关关闭
KB_BACKEND=json python evals/run_retrieval.py \
  --tag rerank_gte_v3 \
  --qrels evals/datasets/retrieval/qrels_v3.yaml \
  --level chunk \
  --rewrite off \
  --retrieval vector \
  --rerank gte \
  --compare rag_chunk_baseline_v3 \
  --method "rerank: vector topN → gte-rerank-v2 → k"
```

## 候选方案（尚未实现则勿写进 method）

| 方案 | 位置 | 预期影响 | 状态 |
|------|------|----------|------|
| 查询改写（规则） | `src/kb/rewrite.py` → `kb_search` | 抗口语；书面金标上见 R17 | ✅ 已落地；qrels_v3 上 P/R 略降、MRR 持平 |
| 查询改写（LLM） | 同上 `KB_QUERY_REWRITE=llm` | 规范化 | ⬜ 未在 v3 金标上 live |
| 混合检索 | `src/kb/keyword.py` + `hybrid.py`；`KB_RETRIEVAL_MODE` | 专名、年份 | ✅ 已落地；等权 RRF 见 R18（P/R↓）；默认 `vector` + 加权 3:1 |
| rerank | `src/kb/rerank.py`；`KB_RERANK` | 偏准确率 | ✅ 已落地；gte/lexical 见 R19（P/R↓）；默认 off |
