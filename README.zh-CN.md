# NEV Research Agent

[English](README.md) | [中文](README.zh-CN.md)

新能源汽车行业研究助手：基于本地知识库（可选联网检索）回答问题，输出带引用的短答或分析。

**范围：** 整车、电池、供应链、政策、竞品。非新能源相关问题会拒答。

**角色：**

| 用户 | 说明 |
|------|------|
| `dev` | 普通用户：简要思考过程，默认走知识库问答 |
| `admin` | 管理员：完整思考与工具调用；可手动扩库爬取 |

用户定义见 `config/users.yaml`。

---

## 项目结构

```
nev-research-agent/
├── main.py                 # CLI 入口
├── config/users.yaml       # 测试用户（admin / alice / dev）+ 密码
├── requirements.txt
├── .env.example
├── src/
│   ├── agent/              # 研究 Agent
│   │   ├── cli.py          # 命令行交互
│   │   ├── runtime.py      # 运行入口
│   │   ├── display.py      # 角色分流展示
│   │   ├── s1/             # 单 Agent（ReAct）
│   │   ├── s2/             # 四节点：Supervisor / Research / Analyst / Verifier
│   │   ├── tools/          # kb_search / web_search
│   │   └── memory/         # 会话与长期记忆
│   └── kb/                 # 分块、入库、检索、爬取清洗
├── scripts/
│   ├── ingest_kb.py        # 语料 → 向量索引
│   ├── ask_kb.py           # 仅知识库问答（不经多 Agent）
│   ├── crawl_to_staging.py # 扩库爬取 → 待审
│   ├── promote_staging.py  # 审核通过 → 正式语料
│   └── clean_staging.py    # 清洗待审稿
├── data/                   # 本机数据（gitignore）
│   ├── kb/                 # 正式语料
│   ├── kb_staging/         # 爬取待审
│   ├── kb_index/           # JSON 向量索引
│   ├── milvus/             # Milvus Lite（可选）
│   └── crawl/search_queries.yaml  # 批量扩库搜索词
├── evals/                  # 评测数据集与脚本
└── tests/
```

Gradio Web UI 在分支 `front`（含 `web_app.py`、`src/agent/web/`）；本分支（`main`）仅保留 CLI。

---

## 安装

```bash
cd nev-research-agent
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
cp .env.example .env              # 填写 API Key 等
```

`.env` 至少需要：

- `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL`（兼容 DashScope 等）
- `EMBEDDING_MODEL`（入库与检索用）
- 可选：`TAVILY_API_KEY`、`KB_BACKEND=milvus`、`NEV_USER_ID`

未执行 `pip install -e .` 时，`main.py` 与 `scripts/` 会自动把 `src/` 加入路径。

### 知识库入库

```bash
# 默认：写入 data/kb_index/（本地 JSON 余弦）
python scripts/ingest_kb.py

# 可选：同时写入 Milvus Lite（无 Docker）
pip install "grpcio==1.67.1" "setuptools>=69,<81" "pymilvus==2.5.14" "milvus-lite==2.4.12"
python scripts/ingest_kb.py --also-milvus
# .env 中设 KB_BACKEND=milvus
```

---

## 怎么用

需要浏览器 UI 时请切换到 `front` 分支（`python web_app.py`）。

### 工具结果缓存（可选 Redis）

只缓存成功的 `kb_search` / `web_search` JSON；未配 `REDIS_URL` 或 Redis 宕机时失败旁路。同轮内相同工具调用按 `call_id` 去重。详见 [`docs/usage.md`](docs/usage.md)。

```bash
docker run -d --name nev-redis -p 6379:6379 redis:7-alpine
# .env: REDIS_URL=redis://127.0.0.1:6379/0
```

### 问答（CLI）

```bash
# 普通用户（默认）
python main.py --user alice
python main.py --user alice "比亚迪刀片电池的核心卖点是什么？"

# 管理员（可见工具调用细节）
python main.py --user admin "分析比亚迪海外扩张"

# 开启长期记忆
python main.py --memory --user alice "它和宁德时代比呢？"

# 仅知识库（不经多 Agent 编排）
python scripts/ask_kb.py "比亚迪刀片电池的核心卖点是什么？"
```

`AGENT_MODE=s1` 可切回单 Agent；默认 `s2`。

### RAG 查询改写与混合检索

`kb_search` 入库检索前会改写查询（去口语/任务动词、补实体别名）。默认规则改写：

```bash
# .env
KB_QUERY_REWRITE=rules   # 默认
# KB_QUERY_REWRITE=llm   # 可选：LLM 改写，失败回退 rules
# KB_QUERY_REWRITE=off

# 检索评测对照
python evals/run_retrieval.py --tag rewrite_off --rewrite off --method "无改写"
python evals/run_retrieval.py --tag rewrite_rules --rewrite rules --method "规则改写" --compare rewrite_off
```

混合检索：向量 + 内存 BM25。默认仍为纯向量（qrels_v3 上等权 RRF 会降 P/R，见 eval-log R18）：

```bash
# .env
KB_RETRIEVAL_MODE=vector   # 默认；keyword | hybrid
# KB_HYBRID_FUSION=rrf     # 或 convex
# KB_HYBRID_WEIGHTS=3,1    # 向量:关键词（rrf）
# KB_HYBRID_ALPHA=0.85     # 向量占比（convex）
# KB_HYBRID_CANDIDATES=24
# KB_RRF_K=60

python evals/run_retrieval.py --tag hybrid_rrf_w31_v3 --rewrite off --retrieval hybrid \
  --qrels evals/datasets/retrieval/qrels_v3.yaml --level chunk \
  --compare rag_chunk_baseline_v3 --method "hybrid: weighted RRF 3:1"
```

重排：向量初检 topN 后再截 k。默认 off：

```bash
# .env
KB_RERANK=off          # lexical | gte（DashScope gte-rerank-v2）
# KB_RERANK_CANDIDATES=32
# KB_RERANK_MODEL=gte-rerank-v2

python evals/run_retrieval.py --tag rerank_gte_v3 --rewrite off --retrieval vector \
  --rerank gte --qrels evals/datasets/retrieval/qrels_v3.yaml --level chunk \
  --compare rag_chunk_baseline_v3 --method "rerank: gte-rerank-v2"
```

### 管理员扩库

爬取结果进入 `data/kb_staging/`，需审核后再入库。

```bash
# 手动：关键词 + 时间范围（需管理员）
python scripts/crawl_to_staging.py --admin --user admin
python scripts/crawl_to_staging.py --user admin \
  --keyword "比亚迪 海外" --from 2026-01-01 --to 2026-06-30

# 批量：按 data/crawl/search_queries.yaml
python scripts/crawl_to_staging.py --from-config

# 审核通过 → data/kb/crawl/，再重建索引
python scripts/promote_staging.py --approve <doc_id>
python scripts/ingest_kb.py --also-milvus
```

### 评测与单测

```bash
pytest -q
python evals/run_retrieval.py --tag kb_expand
python evals/run_eval.py --suite golden --tag after_s2 --ids Q1,Q2
python evals/run_memory_eval.py --tag after_memory
```
