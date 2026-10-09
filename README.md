# NEV Research Agent

[English](README.md) | [中文](README.zh-CN.md)

NEV (new-energy vehicle) research assistant: answers from a local knowledge base (optional web search) with cited short answers or analysis.

**Scope:** OEMs, batteries, supply chain, policy, competitors. Non-NEV questions are refused.

**Roles:**

| User | Description |
|------|-------------|
| `dev` | End user: brief thinking trace; KB Q&A by default |
| `admin` | Admin: full tool traces; can run crawl/expand |

Users are defined in `config/users.yaml`.

---

## Layout

```
nev-research-agent/
├── main.py                 # CLI entry
├── config/users.yaml       # Test users (admin / alice / dev) + passwords
├── requirements.txt
├── .env.example
├── src/
│   ├── agent/              # Research agent
│   │   ├── cli.py          # Interactive CLI
│   │   ├── runtime.py      # Runtime entry
│   │   ├── display.py      # Role-based display
│   │   ├── s1/             # Single ReAct agent
│   │   ├── s2/             # Supervisor / Research / Analyst / Verifier
│   │   ├── tools/          # kb_search / web_search
│   │   └── memory/         # Session + long-term memory
│   └── kb/                 # Chunking, ingest, retrieval, crawl/clean
├── scripts/
│   ├── ingest_kb.py        # Corpus → vector index
│   ├── ask_kb.py           # KB-only Q&A (no multi-agent)
│   ├── crawl_to_staging.py # Crawl → staging
│   ├── promote_staging.py  # Approve staging → corpus
│   └── clean_staging.py    # Clean staging drafts
├── data/                   # Local data (gitignored)
│   ├── kb/                 # Production corpus
│   ├── kb_staging/         # Crawl pending review
│   ├── kb_index/           # JSON vector index
│   ├── milvus/             # Milvus Lite (optional)
│   └── crawl/search_queries.yaml  # Batch crawl queries
├── evals/                  # Eval datasets + runners
└── tests/
```

The Gradio Web UI lives on branch `front` (`web_app.py`, `src/agent/web/`). This branch (`main`) is CLI-only.

---

## Install

```bash
cd nev-research-agent
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
cp .env.example .env              # fill in API keys, etc.
```

Minimum `.env`:

- `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL` (OpenAI-compatible, e.g. DashScope)
- `EMBEDDING_MODEL` (ingest + retrieval)
- Optional: `TAVILY_API_KEY`, `KB_BACKEND=milvus`, `NEV_USER_ID`

If you skip `pip install -e .`, `main.py` and `scripts/` add `src/` to `sys.path` automatically.

### Ingest the knowledge base

```bash
# Default: write data/kb_index/ (local JSON cosine)
python scripts/ingest_kb.py

# Optional: also write Milvus Lite (no Docker)
pip install "grpcio==1.67.1" "setuptools>=69,<81" "pymilvus==2.5.14" "milvus-lite==2.4.12"
python scripts/ingest_kb.py --also-milvus
# set KB_BACKEND=milvus in .env
```

---

## Usage

For a browser UI, switch to the `front` branch (`python web_app.py`).

### Tool-result cache (optional Redis)

Caches successful `kb_search` / `web_search` JSON only. Fail-open if `REDIS_URL` is unset or Redis is down. Same-turn duplicate tool calls are deduped by `call_id`. See [`docs/usage.md`](docs/usage.md).

```bash
docker run -d --name nev-redis -p 6379:6379 redis:7-alpine
# .env: REDIS_URL=redis://127.0.0.1:6379/0
```

### Q&A (CLI)

```bash
# End user (default)
python main.py --user alice
python main.py --user alice "比亚迪刀片电池的核心卖点是什么？"

# Admin (full tool-call detail)
python main.py --user admin "分析比亚迪海外扩张"

# Long-term memory
python main.py --memory --user alice "它和宁德时代比呢？"

# KB only (no multi-agent orchestration)
python scripts/ask_kb.py "What are the key selling points of BYD Blade Battery?"
```

Set `AGENT_MODE=s1` for the single agent; default is `s2`.

### RAG: query rewrite & hybrid retrieval

`kb_search` rewrites the query before retrieval (strip chatty verbs, expand entity aliases). Default is rule-based:

```bash
# .env
KB_QUERY_REWRITE=rules   # default
# KB_QUERY_REWRITE=llm   # optional LLM rewrite; falls back to rules
# KB_QUERY_REWRITE=off

# Retrieval eval A/B
python evals/run_retrieval.py --tag rewrite_off --rewrite off --method "no rewrite"
python evals/run_retrieval.py --tag rewrite_rules --rewrite rules --method "rule rewrite" --compare rewrite_off
```

Hybrid retrieval: dense + in-memory BM25. Default remains dense-only (equal-weight RRF hurt P/R on qrels_v3; see eval-log R18):

```bash
# .env
KB_RETRIEVAL_MODE=vector   # default; keyword | hybrid
# KB_HYBRID_FUSION=rrf     # or convex
# KB_HYBRID_WEIGHTS=3,1    # vector:keyword (rrf)
# KB_HYBRID_ALPHA=0.85     # vector share (convex)
# KB_HYBRID_CANDIDATES=24
# KB_RRF_K=60

python evals/run_retrieval.py --tag hybrid_rrf_w31_v3 --rewrite off --retrieval hybrid \
  --qrels evals/datasets/retrieval/qrels_v3.yaml --level chunk \
  --compare rag_chunk_baseline_v3 --method "hybrid: weighted RRF 3:1"
```

Rerank: retrieve top-N then cut to k. Default off:

```bash
# .env
KB_RERANK=off          # lexical | gte (DashScope gte-rerank-v2)
# KB_RERANK_CANDIDATES=32
# KB_RERANK_MODEL=gte-rerank-v2

python evals/run_retrieval.py --tag rerank_gte_v3 --rewrite off --retrieval vector \
  --rerank gte --qrels evals/datasets/retrieval/qrels_v3.yaml --level chunk \
  --compare rag_chunk_baseline_v3 --method "rerank: gte-rerank-v2"
```

### Admin corpus expand

Crawls land in `data/kb_staging/` and need review before ingest.

```bash
# Manual: keyword + date range (admin)
python scripts/crawl_to_staging.py --admin --user admin
python scripts/crawl_to_staging.py --user admin \
  --keyword "比亚迪 海外" --from 2026-01-01 --to 2026-06-30

# Batch: data/crawl/search_queries.yaml
python scripts/crawl_to_staging.py --from-config

# Approve → data/kb/crawl/, then rebuild index
python scripts/promote_staging.py --approve <doc_id>
python scripts/ingest_kb.py --also-milvus
```

### Eval & tests

```bash
pytest -q
python evals/run_retrieval.py --tag kb_expand
python evals/run_eval.py --suite golden --tag after_s2 --ids Q1,Q2
python evals/run_memory_eval.py --tag after_memory
```
