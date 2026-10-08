# NEV Research Agent

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
├── web_app.py              # Gradio + SSE Web 入口
├── config/users.yaml       # 测试用户（admin / alice / dev）+ 密码
├── requirements.txt
├── .env.example
├── src/
│   ├── agent/              # 研究 Agent
│   │   ├── cli.py          # 命令行交互
│   │   ├── runtime.py      # 运行入口
│   │   ├── display.py      # 角色分流展示
│   │   ├── web/            # Gradio UI + FastAPI SSE
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

### Gradio Web（推荐）

```bash
pip install -r requirements.txt   # Gradio 3.x，兼容 Python 3.9
python web_app.py
# 浏览器打开 http://127.0.0.1:7860/ui/
```

| 能力 | 说明 |
|------|------|
| 登录 | `config/users.yaml`；默认 `admin/admin`、`alice/alice` |
| 会话 | 左侧列表，新建/切换/删除；状态落盘 `data/sessions/`（消息、工具事件、任务进展、STM） |
| 对话 | 右侧输入输出；流式思考 + 回答（底层 SSE：`POST /api/sessions/{id}/stream`） |
| 角色 | 管理员：完整工具调用；普通用户：仅简要思考阶段 |
| 管理 | 管理员页：用户 CRUD、手动/批量爬虫、staging 审核 promote |

环境变量：`NEV_WEB_HOST`（默认 `127.0.0.1`）、`NEV_WEB_PORT`（默认 `7860`）。

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
