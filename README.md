# NEV Research Agent

新能源汽车行业研究分析 Agent：本地知识库 + 联网检索，输出带引用的短答或分析文档。

路径：KB → S1 → S1.5 记忆 → **S2 四智能体（当前默认）** → 缓存 / GraphRAG。

只回答整车 / 电池 / 供应链 / 政策 / 竞品。非新能源直接拒答。「你会做什么」走固定介绍。

## 仓库地图

| 路径 | 做什么 |
|------|--------|
| `main.py` | CLI 入口（薄封装 → `src/agent/cli.py`） |
| `src/agent/` | 研究 Agent：runtime、core、memory、tools、s1、s2 |
| `src/kb/` | 本地语料入库 / 余弦检索 |
| `evals/` | 评测：`datasets/` 金标、`runners/` 脚本、`baselines/` 快照 |
| `scripts/` | 运维：`ingest_kb` / `ask_kb` |
| `tests/` | 单测 |
| `docs/` | 说明（本机留存，gitignore） |
| `data/` | 语料与索引、用户记忆（gitignore） |

## 安装

```bash
cd nev-research-agent
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .                 # 把 src/agent、src/kb 装进环境
cp .env.example .env             # 填 OPENAI/DASHSCOPE；可选 TAVILY_API_KEY
python scripts/ingest_kb.py
python main.py
```

未 `pip install -e .` 时，`main.py` / `evals/run_*.py` / `scripts/*` 会自动把 `src/` 加进 `sys.path`。

## 当前：S2 + 记忆

- 默认 `AGENT_MODE=s2`：`src/agent/s2/{supervisor,research,analyst,verifier}/`
- 回退：`AGENT_MODE=s1` → `src/agent/s1/`
- KB：`data/kb/` → `data/kb_index/`；联网 Tavily → DDG → Bing
- 黄金集关记忆；引用必须来自本轮工具

```bash
python main.py "比亚迪刀片电池的核心卖点是什么？"
python main.py --memory --user alice "分析比亚迪海外扩张"
python scripts/ask_kb.py "比亚迪刀片电池的核心卖点是什么？"
```

## 评测

```bash
pytest -q
python evals/run_retrieval.py --tag kb_expand --method "本地余弦检索 13 篇 KB"
python evals/run_eval.py --suite golden --tag after_s2 --ids Q1,Q2,G1 \
  --method "S2 四节点" --compare after_tavily
python evals/run_memory_eval.py --tag after_memory
```

实现位于 `evals/runners/`；根下 `evals/run_*.py` 为兼容入口。
