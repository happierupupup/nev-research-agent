# NEV Research Agent

新能源汽车行业研究分析 Agent：本地知识库 + 联网检索，输出带引用的短答或分析文档。

路径：KB → S1 单 Agent → S1.5 记忆 → **S2 四智能体（当前默认）** → 缓存 / GraphRAG。

只回答整车 / 电池 / 供应链 / 政策 / 竞品。非新能源直接拒答（不调模型）。「你会做什么」走固定介绍。

## 当前：S2 + 记忆

- LangGraph：**Supervisor → Research → Analyst → Verifier**（代码在 `agent/agents/{角色}/`，各有独立 prompts + 逻辑；`AGENT_MODE=s2`）；`AGENT_MODE=s1` 回退单 Agent ReAct
- Research：`kb_search` + 按需 `web_search`（Tavily → DuckDuckGo → Bing HTML），只产素材
- KB：`data/kb/` 13 篇摘要 → `data/kb_index/` JSON 余弦索引（不是 Chroma）
- REPL 短期记忆（近窗回合 + 溢出摘要）+ 按用户长期画像；黄金集默认关记忆
- 引用必须来自本轮工具，记忆不当 `(来源: …)`

```bash
cd nev-research-agent
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # OPENAI_API_KEY 或 DASHSCOPE；可选 TAVILY_API_KEY
python scripts/ingest_kb.py
python main.py                # REPL：思考过程 → 回答 → 信息来源；退出时写长期记忆
```

```bash
python main.py "比亚迪刀片电池的核心卖点是什么？"          # one-shot，默认关记忆
python main.py --memory --user alice "分析比亚迪海外扩张"  # one-shot 也读写该用户画像
python scripts/ask_kb.py "比亚迪刀片电池的核心卖点是什么？"  # 仅 KB，不走 Agent
```

## 评测

黄金集 live 必须 `--method`。每次写入 `{tag}_{UTC}.json` 快照 + `{tag}.json` 最新别名。

```bash
pytest -q
python evals/run_retrieval.py --tag kb_expand --method "本地余弦检索 13 篇 KB"
python evals/run_eval.py --suite golden --tag s1_tokens --ids Q1,Q2,G1 \
  --method "S1 ReAct kb+web；关记忆；token 基线" --compare s1_baseline
python evals/run_memory_eval.py --tag after_memory
```



