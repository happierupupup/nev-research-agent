# NEV Research Agent

新能源汽车行业研究分析 Agent：信息收集、检索、分析总结，输出带引用的分析文档。

路径：知识库 → **S1 单 Agent（当前）** → 拆四智能体 → 优化 → GraphRAG。详见 [`plan.md`](./plan.md)。

## 当前：S1 单 Agent

```bash
cd nev-research-agent
pip install -r requirements.txt
cp .env.example .env          # API Key；可选 TAVILY_API_KEY
python scripts/ingest_kb.py
python main.py                # 思考过程实时输出 → 带引用回答 → 信息来源
python evals/run_eval.py --tag s1_baseline --ids Q1,Q2
```

仅知识库问答（无 Agent）：`python scripts/ask_kb.py "..."`。

## 文档

| 文档 | 说明 |
|------|------|
| [plan.md](./plan.md) | 需求对齐与实现计划 |
| [docs/architecture.md](./docs/architecture.md) | 说明文档 |
| [docs/usage.md](./docs/usage.md) | 使用文档 |
| [docs/testing.md](./docs/testing.md) | 测试与评测 |
| [docs/failure-log.md](./docs/failure-log.md) | 问题与改进记录 |
