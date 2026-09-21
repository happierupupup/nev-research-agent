# NEV Research Agent

新能源汽车行业研究分析 Agent：信息收集、检索、分析总结，输出带引用的分析文档。

路径：**知识库问答（当前）** → 单 Agent → 拆四智能体 → 优化 → GraphRAG。详见 [`plan.md`](./plan.md)。

## 当前：知识库问答

```bash
cd nev-research-agent
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # 填写 API Key；百炼见 docs/usage.md
python scripts/ingest_kb.py   # 嵌入 data/kb → data/kb_index
python scripts/ask_kb.py "比亚迪刀片电池的核心卖点是什么？"
```

## 路径（完整）

1. **S0** 评测底座  
2. **KB** 本地知识库检索 + LLM 作答 ← **现在**  
3. **S1** 单 Agent（+ 联网搜索）  
4. **S2** 拆四 Agent → **S3** 优化 → **S4** GraphRAG → **S5** 文档/Skill  

## 文档

| 文档 | 说明 |
|------|------|
| [plan.md](./plan.md) | 需求对齐与实现计划 |
| [docs/architecture.md](./docs/architecture.md) | 说明文档 |
| [docs/usage.md](./docs/usage.md) | 使用文档 |
| [docs/testing.md](./docs/testing.md) | 测试与评测 |
| [docs/failure-log.md](./docs/failure-log.md) | 问题与改进记录（每次踩坑/改动追加一条） |

## 评测

```bash
python evals/run_eval.py --suite golden --tag before_xxx --dry-run
python evals/run_eval.py --suite golden --tag after_xxx --compare before_xxx
```
