# NEV Research Agent

新能源汽车行业研究分析 Agent：输入研究命题 → 规划与多源采集 →（后续）量化 / 关系推理 / 质量闭环 → 带引用的结构化报告。

当前仓库处于 **Day 0**：评估入口与 `plan.md` 已就位，Agent 主链路按 Phase 1 推进。

## 评估先行（前后对比）

```bash
cd nev-research-agent
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 打通指标管道（无 Agent 时）
python evals/run_eval.py --suite golden --tag baseline_phase1 --dry-run

# Agent 就绪后去掉 --dry-run；改完再对比
python evals/run_eval.py --suite golden --tag after_xxx --compare baseline_phase1
```

| 文件 | 作用 |
|------|------|
| `evals/rubric.yaml` | 五维评分（覆盖度 / 引用 / 数据 / 逻辑 / 深度） |
| `evals/golden/queries.yaml` | G1–G3 黄金命题 |
| `evals/run_eval.py` | 统一评测入口 |
| `evals/baselines/` | 各 tag 快照 |

阶段目标与验收以 [`plan.md`](./plan.md) 为准。

## 典型命题

- 分析比亚迪 2024 年海外扩张战略及风险评估
- 梳理宁德时代电池供应链的集中度与潜在风险
- 对比特斯拉 vs 比亚迪：成本结构、技术路线、市场份额变化

## 安装与运行（Phase 1 stub）

```bash
cp .env.example .env   # 填入 API Key
python main.py
```

## 技术路线（摘要）

LangGraph 编排 → Milvus RAG →（P2）沙箱数据分析 →（P3）Verifier Issue 闭环 →（P4）Supervisor+3 Specialist →（P5）GraphRAG。
