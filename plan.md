# nev-research-agent 实现计划

> **定位：** 新能源汽车行业研究分析 Agent —— 输入研究命题 → 自主规划 → 多源采集 → 量化/关系推理 → 质量闭环 → 带引用报告  
> **不是：** 简单 RAG 客服、固定 Workflow 流水线  
> **原则：** 评估先行 → 单 Agent 验证主链路 → 工具增强 → 自验证 → 多智能体 → GraphRAG；窄领域先跑通，不堆复杂度  
> **评估入口：** `evals/`（rubric + 黄金用例 + baseline 快照，用于前后对比）

---

## 0. 一句话

研究分析 = **规划 → 多源检索 → 量化分析 → 关系推理 → 综合研判 → 质量闭环**。  
典型命题：「分析比亚迪 2024 年海外扩张战略及风险评估」。

---

## 1. 为什么值得做 Agent（失败代价）

| 失败 | 代价 |
|------|------|
| 事实无引用 / 幻觉 | 投研会上用错材料，误导投资判断 |
| 多源数据矛盾未标注 | 把分歧当定论，风险被掩盖 |
| 关系推理错误（股权/供应链） | 错判集中度与依赖风险 |
| 报告空泛无数据支撑 | 咨询/战略产出不可用，人力白耗 |

本项目追求：**输出可信度可量化、可追溯、可前后对比**；不是堆 Agent 数量。

---

## 2. 评估先行（Day 0 必做）

> 避坑清单第 5 条：评估体系先行于代码。  
> **刚开始就必须有测试指标入口**，否则 Phase 后期无法证明「质量提升 ≥20%」。

### 2.1 五维 Rubric（总分 100）

| 维度 | 权重 | 评分要点 |
|------|------|----------|
| D1 覆盖度 | 20 | 研究框架完整：战略/数据/风险/竞品等关键维度是否覆盖 |
| D2 引用准确性 | 25 | 事实点有来源；引用可回溯；无「空口事实」 |
| D3 数据支撑 | 20 | 量化结论有数；图表/指标与正文一致 |
| D4 逻辑一致性 | 20 | 前后无矛盾；推理不跳跃；结论可由证据推出 |
| D5 深度与可行动性 | 15 | 非百科堆砌；有风险/启示/可验证判断 |

**质量分公式：** `score = Σ(维度分 × 权重)`；人工抽检 +（Phase 3+）Verifier 代理评分共用同一 rubric。

### 2.2 黄金用例集（Phase 1 至少 3 条，目标 8～10 条）

| ID | 命题 | 重点验收 |
|----|------|----------|
| G1 | 分析比亚迪 2024 年海外扩张战略及风险评估 | 主链路、引用 ≥5、结构完整 |
| G2 | 梳理宁德时代电池供应链的集中度与潜在风险 | 供应链维度、多源、风险识别 |
| G3 | 对比特斯拉 vs 比亚迪：成本结构、技术路线、市场份额变化 | 对比结构、数据章占比 |

完整定义见 `evals/golden/queries.yaml`；人工期望见同目录 `expectations/`。

### 2.3 硬指标（前后对比用）

| 指标 | Phase 1 门槛 | Phase 3 | Phase 4 | Phase 5 |
|------|-------------|---------|---------|---------|
| 报告字数 | ≥5000 | — | — | — |
| 带引用事实点 | ≥5 | 关键数据 ≥2 源 | — | — |
| 自主检索轮次 | 2～3 | ≤3 总轮次 | — | — |
| 端到端耗时 | ≤5 min | — | ≤6 min | — |
| Rubric 均分 | 基线记录 | ≥基线 | 较基线 **+20%** | — |
| 数据洞察章节占比 | — | — | — | — |
| 数据分析准确率 | — | ≥90%（P2） | — | — |
| 逻辑矛盾率 | — | <5% | — | — |
| 自动修复成功率 | — | ≥70% | — | — |
| 关系类洞察占比 | — | — | — | ≥20% |

**跑法：**

```bash
# 记录当前实现为 baseline（改架构前必跑）
python evals/run_eval.py --suite golden --tag baseline_phase1

# 改完再跑，自动对比
python evals/run_eval.py --suite golden --tag after_xxx --compare baseline_phase1
```

产物：`evals/baselines/<tag>.json` + 控制台 diff（各维度分、耗时、引用数）。

### 2.4 评估入口文件

| 路径 | 作用 |
|------|------|
| `evals/rubric.yaml` | 五维定义与权重（Verifier / 人工共用） |
| `evals/golden/queries.yaml` | 黄金命题 + 硬门槛 |
| `evals/run_eval.py` | 统一评测入口（前后对比） |
| `evals/baselines/` | 各 tag 快照，勿手改 |
| `evals/metrics.py` | 指标抽取：字数、引用数、耗时、rubric 聚合 |

---

## 3. 架构：四层能力 + Supervisor 编排

### 3.1 能力分层

| 层 | 能力 | 核心价值 |
|----|------|----------|
| 编排层 | 多智能体分工与协作 | 专业化、可对抗、可并行 |
| 推理层 | GraphRAG 关系发现与多跳 | 超越片段检索 |
| 量化层 | 统计分析 + 可视化 | 财务/市场数据开口说话 |
| 质量层 | 异常检测与自我修正 | 可信度可量化、可追溯 |

### 3.2 多智能体（Supervisor + 3 Specialist）

```
START → Supervisor ─┬→ Research  ─┐
                    ├→ Analyst   ─┤→ Supervisor（汇聚裁决）
                    └→ Verifier  ─┘
                                      ├─ 通过 → END
                                      └─ 不通过 → 按 Issue 类型路由回对应 Agent
```

| 角色 | 职责 | 不做 |
|------|------|------|
| **Supervisor** | 拆解命题、路由、裁决 Issue、判定达标 | 不写长分析（System Prompt ≤1000 tokens） |
| **Research** | 多源采集，输出带元数据的原始素材 | 不做总结 |
| **Analyst** | 文本 + 数据 + 图谱 → 中间报告 | 不做最终质量闸 |
| **Verifier** | 对抗挑错 → Issue List（级别+Citation） | 不直接改报告正文 |

**交互：** Research→Analyst（素材流）→ Verifier（结论流）→ Supervisor（Issue）；Verifier 可定向要求 Research 补采。

### 3.3 LangGraph 状态（分区，禁止扁平大 dict）

```python
class ResearchState(TypedDict):
    query: str
    research_plan: Dict
    research_workspace: List[Dict]   # 原始素材区
    analysis_workspace: Dict         # 中间结论 + 图表
    verification_issues: List[Dict]  # Issue List
    current_phase: str               # researching|analyzing|verifying|finalizing
    iteration_count: int             # ≤3
    final_report: str
```

选型理由：多节点状态图、Conditional Edge 打回、Cycle+递归限制、Interrupt HITL、Checkpoint 断点续跑、LangSmith/Langfuse 可观测。

### 3.4 双索引（Phase 1 只用向量；Phase 5 补图）

| 能力 | Milvus | 图存储（PG + LlamaIndex Property Graph） |
|------|--------|------------------------------------------|
| 文档片段 / 相似度 | ✅ | ❌ |
| 多跳 / 邻域 / 社区摘要 / 关系聚合 | ❌ | ✅ |

### 3.5 异常检测四层

| 层 | 类型 | 检测 | 修复 |
|----|------|------|------|
| L1 | 数据源失效/超时/限流 | 工具异常捕获 | 备用源 + 指数退避 |
| L2 | 多源事实矛盾 | Cross-source diff | 第三源 / 标注分歧+置信度 |
| L3 | 逻辑矛盾、推理跳跃 | LLM 自洽检查 | Analyst 重写 + 再验 |
| L4 | 深度不足、缺维度 | Rubric（Supervisor） | 重规划补子任务 |

**Issue 铁律：** Critical 必修 / Minor 可标注放过；必须带 Citation；重试 ≤3 后 HITL；Issue 与修复落库反哺。

---

## 4. 技术栈

| 层 | 方案 | 说明 |
|----|------|------|
| 编排 | LangGraph | 与本架构匹配 |
| LLM | 强模型做 Analyst/Verifier；轻量做 Supervisor | 降本 |
| 向量 | Milvus | 传统 RAG |
| 图（P1–4 轻量 / P5） | PostgreSQL + LlamaIndex Property Graph | 可升 Nebula/Neo4j |
| 数据沙箱 | E2B / Modal / Docker | **禁止**宿主 `exec` LLM 代码 |
| 可观测 | LangSmith / Langfuse | 多 Agent 必接 |
| 参考实现 | GPT-Researcher / STORM / LangGraph examples / MS GraphRAG / nano-graphrag | 见 §9 |

**Analyst 工具化（非角色化）：** `data_analysis_toolkit`（沙箱）∥ `graph_rag_toolkit`（图）∥ `synthesize`（综合）。

---

## 5. 目标目录

```
nev-research-agent/
├── plan.md                 # 本文（阶段与验收以本文为准）
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── main.py                 # CLI：输入命题 → 报告
├── agent/
│   ├── state.py
│   ├── graph.py
│   ├── schemas.py          # Issue / Plan / Citation JSON Schema
│   ├── nodes/              # supervisor / research / analyst / verifier（Phase 4 拆分）
│   ├── tools/              # search / scrape / milvus / sql / sandbox / graph
│   ├── memory.py           # 任务级压缩与 checkpoint 辅助
│   └── prompts/
├── data/                   # 领域词典、样例语料、seed（后期）
├── evals/                  # ★ Day 0 评估入口
│   ├── rubric.yaml
│   ├── metrics.py
│   ├── run_eval.py
│   ├── golden/
│   │   └── queries.yaml
│   └── baselines/
├── scripts/
├── tests/
└── notes/                  # failure-log / timing（面试与迭代素材）
```

Phase 1 可先单文件 `nodes.py`，Phase 4 再拆 `nodes/`。

---

## 6. 五阶段交付

| Phase | 目标 | 周期 | 验收标志 |
|-------|------|------|----------|
| **1** 单 Agent MVP | 命题→采集→报告最小闭环 | 2–3 周 | 主流程跑通；evals 有 baseline |
| **2** 工具增强 | 财务/SQL/图表 + 多源并行 | 3–4 周 | 数据洞察 ≥30%；图表 ≥3 种 |
| **3** 自验证 | Issue + 定向修复闭环 | 3–4 周 | 关键数据 ≥2 源；修复成功率 ≥70% |
| **4** 多智能体 | Supervisor + 3 Specialist | 4–5 周 | Rubric 较 Phase1 基线 **+≥20%** |
| **5** GraphRAG | 股权/供应链关系推理 | 5–6 周 | 关系洞察 ≥20%；图规模达标 |

**总估：** 17–22 周。里程碑：P1 demo → P3 内部 Beta → P5 可商业化。

### Phase 1：单 Agent MVP（缓存减 token）

- 命题解析、Milvus 基础 RAG、搜索+网页抓取 2–3 工具、LangGraph 简单循环、Markdown 报告  
- 验收：G1 出 ≥5000 字结构化报告；≥5 带引用事实；自主检索 2–3 轮；≤5 min  
- **必须：** `run_eval.py --tag baseline_phase1` 落盘

### Phase 2：工具增强

- Python 沙箱分析、Milvus + SQL + 财务 API、结果融合与工具路由  
- 验收：SQL 正确；折线/柱/饼 ≥3；数据洞察章 ≥30%；抽检准确率 ≥90%

### Phase 3：自验证

- Cross-source、逻辑一致性、Z-score 数据异常、Issue JSON Schema、定向修复路由  
- 验收：关键数据 ≥2 源；逻辑矛盾率 <5%；自动修复 ≥70%；平均 ≤3 轮

### Phase 4：多智能体

- Supervisor 编排、Conditional Edge、共享工作区、Verifier Issue List  
- 验收：拆解 ≥5 子问题；Research 效率 ↑50%；Verifier 发现问题数 ≥2× 单 Agent；耗时 <6 min；Rubric +≥20%

### Phase 5：GraphRAG

- 实体关系构建、Local/Global/Drift、Dual RAG 融合  
- 验收：≥1000 实体 / ≥5000 关系（窄域如动力电池供应链）；查询准确率 ≥85%；关系洞察 ≥20%；建图 <24h

**顺序铁律：** Phase 4 不稳，不上 Phase 5；图谱先窄域，不全行业。

---

## 7. 可优化点（后置，不挡主路径）

| 级别 | 内容 | 实现 |
|------|------|------|
| 任务级记忆 | 素材/结论/Issue | LangGraph State + 向量缓存 + 压缩 |
| 用户级记忆 | 报告风格、关注公司 | 画像 DB + 个性化 prompt |
| 组织级记忆 | 历史报告与修复模式 | 归档 + 案例检索 |
| 安全 | 脱敏、审计日志、内容审核 | 正则+NER / 结构化日志 / 审核模型 |
| 权限 | RBAC、数据源令牌、高危双人复核 | 后期产品化再做 |

---

## 8. 落地避坑（执行清单）

1. 图谱先窄域（动力电池供应链），验证价值再扩  
2. Verifier **必须** JSON Schema，禁止散文式吐槽  
3. 沙箱隔离，禁止宿主裸跑 LLM 代码  
4. Checkpoint + 上下文压缩，控 token  
5. **评估先行**（本文 §2 + `evals/`）  
6. 接 LangSmith / Langfuse，否则多 Agent 不可查  
7. Supervisor prompt ≤1000 tokens，只路由与决策  
8. Phase 4 稳定前不上 GraphRAG  

---

## 9. 参考仓库

| 仓库 | 价值 | 我们怎么用 |
|------|------|------------|
| [GPT-Researcher](https://github.com/assafelovic/gpt-researcher) | 研究→报告主流程 | Phase 1 最佳参考 / fork 改造 |
| [STORM](https://github.com/stanford-oval/storm) | Writer+Critic 对抗闭环 | Phase 3–4 Verifier 教科书 |
| [CrewAI](https://github.com/crewAIInc/crewAI) | 角色定义简洁 | 学角色划分；状态/checkpoint 弱，不主用 |
| [LangGraph examples](https://github.com/langchain-ai/langgraph/tree/main/examples) | multi_agent / research_assistant | **主技术栈，优先改** |
| [MS GraphRAG](https://github.com/microsoft/graphrag) | Local/Global/Drift | Phase 5 对齐查询模式（重、贵） |
| [nano-graphrag](https://github.com/gusye1234/nano-graphrag) | 轻量 GraphRAG | 中小起步备选 |
| [LlamaIndex Property Graph](https://github.com/run-llama/llama_index) | 属性图 + Milvus 友好 | 项目内 GraphRAG 首选 |

**组合策略：** GPT-Researcher（主流程）→ LangGraph 重构（Supervisor）→ LlamaIndex PG（图）→ E2B/Modal（沙箱）→ Milvus（向量）。

---

## 10. 本周立刻做（不写业务 Agent 逻辑也要先做完）

| # | 任务 | 产出 |
|---|------|------|
| 1 | 固化 rubric + 黄金 3 题 | `evals/rubric.yaml`、`evals/golden/queries.yaml` |
| 2 | 评测入口可跑（无 Agent 时打桩） | `evals/run_eval.py` 能写 baseline JSON |
| 3 | 空壳 CLI + 依赖清单 | `main.py`、`requirements.txt`、`.env.example` |
| 4 | Phase 1 最小图骨架（单节点也可） | `agent/state.py` + `agent/graph.py` stub |

改完任何阶段：**先 `--compare` 再宣称提升。**
