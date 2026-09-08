# GraphRAG 多 Agent 知识检索系统（v2.2）

针对传统向量检索（Vector-RAG）在海量企业级文档库中**多跳推理、全局概览、跨文档对比**三类查询上的短板，构建了融合**知识图谱 + 多智能体协作**的问答系统。v2.2 新增**证据验证与冲突裁决**，把"检索相关"和"支持结论"分开，证据不足时**拒答**而非让 LLM 编造，并用 Ragas + Evaluation 闭环验证。

> 一句话定位：检索到相关内容 ≠ 能回答你的问题——用确定性规则验证证据是否足以支撑结论，不足就补召回、仍不足就拒答；图谱与文本冲突时按时间/来源/置信度裁决，防幻觉。

## 系统架构（v2.2 LangGraph 状态机）

```
用户 query
   │
   ▼
┌───────────────┐
│ router_node   │ RouterAgent：规则关键词（80%）→ LLM 兜底（模糊 case）→ 默认 local
└───────┬───────┘
        │ route_decision（条件边；step_count ≥ max_steps 或 error_count ≥ 3 → 直接 answer 收尾）
   ┌────┴────────────┐
   ▼ local           ▼ global
┌────────────┐   ┌────────────┐
│ local_node │   │ global_node│ GlobalAnalysisAgent：全图扫描 项目-使用/依赖 → 概览+统计摘要
└─────┬──────┘   └─────┬──────┘
      │ ① 语义缓存检查（L1 精确+L2 embedding，命中→from_cache）      │
      │ ② HybridRetriever 双路检索                                  │
      │    Neo4j 图谱（1跳直接三元组 + hops=2 两跳邻居扩展）            │
      │    Milvus 向量（top_k=5）                                    │
      │    → RRF 融合（k=60）取 top5 → BGEReranker 精排 Top-3        │
      │ ③ 失败重试 3 次（指数退避）                                    │
      │                                                             │
      │ local_after 条件边                                          │
      │  ├─ 3 次全失败 → fallback_to_global ──────────────► global ◄─┘（降级兜底）
      │  └─ 成功 ──► evidence_validator
      ▼
┌──────────────────┐
│ evidence_validator│ EvidenceValidator（v2.2，确定性规则，不调 LLM）
└────────┬─────────┘  · 从 query 提取 target_relation / target_entity
         │            · 图谱 relation 结构化匹配正向证据
         │            · 文本方向判断（支持/反对/否定词同现）
         │ next_action 条件边
   ┌─────┼───────────────────────────────┐
   ▼     ▼                               ▼
answer  targeted_graph                  refuse
   │     （定向图谱补充召回：             （模板拒答文案）
   │      entity + relation_type 精确查询 → 回 evidence_validator 二次判断；
   │      仍不足 → refuse；step 超限 → refuse）
   ▼
conflict
   │ ConflictResolver（v2.2，确定性规则：时间信号 → 来源 → 置信度）
   │  ├─ resolved   → 基于胜出证据模板化生成答案
   │  └─ unresolved → 拒答（"存在冲突信息，无法可靠确认"）
   ▼
answer_node
   · from_cache 命中 → 跳过 LLM 直接复用缓存答案
   · evidence_status ∈ {insufficient, conflict} → 跳过 LLM（用拒答/冲突文案）
   · 否则 AnswerAgent 生成（模板 / LLM），生成成功后才写入语义缓存
```

## 核心设计

### 1. 双路混合检索 + RRF 融合
- **Neo4j 图谱**：存实体-关系三元组（`Entity -[RELATES_TO]-> Entity`），v2.1 起过滤下推 Cypher（不再 LIMIT 100 截断）、doc_id 唯一（修复 RRF key 坍缩）、支持两跳邻居扩展（回答依赖链/跨实体对比）
- **Milvus 向量**：文档块（384 维 MiniLM embedding）语义检索
- **RRF 融合**：两路分数量纲不同，RRF 只依据排名融合（`1/(k+rank)`，k=60），免疫量纲问题

### 2. LangGraph 状态机编排
- 节点：router → local/global → evidence_validator →（answer / targeted_graph / refuse / conflict）→ answer
- 条件边强约束路由轨迹，解决纯 ReAct 的**路由漂移**与**无限循环**问题
- `max_steps` + `error_count` 双保险丝兜底

### 3. 证据验证与拒答（v2.2）
- 核心语义："相关（relevant）" ≠ "支持结论（support）"。检索到"张三参与项目A"不能证明"张三负责项目A"
- 关键陷阱规避：`relation != expected_relation` 只判 **insufficient**（证据不足），**绝不**生成负向证据（"没有证据证明负责" ≠ "证明不负责"）
- 不足 → **定向图谱补充召回**（targeted graph，entity + relation_type 精确查询）→ 仍不足 → **拒答**，不让 LLM 基于不充分证据自由发挥

### 4. 冲突裁决（v2.2）
- 图谱说"负责"、文本说"不再负责"（否定词+实体+关系同现）→ 判定 conflict
- ConflictResolver 确定性裁决：**时间信号 → 来源 → 置信度**；可裁决 → 模板化输出胜方答案；不可裁决 → 明确说明"存在冲突信息"，不编造

### 5. 工程化保障
- **语义缓存**：L1 精确 + L2 语义（embedding 余弦）两层；v2.1 修复写入时机（答案生成后写入，命中跳过 LLM，不再存空 answer）；v2.2 起 threshold/ttl/max_size/knowledge_version 可配置，知识更新 bump 版本自动失效旧缓存
- **降级链**：local 失败 3 次 → 自动切 global → 都失败返回固定文案，用户永远拿到可读响应
- **可观测性**：Span 树记录每个环节耗时/结果（OpenTelemetry）

## 迭代记录

| 版本 | 变更 |
|------|------|
| v1.0 | 关键词 Router + if-else 编排，双路检索 + RRF |
| v2.0 | LangGraph StateGraph 重构，替代纯 ReAct 路由；语义缓存、Ragas 评估、链路追踪 |
| v2.1 | 修复 7 处：语义缓存写入时机、RRF key 坍缩、图谱检索 LIMIT 截断、补多跳扩展、LLM 兜底路由实现、max_steps 生效、local 失败自动切 global |
| v2.2 | **证据验证与拒答**（EvidenceValidator）、**定向图谱补充召回**（targeted_graph）、**冲突裁决**（ConflictResolver）、缓存/检索参数配置化、Evaluation 闭环（13 条场景数据集 + 4 项指标回归） |

## 评测

### Ragas 框架（v2.0）
四项指标（Faithfulness / Answer Correctness / Context Precision / Context Recall），自建覆盖单跳/多跳/对比查询的测试集，相对 Vector-RAG 基线：
- **Answer Correctness 提升 38%**
- **Context Precision 提升 28%**

### Evaluation 闭环（v2.2）
`tests/evaluation/` 走真实生产路径（逻辑层）：query → Router → Retrieval → RRF → Reranker → EvidenceValidator → ConflictResolver → Answer/Refuse，注入假 Neo4j/Milvus/Reranker，不依赖 Docker/外部 LLM。
- 13 条场景数据集：normal_fact / relation_single_hop / multi_hop / conflict_resolved / conflict_unresolved / insufficient_refuse / not_in_kb / semantic_trap 等
- 4 项指标：**Retrieval Recall**（目标实体/目标是否进入候选）、**Evidence Validation Accuracy**（expected_evidence vs 实际）、**Refusal Accuracy**（应拒答是否真拒答）、**Answer Accuracy**（确定性答案是否命中目标）

## 快速开始

```bash
# 1. 启动基础设施（Neo4j + Milvus + etcd + MinIO）
docker compose up -d

# 2. 配置 API key（仅 LLM 生成/路由需要；模板模式可不配）
cp .env.example .env   # 若存在；否则手动设置 DEEPSEEK_API_KEY 环境变量

# 3. 安装依赖
pip install -r requirements.txt

# 4. 运行测试
python tests/sanity_check_fixes.py        # 核心逻辑验证（v2.1 修复回归，无 Docker）
python tests/sanity_check_fallback.py     # 降级路径验证（local→global 切换）
python tests/evaluation/run_evaluation.py # v2.2 Evaluation 闭环（13 条场景 4 项指标）
pytest tests/ -v                          # 单元测试（含 evidence/conflict 测试）
```

## 目录结构

```
graphrag_system/
├── src/
│   ├── agents/
│   │   ├── orchestrator_v2.py      # LangGraph 编排器（8 节点状态机）
│   │   ├── evidence_validator.py   # v2.2 证据验证器（确定性规则）
│   │   └── conflict_resolver.py    # v2.2 冲突裁决器（时间→来源→置信度）
│   ├── database/
│   │   ├── neo4j_client.py         # Neo4j 图数据库客户端
│   │   └── milvus_client.py        # Milvus 向量数据库客户端
│   ├── extractors/
│   │   └── triple_extractor.py     # LLM 实体关系三元组抽取（JSON Schema 约束）
│   ├── retriever/
│   │   └── hybrid_retriever.py     # 双路检索 + 两跳扩展 + RRF 融合
│   ├── reranker/
│   │   └── reranker.py             # BGE/MiniLM 精排
│   ├── schemas/
│   │   └── graph_schema.py         # Pydantic 图谱数据模型
│   ├── utils/
│   │   ├── semantic_cache.py       # L1 精确 + L2 语义缓存
│   │   ├── retry_decorator.py      # 指数退避重试
│   │   └── tracing.py              # OpenTelemetry Span 追踪
│   └── config.py                   # 全局配置 + fail-fast 校验
├── tests/
│   ├── sanity_check_fixes.py       # v2.1 修复回归验证
│   ├── sanity_check_fallback.py    # 降级路径验证
│   ├── test_evidence_validator.py  # v2.2 证据验证测试
│   ├── test_conflict_resolver.py   # v2.2 冲突裁决测试
│   ├── test_ragas_eval.py          # Ragas 评估
│   ├── test_agents.py / test_reranker.py / test_hybrid_retrieval.py ...
│   └── evaluation/
│       ├── dataset.json            # 13 条场景数据集
│       └── run_evaluation.py       # Evaluation Runner（4 项指标）
├── data/eval_queries.json          # 示例评测查询
├── docker-compose.yml              # Neo4j + Milvus + etcd + MinIO
└── requirements.txt                # Python 依赖
```

## 技术栈

Python · LangGraph · LangChain · DeepSeek API · Milvus · Neo4j · SentenceTransformer · RRF · Reranker · OpenTelemetry · Pydantic
