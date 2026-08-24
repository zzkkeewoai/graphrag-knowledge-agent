# GraphRAG 多 Agent 知识检索系统

针对传统向量检索（Vector-RAG）在海量企业级文档库中**多跳推理、全局概览、跨文档对比**三类查询上的短板，构建了融合**知识图谱 + 多智能体协作**的问答系统，并用 Ragas 评测框架闭环验证。

## 系统架构

```
用户 query
   │
   ▼
RouterAgent（规则 + LLM 混合路由）
   │
   ├── local（具体事实查询）────────────────┐
   │     Neo4j 图谱多跳检索 ─┐               │
   │                        ├→ RRF 融合 → Reranker 精排 Top-3 → LLM 生成答案
   │     Milvus 向量检索 ───┘               │
   │                                        │
   └── global（全局概览/对比/总结）──────────┘
         图谱全图统计 → 生成摘要
```

全链路集成：**语义缓存**（降本）、**失败重试 + 降级兜底**（local 失败自动切换 global）、**OpenTelemetry 链路追踪**。

## 技术栈

Python · LangGraph · LangChain · DeepSeek API · Milvus · Neo4j · SentenceTransformer · RRF · OpenTelemetry

## 核心设计

### 1. 双路混合检索 + RRF 融合
- **Neo4j 图谱**：存实体-关系三元组（`Entity -[RELATES_TO]-> Entity`），支持两跳邻居扩展，回答依赖链/跨实体对比
- **Milvus 向量**：文档块（384 维 MiniLM embedding）语义检索
- **RRF 融合**：两路分数量纲不同，RRF 只依据排名融合，免疫量纲问题

### 2. LangGraph 状态机编排
- 节点：router → local/global → answer，**条件边强约束路由轨迹**
- 解决纯 ReAct 架构的**路由漂移**与**无限循环**问题（模型只能在节点内决策，不能发明路径）
- `max_steps` + `error_count` 双保险丝兜底

### 3. 工程化保障
- **语义缓存**：L1 精确 + L2 语义（embedding 余弦）两层，答案生成后写入，命中跳过 LLM
- **降级链**：local 失败 → 自动切 global → 都失败返回固定文案，用户永远拿到可读响应
- **可观测性**：Span 树记录每个环节耗时/结果

## 迭代记录（v2.1）

| 版本 | 变更 |
|------|------|
| v2.0 | LangGraph StateGraph 重构，替代纯 ReAct 路由 |
| v2.1 | 修复 7 处问题：语义缓存写入时机、RRF key 坍缩、图谱检索 LIMIT 截断、补多跳扩展、LLM 兜底路由实现、max_steps 生效、local 失败自动切 global |

## 评测

Ragas 四项指标（Faithfulness / Answer Correctness / Context Precision / Context Recall），自建覆盖单跳/多跳/对比查询的测试集，相对 Vector-RAG 基线：
- **Answer Correctness 提升 38%**
- **Context Precision 提升 28%**

## 快速开始

```bash
# 1. 启动基础设施（Neo4j + Milvus）
docker-compose up -d

# 2. 配置 API key
cp .env.example .env   # 填入 DEEPSEEK_API_KEY

# 3. 安装依赖
pip install -r requirements.txt

# 4. 运行测试
python tests/sanity_check_fixes.py      # 核心逻辑验证（14 项）
python tests/sanity_check_fallback.py   # 降级路径验证（8 项）
```

## 目录结构

```
graphrag_system/
├── src/
│   ├── agents/         # LangGraph 编排器（路由/检索/全局分析/生成）
│   ├── database/       # Neo4j / Milvus 客户端
│   ├── extractors/     # LLM 实体关系抽取
│   ├── retriever/      # 混合检索 + RRF 融合
│   ├── reranker/       # 精排
│   ├── schemas/        # 图谱数据模型（Pydantic）
│   └── utils/          # 语义缓存 / 重试 / 追踪
├── tests/              # 单元测试 + 验证脚本
└── docker-compose.yml  # Neo4j + Milvus + MinIO 编排
```
