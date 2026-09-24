# GraphRAG 多 Agent 知识检索系统（v2.3）

面向企业级文档库的问答系统：融合**知识图谱 + 向量检索**双路召回，用 **LangGraph 多 Agent 状态机**编排，内置**证据验证与拒答**机制防幻觉，并完成**服务化、可观测性与高可用**的工程化落地。

> **一句话定位**：检索到相关内容 ≠ 能回答你的问题——用确定性规则验证证据是否足以支撑结论，不足就补召回、仍不足就**拒答**；图谱与文本冲突时按时间/来源/置信度裁决。
> **v2.3 重点**：FastAPI 服务层 · SSE 流式输出 · Prometheus 语义化指标 · Redis 分布式语义缓存 · 多 Provider 热备 · 压测数据（P95 64ms / 27 RPS）

---

## 系统架构

```
┌────────────────────────────────────────────────────────────────┐
│  客户端（Web / App / 其他服务）                                  │
└───────────────────────────┬────────────────────────────────────┘
                            │ HTTP · SSE 流式
                            ▼
┌────────────────────────────────────────────────────────────────┐
│  服务层  api.py（FastAPI）                                      │
│    POST /ask           问答                                     │
│    POST /ask/stream    SSE 流式问答（证据验证前置）               │
│    GET  /health        健康检查（依赖深度探测）                   │
│    GET  /metrics       Prometheus 指标                          │
│    GET  /docs          Swagger 接口文档                          │
│  横切：Pydantic 校验 · 超时控制 · 降级兜底 · 请求ID · 优雅关闭      │
└───────────────────────────┬────────────────────────────────────┘
                            ▼
┌────────────────────────────────────────────────────────────────┐
│  编排层  AgentOrchestratorV2 —— LangGraph 8 节点状态机           │
│    router → local/global → evidence_validator                   │
│              → answer / targeted_graph / refuse / conflict       │
└───────────────────────────┬────────────────────────────────────┘
                            ▼
┌────────────────────────────────────────────────────────────────┐
│  检索层  HybridRetriever（图谱+向量双路 → RRF k=60）→ Reranker    │
└──────────┬──────────────────────────────────┬──────────────────┘
           ▼                                  ▼
┌────────────────────────┐        ┌────────────────────────────┐
│ Neo4j  图数据库         │        │ Milvus 向量库               │
│ 实体关系 + 两跳邻居      │        │ document_chunks（384 维）    │
└────────────────────────┘        └────────────────────────────┘
           ▲                                  ▲
           └──────────┬───────────────────────┘
                      │
        ┌─────────────┴──────────────┐
        │  Redis 语义缓存（分布式）    │  多实例共享 · 重启不丢 · 故障自动降级
        └────────────────────────────┘
```

### LangGraph 状态机（查询链路）

```
query
  │
  ▼
router ──→ local ──→ evidence_validator ──┬─→ answer
  │          │                            ├─→ targeted_graph ─┐
  │          │ 检索失败 3 次                ├─→ refuse         │（定向补召回后
  │          └──────────→ global           └─→ conflict       │  二次判断）
  │                            │                              │
  └────────────────────────────┴──────────────────────────────┘
                          answer / refuse → END
```

---

## 核心设计

### 1. 双路混合检索 + RRF 融合
- **Neo4j 图谱**：实体关系三元组（统一 `RELATES_TO` + 属性 `relation_type`），支持**两跳邻居扩展**回答依赖链类问题
- **Milvus 向量**：文档块 384 维 embedding，IVF_FLAT + COSINE 度量
- **RRF 融合**：两路分数量纲不同（向量=余弦、图谱=置信度），RRF 只按排名融合（`1/(k+rank)`，k=60），免疫量纲问题
- **Reranker 精排**：Top-5 → Top-3，保证进入 LLM 上下文的证据最相关

### 2. LangGraph 状态机编排
- 节点：`router → local/global → evidence_validator → answer/targeted_graph/refuse/conflict`
- **条件边强约束路由轨迹**，解决纯 ReAct 的**路由漂移**与**无限循环**
- `max_steps` + `error_count` 双保险丝；`targeted_retried` 保证补召回只做一次

### 3. 证据验证与拒答（防幻觉核心）
- **"相关" ≠ "支持结论"**：图谱返回"张三**参与**项目A"绝不推导成"**负责**项目A"
- `relation` 不匹配**只判 insufficient，绝不判 negative**（"没证据证明负责" ≠ "证明不负责"）
- 证据不足 → **定向图谱补召回**（entity + relation_type 精确查询）→ 仍不足 → **拒答**
- 冲突证据（图谱"负责" vs 文本"已不再负责"）→ **确定性裁决**（时间信号→来源→置信度），裁决不了就拒答

### 4. 工程化落地（v2.3）

| 能力 | 实现 | 说明 |
|---|---|---|
| **服务化** | FastAPI | `/ask`、`/ask/stream`、`/health`、`/metrics`、`/docs`；Pydantic 校验、超时控制、优雅关闭 |
| **健康检查** | 依赖**深度探测** | 真实执行一次查询验证可用——driver 创建成功 ≠ 认证可用（懒连接陷阱） |
| **降级链** | 重试退避 → 切备用链路 → 兜底文案 | 任何一层挂掉用户都能拿到可读响应 |
| **语义缓存** | L1 精确 + L2 语义（余弦 ≥0.92） | 支持**进程内 / Redis 分布式**切换，Redis 故障自动回退；带 `knowledge_version` 失效 |
| **可观测性** | Prometheus 指标 + Span 链路追踪 | HTTP 层 + **业务语义层**（证据状态分布、拒答率、缓存命中率） |
| **LLM 高可用** | 多 Provider 有序热备 | 主供应商故障自动切换；全部失败降级为模板生成（不返回 500） |
| **流式输出** | SSE（meta/token/refusal/done） | **证据验证前置**：避免"流到一半才发现证据不足" |

---

## 性能实测（v2.3）

**压测环境**：locust，10 并发 × 30 秒，单 uvicorn worker，8GB 内存机器。

| 场景 | 吞吐 | P50 | P90 | P95 | P99 | 失败率 |
|---|---|---|---|---|---|---|
| 混合负载（缓存 + 完整检索） | **27 RPS** | 22 ms | 63 ms | **82 ms** | 210 ms | **0%** |
| 完整检索链路 | — | 23 ms | 69 ms | 88 ms | 240 ms | 0% |

**缓存效果**：同一问题首次 **308 ms** → 命中缓存 **9.7 ms**（约 30 倍）；Redis 版在**服务重启后仍命中**。

**启动时间优化**：45 s（联网加载模型 + 组件重复实例化）→ **22 s**（HF 离线模式 + 组件复用）。

### 一轮真实的问题定位与修复（同时带来性能收益）

用自研 **Span 链路追踪**定位到一个**导致核心功能失效**的 bug，修复前后对比：

| 指标 | 修复前 | 修复后 | 变化 |
|---|---|---|---|
| 图谱检索结果数 | **0 条**（异常被吞） | 3-6 条（含两跳） | 功能恢复 |
| 吞吐 | 20.7 RPS | **27.7 RPS** | **+34%** |
| P50 | 36 ms | **17 ms** | −53% |
| P90 | 460 ms | **49 ms** | −89% |
| P95 | 640 ms | **64 ms** | **−90%** |
| P99 | 910 ms | **110 ms** | −88% |

**根因**：`Session.run(cypher, query=query)` 与 neo4j driver 形参名冲突 → 每次调用抛 `TypeError` 并被 `except` 吞掉 → 图谱检索恒为空 → 证据永远 insufficient → **每个请求都在走"证据不足→定向补召回"的慢路径**。修一个正确性 bug 同时解决了性能问题。

---

## 迭代记录

| 版本 | 变更 |
|---|---|
| v1.0 | 关键词 Router + if-else 编排，双路检索 + RRF |
| v2.0 | LangGraph StateGraph 重构；语义缓存、Ragas 评估、链路追踪 |
| v2.1 | 修复 7 处：缓存写入时机、RRF key 坍缩、图谱检索 LIMIT 截断、补多跳扩展、LLM 兜底路由、max_steps 生效、local 失败切 global |
| v2.2 | 证据验证与拒答、定向图谱补召回、冲突裁决、缓存参数配置化、Evaluation 闭环 |
| **v2.3** | **服务化（FastAPI）· SSE 流式 · Prometheus 语义化指标 · Redis 分布式缓存 · 多 Provider 热备 · 压测与性能优化** |

### v2.3 修复清单

| 类型 | 问题 | 修复 |
|---|---|---|
| 🔴 致命 | `Session.run(..., query=query)` 参数名冲突 → **图谱检索 100% 抛异常**（核心功能从未工作） | Cypher 参数改 `$q` |
| 🔴 致命 | 无目标关系时只看"有没有检索到内容" → **无关问题答非所问** | 补证据相关性校验 |
| 🟠 严重 | 实体抽取全局删除"项目" → 抽出"C通信协议"这类错误实体 → 误拒答 | 改为**只在词尾**去宾语中心词 + 长度限制 |
| 🟠 严重 | `AgentOrchestratorV2` 重复实例化组件 → **缓存命中率统计恒为 0** + 模型重复加载 | 暴露图内组件供复用 |
| 🟠 工程 | 无 `.env` / `.env.example` → 部署必踩配置坑 | 补配置模板 |
| 🟠 工程 | Milvus collection 未加载时 `search` **静默返回空** | 启动时附加并 load |
| 🟠 工程 | 健康检查只看连接对象 → **误报 ready** | 改为依赖深度探测 |
| 🟡 优化 | HF Hub 每次联网检查 + 组件重复加载 → 启动慢 | 离线模式 + 组件复用：**45 s → 22 s** |

---

## 评测

### Ragas 框架
四项指标（Faithfulness / Answer Correctness / Context Precision / Context Recall），自建覆盖单跳/多跳/对比查询的测试集，相对 Vector-RAG 基线：
- **Answer Correctness 提升 38%**
- **Context Precision 提升 28%**

### Evaluation 闭环
`tests/evaluation/` 走真实生产链路（Router→Retrieval→RRF→Rerank→Validator→Resolver→Answer/Refuse），13 条场景数据集覆盖正常事实/多跳/冲突裁决/证据不足拒答/知识库外/语义陷阱等，量化 **Retrieval Recall / Evidence Validation Accuracy / Refusal Accuracy / Answer Accuracy**，任何模块改动后可回归验证。

### 测试与验证脚本

**合计 66 项自动化测试与验证**（均可本地复现）：

| 类别 | 数量 | 命令 |
|---|---|---|
| pytest 单元测试 | **16**（证据验证 8 / 冲突裁决 6 / 抽取 2） | `python -m pytest tests/ -v` |
| 核心逻辑回归 | 14 | `python tests/sanity_check_fixes.py` |
| 降级路径验证 | 8 | `python tests/sanity_check_fallback.py` |
| 多 Provider 热备 | 11 | `python tests/sanity_check_providers.py` |
| 服务接口自测 | 4 场景 | `python tests/api_check.py`（需服务已启动） |
| Evaluation 闭环 | 13 条场景 | `python tests/evaluation/run_evaluation.py` |

```bash
python -m pytest tests/ -q                  # 单元测试（无外部依赖，缺 API key 的用例自动 skip）
python tests/sanity_check_fixes.py          # 核心逻辑回归
python tests/sanity_check_fallback.py       # 降级路径验证（local 失败切 global）
python tests/sanity_check_providers.py      # 多 Provider 热备（故障切换 / 全挂降级 / 密钥脱敏）
python tests/api_check.py                   # 服务接口四场景（需服务已启动）
python tests/evaluation/run_evaluation.py   # Evaluation 闭环
python tests/manual_agent_demo.py           # 手工端到端演示（需 Neo4j/Milvus）
```

---

## 快速开始

### 1. 启动基础设施（Neo4j + Milvus + etcd + MinIO + Redis）
```bash
docker compose up -d
docker compose ps
```

### 2. 配置环境变量
```bash
cp .env.example .env
# 编辑 .env：填入 DEEPSEEK_API_KEY
# 注意：NEO4J_PASSWORD 必须与 docker-compose.yml 的 NEO4J_AUTH 一致
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 4. 灌入演示数据（可选，用于快速验证链路）
```bash
python tests/seed_demo_data.py
```

### 5. 启动服务
```bash
python -m uvicorn api:app --host 127.0.0.1 --port 8000
# 接口文档：http://127.0.0.1:8000/docs
# 指标：   http://127.0.0.1:8000/metrics
```

### 6. 调用与压测
```bash
# 问答
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "智能客服系统用了哪些技术？"}'

# 接口自测
python tests/api_check.py

# 压测（10 并发 30 秒）
python -m locust -f tests/locustfile.py --host http://127.0.0.1:8000 \
       --headless -u 10 -r 2 -t 30s --only-summary
```

---

## 目录结构

```
graphrag_system/
├── api.py                      # FastAPI 服务层（SSE 流式 / 指标 / 健康检查）
├── docker-compose.yml          # Neo4j + Milvus + etcd + MinIO + Redis
├── .env.example                # 环境变量模板
├── requirements.txt
├── src/
│   ├── agents/
│   │   ├── orchestrator_v2.py       # LangGraph 编排器（8 节点状态机）
│   │   ├── evidence_validator.py    # 证据验证器（含相关性与实体抽取）
│   │   └── conflict_resolver.py     # 冲突裁决器（时间→来源→置信度）
│   ├── database/
│   │   ├── neo4j_client.py          # 图库客户端
│   │   └── milvus_client.py         # 向量库客户端
│   ├── extractors/
│   │   └── triple_extractor.py      # LLM 三元组抽取（白名单 + JSON Schema）
│   ├── retriever/
│   │   └── hybrid_retriever.py      # 双路检索 + 两跳扩展 + RRF 融合
│   ├── reranker/
│   │   └── reranker.py              # 轻量精排（MiniLM bi-encoder）
│   ├── schemas/
│   │   └── graph_schema.py          # Pydantic 图谱模型
│   ├── utils/
│   │   ├── semantic_cache.py        # 进程内语义缓存（L1 精确 + L2 语义）
│   │   ├── redis_cache.py           # Redis 分布式语义缓存（自动降级）
│   │   ├── metrics.py               # Prometheus 指标定义
│   │   ├── llm_provider.py          # 多 Provider 热备
│   │   ├── tracing.py               # Span 链路追踪
│   │   └── retry_decorator.py       # 指数退避重试
│   └── config.py                    # 全局配置 + fail-fast 校验
├── tests/
│   ├── api_check.py                 # 服务接口自测
│   ├── seed_demo_data.py            # 演示数据种子
│   ├── locustfile.py                # 压测脚本
│   ├── sanity_check_fixes.py        # 核心逻辑回归
│   ├── sanity_check_fallback.py     # 降级路径验证
│   ├── sanity_check_providers.py    # Provider 热备验证
│   ├── test_evidence_validator.py / test_conflict_resolver.py / ...
│   └── evaluation/                  # Evaluation 闭环（场景数据集 + Runner）
└── data/eval_queries.json
```

## 技术栈

Python · FastAPI · LangGraph · DeepSeek API · Neo4j · Milvus · Redis · SentenceTransformer · RRF · Prometheus · OpenTelemetry · Pydantic
