"""
基于 LangGraph 的 Multi-Agent 编排器

架构升级（v2.0）：
- 用 LangGraph StateGraph 替代纯 ReAct/关键词路由
- 显式状态机强约束路由轨迹 → 解决路由漂移与死循环问题
- 集成语义缓存、重试、降级、全链路追踪
- 支持 Human-in-the-Loop 扩展点
"""
import logging
import sys
import os
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from typing_extensions import NotRequired

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.database.neo4j_client import Neo4jClient
from src.database.milvus_client import MilvusClient
from src.config import Config
from src.retriever.hybrid_retriever import HybridRetriever
from src.reranker.reranker import BGEReranker
from src.utils.semantic_cache import SemanticCache
from src.utils.tracing import tracer
from src.agents.evidence_validator import EvidenceValidator, extract_target_relation, extract_target_entity
from src.agents.conflict_resolver import ConflictResolver

logger = logging.getLogger(__name__)


# ============================================================
# LangGraph State 定义
# ============================================================

class AgentState(TypedDict):
    """Agent 全局状态，在 LangGraph 节点间流转"""
    # 输入
    query: str
    history: NotRequired[List[Dict[str, str]]]

    # 路由
    route: str                                    # "local" | "global"

    # 检索结果
    graph_results: NotRequired[List[Dict]]
    vector_results: NotRequired[List[Dict]]
    fused_results: NotRequired[List[Dict]]
    documents: NotRequired[List[Dict]]            # 精排后的 Top-K 文档

    # 生成结果
    answer: NotRequired[str]

    # 缓存命中标记：命中后 answer_node 跳过 LLM，直接复用缓存答案
    from_cache: NotRequired[bool]

    # 降级标记：local 检索全部失败 → 条件边切换到 global 分析（v2.1 新增）
    fallback_to_global: NotRequired[bool]

    # 证据验证（v2.2 新增）
    # 用户问题的目标关系类型（如"负责"），由 EvidenceValidator 从 query 提取
    target_relation: NotRequired[str]
    # 目标实体（如"张三"），供 targeted graph retrieval 使用
    target_entity: NotRequired[str]
    # 证据状态：sufficient / insufficient / conflict / unknown
    evidence_status: NotRequired[str]
    # EvidenceValidator 决定的下一步：answer / targeted_graph / refuse / conflict
    next_action: NotRequired[str]
    # 是否已做过一次 targeted graph retrieval（避免无限循环）
    targeted_retried: NotRequired[bool]
    # 冲突检测状态：none / detected / resolved / unresolved（v2.2 Conflict Resolver）
    conflict_status: NotRequired[str]
    # 证据明细（供 answer_node 使用 / 日志）
    evidence_detail: NotRequired[List[Dict]]
    # 冲突双方证据（EvidenceValidator 收集，供 ConflictResolver 裁决）
    conflict_evidence: NotRequired[List[Dict]]
    # 冲突裁决结果说明（resolved 时记录裁决理由）
    conflict_resolution: NotRequired[str]

    # 执行控制
    step_count: int                               # 当前步数，防死循环
    max_steps: int                                # 最大步数限制
    error_count: int                              # 错误计数
    error_message: NotRequired[str]

    # 元信息
    latency_ms: NotRequired[float]
    token_estimate: NotRequired[int]


# ============================================================
# 单个 Agent 组件（保持单一职责）
# ============================================================

class RouterAgent:
    """
    路由决策 Agent
    v2.0: 升级为规则 + LLM 混合路由
    - 规则层：关键词快速过滤 80% 的明显 case
    - LLM 层（可选）：处理复杂语义判断 20% 的模糊 case
    """

    def __init__(self, use_llm_fallback: bool = False):
        self.use_llm_fallback = use_llm_fallback
        # 扩展关键词列表
        self.global_keywords = [
            "总结", "对比", "概述", "异同", "分析", "趋势",
            "所有", "整体", "全局", "概况", "汇总", "统计",
            "列出全部", "一共有多少", "有哪些项目"
        ]
        self.local_keywords = [
            "是什么", "谁", "什么时候", "哪些", "如何",
            "为什么", "多少", "哪个", "什么关系", "用什么",
            "依赖于", "属于", "领导了", "负责"
        ]

    def route(self, query: str) -> str:
        """路由决策：规则优先 + LLM 兜底"""
        # Step 1: 规则路由（覆盖 80% 场景）
        for kw in self.global_keywords:
            if kw in query:
                logger.info(f"[Router] global 关键词: '{kw}' → global")
                return "global"

        for kw in self.local_keywords:
            if kw in query:
                logger.info(f"[Router] local 关键词: '{kw}' → local")
                return "local"

        # Step 2: LLM 兜底（可选，处理模糊查询）
        if self.use_llm_fallback:
            return self._llm_route(query)

        # 默认 local（更安全）
        logger.info(f"[Router] 无明确关键词，默认 → local")
        return "local"

    def _llm_route(self, query: str) -> str:
        """LLM 兜底路由：处理关键词无法判定的模糊查询（生产环境启用）

        设计说明：
        - 规则层覆盖约 80% 明显 case（关键词命中即返回，零成本）
        - LLM 层只处理剩余的模糊 case，控制调用成本与延迟
        - temperature=0 + max_tokens=10：只允许输出 local/global 一个词，
          从输出侧约束路由的确定性（比解析自由文本更稳）
        - 任何异常（网络、解析、缺 key）都回退 local，路由绝不抛错
        """
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
            )
            prompt = (
                "你是查询路由分类器。判断用户查询适合哪种检索策略：\n"
                "- 如果查询要求全局概览、对比、汇总、趋势、统计、列出全部 → 输出 global\n"
                "- 如果查询针对具体实体或事实（是谁、是什么、用什么、依赖谁）→ 输出 local\n"
                "只输出一个词：local 或 global。\n\n"
                f"查询：{query}"
            )
            resp = client.chat.completions.create(
                model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10
            )
            route = resp.choices[0].message.content.strip().lower()
            if route in ("local", "global"):
                logger.info(f"[Router] LLM 兜底路由: {query[:30]}... → {route}")
                return route
            logger.warning(f"[Router] LLM 输出非法路由值: {route!r}，回退 local")
        except Exception as e:
            logger.warning(f"[Router] LLM 路由失败，回退 local: {e}")
        return "local"


class LocalRetrievalAgent:
    """
    本地检索 Agent：混合检索 + Reranker 精排
    """

    def __init__(self, retriever: HybridRetriever, reranker: BGEReranker):
        self.retriever = retriever
        self.reranker = reranker

    def execute(self, query: str) -> Dict[str, Any]:
        """执行本地检索管线"""
        logger.info(f"[LocalAgent] 检索: {query[:50]}...")

        # Step 1: 混合检索
        results = self.retriever.retrieve(query)

        # Step 2: text_map（融合结果 → 原文）
        text_map = {}
        for r in results.get('graph_results', []):
            key = f"{r.get('doc_id', '')}||{r.get('chunk_index', 0)}"
            text_map[key] = r.get('text', '')
        for r in results.get('vector_results', []):
            key = f"{r.get('doc_id', '')}||{r.get('chunk_index', 0)}"
            text_map[key] = r.get('text', '')[:200]

        # Step 3: 用真实文本填充 fused_results
        documents = []
        for item in results.get('fused_results', []):
            key = f"{item.get('doc_id', '')}||{item.get('chunk_index', 0)}"
            real_text = text_map.get(key, f"实体: {item.get('doc_id', '未知')}")
            documents.append({
                'text': real_text,
                'doc_id': item.get('doc_id', ''),
                'chunk_index': item.get('chunk_index', 0),
                'fusion_score': item.get('fusion_score', 0)
            })

        # Step 4: Reranker 精排 → Top-3
        if documents:
            reranked = self.reranker.rerank(query, documents, top_k=3)
        else:
            reranked = []

        return {
            "type": "local",
            "query": query,
            "documents": reranked,
            # v2.2 Evaluation 修复：返回结构化图谱证据供 EvidenceValidator 使用
            # （否则证据验证只能靠 documents 文本，relation 结构化匹配会退化）
            "graph_results": results.get('graph_results', []),
            "vector_results": results.get('vector_results', []),
            "graph_count": len(results.get('graph_results', [])),
            "vector_count": len(results.get('vector_results', [])),
        }


class GlobalAnalysisAgent:
    """
    全局分析 Agent：扫描全图谱，生成统计摘要
    """

    def __init__(self, neo4j_client: Neo4jClient):
        self.neo4j = neo4j_client

    def execute(self, query: str) -> Dict[str, Any]:
        """执行全局图谱分析"""
        logger.info(f"[GlobalAgent] 全局分析: {query[:50]}...")

        relationships = []
        try:
            with self.neo4j.driver.session() as session:
                result = session.run("""
                    MATCH (p:Entity {type: '项目'})-[r:RELATES_TO]->(t:Entity)
                    WHERE r.relation_type IN ['使用', '依赖']
                    RETURN p.name AS project,
                           r.relation_type AS relation,
                           t.name AS target,
                           r.confidence AS confidence
                    ORDER BY p.name
                """)
                relationships = [record.data() for record in result]
        except Exception as e:
            logger.warning(f"图谱查询异常: {e}")

        # 构建文本上下文
        if not relationships:
            context = "图谱中暂无项目关系数据。"
        else:
            context = "📊 全局项目关系概览:\n"
            for rel in relationships:
                context += (
                    f"  • {rel['project']} {rel['relation']} "
                    f"{rel['target']} (置信度: {rel['confidence']:.2f})\n"
                )

        # 统计摘要
        projects = set()
        techs = set()
        deps = []
        for rel in relationships:
            projects.add(rel['project'])
            if rel['relation'] == '使用':
                techs.add(rel['target'])
            elif rel['relation'] == '依赖':
                deps.append(f"{rel['project']} → {rel['target']}")

        summary = f"""
📈 统计摘要:
  • 项目数: {len(projects)}
  • 技术栈数: {len(techs)}
  • 依赖关系数: {len(deps)}
  • 依赖链: {', '.join(deps) if deps else '无'}
        """

        return {
            "type": "global",
            "query": query,
            "context": context,
            "summary": summary,
            "relationships": relationships,
        }


class AnswerAgent:
    """
    答案生成 Agent：模板（开发） / LLM（生产）
    """

    def __init__(self, use_llm: bool = False):
        self.use_llm = use_llm
        self._client = None
        if use_llm:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
            )

    def generate(self, query: str, agent_result: Dict[str, Any]) -> str:
        if self.use_llm and self._client:
            return self._generate_with_llm(query, agent_result)
        return self._generate_with_template(query, agent_result)

    def _generate_with_template(self, query: str, result: Dict[str, Any]) -> str:
        """模板生成（零成本快速验证）"""
        if result["type"] == "local":
            docs = result.get("documents", [])
            if not docs:
                return "未找到相关信息。请尝试更具体的问题。"
            answer = f"🔍 找到 {len(docs)} 条相关信息：\n"
            for i, doc in enumerate(docs, 1):
                text = doc.get('text', '')[:100]
                score = doc.get('rerank_score', 0)
                answer += f"  {i}. {text}... (相关性: {score:.3f})\n"
            return answer
        else:
            context = result.get("context", "")
            summary = result.get("summary", "")
            return f"{context}\n{summary}"

    def _generate_with_llm(self, query: str, result: Dict[str, Any]) -> str:
        """LLM 生成（高质量）"""
        if result["type"] == "local":
            docs = result.get("documents", [])
            context = "\n".join([f"- {d.get('text', '')}" for d in docs[:5]])
            prompt = (
                f"仅根据以下信息回答用户问题。如果信息不足，请明确说明。\n\n"
                f"用户问题: {query}\n\n"
                f"相关信息:\n{context}"
            )
        else:
            context = result.get("context", "")
            prompt = (
                f"根据以下全局分析总结回答用户问题。\n\n"
                f"用户问题: {query}\n\n"
                f"全局信息:\n{context}"
            )

        response = self._client.chat.completions.create(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )
        return response.choices[0].message.content


# ============================================================
# 降级兜底机制
# ============================================================

FALLBACK_ANSWERS = {
    "no_results": "抱歉，在当前知识库中未找到与您问题相关的信息。请尝试：\n1. 使用更具体的关键词\n2. 换一种表述方式\n3. 确认相关文档已导入系统",
    "timeout": "抱歉，查询处理超时。系统正在优化中，请稍后重试。",
    "error": "抱歉，处理您的请求时遇到了技术问题。我们的工程师已收到通知，请稍后重试。",
    "empty_graph": "当前知识图谱尚未建立。请先导入文档以构建知识库。"
}


# ============================================================
# LangGraph 编排器（核心升级）
# ============================================================

def create_orchestrator_graph(
    neo4j_client: Neo4jClient,
    milvus_client: MilvusClient,
    use_llm: bool = False,
    use_cache: bool = True,
    max_retries: int = 3,
    max_steps: int = 15
):
    """
    构建 LangGraph StateGraph

    节点:
      router_node → [conditional] → local_node or global_node → answer_node

    状态机强约束：每个状态只能转移到预定义的下一个状态
    → 解决 ReAct 的路径漂移和死循环问题
    """
    # 初始化组件
    # 修复 v2.1：LLM 兜底路由改为配置驱动（USE_LLM_ROUTER），
    # 不再硬编码 use_llm_fallback=False
    router = RouterAgent(use_llm_fallback=Config.USE_LLM_ROUTER)
    retriever = HybridRetriever(neo4j_client, milvus_client)
    reranker = BGEReranker()
    local_agent = LocalRetrievalAgent(retriever, reranker)
    global_agent = GlobalAnalysisAgent(neo4j_client)
    answer_agent = AnswerAgent(use_llm=use_llm)

    # 语义缓存（v2.2: threshold/max_size/knowledge_version 配置化）
    cache = SemanticCache(
        encoder=reranker.model,
        similarity_threshold=Config.CACHE_SIMILARITY_THRESHOLD,
        ttl=Config.CACHE_TTL,
        max_size=Config.CACHE_MAX_SIZE,
        knowledge_version=Config.KNOWLEDGE_VERSION,
    ) if use_cache else None

    # 证据验证器（v2.2 新增，确定性规则，不调 LLM）
    evidence_validator = EvidenceValidator()
    # 冲突解决器（v2.2 新增，确定性规则：时间→来源→置信度）
    conflict_resolver = ConflictResolver()

    # ============================================================
    # 节点函数
    # ============================================================

    def router_node(state: AgentState) -> AgentState:
        """路由节点"""
        with tracer.start_span("router", query=state["query"][:50]) as span:
            route = router.route(state["query"])
            span.set_attribute("route", route)
            state["route"] = route
            state["step_count"] += 1
        return state

    def local_node(state: AgentState) -> AgentState:
        """本地检索节点"""
        query = state["query"]

        # 检查语义缓存
        if cache:
            cached = cache.get(query)
            if cached:
                logger.info("[Local] 语义缓存命中")
                # 修复 v2.1：命中缓存必须标记 from_cache，
                # answer_node 据此跳过 LLM 直接复用缓存答案
                state["from_cache"] = True
                state["answer"] = cached.get("answer", "")
                state["documents"] = cached.get("documents", [])
                state["latency_ms"] = 0.0
                return state

        # 执行检索 + 重排序（带重试）
        last_error = None
        for attempt in range(max_retries):
            try:
                with tracer.start_span("local_retrieval", attempt=attempt + 1) as span:
                    result = local_agent.execute(query)
                    span.set_attribute("document_count", len(result.get("documents", [])))

                    state["documents"] = result.get("documents", [])
                    state["graph_results"] = result.get("graph_results", [])
                    state["vector_results"] = result.get("vector_results", [])
                    state["fused_results"] = result.get("fused_results", [])
                    state["error_count"] = 0  # 重置错误计数

                    # 修复 v2.1：不再在这里写缓存！
                    # 旧实现把 state["answer"]（此时还是空字符串）写进缓存，
                    # 导致命中缓存的请求拿到空 answer，且 answer_node 仍会
                    # 重新调用 LLM —— 缓存既省不了钱还返回错误结果。
                    # 缓存写入统一挪到 answer_node（生成最终答案之后）。

                    return state
            except Exception as e:
                last_error = e
                logger.warning(f"[Local] 第 {attempt + 1} 次重试: {e}")
                import time
                time.sleep(1.0 * (2 ** attempt))  # 指数退避

        # 所有重试失败 → 降级切换：local 挂了切 global（对应简历"降级兜底机制"）
        # 修复 v2.1：旧实现直接给 FALLBACK_ANSWERS["error"] 就结束了，
        # "local挂了切global"只是文档里的说法，代码里没有这个行为。
        # 现在：标记 fallback_to_global → 条件边把流程切到 global_node，
        # 用全图分析兜底回答（global 不需要向量库，图谱挂了才真的失败）。
        logger.error(f"[Local] {max_retries} 次重试全部失败，降级切换 global: {last_error}")
        state["error_count"] += 1
        state["error_message"] = str(last_error)
        state["documents"] = []
        state["answer"] = ""                      # 清空，等 global 填
        state["route"] = "global"                 # 关键：answer_node 按 route 选择上下文
        state["fallback_to_global"] = True
        return state

    def evidence_validator_node(state: AgentState) -> AgentState:
        """证据验证节点（v2.2 新增）

        判断检索到的证据是否足以支持用户问题：
          - sufficient → answer（进 AnswerAgent）
          - insufficient（首次）→ targeted_graph（定向图谱补充召回）
          - insufficient（已重试）→ refuse（拒答，不交给 LLM 自由发挥）
          - conflict → conflict（进入冲突处理分支，v2.2 预留）
        """
        query = state["query"]
        known_entities = [r.get("entity", "") for r in (state.get("graph_results") or [])]

        with tracer.start_span("evidence_validation", query=query[:50]) as span:
            result = evidence_validator.validate(
                query=query,
                graph_results=state.get("graph_results", []),
                documents=state.get("documents", []),
                targeted_retried=bool(state.get("targeted_retried")),
                known_entities=known_entities,
            )
            span.set_attribute("evidence_status", result["evidence_status"])
            span.set_attribute("next_action", result["next_action"])

            state["evidence_status"] = result["evidence_status"]
            state["next_action"] = result["next_action"]
            state["conflict_status"] = result.get("conflict_status", "unknown")
            state["target_relation"] = result.get("target_relation")
            state["target_entity"] = result.get("target_entity")
            state["evidence_detail"] = result.get("evidence_detail", [])
            # 修复：conflict 分支必须把正反证据写入 state，供 ConflictResolver 裁决
            state["conflict_evidence"] = result.get("conflict_evidence", [])

        logger.info(f"[Evidence] status={state['evidence_status']} action={state['next_action']}")
        return state

    def targeted_graph_node(state: AgentState) -> AgentState:
        """定向图谱检索节点（v2.2 新增）

        当 Evidence Validator 判定 insufficient 时触发：
        针对 query 中的 entity + 期望 relation_type 做精确结构化查询，
        补充"正向证据"，然后回到 evidence_validator_node 二次判断。
        """
        query = state["query"]
        entity = state.get("target_entity") or extract_target_entity(query, [])
        rel = state.get("target_relation") or extract_target_relation(query)

        with tracer.start_span("targeted_graph_retrieval", query=query[:50]) as span:
            span.set_attribute("entity", entity or "")
            span.set_attribute("relation", rel or "")

            if entity and rel:
                targeted = neo4j_client.query_entity_by_relation(entity, rel)
                span.set_attribute("targeted_count", len(targeted))
            else:
                targeted = []

            # 与已有 graph_results 合并（避免重复），统一格式
            seen = set()
            merged = list(state.get("graph_results") or [])
            for item in state.get("graph_results") or []:
                seen.add((item.get("entity"), item.get("relation"), item.get("target")))

            for t in targeted:
                key = (t.get("entity"), t.get("relation"), t.get("target"))
                if key not in seen:
                    seen.add(key)
                    merged.append({
                        "doc_id": f"graph:{t.get('entity', '')}",
                        "chunk_index": len(merged),
                        "text": f"{t.get('entity')} {t.get('relation')} {t.get('target')}",
                        "entity": t.get("entity", ""),
                        "relation": t.get("relation", ""),
                        "target": t.get("target", ""),
                        "confidence": t.get("confidence", 0.5),
                        "source": "neo4j",
                        "hop": 1,
                    })

            state["graph_results"] = merged
            state["targeted_retried"] = True
            state["step_count"] += 1

        return state

    def refuse_node(state: AgentState) -> AgentState:
        """拒答节点（v2.2 新增）

        证据不足（含 targeted 补充后仍不足）时拒答，
        不让 LLM 基于不充分证据自由发挥/推断。
        """
        entity = state.get("target_entity") or ""
        rel = state.get("target_relation") or ""
        if entity and rel:
            state["answer"] = f"当前知识库中的信息不足以确认{entity}{rel}的结论。"
        else:
            state["answer"] = "当前知识库中的信息不足以回答您的问题。"
        state["evidence_status"] = "insufficient"
        logger.info(f"[Refuse] 证据不足，拒答: {state['answer']}")
        return state

    def conflict_node(state: AgentState) -> AgentState:
        """冲突处理节点（v2.2 Conflict Resolver 接入）

        EvidenceValidator 检测到冲突（支持 vs 反对证据并存）后进入本节点：
          - ConflictResolver 用确定性规则裁决（时间 → 来源 → 置信度）
          - resolved → 设置 winner 答案（模板化，不调 LLM），走 answer
          - unresolved → 拒答，说明冲突存在
        """
        entity = state.get("target_entity") or ""
        rel = state.get("target_relation") or ""
        conflict_evidence = state.get("conflict_evidence") or []
        evidence_detail = state.get("evidence_detail") or []

        with tracer.start_span("conflict_resolution", query=(state.get("query") or "")[:50]) as span:
            result = conflict_resolver.resolve(
                query=state.get("query", ""),
                conflict_evidence=conflict_evidence,
                evidence_detail=evidence_detail,
            )
            span.set_attribute("conflict_status", result["conflict_status"])
            span.set_attribute("next_action", result["next_action"])

            state["conflict_status"] = result["conflict_status"]
            state["next_action"] = result["next_action"]
            state["conflict_resolution"] = result.get("conflict_resolution", "")

        if result["next_action"] == "refuse":
            # unresolved → 拒答（明确说明存在冲突，不编造答案）
            state["answer"] = (
                f"知识库中存在关于{entity}是否{rel}的冲突信息，"
                f"当前无法可靠确认。建议核实最新状态后再确认。"
            )
            logger.info(f"[Conflict] unresolved，拒答: {state['answer']}")
            return state

        # resolved → 基于胜出证据生成确定答案（模板化，不调 LLM）
        winner_side = result.get("winner_side")
        winner_ev = result.get("winner_evidence") or {}
        if winner_side == "oppose":
            # 反对侧胜出（如文本"不再负责"）→ 明确否定式回答
            state["answer"] = (
                f"根据知识库中的最新信息，{entity}不再{rel}"
                f"（{winner_ev.get('text', '')[:60]}）。"
            )
        else:
            # 支持侧胜出 → 正向回答
            target = winner_ev.get("target") or winner_ev.get("text", "")[:30]
            state["answer"] = f"根据知识库信息，{entity}{rel}{target}。"
        state["evidence_status"] = "resolved"
        logger.info(f"[Conflict] resolved（{winner_side}），answer: {state['answer']}")
        return state

    def global_node(state: AgentState) -> AgentState:
        """全局分析节点"""
        query = state["query"]

        # 检查语义缓存
        if cache:
            cached = cache.get(query)
            if cached:
                logger.info("[Global] 语义缓存命中")
                state["from_cache"] = True
                state["answer"] = cached.get("answer", "")
                state["latency_ms"] = 0.0
                return state

        # 执行全局分析（带重试）
        for attempt in range(max_retries):
            try:
                with tracer.start_span("global_analysis", attempt=attempt + 1) as span:
                    result = global_agent.execute(query)
                    span.set_attribute("relationship_count", len(result.get("relationships", [])))

                    # 先写入分析上下文，answer_node 会基于它生成最终答案
                    state["answer"] = (
                        result.get("context", "") + "\n" + result.get("summary", "")
                    )
                    state["error_count"] = 0

                    # 修复 v2.1：不再在这里写缓存（理由同 local_node）
                    return state
            except Exception as e:
                logger.warning(f"[Global] 第 {attempt + 1} 次重试: {e}")
                import time
                time.sleep(1.0 * (2 ** attempt))

        # 降级
        state["error_count"] += 1
        state["answer"] = FALLBACK_ANSWERS["error"]
        return state

    def answer_node(state: AgentState) -> AgentState:
        """答案生成节点"""
        query = state["query"]
        route = state["route"]

        # 修复 v2.1：语义缓存命中 → 跳过 LLM，直接复用缓存里的最终答案
        if state.get("from_cache"):
            logger.info(f"[Answer] 缓存命中，跳过 LLM 生成（route={route}）")
            return state

        # v2.2：证据不足或冲突 → 不调 LLM，直接返回拒答/冲突文案
        # 关键：不让 LLM 基于"参与"之类不充分证据自由推断"负责"
        if state.get("evidence_status") in ("insufficient", "conflict"):
            logger.info(f"[Answer] 证据状态={state.get('evidence_status')}，跳过 LLM，拒答")
            return state

        with tracer.start_span("answer_generation", route=route) as span:
            # 构建 agent_result 供 AnswerAgent 使用
            if route == "local":
                agent_result = {
                    "type": "local",
                    "documents": state.get("documents", []),
                }
            else:
                agent_result = {
                    "type": "global",
                    "context": state.get("answer", ""),
                }

            try:
                answer = answer_agent.generate(query, agent_result)
                state["answer"] = answer
                span.set_attribute("answer_length", len(answer))

                # 修复 v2.1：答案生成完成后才写缓存（local 和 global 共用）
                # 命中后 answer_node 直接返回缓存答案，既省 LLM 调用又保证正确
                if cache:
                    cache.set(query, {
                        "answer": answer,
                        "documents": state.get("documents", []),
                    })
            except Exception as e:
                logger.error(f"[Answer] 生成失败: {e}")
                state["answer"] = FALLBACK_ANSWERS["error"]
                span.set_error(str(e))

        return state

    # ============================================================
    # 条件边：根据 route 决定走 local 还是 global
    # ============================================================

    def route_decision(state: AgentState) -> str:
        # 修复 v2.1：max_steps 真正生效
        # 旧实现 max_steps 形同虚设——图是 DAG（最多 3 步），永远走不到 15；
        # 现在只要超过上限就强制收尾进 answer，作为状态机的最后一道保险
        if state.get("step_count", 0) >= state.get("max_steps", 15):
            logger.warning(f"[Route] 步数达上限 {state.get('max_steps')}，强制收尾")
            return "answer"
        if state.get("error_count", 0) >= 3:
            logger.warning("[Route] 错误次数过多，终止")
            return "answer"  # 有错误也尝试生成答案
        route = state.get("route", "local")
        logger.info(f"[Route] 决策: {route}")
        return route

    # ============================================================
    # 构建 StateGraph
    # ============================================================

    from langgraph.graph import StateGraph, END

    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("router", router_node)
    workflow.add_node("local", local_node)
    workflow.add_node("global", global_node)
    workflow.add_node("evidence_validator", evidence_validator_node)
    workflow.add_node("targeted_graph", targeted_graph_node)
    workflow.add_node("refuse", refuse_node)
    workflow.add_node("conflict", conflict_node)
    workflow.add_node("answer", answer_node)

    # 设置入口
    workflow.set_entry_point("router")

    # 条件边：router → local 或 global
    workflow.add_conditional_edges(
        "router",
        route_decision,
        {
            "local": "local",
            "global": "global",
            "answer": "answer"   # error 情况
        }
    )

    # 修复 v2.1：local → global 降级切换
    # 旧实现 local/global 都是直连 answer，"local挂了切global"不存在。
    # 现在 local 失败（fallback_to_global=True）→ 条件边切 global 兜底；
    # local 成功 → 正常进 answer。
    def local_after(state: AgentState) -> str:
        if state.get("fallback_to_global"):
            logger.warning("[Local] 检索失败，降级切换到 global 分析")
            return "global"
        return "evidence_validator"

    workflow.add_conditional_edges(
        "local",
        local_after,
        {
            "global": "global",          # local 挂了 → global 兜底
            "evidence_validator": "evidence_validator"   # v2.2: local 成功 → 证据验证
        }
    )

    # v2.2：证据验证 → 按 next_action 分流
    #  - answer          → 证据充足，正常生成
    #  - targeted_graph  → 证据不足（首次）→ 定向图谱补充召回
    #  - refuse          → 证据不足（已重试）→ 拒答
    #  - conflict        → 冲突证据 → 冲突处理（预留）
    def evidence_after(state: AgentState) -> str:
        action = state.get("next_action", "answer")
        # 保险：超过 max_steps 不再重查，避免循环
        if action == "targeted_graph" and state.get("step_count", 0) >= state.get("max_steps", 15):
            logger.warning("[Evidence] 步数超限，targeted_graph 转 refuse")
            return "refuse"
        logger.info(f"[Evidence] 决策: {action}")
        return action

    workflow.add_conditional_edges(
        "evidence_validator",
        evidence_after,
        {
            "answer": "answer",
            "targeted_graph": "targeted_graph",
            "refuse": "refuse",
            "conflict": "conflict",
        }
    )

    # v2.2：定向图谱检索后 → 回到证据验证（二次判断）
    # 此时 targeted_retried=True，若仍不足 → refuse
    workflow.add_edge("targeted_graph", "evidence_validator")

    workflow.add_edge("global", "answer")
    # 拒答/冲突直接结束（answer 已在节点内设置）
    workflow.add_edge("refuse", END)
    workflow.add_edge("conflict", END)

    # answer → 结束
    workflow.add_edge("answer", END)

    # 编译图
    app = workflow.compile()
    logger.info("LangGraph Agent 编排器已就绪")
    return app


# ============================================================
# 对外接口（兼容旧版 AgentOrchestrator 接口）
# ============================================================

class AgentOrchestratorV2:
    """
    LangGraph 版 Agent 编排器
    对外接口兼容旧版 AgentOrchestrator.process()
    """

    def __init__(
        self,
        neo4j_client: Neo4jClient,
        milvus_client: MilvusClient,
        use_llm: bool = False,
        use_cache: bool = True,
        max_retries: int = 3,
        max_steps: int = 15
    ):
        self.neo4j = neo4j_client
        self.milvus = milvus_client
        self.use_llm = use_llm
        self.use_cache = use_cache

        # 构建 LangGraph
        self.graph = create_orchestrator_graph(
            neo4j_client=neo4j_client,
            milvus_client=milvus_client,
            use_llm=use_llm,
            use_cache=use_cache,
            max_retries=max_retries,
            max_steps=max_steps
        )

        # 各组件引用（供外部直接使用）
        self.retriever = HybridRetriever(neo4j_client, milvus_client)
        self.reranker = BGEReranker()
        self.cache = SemanticCache(
            encoder=self.reranker.model,
            similarity_threshold=Config.CACHE_SIMILARITY_THRESHOLD,
            ttl=Config.CACHE_TTL,
            max_size=Config.CACHE_MAX_SIZE,
            knowledge_version=Config.KNOWLEDGE_VERSION,
        ) if use_cache else None

    def process(self, query: str) -> Dict[str, Any]:
        """
        处理用户查询（兼容旧接口）

        执行流程（LangGraph 状态机约束）:
          ① router → ② local/global → ③ answer → 返回

        状态机保证：
        - 路由轨迹受强约束，不会出现死循环
        - 每步都有降级兜底
        - 全链路 Trace 记录
        """
        import time
        start = time.time()

        # 初始化状态
        initial_state: AgentState = {
            "query": query,
            "route": "local",
            "step_count": 0,
            "max_steps": 15,
            "error_count": 0,
            "documents": [],
            "answer": "",
        }

        # 执行状态机
        logger.info(f"[OrchestratorV2] 开始处理: {query[:50]}...")
        try:
            with tracer.start_span("orchestrator_pipeline", query=query[:50]) as root_span:
                final_state = self.graph.invoke(initial_state)
                root_span.set_attribute("route", final_state.get("route", "unknown"))
                root_span.set_attribute("steps", final_state.get("step_count", 0))
        except Exception as e:
            logger.error(f"[OrchestratorV2] 执行异常: {e}")
            final_state = {
                "answer": FALLBACK_ANSWERS["error"],
                "route": "local",
                "documents": [],
                "error_message": str(e),
            }

        elapsed_ms = (time.time() - start) * 1000

        # 构建返回结果
        return {
            "query": query,
            "route": final_state.get("route", "local"),
            "answer": final_state.get("answer", FALLBACK_ANSWERS["no_results"]),
            "documents": final_state.get("documents", []),
            "latency_ms": round(elapsed_ms, 1),
            "step_count": final_state.get("step_count", 0),
            "error_message": final_state.get("error_message", None),
            "trace": tracer.get_trace_tree(),
        }

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        if self.cache:
            return self.cache.stats
        return {"enabled": False}

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "use_llm": self.use_llm,
            "use_cache": self.use_cache,
            "cache": self.get_cache_stats(),
        }
