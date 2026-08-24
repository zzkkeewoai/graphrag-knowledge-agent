import logging
from typing import List, Dict, Any
from src.database.neo4j_client import Neo4jClient
from src.database.milvus_client import MilvusClient

logger = logging.getLogger(__name__)


class HybridRetriever:
    """混合检索器：结合图谱检索 + 向量检索"""

    def __init__(self, neo4j_client: Neo4jClient, milvus_client: MilvusClient):
        self.neo4j = neo4j_client
        self.milvus = milvus_client

    def reciprocal_rank_fusion(self, results_list: List[List[Dict]], k: int = 60) -> List[Dict]:
        """
        RRF（Reciprocal Rank Fusion）融合算法
        为什么用 RRF 而不是加权求和？
        因为不同模态的分数（向量余弦相似度 vs 图谱邻近度）量纲不同，
        直接加权求和需要人为定权重且随数据漂移；RRF 只依据排名融合，免疫量纲问题
        """
        scores = {}
        for results in results_list:
            for rank, item in enumerate(results, 1):
                # 用 "||" 分隔，避免 doc_id 内含 _ 导致拆分错误（如 doc_001）
                key = f"{item.get('doc_id', '')}||{item.get('chunk_index', rank)}"
                if key not in scores:
                    scores[key] = 0
                scores[key] += 1 / (k + rank)

        # 按分数排序
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # 返回前N个结果
        fused_results = []
        for key, score in sorted_items[:5]:
            # 用 rsplit 从右边拆，保留 doc_id 中的 _
            parts = key.rsplit('||', 1)
            if len(parts) == 2:
                doc_id = parts[0]
                chunk_index = int(parts[1]) if parts[1].isdigit() else 0
                fused_results.append({
                    "doc_id": doc_id,
                    "chunk_index": chunk_index,
                    "fusion_score": score
                })
        return fused_results

    def retrieve(self, query: str, top_k: int = 5, hops: int = 2) -> Dict[str, Any]:
        """
        混合检索主流程

        Args:
            query: 用户查询
            top_k: 向量检索返回条数
            hops: 图谱多跳扩展深度（默认 2，支持依赖链/跨实体对比查询）
        """
        logger.info(f"混合检索: '{query}' (hops={hops})")

        # 1. 图谱检索（从Neo4j获取相关实体 + 多跳邻居）
        graph_results = self._graph_retrieve(query, hops=hops)
        logger.info(f"图谱检索: {len(graph_results)} 个结果 (hops={hops})")

        # 2. 向量检索（从Milvus获取相关文档块）
        vector_results = self._vector_retrieve(query, top_k)
        logger.info(f"向量检索: {len(vector_results)} 个结果")

        # 3. RRF融合
        fused = self.reciprocal_rank_fusion([graph_results, vector_results])
        logger.info(f"RRF融合: {len(fused)} 个结果")

        return {
            "graph_results": graph_results,
            "vector_results": vector_results,
            "fused_results": fused,
            "query": query
        }

    def _graph_retrieve(self, query: str, hops: int = 1) -> List[Dict]:
        """
        图谱检索：从 Neo4j 实时查询匹配的实体及关系，支持多跳邻居扩展。

        【迭代记录 v2.1 —— 原实现的两个真 bug】
        旧实现：MATCH ... LIMIT 100 拉全表 → Python 侧做子串匹配
          ① 召回截断：图谱关系超过 100 条时，后面的实体永远检索不到；
          ② RRF key 坍缩：所有图谱结果 doc_id='graph'、chunk_index=0，
             RRF 融合时全部落在同一个 key "graph||0" 上 → 图谱路退化成
             只有一个融合位，且图谱文本在 orchestrator 的 text_map 中
             永远匹配不到真实文本（fallback 成 "实体: xxx"）。
        新实现：
          ① 过滤下推到 Cypher（$query CONTAINS 实体名 + 关键词 CONTAINS），
             只保留安全上限 LIMIT 2000，不再截断；
          ② 每个图谱结果使用独立 doc_id（graph:实体）+ 递增 chunk_index，
             RRF key 唯一，图谱路不再坍缩；
          ③ hops=2 时对命中实体做两跳邻居扩展（如"项目A -依赖-> X <-使用- 项目B"
             这类跨实体依赖链），回答"多跳推理/对比"类查询。
        """
        results = []
        try:
            # 提取 query 中的关键词（按标点切分，不加滑动窗口，避免 2 字碎片噪声）
            query_words = self._extract_keywords(query, include_windows=False)
            with self.neo4j.driver.session() as session:
                # 过滤在 Cypher 内完成：query 直接包含实体名，或实体名包含任一关键词
                node_result = session.run(
                    """
                    MATCH (e:Entity)-[r:RELATES_TO]->(t:Entity)
                    WHERE $query CONTAINS e.name
                       OR $query CONTAINS t.name
                       OR any(kw IN $keywords WHERE e.name CONTAINS kw OR t.name CONTAINS kw)
                    RETURN e.name AS entity, e.type AS etype,
                           r.relation_type AS rel, r.confidence AS conf,
                           t.name AS target, t.type AS ttype
                    LIMIT 2000
                    """,
                    query=query, keywords=list(query_words)
                )
                rows = list(node_result)
        except Exception as e:
            logger.warning(f"图谱检索异常: {e}")
            rows = []

        # 第一跳：直接命中的三元组（每行独立 doc_id + 行号，保证 RRF key 唯一）
        seen = set()
        for idx, row in enumerate(rows):
            entity, target = row['entity'], row['target']
            rel = row['rel']
            dedup_key = f"{entity}|{rel}|{target}"
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            results.append({
                "doc_id": f"graph:{entity}",
                "chunk_index": idx,
                "text": f"{entity} {rel} {target}",
                "entity": entity,
                "relation": rel,
                "target": target,
                "confidence": row['conf'],
                "score": 0.9,
                "source": "neo4j",
                "hop": 1
            })

        # 第二跳：多跳扩展（依赖链 / 跨实体对比）
        if hops >= 2 and rows:
            seeds = sorted({r['entity'] for r in rows} | {r['target'] for r in rows})
            extra = self._graph_expand_two_hop(seeds, seen)
            results.extend(extra)

        # 兜底：一个实体都没命中，返回全图实体概览
        if not results:
            entity_set = set()
            for idx, row in enumerate(rows):
                key = row['entity']
                if key not in entity_set:
                    entity_set.add(key)
                    results.append({
                        "doc_id": f"graph:{key}",
                        "chunk_index": idx,
                        "text": f"已知实体: {key} (类型: {row['etype']})",
                        "entity": key,
                        "relation": "",
                        "target": "",
                        "confidence": 1.0,
                        "score": 0.5,
                        "source": "neo4j",
                        "hop": 1
                    })

        return results

    def _graph_expand_two_hop(self, seed_names: List[str], seen: set) -> List[Dict]:
        """
        两跳邻居扩展：seed 实体经过任意中间实体到达的邻居。
        对应 Cypher 多跳查询：
            MATCH (e)-[r1]->(m)-[r2]->(n) WHERE e.name IN $seeds ...
        """
        extra = []
        try:
            with self.neo4j.driver.session() as session:
                result = session.run(
                    """
                    MATCH (e:Entity)-[r1:RELATES_TO]->(m:Entity)-[r2:RELATES_TO]->(n:Entity)
                    WHERE e.name IN $seeds
                      AND n.name <> e.name
                      AND NOT EXISTS((e)-[:RELATES_TO]->(n))
                    RETURN e.name AS entity, r1.relation_type AS rel1, m.name AS mid,
                           r2.relation_type AS rel2, n.name AS target,
                           r1.confidence AS conf1, r2.confidence AS conf2
                    LIMIT 500
                    """,
                    seeds=seed_names
                )
                for idx, row in enumerate(result):
                    entity, target = row['entity'], row['target']
                    dedup_key = f"{entity}|{row['rel2']}|{target}"
                    if dedup_key in seen:  # 与第一跳直接关系去重
                        continue
                    seen.add(dedup_key)
                    extra.append({
                        "doc_id": f"graph2hop:{entity}",
                        "chunk_index": idx,
                        "text": (f"{entity} {row['rel1']} {row['mid']} "
                                 f"{row['rel2']} {row['target']}"),
                        "entity": entity,
                        "relation": f"{row['rel1']}-{row['rel2']}",
                        "target": target,
                        "confidence": min(float(row['conf1']), float(row['conf2'])),
                        "score": 0.7,
                        "source": "neo4j",
                        "hop": 2
                    })
        except Exception as e:
            logger.warning(f"两跳扩展异常: {e}")
        return extra

    @staticmethod
    def _extract_keywords(text: str, include_windows: bool = True) -> set:
        """从文本中提取可能的关键词（轻量版，无外部依赖）

        include_windows=True: 额外生成 2-4 字滑动窗口子串（用于 Python 侧粗匹配）
        include_windows=False: 只保留按标点切分的整段词（用于 Cypher CONTAINS，
                               避免 2 字碎片在数据库里造成大量误匹配）
        """
        kw = set()
        # 按标点和空格切分
        import re
        parts = re.split(r'[，。！？、；：""''（）\s]+', text)
        for part in parts:
            part = part.strip()
            if len(part) >= 2:
                kw.add(part)
                if include_windows:
                    # 滑动窗口取子串（2-4 字）
                    for win in range(2, min(5, len(part) + 1)):
                        for i in range(len(part) - win + 1):
                            kw.add(part[i:i + win])
            elif len(part) == 1:
                kw.add(part)
        return kw

    def _vector_retrieve(self, query: str, top_k: int) -> List[Dict]:
        """向量检索"""
        results = self.milvus.search(query, top_k=top_k)
        # 统一格式
        formatted = []
        for r in results:
            formatted.append({
                "doc_id": r.get("doc_id", "unknown"),
                "chunk_index": r.get("chunk_index", 0),
                "text": r.get("chunk_text", ""),
                "score": r.get("score", 0),
                "source": "milvus"
            })
        return formatted