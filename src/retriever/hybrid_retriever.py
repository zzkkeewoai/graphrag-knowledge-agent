import logging
import os
import re
from typing import List, Dict, Any
from src.database.neo4j_client import Neo4jClient
from src.database.milvus_client import MilvusClient
from src.retriever.bm25_index import BM25Index

logger = logging.getLogger(__name__)


class HybridRetriever:
    """混合检索器：图谱 + 向量 + BM25 三路召回，RRF 融合。

    为什么加 BM25：向量擅长语义相近，但对精确术语不敏感——"第十五条""净资产 50%"
    这种字面唯一的片段，倒排索引往往比 embedding 更准。企业制度/年报里全是这类表述。

    可用 USE_BM25=false 关闭这一路，用于做 A/B 对照。
    """

    def __init__(self, neo4j_client: Neo4jClient, milvus_client: MilvusClient):
        self.neo4j = neo4j_client
        self.milvus = milvus_client
        self._bm25: BM25Index | None = None
        self._entity_domains = None      # 实体→域 的懒加载映射（图谱层过滤用）

    def _entity_domain_map(self):
        """懒构建"实体→域"映射：用各域块文本反推实体归属。

        图里没有 source_chunk，没法直接回填；重灌要 400+ 次抽取调用。
        块文本里有原文，反推零成本——代价是近似（见 entity_domain.py 的说明）。
        """
        if self._entity_domains is None:
            from src.retriever.domain import domain_of
            from src.retriever.entity_domain import EntityDomainMap

            by_domain: dict[str, list[str]] = {}
            titles_by_domain: dict[str, list[str]] = {}
            for text, meta in zip(self.milvus.texts, self.milvus.metas):
                doc = domain_of(meta["doc_id"])
                by_domain.setdefault(doc, []).append(text)
                # doc_id 去掉序号前缀就是文档标题（"1_对外担保管理制度" → "对外担保管理制度"）
                title = re.sub(r"^\d+[_\-\s]+", "", meta["doc_id"])
                if len(title) >= 4:
                    titles_by_domain.setdefault(doc, [])
                    if title not in titles_by_domain[doc]:
                        titles_by_domain[doc].append(title)

            names = []
            try:
                with self.neo4j.driver.session() as s:
                    names = [r["n"] for r in s.run("MATCH (n:Entity) RETURN n.name AS n")]
            except Exception as e:  # noqa: BLE001
                logger.warning(f"取实体名失败，图谱层不做域过滤: {e}")

            self._entity_domains = EntityDomainMap().build(names, by_domain, titles_by_domain)
        return self._entity_domains

    def _bm25_retrieve(self, query: str, top_k: int) -> List[Dict]:
        """BM25 一路。懒构建索引：第一次查询时才建，避免启动变慢。"""
        if os.getenv("USE_BM25", "true").lower() != "true":
            return []
        try:
            if self._bm25 is None:
                self._bm25 = BM25Index.from_vector_store(self.milvus)
            return self._bm25.search(query, top_k)
        except Exception as e:  # noqa: BLE001
            # BM25 挂了不能影响主链路
            logger.warning(f"BM25 检索失败，本次跳过该路: {e}")
            return []

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
        # 注意：原来这里写死 [:5]，Config.FUSION_TOP_K（=8）形同虚设。
        # 加了第三路（BM25）之后这个问题被放大：三路各 8 条竞争 5 个位置，
        # 融合窗口越小，越容易把原文挤掉。
        from src.config import Config

        limit = max(5, getattr(Config, "FUSION_TOP_K", 8))
        fused_results = []
        for key, score in sorted_items[:limit]:
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

    def retrieve(self, query: str, top_k: int = 5, hops: int = 2,
                 domains: List[str] | None = None) -> Dict[str, Any]:
        """混合检索主流程。

        Args:
            query: 用户查询
            top_k: 向量检索返回条数
            hops: 图谱多跳扩展深度（默认 2，支持依赖链/跨实体对比查询）
            domains: 只在这些域里检索（如 ["pdf"] / ["simco","project"]）；
                     为 None 时不过滤——多域混检，噪声会互相挤占
        """
        from src.retriever.domain import keep

        logger.info(f"混合检索: '{query}' (hops={hops}, domains={domains})")

        # 有域过滤时多取一些候选，过滤后再截断，避免"取 5 条过滤完只剩 1 条"
        fetch_k = top_k * 4 if domains else top_k

        # 1. 图谱检索（从Neo4j获取相关实体 + 多跳邻居）
        graph_results = self._graph_retrieve(query, hops=hops)
        if domains and graph_results:
            # 图谱层也要过滤，否则向量层隔开了、图谱层还是通的（实测串味的根因）。
            # 用三元组级判断：主语常是"公司"这种通用词，真正带域信息的是宾语。
            emap = self._entity_domain_map()
            before = len(graph_results)
            graph_results = [
                r for r in graph_results
                if emap.triple_keep(r.get("entity", ""), r.get("target", ""), domains)
            ]
            logger.info(f"图谱检索: {before} → {len(graph_results)} 个结果（域过滤后）")
        else:
            logger.info(f"图谱检索: {len(graph_results)} 个结果 (hops={hops})")

        # 2. 向量检索（从Milvus获取相关文档块）
        vector_results = self._vector_retrieve(query, fetch_k)
        if domains:
            vector_results = [r for r in vector_results if keep(r.get("doc_id", ""), domains)][:top_k]
        logger.info(f"向量检索: {len(vector_results)} 个结果（域过滤后）")

        # 2.5 BM25 关键词检索（精确术语靠它，向量对"第十五条""50%"这类不敏感）
        bm25_results = self._bm25_retrieve(query, fetch_k)
        if domains:
            bm25_results = [r for r in bm25_results if keep(r.get("doc_id", ""), domains)][:top_k]
        logger.info(f"BM25检索: {len(bm25_results)} 个结果（域过滤后）")

        # 3. RRF融合（三路）
        fused = self.reciprocal_rank_fusion([graph_results, vector_results, bm25_results])
        logger.info(f"RRF融合: {len(fused)} 个结果")

        # 3.5 【融合保底配额】
        # RRF 只看排名不看来源，而图谱一路结果数量通常远多于向量一路，
        # 于是向量结果被整体挤出窗口——表现是"文本里明明有答案，模型却说没有"。
        # 实测两个例子：含"1200人"的块向量排第 4、含"事业部"的排第 10，
        # 都没进最终上下文；前者答案就在原文里。
        #
        # 这里强制保留向量前 2 名：宁可多给两条文档，也不能让原文缺席。
        # 精排（rerank）之后仍然按相关性收敛，不会真的把噪声全塞给模型。
        if vector_results:
            fused_keys = {(f.get("doc_id"), f.get("chunk_index")) for f in fused}
            added = 0
            # 向量召回本身只有 VECTOR_TOP_K 条（目前 5），**全部保留**，不参与淘汰。
            # 原因：图谱一路动辄返回几十上百条（实测 90 条），RRF 只看排名不看来源，
            # 几条向量结果很容易被整体挤出窗口 —— 而向量是唯一带原文的一路。
            # 早先版本只补"未入选的前 2 条"，实测第 5 名的原文块仍然会丢
            # （"A项目使用Python"就排第 5），所以这里放开到全量。
            for v in vector_results:
                key = (v.get("doc_id"), v.get("chunk_index"))
                if key not in fused_keys:
                    fused.append({**v, "fusion_score": 0.0})
                    fused_keys.add(key)
                    added += 1
            if added:
                logger.info(f"[保底] 补入 {added} 条向量结果，防止原文被图谱挤出")

        # 3.6 【BM25 保底】同理：BM25 前 2 名代表的往往是最精确的字面命中，
        # 被图谱的海量结果挤掉就白搭了。
        if bm25_results:
            fused_keys = {(f.get("doc_id"), f.get("chunk_index")) for f in fused}
            added = 0
            for b in bm25_results[:2]:
                key = (b.get("doc_id"), b.get("chunk_index"))
                if key not in fused_keys:
                    fused.append({**b, "fusion_score": 0.0})
                    fused_keys.add(key)
                    added += 1
            if added:
                logger.info(f"[保底] 补入 {added} 条 BM25 结果")

        return {
            "graph_results": graph_results,
            "vector_results": vector_results,
            "bm25_results": bm25_results,
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
                # 【v2.2 修复·致命】参数名绝不能叫 query —— Session.run(query, parameters)
                # 的第一个形参就叫 query，再用 query=... 传关键字参数会直接抛：
                #   TypeError: Session.run() got multiple values for argument 'query'
                # 而异常被下面的 except 吞掉，表现为"图谱检索永远返回 0 条结果"，
                # 进而 known_entities 为空、实体抽取走兜底抽错、证据永远 insufficient。
                node_result = session.run(
                    """
                    MATCH (e:Entity)-[r:RELATES_TO]->(t:Entity)
                    WHERE $q CONTAINS e.name
                       OR $q CONTAINS t.name
                       OR any(kw IN $keywords WHERE e.name CONTAINS kw OR t.name CONTAINS kw)
                    RETURN e.name AS entity, e.type AS etype,
                           r.relation_type AS rel, r.confidence AS conf,
                           t.name AS target, t.type AS ttype
                    LIMIT 2000
                    """,
                    q=query, keywords=list(query_words)
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