import sys
import os
import traceback

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.neo4j_client import Neo4jClient
from src.database.milvus_client import MilvusClient
from src.retriever.hybrid_retriever import HybridRetriever
from src.reranker.reranker import BGEReranker
from src.config import Config

# 测试文档（与 test_write_to_neo4j.py 保持一致）
TEST_DOCS = [
    (
        "doc_001",
        """
        智能客服系统于2024年3月启动，目标是将AI能力集成到现有的ERP系统中。
        项目负责人是王明，技术选型采用了Java 17、FastAPI框架和Milvus向量数据库。
        智能客服系统依赖B项目提供的用户权限管理模块。
        """
    ),
    (
        "doc_002",
        """
        B项目是一个微服务架构的基础设施项目，采用Go语言和gRPC通信协议。
        该项目由李婷领导，已于2025年1月完成一期交付。
        B项目为C项目提供了核心的认证服务。
        """
    )
]

# 有意义的查询列表
TEST_QUERIES = [
    "智能客服系统用了哪些技术？",
    "B项目用了什么语言和框架？",
    "智能客服系统依赖哪个项目？",
]


def main():
    print("=" * 60)
    print("混合检索 + Reranker 测试")
    print("=" * 60)

    try:
        # ========================================
        # 初始化客户端
        # ========================================
        print("\n[1/5] 初始化客户端...")
        neo4j = Neo4jClient(Config.NEO4J_URI, Config.NEO4J_USER, Config.NEO4J_PASSWORD)
        neo4j.connect()

        milvus = MilvusClient()
        milvus.connect()

        # ========================================
        # 写入 Milvus：文档切片 → 向量化 → 入库
        # ========================================
        print("\n[2/5] 写入文档块到 Milvus...")
        milvus.create_collection("document_chunks")
        total_chunks = 0
        for doc_id, doc_text in TEST_DOCS:
            chunks = milvus.chunk_text(doc_text)
            n = milvus.insert_chunks(doc_id, chunks)
            total_chunks += n
            print(f"  {doc_id}: {n} 个 chunk 已写入")
        print(f"  总计写入 {total_chunks} 个文档块")

        # ========================================
        # 创建检索器 + Reranker
        # ========================================
        print("\n[3/5] 创建检索器 + 加载 Reranker...")
        retriever = HybridRetriever(neo4j, milvus)
        reranker = BGEReranker()
        print("  Reranker 加载完成")

        # ========================================
        # 逐条查询
        # ========================================
        print("\n[4/5] 执行混合检索 + Reranker 重排")

        for query in TEST_QUERIES:
            print(f"\n{'─' * 50}")
            print(f"  查询: '{query}'")

            # ---- 混合检索 ----
            result = retriever.retrieve(query)
            print(f"  图谱结果: {len(result['graph_results'])} 条  |  向量结果: {len(result['vector_results'])} 条  |  融合结果: {len(result['fused_results'])} 条")

            # ---- 构建真实文本映射 ----
            # 从原始结果中按 (doc_id, chunk_index) 建立文本索引（|| 分隔，避免 doc_id 内含 _ 导致拆分错误）
            text_map = {}
            for r in result['graph_results']:
                key = f"{r.get('doc_id', '')}||{r.get('chunk_index', 0)}"
                text_map[key] = r.get('text', '')
            for r in result['vector_results']:
                key = f"{r.get('doc_id', '')}||{r.get('chunk_index', 0)}"
                text_map[key] = r.get('text', '')[:150]

            # ---- 组装待重排文档 ----
            documents = []
            for item in result['fused_results']:
                key = f"{item.get('doc_id', '')}||{item.get('chunk_index', 0)}"
                real_text = text_map.get(key)
                if not real_text:
                    # 回退：从 Neo4j 查实体关联信息
                    real_text = _query_neo4j_context(neo4j, item.get('doc_id', ''))
                documents.append({
                    'doc_id': item.get('doc_id', ''),
                    'chunk_index': item.get('chunk_index', 0),
                    'text': real_text,
                    'fusion_score': item.get('fusion_score', 0),
                    'source': 'graph' if item.get('doc_id') == 'graph' else 'vector'
                })

            # ---- Reranker 重排 ----
            reranked = reranker.rerank(query, documents, top_k=3)
            print(f"  重排结果: {len(reranked)} 条")
            for i, doc in enumerate(reranked, 1):
                text_preview = doc.get('text', '')[:80].replace('\n', ' ')
                print(f"    {i}. [{doc.get('source', '?')}] {text_preview}...")
                print(f"       score={doc.get('rerank_score', 0):.4f}  fusion={doc.get('fusion_score', 0):.4f}")

        # ========================================
        # 完成
        # ========================================
        print(f"\n[5/5] 清理资源...")
        neo4j.close()
        milvus.disconnect()
        print("\n✅ 测试完成!")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        traceback.print_exc()


def _query_neo4j_context(neo4j: Neo4jClient, entity_name: str) -> str:
    """从 Neo4j 查询实体的关联上下文"""
    try:
        with neo4j.driver.session() as session:
            result = session.run(
                """
                MATCH (e:Entity {name: $name})-[r:RELATES_TO]->(t:Entity)
                RETURN e.name AS entity, r.relation_type AS rel, t.name AS target
                UNION
                MATCH (s:Entity)-[r:RELATES_TO]->(e:Entity {name: $name})
                RETURN s.name AS entity, r.relation_type AS rel, e.name AS target
                """,
                name=entity_name
            )
            rows = list(result)
            if rows:
                parts = [f"{row['entity']} {row['rel']} {row['target']}" for row in rows[:3]]
                return "；".join(parts)
            return f"实体: {entity_name}"
    except Exception:
        return f"实体: {entity_name}"


if __name__ == "__main__":
    main()
