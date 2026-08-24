import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.neo4j_client import Neo4jClient
from src.database.milvus_client import MilvusClient
from src.retriever.hybrid_retriever import HybridRetriever
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


def main():
    print("=" * 60)
    print("🔍 混合检索测试")
    print("=" * 60)

    # 初始化客户端
    neo4j = Neo4jClient(Config.NEO4J_URI, Config.NEO4J_USER, Config.NEO4J_PASSWORD)
    neo4j.connect()

    milvus = MilvusClient()
    milvus.connect()
    milvus.create_collection("document_chunks")

    # Step 0: 将文档切片并写入 Milvus（向量检索的数据来源）
    print("\n📥 写入文档块到 Milvus...")
    total_chunks = 0
    for doc_id, doc_text in TEST_DOCS:
        chunks = milvus.chunk_text(doc_text)
        n = milvus.insert_chunks(doc_id, chunks)
        total_chunks += n
        print(f"  {doc_id}: {n} 个 chunk 已写入")
    print(f"  📊 总计写入 {total_chunks} 个文档块")

    # 创建混合检索器
    retriever = HybridRetriever(neo4j, milvus)

    # 测试查询
    test_queries = [
        "智能客服系统用了哪些技术？",
        "这个项目依赖谁？",
        "智能客服系统是什么时候启动的？"
    ]

    for query in test_queries:
        print(f"\n📌 查询: '{query}'")
        result = retriever.retrieve(query)

        print(f"  📊 图谱结果: {len(result['graph_results'])} 条")
        print(f"  📊 向量结果: {len(result['vector_results'])} 条")
        print(f"  📊 融合结果: {len(result['fused_results'])} 条")

        for item in result['fused_results']:
            print(f"    - {item}")

    neo4j.close()
    milvus.disconnect()
    print("\n✅ 混合检索测试完成!")


if __name__ == "__main__":
    main()