import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.neo4j_client import Neo4jClient
from src.database.milvus_client import MilvusClient
from src.agents.orchestrator import AgentOrchestrator
from src.config import Config

# 测试文档（与 Neo4j 三元组测试的原文一致）
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


def print_section(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def main():
    print("=" * 60)
    print("  多Agent协同 + DeepSeek答案生成 测试")
    print("=" * 60)

    # ========================================
    # 初始化客户端
    # ========================================
    print("\n[1/4] 初始化 Neo4j + Milvus...")
    neo4j = Neo4jClient(Config.NEO4J_URI, Config.NEO4J_USER, Config.NEO4J_PASSWORD)
    neo4j.connect()

    milvus = MilvusClient()
    milvus.connect()

    # ========================================
    # 写入 Milvus（让向量检索有数据可查）
    # ========================================
    print("\n[2/4] 写入文档块到 Milvus...")
    milvus.create_collection("document_chunks")
    total = 0
    for doc_id, doc_text in TEST_DOCS:
        chunks = milvus.chunk_text(doc_text)
        n = milvus.insert_chunks(doc_id, chunks)
        total += n
    print(f"  写入 {total} 个 chunk 到 Milvus")

    # ========================================
    # 初始化 Agent 编排器（启用 LLM 答案生成）
    # ========================================
    print("\n[3/4] 初始化 Agent 编排器 (use_llm=True)...")
    orchestrator = AgentOrchestrator(neo4j, milvus, use_llm=True)
    print("  Agent 编排器初始化完成 (Router + Local + Global + Answer)")

    # ========================================
    # 测试查询
    # ========================================
    test_queries = [
        ("事实性查询", "智能客服系统用了哪些技术？"),
        ("事实性查询", "B项目用了什么语言和框架？"),
        ("事实性查询", "智能客服系统依赖哪个项目？"),
        ("总结性查询", "总结所有项目的关系"),
        ("总结性查询", "对比分析项目的技术栈"),
    ]

    print("\n[4/4] 执行测试查询...")

    for query_type, query in test_queries:
        print_section(f"{query_type}: {query}")

        result = orchestrator.process(query)

        # 显示元信息
        print(f"  路由: {result.get('route', '?')} | "
              f"图谱: {result.get('graph_count', '?')}条 | "
              f"向量: {result.get('vector_count', '?')}条")

        # 显示答案
        print("\n  答案:")
        answer = result.get("answer", "无答案")
        for line in answer.strip().split('\n'):
            print(f"    {line.strip()}")

    # ========================================
    # 清理
    # ========================================
    print_section("清理资源")
    neo4j.close()
    milvus.disconnect()

    print("\n" + "=" * 60)
    print("  多Agent + DeepSeek 测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()


