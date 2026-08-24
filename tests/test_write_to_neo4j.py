import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.extractors.triple_extractor import TripleExtractor
from src.database.neo4j_client import Neo4jClient
from src.config import Config

# 测试文档
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
    print("🚀 开始GraphRAG端到端流程")
    print("=" * 60)

    # 1. 抽取三元组
    print("\n📖 Step 1: 抽取三元组...")
    extractor = TripleExtractor(
        api_key=Config.DEEPSEEK_API_KEY,
        base_url=Config.DEEPSEEK_BASE_URL,
        model_name=Config.DEEPSEEK_MODEL,
        confidence_threshold=0.7
    )

    all_triples = []
    for chunk_id, chunk_text in TEST_DOCS:
        result = extractor.extract(chunk_text, chunk_id)
        all_triples.extend(result.triples)
        print(f"  ✅ {chunk_id}: {len(result.triples)} 个三元组")

    print(f"\n📊 总共抽取: {len(all_triples)} 个三元组")

    # 2. 写入Neo4j
    print("\n💾 Step 2: 写入Neo4j...")
    client = Neo4jClient(
        uri=Config.NEO4J_URI,
        user=Config.NEO4J_USER,
        password=Config.NEO4J_PASSWORD
    )
    client.connect()
    client.create_indexes()

    write_result = client.batch_write(all_triples)
    print(f"  ✅ 成功写入: {write_result['success']}")
    print(f"  ❌ 写入失败: {write_result['failed']}")

    # 3. 查询验证
    print("\n🔍 Step 3: 查询验证")
    tech_stack = client.query_tech_stack("智能客服系统")
    print(f"  📌 智能客服系统技术栈: {', '.join(tech_stack)}")

    deps = client.query_dependencies("智能客服系统")
    for dep in deps:
        print(f"  📌 依赖: {dep['project']} (置信度: {dep['confidence']})")

    client.close()
    print("\n✅ 端到端流程完成!")


if __name__ == "__main__":
    main()