import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.extractors.triple_extractor import TripleExtractor
from src.config import Config

# 测试数据
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


def test_single_extraction():
    """测试单个文档抽取"""
    extractor = TripleExtractor(
        api_key=Config.DEEPSEEK_API_KEY,
        base_url=Config.DEEPSEEK_BASE_URL,
        model_name=Config.DEEPSEEK_MODEL,
        confidence_threshold=Config.CONFIDENCE_THRESHOLD
    )

    chunk_id, chunk_text = TEST_DOCS[0]
    result = extractor.extract(chunk_text, chunk_id)

    print(f"\n=== 抽取结果 ===")
    print(f"耗时: {result.extraction_time_ms:.2f}ms")
    print(f"有效三元组数: {len(result.triples)}")

    for triple in result.triples:
        print(
            f"({triple.head}: {triple.head_type.value}) "
            f"- {triple.relation.value} -> "
            f"({triple.tail}: {triple.tail_type.value}) "
            f"[conf={triple.confidence:.2f}]"
        )

    return result


def test_batch_extraction():
    """测试批量抽取"""
    extractor = TripleExtractor(
        api_key=Config.DEEPSEEK_API_KEY,
        base_url=Config.DEEPSEEK_BASE_URL,
        model_name=Config.DEEPSEEK_MODEL,
        confidence_threshold=Config.CONFIDENCE_THRESHOLD
    )

    results = extractor.batch_extract(TEST_DOCS)

    total_triples = sum(len(r.triples) for r in results)
    print(f"\n=== 批量抽取完成 ===")
    print(f"处理文档数: {len(TEST_DOCS)}")
    print(f"总三元组数: {total_triples}")

    for i, result in enumerate(results):
        print(f"\n文档 {i + 1}: {len(result.triples)} 个三元组")


if __name__ == "__main__":
    # 先跑单个测试
    test_single_extraction()

    # 再跑批量测试
    test_batch_extraction()