"""手工端到端演示：走完整 Agent 编排链路（需要 Neo4j / Milvus 已启动）

为什么叫这个文件名而不是 test_*：
    这是**手工演示脚本**，会连真实数据库、写数据、可能调用外部 LLM。
    它不适合放进 pytest 单元测试（外部依赖会让 CI 不稳定，且 pytest 收集阶段
    的 import 失败会中断整个测试会话）。
    约定：
      - `pytest tests/`            → 无外部依赖的单元测试
      - `python tests/manual_agent_demo.py`      → 手工端到端演示（本文件）
      - `python tests/evaluation/run_evaluation.py` → 假客户端的评测闭环

用法：
    1. docker compose up -d        # 启动 Neo4j + Milvus
    2. python tests/manual_agent_demo.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.agents.orchestrator_v2 import AgentOrchestratorV2  # noqa: E402
from src.config import Config  # noqa: E402
from src.database.milvus_client import MilvusClient  # noqa: E402
from src.database.neo4j_client import Neo4jClient  # noqa: E402

# 演示文档（写入向量库，让检索有数据可查）
TEST_DOCS = [
    (
        "doc_demo_101",
        """
        智能客服系统于2024年3月启动，目标是将AI能力集成到现有的ERP系统中。
        项目负责人是王明，技术选型采用了Java 17、FastAPI框架和Milvus向量数据库。
        智能客服系统依赖B项目提供的用户权限管理模块。
        """,
    ),
    (
        "doc_demo_102",
        """
        B项目是一个微服务架构的基础设施项目，采用Go语言和gRPC通信协议。
        该项目由李婷领导，已于2025年1月完成一期交付。
        B项目为C项目提供了核心的认证服务。
        """,
    ),
]

TEST_QUERIES = [
    ("事实性查询", "智能客服系统用了哪些技术？"),
    ("事实性查询", "B项目用了什么语言和框架？"),
    ("关系查询", "智能客服系统依赖哪个项目？"),
    ("全局概览", "总结所有项目的关系"),
]


def print_section(title: str) -> None:
    print("\n" + "=" * 64)
    print(f"  {title}")
    print("=" * 64)


def main() -> None:
    print("=" * 64)
    print("  GraphRAG 端到端演示（多 Agent 编排 + 证据验证）")
    print("=" * 64)

    print("\n[1/4] 连接 Neo4j + Milvus …")
    neo4j = Neo4jClient(Config.NEO4J_URI, Config.NEO4J_USER, Config.NEO4J_PASSWORD)
    neo4j.connect()
    milvus = MilvusClient(host=Config.MILVUS_HOST, port=Config.MILVUS_PORT)
    milvus.connect()
    milvus.create_collection("document_chunks", drop_existing=False)  # 存在则复用
    milvus.load_if_needed()

    print("\n[2/4] 写入演示文档块到 Milvus …")
    total = 0
    offset = 0
    for doc_id, doc_text in TEST_DOCS:
        chunks = milvus.chunk_text(doc_text)
        n = milvus.insert_chunks(doc_id, chunks, chunk_offset=offset)
        offset += n
        total += n
    print(f"  写入 {total} 个 chunk")

    print("\n[3/4] 初始化编排器（use_llm 由 .env 的 USE_LLM_ANSWER 决定）…")
    orchestrator = AgentOrchestratorV2(
        neo4j_client=neo4j,
        milvus_client=milvus,
        use_llm=Config.USE_LLM_ANSWER,
        use_cache=Config.USE_SEMANTIC_CACHE,
    )
    print("  编排器就绪（router → local/global → evidence_validator → answer）")

    print("\n[4/4] 执行查询 …")
    for query_type, query in TEST_QUERIES:
        print_section(f"{query_type}：{query}")
        result = orchestrator.process(query)
        print(
            f"  route={result.get('route')} | steps={result.get('step_count')} | "
            f"evidence={result.get('evidence_status')} | "
            f"cached={result.get('from_cache')} | latency={result.get('latency_ms')}ms"
        )
        print("\n  答案:")
        for line in str(result.get("answer", "")).strip().split("\n"):
            print(f"    {line.strip()}")

    print_section("清理资源")
    neo4j.close()
    milvus.disconnect()
    print("\n演示完成。")


if __name__ == "__main__":
    main()
