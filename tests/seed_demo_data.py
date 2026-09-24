"""种子演示数据：让 GraphRAG 的检索链路有真实数据可查。

为什么要这个脚本：
    服务能启动 ≠ 链路能出结果。新环境（或数据卷被清空）后，Neo4j 与 Milvus
    都是空的，此时所有查询都会走到"证据不足 → 拒答"，容易被误判为系统故障。
    这个脚本灌入一批可验证的演示数据，并打印写入统计。

写入内容：
    - Neo4j：项目 / 人员 / 技术栈三元组（关系类型用 schemas 白名单里的合法值）
    - Milvus：文档块（供向量检索）

用法：
    cd graphrag_system
    python tests/seed_demo_data.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.config import Config  # noqa: E402
from src.database.neo4j_client import Neo4jClient  # noqa: E402
from src.database.milvus_client import MilvusClient  # noqa: E402
from src.schemas.graph_schema import EntityType, RelationType, Triple  # noqa: E402

COLLECTION = "document_chunks"

# ------------------------------------------------------------
# 三元组（head, head_type, relation, tail, tail_type, confidence, source_chunk）
# ------------------------------------------------------------
TRIPLES = [
    ("智能客服系统", EntityType.PROJECT, RelationType.USES, "FastAPI", EntityType.TECH_STACK, 0.95,
     "智能客服系统采用Java 17和FastAPI框架，部署在K8s集群。"),
    ("智能客服系统", EntityType.PROJECT, RelationType.USES, "Milvus", EntityType.TECH_STACK, 0.93,
     "智能客服系统的向量检索使用Milvus。"),
    ("智能客服系统", EntityType.PROJECT, RelationType.DEPENDS_ON, "B项目", EntityType.PROJECT, 0.92,
     "智能客服系统依赖B项目提供的用户权限管理模块。"),
    ("B项目", EntityType.PROJECT, RelationType.DEPENDS_ON, "C项目", EntityType.PROJECT, 0.90,
     "B项目依赖C项目提供的能力。"),
    ("C项目", EntityType.PROJECT, RelationType.USES, "gRPC", EntityType.TECH_STACK, 0.90,
     "C项目采用gRPC通信协议。"),
    ("张三", EntityType.PERSON, RelationType.LEADS, "B项目", EntityType.PROJECT, 0.95,
     "张三负责B项目的开发和上线工作。"),
]

# ------------------------------------------------------------
# 文档块（供向量检索；与上面的三元组互相印证）
# ------------------------------------------------------------
DOCS = {
    "doc_demo_001": [
        "智能客服系统采用Java 17和FastAPI框架，部署在K8s集群，向量检索使用Milvus。",
    ],
    "doc_demo_002": [
        "C项目采用gRPC通信协议，是微服务基础设施，被B项目依赖。",
    ],
    "doc_demo_003": [
        "张三负责B项目的开发和上线工作，是B项目的负责人。",
    ],
}


def main() -> None:
    print("=" * 60)
    print("写入 Neo4j 三元组")
    print("=" * 60)
    neo4j = Neo4jClient(
        uri=Config.NEO4J_URI, user=Config.NEO4J_USER, password=Config.NEO4J_PASSWORD
    )
    neo4j.connect()
    neo4j.create_indexes()  # 索引：加速实体名/类型查询（生产必需，别只靠 MERGE）

    written = 0
    for head, htype, rel, tail, ttype, conf, chunk in TRIPLES:
        triple = Triple(
            head=head, head_type=htype, relation=rel, tail=tail,
            tail_type=ttype, confidence=conf, source_chunk=chunk,
        )
        if neo4j.write_triple(triple):
            written += 1
            print(f"  + {head} -[{rel.value}]-> {tail}")
    print(f"  写入 {written}/{len(TRIPLES)} 条三元组")

    print()
    print("=" * 60)
    print("写入 Milvus 文档块")
    print("=" * 60)
    milvus = MilvusClient(host=Config.MILVUS_HOST, port=Config.MILVUS_PORT)
    milvus.connect()
    # 存在则复用（drop_existing=False 防止误删数据）
    milvus.create_collection(COLLECTION, drop_existing=False)
    milvus.load_if_needed()

    total_chunks = 0
    chinese_offset = 0
    for doc_id, chunks in DOCS.items():
        n = milvus.insert_chunks(doc_id, chunks, chunk_offset=chinese_offset)
        chinese_offset += n
        total_chunks += n
        print(f"  + {doc_id}: {n} 个块")
    print(f"  写入 {total_chunks} 个块")

    print()
    print("=" * 60)
    print("校验")
    print("=" * 60)
    with neo4j.driver.session() as session:
        nodes = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        rels = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
    print(f"  Neo4j: {nodes} 个节点, {rels} 条关系")
    print(f"  Milvus: collection={COLLECTION}, 实体数={milvus.collection.num_entities}")
    print()
    print("完成。现在可以问：'智能客服系统用了哪些技术？'")

    neo4j.close()
    milvus.disconnect()


if __name__ == "__main__":
    main()
