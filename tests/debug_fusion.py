"""tests/debug_fusion.py —— 看三路召回的原始结果，定位"原文为什么没进上下文"。

链路是：图谱检索 + 向量检索 → RRF 融合 → 精排 → 给模型。
只看最终答案永远分不清是哪一段出的问题。这个脚本把前三段摊开。

用法：
    python tests/debug_fusion.py "公司员工大约有多少人"
    python tests/debug_fusion.py "公司员工大约有多少人" --needle 1200
"""

import argparse
import sys

sys.path.insert(0, ".")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--needle", default="", help="要追踪的原文片段")
    args = parser.parse_args()

    from src.config import Config
    from src.database.local_vector_store import LocalVectorStore
    from src.database.neo4j_client import Neo4jClient
    from src.retriever.hybrid_retriever import HybridRetriever

    neo4j = Neo4jClient(uri=Config.NEO4J_URI, user=Config.NEO4J_USER, password=Config.NEO4J_PASSWORD)
    neo4j.connect()
    store = LocalVectorStore()
    store.create_collection("document_chunks")

    retriever = HybridRetriever(neo4j, store)
    r = retriever.retrieve(args.query) if hasattr(retriever, "retrieve") else retriever.search(args.query)

    for key in ("graph_results", "vector_results", "fused_results"):
        items = r.get(key, [])
        print(f"\n{key}: {len(items)} 条")
        for i, it in enumerate(items[:8], 1):
            text = (it.get("text") or it.get("target") or "")[:60].replace("\n", " ")
            mark = "★" if args.needle and args.needle in (it.get("text") or "") else " "
            print(f"  {mark}{i:>2}. {it.get('doc_id')}#{it.get('chunk_index')}  {text}")

    if args.needle:
        fused = r.get("fused_results", [])
        idx = [i for i, it in enumerate(fused) if args.needle in (it.get("text") or "")]
        print(f"\n「{args.needle}」在融合结果中的位置：{idx[:3] if idx else '不存在'}")

    neo4j.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
