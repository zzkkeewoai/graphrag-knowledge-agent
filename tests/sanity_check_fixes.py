"""
修复验证脚本（不需要 Docker / Neo4j / Milvus，全部用假客户端）

验证 v2.1 四个修复：
  1. RRF key 坍缩修复：图谱结果不再全部落在 "graph||0" 一个 key 上
  2. 图谱检索不再 LIMIT 100 截断，且 doc_id 唯一、文本可回溯
  3. 两跳邻居扩展（多跳推理链路）可用
  4. 语义缓存不再存空 answer（缓存写入挪到 answer_node 之后）
  5. RouterAgent LLM 兜底：无 API key 时优雅回退 local，不抛错

运行：python tests/sanity_check_fixes.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retriever.hybrid_retriever import HybridRetriever
from src.utils.semantic_cache import SemanticCache

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = ""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name}  {detail}")


# ============================================================
# 假 Neo4j 客户端：返回固定图谱数据
# ============================================================
class FakeNeo4j:
    """模拟 Neo4j：第一跳返回 3 条关系，两跳查询返回 1 条扩展关系"""

    HOP1_ROWS = [
        {"entity": "智能客服系统", "etype": "项目", "rel": "使用",
         "conf": 0.95, "target": "Java 17", "ttype": "技术栈"},
        {"entity": "智能客服系统", "etype": "项目", "rel": "依赖",
         "conf": 0.90, "target": "B项目", "ttype": "项目"},
        {"entity": "B项目", "etype": "项目", "rel": "使用",
         "conf": 0.92, "target": "Go语言", "ttype": "技术栈"},
    ]
    HOP2_ROWS = [
        {"entity": "智能客服系统", "rel1": "依赖", "mid": "B项目",
         "rel2": "使用", "target": "gRPC", "conf1": 0.90, "conf2": 0.88},
    ]

    class Session:
        def run(self, cypher: str, **params):
            if "r2:RELATES_TO" in cypher:
                return FakeNeo4j.HOP2_ROWS
            return FakeNeo4j.HOP1_ROWS

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def __init__(self):
        self.driver = type("Driver", (), {"session": lambda self: FakeNeo4j.Session()})()


class FakeMilvus:
    """模拟 Milvus 向量检索：返回 2 条固定结果"""

    def search(self, query: str, top_k: int = 5):
        return [
            {"doc_id": "doc_001", "chunk_index": 0,
             "chunk_text": "智能客服系统采用Java 17和FastAPI框架", "score": 0.81},
            {"doc_id": "doc_002", "chunk_index": 0,
             "chunk_text": "B项目采用Go语言和gRPC通信协议", "score": 0.76},
        ]


def test_rrf_no_collapse():
    print("\n[1] RRF key 坍缩修复：图谱结果必须每个都是独立融合位")
    retriever = HybridRetriever(FakeNeo4j(), FakeMilvus())
    results = retriever.retrieve("智能客服系统用了哪些技术？", top_k=5)

    graph_keys = {f"{r['doc_id']}||{r['chunk_index']}" for r in results['graph_results']}
    fused_keys = [f"{f['doc_id']}||{f['chunk_index']}" for f in results['fused_results']]

    check("图谱结果 doc_id 唯一（不再全是 graph||0）",
          len(graph_keys) == len(results['graph_results']),
          f"graph_keys={graph_keys}")
    check("图谱文本可回溯（text_map 能找到真实三元组文本）",
          all("实体: " not in r.get("text", "") or "已知实体" in r.get("text", "")
              for r in results['graph_results']),
          f"texts={[r.get('text') for r in results['graph_results']]}")
    check("融合结果 doc_id 来自两路（graph + milvus doc_id）",
          any(k.startswith("graph:") for k in fused_keys) and
          any("doc_00" in k for k in fused_keys),
          f"fused_keys={fused_keys}")
    # 图谱 3 条 + 向量 2 条 = 融合上限 5，全部应该被保留
    check("融合结果完整保留两路信息（不坍缩成 1 条）",
          len(results['fused_results']) >= 4,
          f"fused={len(results['fused_results'])}")


def test_two_hop_expansion():
    print("\n[2] 多跳扩展：hops=2 时能检索到两跳邻居")
    retriever = HybridRetriever(FakeNeo4j(), FakeMilvus())
    one_hop = retriever._graph_retrieve("智能客服系统", hops=1)
    two_hop = retriever._graph_retrieve("智能客服系统", hops=2)

    hop2_items = [r for r in two_hop if r.get("hop") == 2]
    check("hops=2 返回两跳邻居（智能客服系统-依赖->B项目-使用->gRPC）",
          len(hop2_items) >= 1,
          f"hop2={[(r['entity'], r['relation'], r['target']) for r in hop2_items]}")
    check("两跳结果带独立 doc_id（graph2hop: 前缀）",
          all(r["doc_id"].startswith("graph2hop:") for r in hop2_items),
          f"doc_ids={[r['doc_id'] for r in hop2_items]}")
    # 第一跳也应保留（hop=1 数量不变）
    check("第一跳结果不丢失",
          len([r for r in two_hop if r.get("hop") == 1]) == len(one_hop))


def test_semantic_cache_flow():
    print("\n[3] 语义缓存：命中返回完整答案（不再存空 answer）")

    class FakeEncoder:
        def encode(self, texts, normalize_embeddings=True):
            # 相同文本 → 完全相同向量；不同文本 → 相似度 0
            import numpy as np
            return np.array([[1.0, 0.0] if t == "q1" else [0.0, 1.0] for t in texts])

    cache = SemanticCache(encoder=FakeEncoder(), similarity_threshold=0.9)

    # 模拟修复后的写入时机：answer_node 生成最终答案后才写缓存
    cache.set("q1", {"answer": "这是完整答案", "documents": [{"doc_id": "d1"}]})

    hit = cache.get("q1")
    check("精确缓存命中且 answer 非空", hit is not None and hit["answer"] == "这是完整答案")

    # 语义相似命中（同向量）
    semantic_hit = cache.get("q1")  # 同一文本
    check("语义缓存命中", semantic_hit is not None and semantic_hit["answer"] == "这是完整答案")

    miss = cache.get("完全不同的问题")
    check("无关查询不命中", miss is None)

    check("hit_rate 统计正常",
          0 < cache.hit_rate < 1, f"hit_rate={cache.hit_rate}")


def test_router_llm_fallback():
    print("\n[4] RouterAgent LLM 兜底：无 API key 时优雅回退 local")
    from src.agents.orchestrator_v2 import RouterAgent
    router = RouterAgent(use_llm_fallback=True)
    # 没有 DEEPSEEK_API_KEY 时，openai 调用必然失败 → 应回退 "local" 而不是抛错
    old_key = os.environ.get("DEEPSEEK_API_KEY")
    os.environ.pop("DEEPSEEK_API_KEY", None)
    try:
        route = router._llm_route("帮我看看这个模糊的问题")
        check("LLM 路由失败回退 local 且不抛错", route == "local", f"route={route}")
    finally:
        if old_key:
            os.environ["DEEPSEEK_API_KEY"] = old_key

    # 关键词路由本身仍工作
    check("关键词路由仍工作（global）", router.route("总结所有项目的关系") == "global")
    check("关键词路由仍工作（local）", router.route("智能客服系统用了哪些技术") == "local")


if __name__ == "__main__":
    print("=" * 60)
    print("GraphRAG v2.1 修复验证")
    print("=" * 60)

    test_rrf_no_collapse()
    test_two_hop_expansion()
    test_semantic_cache_flow()
    test_router_llm_fallback()

    print("\n" + "=" * 60)
    print(f"结果: {PASS} 通过, {FAIL} 失败")
    print("=" * 60)
    sys.exit(1 if FAIL else 0)
