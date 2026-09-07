"""
降级路径验证：local 挂了自动切 global（不依赖真实 langgraph / Docker）

原理：在 sys.modules 里注入一个"假的 langgraph.graph"（StateGraph 记录节点与边，
compile 返回一个按边执行真实节点闭包的 FakeApp），再用假 Neo4j/Milvus/Reranker
客户端实例化 create_orchestrator_graph —— 执行的是 orchestrator_v2.py 里
真实的 local_node / global_node / answer_node 闭包。

场景：
  1. 正常查询 → local 成功 → route=local，answer 来自模板
  2. 故障查询（Milvus 抛异常）→ local 重试 3 次全挂 → fallback_to_global
     → 条件边切 global → global 成功 → route=global，answer 来自全局分析

运行：python tests/sanity_check_fallback.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = ""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  OK  {name}")
    else:
        FAIL += 1
        print(f"  XX  {name}  {detail}")


# ============================================================
# 假的 langgraph.graph 模块
# ============================================================
class FakeStateGraph:
    def __init__(self, state_schema):
        self.nodes = {}
        self.plain_edges = []            # [(from, to)]
        self.conditional = {}            # from -> (decider_fn, mapping)
        self.entry = None

    def add_node(self, name, fn):
        self.nodes[name] = fn

    def add_edge(self, a, b):
        self.plain_edges.append((a, b))

    def add_conditional_edges(self, a, decider, mapping):
        self.conditional[a] = (decider, mapping)

    def set_entry_point(self, name):
        self.entry = name

    def compile(self):
        return FakeApp(self)


class FakeApp:
    """按边执行真实节点闭包的最小图执行器"""

    def __init__(self, graph: FakeStateGraph):
        self.graph = graph

    def invoke(self, state):
        current = self.graph.entry
        steps = 0
        while current != "END" and steps < 100:
            node_fn = self.graph.nodes[current]
            returned = node_fn(dict(state))          # 执行真实节点闭包
            state.update(returned)
            if current in self.graph.conditional:
                decider, mapping = self.graph.conditional[current]
                nxt = decider(state)                  # 执行真实条件边函数
                current = mapping.get(nxt, "END")
            else:
                nxts = [b for (a, b) in self.graph.plain_edges if a == current]
                current = nxts[0] if nxts else "END"
            steps += 1
        state["_steps"] = steps
        return state


fake_langgraph_module = type(sys)("langgraph.graph")
fake_langgraph_module.StateGraph = FakeStateGraph
fake_langgraph_module.END = "END"
sys.modules["langgraph"] = type(sys)("langgraph")
sys.modules["langgraph.graph"] = fake_langgraph_module


# ============================================================
# 假客户端（Neo4j / Milvus / Reranker）
# ============================================================
class FakeRecord(dict):
    """既支持 row['key'] 索引，又有 .data() 方法（模拟 neo4j Record）"""
    def data(self):
        return dict(self)


class FakeNeo4j:
    HOP1_ROWS = [
        {"entity": "智能客服系统", "etype": "项目", "rel": "使用",
         "conf": 0.95, "target": "Java 17", "ttype": "技术栈"},
        {"entity": "智能客服系统", "etype": "项目", "rel": "依赖",
         "conf": 0.90, "target": "B项目", "ttype": "项目"},
    ]
    GLOBAL_ROWS = [
        {"project": "智能客服系统", "relation": "使用", "target": "Java 17", "confidence": 0.95},
        {"project": "智能客服系统", "relation": "依赖", "target": "B项目", "confidence": 0.90},
    ]

    class Session:
        def run(self, cypher: str, **params):
            if "MATCH (p:Entity" in cypher:          # global 全图统计
                return [FakeRecord(r) for r in FakeNeo4j.GLOBAL_ROWS]
            if "r2:RELATES_TO" in cypher:            # 两跳扩展
                return []
            return [FakeRecord(r) for r in FakeNeo4j.HOP1_ROWS]  # 一跳

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def __init__(self):
        self.driver = type("Driver", (), {"session": lambda self: FakeNeo4j.Session()})()

    # v2.2: Evidence Validator 的定向图谱检索（测试适配）
    def query_entity_by_relation(self, entity_name: str, relation_type: str) -> list:
        """返回与 entity + relation_type 精确匹配的边（模拟 targeted retrieval）"""
        results = []
        for row in FakeNeo4j.HOP1_ROWS:
            if row["entity"] == entity_name and row["rel"] == relation_type:
                results.append({
                    "entity": row["entity"],
                    "relation": row["rel"],
                    "target": row["target"],
                    "target_type": row.get("ttype", ""),
                    "confidence": row["conf"],
                })
        return results


class FakeMilvus:
    """fail_mode=True 时模拟向量库故障（local 检索必然失败）"""
    fail_mode = False

    def search(self, query: str, top_k: int = 5):
        if FakeMilvus.fail_mode:
            raise ConnectionError("Milvus 连接超时（模拟故障）")
        return [
            {"doc_id": "doc_001", "chunk_index": 0,
             "chunk_text": "智能客服系统采用Java 17和FastAPI框架", "score": 0.81},
        ]


class FakeReranker:
    """避免加载真实 MiniLM 模型"""
    def __init__(self, *a, **k):
        self.model = None

    def rerank(self, query, documents, top_k=3):
        return documents[:top_k]


# ============================================================
# 测试主体
# ============================================================
def run_pipeline(query: str):
    """实例化编排器并跑一次完整流程（真实节点闭包 + 假边执行器）"""
    import src.agents.orchestrator_v2 as orch
    orch.BGEReranker = FakeReranker          # patch 重排器，避免加载模型

    neo4j = FakeNeo4j()
    milvus = FakeMilvus()
    graph = orch.create_orchestrator_graph(
        neo4j_client=neo4j,
        milvus_client=milvus,
        use_llm=False,        # 模板生成，零 API 依赖
        use_cache=False,
        max_retries=3,
    )
    initial = {
        "query": query,
        "route": "local",
        "step_count": 0,
        "max_steps": 15,
        "error_count": 0,
        "documents": [],
        "answer": "",
    }
    final = graph.invoke(initial)
    return final


def test_normal_local():
    print("\n[场景1] 正常查询 → local 成功")
    final = run_pipeline("智能客服系统用了哪些技术")
    check("路由保持 local", final.get("route") == "local", f"route={final.get('route')}")
    check("没有触发降级标记", not final.get("fallback_to_global"), str(final.get("fallback_to_global")))
    check("返回了文档", len(final.get("documents", [])) > 0, f"docs={len(final.get('documents', []))}")
    check("answer 非空（模板生成）", bool(final.get("answer")), f"answer={final.get('answer','')[:30]}")
    return final


def test_fallback_to_global():
    print("\n[场景2] Milvus 故障 → local 重试 3 次全挂 → 自动切 global")
    FakeMilvus.fail_mode = True
    try:
        final = run_pipeline("智能客服系统用了哪些技术")
    finally:
        FakeMilvus.fail_mode = False

    check("触发了降级标记", final.get("fallback_to_global") is True, str(final.get("fallback_to_global")))
    check("路由被改写为 global", final.get("route") == "global", f"route={final.get('route')}")
    # local 失败 error_count=1 → global 成功后被重置为 0（降级成功=系统自愈，合理行为）
    check("error_count 被 global 成功重置（自愈）",
          final.get("error_count", -1) == 0, f"error_count={final.get('error_count')}")
    check("answer 来自全局分析兜底（含项目关系摘要）",
          bool(final.get("answer")) and "智能客服系统" in final.get("answer", ""),
          f"answer={final.get('answer','')[:50]}")
    return final


if __name__ == "__main__":
    print("=" * 60)
    print("降级路径验证：local 挂了自动切 global")
    print("=" * 60)
    test_normal_local()
    test_fallback_to_global()
    print("\n" + "=" * 60)
    print(f"结果: {PASS} 通过, {FAIL} 失败")
    print("=" * 60)
    sys.exit(1 if FAIL else 0)
