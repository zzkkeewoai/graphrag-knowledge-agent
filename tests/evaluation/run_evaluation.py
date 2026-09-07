"""
run_evaluation.py —— RAG Evaluation Runner（v2.2）

走真实生产路径（逻辑层）：
  query → Router → Retrieval(local) → RRF → Reranker
        → EvidenceValidator → ConflictResolver → Answer / Refuse

数据源：tests/evaluation/dataset.json 中每个 case 的 knowledge（graph_edges + texts）
注入假 Neo4j/Milvus/Reranker；执行真实节点闭包（复用 sanity 的 fake StateGraph
执行器，不依赖 langgraph 库 / Docker / 外部 LLM，use_llm=False 模板模式）。

指标：
  Retrieval Recall              : expected entity/target 是否进入候选
  Evidence Validation Accuracy  : expected_evidence vs 实际 evidence_status
  Refusal Accuracy              : 应拒答 → 是否真拒答
  Answer Accuracy               : 确定性 answer 类，target/entity 是否进最终答案
  Overall Summary               : 汇总

运行：python tests/evaluation/run_evaluation.py
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.agents.orchestrator_v2 import create_orchestrator_graph

DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.json")


# ============================================================
# 假的 langgraph.graph 模块（复用 sanity 思路）
# ============================================================
class FakeStateGraph:
    def __init__(self, state_schema):
        self.nodes = {}
        self.plain_edges = []
        self.conditional = {}
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
    def __init__(self, graph):
        self.graph = graph

    def invoke(self, state):
        current = self.graph.entry
        steps = 0
        while current != "END" and steps < 100:
            node_fn = self.graph.nodes[current]
            returned = node_fn(dict(state))
            state.update(returned)
            if current in self.graph.conditional:
                decider, mapping = self.graph.conditional[current]
                nxt = decider(state)
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
# 假客户端（每 case 注入 knowledge）
# ============================================================
class FakeRecord(dict):
    def data(self):
        return dict(self)


class FakeNeo4j:
    """模拟 Neo4j：graph_edges 作为图谱内容。
    - _graph_retrieve 的 Cypher 查询 → 返回所有 edges（简化：实体匹配在 Python 侧已由
      hybrid_retriever 的 keyword 过滤完成，这里直接给全量行）
    - query_entity_by_relation（targeted）→ 按 relation 过滤
    """
    edges = []  # 每 case 注入

    class Session:
        def run(self, cypher: str, **params):
            if "r2:RELATES_TO" in cypher:      # 两跳扩展
                return []
            if "MATCH (p:Entity" in cypher:    # global
                return []
            # 一跳图谱检索 → 返回 edges 作为行（含 head/tail 形式由 _graph_retrieve 使用）
            rows = []
            for e in FakeNeo4j.edges:
                rows.append(FakeRecord({
                    "entity": e["entity"],
                    "etype": "项目",
                    "rel": e["relation"],
                    "conf": e.get("confidence", 0.9),
                    "target": e["target"],
                    "ttype": "项目",
                }))
            return rows

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def __init__(self):
        self.driver = type("Driver", (), {"session": lambda self: FakeNeo4j.Session()})()

    def query_entity_by_relation(self, entity_name: str, relation_type: str) -> list:
        out = []
        for e in FakeNeo4j.edges:
            if e["entity"] == entity_name and e["relation"] == relation_type:
                out.append({
                    "entity": e["entity"], "relation": e["relation"],
                    "target": e["target"], "target_type": "项目",
                    "confidence": e.get("confidence", 0.9),
                })
        return out


class FakeMilvus:
    texts = []  # 每 case 注入

    def search(self, query: str, top_k: int = 5):
        out = []
        for i, t in enumerate(FakeMilvus.texts[:top_k]):
            out.append({
                "doc_id": t.get("doc_id", f"doc_{i}"),
                "chunk_index": i,
                "chunk_text": t["text"],
                "score": 0.8,
            })
        return out


class FakeReranker:
    """避免加载真实 MiniLM。Rerank 保持原序（评估关心 Evidence 而非排序质量）。"""
    def __init__(self, *a, **k):
        self.model = None

    def rerank(self, query, documents, top_k=3):
        return documents[:top_k]


# ============================================================
# 指标计算
# ============================================================
class Metrics:
    def __init__(self):
        self.retrieval_hits = 0
        self.retrieval_total = 0
        self.ev_ok = 0
        self.refusal_correct = 0
        self.refusal_total = 0
        self.answer_correct = 0
        self.answer_total = 0
        self.insufficient_cases = 0
        self.correct_refusal = 0
        self.incorrect_answer_on_refuse = 0
        self.conflict_cases = 0
        self.conflict_resolved = 0
        self.conflict_unresolved = 0


def run_single(case):
    """跑一个 case，返回 (state, result_summary)"""
    FakeNeo4j.edges = case.get("knowledge", {}).get("graph_edges", [])
    FakeMilvus.texts = case.get("knowledge", {}).get("texts", [])

    import src.agents.orchestrator_v2 as orch
    orch.BGEReranker = FakeReranker  # patch 避免加载模型

    neo4j = FakeNeo4j()
    milvus = FakeMilvus()
    app = create_orchestrator_graph(
        neo4j_client=neo4j,
        milvus_client=milvus,
        use_llm=False,      # 模板模式：本地可跑，不依赖外部 LLM
        use_cache=False,
        max_retries=2,
    )
    initial = {
        "query": case["query"],
        "route": "local",
        "step_count": 0,
        "max_steps": 15,
        "error_count": 0,
        "documents": [],
        "answer": "",
    }
    final = app.invoke(initial)
    return final


def evaluate():
    with open(DATASET_PATH, encoding="utf-8") as f:
        cases = json.load(f)["cases"]

    m = Metrics()
    details = []

    for case in cases:
        state = run_single(case)
        query = case["query"]
        exp_behavior = case["expected_behavior"]
        exp_evidence = case.get("expected_evidence")
        exp_entities = case.get("expected_entities") or []
        exp_targets = case.get("expected_targets") or []
        exp_resolution = case.get("expected_resolution")

        answer = state.get("answer", "")
        ev_status = state.get("evidence_status", "unknown")
        next_action = state.get("next_action", "")
        conflict_status = state.get("conflict_status", "none")
        # 检索到的候选（graph_results + documents 文本）
        graph_results = state.get("graph_results") or []
        documents = state.get("documents") or []
        candidate_text = " ".join(
            [r.get("text", "") for r in graph_results]
            + [d.get("text", "") for d in documents]
        )

        # ---- 1. Retrieval Recall ----
        # expected entity/target 是否进入候选（对 answer 类有意义；refuse 类若 exp 为空跳过）
        exp_all = exp_entities + exp_targets
        if exp_all:
            m.retrieval_total += 1
            if all(x in candidate_text or x in query for x in exp_all):
                m.retrieval_hits += 1

        # ---- 2. Evidence Validation Accuracy ----
        # expected_evidence: sufficient / insufficient / conflict
        # 注意：conflict resolved 后 evidence_status 可能为 "resolved"，视为检测到 conflict
        ev_actual = ev_status
        if ev_actual == "resolved":
            ev_actual_for_check = "conflict"
        else:
            ev_actual_for_check = ev_actual

        if exp_evidence == "conflict":
            m.conflict_cases += 1
            if conflict_status == "resolved":
                m.conflict_resolved += 1
            elif conflict_status == "unresolved":
                m.conflict_unresolved += 1

        ev_ok = (exp_evidence == ev_actual_for_check)
        if ev_ok:
            m.ev_ok += 1

        # ---- 3. Refusal Accuracy ----
        refused = (next_action == "refuse") or ("无法" in answer or "不足" in answer
                                                or "不存在" in answer or "冲突信息" in answer)
        if exp_behavior == "refuse":
            m.refusal_total += 1
            if refused:
                m.refusal_correct += 1
            else:
                # 应拒答但系统给了实体答案
                m.incorrect_answer_on_refuse += 1
            # insufficient 子统计
            if exp_evidence == "insufficient":
                m.insufficient_cases += 1
                if refused:
                    m.correct_refusal += 1
        else:
            # expected answer，但系统拒了 → 也算 refusal 侧失败（误拒）
            if refused:
                m.refusal_total += 1
                # 误拒：expected answer 却拒答
                # 记到 refusal_total 但不加 correct，同时 Answer Accuracy 记失败
                pass

        # ---- 4. Answer Accuracy ----
        if exp_behavior == "answer":
            m.answer_total += 1
            # 确定性规则：expected target 出现在 answer 中（或 expected_entities + relation 短语）
            ok = False
            if exp_targets:
                ok = any(t in answer for t in exp_targets)
            elif exp_entities:
                ok = any(e in answer for e in exp_entities)
            # conflict resolved(oppose)：answer 应含否定（"不再"）
            if exp_resolution == "oppose":
                ok = ok and ("不再" in answer or "不负责" in answer or "已交接" in answer or "不" in answer)
            if ok:
                m.answer_correct += 1

        # 详细输出
        status = "PASS" if _case_pass(case, state, ev_ok, refused, answer) else "FAIL"
        details.append({
            "status": status, "id": case["id"], "query": query,
            "expected_behavior": exp_behavior, "actual_behavior": "refuse" if refused else "answer",
            "expected_evidence": exp_evidence, "actual_evidence": ev_status,
            "conflict_status": conflict_status,
            "answer_preview": answer[:80],
            "ev_ok": ev_ok,
        })

    # ---- 打印 ----
    print("\n" + "=" * 70)
    print("RAG Evaluation Results")
    print("=" * 70)

    for d in details:
        mark = "[PASS]" if d["status"] == "PASS" else "[FAIL]"
        print(f"{mark} {d['id']}")
        print(f"   Query: {d['query']}")
        print(f"   Expected: {d['expected_behavior']} / evidence={d['expected_evidence']}")
        print(f"   Actual:   {d['actual_behavior']} / evidence={d['actual_evidence']} / conflict={d['conflict_status']}")
        print(f"   Answer: {d['answer_preview']}")

    print("\n" + "=" * 70)
    print("Evaluation Summary")
    print("-" * 70)
    total = len(cases)
    rr = m.retrieval_hits / m.retrieval_total if m.retrieval_total else 0
    eva = m.ev_ok / total if total else 0
    ra = m.refusal_correct / m.refusal_total if m.refusal_total else 0
    aa = m.answer_correct / m.answer_total if m.answer_total else 0
    print(f"Total: {total}")
    print(f"Retrieval Recall: {rr:.2f}  ({m.retrieval_hits}/{m.retrieval_total})")
    print(f"Evidence Validation Accuracy: {eva:.2f}  ({m.ev_ok}/{total})")
    print(f"Refusal Accuracy: {ra:.2f}  ({m.refusal_correct}/{m.refusal_total})")
    print(f"Answer Accuracy: {aa:.2f}  ({m.answer_correct}/{m.answer_total})")
    print(f"\nInsufficient cases: {m.insufficient_cases}")
    print(f"  Correct refusal: {m.correct_refusal}")
    print(f"  Incorrect answer: {m.incorrect_answer_on_refuse}")
    print(f"Conflict cases: {m.conflict_cases}")
    print(f"  Resolved: {m.conflict_resolved}")
    print(f"  Unresolved: {m.conflict_unresolved}")

    failed = [d for d in details if d["status"] == "FAIL"]
    print(f"\nFailed cases: {len(failed)}")
    for d in failed:
        print(f"  - {d['id']}: expected {d['expected_behavior']}/{d['expected_evidence']}, "
              f"got {d['actual_behavior']}/{d['actual_evidence']}")


def _case_pass(case, state, ev_ok, refused, answer):
    exp_behavior = case["expected_behavior"]
    exp_evidence = case.get("expected_evidence")

    # evidence 状态必须对（sufficient/insufficient/conflict）
    ev_actual = state.get("evidence_status", "")
    ev_actual_check = "conflict" if ev_actual == "resolved" else ev_actual
    if exp_evidence != ev_actual_check:
        return False

    if exp_behavior == "refuse":
        return refused
    # answer 类：不拒答即可（target 匹配已在 Answer Accuracy 单独统计）
    return not refused


if __name__ == "__main__":
    evaluate()
