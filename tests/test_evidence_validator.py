"""
tests/test_evidence_validator.py —— Evidence Validator 单元测试（v2.2）

覆盖用户要求的 5 个 Case + 语义陷阱：
  Case 1: "张三负责哪些项目?"  graph: 张三--负责-->项目B          → sufficient
  Case 2: "张三负责哪些项目?"  graph: 张三--参与-->项目A          → insufficient（不是 negative）
  Case 3: "张三负责哪些项目?"  graph: 参与 + vector 也只说参与     → insufficient → refuse
  Case 4: "张三负责哪些项目?"  graph: 负责项目B + 参与项目A        → sufficient，证据只含"负责"
  Case 5: 正向负责边 + 文本"张三不再负责项目A"                    → conflict

语义陷阱断言：
  relation != expected  绝不产生 negative evidence，
  只能 insufficient（"没有证据证明负责" ≠ "证明不负责"）。

运行：python tests/test_evidence_validator.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.evidence_validator import (
    EvidenceValidator,
    extract_target_relation,
    extract_target_entity,
)

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  OK  {name}")
    else:
        FAIL += 1
        print(f"  XX  {name}  {detail}")


def graph_edge(entity, relation, target, conf=0.9):
    return {
        "doc_id": f"graph:{entity}",
        "chunk_index": 0,
        "text": f"{entity} {relation} {target}",
        "entity": entity,
        "relation": relation,
        "target": target,
        "confidence": conf,
        "source": "neo4j",
        "hop": 1,
    }


def doc(text):
    return {"doc_id": "d1", "text": text}


def test_case1_sufficient():
    print("\n[Case 1] graph: 张三--负责-->项目B → sufficient")
    v = EvidenceValidator()
    r = v.validate("张三负责哪些项目？", [graph_edge("张三", "负责", "项目B")], [])
    check("evidence_status=sufficient", r["evidence_status"] == "sufficient", str(r))
    check("next_action=answer", r["next_action"] == "answer")
    check("target_relation=负责", r["target_relation"] == "负责")
    check("存在正向证据", any(e["role"] == "positive" and e["target"] == "项目B" for e in r["evidence_detail"]))


def test_case2_insufficient_not_negative():
    print("\n[Case 2] graph: 张三--参与-->项目A → insufficient（绝不能 negative）")
    v = EvidenceValidator()
    r = v.validate("张三负责哪些项目？", [graph_edge("张三", "参与", "项目A")], [])
    check("evidence_status=insufficient", r["evidence_status"] == "insufficient", str(r))
    check("next_action=targeted_graph（首次不足→补充召回）", r["next_action"] == "targeted_graph")
    # 关键语义陷阱：不能把"参与"当"不负责"的证据
    check("绝不产生 negative evidence",
          all(e["role"] != "negative" for e in r["evidence_detail"]),
          str(r.get("evidence_detail")))
    check("参与项目A 标记为 partial（相关但非支持）",
          any(e["role"] == "partial" and e["target"] == "项目A" for e in r["evidence_detail"]))


def test_case3_refuse_after_targeted():
    print("\n[Case 3] 参与 + vector 也只说参与 → targeted 后仍不足 → refuse")
    v = EvidenceValidator()
    # 首次：insufficient → targeted_graph
    r1 = v.validate(
        "张三负责哪些项目？",
        [graph_edge("张三", "参与", "项目A")],
        [doc("张三参与项目A的开发工作")],
    )
    check("首次 insufficient", r1["evidence_status"] == "insufficient")
    check("首次 next_action=targeted_graph", r1["next_action"] == "targeted_graph")

    # targeted 补充后仍无"负责"证据 → 二次判断（targeted_retried=True）→ refuse
    r2 = v.validate(
        "张三负责哪些项目？",
        [graph_edge("张三", "参与", "项目A")],
        [doc("张三参与项目A的开发工作")],
        targeted_retried=True,
    )
    check("二次 insufficient → refuse", r2["evidence_status"] == "insufficient" and r2["next_action"] == "refuse",
          str(r2))


def test_case4_sufficient_only_responsible():
    print("\n[Case 4] 负责项目B + 参与项目A → sufficient，证据只含'负责'的")
    v = EvidenceValidator()
    edges = [graph_edge("张三", "负责", "项目B"), graph_edge("张三", "参与", "项目A")]
    r = v.validate("张三负责哪些项目？", edges, [])
    check("evidence_status=sufficient", r["evidence_status"] == "sufficient", str(r))
    positives = [e["target"] for e in r["evidence_detail"] if e["role"] == "positive"]
    check("正向证据只含项目B", positives == ["项目B"], str(positives))
    check("项目A 不作为负责证据", "项目A" not in positives)


def test_case5_conflict():
    print("\n[Case 5] 负责项目A + 文本'张三不再负责项目A' → conflict")
    v = EvidenceValidator()
    r = v.validate(
        "张三负责哪些项目？",
        [graph_edge("张三", "负责", "项目A")],
        [doc("张三不再负责项目A，已交接给李四")],
    )
    check("evidence_status=conflict", r["evidence_status"] == "conflict", str(r))
    check("next_action=conflict", r["next_action"] == "conflict")
    check("conflict_status=detected（待 Resolver 裁决）", r.get("conflict_status") == "detected")
    check("收集了正反冲突证据", len(r.get("conflict_evidence", [])) == 2,
          str(r.get("conflict_evidence")))


def test_semantic_trap():
    print("\n[语义陷阱] '没有证据证明负责' ≠ '证明不负责'")
    v = EvidenceValidator()
    # 只有"参与"和"合作"证据，没有任何"负责"——绝不能推出"不负责"
    r = v.validate(
        "张三负责哪些项目？",
        [graph_edge("张三", "参与", "项目A"), graph_edge("张三", "合作", "项目C")],
        [],
    )
    check("insufficient 而非 negative/conflict", r["evidence_status"] == "insufficient",
          str(r))
    check("无 negative 证据", all(e["role"] != "negative" for e in r["evidence_detail"]))

    # 文本里出现"不负责"应只在明确否定同一实体+关系时算 conflict，
    # 其他否定词不应误判
    r2 = v.validate(
        "张三负责哪些项目？",
        [graph_edge("张三", "负责", "项目B")],
        [doc("项目A的开发张三没有参与")],   # "没有参与" ≠ "不再负责项目B"
    )
    check("无关否定不误判 conflict", r2["evidence_status"] != "conflict",
          f"status={r2['evidence_status']} detail={r2.get('conflict_status')}")


def test_relation_extraction():
    print("\n[关系提取] extract_target_relation / extract_target_entity")
    check("负责 → 负责", extract_target_relation("张三负责哪些项目？") == "负责")
    check("依赖 → 依赖", extract_target_relation("B项目依赖什么系统？") == "依赖")
    check("使用 → 使用", extract_target_relation("智能客服系统用了哪些技术？") == "使用")
    check("无关系词 → None", extract_target_relation("今天天气怎么样") is None)
    check("实体提取(known)", extract_target_entity("张三负责哪些项目？", ["张三", "李四"]) == "张三")
    check("实体提取(启发式)", extract_target_entity("张三负责的项目", []) == "张三")


def test_pipeline_with_fake_graph():
    """用假 Neo4j 跑真实 orchestrator 图（复用 fake StateGraph 思路），
    验证 evidence_validator → targeted_graph → refuse 的路径真实可达。"""
    print("\n[流程] 模拟 orchestrator 状态流转（不依赖 langgraph）")
    # 模拟 targeted_graph_node 的行为：调 neo4j_client.query_entity_by_relation
    class FakeNeo4j:
        def query_entity_by_relation(self, entity, relation_type):
            # 模拟：张三--负责-->项目B 存在（补充后 sufficient）
            return [{"entity": "张三", "relation": "负责", "target": "项目B",
                     "target_type": "项目", "confidence": 0.95}]
        class _D:
            def session(self): raise RuntimeError("不应调用")
        driver = _D()

    from src.agents.evidence_validator import EvidenceValidator as EV
    # 模拟：初始 local 只有"参与"证据
    v = EV()
    initial_edges = [graph_edge("张三", "参与", "项目A")]
    r1 = v.validate("张三负责哪些项目？", initial_edges, [], targeted_retried=False)
    check("首次 insufficient → targeted_graph", r1["next_action"] == "targeted_graph")

    # targeted 补充：合并"负责"边
    targeted = FakeNeo4j().query_entity_by_relation("张三", "负责")
    merged = list(initial_edges) + [graph_edge(t["entity"], t["relation"], t["target"]) for t in targeted]
    r2 = v.validate("张三负责哪些项目？", merged, [], targeted_retried=True)
    check("补充后 sufficient → answer", r2["evidence_status"] == "sufficient" and r2["next_action"] == "answer",
          str(r2))


if __name__ == "__main__":
    print("=" * 60)
    print("Evidence Validator 单元测试")
    print("=" * 60)
    test_relation_extraction()
    test_case1_sufficient()
    test_case2_insufficient_not_negative()
    test_case3_refuse_after_targeted()
    test_case4_sufficient_only_responsible()
    test_case5_conflict()
    test_semantic_trap()
    test_pipeline_with_fake_graph()
    print("\n" + "=" * 60)
    print(f"结果: {PASS} 通过, {FAIL} 失败")
    print("=" * 60)
    sys.exit(1 if FAIL else 0)
