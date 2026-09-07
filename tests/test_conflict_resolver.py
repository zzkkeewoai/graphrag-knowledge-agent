"""
tests/test_conflict_resolver.py —— Conflict Resolver 单元测试（v2.2）

覆盖用户要求的 Case 1-4 + insufficient vs conflict 区分：
  Case 1: Graph"负责项目A" + Vector"张三负责项目A"（正向一致）      → sufficient / no conflict
  Case 2: Graph"负责项目A" + Vector"不再负责，目前由李四负责"（有时间信号）
                                                               → conflict detected → resolved(oppose) → Answer
  Case 3: Graph"负责项目A" + Vector"张三不负责项目A"（无时间无来源）
                                                               → conflict → unresolved → refuse
  Case 4: Graph"参与项目A" + Vector"张三负责项目A"
         （参与≠负责不是矛盾；Vector 支持负责）                 → sufficient → Answer

关键语义：
  insufficient：证据不足（缺正向）
  conflict：正反证据并存
  绝不能把 insufficient 当 conflict，也不能因 relation != expected 生成 negative。

运行：python tests/test_conflict_resolver.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.evidence_validator import EvidenceValidator
from src.agents.conflict_resolver import ConflictResolver

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


def graph_edge(entity, relation, target, conf=0.9, **kw):
    e = {
        "doc_id": f"graph:{entity}", "chunk_index": 0,
        "text": f"{entity} {relation} {target}",
        "entity": entity, "relation": relation, "target": target,
        "confidence": conf, "source": "neo4j", "hop": 1,
    }
    e.update(kw)
    return e


def doc(text, **kw):
    e = {"doc_id": "d1", "text": text}
    e.update(kw)
    return e


def test_case1_no_conflict():
    """Graph 负责 + Vector 正向负责 → sufficient，无冲突"""
    print("\n[Case 1] Graph'负责' + Vector'张三负责项目A' → sufficient / no conflict")
    v = EvidenceValidator()
    r = v.validate(
        "张三负责项目A吗？",
        [graph_edge("张三", "负责", "项目A")],
        [doc("张三负责项目A的日常运营")],
    )
    check("evidence_status=sufficient", r["evidence_status"] == "sufficient", str(r))
    check("next_action=answer", r["next_action"] == "answer")
    check("conflict_status=no_conflict", r.get("conflict_status") == "no_conflict")


def test_case2_resolved_oppose():
    """Graph 负责 + Vector 不再负责（有时间信号）→ conflict → resolved(oppose) → answer"""
    print("\n[Case 2] Graph'负责' + Vector'不再负责，目前由李四负责' → conflict → resolved(oppose)")
    v = EvidenceValidator()
    ev = v.validate(
        "张三负责项目A吗？",
        [graph_edge("张三", "负责", "项目A")],
        [doc("张三已经不再负责项目A，目前由李四负责，已于2025年交接完成")],
    )
    check("evidence_status=conflict", ev["evidence_status"] == "conflict", str(ev))
    check("conflict_status=detected", ev.get("conflict_status") == "detected")
    check("有 conflict_evidence 正反双方",
          len(ev.get("conflict_evidence", [])) == 2,
          str(ev.get("conflict_evidence")))

    resolver = ConflictResolver()
    r = resolver.resolve("张三负责项目A吗？",
                         ev.get("conflict_evidence", []),
                         ev.get("evidence_detail", []))
    check("conflict_status=resolved", r["conflict_status"] == "resolved", str(r))
    check("next_action=answer", r["next_action"] == "answer")
    check("winner_side=oppose（文本时间信号最新）", r["winner_side"] == "oppose",
          f"winner={r['winner_side']}")


def test_case3_unresolved_refuse():
    """Graph 负责 + Vector 不负责（无时间无来源）→ conflict → unresolved → refuse"""
    print("\n[Case 3] Graph'负责' + Vector'不负责'（无时间信息）→ conflict → unresolved → refuse")
    v = EvidenceValidator()
    ev = v.validate(
        "张三负责项目A吗？",
        [graph_edge("张三", "负责", "项目A")],
        [doc("张三不负责项目A。")],
    )
    check("evidence_status=conflict", ev["evidence_status"] == "conflict", str(ev))

    resolver = ConflictResolver()
    r = resolver.resolve("张三负责项目A吗？",
                         ev.get("conflict_evidence", []),
                         ev.get("evidence_detail", []))
    check("conflict_status=unresolved", r["conflict_status"] == "unresolved", str(r))
    check("next_action=refuse", r["next_action"] == "refuse")
    check("winner_side=None（不强行选择）", r["winner_side"] is None)


def test_case4_participate_vs_responsible():
    """Graph 参与 + Vector 负责 → 不是 conflict，是 sufficient（Vector 支持负责）"""
    print("\n[Case 4] Graph'参与' + Vector'张三负责项目A' → sufficient（参与≠负责不是矛盾）")
    v = EvidenceValidator()
    r = v.validate(
        "张三负责项目A吗？",
        [graph_edge("张三", "参与", "项目A")],
        [doc("张三负责项目A的开发与上线")],
    )
    check("evidence_status=sufficient（Vector 支持负责）",
          r["evidence_status"] == "sufficient", str(r))
    check("next_action=answer", r["next_action"] == "answer")
    check("不是 conflict", r.get("conflict_status") != "conflict")


def test_insufficient_vs_conflict():
    """核心区分：insufficient ≠ conflict"""
    print("\n[区分] insufficient（证据不足）≠ conflict（正反证据并存）")
    v = EvidenceValidator()

    # insufficient：只有"参与"，无任何"负责"证据（graph + text 都没有）
    r1 = v.validate(
        "张三负责哪些项目？",
        [graph_edge("张三", "参与", "项目A")],
        [doc("张三参与了项目A的开发工作")],
    )
    check("只有参与 → insufficient（非 conflict）",
          r1["evidence_status"] == "insufficient" and r1.get("conflict_status") != "conflict",
          str(r1))

    # conflict：graph"负责" + text"不再负责"
    r2 = v.validate(
        "张三负责哪些项目？",
        [graph_edge("张三", "负责", "项目B")],
        [doc("张三不再负责项目B")],
    )
    check("负责 vs 不再负责 → conflict",
          r2["evidence_status"] == "conflict", str(r2))

    # 无任何证据（graph 空 + text 空）→ insufficient
    r3 = v.validate("张三负责哪些项目？", [], [])
    check("无证据 → insufficient", r3["evidence_status"] == "insufficient", str(r3))


def test_no_false_conflict_on_unrelated_negation():
    """无关否定不误判：'张三没有参与项目C' 不是对'负责项目A'的冲突"""
    print("\n[防误判] 无关否定不产生 conflict")
    v = EvidenceValidator()
    r = v.validate(
        "张三负责项目A吗？",
        [graph_edge("张三", "负责", "项目A")],
        [doc("张三没有参与项目C的开发。")],   # 否定"参与项目C"，与"负责项目A"无关
    )
    check("无关否定 → sufficient（不误判 conflict）",
          r["evidence_status"] == "sufficient", str(r))


if __name__ == "__main__":
    print("=" * 60)
    print("Conflict Resolver 单元测试")
    print("=" * 60)
    test_case1_no_conflict()
    test_case2_resolved_oppose()
    test_case3_unresolved_refuse()
    test_case4_participate_vs_responsible()
    test_insufficient_vs_conflict()
    test_no_false_conflict_on_unrelated_negation()
    print("\n" + "=" * 60)
    print(f"结果: {PASS} 通过, {FAIL} 失败")
    print("=" * 60)
    sys.exit(1 if FAIL else 0)
