"""tests/try_extract.py —— 用 3 篇小文档试抽，看新实体/关系类型能不能被抽出来。

【为什么不直接灌库】
抽取质量要在**入库之前**看。先灌进 Neo4j 再看，你会分不清"抽错了"还是"写错了"，
而且清库成本高。这个脚本只调抽取器、不碰数据库，可以反复跑。

【为什么把 confidence_threshold 设成 0】
默认 0.7，会把低置信度的三元组直接过滤掉。第一次试抽要看**全貌**：
如果模型抽出来了但置信度低，说明它"隐约知道"，那是调 prompt 能救的；
如果根本抽不出来，说明它"完全没这个概念"，那是 schema 或示例的问题。
先看全貌，再决定阈值调多少。

【怎么判断通过】
每篇文档都写了一个"期望三元组"，脚本会按 关系 + 头实体 + 尾实体 去匹配。
命中 0/3：新类型完全没被用上 → 必须改 few-shot 示例
命中 1-2/3：部分被用上 → 补示例或补类型说明
命中 3/3：schema + 示例都够用，可以开始造语料

运行：
    python tests/try_extract.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from src.config import Config  # noqa: E402
from src.extractors.triple_extractor import TripleExtractor  # noqa: E402

# ============================================================
# 三篇试抽文档：每篇只考一件事，避免"错在哪都看不出来"
# ============================================================
DOCS = {
    # 考「角色 —审批→ 流程」：新增的 角色 / 流程 / 审批
    "doc1_审批": (
        "《采购付款审批流程》第 3.2 条规定：单笔金额达到 5 万元及以上的采购付款，"
        "须经部门负责人与财务总监二级审批；5 万元以下由部门负责人审批。"
    ),
    # 考「条款 —截止于→ 时限」：新增的 条款 / 时限 / 截止于
    "doc2_时限": (
        "《发票管理规定》第 4.2 条：跨年发票须在次年 3 月 31 日前完成抵扣，"
        "逾期不予受理。"
    ),
    # 考「新版 —替代→ 旧版」：新增的 替代
    "doc3_版本": (
        "《发票管理规定》（LH-FIN-2026-002）自 2026 年 1 月 1 日起施行，"
        "原《发票管理规定》（LH-FIN-2025-002）同时废止。"
    ),
}

# 每篇期望抽出的三元组：用 (关系, 头实体包含, 尾实体包含) 做匹配
EXPECTED = {
    "doc1_审批": [("审批", "财务总监", "采购付款")],
    "doc2_时限": [("截止于", "发票管理规定", "3 月 31")],
    "doc3_版本": [("替代", "2026", "2025")],
}

NEW_ENTITIES = {"条款", "阈值", "时限", "流程", "角色", "组织"}
NEW_RELATIONS = {"审批", "适用于", "截止于", "依据", "替代"}


def norm(s: str) -> str:
    """去掉空白再比较，避免"3 月 31"和"3月31"对不上。"""
    return "".join((s or "").split())


def match(triples, relation: str, head_kw: str, tail_kw: str) -> bool:
    """期望三元组是否被抽出（关系精确、头尾用包含匹配）。"""
    for t in triples:
        if t.relation.value != relation:
            continue
        if norm(head_kw) in norm(t.head) and norm(tail_kw) in norm(t.tail):
            return True
        # 反向也认：有些模型会把头尾写反，先给一个宽松判定
        if norm(head_kw) in norm(t.tail) and norm(tail_kw) in norm(t.head):
            return True
    return False


def main() -> int:
    api_key = Config.DEEPSEEK_API_KEY
    if not api_key:
        print("[FAIL] .env 里没有 DEEPSEEK_API_KEY")
        return 1

    extractor = TripleExtractor(
        api_key=api_key,
        base_url=Config.DEEPSEEK_BASE_URL,
        model_name=Config.DEEPSEEK_MODEL,
        temperature=0.1,
        confidence_threshold=0.0,  # 先关掉阈值，看全貌
    )

    total_expected = 0
    total_hit = 0
    new_entity_hits = 0
    new_relation_hits = 0

    for doc_id, text in DOCS.items():
        print("=" * 66)
        print(f"【{doc_id}】{text[:40]}…")
        print("=" * 66)

        result = extractor.extract(text, chunk_id=doc_id)
        triples = result.triples

        if not triples:
            print("  （一条三元组都没抽出来）")

        for t in triples:
            mark_e = "★" if t.head_type.value in NEW_ENTITIES or t.tail_type.value in NEW_ENTITIES else " "
            mark_r = "★" if t.relation.value in NEW_RELATIONS else " "
            print(
                f"  {mark_e}{mark_r} {t.head}({t.head_type.value})"
                f" —{t.relation.value}→ {t.tail}({t.tail_type.value})"
                f"  conf={t.confidence:.2f}"
            )
            if t.head_type.value in NEW_ENTITIES or t.tail_type.value in NEW_ENTITIES:
                new_entity_hits += 1
            if t.relation.value in NEW_RELATIONS:
                new_relation_hits += 1

        for relation, head_kw, tail_kw in EXPECTED[doc_id]:
            total_expected += 1
            hit = match(triples, relation, head_kw, tail_kw)
            total_hit += 1 if hit else 0
            print(
                f"\n  期望：{head_kw} —{relation}→ {tail_kw}    "
                f"{'✓ 命中' if hit else '✗ 未命中'}"
            )
        print()

    print("=" * 66)
    print(f"期望三元组命中：{total_hit}/{total_expected}")
    print(f"用到新实体类型的三元组：{new_entity_hits} 条")
    print(f"用到新关系类型的三元组：{new_relation_hits} 条")
    print("=" * 66)

    if total_hit == total_expected:
        print("结论：新类型可被抽出，可以开始造语料。")
    elif total_hit == 0:
        print("结论：新类型完全没被用上 —— 大概率是 few-shot 示例全是项目域，")
        print("      模型只会往旧类型上靠。下一步：补一个企业域示例。")
    else:
        print("结论：部分命中 —— 补示例或加强类型说明后再试。")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
