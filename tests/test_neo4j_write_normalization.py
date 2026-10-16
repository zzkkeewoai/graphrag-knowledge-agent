"""tests/test_neo4j_write_normalization.py —— 图谱写入路径的实体名规范化

为什么单独测这一层：
    `entity_normalizer` 的单元测试只证明"规范化函数本身是对的"，
    **不证明数据库层真的调用了它**。而"策略写好了但某条路径忘了用"
    恰恰是这类改动最容易留下的洞（写路径有单条和批量两条）。

    所以这里直接测 `Neo4jClient._entity_row` —— 它是两条写路径**共用**的
    参数构造入口，而且不需要连 Neo4j 就能测。

运行：python tests/test_neo4j_write_normalization.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.neo4j_client import Neo4jClient  # noqa: E402
from src.schemas.graph_schema import (  # noqa: E402
    EntityType,
    RelationType,
    Triple,
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


def section(title):
    print(f"\n{'=' * 62}\n  {title}\n{'=' * 62}")


def make_triple(head, tail="项目B"):
    return Triple(
        head=head,
        head_type=EntityType.PROJECT,
        relation=RelationType.DEPENDS_ON,
        tail=tail,
        tail_type=EntityType.PROJECT,
        confidence=0.9,
        source_chunk="测试用原文片段",
    )


def main():
    global PASS, FAIL

    section("① 写入参数里的 name 必须是规范化后的键")
    row = Neo4jClient._entity_row(make_triple("  项目A  "))
    check("去掉首尾空格", row["head"] == "项目A", f"实际 {row['head']!r}")
    check("同时保留原始写法到 display", row["head_display"] == "  项目A  ",
          f"实际 {row['head_display']!r}")

    row = Neo4jClient._entity_row(make_triple("Ａ项目"))
    check("全角 A 被归一", row["head"] == "A项目", f"实际 {row['head']!r}")

    row = Neo4jClient._entity_row(make_triple("Python"))
    check("ASCII 大小写归一", row["head"] == "python", f"实际 {row['head']!r}")

    section("② 同一实体的多种写法 → 同一个写入键（这才是修复的目的）")
    # 注意：全角变体必须是"Ａ在末尾"（项目Ａ）——「Ａ项目」是全角 A 在**开头**，
    # 它是另一个实体，不能混进这组。
    variants = ["项目A", "项目A ", " 项目A", "项目 A", "项目Ａ", "项目A\u200b"]
    keys = {Neo4jClient._entity_row(make_triple(v))["head"] for v in variants}
    check(f"{len(variants)} 种写法只产生 1 个键", len(keys) == 1, f"实际 {keys}")

    section("②b 反向验证 —— 位置不同的 A 是两个实体")
    k_head = Neo4jClient._entity_row(make_triple("Ａ项目"))["head"]
    k_tail = Neo4jClient._entity_row(make_triple("项目Ａ"))["head"]
    check("「Ａ项目」≠「项目Ａ」", k_head != k_tail, f"{k_head!r} vs {k_tail!r}")

    section("③ 反向验证 —— 不同实体必须得到不同的键")
    k1 = Neo4jClient._entity_row(make_triple("项目A"))["head"]
    k2 = Neo4jClient._entity_row(make_triple("项目B"))["head"]
    check("项目A ≠ 项目B", k1 != k2, f"{k1!r} vs {k2!r}")

    section("④ 参数完整性 —— 不能因为改了参数构造就漏字段")
    row = Neo4jClient._entity_row(make_triple("项目A"))
    expected = {
        "head", "head_display", "head_type",
        "tail", "tail_display", "tail_type",
        "relation_type", "confidence",
    }
    check("字段集合完全匹配 Cypher 里用到的参数",
          set(row.keys()) == expected,
          f"缺: {expected - set(row.keys())}  多: {set(row.keys()) - expected}")
    check("head_type 是枚举值字符串", row["head_type"] == EntityType.PROJECT.value)
    check("relation_type 是枚举值字符串", row["relation_type"] == RelationType.DEPENDS_ON.value)
    check("confidence 原样透传", row["confidence"] == 0.9)

    section("⑤ 单条写路径的 Cypher 必须用上规范化后的键和 display")
    import inspect
    src = inspect.getsource(Neo4jClient.write_triple)
    check("write_triple 用 $head 而不是原名字", "MERGE (h:Entity {name: $head})" in src)
    check("write_triple 写入了 display_name", "display_name" in src)
    check("write_triple 通过 _entity_row 构造参数", "_entity_row" in src)

    src_batch = inspect.getsource(Neo4jClient.batch_write)
    check("batch_write 也走 _entity_row（两条路径共用）", "_entity_row" in src_batch)
    check("batch_write 也写入了 display_name", "display_name" in src_batch)

    section("⑥ 非法实体名在写入路径必须 fail-fast（不能静默写脏数据）")
    for bad in ["", "   ", "\u200b"]:
        try:
            Neo4jClient._entity_row(make_triple(bad))
            check(f"{bad!r} → 应该报错", False, "居然没报错")
        except ValueError:
            check(f"{bad!r} → ValueError", True)

    print(f"\n{'=' * 62}")
    print(f"  结果: {PASS} 通过 / {FAIL} 失败")
    print(f"{'=' * 62}")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
