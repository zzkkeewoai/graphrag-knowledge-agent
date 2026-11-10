"""tests/check_graph.py —— 直接查 Neo4j，验证图建得对不对（不用 LLM、不用 Milvus）。

为什么要单独做这个检查：
    端到端问答一旦答错，原因可能是 ①图建错了 ②检索没走到 ③生成编了。
    三者混在一起非常难查。先把 ① 单独验证掉，剩下就只有两种可能。

它回答四个问题：
    1. 各类实体有多少个？（某类为 0 = 抽取漏了）
    2. 各类关系有多少条？
    3. 「第2.2条」这个再化节点，是不是同时连着阈值/角色/流程？
    4. 「采购付款」的两跳邻居能不能走到审批人？

用法：
    python tests/check_graph.py
    python tests/check_graph.py --entity 第2.2条
"""

import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv()

from src.config import Config  # noqa: E402
from src.database.neo4j_client import Neo4jClient  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default="第2.2条", help="要看邻居的实体名")
    args = parser.parse_args()

    client = Neo4jClient(
        uri=Config.NEO4J_URI, user=Config.NEO4J_USER, password=Config.NEO4J_PASSWORD
    )
    client.connect()

    with client.driver.session() as session:
        # ---------- 1. 实体分布 ----------
        print("=" * 66)
        print("① 实体类型分布")
        print("=" * 66)
        rows = session.run(
            "MATCH (n:Entity) RETURN n.type AS t, n.display_name AS dn, n.name AS name"
        )
        types = Counter()
        names = []
        for r in rows:
            types[r["t"]] += 1
            names.append(r["dn"] or r["name"])
        for t, c in types.most_common():
            print(f"  {t or '(无类型)':<8}{c:>4} 个")

        total_nodes = sum(types.values())
        print(f"  {'合计':<8}{total_nodes:>4} 个")

        # ---------- 2. 关系分布 ----------
        print("\n" + "=" * 66)
        print("② 关系类型分布")
        print("=" * 66)
        rows = session.run(
            "MATCH ()-[r:RELATES_TO]->() RETURN r.relation_type AS rt"
        )
        rels = Counter(r["rt"] for r in rows)
        for t, c in rels.most_common():
            print(f"  {t or '(无类型)':<8}{c:>4} 条")
        print(f"  {'合计':<8}{sum(rels.values()):>4} 条")

        # ---------- 3. 再化节点检查 ----------
        print("\n" + "=" * 66)
        print(f"③ 再化节点检查：{args.entity}")
        print("=" * 66)
        rows = session.run(
            """
            MATCH (c:Entity {name: $name})-[r:RELATES_TO]-(o:Entity)
            RETURN r.relation_type AS rel, o.name AS other, o.type AS otype
            """,
            name=args.entity,
        )
        found = list(rows)
        if not found:
            print(f"  没找到「{args.entity}」这个节点。")
            print("  可能原因：抽取时条款名写法不同（比如带了前缀），或者它没被抽出来。")
            print(f"  库里现有名字样例：{', '.join(names[:12])}")
        else:
            for r in found:
                print(f"  {args.entity} —{r['rel']}→ {r['other']}({r['otype']})")
            kinds = {r["otype"] for r in found}
            expect = {"阈值", "角色", "流程"}
            if expect <= kinds:
                print(f"\n  ✅ 阈值 / 角色 / 流程 都连上了 —— 再化成功，条件配对没丢。")
            else:
                print(f"\n  ⚠️ 缺少类型：{expect - kinds} —— 条件可能没挂全。")

        # ---------- 4. 两跳可达性 ----------
        print("\n" + "=" * 66)
        print("④ 两跳可达性：采购付款 → 条款 → 审批人")
        print("=" * 66)
        rows = session.run(
            """
            MATCH (p:Entity {name: '采购付款'})-[r1:RELATES_TO]-(c:Entity {type: '条款'})
                  -[r2:RELATES_TO]->(role:Entity {type: '角色'})
            RETURN DISTINCT c.name AS clause, role.name AS approver
            """
        )
        hops = list(rows)
        if not hops:
            print("  走不通。可能 采购付款 没连到条款，或者条款没连到角色。")
        else:
            for h in hops:
                print(f"  采购付款 → {h['clause']} → {h['approver']}")
            approvers = sorted({h["approver"] for h in hops})
            print(f"\n  可达审批人：{', '.join(approvers)}")

    client.close()
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
