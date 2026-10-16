"""tests/check_entity_normalization.py —— 检查已有图谱里的实体名是否已规范化

【为什么必须有这个脚本】
    实体名规范化是在**写入路径**上做的（见 Neo4jClient._entity_row）。
    这意味着：**在这次改动之前写进去的节点，名字是没规范化的**。

    后果很隐蔽：
      · 老节点  name="  项目A  "（带空格）
      · 新查询  name="项目A"（规范化后）
      → 查询命中不了老节点，表现为"数据明明在库里却查不到"

    所以升级之后**必须跑一次这个脚本**，看有没有存量脏数据。

【怎么处理存量数据】
    本项目是演示库，最干净的做法是**清库重灌**：
        # 1. 清掉导入的演示数据（或直接删掉 Neo4j 里的 Entity 节点）
        # 2. 重新灌入 —— 这次写入路径会做规范化
        python tests/seed_demo_data.py

    生产环境不能清库，那就要做真正的迁移（重命名 + 合并节点），
    那是另一套工作，不在本项目范围内 —— 这里只**如实报告**，不假装能自动搞定。

【运行】
    python tests/check_entity_normalization.py

    需要 Neo4j 在运行。连不上时会明确报错并返回非零码，不会静默通过。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.entity_normalizer import (  # noqa: E402
    find_duplicate_groups,
    find_similar_candidates,
    is_normalized,
    normalize_entity_name,
)


def fetch_entity_names():
    """从 Neo4j 读出所有实体名。

    连不上库就抛错 —— 不能返回空列表，否则脚本会"看起来跑通了但其实什么都没查"。
    """
    from src.config import Config
    from src.database.neo4j_client import Neo4jClient

    client = Neo4jClient(Config.NEO4J_URI, Config.NEO4J_USER, Config.NEO4J_PASSWORD)
    client.connect()
    try:
        with client.driver.session() as session:
            result = session.run("MATCH (e:Entity) RETURN e.name AS name")
            return [record["name"] for record in result]
    finally:
        client.close()


def main() -> int:
    try:
        names = fetch_entity_names()
    except Exception as e:  # noqa: BLE001 - 这里要的是"明确失败并说明原因"
        print(f"[FAIL] 读取图谱失败：{type(e).__name__}: {e}")
        print("       请确认 Neo4j 已启动、.env 里的连接配置正确。")
        return 1

    print(f"图谱中共有 {len(names)} 个实体节点\n")

    # ---------- ① 未规范化的节点 ----------
    dirty = [n for n in names if not is_normalized(n)]
    print("=" * 62)
    print("① 未规范化的节点名（说明是本次改动之前写入的）")
    print("=" * 62)
    if dirty:
        print(f"  发现 {len(dirty)} 个：")
        for n in dirty[:30]:
            print(f"    {n!r}  →  应为 {normalize_entity_name(n)!r}")
        if len(dirty) > 30:
            print(f"    ... 还有 {len(dirty) - 30} 个")
        print("\n  ⚠️ 这些节点会**查不到**（查询用的是规范键）。")
        print("     演示库建议清库重灌：python tests/seed_demo_data.py")
    else:
        print("  无 —— 库里所有实体名都已是规范形式 ✓")

    # ---------- ② 规范化后会重复的节点 ----------
    print()
    groups = find_duplicate_groups(names)
    print("=" * 62)
    print("② 规范化后会撞到一起的节点（这些是真重复）")
    print("=" * 62)
    if groups:
        print(f"  发现 {len(groups)} 组：")
        for key, raws in list(groups.items())[:20]:
            print(f"    {key}  <=  {raws}")
        print(f"\n  统计：这 {len(groups)} 组共涉及 "
              f"{sum(len(v) for v in groups.values())} 个节点，"
              f"规范化后应收敛为 {len(groups)} 个。")
    else:
        print("  无 —— 没有发现「同实体多写法」的情况 ✓")

    # ---------- ③ 疑似同义实体（只报告，不合并） ----------
    print()
    similar = find_similar_candidates(names)
    print("=" * 62)
    print("③ 疑似同义实体（**需人工确认，脚本不会自动合并**）")
    print("=" * 62)
    if similar:
        print(f"  发现 {len(similar)} 对候选：")
        for a, b, r in similar[:20]:
            print(f"    {a}  ~  {b}   (相似度 {r})")
        print("\n  ⚠️ 相似 ≠ 同一实体（「张三」和「张四」也很像）。")
        print("     自动合并的风险是把两个真实实体错误地并成一个，")
        print("     数据被污染且很难还原 —— 所以这里只出候选。")
    else:
        print("  没有发现可疑候选 ✓")

    print()
    print("=" * 62)
    if dirty or groups:
        print("  结论：存量数据需要处理（见上面 ① ②）")
        return 1
    print("  结论：存量数据健康，无需迁移")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
