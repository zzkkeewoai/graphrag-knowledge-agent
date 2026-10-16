"""tests/test_entity_normalizer.py —— 实体名规范化单元测试

覆盖四类"静默失效"的格式差异，加三条**反向验证**（防误合并）。

为什么必须做反向验证：
    一个"把什么都合并成一个"的规范化函数，能让所有正例都通过 ——
    但它比不做规范化更糟。所以这里必须验证「不同实体保持不同」。

运行：python tests/test_entity_normalizer.py
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


def main():
    global PASS, FAIL

    # ------------------------------------------------------------
    section("① 空白差异 —— 同一实体必须归一到同一个键")
    for raw in ["张三", "张三 ", " 张三", "张 三", "张  三", "张\t三", "张\n三", "  张三  "]:
        check(f"{raw!r} → 张三", normalize_entity_name(raw) == "张三")

    # ------------------------------------------------------------
    section("② 全半角差异（NFKC）")
    check("全角Ａ项目 → A项目", normalize_entity_name("Ａ项目") == "A项目")
    check("全角数字１２３ → 123", normalize_entity_name("１２３") == "123")
    check("全角空格也算空白（U+3000）", normalize_entity_name("张\u3000三") == "张三")

    # ------------------------------------------------------------
    section("③ 大小写差异（仅纯 ASCII 归一）")
    check("Python → python", normalize_entity_name("Python") == "python")
    check("PYTHON → python", normalize_entity_name("PYTHON") == "python")
    check("三种写法同一个键",
          len({normalize_entity_name(x) for x in ["Python", "python", "PYTHON"]}) == 1)
    check("混合大小写也归一", normalize_entity_name("PyTorch") == "pytorch")

    # ------------------------------------------------------------
    section("④ 不可见字符（从网页复制粘贴带进来的）")
    check("零宽空格被删除", normalize_entity_name("张三\u200b") == "张三")
    check("零宽连接符被删除", normalize_entity_name("张\u200d三") == "张三")
    check("BOM 被删除", normalize_entity_name("\ufeff张三") == "张三")

    # ------------------------------------------------------------
    section("⑤ 反向验证 —— 不同的实体绝不能被合并（否则比不做更糟）")
    check("项目A ≠ 项目B", normalize_entity_name("项目A") != normalize_entity_name("项目B"))
    check("张三 ≠ 李四", normalize_entity_name("张三") != normalize_entity_name("李四"))
    check("张三 ≠ 张三丰", normalize_entity_name("张三") != normalize_entity_name("张三丰"))
    check("非 ASCII 不做大小写归一（已知局限，保守选择）",
          normalize_entity_name("Ä") != normalize_entity_name("ä"),
          "这是刻意的：只对纯 ASCII 做 casefold，避免其他语言的意外映射")

    # ------------------------------------------------------------
    section("⑥ 幂等性 —— 规范化两次必须等于一次")
    samples = ["张三 ", "Ａ项目", "Python", "张三\u200b", "张  三", "PyTorch"]
    for s in samples:
        once = normalize_entity_name(s)
        twice = normalize_entity_name(once)
        check(f"normalize(normalize({s!r})) == normalize({s!r})", once == twice,
              f"{once!r} vs {twice!r}")

    # ------------------------------------------------------------
    section("⑦ 非法输入必须 fail-fast，不能静默放过")
    try:
        normalize_entity_name("")
        check("空字符串 → ValueError", False, "居然没报错")
    except ValueError:
        check("空字符串 → ValueError", True)

    try:
        normalize_entity_name("   ")
        check("纯空白 → ValueError", False, "居然没报错")
    except ValueError:
        check("纯空白 → ValueError", True)

    try:
        normalize_entity_name("\u200b")
        check("纯零宽字符 → ValueError", False, "居然没报错")
    except ValueError:
        check("纯零宽字符 → ValueError", True)

    try:
        normalize_entity_name(None)
        check("None → TypeError", False, "居然没报错")
    except TypeError:
        check("None → TypeError", True)

    try:
        normalize_entity_name(123)
        check("数字 → TypeError", False, "居然没报错")
    except TypeError:
        check("数字 → TypeError", True)

    # 为什么要拒绝空名：空字符串会让所有"空名字"的实体合并到同一个节点上，
    # 那比"不合并"严重得多 —— 数据被污染，且很难还原。

    # ------------------------------------------------------------
    section("⑧ is_normalized —— 用来检查上游有没有漏掉这一步")
    check("已规范化的名字返回 True", is_normalized("张三") is True)
    check("带空格的名字返回 False", is_normalized("张三 ") is False)
    check("全角名字返回 False", is_normalized("Ａ项目") is False)
    check("非法输入不抛错，返回 False", is_normalized(None) is False)

    # ------------------------------------------------------------
    section("⑨ find_duplicate_groups —— 找出「确定要合并」的组")
    names = ["张三", "张三 ", "张三\u200b", "李四", "李四", "项目A"]
    groups = find_duplicate_groups(names)
    check("找出 1 组重复", len(groups) == 1, f"实际 {groups}")
    check("重复组里包含 3 种原始写法",
          len(groups.get("张三", [])) == 3, f"实际 {groups.get('张三')}")
    check("李四只有一种写法 → 不算重复", "李四" not in groups)
    check("完全不同的名字不产生重复组",
          find_duplicate_groups(["A", "B", "C"]) == {})

    # ------------------------------------------------------------
    section("⑩ find_similar_candidates —— 只出候选，绝不自动合并")
    cands = find_similar_candidates(["张三", "张三老师", "李四", "完全无关的名字"])
    pairs = {(a, b) for a, b, _ in cands}
    check("「张三 / 张三老师」被判为候选",
          ("张三", "张三老师") in pairs, f"实际 {pairs}")
    check("返回的是三元组（含相似度）",
          all(len(t) == 3 for t in cands), f"实际 {cands}")
    check("无关实体不进入候选",
          not any("李四" in p and "完全无关" in p for p in
                  {(a, b) for a, b, _ in cands}))
    # 注意：候选不等于合并 —— 自动合并同义实体的风险是
    # 把两个真实实体错误并成一个（数据被污染且难还原），所以这里只报告。

    # ------------------------------------------------------------
    print(f"\n{'=' * 62}")
    print(f"  结果: {PASS} 通过 / {FAIL} 失败")
    print(f"{'=' * 62}")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
