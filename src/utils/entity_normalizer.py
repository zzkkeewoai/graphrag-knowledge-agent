"""实体名规范化 —— 让「同一个实体的不同写法」落到同一个图谱节点

【为什么需要它】
    图谱节点用**实体名**做唯一键（`MERGE (e:Entity {name: ...})`），
    这带来一个**静默失效**：同一个实体的不同写法会变成两个节点。

    四种典型情况：
      · 空白差异      "张三 " / " 张三" / "张  三"
      · 全半角差异    "Ａ项目"（全角 A） / "A项目"
      · 大小写差异    "Python" / "python"
      · 不可见字符    "张三\\u200b"（零宽空格，从网页复制粘贴带进来的）

    后果是**静默的召回损失**：它表现为"检索效果不太好"，
    不会报错、不会告警、日志里也看不出来 —— 你可能永远发现不了。

    这正是"失败模式"里最难防的一类：**损坏得悄无声息**。

【它解决什么、不解决什么（必须先划清界限）】
    解决：**格式层面的同一性** —— 空白 / 全半角 / 大小写 / 不可见字符
    不解决：**语义层面的同一性**——
        · 同义实体："张三" 和 "张伟老师" 是同一个人
        · 同名实体：两个不同的人恰好都叫"张三"（这个更危险：会被错误合并）
        → 那需要实体消歧（别名表 / 向量聚类 / 人工确认），**本项目未实现**。

    所以本模块的定位是：**把"确定能合并的"先合并掉**，
    而不是假装解决了实体消歧。`find_similar_candidates()` 只产出**候选报告**，
    交人工判断，绝不自动合并。

【一条必须记住的运维性质】
    规范化规则一旦上线，**它就成了数据契约**：
    改规则会让"旧规则写进去的节点"和"新规则写进去的节点"对不上，
    需要重跑建图。所以规则要保持稳定；确实要改，就得当作一次数据迁移来做。
"""
import unicodedata
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Tuple

# 判定"疑似同义实体"的相似度阈值。
# 之所以不算低（比如 0.8），是因为这个值越低误报越多 ——
# 而误报的代价是"人工要去看一堆无关候选"，很快就不看了。
DEFAULT_SIMILARITY_THRESHOLD = 0.86


def normalize_entity_name(raw: str) -> str:
    """把实体名规范化成**用于 MERGE 的稳定键**。

    处理顺序（顺序不能换，每一步都依赖前一步的干净输入）：

      1. NFKC 归一 —— 全角转半角、兼容字符统一（stdlib，无新依赖）
      2. **删掉所有空白**（首尾 / 中间 / 全角空格 / 制表符 / 换行）
      3. 删掉不可见字符（Unicode 类别 Cf/Cc，比如零宽空格）
      4. **仅当结果是纯 ASCII 时**才做大小写归一

    第 2 步为什么是"删掉"而不是"折叠成单个空格"：
        实体名是**短标签**（人名 / 项目名 / 技术栈），**内部空格不承载语义**——
        "张 三" 和 "张三"、"Spring Boot" 和 "SpringBoot" 都是同一个实体。
        删掉空白能一次覆盖"多个空格 / 空格 vs 制表符 / 全角空格"这些情况，
        而"折叠成单个空格"只能处理前一半（实测踩到过：`张  三` 归一后
        仍是 `张 三`，和 `张三` 对不上）。
        代价：如果有"内部空格有语义"的实体名（严格区分的产品型号等），
        这条会误合并 —— 那种场景需要把这一步去掉。本项目的实体都是
        上面的三类短标签，所以取这个更宽松的规则。

    为什么第 5 步只对纯 ASCII 做：
        对中文来说大小写归一是空操作，加了没坏处；
        但对其他语言可能产生意外映射（土耳其语 İ、德语 ß 等）。
        与其引入一个"说不清会怎么变"的规则，不如**保守地只处理确定的情况**。
        代价是 "Ä" 和 "ä" 不会被合并 —— 这是已知局限，写在上面了。

    Args:
        raw: 原始实体名

    Returns:
        规范化后的键（非空）

    Raises:
        TypeError: 传入的不是字符串
        ValueError: 规范化后为空 —— **必须拒绝**，
            因为空字符串会让所有"空名字"的实体合并到同一个节点上，
            那是比"不合并"严重得多的错误。
    """
    if not isinstance(raw, str):
        raise TypeError(f"实体名必须是字符串，实际收到 {type(raw).__name__}: {raw!r}")

    s = unicodedata.normalize("NFKC", raw)
    # 删掉所有空白：这样 "张  三" / "张\t三" / "张\u3000三" / " 张三 " 都落到 "张三"
    s = "".join(ch for ch in s if not ch.isspace())
    # 再删不可见字符（零宽空格、BOM 等）—— 它们 isspace() 为 False，必须单独处理
    s = "".join(ch for ch in s if unicodedata.category(ch) not in ("Cf", "Cc"))
    if s.isascii():
        s = s.casefold()

    if not s:
        raise ValueError(f"实体名规范化后为空，拒绝入库: {raw!r}")
    return s


def is_normalized(raw: str) -> bool:
    """判断一个实体名本身是否已经是规范形式。

    用途：检查上游（抽取器 / 演示数据 / 人工录入）有没有漏掉规范化这一步。
    """
    try:
        return normalize_entity_name(raw) == raw
    except (TypeError, ValueError):
        return False


def find_duplicate_groups(names: Iterable[str]) -> Dict[str, List[str]]:
    """找出「多个不同原始写法 → 同一个规范键」的组。

    这些是**确定需要合并**的 —— 规范化之后它们本来就是同一个实体。

    Returns:
        {规范键: [原始写法, ...]}，只包含原始写法多于一种的组
    """
    buckets: Dict[str, set] = {}
    for raw in names:
        try:
            key = normalize_entity_name(raw)
        except (TypeError, ValueError):
            continue  # 非法名字单独用 is_normalized 检查，这里只做分组
        buckets.setdefault(key, set()).add(raw)
    return {k: sorted(v) for k, v in buckets.items() if len(v) > 1}


def find_similar_candidates(
    names: Iterable[str],
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    limit: int = 50,
) -> List[Tuple[str, str, float]]:
    """找出「写法相近、但规范化后仍然不同」的实体对 —— 疑似同义实体。

    ⚠️ 这是**候选报告，不是自动合并**。
        字符串相似不等于同一实体：「张三」和「张四」也很像，但可能是两个人。
        自动合并同义实体的风险是**把两个真实实体错误地并成一个**，
        那比"没合并"严重得多（数据被污染，而且很难还原）。

    Args:
        names: 实体名集合
        threshold: 相似度阈值（默认 0.86）
        limit: 最多返回多少对

    Returns:
        [(实体A, 实体B, 相似度), ...]，按相似度降序
    """
    uniq = sorted({normalize_entity_name(n) for n in names
                   if isinstance(n, str) and n.strip()})
    out: List[Tuple[str, str, float]] = []
    for i, a in enumerate(uniq):
        for b in uniq[i + 1:]:
            ratio = SequenceMatcher(None, a, b).ratio()
            # 包含关系也算候选：「张三」与「张三老师」相似度只有 0.8，
            # 但短名被长名完整包含，是很强的同义信号。
            contained = (len(a) >= 2 and a in b) or (len(b) >= 2 and b in a)
            if ratio >= threshold or contained:
                out.append((a, b, round(ratio, 3)))
    out.sort(key=lambda x: (-x[2], x[0], x[1]))
    return out[:limit]


def report(names: Iterable[str]) -> str:
    """把上面两个检查合成一段可读报告（给排查脚本用）。"""
    names = list(names)
    dupes = find_duplicate_groups(names)
    similar = find_similar_candidates(names)

    lines = [f"实体名规范化检查：共 {len(names)} 个实体名"]
    lines.append(f"  确定重复（规范化后相同，必须合并）：{len(dupes)} 组")
    for key, raws in list(dupes.items())[:20]:
        lines.append(f"    {key}  <=  {raws}")
    lines.append(f"  疑似同义（需人工确认，不自动合并）：{len(similar)} 对")
    for a, b, r in similar[:20]:
        lines.append(f"    {a}  ~  {b}   (相似度 {r})")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    demo = [
        "张三", "张三 ", " 张三", "张  三", "张三\u200b",
        "Python", "python", "PYTHON",
        "Ａ项目", "A项目",
        "张三老师",
        "项目A", "项目B",
    ]
    print(report(demo))
