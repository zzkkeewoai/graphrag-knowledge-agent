"""tests/try_extract_file.py —— 对任意一篇文档做试抽，看抽出了什么类型的实体/关系。

为什么要有这个脚本：
    tests/try_extract.py 里只硬编码了 3 篇试抽样例，改一篇就要改代码。
    造语料阶段要反复抽不同文件，所以单独做一个「传文件路径就能跑」的版本。

它回答三个问题：
    1. 这篇文档抽出了多少三元组？（太少 = 句子太绕或实体太稀）
    2. 类型分布是什么？（"概念"占比过高 = 该标的没标出来）
    3. 有没有明显抽错的？（头尾反了、关系乱配）

用法：
    python tests/try_extract_file.py data/simco/org.md
    python tests/try_extract_file.py data/simco/org.md --chunk paragraph
"""

import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from src.config import Config  # noqa: E402
from src.extractors.triple_extractor import TripleExtractor  # noqa: E402

NEW_ENTITIES = {"条款", "阈值", "时限", "流程", "角色", "组织"}
NEW_RELATIONS = {"审批", "适用于", "截止于", "依据", "替代"}


def split_paragraphs(text: str) -> list[str]:
    """按空行切段，并丢掉纯标题行。

    为什么不按句子切：中文句子边界靠标点，容易切碎（"第 3.2 条：" 会被切成两半）。
    按段落切更稳，代价是一段里可能有多个关系——但试抽阶段要看的是"能不能抽出来"，
    不是"抽得干不干净"。
    """
    blocks = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        # 纯标题（以 # 开头且只有一行）没有实体，抽它浪费时间
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if len(lines) == 1 and lines[0].lstrip().startswith("#"):
            continue
        blocks.append(block)
    return blocks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="要试抽的文档路径（.md / .txt）")
    parser.add_argument("--limit", type=int, default=0, help="只抽前 N 段（0 = 全部）")
    args = parser.parse_args()

    if not os.path.exists(args.path):
        print(f"[FAIL] 找不到文件：{args.path}")
        return 1

    api_key = Config.DEEPSEEK_API_KEY
    if not api_key:
        print("[FAIL] .env 里没有 DEEPSEEK_API_KEY")
        return 1

    with open(args.path, encoding="utf-8") as f:
        text = f.read()

    blocks = split_paragraphs(text)
    if args.limit:
        blocks = blocks[: args.limit]

    print(f"文件：{args.path}")
    print(f"段落数：{len(blocks)}（已跳过纯标题）")
    print("=" * 70)

    extractor = TripleExtractor(
        api_key=api_key,
        base_url=Config.DEEPSEEK_BASE_URL,
        model_name=Config.DEEPSEEK_MODEL,
        temperature=0.1,
        confidence_threshold=0.0,
    )

    all_triples = []
    for i, block in enumerate(blocks, 1):
        preview = block.replace("\n", " ")[:46]
        print(f"\n[{i}/{len(blocks)}] {preview}…")
        try:
            result = extractor.extract(block, chunk_id=f"p{i}")
        except Exception as e:  # noqa: BLE001
            print(f"  [抽取失败] {type(e).__name__}: {e}")
            continue

        if not result.triples:
            print("  （无三元组）")
        for t in result.triples:
            all_triples.append(t)
            mark = "★" if (t.head_type.value in NEW_ENTITIES or t.tail_type.value in NEW_ENTITIES) else " "
            print(
                f"  {mark} {t.head}({t.head_type.value})"
                f" —{t.relation.value}→ {t.tail}({t.tail_type.value})"
                f"  {t.confidence:.2f}"
            )

    # ---------------- 汇总 ----------------
    print("\n" + "=" * 70)
    print(f"合计三元组：{len(all_triples)} 条")

    if not all_triples:
        print("一条都没抽出来 —— 先检查 .env 的 key 和网络。")
        return 0

    ent = Counter()
    rel = Counter()
    for t in all_triples:
        ent[t.head_type.value] += 1
        ent[t.tail_type.value] += 1
        rel[t.relation.value] += 1

    print("\n实体类型分布（头+尾）：")
    for name, cnt in ent.most_common():
        star = " ★新增" if name in NEW_ENTITIES else ""
        print(f"  {name:<6}{cnt:>4} 次{star}")

    print("\n关系类型分布：")
    for name, cnt in rel.most_common():
        star = " ★新增" if name in NEW_RELATIONS else ""
        print(f"  {name:<6}{cnt:>4} 次{star}")

    concept_ratio = ent.get("概念", 0) / max(1, sum(ent.values()))
    print(f"\n「概念」占比：{concept_ratio:.0%}")
    if concept_ratio > 0.5:
        print("  ⚠️ 超过一半都落到「概念」，说明该标的类型没被认出来 ——")
        print("     要么句子写得太含糊，要么 prompt 的类型定义还不够。")
    else:
        print("  ✅ 类型分布合理，没有明显堆积在「概念」上。")

    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
