"""tests/convert_eval_queries.py —— 把原项目域评测集转成规则判分格式。

背景：
    data/eval_queries.json 是给 Ragas（LLM 判官）用的，字段是
    query / expected_answer / relevant_docs，没有可机器判分的关键词。
    要拿它做"扩域后有没有退化"的回归，得先转成 datasets/*.jsonl 的格式。

两类题要分开处理：
    单跳事实查询（5 题）——expected_answer 是具体答案（"张伟属于技术部"），
                          可以抽关键词自动判分。
    多跳 / 全局（10 题）——expected_answer 是描述性的（"需要查找使用 Python
                          或 Neo4j 的其他项目"），规则判不了，标 manual 交人工。

关键词怎么抽（启发式）：
    把 expected_answer 按动词/连接词/标点切开，取**问题里没出现过**的词。
    例：Q"A项目使用了哪些技术？" A"A项目使用Python和Neo4j"
        → 切出 [A项目, Python, Neo4j]，去掉问题里有的 A项目 → [Python, Neo4j]

    这个启发式对单跳够用；对多跳会抽出噪声，所以多跳一律标 manual。

用法：
    python tests/convert_eval_queries.py
    python tests/convert_eval_queries.py --out datasets/project15.jsonl
"""

import argparse
import json
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 只在"连接词/标点"处切，**不按空格切**：
#   "Python和Neo4j"   → Python | Neo4j          （两项，正确）
#   "Java和Spring Boot" → Java | Spring Boot     （两项，Boot 不会被拆走）
# 之前按空格切再合并英文，会把 Java 和 Spring 粘成一个词，属于自己制造的假阴性。
SPLIT = re.compile(
    r"[和与、，,。；;：:（）()]+|使用|依赖|领导|属于|包含|包括|预计|需要|查找"
    r"|上线|投用|发布|完成|通过|实施"
)
STOP = {"项目", "部门", "技术", "公司", "人员", "团队", "情况", "情况如何"}


def norm(s: str) -> str:
    return "".join((s or "").split())


def extract_keywords(query: str, answer: str, limit: int = 3) -> list[str]:
    q = norm(query)
    parts = [p.strip() for p in SPLIT.split(answer) if p and p.strip()]
    out = []
    for p in parts:
        p = norm(p)
        if len(p) < 2 or p in STOP:
            continue
        if p in q:          # 问题里已有的词不算信息（那是题干）
            continue
        if p not in out:
            out.append(p)
    return out[:limit]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/eval_queries.json")
    parser.add_argument("--out", default="datasets/project15.jsonl")
    args = parser.parse_args()

    cases = json.loads(Path(args.src).read_text(encoding="utf-8"))
    rows = []
    manual = 0

    print(f"读取 {len(cases)} 条\n")
    print(f"{'id':<12}{'类型':<8}{'关键词/说明'}")
    print("-" * 70)

    for c in cases:
        is_single = "单跳" in c["category"]
        if is_single:
            kws = extract_keywords(c["query"], c["expected_answer"])
            row = {
                "id": c["id"],
                "category": "单跳",
                "query": c["query"],
                "expect": {"all": [[k] for k in kws], "none": []},
                "note": c["expected_answer"],
            }
            print(f"{c['id']:<12}{'单跳':<8}{kws}")
            if not kws:
                print(f"   ⚠️ 抽不出关键词，需人工补：{c['expected_answer']}")
        else:
            manual += 1
            row = {
                "id": c["id"],
                "category": c["category"],
                "query": c["query"],
                "expect": {"all": [], "none": []},
                "manual": True,          # run_eval 见到它只打答案、不判分
                "note": c["expected_answer"],
            }
            print(f"{c['id']:<12}{'人工':<8}{c['expected_answer'][:40]}")
        rows.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )

    print("-" * 70)
    print(f"已写入 {out}：可自动判分 {len(rows) - manual} 条，需人工 {manual} 条")
    print("\n下一步：打开该文件核一遍关键词（尤其 expect.all 里的），再用 --file 跑。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
