"""tests/show_eval.py —— 把 run_eval 存下的 json 汇总成人看的报告。

为什么单独一个脚本：
    run_eval 在终端里只打一行一题，20 题就刷屏了；而且终端输出存不下来。
    --out 存了 json 之后，用这个脚本可以反复看、按分类看、只看失败题。

用法：
    python tests/show_eval.py runs/eval-02.json
    python tests/show_eval.py runs/eval-02.json --all
"""

import json
import sys
from collections import Counter


def main() -> int:
    if len(sys.argv) < 2:
        print("用法：python tests/show_eval.py <eval.json> [--all]")
        return 1
    path = sys.argv[1]
    show_all = "--all" in sys.argv

    rows = json.load(open(path, encoding="utf-8"))
    # ok=None 的是人工题（规则判不了），不参与自动计分
    auto = [r for r in rows if r.get("ok") is not None]
    manual = [r for r in rows if r.get("ok") is None]
    total = len(auto)
    passed = sum(1 for r in auto if r.get("ok"))

    print(f"文件：{path}")
    if total:
        print(f"总体：{passed}/{total} = {passed / total:.0%}")
    if manual:
        print(f"人工题：{len(manual)} 道（未计分）")
    print()

    hit: Counter = Counter()
    cnt: Counter = Counter()
    for r in auto:
        cnt[r["category"]] += 1
        if r.get("ok"):
            hit[r["category"]] += 1

    print("分类：")
    for cat in sorted(cnt):
        print(f"  {cat:<6}{hit[cat]}/{cnt[cat]} = {hit[cat] / cnt[cat]:.0%}")

    lats = sorted(r["latency_ms"] for r in rows if r.get("latency_ms"))
    if lats:
        print(f"\n延迟：中位 {lats[len(lats) // 2]:.0f}ms  最大 {lats[-1]:.0f}ms")

    bad = [r for r in auto if not r.get("ok")]
    print(f"\n失败 {len(bad)} 题：")
    for r in bad:
        print(f"\n[{r['category']}] {r['id']}  {r['query']}")
        print(f"  missing={r.get('missing')}  forbidden={r.get('forbidden')}")
        print(f"  route={r.get('route')}  steps={r.get('steps')}  degraded={r.get('degraded')}")
        answer = (r.get("answer") or "").replace("\n", " ")
        print(f"  回答：{answer[:240]}")

    if show_all:
        print("\n全部明细：")
        for r in rows:
            flag = "OK" if r.get("ok") else "XX"
            answer = (r.get("answer") or "").replace("\n", " ")[:100]
            print(f"  {flag} {r['id']:<4}{answer}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
