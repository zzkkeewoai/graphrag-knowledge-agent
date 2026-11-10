"""tests/demo.py —— 30 秒演示：四句话展示四类能力。

为什么需要：
    面试时没人有耐心看你跑 105 道评测。这个脚本按"能力递进"挑四道题，
    一次性展示：数值定位 → 跨文档多跳 → 权限边界 → 拒答。
    跑完带一行结论，适合直接投屏。

用法：
    python tests/demo.py                 # 默认连本地 8000
    python tests/demo.py --trace         # 额外打印链路步数
"""

import argparse
import json
import sys
import urllib.error
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# (标签, 问题, 一句话说明这道题在展示什么)
CASES = [
    ("① 数值定位", "单笔报销6000元，由谁审批？",
     "要把 6000 落进「5000元至5万」这一档，再定位到财务经理"),
    ("② 跨文档多跳", "采购一台25万元的设备，采购方式是什么，付款由谁审批？",
     "一份文件定采购方式（招标），另一份定审批人（财务总监）"),
    ("③ 权限边界", "销售部可以查看员工薪酬吗？",
     "考的是「知道谁不能看」，而不是「知道谁在看」"),
    ("④ 拒答", "公司有工会吗？",
     "知识库里没有就说没有，不许用常识补"),
]


def ask(url: str, query: str, timeout: float = 120.0) -> dict:
    payload = json.dumps({"query": query}, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"{url.rstrip('/')}/ask",
        data=payload,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()

    print("=" * 74)
    print("GraphRAG 企业制度问答 · 演示")
    print("=" * 74)

    ok = 0
    for idx, (label, query, why) in enumerate(CASES, 1):
        print(f"\n[{idx}/{len(CASES)}] {label}")
        print(f"  Q: {query}")
        print(f"     （{why}）")

        try:
            r = ask(args.url, query)
        except urllib.error.URLError as exc:
            print(f"  ✗ 连不上服务：{exc.reason}")
            return 1

        answer = (r.get("answer") or "").strip().replace("\n", " ")
        print(f"  A: {answer[:220]}")
        print(f"     route={r.get('route')}  steps={r.get('step_count')}  "
              f"{r.get('latency_ms')}ms")
        if not r.get("degraded"):
            ok += 1

    print("\n" + "=" * 74)
    print(f"四类问题全部走通（无降级）：{ok}/{len(CASES)}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
