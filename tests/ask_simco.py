"""tests/ask_simco.py —— 向 /ask 提问题，并结构化打印结果。

【为什么不用 curl / Invoke-RestMethod】
    Windows 上通过 shell 传中文极不可靠：PowerShell 的默认编码会把 UTF-8
    的字节当成 Latin-1 解释，服务器收到的就是乱码（表现为 answer 里一片
    "éè´ä¸å°" 这种字符）。而乱码查询会让检索直接短路（step_count=1、
    latency 只有几毫秒），看起来像"系统不会答"，其实是"问题没传进去"。

    Python 里 str 永远是 Unicode，json.dumps(ensure_ascii=False).encode("utf-8")
    出来就是标准 UTF-8 字节，不经过任何本地代码页。所以：**中文请求用 Python 发**。

用法：
    python tests/ask_simco.py                    # 跑内置的三个问题
    python tests/ask_simco.py --q "自定义问题"
    python tests/ask_simco.py --url http://127.0.0.1:8000
"""

import argparse
import json
import sys
import urllib.error
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

QUESTIONS = [
    "采购一台30万元的设备，需要谁审批？依据哪一条？",
    "财务部可以访问哪些系统？",
    "员工薪酬数据哪些部门可以看？",
]


def ask(url: str, query: str, timeout: float = 120.0, trace: bool = False) -> dict:
    payload = json.dumps(
        {"query": query, "include_trace": bool(trace)}, ensure_ascii=False
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{url.rstrip('/')}/ask",
        data=payload,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        return {"error": f"HTTP {exc.code}", "detail": body}
    except urllib.error.URLError as exc:
        return {"error": "连不上服务", "detail": str(exc.reason)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--q", default="", help="只问一个问题")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--trace", action="store_true", help="打印链路追踪（各节点输入输出）")
    args = parser.parse_args()

    questions = [args.q] if args.q else QUESTIONS

    for i, q in enumerate(questions, 1):
        print("=" * 72)
        print(f"[{i}/{len(questions)}] {q}")
        print("=" * 72)

        result = ask(args.url, q, args.timeout, trace=args.trace)

        if "error" in result:
            print(f"  ✗ {result['error']}: {result.get('detail', '')}")
            continue

        print(f"  route={result.get('route')}  degraded={result.get('degraded')}  "
              f"steps={result.get('step_count')}  latency={result.get('latency_ms')}ms")
        print()
        print("  回答：")
        for line in (result.get("answer") or "").splitlines():
            print(f"    {line}")
        print()

        # trace 里能看到每个节点（路由/检索/验证/生成）的输入输出，
        # 是排查"证据明明有、答案却说没有"的唯一手段。
        if result.get("trace"):
            print("  trace：")
            print(json.dumps(result["trace"], ensure_ascii=False, indent=2)[:3000])
            print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
