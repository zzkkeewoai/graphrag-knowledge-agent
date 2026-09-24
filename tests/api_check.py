"""api.py 服务的接口自测脚本（不依赖 pytest，直接跑就能看结果）

前置：
    1. 启动服务：cd graphrag_system && python -m uvicorn api:app --port 8000
    2. 另开终端：python tests/api_check.py

覆盖 4 个场景：
    1. GET  /health           —— 健康检查
    2. POST /ask 正常查询      —— 真实检索链路
    3. POST /ask 空 query     —— Pydantic 校验（期望 422）
    4. POST /ask 超时         —— 超时控制（期望 route=timeout）

为什么要写这个脚本：服务化之后，"能不能被正确调用"和"参数错了会怎样"
必须可验证——这是从"库调用"到"服务"的工程化标志之一。
"""
import json
import sys
import time
import urllib.error
import urllib.request

# Windows 控制台默认用 GBK，中文会乱码；统一转 UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE = "http://127.0.0.1:8000"


def call(path, method="GET", payload=None, timeout=60):
    """发一个请求，返回 (状态码, 响应体 dict, 耗时ms)。"""
    url = BASE + path
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)

    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return resp.status, json.loads(body), (time.time() - started) * 1000
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = {"raw": body}
        return e.code, parsed, (time.time() - started) * 1000


def show(title, status, body, elapsed_ms, extra=None):
    print(f"\n{'=' * 60}")
    print(f"[{title}]")
    print(f"  HTTP {status} | 耗时 {elapsed_ms:.0f} ms")
    print(f"  {json.dumps(body, ensure_ascii=False, indent=2)[:1200]}")
    if extra:
        print(f"  → {extra}")


def main():
    results = []

    # ---- 场景 1：健康检查 ----
    status, body, ms = call("/health")
    ok = status == 200 and body.get("status") in ("ready", "degraded")
    show("1. GET /health", status, body, ms, f"判定: {'PASS' if ok else 'FAIL'}")
    results.append(("health", ok))

    # ---- 场景 2：正常问答（真实检索链路）----
    status, body, ms = call(
        "/ask",
        method="POST",
        payload={"query": "智能客服系统用了哪些技术？", "timeout_seconds": 30},
        timeout=90,
    )
    ok = status == 200 and body.get("answer") and body.get("route") in (
        "local", "global", "degraded",
    )
    show(
        "2. POST /ask 正常查询",
        status,
        body,
        ms,
        f"判定: {'PASS' if ok else 'FAIL'} | route={body.get('route')} "
        f"服务端耗时={body.get('latency_ms')}ms steps={body.get('step_count')}",
    )
    results.append(("ask_normal", ok))

    # ---- 场景 3：参数校验（空 query 应被挡在入口）----
    status, body, ms = call("/ask", method="POST", payload={"query": ""})
    ok = status == 422  # FastAPI 校验失败返回 422
    show("3. POST /ask 空 query（期望 422）", status, body, ms, f"判定: {'PASS' if ok else 'FAIL'}")
    results.append(("validation", ok))

    # ---- 场景 4：超时控制（期望走 timeout 分支而不是 500）----
    status, body, ms = call(
        "/ask",
        method="POST",
        payload={"query": "项目之间的依赖关系是什么？", "timeout_seconds": 0.01},
        timeout=90,
    )
    ok = status == 200 and body.get("route") == "timeout"
    show(
        "4. POST /ask 超时 0.01s（期望 route=timeout）",
        status,
        body,
        ms,
        f"判定: {'PASS' if ok else 'FAIL'}",
    )
    results.append(("timeout", ok))

    # ---- 汇总 ----
    print(f"\n{'=' * 60}")
    passed = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"  合计: {passed}/{len(results)} 通过")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
