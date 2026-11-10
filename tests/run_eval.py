"""tests/run_eval.py —— 对 GraphRAG 服务跑一遍评测集，出准确率。

【判分规则（沿用 ContextOps 那套）】
    expect.all  = 列表，每个元素可以是字符串或字符串列表
                  · 字符串      → 必须出现（AND）
                  · 字符串列表  → 其中任一出现即可（OR，同义词）
    expect.none = 禁止出现的词（一律 AND 语义：出现即判错）

    为什么同义词要写成嵌套列表：模型说"逾期"还是"过期"都对，
    只认一个词会把正确回答判成错的——假阴性比假阳性更伤，
    因为你会去修一个不存在的问题。

【为什么不直接评"答案对不对"】
    这件事规则做不到。规则只能判"关键事实在不在"，判不了"推理对不对"。
    所以这 20 题都设计成**答案里有确定的关键事实**（人名/数字/系统名），
    用规则判就够；判不了的（表述是否清楚、理由是否成立）留给人工。

用法：
    python tests/run_eval.py
    python tests/run_eval.py --url http://127.0.0.1:8000 --out runs/eval.json
"""

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


_CN_DIGIT = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
             "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}


def _cn2int(s: str) -> str | None:
    """把「十」「二十」「三十五」这类中文数字转成阿拉伯数字。"""
    if s.isdigit():
        return s
    if s == "十":
        return "10"
    if "十" in s:
        a, _, b = s.partition("十")
        hi = _CN_DIGIT.get(a, 1) if a else 1
        lo = _CN_DIGIT.get(b, 0) if b else 0
        return str(hi * 10 + lo)
    if all(ch in _CN_DIGIT for ch in s):
        return "".join(str(_CN_DIGIT[ch]) for ch in s)
    return None


def norm(s: str) -> str:
    """归一化：去空白、去千分位逗号、中文数字转阿拉伯数字。

    法律/制度文本里这两类写法极多：
        「百分之十」      vs 关键词 "10%"
        「5,000万元」     vs 关键词 "5000万"
    不归一化就会把正确答案判成错——判分器假阴性比系统 bug 更浪费时间。
    """
    t = "".join((s or "").split())
    t = t.replace(",", "").replace("，", "")
    # 百分之X → X%
    t = re.sub(r"百分之([零一二两三四五六七八九十\d]+)",
               lambda m: (_cn2int(m.group(1)) or m.group(1)) + "%", t)
    return t


# 拒答题不能用"必含某词"判——模型的拒答措辞很自由
# （"无法回答""未提供""没有提到"…），只认几个词会把正确答案判成错的。
REFUSAL_MARKERS = [
    "无法回答", "无法确定", "无法判断", "未找到", "未规定", "没有规定",
    "未提及", "未提供", "未涉及", "没有相关", "没有提到", "暂无",
    # 这三种是实测中真的出现过的措辞（"信息不足以回答"），
    # 漏了它们会把正确拒答判成错——判分器最常见的假阴性来源。
    "不足以", "不足够", "没有足够",
    # "无法得知""未包含" 是跑真实 PDF 语料时出现的第三种说法。
    # 同一个模型对同一道题的三次拒答分别是「无法回答 / 无法得知」，措辞会变，
    # 判分器跟不上就会把稳定拒答误判成"退化成幻觉"。
    "无法得知", "无法获知", "未包含", "未涵盖", "没有涉及",
]


def is_refusal(answer: str) -> bool:
    return any(m in (answer or "") for m in REFUSAL_MARKERS)


def score(answer: str, expect: dict) -> dict:
    text = norm(answer)
    missing = []
    for key in expect.get("all", []):
        variants = key if isinstance(key, list) else [key]
        if not any(norm(v) in text for v in variants):
            missing.append("|".join(variants))
    forbidden = [k for k in expect.get("none", []) if norm(k) in text]
    return {"ok": not missing and not forbidden, "missing": missing, "forbidden": forbidden}


def ask(url: str, query: str, timeout: float = 120.0, domain: str | None = None) -> dict:
    body = {"query": query}
    if domain:
        body["domain"] = domain          # 业务域过滤：多域共存时防串味
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"{url.rstrip('/')}/ask",
        data=payload,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        data["_elapsed_ms"] = round((time.time() - started) * 1000, 1)
        return data
    except urllib.error.HTTPError as exc:
        return {"error": f"HTTP {exc.code}", "detail": exc.read().decode("utf-8", "replace")[:300]}
    except urllib.error.URLError as exc:
        return {"error": "连不上服务", "detail": str(exc.reason)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--file", default="datasets/simco_eval.jsonl")
    parser.add_argument("--out", default="")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--domain", default="", help="只在该业务域检索（如 pdf / simco）")
    args = parser.parse_args()

    raw = [line for line in Path(args.file).read_text(encoding="utf-8").splitlines() if line.strip()]
    cases = []
    for lineno, line in enumerate(raw, 1):
        try:
            cases.append(json.loads(line))
        except json.JSONDecodeError as exc:
            # 最常见的错误是行尾多一个逗号（JSON 数组习惯），报错信息却只给列号，
            # 很难定位。这里直接指到行，并给出最可能的原因。
            print(f"[FAIL] {args.file} 第 {lineno} 行不是合法 JSON：{exc.msg}（第 {exc.colno} 列）")
            print("       最常见原因：行末多了逗号。JSONL 每行必须是独立完整的对象。")
            print(f"       该行内容：{line[:120]}")
            return 1
    print(f"评测集：{args.file}   共 {len(cases)} 题\n")

    rows = []
    for i, case in enumerate(cases, 1):
        result = ask(args.url, case["query"], args.timeout,
                     domain=(args.domain or case.get("domain") or None))

        if "error" in result:
            print(f"  ✗ {case['id']} {case['query']}  → {result['error']}")
            rows.append({**case, "ok": False, "answer": "", "error": result["error"]})
            continue

        # 标记了 manual 的题（原项目域的多跳/全局题）规则判不了，
        # 只把期望和实际并排打出来交人工看，不参与自动计分。
        if case.get("manual"):
            print(f"  · [人工] {case['id']}  {case['query']}")
            print(f"      期望：{(case.get('note') or '')[:120]}")
            print(f"      实际：{(result.get('answer') or '').replace(chr(10), ' ')[:120]}")
            rows.append({**case, "ok": None, "answer": result.get("answer", "")})
            continue

        # 拒答题单独判：只要出现任一拒答措辞即算对。
        # 不能要求"必含某个词"——模型的拒答说法太自由。
        if case.get("category") == "拒答":
            refused = is_refusal(result.get("answer", ""))
            scored = {"ok": refused, "missing": [] if refused else ["未拒答"], "forbidden": []}
        else:
            scored = score(result.get("answer", ""), case["expect"])
        row = {
            **case,
            "ok": scored["ok"],
            "missing": scored["missing"],
            "forbidden": scored["forbidden"],
            "answer": result.get("answer", ""),
            "route": result.get("route"),
            "degraded": result.get("degraded"),
            "steps": result.get("step_count"),
            "latency_ms": result.get("latency_ms"),
            "elapsed_ms": result.get("_elapsed_ms"),
        }
        rows.append(row)

        flag = "✓" if scored["ok"] else "✗"
        print(f"  {flag} [{case['category']}] {case['id']}  {case['query']}")
        if not scored["ok"]:
            if scored["missing"]:
                print(f"      缺少：{scored['missing']}")
            if scored["forbidden"]:
                print(f"      出现禁止词：{scored['forbidden']}")
            print(f"      回答：{(row['answer'] or '')[:160]}")
        else:
            print(f"      ({row['latency_ms']}ms)")

    # ---------------- 汇总 ----------------
    # ok=False/True 参与计分；ok=None 是人工题，单独计数
    auto = [r for r in rows if r.get("ok") is not None]
    manual_rows = [r for r in rows if r.get("ok") is None]
    total = len(auto)
    passed = sum(1 for r in auto if r["ok"])
    print("\n" + "=" * 66)
    if total:
        print(f"总体：{passed}/{total} = {passed / total:.0%}")
    if manual_rows:
        print(f"另有 {len(manual_rows)} 道人工题（未计分），请对照上面「期望 / 实际」自评")

    by_cat: dict[str, list[bool]] = {}
    for r in auto:
        by_cat.setdefault(r["category"], []).append(r["ok"])
    print("\n分类：")
    for cat, marks in by_cat.items():
        n = len(marks)
        p = sum(marks)
        print(f"  {cat:<6}{p}/{n} = {p / n:.0%}")

    lats = [r["latency_ms"] for r in rows if r.get("latency_ms")]
    if lats:
        lats.sort()
        median = lats[len(lats) // 2]
        print(f"\n延迟：中位 {median:.0f}ms  最大 {lats[-1]:.0f}ms")
        # 一次 LLM 调用通常 2-10 秒。中位延迟低于 500ms 基本可以断定：
        # 要么命中了语义缓存，要么 USE_LLM_ANSWER 没开（answer 只是检索原文）。
        # 这两种情况下评测结果都不可信，必须先排除。
        if median < 500:
            print("  ⚠️ 中位延迟 < 500ms：疑似命中缓存或未开启 LLM 生成。")
            print("     请用 run_api.ps1 重启服务（它会关缓存、开生成）后重跑。")
    print("=" * 66)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"明细已写入 {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
