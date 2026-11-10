"""tests/run_all.py —— 一次跑完全部评测集，出一张总表。

为什么需要：
    六个集子分散在 datasets/ 下，逐个跑要六条命令、自己对着看。
    这个脚本复用 run_eval 的判分逻辑，跑完给一张表 + 一份合并 json。

用法：
    python tests/run_all.py
    python tests/run_all.py --only holdout3,multihop10
    python tests/run_all.py --url http://127.0.0.1:8000
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 复用 run_eval 的 ask / score / is_refusal，避免两套判分逻辑走偏
_spec = importlib.util.spec_from_file_location("run_eval", Path(__file__).with_name("run_eval.py"))
run_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_eval)

SETS = [
    # (集子名, 题集文件, 显示名, 业务域)
    # 域用于检索过滤：两套语料可共存，各跑各的域，互不串味
    ("simco_eval", "datasets/simco_eval.jsonl", "调优集", "simco"),
    ("holdout20", "datasets/holdout20.jsonl", "开发集", "simco"),
    ("holdout2", "datasets/holdout2.jsonl", "验收集", "simco"),
    ("holdout3", "datasets/holdout3.jsonl", "新语料", "simco"),
    ("multihop10", "datasets/multihop10.jsonl", "跨文档多跳", "simco"),
    ("project15", "datasets/project15.jsonl", "原域回归", "project"),
    ("weijin40", "datasets/weijin40.jsonl", "真实制度PDF", "pdf"),
    ("weijin_hr", "datasets/weijin_hr.jsonl", "年报人事", "pdf"),
    ("kouyu", "datasets/weijin48_kouyu.jsonl", "口语版", "pdf"),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--only", default="", help="只跑指定集子，逗号分隔")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    wanted = {x.strip() for x in args.only.split(",") if x.strip()}
    todo = [s for s in SETS if not wanted or s[0] in wanted]

    started = time.time()
    summary = []
    all_rows = []

    for name, path, label, domain in todo:
        if not Path(path).exists():
            print(f"[skip] {path} 不存在")
            continue

        raw = [l for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
        cases = [json.loads(l) for l in raw]

        passed = auto = manual = 0
        lats = []
        for case in cases:
            result = run_eval.ask(args.url, case["query"], args.timeout, domain=domain)
            if "error" in result:
                print(f"  ✗ {case['id']} 请求失败：{result['error']}")
                manual += 1
                continue

            answer = result.get("answer", "") or ""
            if case.get("manual"):
                manual += 1
                all_rows.append({**case, "ok": None, "answer": answer, "set": name})
                continue

            if case.get("category") == "拒答":
                ok = run_eval.is_refusal(answer)
                scored = {"missing": [] if ok else ["未拒答"], "forbidden": []}
            else:
                s = run_eval.score(answer, case["expect"])
                ok, scored = s["ok"], s

            auto += 1
            passed += 1 if ok else 0
            if result.get("latency_ms"):
                lats.append(result["latency_ms"])
            all_rows.append({**case, "ok": ok, "answer": answer, "set": name, **scored})

        rate = f"{passed}/{auto} = {passed / auto:.0%}" if auto else "—"
        med = sorted(lats)[len(lats) // 2] if lats else 0
        summary.append((label, name, rate, passed, auto, manual, med))
        print(f"  {label:<10}{name:<12}{rate:<12}人工 {manual}  中位 {med:.0f}ms")

    print("\n" + "=" * 64)
    print(f"{'集子':<12}{'结果':<12}{'人工':<6}{'中位延迟'}")
    print("-" * 64)
    for label, name, rate, p, a, m, med in summary:
        print(f"{label:<12}{rate:<12}{m:<6}{med:.0f}ms")

    total_a = sum(s[4] for s in summary)
    total_p = sum(s[3] for s in summary)
    if total_a:
        print("-" * 64)
        print(f"{'合计':<12}{total_p}/{total_a} = {total_p / total_a:.0%}")

    # 0 分几乎一定是"语料没灌"，不是"系统坏了"——
    # 两套语料（澜海科技 / 西部黄金）只能同时存在一套（--reset 会清库）。
    zero = [s for s in summary if s[4] and s[3] == 0]
    if zero:
        print("\n⚠️ 有集子 0 分，大概率是语料不匹配，不是系统问题：")
        for label, name, *_ in zero:
            print(f"   · {label}（{name}）需要对应语料，见 tests/run_all.py 顶部注释")
    print(f"\n总耗时 {(time.time() - started):.0f}s")

    out = Path("runs") / f"all-{time.strftime('%m%d-%H%M')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"明细：{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
