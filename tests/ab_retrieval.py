"""tests/ab_retrieval.py —— 检索层 A/B：BM25 到底有没有帮上忙。

为什么不看端到端分数：
    40 题 + 8 题在开/关 BM25 时都是全对，说明端到端已经到顶（天花板效应），
    比较不出差异。要看一路召回的价值，必须看**它有没有把含答案的块捞进候选**。

指标：hit@k —— 给定题目的期望关键词，top-k 候选里有没有一块包含它。
    这等价于"模型有没有机会看到答案"，是端到端分数的上界。

用法：
    python tests/ab_retrieval.py
    python tests/ab_retrieval.py --datasets datasets/weijin40.jsonl,datasets/weijin_hr.jsonl --top 8
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_spec = importlib.util.spec_from_file_location("run_eval", Path(__file__).with_name("run_eval.py"))
run_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_eval)


def keywords_of(case) -> list[str]:
    out = []
    for group in case.get("expect", {}).get("all", []):
        variants = group if isinstance(group, list) else [group]
        out.extend(variants)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="datasets/weijin40.jsonl,datasets/weijin_hr.jsonl")
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args()

    from src.config import Config
    from src.database.local_vector_store import LocalVectorStore
    from src.database.neo4j_client import Neo4jClient
    from src.retriever.hybrid_retriever import HybridRetriever

    neo4j = Neo4jClient(uri=Config.NEO4J_URI, user=Config.NEO4J_USER, password=Config.NEO4J_PASSWORD)
    neo4j.connect()
    store = LocalVectorStore()
    store.create_collection("document_chunks")
    retriever = HybridRetriever(neo4j, store)

    cases = []
    for p in args.datasets.split(","):
        p = p.strip()
        if p:
            cases += [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]
    cases = [c for c in cases if c.get("category") != "拒答"]   # 拒答题没有关键词

    results = {"off": [], "on": []}
    for mode, flag in (("off", "false"), ("on", "true")):
        os.environ["USE_BM25"] = flag
        # 每轮重置懒加载的 BM25 索引，保证开关真的生效
        retriever._bm25 = None
        for case in cases:
            r = retriever.retrieve(case["query"], top_k=args.top)
            # 和 orchestrator 一样：候选本身不带 text（RRF 只留 doc_id/chunk_index），
            # 要用向量/BM25 结果拼一张 text_map 再回填。早先版本只统计
            # "自带 text 的候选"，量到的是"谁被保底追加了"，不是"谁进了上下文"。
            text_map = {}
            for x in r.get("vector_results", []) + r.get("bm25_results", []):
                text_map[(x.get("doc_id"), x.get("chunk_index"))] = x.get("text", "")
            blob = "\n".join(
                text_map[(f.get("doc_id"), f.get("chunk_index"))]
                for f in r.get("fused_results", [])
                if (f.get("doc_id"), f.get("chunk_index")) in text_map
            )
            kws = keywords_of(case)
            hit = any(run_eval.norm(k) in run_eval.norm(blob) for k in kws) if kws else None
            results[mode].append((case["id"], hit))

    print(f"{'题号':<6}{'BM25关':<8}{'BM25开':<8}变化")
    print("-" * 34)
    gained, lost = [], []
    for (cid, off), (_, on) in zip(results["off"], results["on"]):
        if off != on:
            mark = "↑ 救回" if on else "↓ 丢失"
            (gained if on else lost).append(cid)
        else:
            mark = ""
        print(f"{cid:<6}{'✓' if off else '✗':<8}{'✓' if on else '✗':<8}{mark}")

    n = len(cases)
    off_hit = sum(1 for _, h in results["off"] if h)
    on_hit = sum(1 for _, h in results["on"] if h)
    print("-" * 34)
    print(f"hit@{args.top}： 关 {off_hit}/{n} = {off_hit/n:.0%}   开 {on_hit}/{n} = {on_hit/n:.0%}")
    if gained:
        print(f"BM25 救回：{gained}")
    if lost:
        print(f"BM25 导致丢失：{lost}")

    neo4j.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
