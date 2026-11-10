"""tests/debug_retrieval.py —— 看一个查询到底召回了什么（不经过服务、不调 LLM）。

为什么需要它：
    "答不出来"有四种可能：块没入库、入库了没召回、召回了没进精排、
    精排进了但模型没用。这个脚本只看**第一层**——向量检索召回了什么、排第几。
    把这一层单独验证掉，剩下三层才不会瞎猜。

用法：
    python tests/debug_retrieval.py "审批权限矩阵什么时候生效"
    python tests/debug_retrieval.py "采购部对接哪些供应商" --top 10
    python tests/debug_retrieval.py "生效日期" --contains 元数据
"""

import argparse
import sys

sys.path.insert(0, ".")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--contains", default="", help="只标记包含该词的块")
    args = parser.parse_args()

    from src.database.local_vector_store import LocalVectorStore

    store = LocalVectorStore()
    store.create_collection("document_chunks")

    total = len(store.texts)
    print(f"库里共 {total} 个块\n")

    hits = store.search(args.query, args.top)
    print(f"查询：{args.query}")
    print("-" * 70)
    for i, h in enumerate(hits, 1):
        mark = "★" if args.contains and args.contains in h["chunk_text"] else " "
        text = h["chunk_text"].replace("\n", " ")[:52]
        print(f"{mark} {i:>2}. {h['score']:.3f}  [{h['doc_id']}]  {text}")

    # 全库搜一遍，看目标块的真实排名（只在前 top 里找不到时有意义）
    if args.contains:
        allhits = store.search(args.query, total)
        ranks = [
            (i + 1, h)
            for i, h in enumerate(allhits)
            if args.contains in h["chunk_text"]
        ]
        print()
        if ranks:
            print(f"「{args.contains}」相关块共 {len(ranks)} 个，排名：")
            for rank, h in ranks:
                print(f"  rank={rank:<3} score={h['score']:.3f}  {h['chunk_text'][:44]}")
        else:
            print(f"⚠️ 全库里没有包含「{args.contains}」的块 —— 说明根本没入库。")

    return 0


if __name__ == "__main__":
    sys.exit(main())
