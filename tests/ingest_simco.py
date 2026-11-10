"""tests/ingest_simco.py —— 把 data/simco/ 下的文档灌进图谱与向量库。

和 seed_demo_data.py 的区别：
    seed 脚本是**硬编码**几条三元组，用来"让服务有数据可查"；
    本脚本是**真读文档**：切段 → 抽三元组 → 写 Neo4j → 段落入库 Milvus。
    这才是真实的建图流程，也是验证「文档能不能变成可检索的图」的那一步。

用法：
    # 只建图（不依赖 Milvus / sentence-transformers，先验证多跳能不能走通）
    python tests/ingest_simco.py --reset --skip-milvus

    # 建图 + 向量（需要 pymilvus 与 sentence-transformers）
    python tests/ingest_simco.py --reset

    # 只灌某几篇
    python tests/ingest_simco.py --only org.md,approval_matrix.md

注意：
    --reset 会**清空 Neo4j 里的所有节点**（演示库才这么干）。
    不清也能跑，但旧的项目域数据会和公司域数据混在一起，检索会互相干扰。
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from src.config import Config  # noqa: E402
from src.database.neo4j_client import Neo4jClient  # noqa: E402
from src.extractors.triple_extractor import TripleExtractor  # noqa: E402

COLLECTION = "document_chunks"


def split_paragraphs(text: str) -> list[str]:
    """按空行切段，跳过纯标题行和元数据行。

    和试抽脚本用同一套切法 —— 建库和试抽必须一致，
    否则会出现"试抽看着好、灌进去却不对"的鬼故事。
    """
    blocks = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if len(lines) == 1 and lines[0].startswith("#"):
            continue
        # 文档头部的元数据块（"- 文档编号：… / - 生效日期：…"）**也要入库**。
        # 之前这里直接 continue 掉，导致"审批矩阵什么时候生效"这类问题
        # 永远答不出来——不是模型不会，是那条信息从来没进过库。
        # 它没有实体，所以包一句人话再入库，便于检索命中。
        if all(ln.startswith("- ") for ln in lines):
            parts = [ln[2:].strip() for ln in lines]
            blocks.append("文档元数据：" + "；".join(parts))
            continue
        blocks.append(block)
    return blocks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="data/simco")
    parser.add_argument("--only", default="", help="只灌指定文件，逗号分隔")
    parser.add_argument("--reset", action="store_true", help="先清空 Neo4j（演示库专用）")
    parser.add_argument("--skip-milvus", action="store_true", help="只建图，跳过向量库")
    args = parser.parse_args()

    if not Config.DEEPSEEK_API_KEY:
        print("[FAIL] .env 里没有 DEEPSEEK_API_KEY")
        return 1

    files = sorted(f for f in os.listdir(args.dir) if f.endswith(".md"))
    if args.only:
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        files = [f for f in files if f in wanted]
    if not files:
        print(f"[FAIL] {args.dir} 下没有 .md 文件")
        return 1

    print("=" * 66)
    print(f"待灌文档：{len(files)} 篇 → {args.dir}")
    for f in files:
        print(f"  - {f}")
    print("=" * 66)

    neo4j = Neo4jClient(uri=Config.NEO4J_URI, user=Config.NEO4J_USER, password=Config.NEO4J_PASSWORD)
    neo4j.connect()
    neo4j.create_indexes()

    if args.reset:
        with neo4j.driver.session() as session:
            before = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            session.run("MATCH (n) DETACH DELETE n")
        print(f"\n[reset] 已清空 Neo4j：删除 {before} 个节点\n")

    extractor = TripleExtractor(
        api_key=Config.DEEPSEEK_API_KEY,
        base_url=Config.DEEPSEEK_BASE_URL,
        model_name=Config.DEEPSEEK_MODEL,
        temperature=0.1,
        confidence_threshold=0.7,  # 建库用 0.7，过滤掉犹豫的边；试抽时才放开
    )

    milvus = None
    if not args.skip_milvus:
        # 用本地向量库：接口与 MilvusClient 一致，不需要 etcd/minio 容器
        from src.database.local_vector_store import LocalVectorStore

        milvus = LocalVectorStore()
        milvus.connect()
        # 原来这里写死 drop_existing=True，导致第二次灌库（比如再灌项目域）会把
        # 上一次的向量全清掉，两域无法共存。改为跟随 --reset：
        #   带 --reset → 清空重建；不带 → 追加（多域共存）
        milvus.create_collection(COLLECTION, drop_existing=args.reset)
        milvus.load_if_needed()

    started = time.time()
    total_triples = 0
    total_chunks = 0
    chunk_offset = 0

    for name in files:
        doc_id = os.path.splitext(name)[0]
        with open(os.path.join(args.dir, name), encoding="utf-8") as f:
            text = f.read()
        blocks = split_paragraphs(text)

        # 给元数据块补上文档标题。
        # 不加标题，块里只有"文档编号 LH-FIN-2026-010；生效日期 2026-01-01"，
        # 而用户问的是"《审批权限矩阵》什么时候生效"——字面完全对不上，检索必然落空。
        # 这是"块内缺少主体"的典型问题：内容对，但缺了能让人/模型对上号的锚点。
        title = next((ln[2:].strip() for ln in text.splitlines() if ln.startswith("# ")), "")
        if title:
            blocks = [
                f"《{title}》{b}" if b.startswith("文档元数据：") else b
                for b in blocks
            ]

        print(f"\n[{doc_id}] {len(blocks)} 段")
        doc_triples = []
        for i, block in enumerate(blocks, 1):
            # 抽取是 LLM 调用，失败多半是限流/超时——带指数退避重试（1s→2s→4s）
            got, err = None, None
            for attempt in range(3):
                try:
                    got = extractor.extract(block, chunk_id=f"{doc_id}#p{i}").triples
                    err = None
                    break
                except Exception as e:  # noqa: BLE001
                    err = e
                    if attempt < 2:
                        delay = 1.0 * (2 ** attempt)
                        print(f"  段 {i} 抽取失败（{type(e).__name__}），{delay:.0f}s 后重试 {attempt + 2}/3")
                        time.sleep(delay)
            if err is not None:
                print(f"  段 {i} 抽取最终失败：{type(err).__name__}（已重试 3 次）")
                continue
            doc_triples.extend(got)

        if doc_triples:
            stats = neo4j.batch_write(doc_triples)
            total_triples += len(doc_triples)
            print(f"  抽出 {len(doc_triples)} 条三元组，写入成功 {stats.get('success')} 条")
        else:
            print("  （没抽出三元组）")

        if milvus is not None:
            n = milvus.insert_chunks(doc_id, blocks, chunk_offset=chunk_offset)
            chunk_offset += n
            total_chunks += n
            print(f"  向量库写入 {n} 个块")

    elapsed = time.time() - started
    print("\n" + "=" * 66)
    print(f"完成：{total_triples} 条三元组，{total_chunks} 个向量块，耗时 {elapsed:.0f}s")
    with neo4j.driver.session() as session:
        nodes = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        rels = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
    print(f"Neo4j 现状：{nodes} 个节点，{rels} 条关系")
    print("=" * 66)

    neo4j.close()
    if milvus is not None:
        milvus.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
