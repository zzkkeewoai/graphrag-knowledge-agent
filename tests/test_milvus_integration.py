import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.milvus_client import MilvusClient


def main():
    print("=" * 60)
    print("🚀 测试Milvus向量存储")
    print("=" * 60)

    # 测试文档
    test_doc = """
    智能客服系统于2024年3月启动，目标是将AI能力集成到现有的ERP系统中。
    项目负责人是王明，技术选型采用了Java 17、FastAPI框架和Milvus向量数据库。
    智能客服系统依赖B项目提供的用户权限管理模块。
    """

    # 1. 连接Milvus
    print("\n📊 Step 1: 连接Milvus...")
    client = MilvusClient()
    client.connect()

    # 2. 创建集合
    print("\n📊 Step 2: 创建集合...")
    client.create_collection("test_chunks")

    # 3. 文本切片
    print("\n📊 Step 3: 文本切片...")
    chunks = client.chunk_text(test_doc, chunk_size=100, overlap=20)
    print(f"  切分成 {len(chunks)} 个块")
    for i, chunk in enumerate(chunks):
        print(f"  块{i + 1}: {chunk[:50]}...")

    # 4. 插入向量
    print("\n📊 Step 4: 插入向量...")
    count = client.insert_chunks("doc_001", chunks)
    print(f"  插入 {count} 个向量")

    # 5. 向量检索
    print("\n🔍 Step 5: 向量检索...")
    query = "这个项目用了什么技术？"
    results = client.search(query, top_k=3)
    print(f"  查询: '{query}'")
    for r in results:
        print(f"    Score: {r['score']:.4f} | {r['chunk_text'][:60]}...")

    client.disconnect()
    print("\n✅ 测试完成!")


if __name__ == "__main__":
    main()