import logging
from typing import List, Dict, Any
from pymilvus import (
    connections, Collection, CollectionSchema,
    FieldSchema, DataType, utility
)
from sentence_transformers import SentenceTransformer
import tiktoken

logger = logging.getLogger(__name__)


class MilvusClient:
    """Milvus向量数据库客户端"""

    def __init__(self, host: str = "localhost", port: int = 19530):
        self.host = host
        self.port = port
        self.collection = None
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')  # 384维，轻量快速

    def connect(self):
        """连接Milvus"""
        connections.connect(host=self.host, port=self.port)
        logger.info(f"已连接到Milvus: {self.host}:{self.port}")

    def disconnect(self):
        """断开连接"""
        connections.disconnect("default")

    def create_collection(self, collection_name: str = "document_chunks"):
        """创建向量集合"""
        # 检查是否已存在
        if utility.has_collection(collection_name):
            logger.info(f"集合 {collection_name} 已存在，删除重建")
            utility.drop_collection(collection_name)

        # 定义Schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="chunk_text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)  # 384维
        ]
        schema = CollectionSchema(fields, description="文档块向量存储")

        self.collection = Collection(collection_name, schema)
        logger.info(f"集合 {collection_name} 创建成功")

        # 创建索引（加速检索）
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        self.collection.create_index("embedding", index_params)
        logger.info("向量索引创建完成")

        return self.collection

    def chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """文本切片"""
        # 使用tiktoken按token切分
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)

        chunks = []
        for i in range(0, len(tokens), chunk_size - overlap):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = encoding.decode(chunk_tokens)
            chunks.append(chunk_text)

        return chunks

    def insert_chunks(self, doc_id: str, chunks: List[str]) -> int:
        """插入文档块"""
        if not chunks:
            return 0

        # 生成向量
        embeddings = self.encoder.encode(chunks, show_progress_bar=False)

        # 准备数据
        data = [
            chunks,  # chunk_text
            [doc_id] * len(chunks),  # doc_id
            list(range(len(chunks))),  # chunk_index
            embeddings.tolist()  # embedding
        ]

        # 插入
        self.collection.insert(data)
        self.collection.flush()
        logger.info(f"插入 {len(chunks)} 个文档块到Milvus")

        return len(chunks)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """向量检索"""
        if self.collection is None:
            logger.warning("Milvus 集合未初始化，返回空结果")
            return []
        # 加载集合到内存
        self.collection.load()

        # 查询向量
        query_vector = self.encoder.encode([query])

        # 检索
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        results = self.collection.search(
            query_vector,
            "embedding",
            search_params,
            limit=top_k,
            output_fields=["chunk_text", "doc_id", "chunk_index"]
        )

        # 格式化结果
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    "chunk_text": hit.entity.get('chunk_text'),
                    "doc_id": hit.entity.get('doc_id'),
                    "chunk_index": hit.entity.get('chunk_index'),
                    "score": hit.score
                })

        return formatted_results