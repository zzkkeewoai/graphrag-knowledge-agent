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
        self._loaded = False        # v2.2: 集合加载状态标记（避免热路径重复 load）
        self._embed_batch_size = 64  # v2.2: batch embedding 批次大小

    def connect(self):
        """连接Milvus"""
        connections.connect(host=self.host, port=self.port)
        logger.info(f"已连接到Milvus: {self.host}:{self.port}")

    def disconnect(self):
        """断开连接"""
        connections.disconnect("default")

    def create_collection(self, collection_name: str = "document_chunks", drop_existing: bool = False):
        """创建向量集合

        工程化修复（v2.2）：原实现"集合存在则直接 drop 重建"——正常启动可能
        导致数据丢失。改为 drop_existing 参数控制，默认 False（存在则复用，
        仅当集合不存在时创建），显式传入 True 才重建。
        """
        # 检查是否已存在
        if utility.has_collection(collection_name):
            if drop_existing:
                logger.warning(f"集合 {collection_name} 已存在，按 drop_existing=True 删除重建")
                utility.drop_collection(collection_name)
            else:
                logger.info(f"集合 {collection_name} 已存在，复用（如需重建请传 drop_existing=True）")
                self.collection = Collection(collection_name)
                self._loaded = False
                return self.collection

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

    def insert_chunks(self, doc_id: str, chunks: List[str], chunk_offset: int = 0) -> int:
        """插入文档块

        工程化修复（v2.2）：
        1. batch embedding（避免一次性 encode 超大列表）
        2. chunk_index 支持全局偏移（batch 处理时跨批次连续，不每批从 0 开始）
        3. flush 只在整批插入后执行一次（非逐条）
        4. 插入后标记集合可 load（search 前只需 load 一次）
        """
        if not chunks:
            return 0

        # batch embedding（分片避免内存峰值）
        all_embeddings = []
        for i in range(0, len(chunks), self._embed_batch_size):
            batch = chunks[i:i + self._embed_batch_size]
            batch_emb = self.encoder.encode(batch, show_progress_bar=False)
            all_embeddings.extend(batch_emb.tolist())

        # 准备数据（chunk_index 从 chunk_offset 开始，支持全局连续）
        data = [
            chunks,                                   # chunk_text
            [doc_id] * len(chunks),                   # doc_id
            list(range(chunk_offset, chunk_offset + len(chunks))),  # chunk_index（全局连续）
            all_embeddings,                           # embedding
        ]

        # 整批插入 + 一次 flush
        self.collection.insert(data)
        self.collection.flush()
        logger.info(f"插入 {len(chunks)} 个文档块到Milvus (chunk_offset={chunk_offset})")

        return len(chunks)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """向量检索

        工程化修复（v2.2）：
        - 原实现每次 search 都调用 collection.load() —— 高频查询热路径上
          反复加载集合索引，是明显性能问题。
        - 改为：load() 只发生在集合创建/数据写入后的初始化阶段（见 load_if_needed /
          insert_chunks），search 热路径不再重复 load。
        """
        if self.collection is None:
            logger.warning("Milvus 集合未初始化，返回空结果")
            return []
        # 修复：load 不在 search 热路径执行。若集合尚未 load（例如进程重启后
        # 只调 search 未走 insert），做一次幂等 load 兜底（内部有状态标记，非每次）。
        self._ensure_loaded()

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

    # ----------------------------------------------------------
    # 集合加载生命周期（v2.2 新增）
    # ----------------------------------------------------------
    def _ensure_loaded(self):
        """幂等加载集合（仅当尚未加载时执行一次）。

        Milvus collection.load() 之后再次调用是幂等的，但每次调用都有
        状态检查开销；这里用本地标记避免热路径重复判断。
        """
        if self._loaded:
            return
        self.collection.load()
        self._loaded = True
        logger.debug("Milvus 集合已加载")

    def load_if_needed(self):
        """显式加载集合（初始化/启动阶段调用，供编排器 startup 使用）"""
        if self.collection is not None:
            self._ensure_loaded()