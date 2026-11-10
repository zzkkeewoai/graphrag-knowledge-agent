"""本地向量库：和 MilvusClient 接口完全一致的轻量替代。

【为什么要有它】
    Milvus 需要 etcd + minio + milvus 三个容器，占 3-4GB 内存；
    在 8GB 的开发机上经常起不来，而且镜像拉取还容易被墙。
    但本项目对向量库的全部需求只有三件事：
        建集合 / 写入块 / 按余弦相似度取 top-k
    这三件事用 numpy + 一个 npz 文件就能做，几百 MB 内存。

【为什么接口要和 MilvusClient 一模一样】
    调用方（hybrid_retriever / api / 灌库脚本）不该关心底层是哪个库。
    同接口 = 换一行构造代码就能切，业务代码零改动。
    这也是"可替换设计"的最小实践：**不引入抽象层，只保证签名一致**。

【它不做什么（划清界限）】
    - 不做近似检索（ANN），是暴力余弦相似度 → 几千条以内够快，上万条会慢
    - 不做分布式、不做增量索引
    - 不支持并发写
    这些正是 Milvus 存在的理由。所以：**开发/评测用它，生产切 Milvus**。

用法：
    store = LocalVectorStore()
    store.connect()
    store.create_collection("document_chunks", drop_existing=True)
    store.insert_chunks("org", ["段落1", "段落2"])
    hits = store.search("谁审批采购", top_k=5)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

import numpy as np

from src.config import Config

logger = logging.getLogger(__name__)

DEFAULT_DIR = "data/local_vectors"


class _CollectionShim:
    """只为了兼容 `milvus.collection.num_entities` 这种写法。"""

    def __init__(self, num_entities: int = 0):
        self.num_entities = num_entities


class LocalVectorStore:
    """本地向量库（暴力余弦检索）。接口与 MilvusClient 保持一致。"""

    def __init__(self, host: str = "", port: int = 0, store_dir: str = DEFAULT_DIR):
        # 故意保留 host/port 参数：调用方写的是 MilvusClient(host=..., port=...)，
        # 换成这个类时不需要改调用代码。
        self.store_dir = store_dir
        os.makedirs(self.store_dir, exist_ok=True)

        self._encoder = None
        self._name = "document_chunks"
        self.collection = None
        self.texts: List[str] = []
        self.metas: List[Dict[str, Any]] = []      # {doc_id, chunk_index}
        self.vectors: np.ndarray | None = None      # (N, dim)
        # 只有"真的改过数据"才允许写回磁盘。
        # 否则一个只读进程（例如查库的服务）退出时会把文件覆盖成它启动时
        # 加载的旧快照 —— 这正是"灌完 27 条、跑一次服务又变回 23 条"的原因。
        self._dirty = False

    # ---------------- 编码器（懒加载：不装模型也能 import 这个模块） ----------------
    @property
    def encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
            logger.info(f"加载 embedding 模型：{model}")
            self._encoder = SentenceTransformer(model)
        return self._encoder

    # ---------------- 生命周期 ----------------
    def connect(self):
        logger.info(f"本地向量库就绪：{self.store_dir}（无需外部服务）")
        return self

    def disconnect(self):
        self._save()

    # ---------------- 集合 ----------------
    def _paths(self, name: str) -> tuple[str, str]:
        return (
            os.path.join(self.store_dir, f"{name}.json"),
            os.path.join(self.store_dir, f"{name}.npy"),
        )

    def create_collection(self, collection_name: str = "document_chunks", drop_existing: bool = False):
        self._name = collection_name
        meta_path, vec_path = self._paths(collection_name)

        if drop_existing:
            for p in (meta_path, vec_path):
                if os.path.exists(p):
                    os.remove(p)
            self._dirty = True
            logger.warning(f"本地集合 {collection_name} 已清空（drop_existing=True）")

        if os.path.exists(meta_path) and os.path.exists(vec_path):
            data = json.loads(open(meta_path, encoding="utf-8").read())
            self.texts = data["texts"]
            self.metas = data["metas"]
            self.vectors = np.load(vec_path)
            logger.info(f"本地集合 {collection_name} 已加载：{len(self.texts)} 条")
        else:
            self.texts, self.metas, self.vectors = [], [], None
            logger.info(f"本地集合 {collection_name} 已创建（空）")

        self.collection = _CollectionShim(len(self.texts))
        return self.collection

    # ---------------- 写入 ----------------
    def chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """按 token 切片（与 MilvusClient 的实现保持一致）。"""
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text)
        chunks = []
        step = max(1, chunk_size - overlap)
        for i in range(0, len(tokens), step):
            chunks.append(enc.decode(tokens[i:i + chunk_size]))
        return chunks

    def insert_chunks(self, doc_id: str, chunks: List[str], chunk_offset: int = 0) -> int:
        if not chunks:
            return 0

        vecs = self.encoder.encode(chunks, show_progress_bar=False, normalize_embeddings=True)

        self.texts.extend(chunks)
        for i, _ in enumerate(chunks):
            self.metas.append({"doc_id": doc_id, "chunk_index": chunk_offset + i})
        self.vectors = vecs if self.vectors is None else np.vstack([self.vectors, vecs])

        self.collection = _CollectionShim(len(self.texts))
        self._dirty = True
        self._save()
        logger.info(f"本地向量库写入 {len(chunks)} 个块（累计 {len(self.texts)}）")
        return len(chunks)

    # ---------------- 检索 ----------------
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.texts or self.vectors is None:
            logger.warning("本地向量库为空，返回空结果")
            return []

        q = self.encoder.encode([query], show_progress_bar=False, normalize_embeddings=True)[0]
        # 向量已归一化，点积即余弦相似度
        scores = self.vectors @ q
        idx = np.argsort(-scores)[:top_k]

        return [
            {
                "chunk_text": self.texts[i],
                "doc_id": self.metas[i]["doc_id"],
                "chunk_index": self.metas[i]["chunk_index"],
                "score": float(scores[i]),
            }
            for i in idx
        ]

    # ---------------- 兼容空实现 ----------------
    def load_if_needed(self):
        return None

    def _ensure_loaded(self):
        return None

    # ---------------- 持久化 ----------------
    def _save(self):
        if not self._dirty:
            return  # 没改过就不写，避免只读进程覆盖别人的写入
        if not self.texts:
            return
        meta_path, vec_path = self._paths(self._name)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"texts": self.texts, "metas": self.metas}, f, ensure_ascii=False)
        np.save(vec_path, self.vectors)
