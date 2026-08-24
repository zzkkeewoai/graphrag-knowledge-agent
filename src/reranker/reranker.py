import logging
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class BGEReranker:
    """
    轻量 Reranker：复用 embedding 模型做余弦相似度重排。

    为什么不用 CrossEncoder（如 bge-reranker-v2-m3）？
    - bge-reranker-v2-m3 需 2.2GB 内存，8GB 机器上会 OOM/Segfault
    - base 版也要 ~1.1GB 且需从 HF 下载，耗时很长
    - 用已有的 MiniLM 模型做 bi-encoder 重排，零额外开销，效果足够用

    为什么用 Reranker（精排）？
    - 向量检索（双塔模型）快速但粗糙
    - 用 Reranker 在最后一步做"精排"，弥补粗排的精度不足
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """复用轻量 embedding 模型，384维，已缓存在本地"""
        logger.info(f"加载 Reranker embedding 模型: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info("Reranker 模型加载完成")

    def rerank(self, query: str, documents: List[Dict], top_k: int = 3) -> List[Dict]:
        """
        对文档列表进行重排（基于余弦相似度）
        """
        if not documents:
            return []

        # 提取文本
        texts = []
        for doc in documents:
            if 'text' in doc:
                texts.append(doc['text'])
            elif 'chunk_text' in doc:
                texts.append(doc['chunk_text'])
            else:
                texts.append(str(doc))

        # 计算 query 和每个文档的 embedding
        query_emb = self.model.encode([query], normalize_embeddings=True)[0]
        doc_embs = self.model.encode(texts, normalize_embeddings=True)

        # 余弦相似度（因为已归一化，点积即余弦相似度）
        scores = np.dot(doc_embs, query_emb)

        # 添加分数并排序
        for i, doc in enumerate(documents):
            doc['rerank_score'] = float(scores[i])

        reranked = sorted(documents, key=lambda x: x.get('rerank_score', 0), reverse=True)

        return reranked[:top_k]