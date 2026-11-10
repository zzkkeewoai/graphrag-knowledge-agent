import logging
import os
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# 默认用中文向量模型。
# 原来写死 all-MiniLM-L6-v2（英文模型），中文查询会被编码成近乎相同的向量——
# 实测两个完全不同的中文问题余弦相似度 > 0.92，既导致向量检索乱召回，
# 又导致语义缓存把 A 问题的答案返回给 B 问题。
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")


class BGEReranker:
    """
    轻量 Reranker：复用 embedding 模型做余弦相似度重排。

    为什么不用 CrossEncoder（如 bge-reranker-v2-m3）？
    - bge-reranker-v2-m3 需 2.2GB 内存，8GB 机器上会 OOM/Segfault
    - base 版也要 ~1.1GB 且需从 HF 下载，耗时很长
    - 用已有的 embedding 模型做 bi-encoder 重排，零额外开销

    为什么用 Reranker（精排）？
    - 向量检索（双塔模型）快速但粗糙
    - 用 Reranker 在最后一步做"精排"，弥补粗排的精度不足

    注意：它和向量库必须用**同一个**模型，否则查询向量和库里的向量
    不在同一空间，相似度没有意义。所以两者都读 EMBEDDING_MODEL。
    """

    def __init__(self, model_name: str | None = None):
        model_name = model_name or DEFAULT_EMBEDDING_MODEL
        logger.info(f"加载 Reranker embedding 模型: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info("Reranker 模型加载完成")

    def rerank(self, query: str, documents: List[Dict]) -> List[Dict]:
        """对候选做相关性重排打分，**只排序、不截断**。

        为什么去掉 top_k：
            打分是"相关性判断"，截断是"预算决策"，两件事不该由同一个组件决定。
            以前这里自带 `return reranked[:top_k]`，结果截断规则散落在两处
            （精排里一批、拼 prompt 时又一批），改配置只改一处、效果没变还不知道为什么。
            现在统一在调用方（orchestrator）做两处截断：
              · rerank 前：RERANK_POOL 候选池上限（成本保护）
              · rerank 后：CONTEXT_TOKEN_BUDGET token 预算（上下文预算）
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

        return reranked