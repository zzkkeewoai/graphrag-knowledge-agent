"""
语义缓存：基于 Embedding 相似度缓存高频查询结果
减少重复 LLM 调用，降低延迟和成本
"""
import time
import hashlib
import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class SemanticCache:
    """
    语义缓存：不只是精确匹配，语义相似的查询也能命中缓存

    两层缓存架构：
    - L1 精确缓存（dict）：O(1) 查找，命中率高时极快
    - L2 语义缓存（embedding 相似度）：匹配语义相似的查询
    """

    def __init__(
        self,
        encoder=None,               # SentenceTransformer 编码器（复用）
        similarity_threshold: float = 0.92,
        ttl: int = 3600,            # 缓存过期时间（秒）
        max_size: int = 1000,       # 最大缓存条目
        knowledge_version: str = "1"  # v2.2: 知识库版本，知识更新时 bump → 旧缓存失效
    ):
        if encoder is None:
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.encoder = encoder
        self.similarity_threshold = similarity_threshold
        self.ttl = ttl
        self.max_size = max_size
        self.knowledge_version = str(knowledge_version)

        # L1: 精确缓存 {md5_hash: (result, timestamp, knowledge_version)}
        self.exact_cache: Dict[str, Tuple[Any, float, str]] = {}

        # L2: 语义缓存 {md5_hash: (query_text, embedding, result, timestamp, knowledge_version)}
        self.semantic_cache: Dict[str, Tuple[str, np.ndarray, Any, float, str]] = {}

        self.hit_count = 0
        self.miss_count = 0

    def _hash(self, text: str) -> str:
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def _is_expired(self, timestamp: float) -> bool:
        return (time.time() - timestamp) > self.ttl

    def _is_stale_version(self, version) -> bool:
        """v2.2: 缓存条目的知识库版本与当前版本不一致 = 旧数据（知识已更新）"""
        return str(version) != self.knowledge_version

    def _evict_if_full(self):
        """缓存满了，移出最旧的条目"""
        if len(self.semantic_cache) >= self.max_size:
            oldest_key = min(
                self.semantic_cache.keys(),
                key=lambda k: self.semantic_cache[k][3]
            )
            del self.semantic_cache[oldest_key]
            logger.debug(f"缓存淘汰: {oldest_key[:8]}...")

        if len(self.exact_cache) >= self.max_size:
            oldest_key = min(
                self.exact_cache.keys(),
                key=lambda k: self.exact_cache[k][1]
            )
            del self.exact_cache[oldest_key]

    def get(self, query: str) -> Optional[Any]:
        """
        查询缓存：L1 精确 → L2 语义
        返回: 缓存结果或 None
        v2.2: 命中还需校验 knowledge_version —— 知识更新后旧缓存视为 miss
        """
        # L1: 精确匹配
        query_hash = self._hash(query)
        if query_hash in self.exact_cache:
            result, timestamp, version = self.exact_cache[query_hash]
            if self._is_stale_version(version):
                # 知识已更新，旧缓存作废
                del self.exact_cache[query_hash]
            elif not self._is_expired(timestamp):
                self.hit_count += 1
                logger.debug(f"L1 精确缓存命中: {query[:50]}...")
                return result
            else:
                del self.exact_cache[query_hash]

        # L2: 语义匹配
        try:
            query_emb = self.encoder.encode([query], normalize_embeddings=True)[0]

            best_similarity = 0
            best_key = None
            best_result = None

            for key, (cached_query, cached_emb, result, timestamp, version) in self.semantic_cache.items():
                if self._is_expired(timestamp) or self._is_stale_version(version):
                    continue   # 过期或版本过旧，跳过
                sim = float(np.dot(query_emb, cached_emb))  # 余弦相似度
                if sim > best_similarity:
                    best_similarity = sim
                    best_key = key
                    best_result = result

            if best_similarity >= self.similarity_threshold and best_result is not None:
                self.hit_count += 1
                logger.debug(f"L2 语义缓存命中 (相似度={best_similarity:.3f}): {query[:50]}...")
                return best_result
        except Exception as e:
            logger.warning(f"语义缓存查询异常: {e}")

        self.miss_count += 1
        return None

    def set(self, query: str, result: Any):
        """写入缓存（v2.2: 记录当前 knowledge_version）"""
        query_hash = self._hash(query)
        now = time.time()

        # L1: 精确缓存
        self.exact_cache[query_hash] = (result, now, self.knowledge_version)

        # L2: 语义缓存
        try:
            query_emb = self.encoder.encode([query], normalize_embeddings=True)[0]
            self._evict_if_full()
            self.semantic_cache[query_hash] = (query, query_emb, result, now, self.knowledge_version)
        except Exception as e:
            logger.warning(f"语义缓存写入异常: {e}")

    def clear(self):
        """清空所有缓存"""
        self.exact_cache.clear()
        self.semantic_cache.clear()
        self.hit_count = 0
        self.miss_count = 0

    @property
    def hit_rate(self) -> float:
        total = self.hit_count + self.miss_count
        if total == 0:
            return 0.0
        return self.hit_count / total

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "exact_cache_size": len(self.exact_cache),
            "semantic_cache_size": len(self.semantic_cache),
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": f"{self.hit_rate:.2%}",
            "similarity_threshold": self.similarity_threshold,
            "ttl": self.ttl,
            "knowledge_version": self.knowledge_version,
        }
