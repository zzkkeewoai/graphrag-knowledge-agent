"""基于 Redis 的分布式语义缓存

为什么需要（相对于进程内 dict 缓存）：
    1. **多实例共享**——服务水平扩容后，进程内缓存不共享，每个实例都要重新
       建立缓存，命中率被稀释（N 个实例 ≈ 命中率降到 1/N）；
    2. **重启不丢**——进程重启后进程内缓存清零，流量瞬间全打到检索后端；
    3. **TTL 跨实例一致**——用 Redis 原生过期，不需要各自维护时间戳。

接口与 `SemanticCache` 完全一致（get / set / stats / hit_rate），
所以上层（orchestrator）可以按配置无感切换。

设计要点：
    - L1 精确：`graphrag:cache:exact:{md5}` → JSON（answer / documents / version）
    - L2 语义：`graphrag:cache:semantic:{md5}` → JSON（query / embedding / answer / version）
      另有索引集合 `graphrag:cache:index`（存所有 md5，供 L2 遍历）
    - TTL 用 Redis EXPIRE；knowledge_version 不一致视为失效
    - **降级**：Redis 不可用时自动回退到进程内 `SemanticCache`，
      缓存层故障绝不能影响问答主链路

方案边界（诚实说明）：
    L2 语义匹配需要遍历缓存条目算余弦相似度。小规模（数百~数千条）用
    SCAN + 应用层计算完全够用；规模再上去应改用 **Redis Stack 的向量检索**
    （RediSearch VSS）或直接复用 Milvus 存缓存向量，而不是自己遍历。
"""
import json
import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
import redis
from redis.exceptions import RedisError

from src.utils.semantic_cache import SemanticCache

logger = logging.getLogger(__name__)

KEY_PREFIX = "graphrag:cache"
EXACT_PREFIX = f"{KEY_PREFIX}:exact"
SEMANTIC_PREFIX = f"{KEY_PREFIX}:semantic"
INDEX_KEY = f"{KEY_PREFIX}:index"
STAT_HIT = f"{KEY_PREFIX}:stat:hit"
STAT_MISS = f"{KEY_PREFIX}:stat:miss"


class RedisSemanticCache:
    """Redis 版语义缓存（接口兼容 SemanticCache，Redis 故障时自动降级到进程内缓存）"""

    def __init__(
        self,
        encoder,
        host: str = "localhost",
        port: int = 6379,
        similarity_threshold: float = 0.92,
        ttl: int = 3600,
        max_size: int = 1000,
        knowledge_version: str = "1",
        db: int = 0,
    ):
        self.encoder = encoder
        self.similarity_threshold = similarity_threshold
        self.ttl = ttl
        self.max_size = max_size
        self.knowledge_version = str(knowledge_version)
        self.backend = "redis"

        # 降级用的进程内缓存（Redis 不可用时接管）
        self._fallback = SemanticCache(
            encoder=encoder,
            similarity_threshold=similarity_threshold,
            ttl=ttl,
            max_size=max_size,
            knowledge_version=knowledge_version,
        )

        try:
            self.client = redis.Redis(
                host=host, port=port, db=db,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            self.client.ping()
            logger.info(f"Redis 语义缓存已连接: {host}:{port}/{db}")
        except RedisError as e:
            self.client = None
            self.backend = "in-process(fallback)"
            logger.warning(f"Redis 连接失败，语义缓存降级为进程内实现: {e}")

    # ----------------------------------------------------------
    # 内部工具
    # ----------------------------------------------------------
    @staticmethod
    def _hash(text: str) -> str:
        import hashlib
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _degrade(self, op: str, err: Exception) -> None:
        """运行期 Redis 故障 → 标记降级并回退进程内缓存。"""
        logger.warning(f"Redis {op} 失败，切换进程内缓存: {err}")
        self.backend = "in-process(fallback)"
        self.client = None

    def _embed(self, text: str) -> Optional[List[float]]:
        try:
            vec = self.encoder.encode([text], normalize_embeddings=True)[0]
            return [float(x) for x in vec]
        except Exception as e:  # noqa: BLE001
            logger.warning(f"缓存编码失败: {e}")
            return None

    def _is_fresh(self, payload: Dict[str, Any]) -> bool:
        return str(payload.get("version")) == self.knowledge_version

    # ----------------------------------------------------------
    # 读
    # ----------------------------------------------------------
    def get(self, query: str) -> Optional[Any]:
        if self.client is None:
            return self._fallback.get(query)

        key_hash = self._hash(query)
        try:
            # L1 精确匹配
            raw = self.client.get(f"{EXACT_PREFIX}:{key_hash}")
            if raw:
                payload = json.loads(raw)
                if self._is_fresh(payload):
                    self.client.incr(STAT_HIT)
                    return payload.get("result")
                # 版本过期 → 删除并继续走 L2
                self.client.delete(f"{EXACT_PREFIX}:{key_hash}")

            # L2 语义匹配
            query_emb = self._embed(query)
            if query_emb is None:
                self.client.incr(STAT_MISS)
                return None
            qv = np.asarray(query_emb, dtype=np.float32)

            best_sim, best_result = 0.0, None
            members = list(self.client.smembers(INDEX_KEY))
            if members:
                # 【优化】用 MGET 一次取回所有候选条目。
                # 原来在循环里逐条 GET —— N 条缓存就是 N 次网络往返，
                # 压测中缓存条目增长会让 L2 延迟线性上升（P95 从 64ms 涨到 120ms）。
                # 改成一次 MGET 后，往返次数与条目数无关。
                keys = [f"{SEMANTIC_PREFIX}:{m}" for m in members]
                raws = self.client.mget(keys)
                for member, raw in zip(members, raws):
                    if not raw:
                        self.client.srem(INDEX_KEY, member)  # 过期条目顺手清理
                        continue
                    payload = json.loads(raw)
                    if not self._is_fresh(payload):
                        continue
                    cached_vec = np.asarray(payload.get("embedding") or [], dtype=np.float32)
                    if cached_vec.size != qv.size:
                        continue
                    sim = float(np.dot(qv, cached_vec))
                    if sim > best_sim:
                        best_sim, best_result = sim, payload.get("result")

            if best_sim >= self.similarity_threshold and best_result is not None:
                self.client.incr(STAT_HIT)
                logger.debug(f"L2 语义缓存命中 (相似度={best_sim:.3f})")
                return best_result
        except (RedisError, json.JSONDecodeError, ValueError) as e:
            self._degrade("get", e)
            return self._fallback.get(query)

        self.client.incr(STAT_MISS)
        return None

    # ----------------------------------------------------------
    # 写
    # ----------------------------------------------------------
    def set(self, query: str, result: Any) -> None:
        if self.client is None:
            self._fallback.set(query, result)
            return

        key_hash = self._hash(query)
        try:
            payload = {
                "result": result,
                "version": self.knowledge_version,
                "timestamp": time.time(),
            }
            self.client.setex(f"{EXACT_PREFIX}:{key_hash}", self.ttl, json.dumps(payload, ensure_ascii=False))

            embedding = self._embed(query)
            if embedding is not None:
                sem_payload = {
                    "query": query,
                    "embedding": embedding,
                    "result": result,
                    "version": self.knowledge_version,
                    "timestamp": time.time(),
                }
                self.client.setex(
                    f"{SEMANTIC_PREFIX}:{key_hash}", self.ttl,
                    json.dumps(sem_payload, ensure_ascii=False),
                )
                self.client.sadd(INDEX_KEY, key_hash)
                self.client.expire(INDEX_KEY, self.ttl * 2)
                # 容量控制：超出上限时清理最早的条目（近似 FIFO）
                if self.client.scard(INDEX_KEY) > self.max_size:
                    self._evict_oldest()
        except RedisError as e:
            self._degrade("set", e)
            self._fallback.set(query, result)

    def _evict_oldest(self) -> None:
        """超出容量时按 timestamp 淘汰最旧条目（近似 FIFO）。"""
        try:
            candidates = []
            for member in self.client.smembers(INDEX_KEY):
                raw = self.client.get(f"{SEMANTIC_PREFIX}:{member}")
                if not raw:
                    self.client.srem(INDEX_KEY, member)
                    continue
                candidates.append((json.loads(raw).get("timestamp", 0), member))
            overflow = len(candidates) - self.max_size
            for _, member in sorted(candidates)[: max(overflow, 1)]:
                self.client.delete(f"{SEMANTIC_PREFIX}:{member}")
                self.client.delete(f"{EXACT_PREFIX}:{member}")
                self.client.srem(INDEX_KEY, member)
        except (RedisError, json.JSONDecodeError) as e:
            logger.warning(f"缓存淘汰异常（忽略）: {e}")

    def clear(self) -> None:
        if self.client is None:
            self._fallback.clear()
            return
        try:
            for member in self.client.smembers(INDEX_KEY):
                self.client.delete(f"{SEMANTIC_PREFIX}:{member}")
                self.client.delete(f"{EXACT_PREFIX}:{member}")
            self.client.delete(INDEX_KEY, STAT_HIT, STAT_MISS)
        except RedisError as e:
            self._degrade("clear", e)

    # ----------------------------------------------------------
    # 统计
    # ----------------------------------------------------------
    @property
    def hit_rate(self) -> float:
        if self.client is None:
            return self._fallback.hit_rate
        try:
            hits = int(self.client.get(STAT_HIT) or 0)
            misses = int(self.client.get(STAT_MISS) or 0)
            total = hits + misses
            return hits / total if total else 0.0
        except RedisError:
            return self._fallback.hit_rate

    @property
    def stats(self) -> Dict[str, Any]:
        if self.client is None:
            data = dict(self._fallback.stats)
            data["backend"] = self.backend
            return data
        try:
            hits = int(self.client.get(STAT_HIT) or 0)
            misses = int(self.client.get(STAT_MISS) or 0)
            total = hits + misses
            return {
                "backend": self.backend,
                # 用 SCAN 而非 KEYS：KEYS 会阻塞 Redis 单线程主循环（生产禁忌），
                # SCAN 是游标式增量遍历，不会长时间阻塞。
                "exact_cache_size": sum(
                    1 for _ in self.client.scan_iter(f"{EXACT_PREFIX}:*", count=500)
                ),
                "semantic_cache_size": self.client.scard(INDEX_KEY),
                "hit_count": hits,
                "miss_count": misses,
                "hit_rate": f"{(hits / total if total else 0):.2%}",
                "similarity_threshold": self.similarity_threshold,
                "ttl": self.ttl,
                "knowledge_version": self.knowledge_version,
            }
        except RedisError as e:
            self._degrade("stats", e)
            return self._fallback.stats
