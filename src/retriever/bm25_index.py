"""BM25 关键词检索：给 GraphRAG 补上"倒排"这一路。

为什么需要：
    向量检索擅长语义相近，但对**精确术语**不敏感——"第十五条""净资产 50%"
    这种字面唯一的片段，BM25 往往比 embedding 更准。企业制度/年报语料里
    这类精确表述极多，所以混合检索是标配（多份 JD 里都这么写）。

为什么用字符二元组切词：
    中文没有空格，上 jieba 会引入分词错误、还要维护词典。企业文本里
    "对外担保""募集资金"都是 2-4 字词，**字符二元组**（"对外""外担""担保"）
    对 BM25 已经够用，而且零依赖、不会切错专业术语。

为什么单独建索引而不是塞进向量库：
    两者更新频率和检索方式不同；分开也能单独开关做 A/B 对照。
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_ASCII = re.compile(r"[A-Za-z0-9_.%]+")
_CJK = re.compile(r"[\u4e00-\u9fa5]")


def tokenize(text: str) -> List[str]:
    """中文按字符二元组切，英文数字按整词小写。"""
    if not text:
        return []
    tokens: List[str] = []
    # 英文/数字整词（"bm25"、"50%"、"1.56"）
    tokens.extend(m.group().lower() for m in _ASCII.finditer(text))
    # 中文：取相邻二字
    chars = [c for c in text if _CJK.match(c)]
    tokens.extend(chars[i] + chars[i + 1] for i in range(len(chars) - 1))
    if len(chars) == 1:
        tokens.append(chars[0])
    return tokens


class BM25Index:
    """在一批 (doc_id, chunk_index, text) 上建 BM25 索引。"""

    def __init__(self):
        self.docs: List[Dict[str, Any]] = []
        self._bm25 = None

    def build(self, docs: List[Dict[str, Any]]) -> "BM25Index":
        from rank_bm25 import BM25Okapi

        self.docs = docs
        corpus = [tokenize(d.get("text", "")) for d in docs]
        # 全空语料会让 BM25 除零，兜一下
        if not any(corpus):
            self._bm25 = None
            return self
        self._bm25 = BM25Okapi(corpus)
        logger.info(f"BM25 索引构建完成：{len(docs)} 块")
        return self

    @classmethod
    def from_vector_store(cls, store) -> "BM25Index":
        """直接从本地向量库的块建索引——保证两路检索看到的是同一批文本。"""
        docs = [
            {"doc_id": m["doc_id"], "chunk_index": m["chunk_index"], "text": t}
            for t, m in zip(store.texts, store.metas)
        ]
        return cls().build(docs)

    def search(self, query: str, top_k: int = 8) -> List[Dict[str, Any]]:
        if self._bm25 is None or not self.docs:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        out = []
        for i in order:
            if scores[i] <= 0:
                continue        # 一个词都没命中就不要，避免灌噪声进融合
            d = self.docs[i]
            out.append({
                "doc_id": d["doc_id"],
                "chunk_index": d["chunk_index"],
                "text": d["text"],
                "score": float(scores[i]),
                "source": "bm25",
            })
        return out
