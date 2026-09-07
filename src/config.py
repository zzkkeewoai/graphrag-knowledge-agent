import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    # 抽取参数
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))

    # 图数据库配置
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

    # 向量数据库配置
    MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
    MILVUS_NLIST = int(os.getenv("MILVUS_NLIST", "128"))       # v2.2: 索引 nlist 配置化
    MILVUS_NPROBE = int(os.getenv("MILVUS_NPROBE", "10"))      # v2.2: 检索 nprobe 配置化
    MILVUS_EMBED_BATCH = int(os.getenv("MILVUS_EMBED_BATCH", "64"))  # v2.2: batch embedding

    # 分层 Top-K（v2.2：不同环节用不同 top_k，不共用单一值）
    VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "10"))
    GRAPH_TOP_K = int(os.getenv("GRAPH_TOP_K", "10"))
    FUSION_TOP_K = int(os.getenv("FUSION_TOP_K", "5"))
    RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "3"))

    # 检索超时（v2.2）
    RETRIEVAL_TIMEOUT_SECONDS = float(os.getenv("RETRIEVAL_TIMEOUT_SECONDS", "5"))

    # Agent 编排（v2.0 新增）
    USE_LLM_ANSWER = os.getenv("USE_LLM_ANSWER", "false").lower() == "true"
    USE_SEMANTIC_CACHE = os.getenv("USE_SEMANTIC_CACHE", "true").lower() == "true"
    # v2.1 新增：启用 LLM 兜底路由（处理关键词无法判定的模糊查询）
    USE_LLM_ROUTER = os.getenv("USE_LLM_ROUTER", "false").lower() == "true"
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
    MAX_STEPS = int(os.getenv("MAX_STEPS", "15"))
    CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))
    CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "1000"))   # v2.2: 缓存上限配置化
    CACHE_SIMILARITY_THRESHOLD = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.92"))
    # 知识库版本（v2.2: 知识更新时 bump，旧缓存自动失效）
    KNOWLEDGE_VERSION = os.getenv("KNOWLEDGE_VERSION", "1")

    # ----------------------------------------------------------
    # 启动配置校验（v2.2：缺关键配置快速失败，而不是运行期才报错）
    # ----------------------------------------------------------
    @classmethod
    def validate(cls):
        """校验关键配置，缺失/非法时抛错（fail fast）。

        注意：API Key 只在"需要 LLM 生成"时才强制；纯模板/评测模式可不配置。
        """
        errors = []
        # 数值必须为正
        for name in ("MAX_RETRIES", "MAX_STEPS", "CACHE_TTL", "VECTOR_TOP_K",
                     "GRAPH_TOP_K", "FUSION_TOP_K", "RERANK_TOP_K"):
            val = getattr(cls, name, 0)
            if not isinstance(val, int) or val <= 0:
                errors.append(f"{name}={val} 必须为正整数")
        # 阈值必须在 (0,1)
        for name in ("CONFIDENCE_THRESHOLD", "CACHE_SIMILARITY_THRESHOLD"):
            val = getattr(cls, name, 0)
            if not (0 < float(val) < 1):
                errors.append(f"{name}={val} 必须在 (0,1) 区间")

        if errors:
            raise ValueError("配置校验失败:\n" + "\n".join(f"  - {e}" for e in errors))
        return True