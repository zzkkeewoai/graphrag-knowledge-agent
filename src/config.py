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

    # Agent 编排（v2.0 新增）
    USE_LLM_ANSWER = os.getenv("USE_LLM_ANSWER", "false").lower() == "true"
    USE_SEMANTIC_CACHE = os.getenv("USE_SEMANTIC_CACHE", "true").lower() == "true"
    # v2.1 新增：启用 LLM 兜底路由（处理关键词无法判定的模糊查询）
    USE_LLM_ROUTER = os.getenv("USE_LLM_ROUTER", "false").lower() == "true"
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
    MAX_STEPS = int(os.getenv("MAX_STEPS", "15"))
    CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))
    CACHE_SIMILARITY_THRESHOLD = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.92"))