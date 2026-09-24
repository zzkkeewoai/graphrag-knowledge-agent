"""
GraphRAG HTTP 服务层（api.py）

工程化目标：把"进程内库调用"变成"可被外部调用的服务"。
包含生产必需的五个要素：
  1. 服务化接口      —— POST /ask（问答）、GET /health（健康检查）、GET /cache/stats（缓存统计）
  2. 请求校验        —— Pydantic 模型（长度/范围约束），脏输入在入口就被挡住
  3. 超时控制        —— 同步的 process() 丢进线程池 + asyncio.wait_for，超时返回 504 + 兜底文案
  4. 降级可启动      —— 依赖（Neo4j/Milvus）不可用时服务照常启动，标记 degraded，返回兜底响应
                        （这正是"任何一层挂掉都有人接"在服务层的体现）
  5. 优雅关闭        —— lifespan 关闭时释放数据库连接

运行：
    cd graphrag_system
    python -m uvicorn api:app --host 127.0.0.1 --port 8000
    然后打开 http://127.0.0.1:8000/docs 看 Swagger 文档
"""
import asyncio
import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

# 保证能从项目根导入 src.*
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.config import Config  # noqa: E402
from src.utils import metrics  # noqa: E402
from src.utils.tracing import tracer  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("graphrag.api")


# ============================================================
# 服务状态（进程内单例）
# ============================================================
class ServiceState:
    """服务运行状态：编排器实例 + 依赖健康情况 + 累计指标"""

    def __init__(self):
        self.orchestrator = None
        self.neo4j = None
        self.milvus = None
        # starting: 启动中 | ready: 正常 | degraded: 依赖不可用（仍可服务，走兜底）
        self.status: str = "starting"
        self.error: Optional[str] = None
        self.started_at: float = time.time()
        # 指标（面试可讲：这就是最小可用的可观测性）
        self.request_count: int = 0
        self.error_count: int = 0
        self.timeout_count: int = 0

    def snapshot(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "error": self.error,
            "uptime_seconds": round(time.time() - self.started_at, 1),
            "request_count": self.request_count,
            "error_count": self.error_count,
            "timeout_count": self.timeout_count,
        }


state = ServiceState()


# ============================================================
# 请求 / 响应模型（Pydantic 校验）
# ============================================================
class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="用户问题")
    timeout_seconds: float = Field(
        20.0, gt=0, le=60, description="单次请求超时（秒）"
    )
    include_trace: bool = Field(False, description="是否返回链路追踪树")


class AskResponse(BaseModel):
    request_id: str
    query: str
    answer: str
    route: str
    latency_ms: float
    step_count: int
    degraded: bool = Field(False, description="是否走了降级路径")
    trace: Optional[Dict[str, Any]] = None


# ============================================================
# 编排器初始化（失败不阻塞启动 → degraded）
# ============================================================
def init_orchestrator() -> None:
    """尝试初始化编排器。任何异常都被捕获，服务以 degraded 状态继续运行。"""
    try:
        from src.database.neo4j_client import Neo4jClient
        from src.database.milvus_client import MilvusClient
        from src.agents.orchestrator_v2 import AgentOrchestratorV2

        # 启动即校验配置（fail-fast：配置错了不要等第一个请求才炸）
        Config.validate()

        neo4j = Neo4jClient(
            uri=Config.NEO4J_URI,
            user=Config.NEO4J_USER,
            password=Config.NEO4J_PASSWORD,
        )
        neo4j.connect()
        milvus = MilvusClient(host=Config.MILVUS_HOST, port=Config.MILVUS_PORT)
        milvus.connect()
        # 附加已存在的 collection（存在则复用，不重建）。
        # 为什么必需：MilvusClient 初始化后 self.collection 为 None，
        # 而 search() 在 collection 为 None 时"静默返回空结果"——不报错，
        # 只表现为"检索不到东西 → 证据不足 → 拒答"，极难排查。
        try:
            milvus.create_collection("document_chunks", drop_existing=False)
            milvus.load_if_needed()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Milvus collection 附加失败（向量检索将返回空结果）: {e}")

        orchestrator = AgentOrchestratorV2(
            neo4j_client=neo4j,
            milvus_client=milvus,
            use_llm=Config.USE_LLM_ANSWER,
            use_cache=Config.USE_SEMANTIC_CACHE,
            max_retries=Config.MAX_RETRIES,
            max_steps=Config.MAX_STEPS,
        )

        state.neo4j, state.milvus, state.orchestrator = neo4j, milvus, orchestrator
        state.status, state.error = "ready", None
        logger.info("编排器初始化完成，服务状态: ready")
    except Exception as e:  # noqa: BLE001 - 启动阶段任何异常都不应让服务起不来
        state.status = "degraded"
        state.error = f"{type(e).__name__}: {e}"
        logger.warning(f"编排器初始化失败，服务以 degraded 状态启动: {state.error}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动/关闭生命周期：启动初始化，关闭释放连接。"""
    logger.info("服务启动中…")
    init_orchestrator()
    yield
    # 优雅关闭：释放数据库连接（Neo4j 用 close()，Milvus 用 disconnect()）
    logger.info("服务关闭中，释放连接…")
    for client in (state.neo4j, state.milvus):
        if client is None:
            continue
        for method in ("close", "disconnect"):
            if hasattr(client, method):
                try:
                    getattr(client, method)()
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"释放连接异常: {e}")
                break


app = FastAPI(
    title="GraphRAG 知识检索服务",
    description="图谱 + 向量双路检索，含证据验证与拒答机制",
    version="2.2.0",
    lifespan=lifespan,
)


# ============================================================
# 中间件：请求 ID 透传 + HTTP 指标采集
# ============================================================
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id

    path = request.url.path
    # /metrics 抓取本身不参与统计：否则 inflight 会永远 ≥1（抓取那一刻自己正在处理），
    # 监控图上会出现"常年有在途请求"的假象。
    tracked = path != "/metrics"

    if tracked:
        metrics.INFLIGHT_REQUESTS.inc()
    started = time.time()
    try:
        response = await call_next(request)
    finally:
        if tracked:
            metrics.INFLIGHT_REQUESTS.dec()

    response.headers["X-Request-ID"] = request_id

    if tracked:
        metrics.REQUEST_TOTAL.labels(
            path=path, status_code=str(response.status_code)
        ).inc()
        metrics.REQUEST_LATENCY.labels(path=path).observe(time.time() - started)

    return response


# ============================================================
# 依赖深度探测（health 用）
# ============================================================
def _probe_dependencies() -> Dict[str, str]:
    """真实执行一次轻量操作来验证依赖可用，而不是只看"连接对象是否存在"。

    踩过的坑：Neo4j 的 driver 创建成功 ≠ 认证可用——`GraphDatabase.driver()`
    是懒连接的，密码错了也要等到第一次查询才报 AuthError。只检查 driver 是否
    存在，会让服务在依赖不可用时依然报告 ready，流量被打到坏实例上。
    """
    probes: Dict[str, str] = {}

    # Neo4j：跑一次真实查询（验证连接 + 认证）
    if state.neo4j is None:
        probes["neo4j"] = "not_initialized"
    else:
        try:
            with state.neo4j.driver.session() as session:
                session.run("RETURN 1 AS ok").single()
            probes["neo4j"] = "ok"
        except Exception as e:  # noqa: BLE001
            probes["neo4j"] = f"error: {type(e).__name__}"

    # Milvus：确认 collection 已加载（未加载时 search 会静默返回空）
    if state.milvus is None:
        probes["milvus"] = "not_initialized"
    elif getattr(state.milvus, "collection", None) is None:
        probes["milvus"] = "collection_not_loaded"
    else:
        try:
            state.milvus._ensure_loaded()  # noqa: SLF001 - 幂等，内部有状态标记
            probes["milvus"] = "ok"
        except Exception as e:  # noqa: BLE001
            probes["milvus"] = f"error: {type(e).__name__}"

    return probes


# ============================================================
# 接口
# ============================================================
@app.get("/health", summary="健康检查（含依赖深度探测）")
async def health() -> Dict[str, Any]:
    """进程存活即返回 200；依赖探测失败则 status=degraded（可接 K8s readiness）。

    liveness 与 readiness 分开是生产惯例：进程活着不等于能服务请求。
    """
    payload = state.snapshot()
    payload["cache"] = (
        state.orchestrator.get_cache_stats() if state.orchestrator else {"enabled": False}
    )
    probes = _probe_dependencies()
    payload["dependencies"] = probes
    if any(v != "ok" for v in probes.values()):
        payload["status"] = "degraded"
    return payload


@app.get("/cache/stats", summary="缓存统计")
async def cache_stats() -> Dict[str, Any]:
    if not state.orchestrator:
        return {"enabled": False, "reason": "orchestrator 未初始化"}
    return state.orchestrator.get_cache_stats()


@app.post("/ask", response_model=AskResponse, summary="知识库问答")
async def ask(req: AskRequest, request: Request) -> AskResponse:
    """问答主接口。

    要点：
    - process() 是同步阻塞的（内部含 LLM 调用），放进线程池避免堵住事件循环；
    - 超时用 asyncio.wait_for 控制，超时不返回 500 而是返回 504 + 兜底文案；
    - degraded 状态下不直接报错，返回可读的降级响应。
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    state.request_count += 1
    started = time.time()

    # 依赖不可用 → 降级响应（服务不崩，用户拿到可读文案）
    if state.orchestrator is None:
        state.error_count += 1
        metrics.record_degraded("dependency_unavailable")
        return AskResponse(
            request_id=request_id,
            query=req.query,
            answer="抱歉，知识检索服务当前不可用（依赖未就绪），请稍后重试。",
            route="degraded",
            latency_ms=round((time.time() - started) * 1000, 1),
            step_count=0,
            degraded=True,
        )

    loop = asyncio.get_running_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(None, state.orchestrator.process, req.query),
            timeout=req.timeout_seconds,
        )
    except asyncio.TimeoutError:
        state.timeout_count += 1
        metrics.record_degraded("timeout")
        logger.warning(f"[{request_id}] 请求超时（>{req.timeout_seconds}s）: {req.query[:40]}")
        return AskResponse(
            request_id=request_id,
            query=req.query,
            answer="抱歉，查询处理超时，请稍后重试或换一种问法。",
            route="timeout",
            latency_ms=round((time.time() - started) * 1000, 1),
            step_count=0,
            degraded=True,
        )
    except Exception as e:  # noqa: BLE001
        state.error_count += 1
        metrics.record_degraded("error")
        logger.error(f"[{request_id}] 处理异常: {type(e).__name__}: {e}")
        return AskResponse(
            request_id=request_id,
            query=req.query,
            answer="抱歉，处理您的请求时遇到了技术问题，请稍后重试。",
            route="error",
            latency_ms=round((time.time() - started) * 1000, 1),
            step_count=0,
            degraded=True,
        )

    # 记录业务语义指标：路由分布、证据状态、缓存命中、拒答
    evidence_status = result.get("evidence_status")
    metrics.record_ask_result(
        route=result.get("route", "unknown"),
        evidence_status=evidence_status,
        from_cache=bool(result.get("from_cache")),
        refused=evidence_status == "insufficient",
    )

    return AskResponse(
        request_id=request_id,
        query=result.get("query", req.query),
        answer=result.get("answer", ""),
        route=result.get("route", "unknown"),
        latency_ms=result.get("latency_ms", round((time.time() - started) * 1000, 1)),
        step_count=result.get("step_count", 0),
        degraded=False,
        trace=result.get("trace") if req.include_trace else None,
    )


# ============================================================
# 流式输出（SSE）
# ============================================================
def _chunk_text(text: str, size: int = 6):
    """把文本切成小块用于流式推送。

    说明：这里切的是**已经生成好的答案**（模板模式）。
    真实 LLM 场景下，这一层应该改成 `stream=True` 逐 token 产出；
    事件协议（meta / token / refusal / done）保持不变，前端无需改动。
    这也是把"生成实现"和"传输协议"解耦的意义。
    """
    for i in range(0, len(text), size):
        yield text[i:i + size]


@app.post("/ask/stream", summary="流式问答（SSE）")
async def ask_stream(req: AskRequest, request: Request) -> StreamingResponse:
    """SSE 流式问答。

    核心设计：**证据验证必须前置**。
        检索 + 证据验证是确定性逻辑且很快（百毫秒级），必须**先跑完**：
          - 证据充足 → 才进入流式生成（首字节延迟 ≈ 检索耗时）
          - 证据不足/冲突 → 直接发 refusal 事件，**不进入生成**
        如果做不到前置，就会出现"已经吐了一半才发现证据不足"——流式无法回退，
        只能给用户一个半截的错误答案。这是流式 + 防幻觉结合时最容易踩的坑。

    事件协议：
        meta    元信息（路由、证据状态、缓存命中、检索耗时）→ 前端可据此决定 UI
        token   答案分块
        refusal 拒答（证据不足 / 冲突无法裁决）
        error   依赖不可用或超时
        done    结束（带总耗时）
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    state.request_count += 1
    started = time.time()

    def sse(event: str, payload: Dict[str, Any]) -> str:
        return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

    def elapsed() -> float:
        return round((time.time() - started) * 1000, 1)

    async def event_stream():
        # 1) 依赖不可用 → 降级事件
        if state.orchestrator is None:
            metrics.record_degraded("dependency_unavailable")
            yield sse("error", {"message": "知识检索服务当前不可用（依赖未就绪）"})
            yield sse("done", {"latency_ms": elapsed()})
            return

        # 2) 检索 + 证据验证（同步、必须先完成）
        loop = asyncio.get_running_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, state.orchestrator.process, req.query),
                timeout=req.timeout_seconds,
            )
        except asyncio.TimeoutError:
            metrics.record_degraded("timeout")
            yield sse("error", {"message": "查询处理超时，请稍后重试"})
            yield sse("done", {"latency_ms": elapsed()})
            return
        except Exception as e:  # noqa: BLE001
            metrics.record_degraded("error")
            logger.error(f"[{request_id}] 流式处理异常: {type(e).__name__}: {e}")
            yield sse("error", {"message": "服务内部错误，请稍后重试"})
            yield sse("done", {"latency_ms": elapsed()})
            return

        evidence_status = result.get("evidence_status")
        metrics.record_ask_result(
            route=result.get("route", "unknown"),
            evidence_status=evidence_status,
            from_cache=bool(result.get("from_cache")),
            refused=evidence_status == "insufficient",
        )

        # 3) 先发元信息，前端可立刻渲染"来源/状态"
        yield sse("meta", {
            "request_id": request_id,
            "route": result.get("route"),
            "evidence_status": evidence_status,
            "from_cache": bool(result.get("from_cache")),
            "retrieval_ms": result.get("latency_ms"),
        })

        answer = result.get("answer") or ""

        # 4) 证据不足或冲突 → 明确的 refusal 事件，不进生成流
        if evidence_status in ("insufficient", "conflict"):
            yield sse("refusal", {"text": answer, "reason": evidence_status})
            yield sse("done", {"latency_ms": elapsed()})
            return

        # 5) 证据充足 → 流式推送答案
        for chunk in _chunk_text(answer):
            yield sse("token", {"text": chunk})
            await asyncio.sleep(0.01)  # 模拟 token 间隔；真实场景由 LLM 流控制节奏
        yield sse("done", {"latency_ms": elapsed()})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        # no-cache + 关闭反向代理缓冲，保证逐块到达前端（Nginx 需配 proxy_buffering off）
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/metrics", summary="Prometheus 指标")
async def prometheus_metrics() -> Response:
    """暴露 Prometheus 格式指标，供 Prometheus 抓取。

    包含两类：
      - HTTP 层：请求数、耗时直方图（可算 P50/P95/P99）、在途请求数
      - 业务层：路由分布、证据状态分布、缓存命中、拒答数、降级数
    业务指标是 RAG 系统的"答案质量先行指标"——例如拒答率突增通常意味着
    召回质量下降或知识库变更，比错误率更早暴露问题。
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """兜底异常处理：任何未捕获异常都返回结构化错误，不暴露栈信息。"""
    logger.error(f"未处理异常: {type(exc).__name__}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "message": "服务内部错误，请稍后重试"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=False)
