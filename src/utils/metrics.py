"""Prometheus 指标定义（AI 应用的语义化指标）

为什么需要这个模块：
    普通 Web 监控（QPS/延迟/错误率）只能说明"服务活着"。RAG/Agent 系统还必须看
    **业务语义指标**，因为很多问题不会表现为报错，而是表现为"答案质量下降"：

      - 拒答率突增      → 通常是召回质量下降或知识库变更（最早的预警信号）
      - 证据状态分布    → sufficient 占比掉了说明检索/图谱出问题
      - 缓存命中率      → 直接关联成本与延迟
      - 路由分布        → global 占比异常说明查询模式变了
      - LLM token 消耗  → 成本归因

设计约定：
    - 所有指标集中在此处定义，避免各处重复注册（prometheus_client 重复注册同名
      指标会抛 ValueError）
    - HTTP 层指标（requests/latency/inflight）放中间件记录
    - 业务指标（evidence/route/cache/refusal）在 /ask 处理完成后记录
"""
from prometheus_client import Counter, Gauge, Histogram

# ============================================================
# HTTP 层指标
# ============================================================
REQUEST_TOTAL = Counter(
    "graphrag_requests_total",
    "HTTP 请求总数",
    ["path", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "graphrag_request_latency_seconds",
    "HTTP 请求耗时（秒）",
    ["path"],
    # 分桶覆盖：缓存命中 ~10ms / 检索 ~200ms / 长尾补召回 ~2s
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0),
)

INFLIGHT_REQUESTS = Gauge(
    "graphrag_inflight_requests",
    "正在处理中的请求数",
)

# ============================================================
# 业务语义指标
# ============================================================
ROUTER_DECISION = Counter(
    "graphrag_router_decision_total",
    "路由决策分布",
    ["route"],
)

EVIDENCE_STATUS = Counter(
    "graphrag_evidence_status_total",
    "证据验证结果分布（sufficient/insufficient/conflict）",
    ["status"],
)

CACHE_ACCESS = Counter(
    "graphrag_cache_access_total",
    "缓存访问结果",
    ["result"],  # hit / miss
)

REFUSAL_TOTAL = Counter(
    "graphrag_refusals_total",
    "拒答次数（证据不足或冲突无法裁决）",
)

DEGRADED_TOTAL = Counter(
    "graphrag_degraded_total",
    "降级响应次数",
    ["reason"],  # timeout / error / dependency_unavailable
)

# ============================================================
# 记录辅助函数（业务层调用，避免各处写 labels 拼接）
# ============================================================
def record_ask_result(
    route: str,
    evidence_status: str = None,
    from_cache: bool = False,
    refused: bool = False,
) -> None:
    """在一次 /ask 完成后记录业务指标。"""
    if route:
        ROUTER_DECISION.labels(route=route).inc()
    if evidence_status:
        EVIDENCE_STATUS.labels(status=evidence_status).inc()
    CACHE_ACCESS.labels(result="hit" if from_cache else "miss").inc()
    if refused:
        REFUSAL_TOTAL.inc()


def record_degraded(reason: str) -> None:
    """记录降级响应（超时/异常/依赖不可用）。"""
    DEGRADED_TOTAL.labels(reason=reason).inc()
