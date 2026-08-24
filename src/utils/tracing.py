"""
OpenTelemetry 全链路追踪
为 Agent 管线的每个阶段记录 Trace/Span
"""
import time
import functools
import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# 简化版 Span（不依赖 OpenTelemetry SDK，可无缝替换）
# 生产环境可替换为 from opentelemetry import trace


class Span:
    """轻量 Span，记录单步操作的开始/结束/耗时/元数据"""

    def __init__(self, name: str, parent: Optional["Span"] = None):
        self.name = name
        self.parent = parent
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.attributes: Dict[str, Any] = {}
        self.status = "ok"
        self.error_message: Optional[str] = None
        self.children: list["Span"] = []

    def set_attribute(self, key: str, value: Any):
        self.attributes[key] = value

    def set_error(self, message: str):
        self.status = "error"
        self.error_message = message

    def add_child(self, child: "Span"):
        self.children.append(child)

    def finish(self):
        self.end_time = time.time()

    @property
    def duration_ms(self) -> float:
        if self.end_time is None:
            return (time.time() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "duration_ms": round(self.duration_ms, 2),
            "status": self.status,
            "error": self.error_message,
            "attributes": self.attributes,
            "children": [c.to_dict() for c in self.children]
        }


class Tracer:
    """
    全局追踪器
    管理 Root Span，提供装饰器和上下文管理器
    """

    def __init__(self):
        self.current_root: Optional[Span] = None
        self._span_stack: list[Span] = []

    @contextmanager
    def start_span(self, name: str, **attrs):
        """上下文管理器：自动开始/结束 Span"""
        parent = self._span_stack[-1] if self._span_stack else None
        span = Span(name, parent=parent)
        for k, v in attrs.items():
            span.set_attribute(k, v)

        if parent:
            parent.add_child(span)

        if not self._span_stack:
            self.current_root = span

        self._span_stack.append(span)

        try:
            yield span
        except Exception as e:
            span.set_error(str(e))
            raise
        finally:
            span.finish()
            self._span_stack.pop()

    def trace(self, span_name: str = None):
        """装饰器：自动追踪函数调用"""
        def decorator(func):
            name = span_name or func.__name__

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with self.start_span(name, function=func.__name__) as span:
                    span.set_attribute("args_count", len(args))
                    span.set_attribute("kwargs_keys", list(kwargs.keys()))
                    result = func(*args, **kwargs)
                    if isinstance(result, dict):
                        span.set_attribute("result_keys", list(result.keys()))
                    return result
            return wrapper
        return decorator

    def get_trace_tree(self) -> Optional[Dict[str, Any]]:
        """获取完整追踪树"""
        if self.current_root:
            return self.current_root.to_dict()
        return None

    def print_trace(self):
        """打印追踪树到日志"""
        tree = self.get_trace_tree()
        if tree:
            self._print_node(tree, 0)

    def _print_node(self, node: dict, depth: int):
        indent = "  " * depth
        status_icon = "✅" if node["status"] == "ok" else "❌"
        logger.info(
            f"{indent}{status_icon} {node['name']} "
            f"({node['duration_ms']}ms)"
        )
        if node.get("attributes"):
            for k, v in node["attributes"].items():
                logger.debug(f"{indent}    {k}: {v}")
        for child in node.get("children", []):
            self._print_node(child, depth + 1)


# 全局单例
tracer = Tracer()
