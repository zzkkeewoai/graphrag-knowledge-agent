"""GraphRAG 服务压测脚本（locust）

用法：
    # 无界面快速跑（推荐先跑这个）
    locust -f tests/locustfile.py --host http://127.0.0.1:8000 \
           --headless -u 10 -r 2 -t 30s --only-summary

    # Web UI 模式（可实时看曲线）
    locust -f tests/locustfile.py --host http://127.0.0.1:8000

参数含义：
    -u 并发用户数   -r 每秒启动用户数   -t 持续时长

设计说明（为什么要分两个任务）：
    同一个问题重复问，第二次会命中【语义缓存】，延迟应显著低于完整检索——
    这正是项目里"语义缓存降本"的可量化验证。
    而真实业务里大多数 query 是各不相同的，所以权重设为 1:3。
    压测报告里会用 name= 把两类请求分开统计，可以直接对比 P95。
"""
import json
import random

from locust import HttpUser, between, task

# 覆盖图谱命中 / 向量命中 / 拒答（关系不匹配）三类场景
QUERIES = [
    "智能客服系统用了哪些技术？",
    "智能客服系统依赖哪个项目？",
    "B项目依赖什么？",
    "C项目用了什么通信协议？",
    "张三负责哪个项目？",
]

SAME_QUERY = "智能客服系统用了哪些技术？"


class GraphRAGUser(HttpUser):
    """模拟一个查询知识库的用户。"""

    wait_time = between(0.1, 0.5)

    @task(1)
    def ask_cached(self):
        """重复问同一个问题 → 期望命中语义缓存（延迟低）。"""
        self._ask(SAME_QUERY, name="/ask [cache]")

    @task(3)
    def ask_varied(self):
        """不同问法 → 走完整检索链路（真实成本）。"""
        self._ask(random.choice(QUERIES), name="/ask [full]")

    def _ask(self, query: str, name: str):
        payload = {"query": query, "timeout_seconds": 30}
        with self.client.post(
            "/ask",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json; charset=utf-8"},
            name=name,
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")
                return
            try:
                body = resp.json()
            except Exception as e:  # noqa: BLE001
                resp.failure(f"响应不是合法 JSON: {e}")
                return
            if not body.get("answer"):
                resp.failure("answer 为空")
            elif body.get("degraded"):
                # 降级响应不算"成功查询"——压测中要能看见它
                resp.failure(f"走了降级路径 route={body.get('route')}")
            else:
                resp.success()
