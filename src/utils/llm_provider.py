"""多 LLM Provider 热备：主供应商故障时自动切换备用

为什么需要：
    只接一家 LLM API = **单点故障**。供应商限流、欠费、区域性故障、模型下线
    都会让整个问答系统不可用——而问答系统挂了，用户是直接感知的。
    生产做法是配置**有序 provider 列表**，逐个尝试，全挂了才走最终降级。

配置（环境变量）：
    LLM_PROVIDERS=deepseek,qwen        # 顺序即优先级
    DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL / DEEPSEEK_MODEL
    QWEN_API_KEY     / QWEN_BASE_URL     / QWEN_MODEL
    LLM_TIMEOUT_SECONDS=30             # 单个 provider 的超时

行为约定：
    - 按顺序尝试，**第一个成功即返回**（不做并发竞速，避免成本翻倍）
    - 单 provider 失败（网络/鉴权/限流/超时）→ 记录并切下一个
    - 全部失败 → 抛 ProviderError，由上层走降级（模板生成 / 兜底文案）
    - 记录每个 provider 的调用与失败次数，供监控与容量评估
"""
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# 各 provider 的默认参数（可用环境变量覆盖）
DEFAULTS: Dict[str, Dict[str, str]] = {
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
    },
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4-flash",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
    },
}


class ProviderError(RuntimeError):
    """所有 provider 均不可用（上层据此走降级）。"""


@dataclass
class ProviderConfig:
    name: str
    api_key: str
    base_url: str
    model: str
    call_count: int = 0
    fail_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        # 注意：绝不返回 api_key（避免泄露到日志/接口）
        return {
            "name": self.name,
            "base_url": self.base_url,
            "model": self.model,
            "call_count": self.call_count,
            "fail_count": self.fail_count,
        }


class LLMProviderManager:
    """按优先级顺序调用多个 LLM provider，失败自动切换。"""

    def __init__(self, providers: List[ProviderConfig], timeout: float = 30.0):
        self.providers = providers
        self.timeout = timeout
        if not providers:
            logger.warning("未配置任何可用的 LLM provider（缺少 *_API_KEY）")

    # ----------------------------------------------------------
    # 构建
    # ----------------------------------------------------------
    @classmethod
    def from_env(cls) -> "LLMProviderManager":
        """从环境变量构建 provider 列表（按 LLM_PROVIDERS 顺序）。"""
        raw = os.getenv("LLM_PROVIDERS", "deepseek")
        names = [n.strip().lower() for n in raw.split(",") if n.strip()]
        timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))

        providers: List[ProviderConfig] = []
        for name in names:
            defaults = DEFAULTS.get(name, {})
            prefix = name.upper()
            api_key = os.getenv(f"{prefix}_API_KEY") or ""
            if not api_key:
                # 没配 key 的直接跳过（而不是留一个必然失败的 provider 拖慢切换）
                logger.info(f"provider {name} 未配置 API key，跳过")
                continue
            providers.append(
                ProviderConfig(
                    name=name,
                    api_key=api_key,
                    base_url=os.getenv(f"{prefix}_BASE_URL", defaults.get("base_url", "")),
                    model=os.getenv(f"{prefix}_MODEL", defaults.get("model", "")),
                )
            )

        if not providers:
            logger.warning("LLM_PROVIDERS 里没有任何配置了 API key 的 provider")
        else:
            logger.info(f"LLM provider 列表（按优先级）: {[p.name for p in providers]}")
        return cls(providers, timeout=timeout)

    # ----------------------------------------------------------
    # 调用
    # ----------------------------------------------------------
    def chat(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 500,
    ) -> str:
        """顺序尝试各 provider，返回第一个成功的回答；全失败抛 ProviderError。"""
        if not self.providers:
            raise ProviderError("没有可用的 LLM provider（请配置 *_API_KEY）")

        errors: List[str] = []
        for provider in self.providers:
            try:
                text = self._call(provider, prompt, temperature, max_tokens)
                provider.call_count += 1
                return text
            except Exception as e:  # noqa: BLE001 - 任何失败都应触发切换
                provider.fail_count += 1
                errors.append(f"{provider.name}: {type(e).__name__}: {e}")
                logger.warning(f"provider {provider.name} 调用失败，切换下一个: {e}")

        raise ProviderError("所有 LLM provider 均失败 → " + " | ".join(errors))

    def _call(
        self,
        provider: ProviderConfig,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """单次 provider 调用（独立方法，便于测试时打桩）。"""
        from openai import OpenAI

        client = OpenAI(
            api_key=provider.api_key,
            base_url=provider.base_url,
            timeout=self.timeout,
        )
        resp = client.chat.completions.create(
            model=provider.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    # ----------------------------------------------------------
    # 可观测性
    # ----------------------------------------------------------
    @property
    def stats(self) -> Dict[str, Any]:
        """provider 状态快照（供 /health 或监控使用）。"""
        return {
            "count": len(self.providers),
            "active": self.providers[0].name if self.providers else None,
            "providers": [p.to_dict() for p in self.providers],
            "timeout_seconds": self.timeout,
        }
