"""多 LLM Provider 热备验证（不需要真实 API key，用打桩模拟故障切换）

验证 4 个场景：
    1. 主 provider 故障 → 自动切到备用 provider
    2. 全部 provider 故障 → 抛 ProviderError（上层据此降级）
    3. stats 不泄露 api_key
    4. from_env 只加载配置了 key 的 provider，且保持优先级顺序

运行：python tests/sanity_check_providers.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.utils.llm_provider import (  # noqa: E402
    LLMProviderManager,
    ProviderConfig,
    ProviderError,
)

results = []


def check(name: str, ok: bool, detail: str = ""):
    results.append((name, ok))
    print(f"  {'✅' if ok else '❌'} {name}" + (f" —— {detail}" if detail else ""))


class StubManager(LLMProviderManager):
    """把 _call 打桩：指定名字的 provider 一律失败，其余返回模拟回答。"""

    def __init__(self, providers, fail_names):
        super().__init__(providers)
        self.fail_names = set(fail_names)

    def _call(self, provider, prompt, temperature, max_tokens):  # noqa: D102
        if provider.name in self.fail_names:
            raise RuntimeError(f"模拟 {provider.name} 故障")
        return f"[{provider.name}] 模拟回答"


def make_providers(*names):
    return [
        ProviderConfig(name=n, api_key=f"sk-{n}-secret", base_url=f"https://{n}.example/v1", model=f"{n}-model")
        for n in names
    ]


def main() -> int:
    print("=" * 64)
    print("场景 1：主 provider 故障 → 自动切换备用")
    print("=" * 64)
    mgr = StubManager(make_providers("primary", "backup"), fail_names={"primary"})
    answer = mgr.chat("测试问题")
    check("故障时返回备用 provider 的回答", answer == "[backup] 模拟回答", answer)
    check("主 provider fail_count = 1", mgr.providers[0].fail_count == 1)
    check("备用 provider call_count = 1", mgr.providers[1].call_count == 1)

    print()
    print("=" * 64)
    print("场景 2：全部 provider 故障 → 抛 ProviderError")
    print("=" * 64)
    mgr2 = StubManager(make_providers("a", "b"), fail_names={"a", "b"})
    raised = False
    try:
        mgr2.chat("测试问题")
    except ProviderError as e:
        raised = True
        detail = str(e)
    check("抛出 ProviderError（上层据此走降级）", raised, detail if raised else "")
    check("两个 provider 的 fail_count 均为 1",
          mgr2.providers[0].fail_count == 1 and mgr2.providers[1].fail_count == 1)

    print()
    print("=" * 64)
    print("场景 3：stats 不泄露 api_key（安全）")
    print("=" * 64)
    stats = mgr.stats
    dump = str(stats)
    check("stats 中不含 secret", "secret" not in dump)
    check("stats 含 provider 名称与模型", "primary" in dump and "primary-model" in dump)
    check("stats.active 为主 provider", stats["active"] == "primary")

    print()
    print("=" * 64)
    print("场景 4：from_env 只加载配置了 key 的 provider，并保持顺序")
    print("=" * 64)
    # 造一段环境变量：deepseek 有 key、qwen 没 key、zhipu 有 key
    os.environ["LLM_PROVIDERS"] = "deepseek,qwen,zhipu"
    os.environ["DEEPSEEK_API_KEY"] = "sk-deepseek-test"
    os.environ.pop("QWEN_API_KEY", None)
    os.environ["ZHIPU_API_KEY"] = "sk-zhipu-test"

    env_mgr = LLMProviderManager.from_env()
    names = [p.name for p in env_mgr.providers]
    check("跳过未配 key 的 qwen", "qwen" not in names, f"实际={names}")
    check("保持配置顺序 deepseek → zhipu", names == ["deepseek", "zhipu"], f"实际={names}")
    if names == ["deepseek", "zhipu"]:
        check("deepseek 使用默认 base_url",
              env_mgr.providers[0].base_url == "https://api.deepseek.com")

    print()
    print("=" * 64)
    passed = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"  结果: {passed} 通过, {len(results) - passed} 失败")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
