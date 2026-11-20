"""名字 -> Agent 工厂。CLI 只认名字，不 import 任何具体 Agent。

为什么需要这一层：
    scripts/run_loop.py 原来写死 `from agents.shop import make_shop_agent`。
    那意味着"接入一个新 Agent"要改 CLI——CLI 是 Runtime 的入口，
    改它就等于改核心。加一个 Agent 应该只加一行注册。

为什么存工厂而不是实例：
    Agent 构造要读环境变量（API Key）。在 import 时就构造，
    缺密钥的机器连 `--help` 都打不开。工厂把构造推迟到真要跑的时候。

为什么 import 写在函数里：
    `import agents.builtin` 不该顺带把 langchain 和 httpx 拖进内存，
    更不该在缺密钥时失败。这一点由 tests/test_main_adapter.py 守着。

与 core.registry 的分工：
    这里存的是"能造哪些"（工厂）；自建的 INSTANCES 记"这个进程里造过谁"，只用于自省。
    刻意**不**往 core.registry.DEFAULT 里写：那张表的契约是"默认必须为空"，
    是"删掉整个 agents/ 与 demos/，core 仍可 import 和测试"的运行时护栏。
    接入层往里塞实例，等于让业务 Agent 反向污染核心全局状态。
"""

from __future__ import annotations

from typing import Any, Callable

from core.registry import Registry, UnknownAgentError

# 本进程造过的实例，纯自省用。不碰 core.registry.DEFAULT。
INSTANCES = Registry("builtin")

_FACTORIES: dict[str, Callable[..., Any]] = {}

# 旧名字保持能用：之前几轮真跑用的就是 --agent cold / --agent tuned。
# cs 是 customer 的短别名，给命令行省点打字。
ALIASES = {
    "cold": "shop-cold",
    "tuned": "shop-tuned",
    "shop": "shop-tuned",
    "main_agent": "main",
    "cs": "customer",
}


def register(name: str, factory: Callable[..., Any]) -> None:
    """登记一个工厂。同名重复登记直接报错，不静默覆盖。"""
    if name in _FACTORIES:
        raise ValueError(f"Agent 名字重复: {name!r}")
    _FACTORIES[name] = factory


def names() -> list[str]:
    return sorted(_FACTORIES)


def make(name: str, **kwargs) -> Any:
    """按名字造一个 Agent。kwargs 原样传给工厂。"""
    key = ALIASES.get(name, name)
    try:
        factory = _FACTORIES[key]
    except KeyError:
        available = ", ".join(names()) or "（空）"
        raise UnknownAgentError(
            f"没有名为 {name!r} 的 Agent，当前可用: {available}"
        ) from None

    agent = factory(**kwargs)
    # 登记实例只用于自省（谁被造过、叫什么），失败不该影响执行。
    INSTANCES.register(agent, replace=True)
    return agent


def _shop_factory(variant: str):
    def factory(**kwargs):
        from agents.shop import make_shop_agent

        return make_shop_agent(variant, **kwargs)

    return factory


def _main_factory(**kwargs):
    from agents.main_agent.adapter import MainAgent

    return MainAgent(**kwargs)


def _customer_factory(**kwargs):
    from agents.customer import make_customer_agent

    return make_customer_agent(**kwargs)


register("shop-cold", _shop_factory("cold"))
register("shop-tuned", _shop_factory("tuned"))
register("main", _main_factory)
register("customer", _customer_factory)

DEFAULT_AGENT = "shop-cold"

__all__ = ["register", "names", "make", "ALIASES", "DEFAULT_AGENT"]
