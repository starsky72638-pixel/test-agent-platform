"""名字 -> Agent 的注册表。

刻意做成“空表启动”：
- 默认注册表里没有任何 Agent，核心包不感知任何 demo。
- 谁想跑，谁把自己的 Agent 注册进来。
- 因此删掉整个 demos/ 目录，core / evaluator / memory 仍然可以 import 和测试。

支持两种写法：
    reg = Registry()
    reg.register(MyAgent())            # 直接注册
    @reg.register                      # 装饰器（实例或类都可以）
    class MyAgent: ...
"""

from __future__ import annotations

from typing import Any, Iterator

from core.protocol import Agent, validate_agent


class DuplicateAgentError(ValueError):
    """同名 Agent 已存在。不允许静默覆盖，避免“跑到一半发现换了个实现”。"""


class UnknownAgentError(KeyError):
    """按名字取不到 Agent，错误信息里列出当前可用的名字。"""


class Registry:
    def __init__(self, label: str = "default") -> None:
        self.label = label
        self._agents: dict[str, Agent] = {}

    # -- 注册 -----------------------------------------------------

    def register(self, agent: Any = None, *, name: str | None = None, replace: bool = False):
        """注册一个 Agent。作为装饰器直接传给 register 也可以。"""

        def _do(obj: Any) -> Any:
            validate_agent(obj)
            key = name or obj.name
            if not isinstance(key, str) or not key.strip():
                raise ValueError("注册名不能为空")
            if key in self._agents and not replace:
                raise DuplicateAgentError(
                    f"[{self.label}] 已存在同名 Agent: {key!r}，"
                    f"如需覆盖请显式传 replace=True"
                )
            self._agents[key] = obj
            return obj

        if agent is None:
            return _do
        return _do(agent)

    # -- 取用 -----------------------------------------------------

    def get(self, name: str) -> Agent:
        try:
            return self._agents[name]
        except KeyError:
            available = ", ".join(sorted(self._agents)) or "（空）"
            raise UnknownAgentError(
                f"[{self.label}] 没有名为 {name!r} 的 Agent，当前可用: {available}"
            ) from None

    def names(self) -> list[str]:
        return sorted(self._agents)

    def unregister(self, name: str) -> None:
        if name not in self._agents:
            raise UnknownAgentError(f"[{self.label}] 没有名为 {name!r} 的 Agent")
        del self._agents[name]

    def clear(self) -> None:
        self._agents.clear()

    def __contains__(self, name: object) -> bool:
        return name in self._agents

    def __len__(self) -> int:
        return len(self._agents)

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __repr__(self) -> str:
        return f"Registry(label={self.label!r}, agents={self.names()})"


# 进程级默认表。注意：这里不做任何 import 副作用注册。
DEFAULT = Registry("default")


def register(agent: Any = None, **kwargs):
    """DEFAULT.register 的简写。"""
    return DEFAULT.register(agent, **kwargs)


def get(name: str) -> Agent:
    return DEFAULT.get(name)


def names() -> list[str]:
    return DEFAULT.names()
