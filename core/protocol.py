"""可插拔协议：任何 Agent 只要满足这个形状，就能接进 Runtime。

这是整个项目“换 Agent 不改核心代码”的唯一支点。
核心模块只认 Agent.run(task, ctx) -> Trace，不认任何具体实现、任何框架。

换一个 Agent 的全部工作：
    1. 写一个类，有 name 属性；
    2. 实现 run(task, ctx) 返回 Trace；
    3. 注册进 registry。
没有第 4 步，不需要改 core/、evaluator/、memory/ 任何一行。
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from core.task import Task
from core.trace import Trace


@dataclass
class RunContext:
    """执行一条任务时，Runtime 交给 Agent 的全部外部信息。

    experience 是已渲染好的经验文本（可能为空串）。Agent 可以选择注入到提示词，
    也可以忽略——协议不强制 Agent 用经验，但 Runtime 一定把经验递到手上。

    hits 是这次注入的经验 id 列表，用于事后回填“这条经验有没有起作用”：
    注入后任务仍然失败，对应经验记一次反证。
    """

    task: Task
    experience: str = ""
    hits: list[str] = field(default_factory=list)
    # 预算由 Runtime 定，是唯一生效的来源。
    # Agent 自己的 max_steps 只在"不经过 runner、直接调 run()"时兜底，
    # 不许覆盖这里——否则同一批任务里每个 Agent 跑的步数上限都不一样，
    # 报告里却只写一个数，出问题查不清是谁改的。
    max_steps: int = 6
    extra: dict = field(default_factory=dict)

    def has_experience(self) -> bool:
        return bool(self.experience and self.experience.strip())

    def to_dict(self) -> dict:
        return {
            "task_id": self.task.id,
            "experience_chars": len(self.experience or ""),
            "hits": list(self.hits),
            "max_steps": self.max_steps,
        }


@runtime_checkable
class Agent(Protocol):
    """Agent 的协议形状。

    注意 runtime_checkable 只检查属性存在，不检查签名，所以还要过 validate_agent()。
    """

    name: str

    def run(self, task: Task, ctx: RunContext | None = None) -> Trace:  # pragma: no cover - 协议声明
        ...


class AgentContractError(TypeError):
    """Agent 不符合协议时抛出，错误信息直接指出缺什么。"""


def validate_agent(obj: Any) -> None:
    """校验一个对象能否作为 Agent 接入。不通过就抛 AgentContractError。"""
    name = getattr(obj, "name", None)
    if not isinstance(name, str) or not name.strip():
        raise AgentContractError(f"{type(obj).__name__} 缺少非空 name 属性")

    run = getattr(obj, "run", None)
    if not callable(run):
        raise AgentContractError(f"{name!r} 缺少可调用的 run 方法")

    try:
        sig = inspect.signature(run)
    except (TypeError, ValueError):
        return  # 内建/C 实现读不到签名，放行

    params = [
        p
        for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    if len(params) < 2:
        raise AgentContractError(
            f"{name!r} 的 run 需要至少接收 (task, ctx) 两个位置参数，"
            f"当前签名: {sig}"
        )
    if len(params) > 3:
        raise AgentContractError(
            f"{name!r} 的 run 位置参数过多（{len(params)}），应为 (task, ctx)，"
            f"当前签名: {sig}"
        )
