"""最小可运行 Agent：不调用模型、不连服务、行为完全确定。

用途有两个，都不是“凑一个 demo”：
1. 证明协议可用——一个类加一个 run，就能接进 Runtime。
2. 让闭环可被单测证明——注意回答前缀会随“有没有注入经验”变化，
   于是“经验送达后行为改变”这件事可以毫秒级验证，不必花额度、不必碰外部服务。

它不是业务 Agent，接入层该长什么样由它示范，业务逻辑不在这里。
"""

from __future__ import annotations

import time

from core.protocol import RunContext
from core.task import Task
from core.trace import ToolCall, Trace, Turn

PREFIX_PLAIN = "[echo]"
PREFIX_WITH_EXPERIENCE = "[echo+exp]"


class EchoAgent:
    """把每轮输入原样回显，并为每轮记一次假调用。

    name 可在构造时指定，方便在同一个注册表里放多个同类实例做对比。
    """

    name = "echo"

    def __init__(self, name: str = "echo") -> None:
        self.name = name

    def run(self, task: Task, ctx: RunContext | None = None) -> Trace:
        started = int(time.time() * 1000)
        applied = bool(ctx and ctx.has_experience())
        prefix = PREFIX_WITH_EXPERIENCE if applied else PREFIX_PLAIN

        turns: list[Turn] = []
        for i, text in enumerate(task.user_turns()):
            call = ToolCall(
                name="echo",
                args={"text": text},
                call_id=f"t{i}c0",
                output=f"echo:{text}",
            )
            turns.append(
                Turn(
                    index=i,
                    user_input=text,
                    calls=[call],
                    answer=f"{prefix} {text}",
                    steps=1,
                )
            )

        return Trace(
            task_id=task.id,
            agent=self.name,
            turns=turns,
            split=task.split,
            elapsed_ms=max(0, int(time.time() * 1000) - started),
            meta={
                "experience_applied": applied,
                "hits": list(ctx.hits) if ctx else [],
            },
        )


class SilentAgent:
    """一个字都不说、也不调工具的 Agent。

    专门用于验证“轨迹存在但没有内容”这种情况会被如实记录，
    而不是被当成正常完成。
    """

    name = "silent"

    def run(self, task: Task, ctx: RunContext | None = None) -> Trace:
        turns = [
            Turn(index=i, user_input=text, calls=[], answer="")
            for i, text in enumerate(task.user_turns())
        ]
        return Trace(task_id=task.id, agent=self.name, turns=turns, split=task.split)


class BrokenAgent:
    """每次执行都抛异常。用于验证崩溃隔离：单条挂掉不拖垮整批。"""

    name = "broken"

    def __init__(self, message: str = "boom") -> None:
        self.message = message

    def run(self, task: Task, ctx: RunContext | None = None) -> Trace:
        raise RuntimeError(self.message)


class WrongTypeAgent:
    """返回 dict 而不是 Trace。用于验证协议违规会被明确报错，而不是静默通过。"""

    name = "wrong_type"

    def run(self, task: Task, ctx: RunContext | None = None):
        return {"task_id": task.id, "answer": "我返回了个字典"}
