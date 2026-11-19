"""任务驱动层：把一批任务喂给一个 Agent，收下轨迹。

这一层只做三件事：调 Agent、兜住崩溃、把任务自带的划分盖到轨迹上。
它不判分（判分在 evaluator），不检索经验（检索在 memory），不认任何具体 Agent。

两个刻意的设计：

1. 单条崩溃隔离。
   Agent 抛异常、返回错类型，都转成 status=crashed 的轨迹继续往下跑。
   一批任务里挂一条，剩下的仍然有结果；否则一次外部故障就把整批作废。

2. 经验用回调传入，不直接依赖经验库。
   依赖方向是 memory -> core（经验库要读轨迹）。core 反过来调 memory 就成环。
   所以这里只认一个 (Task) -> (文本, 命中 id 列表) 的函数，谁来提供都行。
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from core.protocol import Agent, RunContext, validate_agent
from core.task import Task
from core.trace import (
    STATUS_CRASHED,
    STATUS_OK,
    Trace,
    now_ms,
    save_traces,
)

# (渲染好的经验文本, 命中的经验 id)
ExperienceProvider = Callable[[Task], "tuple[str, list[str]]"]
# 每跑完一条回调一次，供 CLI 打印进度
ProgressHook = Callable[[Trace], None]


@dataclass
class RunResult:
    """一批任务的执行结果。不含任何通过与否的判断。"""

    traces: list[Trace] = field(default_factory=list)
    elapsed_ms: int = 0

    def by_status(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for t in self.traces:
            out[t.status] = out.get(t.status, 0) + 1
        return out

    def crashed(self) -> list[Trace]:
        return [t for t in self.traces if t.status == STATUS_CRASHED]

    def summary(self) -> dict:
        calls = sum(len(t.all_calls()) for t in self.traces)
        return {
            "tasks": len(self.traces),
            "evaluable": sum(1 for t in self.traces if t.evaluable),
            "turns": sum(len(t.turns) for t in self.traces),
            "calls": calls,
            "by_status": self.by_status(),
            "elapsed_ms": self.elapsed_ms,
        }


def _crash_trace(agent: Agent, task: Task, error: str, meta: dict | None = None, elapsed: int = 0) -> Trace:
    merged = {"task_input": task.input}
    if meta:
        merged.update(meta)
    return Trace(
        task_id=task.id,
        agent=getattr(agent, "name", "unknown"),
        split=task.split,
        status=STATUS_CRASHED,
        error=error,
        elapsed_ms=max(0, elapsed),
        meta=merged,
    )


def run_task(
    agent: Agent,
    task: Task,
    *,
    experience: str = "",
    hits: Sequence[str] | None = None,
    max_steps: int = 6,
    extra: dict | None = None,
) -> Trace:
    """跑一条任务，永远返回 Trace（异常不外抛）。

    agent 返回的轨迹会被就地校正两处：
    - split 以任务为准（划分属于任务，不允许 Agent 自己声明）；
    - task_id 不一致时改写为任务 id，并在 meta.warnings 留下痕迹。
    """
    ctx = RunContext(
        task=task,
        experience=experience or "",
        hits=list(hits or []),
        max_steps=max_steps,
        extra=dict(extra or {}),
    )
    started = now_ms()
    try:
        trace = agent.run(task, ctx)
    except Exception as exc:  # 外部服务、网络、模型权限都在这里兜住
        return _crash_trace(
            agent,
            task,
            f"{type(exc).__name__}: {exc}",
            meta={"traceback": traceback.format_exc(limit=6)},
            elapsed=now_ms() - started,
        )

    if not isinstance(trace, Trace):
        return _crash_trace(
            agent,
            task,
            f"Agent 返回了 {type(trace).__name__}，协议要求 Trace",
            elapsed=now_ms() - started,
        )

    if trace.task_id != task.id:
        trace.meta.setdefault("warnings", []).append(
            f"task_id 不一致: Agent 返回 {trace.task_id!r}，任务为 {task.id!r}"
        )
        trace.task_id = task.id
    trace.split = task.split
    if not trace.agent:
        trace.agent = getattr(agent, "name", "unknown")

    if not trace.elapsed_ms:
        trace.elapsed_ms = max(0, now_ms() - started)
    return trace


def run_tasks(
    agent: Agent,
    tasks: Iterable[Task],
    *,
    experience_provider: ExperienceProvider | None = None,
    max_steps: int = 6,
    on_trace: ProgressHook | None = None,
) -> RunResult:
    """跑一批任务。单条失败不影响后续。"""
    validate_agent(agent)
    started = now_ms()
    traces: list[Trace] = []

    for task in tasks:
        experience: str = ""
        hits: list[str] = []
        provider_error = ""
        if experience_provider is not None:
            try:
                experience, hits = experience_provider(task)
            except Exception as exc:
                provider_error = f"{type(exc).__name__}: {exc}"

        trace = run_task(
            agent,
            task,
            experience=experience,
            hits=hits,
            max_steps=max_steps,
        )
        if provider_error:
            trace.meta.setdefault("warnings", []).append(f"经验提供者失败: {provider_error}")

        traces.append(trace)
        if on_trace is not None:
            on_trace(trace)

    return RunResult(traces=traces, elapsed_ms=max(0, now_ms() - started))


def run_suite(
    agent: Agent,
    tasks: Sequence[Task],
    *,
    split: str | None = None,
    experience_provider: ExperienceProvider | None = None,
    max_steps: int = 6,
    on_trace: ProgressHook | None = None,
    out_path=None,
) -> RunResult:
    """按划分筛选后整批执行，可选落盘为 jsonl。

    split 为 None 表示全都跑。筛选依据是任务自带字段，不看文件名。
    """
    selected = [t for t in tasks if split is None or t.split == split]
    result = run_tasks(
        agent,
        selected,
        experience_provider=experience_provider,
        max_steps=max_steps,
        on_trace=on_trace,
    )
    if out_path is not None and result.traces:
        save_traces(result.traces, out_path)
    return result


def evaluable_trace(trace: Trace) -> bool:
    """辅助判断：轨迹是否具备被评估的前提（状态正常且至少一轮）。

    这不是判分，只是过滤掉无法评价的东西。真正的对错在 evaluator。
    """
    return trace.status == STATUS_OK and trace.evaluable
