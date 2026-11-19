"""验证阶段：同一批 holdout 任务跑两遍，看经验有没有让结果发生变化。

为什么必须跑两遍，而不能跟历史报告比：
历史报告是别的提示词、别的模型版本、别的服务端状态下跑出来的。
拿"今天的注入组"去比"上周的 tuned"，任何差异都分不清是经验带来的、
还是模型换版带来的。唯一干净的对照是：同一个 Agent、同一批任务、
同一个时刻，唯一变量是提示词里有没有那段经验。

三处刻意的设计：

1. 基线先跑、注入组后跑，两侧都只跑 holdout。
   holdout 从头到尾不参与提炼（learn 只碰 fit），所以它是干净的验证集。
   先跑基线还有一个实际好处：如果基线阶段就大面积 infra_error，
   能立刻发现是链路问题，不用等注入组跑完才知道。

2. 经验 id 由 provider 自己记，不信 Agent 写在 Trace.meta 里的东西。
   meta 是 Agent 的自由字段，不同 Agent 写不写、写成什么样都不一定。
   这里包一层 provider，在返回前把 (task_id -> ids) 落进本地字典，
   于是回填效果这件事不依赖任何 Agent 的实现细节。

3. 注入计数与结果回填分开调用，且都是显式调用。
   record_injection 是分母（被注入过几次），record_outcome 是分子。
   两件事都在这里做，不藏在 provider 里——provider 只该返回文本，
   带副作用会让"为什么这条经验被退休了"变得查不清。

一句必须说清的话：这一层不保证"通过率会上升"。样本量是十几条，
temperature=0 也只是"尽量确定"，单条翻转可能就是噪声。所以产出的是
逐题翻转表与归因分布，不是一句"提升 X%"。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from core.protocol import Agent
from core.runner import run_tasks
from core.task import SPLIT_HOLDOUT, Task
from core.trace import Trace, now_ms
from evaluator.verdict import Verdict, evaluate

from memory.inject import make_provider

SideHook = Callable[[str, Task, Trace, Verdict], None]


@dataclass
class SideResult:
    """一组执行的结果。verdicts 按 task_id 索引，因为对比是按题对齐的。"""

    label: str
    traces: list[Trace] = field(default_factory=list)
    verdicts: dict[str, Verdict] = field(default_factory=dict)
    hits: dict[str, dict] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    elapsed_ms: int = 0

    @property
    def total(self) -> int:
        return len(self.verdicts)

    @property
    def passed(self) -> int:
        return sum(1 for v in self.verdicts.values() if v.passed)

    @property
    def rate(self) -> float:
        return round(self.passed / self.total, 4) if self.total else 0.0

    @property
    def infra_failed(self) -> int:
        """链路故障条数：限流、连接、超时。

        这类失败不是 Agent 的能力问题，混进通过率里会让数字失真——
        上一轮就出现过同一道题在基线判 infra_error、在注入组判 PASS，
        差值因此虚高一条。
        """
        return sum(1 for v in self.verdicts.values() if v.failure_type == "infra_error")

    @property
    def evaluable(self) -> int:
        return self.total - self.infra_failed

    @property
    def rate_evaluable(self) -> float:
        """可判口径的通过率：把链路故障从分母里摘出去。"""
        return round(self.passed / self.evaluable, 4) if self.evaluable else 0.0

    def by_failure(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for verdict in self.verdicts.values():
            if not verdict.passed and verdict.failure_type:
                out[verdict.failure_type] = out.get(verdict.failure_type, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))

    def summary(self) -> str:
        head = f"{self.label} {self.passed}/{self.total}"
        if not self.by_failure():
            return head
        tail = "  ".join(f"{k} {v}" for k, v in self.by_failure().items())
        return f"{head}（{tail}）"

    def per_task(self) -> dict[str, dict]:
        """逐题明细：判定 + 这一题实际调了哪些工具。"""
        calls_by_task = {t.task_id: [c.name for c in t.all_calls()] for t in self.traces}
        out: dict[str, dict] = {}
        for tid, verdict in sorted(self.verdicts.items()):
            out[tid] = {
                "passed": bool(verdict.passed),
                "failure_type": verdict.failure_type,
                "fail_turn": verdict.fail_turn,
                "calls": calls_by_task.get(tid, []),
            }
        return out

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "elapsed_ms": self.elapsed_ms,
            "total": self.total,
            "passed": self.passed,
            "rate": self.rate,
            "infra_failed": self.infra_failed,
            "evaluable": self.evaluable,
            "rate_evaluable": self.rate_evaluable,
            "by_failure": self.by_failure(),
            "errors": list(self.errors),
            "per_task": self.per_task(),
            "hits": {tid: dict(info) for tid, info in sorted(self.hits.items())},
        }


@dataclass
class CompareResult:
    """基线 vs 注入组。delta 只做减法，不下结论。"""

    baseline: SideResult
    injected: SideResult
    touched: int = 0
    retired: int = 0

    def flips(self) -> dict[str, list[str]]:
        """逐题翻转。只比两侧都判过的题——一侧判分失败时不参与比较。"""
        base, inj = self.baseline.verdicts, self.injected.verdicts
        common = sorted(set(base) & set(inj))
        up: list[str] = []
        down: list[str] = []
        same_pass: list[str] = []
        same_fail: list[str] = []
        for tid in common:
            b, i = bool(base[tid].passed), bool(inj[tid].passed)
            if not b and i:
                up.append(tid)
            elif b and not i:
                down.append(tid)
            elif i:
                same_pass.append(tid)
            else:
                same_fail.append(tid)
        return {"up": up, "down": down, "same_pass": same_pass, "same_fail": same_fail}

    def coverage(self) -> list[dict]:
        """每条 holdout 到底注入了什么。零注入时这一栏就是"为什么没变化"的答案。"""
        out: list[dict] = []
        for tid in sorted(self.injected.verdicts):
            info = self.injected.hits.get(tid) or {}
            out.append(
                {
                    "task_id": tid,
                    "injected": bool(info.get("ids")),
                    "exp_ids": list(info.get("ids") or []),
                    "chars": int(info.get("chars") or 0),
                }
            )
        return out

    def hit_counts(self) -> dict[str, int]:
        """每条经验被注入到几条任务上。用来分辨"只有一条经验在起作用"。"""
        out: dict[str, int] = {}
        for info in self.injected.hits.values():
            for exp_id in info.get("ids") or []:
                out[exp_id] = out.get(exp_id, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))

    def injected_tasks(self) -> int:
        return sum(1 for i in self.injected.hits.values() if i.get("ids"))

    def delta(self) -> dict:
        return {
            "baseline_rate": self.baseline.rate,
            "injected_rate": self.injected.rate,
            "rate_delta": round(self.injected.rate - self.baseline.rate, 4),
            "baseline_passed": self.baseline.passed,
            "injected_passed": self.injected.passed,
            "flips_up": len(self.flips()["up"]),
            "flips_down": len(self.flips()["down"]),
            # 与顶层 coverage（逐题明细）区分：这里是"有多少条任务真的被注入了"的比例。
            "coverage_rate": (
                round(self.injected_tasks() / self.injected.total, 4)
                if self.injected.total
                else 0.0
            ),
        }

    def to_dict(self) -> dict:
        return {
            "stage": "verify",
            "baseline": self.baseline.to_dict(),
            "injected": self.injected.to_dict(),
            "delta": self.delta(),
            "flips": self.flips(),
            "coverage": self.coverage(),
            "hit_counts": self.hit_counts(),
            "memory": {"touched": self.touched, "retired": self.retired},
        }


def _run_side(
    agent: Agent,
    tasks: Sequence[Task],
    provider,
    label: str,
    allowed_tools: set[str] | None,
    on_judged: SideHook | None,
    max_steps: int,
) -> SideResult:
    by_id = {t.id: t for t in tasks}
    verdicts: dict[str, Verdict] = {}
    errors: list[str] = []
    started = now_ms()

    def hook(trace: Trace) -> None:
        task = by_id.get(trace.task_id)
        if task is None:
            errors.append(f"{trace.task_id}: 轨迹找不到对应任务，未判分")
            return
        try:
            verdict = evaluate(trace, task, allowed_tools=allowed_tools)
        except Exception as exc:
            errors.append(f"{trace.task_id}: 判定异常 {type(exc).__name__}: {exc}")
            return
        verdicts[trace.task_id] = verdict
        if on_judged is not None:
            on_judged(label, task, trace, verdict)

    result = run_tasks(
        agent, tasks, experience_provider=provider, max_steps=max_steps, on_trace=hook
    )
    return SideResult(
        label=label,
        traces=result.traces,
        verdicts=verdicts,
        errors=errors,
        elapsed_ms=max(0, now_ms() - started),
    )


def verify(
    agent: Agent,
    tasks: Sequence[Task],
    store,
    *,
    allowed_tools: set[str] | None = None,
    on_judged: SideHook | None = None,
    top_k: int = 6,
    min_score: float | None = None,
    max_steps: int = 6,
    record: bool = True,
) -> CompareResult:
    """holdout 跑两遍并对比。record=False 时不改动经验库（便于只看不改）。"""
    items = list(tasks or [])
    holdout = [t for t in items if t.split == SPLIT_HOLDOUT]
    if not holdout:
        raise ValueError("没有 holdout 侧任务，验证阶段无从开始（检查任务划分）")

    baseline = _run_side(agent, holdout, None, "baseline", allowed_tools, on_judged, max_steps)

    hits_log: dict[str, dict] = {}
    inner = make_provider(store, top_k=top_k, min_score=min_score)

    def tracking(task: Task) -> tuple[str, list[str]]:
        text, ids = inner(task)
        hits_log[task.id] = {"ids": list(ids), "chars": len(text or "")}
        return text, ids

    injected = _run_side(
        agent, holdout, tracking, "injected", allowed_tools, on_judged, max_steps
    )
    injected.hits = hits_log

    touched = 0
    retired = 0
    if record:
        # 先记分母，再记分子。顺序反了会让 confidence 短暂失真，
        # 而 confidence 会进检索打分。
        for info in hits_log.values():
            if info["ids"]:
                store.record_injection(info["ids"])
        for task_id, info in hits_log.items():
            if not info["ids"]:
                continue
            verdict = injected.verdicts.get(task_id)
            if verdict is None:
                continue
            outcome = store.record_outcome(info["ids"], passed=bool(verdict.passed))
            touched += int(outcome.get("touched") or 0)
            retired += int(outcome.get("retired") or 0)

    return CompareResult(
        baseline=baseline, injected=injected, touched=touched, retired=retired
    )


__all__ = ["SideResult", "CompareResult", "verify"]
