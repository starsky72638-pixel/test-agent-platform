"""学习阶段：拿 fit 侧任务跑一遍，把失败变成可检索的经验。

这是闭环里的"左半边"：

    跑 fit -> Trace -> evaluate -> Verdict -> extract -> quality -> store

四件事按固定顺序做，顺序本身是设计的一部分：

1. 只跑 fit 侧。
   传进来的 tasks 是全量的，这里自己按 split 过滤。之所以不让调用方先过滤，
   是因为 filter_batch 的泄漏闸门需要全量 holdout 的答案值；一旦调用方只递
   fit 进来，闸门就退化成"什么都不比对"——这是最容易被无声绕过的一道门。

2. 学习阶段不注入经验。
   experience_provider 固定为 None。理由：只有这样才能保证提炼出的经验对应
   "没有经验时真实犯的错"。一边注入一边学，就分不清某条经验是被哪条旧经验
   催出来的。代价是多轮迭代学习做不到，只能一轮一轮手动跑。

3. 判定异常只记不抛。
   一条轨迹判分时抛异常，记进 errors 然后跳过，不中断整批。
   理由：前面的 API 额度已经花掉了，为一条判分 bug 丢掉整批结果不合理。
   但 errors 必须返回并打印，不能悄悄吞掉——那会把 bug 伪装成"没有失败"。

4. 提炼和入库之间必须有闸门，且闸门口径与入库口径一致。
   filter_batch 拿到的 trace_index / verdict_index 就是这一步刚产生的轨迹与判定，
   证据能不能回查，比的是真数据，不是提炼器自己的说法。

产出 LearnResult 是一份账目：跑了多少、判了多少、提炼出几条、入库几条、
被拒的每条原因是什么。没有这个账，"入库 3 条"到底是提炼出 3 条还是提炼出 20 条
只剩 3 条，事后无法分辨。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from core.protocol import Agent
from core.runner import run_tasks
from core.task import SPLIT_FIT, Task
from core.trace import Trace, now_ms
from evaluator.verdict import Verdict, evaluate

from memory.extract import extract_from_run
from memory.quality import QualityReport, filter_batch, holdout_answer_values
from memory.schema import Experience

# 判定回调：learn 用它把"这一条跑完并且判过了"透给 CLI 打印进度。
# 不放在 run_tasks 的 on_trace 上，是因为那里还没有 Verdict。
JudgedHook = Callable[[Task, Trace, Verdict], None]


@dataclass
class LearnResult:
    """一次学习过程的完整账目。"""

    traces: list[Trace] = field(default_factory=list)
    verdicts: dict[str, Verdict] = field(default_factory=dict)
    candidates: list[Experience] = field(default_factory=list)
    kept: list[Experience] = field(default_factory=list)
    reports: list[QualityReport] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    store_stats: dict = field(default_factory=dict)
    elapsed_ms: int = 0

    @property
    def judged(self) -> int:
        """成功判分的条数。低于 len(traces) 说明有判定异常。"""
        return len(self.verdicts)

    @property
    def passed(self) -> int:
        return sum(1 for v in self.verdicts.values() if v.passed)

    @property
    def rejected(self) -> list[QualityReport]:
        return [r for r in self.reports if not r.ok]

    def by_failure(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for verdict in self.verdicts.values():
            if not verdict.passed and verdict.failure_type:
                out[verdict.failure_type] = out.get(verdict.failure_type, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))

    def lines(self) -> list[str]:
        """给 CLI 直接打印的几行摘要。"""
        out = [
            f"fit {len(self.traces)} 条，判分 {self.judged} 条，通过 {self.passed} 条",
        ]
        if self.by_failure():
            out.append(
                "失败归因："
                + "  ".join(f"{k} {v}" for k, v in self.by_failure().items())
            )
        out.append(
            f"提炼 {len(self.candidates)} 条 -> 入库 {len(self.kept)} 条"
            f"（拒绝 {len(self.rejected)} 条）"
        )
        return out

    def to_dict(self) -> dict:
        return {
            "stage": "learn",
            "elapsed_ms": self.elapsed_ms,
            "fit_tasks": len(self.traces),
            "judged": self.judged,
            "passed": self.passed,
            "by_failure": self.by_failure(),
            "candidates": len(self.candidates),
            "kept": len(self.kept),
            "rejected": [
                {"exp_id": r.exp_id, "reasons": list(r.reasons)} for r in self.rejected
            ],
            "warnings": [
                {"exp_id": r.exp_id, "warnings": list(r.warnings)}
                for r in self.reports
                if r.ok and r.warnings
            ],
            "errors": list(self.errors),
            "experiences": [
                {
                    "exp_id": e.exp_id,
                    "failure_type": e.failure_type,
                    "pattern": e.pattern,
                    "action": e.action,
                    "guard": e.guard,
                    "support": e.support,
                    "tags": list(e.trigger.tags),
                    "shape": e.trigger.shape,
                    "evidence": [ev.to_dict() for ev in e.evidence],
                }
                for e in self.kept
            ],
            "store": dict(self.store_stats),
        }


def learn(
    agent: Agent,
    tasks: Sequence[Task],
    store,
    *,
    allowed_tools: set[str] | None = None,
    on_judged: JudgedHook | None = None,
    max_steps: int = 6,
    now: int | None = None,
) -> LearnResult:
    """跑 fit 侧、判分、提炼、过闸门、入库。

    tasks 必须是**全量**（fit + holdout）。只递 fit 进来会让泄漏闸门失效，
    这里不做拦截，但会通过 store 里经验的 trigger 与来源体现出来。
    """
    started = now_ms()
    items = list(tasks or [])
    fit = [t for t in items if t.split == SPLIT_FIT]
    if not fit:
        raise ValueError("没有 fit 侧任务，学习阶段无从开始（检查任务划分）")

    by_id = {t.id: t for t in fit}
    verdicts: dict[str, Verdict] = {}
    errors: list[str] = []

    def hook(trace: Trace) -> None:
        task = by_id.get(trace.task_id)
        if task is None:
            errors.append(f"{trace.task_id}: 轨迹找不到对应任务，未判分")
            return
        try:
            verdict = evaluate(trace, task, allowed_tools=allowed_tools)
        except Exception as exc:  # 判分 bug 不该毁掉整批已花掉额度的结果
            errors.append(f"{trace.task_id}: 判定异常 {type(exc).__name__}: {exc}")
            return
        verdicts[trace.trace_id] = verdict
        if on_judged is not None:
            on_judged(task, trace, verdict)

    result = run_tasks(agent, fit, max_steps=max_steps, on_trace=hook)

    triples: list[tuple[Task, Trace, Verdict]] = []
    for trace in result.traces:
        verdict = verdicts.get(trace.trace_id)
        task = by_id.get(trace.task_id)
        if verdict is None or task is None:
            continue
        triples.append((task, trace, verdict))

    candidates = extract_from_run(triples, now=now)

    trace_index = {t.trace_id: t for t in result.traces}
    kept, reports = filter_batch(
        candidates,
        trace_index=trace_index,
        verdict_index=verdicts,
        forbidden_values=holdout_answer_values(items),
    )

    if kept:
        store.add_many(kept, now=now)

    return LearnResult(
        traces=result.traces,
        verdicts=verdicts,
        candidates=candidates,
        kept=kept,
        reports=reports,
        errors=errors,
        store_stats=store.stats(),
        elapsed_ms=max(0, now_ms() - started),
    )


__all__ = ["LearnResult", "JudgedHook", "learn"]
