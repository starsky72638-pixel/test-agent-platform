"""经验入库前的质量校验。

这是"记忆"和"垃圾堆"之间唯一的一道门。五类闸门，任一不过就不入库：

1. 能力性失败闸门
   链路故障、崩溃、空轨迹、未知工具，都属于环境问题，不是 Agent 能力问题。
   把它们沉淀成经验，下次注入的是一条"教 Agent 怎么应对别人挂了"的废话。

2. 来源闸门
   只接受 fit 侧任务提炼的经验。holdout 侧的任何东西（哪怕是失败原因）
   都不许变成经验文本，否则等于把验证集信息带进了训练侧，对比实验就失效了。

3. 证据闸门
   证据必须指向真实存在的轨迹与轮次；该轮在判定里确实是这个失败类型；
   quote 必须能在轨迹原文里找到。做不到这三点的经验，来源不明，不允许入库。

4. 泄漏闸门
   动作文本里只要出现阿拉伯数字，或者出现 holdout 任务的答案值，一律拒绝。
   经验是给人看的"该怎么做"，不是"这道题答案是 899"。

5. 可执行闸门
   太短、或者是"注意准确""再仔细点"这类空话，无法转成行为，拒绝。
   同时要求触发条件非空，否则永远检索不到，等于没存。

为什么一次性收集所有 reasons 而不是遇到第一条就返回：
读报告的人需要知道这条经验踩了几个坑，只报第一个会让人反复修同一批数据。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from core.task import Task

from memory.schema import Experience

# 这些失败类型说明"根本没进到内容层"，不能沉淀成能力经验。
NON_LEARN_FAILURES = frozenset(
    {"infra_error", "crashed", "empty_trace", "invalid_tool", "no_judged_turn"}
)

# 空话黑名单：出现即视为无法转成行为。
VAGUE_PHRASES = (
    "注意准确",
    "更仔细",
    "认真的",
    "小心一点",
    "尽量",
    "努力",
    "要注意",
    "多加注意",
)

ACTION_MIN_CHARS = 8
ACTION_MAX_CHARS = 300

DIGIT_RE = re.compile(r"\d")
# 从断言里抽"答案值"：长数字、字母前缀的单号。只用来做泄漏比对。
ANSWER_VALUE_RE = re.compile(r"\d{2,}|[A-Z]{2,}\d{4,}")


@dataclass
class QualityReport:
    """一次校验的结论。reasons 非空即不合格。"""

    exp_id: str
    ok: bool = True
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:  # 允许 if report:
        return self.ok

    def reject(self, reason: str) -> None:
        self.ok = False
        self.reasons.append(reason)

    def warn(self, text: str) -> None:
        self.warnings.append(text)

    def summary(self) -> str:
        if self.ok:
            return "通过" + (f"（{'; '.join(self.warnings)}）" if self.warnings else "")
        return "拒绝: " + "; ".join(self.reasons)


def holdout_answer_values(tasks: list[Task]) -> set[str]:
    """收集 holdout 任务的答案值，用于泄漏比对。

    只取断言里出现的数字与单号形态的片段：这些是"这道题的答案"，
    出现在经验文本里就等于记住答案。像"已签收"这种状态词不取，
    因为它是通用词汇，写进经验是合理的。
    """
    values: set[str] = set()
    for task in tasks or []:
        if task.split != "holdout":
            continue
        assertions = task.assertions or {}
        texts: list[str] = [str(s) for s in (assertions.get("must_have") or [])]
        for group in assertions.get("must_have_any") or []:
            texts.extend(str(s) for s in (group or []))
        for cfg in assertions.get("grounded_claims") or []:
            texts.append(str((cfg or {}).get("value") or ""))
        for text in texts:
            values.update(ANSWER_VALUE_RE.findall(text))
    return values


def check(
    exp: Experience,
    *,
    trace_index: dict | None = None,
    verdict_index: dict | None = None,
    forbidden_values: set[str] | None = None,
    require_fit: bool = True,
) -> QualityReport:
    """校验一条经验。参数全为可选，但给得越全，闸门越严。

    trace_index / verdict_index 的键都是 trace_id：
      trace_index: trace_id -> core.trace.Trace
      verdict_index: trace_id -> evaluator.verdict.Verdict
    """
    report = QualityReport(exp_id=exp.exp_id)

    # ---- 闸门 1：能力性失败 ----
    if not exp.failure_type:
        report.reject("缺少失败类型")
    elif exp.failure_type in NON_LEARN_FAILURES:
        report.reject(f"{exp.failure_type} 属于环境问题，不沉淀为能力经验")

    # ---- 闸门 2：来源必须是 fit 侧 ----
    if require_fit and exp.origin_split != "fit":
        report.reject(f"经验来自 {exp.origin_split} 侧，会污染验证集")

    # ---- 闸门 3：证据可回查 ----
    if not exp.evidence:
        report.reject("没有证据")
    for ev in exp.evidence:
        if not ev.trace_id:
            report.reject("证据缺少 trace_id")
            continue
        if trace_index is not None:
            trace = trace_index.get(ev.trace_id)
            if trace is None:
                report.reject(f"证据指向的轨迹不存在: {ev.trace_id}")
                continue
            if ev.task_id and trace.task_id != ev.task_id:
                report.reject(f"证据 task_id 与轨迹不一致: {ev.task_id} != {trace.task_id}")
            if not _quote_found(ev.quote, trace, ev.failure_type):
                report.reject(f"证据原文在轨迹里找不到: {ev.quote[:24]!r}")
        if verdict_index is not None:
            verdict = verdict_index.get(ev.trace_id)
            if verdict is None:
                report.reject(f"证据指向的判定不存在: {ev.trace_id}")
                continue
            rounds = {r.turn: r for r in verdict.rounds}
            rv = rounds.get(int(ev.turn))
            if rv is None:
                report.reject(f"证据轮次 {ev.turn} 不在判定里")
            elif rv.failure_type and rv.failure_type != ev.failure_type:
                report.reject(
                    f"证据轮次失败类型不一致: 证据 {ev.failure_type} / 判定 {rv.failure_type}"
                )

    # ---- 闸门 4：不许泄漏答案 ----
    text = exp.text()
    if DIGIT_RE.search(text):
        report.reject("动作文本含具体数字，等于把某道题的答案写进经验")
    if forbidden_values:
        hit = sorted(v for v in forbidden_values if v and v in text)
        if hit:
            report.reject(f"动作文本含验证集答案值: {','.join(hit[:3])}")

    # ---- 闸门 5：可执行 + 可检索 ----
    if len(text) < ACTION_MIN_CHARS:
        report.reject(f"动作文本过短（{len(text)} < {ACTION_MIN_CHARS}）")
    if len(text) > ACTION_MAX_CHARS:
        report.reject(f"动作文本过长（{len(text)} > {ACTION_MAX_CHARS}）")
    if any(p in text for p in VAGUE_PHRASES):
        report.reject("动作文本是空话，无法转成行为")
    if not exp.trigger.tags and not exp.trigger.tools:
        report.reject("触发条件为空，永远检索不到")

    # 提醒项：不拦，但写进报告，方便人工复核
    if not exp.trigger.tags:
        report.warn("没有关键词标签，只能靠任务形态匹配")
    if len(exp.action) > 160:
        report.warn("动作文本偏长，注入时可能挤占上下文")
    return report


def _quote_found(quote: str, trace, failure_type: str) -> bool:
    """证据原文是否真的出现在轨迹里。

    空回答类失败允许 quote 为空：那时候本来就没有回答可引用。
    """
    if not quote:
        return failure_type in {"empty_answer", "no_execution"}
    haystack = "\n".join(t.answer or "" for t in trace.turns)
    if quote in haystack:
        return True
    for call in trace.all_calls():
        if quote in (call.output or ""):
            return True
    return False


def filter_batch(
    exps: list[Experience],
    *,
    trace_index: dict | None = None,
    verdict_index: dict | None = None,
    forbidden_values: set[str] | None = None,
    require_fit: bool = True,
) -> tuple[list[Experience], list[QualityReport]]:
    """批量过滤。返回 (合格经验, 全部校验报告)。

    报告包含通过的条目，这样"提炼了 20 条、入库 12 条"能对上账。
    """
    kept: list[Experience] = []
    reports: list[QualityReport] = []
    for exp in exps:
        report = check(
            exp,
            trace_index=trace_index,
            verdict_index=verdict_index,
            forbidden_values=forbidden_values,
            require_fit=require_fit,
        )
        reports.append(report)
        if report.ok:
            kept.append(exp)
    return kept, reports
