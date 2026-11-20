"""从"一条失败的执行"里提炼一条可执行的经验。

输入是 (Task, Trace, Verdict)，输出是 Experience 列表，一条失败轮次最多产出一条。

为什么不直接把失败回答总结成一段话：
总结出来的东西无法验证、无法检索、无法失效，也无法判断有没有把答案写进去。
这里的做法是：失败类型 -> 该补的动作，动作里只写"做什么"（工具名、参数名、顺序），
不写"答案是什么"（任何数字都被 quality 拦掉）。

按失败模式聚合，不按题目聚合：
同一类错误（例如同一次回答里漏了同一组工具）在不同任务上重复出现，
应该累加成一条并提高 support，而不是攒出几十条一题一条的碎片——
碎片的检索效果更差，注入时还会互相重复。

覆盖的失败类型（与 evaluator.verdict.FAILURE_PRIORITY 对齐）：
missing_tool / wrong_order / wrong_arg / forbidden_action /
ungrounded / hallucination / wrong_fact / non_answer / off_topic /
empty_answer / no_execution / max_steps
不产经验的：infra_error / crashed / invalid_tool（环境问题，由 quality 再兜一次）
"""

from __future__ import annotations

import re
from typing import Iterable

from core.task import Task
from core.trace import ToolCall, Trace
from evaluator.verdict import RoundVerdict, Verdict, calls_of_turn

from memory.schema import (
    Evidence,
    Experience,
    Trigger,
    make_exp_id,
    shape_of_turn_count,
)

# 中文二元组 + 英文/下划线词。纯数字一律丢弃：订单号、金额不是检索特征，
# 把它们留在标签里，等于让检索按"这道题的具体编号"去匹配，泛化不了。
_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_]{2,}")

# 高频无区分度二元组，留着会让所有任务互相都"相似"。
STOPWORDS = frozenset({"帮我", "一下", "请问", "你好", "谢谢", "我想", "能不", "不能", "这个"})

# 不产经验的失败类型：不是能力问题。
SKIP_FAILURES = frozenset({"infra_error", "crashed", "empty_trace", "invalid_tool", "no_judged_turn"})


def keywords(text: str, *, max_items: int = 24) -> list[str]:
    """从一句用户原话抽关键词。"""
    out: set[str] = set()
    for run in _CJK_RUN.findall(str(text or "")):
        for i in range(len(run) - 1):
            gram = run[i : i + 2]
            if gram not in STOPWORDS:
                out.add(gram)
    for word in _WORD.findall(str(text or "")):
        out.add(word.lower())
    return sorted(out)[:max_items]


def task_tags(task: Task, *, max_items: int = 32) -> list[str]:
    """一条任务的全部关键词（各轮取并集）。"""
    out: set[str] = set()
    for turn in task.user_turns():
        out.update(keywords(turn))
    return sorted(out)[:max_items]


def _turn_of(trace: Trace, turn_no: int):
    for t in trace.turns:
        if t.index == turn_no - 1:
            return t
    index = turn_no - 1
    return trace.turns[index] if 0 <= index < len(trace.turns) else None


def _assertions_of(task: Task, turn_no: int) -> dict:
    """取失败那一轮实际生效的断言，顶层兜底。

    多轮任务的断言常常只写在 turn_checks[i] 里，顶层是 null
    （M08 的 expect_tools 在第 5 轮、H12 的 expect_order 在第 1 轮）。
    只看顶层会让这类失败一条经验都产不出来——提炼端与判定端读的不是同一份断言，
    经验池就凭空少一半，而报告上只显示"提炼 2 条"，看不出丢了什么。

    逐轮优先、顶层兜底：单轮任务没有 turn_checks，行为与改前完全一致。
    逐轮里的显式 null 不覆盖顶层——null 表示"这轮不设此断言"，
    让它抹掉整体断言，等于把判定依据也一起抹掉。
    """
    top = dict(task.assertions or {})
    if not turn_no:
        return top

    checks = task.turn_assertions()
    idx = int(turn_no) - 1
    if 0 <= idx < len(checks):
        for key, value in (checks[idx] or {}).items():
            if value is not None:
                top[key] = value
    return top


def _tools_of(trace: Trace, turn_no: int) -> list[str]:
    turn = _turn_of(trace, turn_no)
    return sorted({c.name for c in (calls_of_turn(trace, turn) if turn else [])})


def _quote_of(trace: Trace, turn_no: int) -> str:
    """证据原文：优先取该轮回答，空回答时退回最后一次工具返回。

    只截前 160 字。这段文本会被人拿去核对，所以必须原样来自轨迹，不做改写。
    """
    turn = _turn_of(trace, turn_no)
    if turn is None:
        return ""
    answer = (turn.answer or "").strip()
    if answer:
        return answer[:160]
    for call in reversed(calls_of_turn(trace, turn)):
        if (call.output or "").strip():
            return call.output.strip()[:160]
    return ""


def _detail_of(rv: RoundVerdict) -> str:
    """判定给出的原始线索，便于人工复核时不用再翻报告。"""
    parts: list[str] = []
    for label, msg in (("工具", rv.tool_msg), ("参数", rv.param_msg), ("顺序", rv.order_msg)):
        msg = (msg or "").strip()
        if msg and msg not in {"命中", "未校验"}:
            parts.append(f"{label}:{msg}")
    for label, items in (
        ("缺事实", rv.missing_fact),
        ("缺任一", rv.missing_any),
        ("越界", rv.forbidden_hit),
        ("无依据", rv.ungrounded),
        ("禁调", rv.forbidden_tools),
    ):
        if items:
            parts.append(f"{label}:{','.join(str(x) for x in items)}")
    return " | ".join(parts)[:200]


def _lesson(rv: RoundVerdict, task: Task, trace: Trace) -> tuple[str, str, str] | None:
    """核心策略表：失败类型 -> (pattern, action, guard)。

    pattern 决定 exp_id，所以它必须只由"失败的结构"组成，不能含题目 id、金额、单号。
    """
    shape = shape_of_turn_count(task.turn_count())
    ft = rv.failure_type
    turn = _turn_of(trace, rv.turn)
    calls = calls_of_turn(trace, turn) if turn else []
    names = [c.name for c in calls]
    # 断言必须取失败那一轮的，不能只看顶层：见 _assertions_of 的说明。
    assertions = _assertions_of(task, rv.turn)

    if ft == "missing_tool":
        expect = [t for t in (assertions.get("expect_tools") or []) if t]
        missing = [t for t in expect if t not in names]
        if not missing:
            return None
        pattern = f"missing_tool|{'/'.join(sorted(missing))}|{shape}"
        action = (
            f"遇到这类问题时必须调用 {'、'.join(missing)}；"
            "只做其中一部分查询就作答，会被判为漏查。"
        )
        guard = "已经查到的部分不能代替没查的那部分。"
        return pattern, action, guard

    if ft == "wrong_order":
        want = [str(x) for x in (assertions.get("expect_order") or [])]
        if len(want) < 2:
            return None
        # 找出第一处倒置的位置，只描述那一对，避免写出整条流水账。
        pos, first_bad = 0, None
        for name in names:
            if pos < len(want) and name == want[pos]:
                pos += 1
            elif name in want and want.index(name) > pos:
                first_bad = (want[pos], name)
                break
        if first_bad is None:
            first_bad = (want[0], want[1])
        pattern = f"wrong_order|{'/'.join(want)}|{shape}"
        action = (
            f"必须先调用 {first_bad[0]} 拿到结果，再调用 {first_bad[1]}；"
            "顺序颠倒等于动作已经发出，事后补查不算数。"
        )
        return pattern, action, "不要在没核实前提之前就把动作发出去。"

    if ft == "wrong_arg":
        expect_args = assertions.get("expect_args") or {}
        bad: list[str] = []
        for key, want in expect_args.items():
            if not any(str(c.args.get(key)) == str(want) for c in calls):
                bad.append(str(key))
        if not bad:
            return None
        pattern = f"wrong_arg|{'/'.join(sorted(bad))}|{shape}"
        # 措辞必须同时覆盖两种情形，否则会写出与多轮槽位相反的经验。
        # 原措辞是"不能沿用上一次的值"——对单轮成立，对多轮正好反了：
        # 用户在后续轮次不再重复条件时，正确做法就是沿用最近一次说过的值。
        # M07 第 3 轮"再帮我搜个耳机"丢了第 2 轮的预算，正是这类。
        action = (
            f"调用工具时 {', '.join(sorted(bad))} 必须带上具体值："
            "本轮用户说了就按本轮的，本轮没提但前面说过的，沿用最近一次的值，"
            "不能退回更早的旧值；换关键词继续搜时，之前的限制仍然有效；"
            "不能留空，也不能凭空猜。"
        )
        return pattern, action, "参数不是可选项，漏了就等于查错范围。"

    if ft == "forbidden_action":
        banned = [str(x) for x in (rv.forbidden_tools or [])]
        if not banned:
            return None
        pattern = f"forbidden_action|{'/'.join(sorted(banned))}|{shape}"
        action = f"不要调用 {'、'.join(banned)}，除非用户明确要求执行这个动作。"
        return pattern, action, "查到相关信息不等于可以替用户下决定。"

    if ft == "ungrounded":
        pattern = f"ungrounded|{shape}"
        action = (
            "回答里的具体数值与编号必须来自工具返回；"
            "工具没给就说没查到，不要凭印象补全。"
        )
        return pattern, action, "上一轮的旧值也不能当成本轮结果。"

    if ft == "hallucination":
        pattern = f"hallucination|{shape}"
        action = (
            "工具没有返回的内容不要自己补，尤其是数量、状态、"
            "能不能执行这类结论，必须以工具返回为准。"
        )
        return pattern, action, ""

    if ft == "wrong_fact":
        pattern = f"wrong_fact|{shape}"
        action = "用户问到的关键信息要都报出来，不能只答一半或漏掉其中的一项。"
        return pattern, action, ""

    if ft == "non_answer":
        pattern = f"non_answer|{shape}"
        action = (
            "用中文直接回答用户的问题，不要拒答、不要输出英文说明、"
            "不要声称自己读不到数据。"
        )
        return pattern, action, ""

    if ft == "off_topic":
        pattern = f"off_topic|{shape}"
        action = "只回答用户问的那件事，不要延伸到无关话题。"
        return pattern, action, ""

    if ft == "empty_answer":
        pattern = f"empty_answer|{shape}"
        action = "必须给出实质性回答，不能返回空内容。"
        return pattern, action, ""

    if ft == "no_execution":
        pattern = f"no_execution|{shape}"
        action = "用户已经给出条件时就直接调工具查询，不要先反问。"
        return pattern, action, ""

    if ft == "max_steps":
        pattern = f"max_steps|{shape}"
        action = "不要反复调用同一个工具，信息够了就收敛到回答。"
        return pattern, action, ""

    return None


def extract_one(
    task: Task,
    trace: Trace,
    verdict: Verdict,
    *,
    now: int | None = None,
) -> list[Experience]:
    """从一条执行里提炼经验。失败的每一轮最多一条。"""
    if not trace or not verdict:
        return []
    if not verdict.failed_rounds():
        return []

    out: list[Experience] = []
    for rv in verdict.failed_rounds():
        if not rv.failure_type or rv.failure_type in SKIP_FAILURES:
            continue
        built = _lesson(rv, task, trace)
        if built is None:
            continue
        pattern, action, guard = built
        out.append(
            Experience(
                exp_id=make_exp_id(pattern),
                failure_type=rv.failure_type,
                pattern=pattern,
                action=action,
                guard=guard,
                trigger=Trigger(
                    tags=task_tags(task),
                    tools=_tools_of(trace, rv.turn),
                    shape=shape_of_turn_count(task.turn_count()),
                ),
                evidence=[
                    Evidence(
                        trace_id=trace.trace_id,
                        task_id=task.id,
                        turn=int(rv.turn),
                        failure_type=rv.failure_type,
                        quote=_quote_of(trace, rv.turn),
                        agent=trace.agent,
                        detail=_detail_of(rv),
                    )
                ],
                origin_split=task.split,
                created_at=now or 0,
                updated_at=now or 0,
            )
        )
    return out


def extract_from_run(
    items: Iterable[tuple[Task, Trace, Verdict]],
    *,
    now: int | None = None,
) -> list[Experience]:
    """从一批执行里提炼经验，同模式的自动合并成一条。

    返回的顺序按首次出现排列，便于报告稳定复现。
    """
    learned: dict[str, Experience] = {}
    for task, trace, verdict in items:
        for exp in extract_one(task, trace, verdict, now=now):
            current = learned.get(exp.exp_id)
            if current is None:
                learned[exp.exp_id] = exp
            else:
                current.merge_from(exp, now=now)
    return list(learned.values())


def find_calls(trace: Trace, name: str) -> list[ToolCall]:
    """辅助：某工具在整条轨迹里的全部调用，供人工核对用。"""
    return [c for c in trace.all_calls() if c.name == name]
