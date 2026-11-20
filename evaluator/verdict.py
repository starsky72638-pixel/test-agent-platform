"""五层判定与归因。

输入是 Trace + Task，输出是 Verdict。不含任何业务语义、不依赖任何 Agent 框架。

五层各自在拦什么：
L1 工具选择：该调的工具一次没调。
L2 参数：调了，但参数填错（例如换个订单号去查）。
L3 依据溯源：回答里的数字/状态在指定工具的返回里找不到（唯一允许跨轮取数的一层）。
L4 终态事实：请求的内容缺失、或出现了明确禁止的说法。
L5 调用顺序：该先做的没先做（先提交退款、再回头查订单）。

归因必须定优先级：一条轨迹常同时踩多个坑。不定优先级，同一轨迹换个人归到不同类，
报告就不可比。顺序见 FAILURE_PRIORITY。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from core.task import Task
from core.trace import KIND_TOOL_CALL, STATUS_OK, ToolCall, Trace, Turn

from evaluator.claims import Claim, resolve_all
from evaluator.matching import looks_non_answer, looks_off_topic, matches, same

# 归因优先级：越靠前越优先。前四类是"没真正执行"，必须先排除，
# 否则会把链路故障记成模型的错。
FAILURE_PRIORITY = (
    "infra_error",
    "no_execution",
    "empty_answer",
    "invalid_tool",
    "forbidden_action",
    "max_steps",
    "off_topic",
    "ungrounded",
    "hallucination",
    "missing_tool",
    "wrong_order",
    "wrong_arg",
    "non_answer",
    "wrong_fact",
    "no_judged_turn",
)

PRIORITY_INDEX = {name: i for i, name in enumerate(FAILURE_PRIORITY)}

# 这几类说明"根本没进到内容层"，不该再被 off_topic 覆盖。
PRE_CONTENT_FAILURES = frozenset(
    {"infra_error", "empty_answer", "invalid_tool", "forbidden_action"}
)


@dataclass
class RoundVerdict:
    """一轮的判定结果。turn 是 1 起的轮次号，与报告口径一致。"""

    turn: int
    judged: bool = True
    passed: bool | None = False
    failure_type: str = ""
    tool_ok: bool = False
    param_ok: bool = False
    order_ok: bool = False
    ground_ok: bool = False
    answer_ok: bool = False
    tool_msg: str = ""
    param_msg: str = ""
    order_msg: str = ""
    answer: str = ""
    claims: list[Claim] = field(default_factory=list)
    ungrounded: list[str] = field(default_factory=list)
    missing_fact: list[str] = field(default_factory=list)
    missing_any: list[str] = field(default_factory=list)
    forbidden_hit: list[str] = field(default_factory=list)
    forbidden_tools: list[str] = field(default_factory=list)
    invalid_tools: list[str] = field(default_factory=list)
    non_answer: bool = False
    over_calls: list[list] = field(default_factory=list)
    call_count: int = 0
    # 判定指向的具体节点（节点 id，按出现顺序去重）。
    # Workflow 的失败要能落到"哪个节点"，原生 Agent 没有节点，这里为空。
    # 单独成字段而不是拼进 msg：消费方要能直接取用，不必正则抠字符串。
    fail_nodes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "turn": self.turn,
            "judged": self.judged,
            "passed": self.passed,
            "failure_type": self.failure_type,
            "tool_ok": self.tool_ok,
            "tool_msg": self.tool_msg,
            "param_ok": self.param_ok,
            "param_msg": self.param_msg,
            "order_ok": self.order_ok,
            "order_msg": self.order_msg,
            "ground_ok": self.ground_ok,
            "answer_ok": self.answer_ok,
            "answer": self.answer,
            "claims": [c.to_dict() for c in self.claims],
            "ungrounded": list(self.ungrounded),
            "missing_fact": list(self.missing_fact),
            "missing_any": list(self.missing_any),
            "forbidden_hit": list(self.forbidden_hit),
            "forbidden_tools": list(self.forbidden_tools),
            "invalid_tools": list(self.invalid_tools),
            "non_answer": self.non_answer,
            "over_calls": [list(x) for x in self.over_calls],
            "call_count": self.call_count,
            "fail_nodes": list(self.fail_nodes),
        }


@dataclass
class Verdict:
    """一条任务的完整判定。passed 与 failure_type 是汇总口径。"""

    task_id: str
    passed: bool = False
    failure_type: str = ""
    fail_turn: int | None = None
    rounds: list[RoundVerdict] = field(default_factory=list)
    tool_ok: bool = False
    param_ok: bool = False
    order_ok: bool = False
    ground_ok: bool = False
    answer_ok: bool = False
    tool_msg: str = ""
    param_msg: str = ""
    order_msg: str = ""
    ungrounded: list[str] = field(default_factory=list)
    missing_fact: list[str] = field(default_factory=list)
    missing_any: list[str] = field(default_factory=list)
    forbidden_hit: list[str] = field(default_factory=list)
    forbidden_tools: list[str] = field(default_factory=list)
    invalid_tools: list[str] = field(default_factory=list)
    over_calls: list[list] = field(default_factory=list)
    fail_nodes: list[str] = field(default_factory=list)

    def graded_rounds(self) -> list[RoundVerdict]:
        return [r for r in self.rounds if r.judged]

    def failed_rounds(self) -> list[RoundVerdict]:
        return [r for r in self.graded_rounds() if not r.passed]

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "passed": self.passed,
            "failure_type": self.failure_type,
            "fail_turn": self.fail_turn,
            "rounds": [r.to_dict() for r in self.rounds],
            "tool_ok": self.tool_ok,
            "param_ok": self.param_ok,
            "order_ok": self.order_ok,
            "ground_ok": self.ground_ok,
            "answer_ok": self.answer_ok,
            "tool_msg": self.tool_msg,
            "param_msg": self.param_msg,
            "order_msg": self.order_msg,
            "ungrounded": list(self.ungrounded),
            "missing_fact": list(self.missing_fact),
            "missing_any": list(self.missing_any),
            "forbidden_hit": list(self.forbidden_hit),
            "forbidden_tools": list(self.forbidden_tools),
            "invalid_tools": list(self.invalid_tools),
            "over_calls": [list(x) for x in self.over_calls],
            "fail_nodes": list(self.fail_nodes),
        }


def calls_of_turn(trace: Trace, turn: Turn) -> list[ToolCall]:
    """取这一轮的工具调用序列，判分统一从这里取。

    有 event 时以 event 为准：WorkflowAgent 的 Turn 只承载对话语义
    （问题进、答案出），工具调用发生在节点里，只有事件记全了。
    两者都有时**不能相加**——同一批调用会被判两次，
    expect_order 的子序列匹配会被打乱，call_count 也会虚高。

    没有 event 的老轨迹（ReAct Agent、全部历史 jsonl）走原路径，
    行为与改动前逐字一致。
    """
    if not trace.events:
        return list(turn.calls)
    return [
        e.as_toolcall()
        for e in trace.events
        if e.kind == KIND_TOOL_CALL and e.turn == turn.index
    ]


def _nodes_where(
    calls: Sequence[ToolCall], names: set[str] | None = None
) -> list[str]:
    """这些调用发生在哪些节点：按出现顺序去重，原生 Agent 的空 node_id 忽略。"""
    out: list[str] = []
    for c in calls:
        if names is not None and c.name not in names:
            continue
        if c.node_id and c.node_id not in out:
            out.append(c.node_id)
    return out


def judge_round(
    assertions: dict,
    calls: Sequence[ToolCall],
    answer: str,
    history: list[tuple[int, ToolCall]],
    allowed_tools: set[str] | None = None,
    turn: int = 1,
) -> RoundVerdict:
    """判一轮。

    calls 只含本轮调用（工具选择、参数、顺序都只看本轮）；
    history 含截至本轮的调用（只有依据溯源用它）。
    """
    r = RoundVerdict(turn=turn, answer=str(answer or ""), call_count=len(calls))
    text = str(answer or "")

    if not text.strip():
        r.tool_msg = "无回答"
        r.param_msg = "无回答"
        r.failure_type = "empty_answer"
        return r

    names = [c.name for c in calls]

    if allowed_tools is not None:
        unknown = [n for n in names if n not in allowed_tools]
        if unknown:
            r.invalid_tools = unknown
            r.tool_msg = f"未知工具 {','.join(unknown)}"
            r.param_msg = "未校验"
            r.failure_type = "invalid_tool"
            r.fail_nodes = _nodes_where(calls, set(unknown))
            return r

    banned = [t for t in (assertions.get("forbidden_tools") or []) if t in names]
    if banned:
        r.forbidden_tools = banned
        r.tool_msg = f"禁止调用 {','.join(banned)}"
        r.param_msg = "未校验"
        r.failure_type = "forbidden_action"
        r.fail_nodes = _nodes_where(calls, set(banned))
        return r

    # ---------- L1 工具选择 ----------
    expect = assertions.get("expect_tools")
    if expect is None:
        r.tool_ok, r.tool_msg = True, "未校验"
    else:
        missing = [t for t in expect if t not in names]
        r.tool_ok = not missing
        if missing:
            r.tool_msg = f"缺 {','.join(missing)}"
            # 缺的工具在画布里没有节点，fail_nodes 只能留空——
            # 硬塞现有节点会被读成"这些节点判错了"。
            # 但要把本轮实际出过调用的节点写出来，读者才知道该加节点。
            seen = _nodes_where(calls)
            if seen:
                r.tool_msg += f"（本轮调用来自节点 {','.join(seen)}）"
        else:
            r.tool_msg = "命中"

    # ---------- L2 参数 ----------
    expect_args = assertions.get("expect_args") or {}
    if not expect_args:
        r.param_ok, r.param_msg = True, "未校验"
    else:
        bad: list[str] = []
        nodes: list[str] = []
        for k, v in expect_args.items():
            if any(same((c.args or {}).get(k), v) for c in calls):
                continue
            # 谁把参数传错了：找携带这个键的调用，取它所在的节点。
            carrier = next(
                ((c.node_id, (c.args or {}).get(k)) for c in calls
                 if k in (c.args or {})),
                None,
            )
            if carrier and carrier[0]:
                if carrier[0] not in nodes:
                    nodes.append(carrier[0])
                bad.append(f"{k}={v}（节点 {carrier[0]} 实传 {carrier[1]}）")
            elif carrier:
                # 原生 Agent 的调用不带节点，文案与改动前逐字一致。
                bad.append(f"{k}={v}")
            else:
                bad.append(f"{k}={v}（无调用带该参数）")
        r.param_ok = not bad
        r.param_msg = f"错 {','.join(bad)}" if bad else "命中"
        r.fail_nodes = nodes

    # ---------- L5 调用顺序 ----------
    # 子序列匹配：中间插入别的工具没关系，这一串的先后不能乱。
    want = assertions.get("expect_order") or []
    if not want:
        r.order_ok, r.order_msg = True, "未校验"
    else:
        pos = 0
        for name in names:
            if pos < len(want) and name == want[pos]:
                pos += 1
        r.order_ok = pos == len(want)
        if r.order_ok:
            r.order_msg = "命中"
        elif pos == 0:
            # 第一个该调的工具压根没出现，说"顺序错"会误导。
            # 这种情况下 tool_ok 也会挂，报告里应显示"未校验"。
            r.order_msg = f"未校验，{want[0]} 一次都没调"
        else:
            r.order_msg = f"顺序错，{want[pos]} 没在 {want[pos - 1]} 之后"
            # 落到违反顺序的那次调用：前驱工具最后一次出现的那个节点。
            idx = max(i for i, n in enumerate(names) if n == want[pos - 1])
            r.fail_nodes = _nodes_where([calls[idx]])

    # ---------- L3 依据溯源 ----------
    r.claims = resolve_all(assertions.get("grounded_claims"), text, history)
    r.ungrounded = [c.value for c in r.claims if not c.grounded]
    r.ground_ok = not r.ungrounded

    # ---------- L4 终态事实 ----------
    r.missing_fact = [s for s in (assertions.get("must_have") or []) if not matches(s, text)]
    any_miss = [
        g
        for g in (assertions.get("must_have_any") or [])
        if not any(matches(s, text) for s in g)
    ]
    r.missing_any = [(g[0] if g else "") for g in any_miss]
    r.forbidden_hit = [s for s in (assertions.get("must_not_have") or []) if matches(s, text)]
    r.answer_ok = not (r.missing_fact or r.missing_any or r.forbidden_hit)

    # ---------- 调用次数：只上报 ----------
    limits = assertions.get("max_calls") or {}
    r.over_calls = [
        [name, names.count(name), int(limit)]
        for name, limit in limits.items()
        if names.count(name) > int(limit)
    ]

    round_ok = r.tool_ok and r.param_ok and r.order_ok and r.answer_ok and r.ground_ok
    # 只在整体没过时标：正常回答里夹几个英文词不该被打成拒答。
    r.non_answer = bool(not round_ok and looks_non_answer(text))
    r.passed = round_ok

    if round_ok:
        r.failure_type = ""
    elif r.ungrounded:
        r.failure_type = "ungrounded"
    elif r.forbidden_hit:
        r.failure_type = "hallucination"
    elif not r.tool_ok:
        r.failure_type = "missing_tool"
    elif not r.order_ok:
        # 排在参数前：先动手后核实属于流程违规，参数后来补对也改变不了动作已发出。
        r.failure_type = "wrong_order"
    elif not r.param_ok:
        r.failure_type = "wrong_arg"
    elif r.non_answer:
        # 与 wrong_fact 分开：一个是"没答"，一个是"答错"，修法完全不同。
        r.failure_type = "non_answer"
    else:
        r.failure_type = "wrong_fact"

    return r


def evaluate(
    trace: Trace,
    task: Task,
    *,
    allowed_tools: set[str] | None = None,
) -> Verdict:
    """判一条任务。"""
    if trace.status != STATUS_OK:
        return Verdict(
            task_id=trace.task_id,
            passed=False,
            failure_type="infra_error",
            tool_msg="未执行",
            param_msg="未执行",
            order_msg="未执行",
        )

    if not trace.turns:
        return Verdict(
            task_id=trace.task_id,
            passed=False,
            failure_type="no_execution",
            tool_msg="没有执行轮次",
            param_msg="没有执行轮次",
            order_msg="没有执行轮次",
        )

    raw_checks = task.assertions.get("turn_checks") or []
    aligned = task.turn_assertions()

    judged: list[RoundVerdict] = []
    history: list[tuple[int, ToolCall]] = []
    last = len(trace.turns) - 1

    for i, turn in enumerate(trace.turns):
        calls = calls_of_turn(trace, turn)
        # 依据溯源要看到"截至本轮"的全部调用，所以累积放在判分之前。
        history.extend((turn.index, c) for c in calls)

        if raw_checks:
            ck = aligned[i]
        elif i == last:
            # 没有逐轮标准时，顶层断言只检查最后一轮。
            ck = task.assertions
        else:
            ck = None

        if ck is None:
            judged.append(
                RoundVerdict(
                    turn=i + 1,
                    judged=False,
                    passed=None,
                    answer=turn.answer,
                    call_count=len(calls),
                )
            )
            continue

        r = judge_round(
            ck,
            calls,
            turn.answer,
            history,
            allowed_tools=allowed_tools,
            turn=i + 1,
        )

        # 步数耗尽说明这一轮没得出结论，不该按回答内容猜缺事实还是空回答。
        if turn.meta.get("status") == "max_steps":
            r.passed = False
            r.failure_type = "max_steps"
            r.answer_ok = False
        elif (
            not r.answer_ok
            and not r.non_answer
            and r.failure_type not in PRE_CONTENT_FAILURES
            and calls
            and looks_off_topic(turn.answer, [c.output for c in calls])
        ):
            # 查到了数据却一个字段都没用上：按缺事实记会把真因藏起来。
            r.failure_type = "off_topic"

        judged.append(r)

    graded = [r for r in judged if r.judged]
    failed = [r for r in graded if not r.passed]
    first = failed[0] if failed else None

    def agg(attr: str) -> bool:
        return all(getattr(r, attr) for r in graded) if graded else False

    def union(attr: str) -> list:
        return [x for r in graded for x in (getattr(r, attr) or [])]

    return Verdict(
        task_id=trace.task_id,
        passed=bool(graded) and first is None,
        failure_type=(
            first.failure_type if first else ("no_judged_turn" if not graded else "")
        ),
        fail_turn=(first.turn if first else None),
        rounds=judged,
        tool_ok=agg("tool_ok"),
        param_ok=agg("param_ok"),
        order_ok=agg("order_ok"),
        ground_ok=agg("ground_ok"),
        answer_ok=agg("answer_ok"),
        tool_msg="; ".join(
            r.tool_msg for r in graded if r.tool_msg and not r.tool_ok
        ),
        param_msg="; ".join(
            r.param_msg for r in graded if r.param_msg and not r.param_ok
        ),
        order_msg="; ".join(
            r.order_msg for r in graded if r.order_msg and not r.order_ok
        ),
        ungrounded=union("ungrounded"),
        missing_fact=union("missing_fact"),
        missing_any=union("missing_any"),
        forbidden_hit=union("forbidden_hit"),
        forbidden_tools=union("forbidden_tools"),
        invalid_tools=union("invalid_tools"),
        over_calls=[[r.turn, *oc] for r in graded for oc in (r.over_calls or [])],
        # 只汇总失败轮的节点，去重保序：通过轮不该出现在定位结果里。
        fail_nodes=list(
            dict.fromkeys(n for r in failed for n in (r.fail_nodes or []))
        ),
    )


def evaluate_all(
    traces: Sequence[Trace],
    tasks: dict[str, Task],
    *,
    allowed_tools: set[str] | None = None,
) -> list[Verdict]:
    """批量判定。任务缺失直接报错，不做静默跳过——那会让通过率虚高。"""
    out: list[Verdict] = []
    for trace in traces:
        task = tasks.get(trace.task_id)
        if task is None:
            raise KeyError(f"找不到任务 {trace.task_id!r}，无法判定")
        out.append(evaluate(trace, task, allowed_tools=allowed_tools))
    return out
