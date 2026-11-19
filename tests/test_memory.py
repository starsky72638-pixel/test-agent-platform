"""经验记忆层的单测。

覆盖五件事：
1. 契约：往返、稳定 id、置信度、过期；
2. 质量闸门：五类脏经验必须被拒，一条干净的被接收；
3. 提炼：每类失败都能产出可执行动作，且动作里不含数字；
4. 库与检索：追加写、版本、退休、相似任务能命中；
5. 闭环：cold agent 在 holdout 上从失败变成通过，且变化确实由经验引起。

闭环那条单测的性质要说清楚：它证明的是"管道通、行为可观测地依赖经验"，
不证明语言理解——测试里的 Agent 只看 ctx 有没有经验，不读经验内容。
语言层面的真实效果由真实模型实验给数，这里不替它下结论。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.protocol import RunContext
from core.runner import run_task
from core.task import Task
from core.trace import (
    STATUS_CRASHED,
    STATUS_OK,
    ToolCall,
    Trace,
    Turn,
    now_ms,
)
from evaluator import evaluate
from memory.extract import extract_from_run, extract_one, keywords, task_tags
from memory.inject import EXPERIENCE_HEADER, make_provider, render, render_for_task
from memory.quality import check, filter_batch, holdout_answer_values
from memory.retrieve import jaccard, retrieve
from memory.schema import (
    STATUS_ACTIVE,
    STATUS_RETIRED,
    Evidence,
    Experience,
    Trigger,
    make_exp_id,
)
from memory.store import ExperienceStore

DAY_MS = 86_400_000


# ---------------------------------------------------------------- 构造工具


def mk_call(name: str, args: dict | None = None, output: str = "") -> ToolCall:
    return ToolCall(name=name, args=dict(args or {}), call_id=f"c-{name}", output=output)


def mk_turn(index: int, calls: list[ToolCall] | None = None, answer: str = "") -> Turn:
    return Turn(index=index, user_input=f"u{index}", calls=list(calls or []), answer=answer)


def mk_task(
    tid: str,
    *,
    assertions: dict | None = None,
    turns: list[str] | None = None,
    split: str = "fit",
    text: str = "查一下状态",
) -> Task:
    return Task(
        id=tid,
        input=text,
        turns=list(turns or []),
        assertions=dict(assertions or {}),
        split=split,
    )


def mk_trace(
    task_id: str,
    turns: list[Turn],
    *,
    agent: str = "fake",
    status: str = STATUS_OK,
    trace_id: str | None = None,
) -> Trace:
    """构造一条轨迹。

    第一个参数是**任务 id**，不是 trace id——因为 quality 会核对
    `Evidence.task_id == trace.task_id`，传错就会在"证据与轨迹不一致"那道闸门被拒。

    trace_id 单独给：Trace 在 trace_id 为空时会自动生成 "<task_id>-<uuid8>"，
    那样就和 trace_index / verdict_index 的键对不上，测试也就绕过了想测的闸门。
    生产路径里这两个 id 由执行方各自生成，这里显式指定只是为了测试可读。
    """
    return Trace(
        task_id=task_id,
        agent=agent,
        turns=turns,
        status=status,
        trace_id=trace_id or f"tr-{task_id}",
    )


def mk_exp(**over) -> Experience:
    base = dict(
        exp_id=make_exp_id("missing_tool|lookup|single"),
        failure_type="missing_tool",
        pattern="missing_tool|lookup|single",
        action="遇到这类问题时必须调用 lookup；只做一部分查询就作答会被判为漏查。",
        trigger=Trigger(tags=["状态", "查一"], tools=["lookup"], shape="single"),
        evidence=[Evidence(trace_id="t-1", task_id="T1", turn=1, failure_type="missing_tool")],
        origin_split="fit",
    )
    base.update(over)
    return Experience(**base)


def run_one(agent, task):
    """跑一条任务并判分，返回 (trace, verdict)。"""
    trace = run_task(agent, task)
    return trace, evaluate(trace, task)


# ---------------------------------------------------------------- 1. 契约


def test_experience_roundtrip():
    exp = mk_exp()
    raw = json.loads(json.dumps(exp.to_dict(), ensure_ascii=False))
    again = Experience.from_dict(raw)
    assert again.to_dict() == exp.to_dict()


def test_exp_id_depends_on_pattern_not_task():
    pattern = "missing_tool|lookup|single"
    assert make_exp_id(pattern) == make_exp_id(pattern)
    assert make_exp_id(pattern) != make_exp_id("missing_tool|lookup|multi")


def test_confidence_uses_injection_ratio():
    exp = mk_exp(support=3, contradict=1, injected=4)
    assert exp.confidence == 0.75
    fresh = mk_exp(support=1, contradict=0, injected=0)
    assert fresh.confidence == 0.5


def test_ttl_expiry():
    old = mk_exp(created_at=now_ms() - 40 * DAY_MS, ttl_days=30)
    assert old.is_expired()
    young = mk_exp(created_at=now_ms() - 1 * DAY_MS, ttl_days=30)
    assert not young.is_expired()


def test_trigger_rejects_unknown_shape():
    with pytest.raises(ValueError):
        Trigger(shape="triple")


# ---------------------------------------------------------------- 2. 质量闸门


def test_reject_infra_failure_type():
    report = check(mk_exp(failure_type="crashed"))
    assert not report
    assert any("环境问题" in r for r in report.reasons)


def test_reject_holdout_origin():
    report = check(mk_exp(origin_split="holdout"))
    assert not report
    assert any("污染验证集" in r for r in report.reasons)


def test_reject_without_evidence():
    report = check(mk_exp(evidence=[]))
    assert not report
    assert any("没有证据" in r for r in report.reasons)


def test_reject_digit_in_action():
    report = check(mk_exp(action="回答里必须包含金额 899 元，否则算漏答。"))
    assert not report
    assert any("具体数字" in r for r in report.reasons)


def test_reject_holdout_answer_value():
    report = check(mk_exp(action="看到已签收就直接说已签收。"), forbidden_values={"已签收"})
    assert not report
    assert any("验证集答案值" in r for r in report.reasons)


def test_reject_vague_action():
    report = check(mk_exp(action="要注意准确一点"))
    assert not report
    assert any("空话" in r for r in report.reasons)


def test_reject_unknown_trace():
    exp = mk_exp()
    report = check(exp, trace_index={})
    assert not report
    assert any("轨迹不存在" in r for r in report.reasons)


def test_reject_quote_not_verbatim():
    trace = mk_trace("T1", [mk_turn(0, [mk_call("lookup")], answer="真实回答")], trace_id="t-1")
    exp = mk_exp(evidence=[Evidence(trace_id="t-1", task_id="T1", turn=1,
                                    failure_type="missing_tool", quote="编造的引用")])
    report = check(exp, trace_index={"t-1": trace})
    assert not report
    assert any("找不到" in r for r in report.reasons)


def test_reject_turn_failure_mismatch():
    task = mk_task("T1", assertions={"expect_tools": ["lookup", "trace"]})
    trace = mk_trace("T1", [mk_turn(0, [mk_call("lookup")], answer="查到了")], trace_id="t-1")
    verdict = evaluate(trace, task)
    exp = mk_exp(evidence=[Evidence(trace_id="t-1", task_id="T1", turn=1,
                                    failure_type="wrong_arg")])
    report = check(exp, verdict_index={"t-1": verdict})
    assert not report
    assert any("失败类型不一致" in r for r in report.reasons)


def test_accept_clean_experience():
    task = mk_task("T1", assertions={"expect_tools": ["lookup", "trace"]})
    trace = mk_trace("T1", [mk_turn(0, [mk_call("lookup")], answer="查到了")], trace_id="t-1")
    verdict = evaluate(trace, task)
    exps = extract_one(task, trace, verdict)
    assert len(exps) == 1
    report = check(
        exps[0],
        trace_index={"t-1": trace},
        verdict_index={"t-1": verdict},
        forbidden_values=set(),
    )
    assert report.ok, report.summary()


# ---------------------------------------------------------------- 3. 提炼


def test_extract_missing_tool_names_the_tool():
    task = mk_task("T1", assertions={"expect_tools": ["lookup", "trace"], "must_have": ["完成"]})
    trace = mk_trace("T1", [mk_turn(0, [mk_call("lookup")], answer="完成了")], trace_id="t-1")
    exps = extract_one(task, trace, evaluate(trace, task))
    assert len(exps) == 1
    assert "trace" in exps[0].action
    assert exps[0].failure_type == "missing_tool"
    assert exps[0].evidence[0].trace_id == "t-1"


def test_extract_reads_turn_level_assertions():
    """断言只写在 turn_checks 里时，也要提炼出经验。

    真实数据里 M08 的 expect_tools 只在第 5 轮、顶层是 null。
    改前只看顶层，missing 为空直接 return None，这条失败被静默吞掉。
    """
    task = mk_task(
        "M08",
        turns=["帮我看看", "第一个", "那 1001 呢", "算了", "订单和物流都查一下"],
        assertions={
            "expect_tools": None,
            "turn_checks": [
                {"must_have": ["好的"]},
                {"must_have": ["好的"]},
                {"must_have": ["好的"]},
                {"must_have": ["好的"]},
                {"expect_tools": ["order", "logistics"], "must_have": ["已发货"]},
            ],
        },
    )
    turns = [mk_turn(i, answer="好的") for i in range(4)]
    turns.append(mk_turn(4, [mk_call("order")], answer="已发货"))
    trace = mk_trace("M08", turns, trace_id="tr-M08")

    verdict = evaluate(trace, task)
    assert verdict.failure_type == "missing_tool"
    assert verdict.fail_turn == 5

    exps = extract_one(task, trace, verdict)
    assert len(exps) == 1
    assert "logistics" in exps[0].action


def test_extract_wrong_order_describes_the_pair():
    task = mk_task("T1", assertions={"expect_tools": ["check", "apply"],
                                     "expect_order": ["check", "apply"]})
    trace = mk_trace(
        "T1", [mk_turn(0, [mk_call("apply"), mk_call("check")], answer="办好了")], trace_id="t-1"
    )
    verdict = evaluate(trace, task)
    assert verdict.failure_type == "wrong_order"
    exps = extract_one(task, trace, verdict)
    assert len(exps) == 1
    assert "check" in exps[0].action and "apply" in exps[0].action


def test_extract_wrong_arg_keeps_slot_from_previous_turns():
    """多轮里参数要沿用前面轮次的值，不能写成"不许沿用上一次"。

    M07 的真实病因：第 2 轮把预算提到 900，第 3 轮只说"再帮我搜个耳机"，
    正确做法是沿用 900 去搜。原措辞"不能沿用上一次的值"会把模型往反方向带——
    这条经验一旦入库并注入，比不提炼更坏。
    """
    checks = [
        {"expect_tools": ["search"], "expect_args": {"max_price": 500}},
        {"expect_tools": ["search"], "expect_args": {"max_price": 900}},
        {"expect_tools": ["search"], "expect_args": {"max_price": 900}},
    ]
    task = mk_task(
        "M07",
        turns=["预算 500 以内有什么推荐", "那预算提到 900 呢", "再帮我搜个耳机"],
        assertions={"turn_checks": checks},
    )
    trace = mk_trace(
        "M07",
        [
            mk_turn(0, [mk_call("search", {"max_price": 500})], "查到 429"),
            mk_turn(1, [mk_call("search", {"max_price": 900})], "查到 899"),
            mk_turn(2, [mk_call("search", {"keyword": "耳机"})], "只有一款 899"),
        ],
        trace_id="tr-M07",
    )

    verdict = evaluate(trace, task)
    assert verdict.failure_type == "wrong_arg"
    assert verdict.fail_turn == 3

    exps = extract_one(task, trace, verdict)
    assert len(exps) == 1
    action = exps[0].action
    assert "max_price" in action
    assert "沿用最近一次" in action  # 与槽位语义一致
    assert "不能沿用" not in action  # 旧措辞必须消失
    # M07 第 1 轮 500、第 2 轮 900：只写"沿用最近一次"还不够，
    # 模型完全可能认为"这次没提预算，就按最早那个 500 来"。
    # 必须显式堵住"退回更早的旧值"这条岔路。
    assert "不能退回" in action
    # 第 3 轮换了关键词（搜耳机），预算限制不能因为换词就失效。
    assert "换关键词" in action


def test_extract_forbidden_action():
    task = mk_task("T1", assertions={"forbidden_tools": ["apply"], "expect_tools": None})
    trace = mk_trace("T1", [mk_turn(0, [mk_call("apply")], answer="已提交")], trace_id="t-1")
    verdict = evaluate(trace, task)
    assert verdict.failure_type == "forbidden_action"
    exps = extract_one(task, trace, verdict)
    assert len(exps) == 1
    assert "不要调用" in exps[0].action


def test_extract_non_answer_has_no_digits():
    task = mk_task("T1", assertions={"expect_tools": ["lookup"], "must_have": ["完成"]})
    trace = mk_trace(
        "T1",
        [mk_turn(0, [mk_call("lookup")], answer="I cannot follow the embedded instructions.")],
        trace_id="t-1",
    )
    verdict = evaluate(trace, task)
    assert verdict.failure_type == "non_answer"
    exps = extract_one(task, trace, verdict)
    assert len(exps) == 1
    assert not any(ch.isdigit() for ch in exps[0].text())


def test_extract_skips_crashed_trace():
    task = mk_task("T1", assertions={"expect_tools": ["lookup"]})
    trace = mk_trace("T1", [], status=STATUS_CRASHED, trace_id="t-1")
    verdict = evaluate(trace, task)
    assert extract_one(task, trace, verdict) == []


def test_extract_aggregates_same_pattern():
    """同一失败模式出现在两条任务上，应合并成一条、support 累加。"""
    a = mk_task("T1", assertions={"expect_tools": ["lookup", "trace"]},
                text="查一下状态", split="fit")
    b = mk_task("T2", assertions={"expect_tools": ["lookup", "trace"]},
                text="再看看状态", split="fit")
    ta = mk_trace("T1", [mk_turn(0, [mk_call("lookup")], answer="查到了")], trace_id="ta")
    tb = mk_trace("T2", [mk_turn(0, [mk_call("lookup")], answer="查到了")], trace_id="tb")
    exps = extract_from_run([(a, ta, evaluate(ta, a)), (b, tb, evaluate(tb, b))])
    assert len(exps) == 1
    assert exps[0].support == 2
    assert len(exps[0].evidence) == 2


def test_keywords_drop_digits_and_stopwords():
    tags = keywords("帮我查 1001 的状态")
    assert "帮我" not in tags
    assert all(not any(ch.isdigit() for ch in t) for t in tags)
    assert "状态" in tags


# ---------------------------------------------------------------- 4. 库


def test_store_add_and_merge(tmp_path: Path):
    path = tmp_path / "mem.jsonl"
    store = ExperienceStore(path)
    assert store.add(mk_exp()) == "new"
    assert store.add(mk_exp()) == "merged"
    exp = store.get(mk_exp().exp_id)
    assert exp.version == 2
    assert exp.support == 2
    assert store.stats()["lines"] == 2


def test_store_is_append_only_and_reload_keeps_latest(tmp_path: Path):
    path = tmp_path / "mem.jsonl"
    store = ExperienceStore(path)
    store.add(mk_exp())
    store.add(mk_exp())
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 2  # 旧版本仍在文件里
    reloaded = ExperienceStore(path)
    assert reloaded.get(mk_exp().exp_id).version == 2


def test_store_retire_is_sticky(tmp_path: Path):
    path = tmp_path / "mem.jsonl"
    store = ExperienceStore(path)
    store.add(mk_exp())
    assert store.retire(mk_exp().exp_id, "manual")
    store.add(mk_exp())  # 又观察到一次同类失败
    exp = store.get(mk_exp().exp_id)
    assert exp.status == STATUS_RETIRED
    assert exp.meta.get("reobserved_while_retired") == 1


def test_store_retires_after_contradiction(tmp_path: Path):
    path = tmp_path / "mem.jsonl"
    store = ExperienceStore(path)
    store.add(mk_exp())
    exp_id = mk_exp().exp_id
    store.record_injection([exp_id])
    store.record_injection([exp_id])
    store.record_outcome([exp_id], passed=False)
    assert store.get(exp_id).status == STATUS_ACTIVE
    store.record_outcome([exp_id], passed=False)
    assert store.get(exp_id).status == STATUS_RETIRED


def test_store_retires_expired(tmp_path: Path):
    path = tmp_path / "mem.jsonl"
    store = ExperienceStore(path)
    store.add(mk_exp(created_at=now_ms() - 40 * DAY_MS, ttl_days=30))
    retired = store.refresh_status()
    assert retired == [mk_exp().exp_id]
    assert store.active() == []


def test_store_snapshot_writes_current_view(tmp_path: Path):
    path = tmp_path / "mem.jsonl"
    store = ExperienceStore(path)
    store.add(mk_exp())
    store.add(mk_exp())
    snap = store.snapshot(tmp_path / "snap.jsonl")
    lines = [ln for ln in snap.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 1


# ---------------------------------------------------------------- 5. 检索与注入


def _two_experiences() -> list[Experience]:
    a = mk_exp(exp_id="exp-a", pattern="p-a", trigger=Trigger(tags=["状态", "查询"], shape="single"))
    b = mk_exp(
        exp_id="exp-b",
        pattern="p-b",
        failure_type="wrong_order",
        action="必须先调用 check 再调用 apply，顺序不能颠倒。",
        trigger=Trigger(tags=["取消", "办理"], shape="single"),
    )
    return [a, b]


def test_retrieve_prefers_related_experience():
    task = mk_task("T1", text="查询一下状态")
    hits = retrieve(task, _two_experiences())
    assert hits
    assert hits[0].exp.exp_id == "exp-a"


def test_retrieve_returns_empty_without_keyword_overlap():
    task = mk_task("T1", text="今天天气真好")
    assert retrieve(task, _two_experiences()) == []


def test_retrieve_respects_top_k():
    task = mk_task("T1", text="查询一下状态")
    hits = retrieve(task, _two_experiences(), top_k=1)
    assert len(hits) == 1


def test_retrieve_rejects_shape_mismatch():
    """形态不符直接不注入，哪怕关键词高度重合。

    回归守卫：一条"多轮漏查订单"的经验曾被注入到单轮退款题（213 字），
    把本来 PASS 的任务打成 wrong_fact。形态当时只是 0.15 的加分项，
    扣掉之后分数仍然过线，所以拦不住跨形态混入。
    """
    task = mk_task("T1", text="查询一下订单状态")  # 单轮
    exp = mk_exp(
        exp_id="exp-multi",
        pattern="missing_tool|lookup|multi",
        trigger=Trigger(tags=["查询", "订单", "状态"], tools=["lookup"], shape="multi"),
    )
    # 先确认关键词本身是高度重合的，否则这条测试会因为关键词门槛而假通过
    assert jaccard(task_tags(task), exp.trigger.tags) > 0.2
    assert retrieve(task, [exp]) == []


def test_retrieve_keeps_same_shape_multi():
    """形态相同则照常检索：门槛不能把该注入的也挡掉。"""
    task = mk_task("T1", turns=["查询一下订单状态", "那物流呢"])
    exp = mk_exp(
        exp_id="exp-multi",
        pattern="missing_tool|lookup|multi",
        trigger=Trigger(tags=["查询", "订单", "状态"], tools=["lookup"], shape="multi"),
    )
    hits = retrieve(task, [exp])
    assert len(hits) == 1
    assert hits[0].shape == 1.0


def test_retire_removes_from_store_active(tmp_path: Path):
    store = ExperienceStore(tmp_path / "mem.jsonl")
    store.add(mk_exp())
    assert len(store.active()) == 1
    store.retire(mk_exp().exp_id, "manual")
    assert store.active() == []


def test_render_empty_returns_blank():
    assert render([]) == ""


def test_render_has_header_and_numbering():
    text = render(_two_experiences())
    assert text.startswith(EXPERIENCE_HEADER)
    assert "1." in text and "2." in text


def test_render_caps_items():
    exps = [
        mk_exp(exp_id=f"exp-{i}", pattern=f"p-{i}", action=f"这是第{i + 1}条必须注意的动作要求。")
        for i in range(5)
    ]
    text = render(exps, max_items=2)
    assert text.count("必须注意") == 2


def test_render_makes_experience_yield_to_user_request():
    """经验与用户本轮的明确请求冲突时，必须让位。

    实测反例：S14 用户要求退款，注入的却是"不要调用 apply_refund"，
    模型没去少调工具，直接把回答说错了（PASS -> wrong_fact）。
    尾部这句让位声明是唯一防线，删掉不会报错，只会让注入重新变成使坏。
    """
    text = render(_two_experiences())
    assert "以用户请求为准" in text
    assert "不要因为经验而拒绝" in text


def test_provider_returns_text_and_ids(tmp_path: Path):
    store = ExperienceStore(tmp_path / "mem.jsonl")
    store.add(mk_exp())
    text, ids = render_for_task(store, mk_task("T1", text="查询一下状态"))
    assert text and ids == [mk_exp().exp_id]


def test_provider_with_empty_store(tmp_path: Path):
    store = ExperienceStore(tmp_path / "mem.jsonl")
    text, ids = render_for_task(store, mk_task("T1"))
    assert text == "" and ids == []


# ---------------------------------------------------------------- 6. 闭环


class ColdAgent:
    """没有经验就凭印象作答，拿到经验才去查。用来验证管道，不验证语言理解。"""

    name = "cold-demo"

    def run(self, task: Task, ctx: RunContext | None = None) -> Trace:
        informed = bool(ctx and ctx.has_experience())
        turn = Turn(index=0, user_input=task.user_turns()[0])
        if informed:
            turn.calls = [mk_call("lookup", {"q": task.id}, output='{"status":"完成"}')]
            turn.answer = "查到了，状态：完成"
        else:
            turn.answer = "应该是完成了"
        return Trace(task_id=task.id, agent=self.name, turns=[turn], split=task.split)


def test_memory_loop_improves_holdout(tmp_path: Path):
    assertions = {"expect_tools": ["lookup"], "must_have": ["完成"]}
    fit = mk_task("T1", assertions=assertions, text="查一下 甲 的状态", split="fit")
    holdout = mk_task("T2", assertions=assertions, text="再看一下 乙 的状态", split="holdout")

    agent = ColdAgent()

    # 1) 基线：没经验，holdout 失败
    trace_base, verdict_base = run_one(agent, holdout)
    assert not verdict_base.passed
    assert verdict_base.failure_type == "missing_tool"

    # 2) 在 fit 侧失败中提炼经验，过闸门后入库
    fit_trace, fit_verdict = run_one(agent, fit)
    assert not fit_verdict.passed
    learned = extract_from_run([(fit, fit_trace, fit_verdict)])
    assert len(learned) == 1
    kept, reports = filter_batch(
        learned,
        trace_index={fit_trace.trace_id: fit_trace},
        verdict_index={fit_trace.trace_id: fit_verdict},
        forbidden_values=holdout_answer_values([fit, holdout]),
    )
    assert len(kept) == 1, [r.summary() for r in reports]

    store = ExperienceStore(tmp_path / "mem.jsonl")
    store.add_many(kept)

    # 3) 注入后复跑同一条 holdout 任务
    provider = make_provider(store)
    text, ids = provider(holdout)
    assert ids, "应当检索到刚入库的经验"
    trace_after = run_task(agent, holdout, experience=text, hits=ids)
    verdict_after = evaluate(trace_after, holdout)
    assert verdict_after.passed, verdict_after.failure_type

    # 4) 效果回填：通过则 support 增加
    before = store.get(ids[0]).support
    store.record_outcome(ids, passed=verdict_after.passed)
    assert store.get(ids[0]).support == before + 1


# ---------------------------------------------------------------- 7. 解耦


BUSINESS_TOKENS = ("shop", "product", "logistics", "refund", "main_agent", "import demos")


def test_memory_has_no_business_dependency():
    """memory/ 里不许出现业务实现或具体 Agent。

    特意不检查 "order"：evaluator 的失败类型叫 wrong_order（顺序），
    和业务上的"订单"是两回事，用字符串检查会误伤。
    """
    root = Path(__file__).resolve().parents[1] / "memory"
    offenders: list[str] = []
    for path in sorted(root.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for token in BUSINESS_TOKENS:
            if token in text:
                offenders.append(f"{path.name}: {token}")
    assert not offenders, offenders


def test_memory_imports_without_demos():
    """import memory 不应牵连 demos 或任何具体 Agent。

    检查改成 'agents' 前缀：main_agent 已经搬进 agents/ 下，
    再按 'main_agent' 匹配就永远命不中了，这条测试会变成永远通过的摆设。
    """
    import subprocess
    import sys

    code = (
        "import sys; import memory; "
        "bad=[m for m in sys.modules if m.startswith(('demos','agents'))]; "
        "print(bad); raise SystemExit(1 if bad else 0)"
    )
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
