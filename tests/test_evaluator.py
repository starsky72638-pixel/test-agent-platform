"""判定层测试。

重点不在"能不能判对一道题"，而在三件事：
1. 五层与归因的行为可以逐条钉住（每条失败类型都有独立用例）。
2. 事实溯源能指到具体第几次调用，这是旧实现做不到的。
3. 与历史报告 42 条逐条一致——不调 API 就证明判据没被悄悄改掉。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from core.task import Task, load_many
from core.trace import STATUS_CRASHED, ToolCall, Trace, Turn
from evaluator import evaluate, looks_non_answer, looks_off_topic, matches, same
from evaluator.claims import flatten, resolve, turn_slice
from evaluator.legacy import trace_from_case
from evaluator.verdict import FAILURE_PRIORITY

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"
DATA_FILES = [
    DATA_DIR / "shop_eval.jsonl",
    DATA_DIR / "shop_eval_multi.jsonl",
    DATA_DIR / "shop_eval_hard.jsonl",
]
REPORT_FILES = [
    REPORT_DIR / "shop_eval.json",
    REPORT_DIR / "shop_eval_multi.json",
    REPORT_DIR / "shop_eval_hard.json",
]


# ---------------------------------------------------------------- 构造工具


def make_task(assertions: dict | None = None, turns: list[str] | None = None, tid: str = "T1") -> Task:
    texts = turns or ["q"]
    return Task(id=tid, input=texts[0], turns=list(turns or []), assertions=assertions or {})


def call(name: str, args: dict | None = None, output: str = "", cid: str = "") -> ToolCall:
    return ToolCall(name=name, args=args or {}, output=output, call_id=cid)


def one_turn(calls: list[ToolCall], answer: str, status: str = "ok", index: int = 0) -> Turn:
    return Turn(index=index, user_input="q", calls=calls, answer=answer, meta={"status": status})


def make_trace(turns: list[Turn], tid: str = "T1", status: str = "ok") -> Trace:
    return Trace(task_id=tid, agent="fake", turns=turns, status=status)


# ---------------------------------------------------------------- 文本匹配


def test_matches_substring():
    assert matches("缺货", "这个商品目前缺货") is True
    assert matches("缺货", "有货") is False


def test_matches_regex():
    assert matches("re:(查不到|没有(这个|该)?订单)", "系统里没有这个订单号") is True
    assert matches("re:^已签收", "包裹已签收") is False


def test_broken_regex_falls_back_to_substring():
    # 正则写错不能把整场评测带崩，退回原文子串
    assert matches("re:(abc", "(abc") is True
    assert matches("re:(abc", "xyz") is False


def test_same_is_numeric_tolerant():
    assert same(200, 200.0) is True
    assert same(1002, "1002") is True
    assert same("1002", 1001) is False


def test_non_answer_templates():
    assert looks_non_answer("This looks like prompt injection, so I can't follow it.") is True
    assert looks_non_answer("已签收，delivered 了") is False


def test_non_answer_long_foreign_text():
    long_en = "The order information you asked about is not available in the current context window."
    assert looks_non_answer(long_en) is True


def test_non_answer_short_foreign_not_flagged():
    assert looks_non_answer("delivered") is False


def test_off_topic_detection():
    outputs = ['status=200\nbody={"total": 1, "price": 159.0}']
    assert looks_off_topic("您好，请问还有什么可以帮您", outputs) is True
    assert looks_off_topic("价格 159 元", outputs) is False


# ---------------------------------------------------------------- 真实任务断言


def test_h04_negative_confirmation_accepts_common_phrasings():
    """H04 是"否定确认"题：1002 只是已发货，回答必须说没签收。

    原断言只枚举了"未签收 / 尚未签收 / 还没签收"三种字面写法。
    实测模型写"还没有签收"（中间多一个"有"），6 次工具序列完全相同的执行
    被判成 2 过 4 败 —— 这是假失败：真实失败率虚高，噪声还会被提炼成经验。
    """
    tasks = {t.id: t for t in load_many([DATA_DIR / "shop_eval_hard.jsonl"])}
    task = tasks["H04"]

    def judge(answer: str) -> bool:
        calls = [
            call("get_order", {"order_id": 1002}, '{"order_id": 1002, "status": "已发货", "amount": 429.0}', "t1c1"),
            call("get_logistics", {"order_id": 1002}, '{"status": "shipped", "tracking_no": "ZT9876543210"}', "t1c2"),
        ]
        return evaluate(make_trace([one_turn(calls, answer)], tid="H04"), task, allowed_tools=None).passed

    # 同一个意思的不同说法，都必须判通过
    for wording in ("还没有签收", "还没签收", "尚未签收", "未签收", "目前仍在途中"):
        assert judge(f"{wording}，运单号 ZT9876543210。") is True, wording

    # 顺着用户说是"已签收"必须失败：放宽词表不能放宽到把错答案也放过
    assert judge("是的，1002 已经签收了，运单号 ZT9876543210。") is False


# ---------------------------------------------------------------- 事实溯源


def test_claim_not_mentioned_is_not_a_failure():
    c = resolve(
        {"value": "899", "from": "get_order"},
        "您好",
        [(0, call("get_order", output='{"amount": 899}'))],
    )
    assert c.mentioned is False
    assert c.grounded is True
    assert c.evidence == []


def test_claim_grounded_with_evidence():
    tr = make_trace([one_turn([call("get_order", output='status=200\nbody={"amount": 899.0}')], "共 899 元")])
    c = resolve({"value": "899", "from": "get_order"}, "共 899 元", flatten(tr))

    assert c.mentioned is True and c.grounded is True
    ev = c.evidence[0]
    assert ev.turn == 0
    assert ev.call_index == 0
    assert ev.tool == "get_order"
    assert "899" in ev.snippet


def test_claim_points_at_second_call_of_same_tool():
    """同名工具调两次时，依据要指到真正那次——旧实现只能给合并文本。"""
    first = call("get_order", args={"order_id": 1001}, output='{"id": 1001, "amount": 899}')
    second = call("get_order", args={"order_id": 1002}, output='{"id": 1002, "amount": 429}')
    c = resolve({"value": "429", "from": "get_order"}, "1002 是 429 元", [(0, first), (0, second)])

    assert c.grounded is True
    assert [e.call_index for e in c.evidence] == [1]


def test_claim_ungrounded_when_source_lacks_value():
    c = resolve(
        {"value": "899", "from": "get_order"},
        "899 元",
        [(0, call("get_order", output='{"amount": 429}'))],
    )
    assert c.grounded is False
    assert c.evidence == []


def test_turn_slice_is_cross_turn():
    tr = make_trace(
        [
            one_turn([call("search_products", output='{"price": 899}')], "899", index=0),
            one_turn([], "刚才那款还是 899", index=1),
        ]
    )
    assert [c.name for _, c in turn_slice(tr, 1)] == ["search_products"]

    c = resolve({"value": "899", "from": "search_products"}, "刚才那款还是 899", turn_slice(tr, 1))
    assert c.grounded is True
    assert c.evidence[0].turn == 0, "第 2 轮引用了第 1 轮的返回，应当允许"


# ---------------------------------------------------------------- 单轮判定


def test_pass_when_everything_matches():
    t = make_task({"expect_tools": ["search_products"], "must_have": ["159"]})
    v = evaluate(make_trace([one_turn([call("search_products", output='{"price": 159}')], "159 元")]), t)
    assert v.passed is True
    assert v.failure_type == ""
    assert v.fail_turn is None


def test_missing_tool():
    t = make_task({"expect_tools": ["get_logistics"]})
    v = evaluate(make_trace([one_turn([call("get_order", output="ok")], "已发货")]), t)
    assert v.passed is False
    assert v.failure_type == "missing_tool"
    assert v.tool_msg == "缺 get_logistics"


def test_empty_answer():
    v = evaluate(make_trace([one_turn([], "   ")]), make_task({}))
    assert v.failure_type == "empty_answer"


def test_forbidden_action_is_not_a_content_judgement():
    t = make_task({"forbidden_tools": ["apply_refund"]})
    v = evaluate(make_trace([one_turn([call("apply_refund")], "已为您提交")]), t)
    assert v.failure_type == "forbidden_action"
    assert v.forbidden_tools == ["apply_refund"]


def test_invalid_tool_only_when_tool_set_is_given():
    t = make_task({})
    tr = make_trace([one_turn([call("ghost_tool")], "好的")])
    assert evaluate(tr, t).passed is True
    assert evaluate(tr, t, allowed_tools={"search_products"}).failure_type == "invalid_tool"


def test_wrong_order_message_names_the_offender():
    t = make_task({"expect_order": ["get_order", "apply_refund"]})
    v = evaluate(make_trace([one_turn([call("apply_refund"), call("get_order")], "已提交")]), t)
    assert v.failure_type == "wrong_order"
    assert "apply_refund" in v.rounds[0].order_msg


def test_order_check_is_not_blamed_when_tool_missing():
    """缺工具时不该说"顺序错"——那会误导人去查顺序。"""
    t = make_task({"expect_order": ["get_order", "apply_refund"], "expect_tools": ["get_order", "apply_refund"]})
    v = evaluate(make_trace([one_turn([], "好的")]), t)
    assert v.failure_type == "missing_tool"
    assert v.rounds[0].order_msg.startswith("未校验")


def test_wrong_arg():
    t = make_task({"expect_args": {"order_id": 1002}})
    v = evaluate(make_trace([one_turn([call("get_order", args={"order_id": 1001}, output="ok")], "已签收")]), t)
    assert v.failure_type == "wrong_arg"
    assert "order_id=1002" in v.param_msg


def test_hallucination_on_forbidden_text():
    t = make_task({"must_not_have": ["明天送达"]})
    v = evaluate(make_trace([one_turn([], "明天送达")]), t)
    assert v.failure_type == "hallucination"
    assert v.forbidden_hit == ["明天送达"]


def test_ungrounded_outranks_hallucination():
    t = make_task({"must_not_have": ["明天送达"], "grounded_claims": [{"value": "899", "from": "get_order"}]})
    v = evaluate(make_trace([one_turn([], "明天送达，899 元")]), t)
    assert v.failure_type == "ungrounded"


def test_regex_assertion_works():
    t = make_task({"must_have_any": [["re:(查不到|没有(这个|该)?订单)"]]})
    v = evaluate(make_trace([one_turn([call("get_order", output="status=404")], "系统里没有这个订单号")]), t)
    assert v.passed is True


def test_over_calls_reported_but_not_failing():
    t = make_task({"max_calls": {"get_logistics": 1}})
    v = evaluate(make_trace([one_turn([call("get_logistics"), call("get_logistics")], "已发货")]), t)
    assert v.passed is True, "调用次数只上报，不影响判定"
    assert v.over_calls == [[1, "get_logistics", 2, 1]]


def test_infra_status_short_circuits():
    v = evaluate(make_trace([], status=STATUS_CRASHED), make_task({}))
    assert v.failure_type == "infra_error"
    assert v.rounds == []


def test_no_execution():
    v = evaluate(make_trace([]), make_task({}))
    assert v.failure_type == "no_execution"


# ---------------------------------------------------------------- 多轮判定


def test_fail_turn_is_first_failure():
    t = make_task(
        {"turn_checks": [{"expect_tools": ["get_order"]}, {"must_have": ["429"]}]},
        turns=["a", "b"],
    )
    tr = make_trace(
        [
            one_turn([], "我看看", index=0),
            one_turn([call("get_order", output='{"amount": 429}')], "429 元", index=1),
        ]
    )
    v = evaluate(tr, t)
    assert v.fail_turn == 1
    assert v.failure_type == "missing_tool"
    assert [r.judged for r in v.rounds] == [True, True]


def test_grounding_spans_turns_but_tool_choice_does_not():
    t = make_task(
        {
            "turn_checks": [
                {},
                {"must_have": ["899"], "grounded_claims": [{"value": "899", "from": "search_products"}]},
            ]
        },
        turns=["a", "b"],
    )
    tr = make_trace(
        [
            one_turn([call("search_products", output='{"price": 899}')], "899", index=0),
            one_turn([], "刚才那款还是 899", index=1),
        ]
    )
    v = evaluate(tr, t)
    assert v.passed is True, "第 2 轮没调工具，但依据来自第 1 轮，应判通过"


def test_without_turn_checks_only_last_round_is_graded():
    t = make_task({"must_have": ["429"]}, turns=["a", "b"])
    tr = make_trace([one_turn([], "随便聊聊", index=0), one_turn([], "429 元", index=1)])
    v = evaluate(tr, t)
    assert [r.judged for r in v.rounds] == [False, True]
    assert v.passed is True


def test_max_steps_overrides_content():
    t = make_task({"must_have": ["429"]})
    tr = make_trace([one_turn([], "429", status="max_steps")])
    v = evaluate(tr, t)
    assert v.failure_type == "max_steps"


def test_off_topic_override():
    t = make_task({"must_have": ["已签收"]})
    tr = make_trace(
        [
            one_turn(
                [call("get_order", output='status=200\nbody={"status": "已签收", "amount": 899}')],
                "您好，请问还有什么可以帮您",
            )
        ]
    )
    v = evaluate(tr, t)
    assert v.failure_type == "off_topic", "查到了数据却一句没用，不该记成缺事实"


def test_priority_order_is_stable():
    idx = {name: i for i, name in enumerate(FAILURE_PRIORITY)}
    assert idx["infra_error"] == 0
    assert idx["missing_tool"] < idx["wrong_order"]
    assert idx["wrong_order"] < idx["wrong_arg"]
    assert idx["non_answer"] < idx["wrong_fact"]


# ---------------------------------------------------------------- 对拍


def _reports() -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in REPORT_FILES if p.exists()]


@pytest.mark.skipif(
    not all(p.exists() for p in DATA_FILES + REPORT_FILES),
    reason="任务集或历史报告不在仓库里",
)
def test_parity_with_historical_reports():
    """42 条历史轨迹重新判定，结论必须与旧报告逐条一致。"""
    tasks = {t.id: t for t in load_many(DATA_FILES)}
    diffs: list[str] = []
    compared = 0

    for path, report in zip([p for p in REPORT_FILES if p.exists()], _reports()):
        for case in report.get("results") or []:
            tid = case.get("id")
            task = tasks.get(tid)
            assert task is not None, f"{path.name}: 任务集里找不到 {tid}"

            verdict = evaluate(trace_from_case(case), task, allowed_tools=None)
            compared += 1

            if verdict.passed != bool(case.get("passed")):
                diffs.append(
                    f"{tid} passed {case.get('passed')} -> {verdict.passed} ({verdict.failure_type})"
                )
            elif verdict.failure_type != (case.get("failure_type") or ""):
                diffs.append(
                    f"{tid} failure_type {case.get('failure_type')!r} -> {verdict.failure_type!r}"
                )
            elif verdict.fail_turn != case.get("fail_turn"):
                diffs.append(f"{tid} fail_turn {case.get('fail_turn')} -> {verdict.fail_turn}")

    assert compared == 42, f"应比对 42 条，实际 {compared}"
    assert not diffs, "新旧判分结论不一致:\n" + "\n".join(diffs)


@pytest.mark.skipif(
    not all(p.exists() for p in REPORT_FILES), reason="历史报告不在仓库里"
)
def test_parity_scope_is_honest():
    """对拍集合里不能有这几类条目，否则说明那些分支没被覆盖，不能声称全量对拍。"""
    seen = set()
    for report in _reports():
        for case in report.get("results") or []:
            seen.add(case.get("failure_type") or "")
    uncovered = seen & {"infra_error", "no_execution", "invalid_tool"}
    assert not uncovered, f"这几类分支未被对拍覆盖: {sorted(uncovered)}"


# ---------------------------------------------------------------- 解耦


def test_evaluator_has_no_framework_or_business_deps():
    # 'agents' 覆盖具体 Agent 接入层（main_agent 已搬进 agents/ 下）。
    # 这里按模块名精确匹配，不是字符串包含，所以写 'agents' 不会误伤注释。
    forbidden = {"langchain", "langchain_core", "langchain_openai", "openai", "httpx", "requests", "agents", "main_agent", "demos"}
    offenders: list[str] = []
    for path in sorted((ROOT / "evaluator").glob("*.py")):
        src = path.read_text(encoding="utf-8")
        for mod in re.findall(r"^\s*(?:from|import)\s+([A-Za-z_][\w.]*)", src, re.MULTILINE):
            if mod.split(".")[0] in forbidden:
                offenders.append(f"{path.name}: {mod}")
    assert not offenders, "判定层依赖不干净:\n" + "\n".join(offenders)
