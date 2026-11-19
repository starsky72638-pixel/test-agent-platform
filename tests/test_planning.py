"""计划层的离线测试。不连模型、不花额度。

守四件事，都是"兜底真的会动手"的证据，而不是提示词里写了什么：

1. 模型给的 JSON 能从各种包装里解析出来，认不出的工具名要被丢掉；
2. 一个调用只能抵一条待办——两个订单只查了一个，必须仍判为漏；
3. 模型想收尾但待办没做完时，循环会顶它一次，且只顶一次；
4. 开关关掉时行为与改动前一致（不多调一次模型）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from agents.shop.core import planning
from agents.shop.core.agent import run_loop
from core.task import load_many

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TASK_FILES = [
    DATA_DIR / "shop_eval.jsonl",
    DATA_DIR / "shop_eval_multi.jsonl",
    DATA_DIR / "shop_eval_hard.jsonl",
]


class FakeTool:
    def __init__(self, name: str, output: str = "ok") -> None:
        self.name = name
        self.output = output

    def invoke(self, args):
        return self.output


class ScriptedLLM:
    """按脚本逐次返回，并记下每次收到的消息。"""

    def __init__(self, steps) -> None:
        self.steps = list(steps)
        self.calls = 0
        self.seen: list[list] = []

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.seen.append(list(messages))
        step = self.steps[self.calls] if self.calls < len(self.steps) else "结束"
        self.calls += 1
        if isinstance(step, AIMessage):
            return step
        return AIMessage(content=str(step))


def _tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def _plan_json(*steps) -> str:
    return json.dumps({"steps": list(steps)}, ensure_ascii=False)


# =========================
# 一、解析
# =========================


def test_parse_plan_handles_fence_and_rejects_unknown_tool():
    text = "好的，先拆一下：\n```json\n" + _plan_json(
        {"goal": "查订单 1001", "tool": "get_order", "args": {"order_id": 1001}},
        {"goal": "查天气", "tool": "get_weather", "args": {}},
    ) + "\n```"
    steps = planning.parse_plan(text, {"get_order"})

    assert len(steps) == 1
    assert steps[0]["tool"] == "get_order"
    assert steps[0]["args"]["order_id"] == 1001
    # 认不出的工具必须丢掉：留着会让这条待办永远算不成交付，兜底就永远触发。
    assert all(s["tool"] != "get_weather" for s in steps)


def test_parse_plan_returns_empty_on_garbage():
    assert planning.parse_plan("我不太确定该怎么拆") == []
    assert planning.parse_plan("") == []


def test_render_plan_is_empty_for_no_steps():
    assert planning.render_plan([]) == ""


# =========================
# 二、覆盖判定
# =========================


def test_one_call_covers_only_one_step():
    """两个订单只查了一个，必须仍判为漏——这是 M08 那种场景的命门。"""
    steps = [
        {"goal": "查 1001", "tool": "get_order", "args": {"order_id": 1001}},
        {"goal": "查 1002", "tool": "get_order", "args": {"order_id": 1002}},
    ]
    calls = [{"name": "get_order", "args": {"order_id": 1001}}]
    missed = planning.uncovered(steps, calls)

    assert len(missed) == 1
    assert missed[0]["args"]["order_id"] == 1002


def test_wrong_arg_does_not_cover():
    """参数不一致不算覆盖，否则"查了另一个对象"会被当成做完了。"""
    steps = [{"goal": "查 1002", "tool": "get_order", "args": {"order_id": 1002}}]
    calls = [{"name": "get_order", "args": {"order_id": 1001}}]

    assert planning.uncovered(steps, calls) == steps


def test_number_and_string_args_are_equal():
    """模型把 order_id 写成字符串是常态，不能因此判成没做。"""
    steps = [{"goal": "查 1001", "tool": "get_order", "args": {"order_id": "1001"}}]
    calls = [{"name": "get_order", "args": {"order_id": 1001}}]

    assert planning.uncovered(steps, calls) == []


# =========================
# 三、循环里的顶回
# =========================


def test_loop_nudges_once_when_plan_not_done():
    llm = ScriptedLLM(
        [
            _plan_json(
                {"goal": "查订单 1001", "tool": "get_order", "args": {"order_id": 1001}},
                {"goal": "查订单 1002", "tool": "get_order", "args": {"order_id": 1002}},
            ),
            _tool_call("get_order", {"order_id": 1001}, "c1"),
            "1001 花了 899 元",  # 想收尾，但第二项没做
            "1001 花了 899 元，1002 需要再查",
        ]
    )
    result = run_loop(
        llm,
        [FakeTool("get_order", "899")],
        "系统提示",
        "那两个订单各花了多少",
        max_steps=4,
        use_plan=True,
    )

    assert result["nudges"] == 1
    assert [s["args"]["order_id"] for s in result["uncovered"]] == [1002]
    # 顶回的话必须真的进了模型上下文，否则"顶回去"只是代码里的一个自增。
    assert any("待办里还有没做的" in str(m.content) for m in llm.seen[-1])


def test_loop_does_not_nudge_when_plan_done():
    llm = ScriptedLLM(
        [
            _plan_json({"goal": "查 1001", "tool": "get_order", "args": {"order_id": 1001}}),
            _tool_call("get_order", {"order_id": 1001}, "c1"),
            "899 元",
        ]
    )
    result = run_loop(
        llm,
        [FakeTool("get_order", "899")],
        "系统提示",
        "查一下 1001 的订单和物流",
        max_steps=4,
        use_plan=True,
    )

    assert result["nudges"] == 0
    assert result["uncovered"] == []
    assert result["answer"] == "899 元"


def test_plan_only_for_multi_intent():
    """计划层只该管"多件事别漏"，不该插手单意图题目。

    S08 原文一个并列词都没有，但它隐含要查两次（订单状态 + 物流状态），
    原来靠提示词规则 8 兜住。一旦对它开计划，模型只把"查订单"写进清单，
    答完就收尾，反而把规则 8 覆盖掉——实测就是这样从 PASS 掉成 missing_tool 的。
    """
    assert not planning.needed("1002 现在什么状态")
    assert not planning.needed("帮我给订单 1002 申请退款，耳机有杂音")
    assert planning.needed("那两个订单加起来一共多少钱")
    assert planning.needed("查一下 1002 的订单和物流")


@pytest.mark.skipif(
    not all(p.exists() for p in TASK_FILES), reason="真实任务集不在仓库里"
)
def test_plan_fires_only_on_these_turns_in_corpus():
    """把"计划会在哪些轮次触发"钉死在三份真实题干上。

    为什么要这道守卫：上一版标记表里有「还有」，S04「智能手表 S2 还有货吗」
    被误判成多意图——"还有货"是动词短语，不是并列。
    有了这条，以后任何人往标记表里加词，都会先看到它把哪些题拉进计划，
    而不是等跑完一整轮实验才发现有三道 PASS 掉成了 FAIL。

    期望集合是当前语料的全部触发点：
    H02「机械键盘和充电宝一起买，一共多少钱」——真多意图，两个商品要分别搜再聚合；
    S17「订单和物流」、S20「耳机和机械键盘」——两件事分别查；
    M08 第 5 轮「那两个订单加起来一共多少钱」、M09 第 6 轮「键盘多少钱、1001 那单花了多少」。

    H02 一开始被我漏在期望集合外，是测试写错了不是代码错了：它命中「和」+「一共」，
    且任务要求跨两个商品聚合，本来就该拆成两条 search_products。
    """
    tasks = load_many(TASK_FILES)
    fired = {
        (task.id, i)
        for task in tasks
        for i, text in enumerate(task.user_turns(), 1)
        if planning.needed(text)
    }

    assert fired == {("H02", 1), ("S17", 1), ("S20", 1), ("M08", 5), ("M09", 6)}


def test_plan_failure_is_reported_not_silently_empty():
    """计划拆不出来必须留下原因。

    为什么这条必须有：meta 里没有 plan 字段有两种完全相反的含义——
    "这道题不需要拆"（设计）和"门开了但模型没给 JSON"（缺陷）。
    实测 M08 第 5 轮同一题干，一次有计划一次没有，没有这个字段就只能猜。
    """
    llm = ScriptedLLM(["这道题我拆不了，建议直接回答"])
    report: dict = {}

    steps = planning.make_plan(
        llm, "系统提示", "那两个订单加起来一共多少钱", [FakeTool("get_order")], report=report
    )

    assert steps == []
    assert report["called"] is True
    assert report["error"].startswith("no_steps:")


def test_plan_invoke_failure_becomes_error_not_crash():
    """计划调用本身抛异常时，整轮任务不能跟着崩。"""

    class BoomLLM:
        def invoke(self, messages):
            raise TimeoutError("upstream timeout")

    report: dict = {}
    steps = planning.make_plan(BoomLLM(), "系统提示", "两个都要", [FakeTool("get_order")], report=report)

    assert steps == []
    assert report["error"] == "invoke:TimeoutError"


def test_loop_surfaces_plan_error_in_result():
    """计划失败要能落到 run_loop 的返回值里，probe 才打得出来。"""
    llm = ScriptedLLM(["拆不出来", "899 元"])
    result = run_loop(
        llm,
        [FakeTool("get_order", "899")],
        "系统提示",
        "那两个订单加起来一共多少钱",
        max_steps=3,
        use_plan=True,
    )

    assert result["plan"] == []
    assert result["plan_report"]["called"] is True
    assert result["plan_report"]["error"].startswith("no_steps:")


def test_plan_retries_once_after_garbage():
    """第一次回了空 steps，重发一次能纠回来的，不该判为无计划。

    实测 M08 有 3/6 次是 {"steps": []} 或一段分析文字，
    而通过与否和拆没拆出来完全吻合，所以这一次重发不是锦上添花。
    """
    llm = ScriptedLLM(
        [
            "这道题不用拆吧",
            _plan_json(
                {"goal": "查 1001", "tool": "get_order", "args": {"order_id": 1001}},
                {"goal": "查 1002", "tool": "get_order", "args": {"order_id": 1002}},
            ),
        ]
    )
    report: dict = {}

    steps = planning.make_plan(
        llm, "系统提示", "那两个订单加起来一共多少钱", [FakeTool("get_order")], report=report
    )

    assert len(steps) == 2
    assert report["attempts"] == 2
    assert "error" not in report  # 重试成功就不该报错


def test_plan_retry_is_bounded_at_one():
    """两次都拆不出来就收手，不能无限重发。"""
    llm = ScriptedLLM(["第一次不行", "第二次还是不行", "第三次不该发生"])
    report: dict = {}

    steps = planning.make_plan(
        llm, "系统提示", "两个都要", [FakeTool("get_order")], report=report
    )

    assert steps == []
    assert report["attempts"] == 2  # 1 次原始 + 1 次重试，仅此而已
    assert llm.calls == 2
    assert report["error"].startswith("no_steps:")


def test_plan_accepts_params_alias_without_retry():
    """把 args 写成 params 是模型常见偏差，就地认掉，不花额外调用。"""
    text = json.dumps(
        {"steps": [{"goal": "查 1001", "name": "get_order", "params": {"order_id": 1001}}]},
        ensure_ascii=False,
    )
    steps = planning.parse_plan(text, {"get_order"})

    assert steps == [{"goal": "查 1001", "tool": "get_order", "args": {"order_id": 1001}}]


def test_plan_drops_steps_without_tool_name():
    """没有工具名的待办必须丢掉：留着要么假覆盖，要么永久漏项。"""
    text = json.dumps({"steps": [{"goal": "查一下库存", "params": {"q": "耳机"}}]}, ensure_ascii=False)

    assert planning.parse_plan(text, {"search_products"}) == []


def test_single_intent_runs_without_plan_call():
    llm = ScriptedLLM(["899 元"])
    result = run_loop(
        llm, [FakeTool("get_order")], "系统提示", "1002 现在什么状态", max_steps=3, use_plan=True
    )

    assert llm.calls == 1  # 没有多出来那次计划调用
    assert result["plan"] == []


def test_plan_off_costs_no_extra_call():
    """开关关掉时不能多调一次模型——否则对比实验的差异就不只一个变量。"""
    llm = ScriptedLLM(["899 元"])
    result = run_loop(llm, [], "系统提示", "1001 花了多少", max_steps=3, use_plan=False)

    assert llm.calls == 1
    assert result["plan"] == []
    assert result["answer"] == "899 元"
