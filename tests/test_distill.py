"""数据回流（distill）的离线测试。不读真实实验目录、不连模型。

守的是三道闸门真的会拦：
    失败轨迹不进 SFT、工具报错的轨迹不进 SFT、重复轨迹只留一条。
这三条如果只写在注释里，训练集就会悄悄变成"含错误示范"的集合，
而且看不出来——所以每条都钉一个用例。
"""

from __future__ import annotations

from core.task import Task
from core.trace import ToolCall, Trace, Turn

from scripts.distill import distill_traces


def make_task(task_id: str = "T1") -> Task:
    return Task(
        id=task_id,
        input="1001 那单花了多少钱",
        split="fit",
        assertions={"expect_tools": ["get_order"], "must_have": ["899"]},
    )


def make_trace(*, answer: str = "订单 1001 一共 899 元", ok: bool = True) -> Trace:
    turn = Turn(
        index=0,
        user_input="1001 那单花了多少钱",
        calls=[
            ToolCall(
                name="get_order",
                args={"order_id": 1001},
                call_id="t0c0",
                output='{"order_id": 1001, "amount": 899}',
                ok=ok,
            )
        ],
        answer=answer,
        steps=2,
        elapsed_ms=100,
        meta={"status": "ok"},
    )
    return Trace(task_id="T1", agent="fake", turns=[turn], split="fit")


def test_pass_trace_becomes_sft_sample():
    sft, failed, extra = distill_traces([make_trace()], {"T1": make_task()})

    assert len(sft) == 1 and not failed
    roles = [m["role"] for m in sft[0]["messages"]]
    # 一问 → 一次带 tool_calls 的 assistant → 一条 tool 返回 → 最终回答
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert sft[0]["messages"][1]["tool_calls"][0]["function"]["name"] == "get_order"
    assert "899" in sft[0]["messages"][2]["content"]
    assert extra["dropped"] == {}


def test_failed_trace_goes_to_failed_set_with_reason():
    trace = make_trace(answer="一共 899 元")  # 没调工具，答得对也不算过
    trace.turns[0].calls = []

    sft, failed, _ = distill_traces([trace], {"T1": make_task()})

    assert not sft
    assert len(failed) == 1
    assert failed[0]["failure_type"] == "missing_tool"
    # 失败样本必须留档原因与轮次，否则后训练里没法做对比/做 reward。
    # fail_turn 是 1 起的轮次编号（与判定输出里的 @t1 同一套），不是 list 下标。
    assert failed[0]["fail_turn"] == 1


def test_tool_error_trace_is_dropped_not_trained():
    sft, failed, extra = distill_traces([make_trace()], {"T1": make_task()})
    assert len(sft) == 1

    sft2, failed2, extra2 = distill_traces(
        [make_trace(ok=False)], {"T1": make_task()}
    )
    assert not sft2
    # 工具报错但文本对得上，会先被判为通过，然后被 tool_error 闸门拦下。
    assert extra2["dropped"].get("tool_error") == 1 or failed2


def test_duplicates_are_collapsed():
    sft, _, extra = distill_traces(
        [make_trace(), make_trace()], {"T1": make_task()}
    )

    assert len(sft) == 1
    assert extra["duplicates"] == 1


def test_unknown_task_is_dropped():
    sft, failed, extra = distill_traces([make_trace()], {})

    assert not sft and not failed
    assert extra["dropped"].get("no_task") == 1
