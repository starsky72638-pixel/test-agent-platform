"""数据回流（distill）的离线测试。不读真实实验目录、不连模型。

守的是三道闸门真的会拦：
    失败轨迹不进 SFT、工具报错的轨迹不进 SFT、重复轨迹只留一条。
这三条如果只写在注释里，训练集就会悄悄变成"含错误示范"的集合，
而且看不出来——所以每条都钉一个用例。
"""

from __future__ import annotations

import json

from core.task import Task
from core.trace import (
    KIND_NODE_END,
    KIND_NODE_START,
    KIND_TOOL_CALL,
    ToolCall,
    Trace,
    Turn,
    save_traces,
)

from scripts.distill import distill_traces, main as distill_main
from scripts.p5c_probe import _check_sample


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


def make_event_trace(*, ok: bool = True) -> Trace:
    """工具调用只记在 Event 里、Turn.calls 为空的轨迹。

    这就是 workflow 跑出来的形态：Turn 只承载问答，节点细节全走事件。
    """
    trace = make_trace()
    trace.turns[0].calls = []
    trace.add_event(KIND_NODE_START, node_id="start", node_type="start")
    trace.add_event(
        KIND_TOOL_CALL,
        node_id="t1",
        node_type="tool",
        name="get_order",
        args={"order_id": 1001},
        output='{"order_id": 1001, "amount": 899}',
        ok=ok,
    )
    trace.add_event(KIND_NODE_END, node_id="end", node_type="end")
    return trace


def test_pass_trace_becomes_sft_sample():
    sft, failed, extra = distill_traces([make_trace()], {"T1": make_task()})

    assert len(sft) == 1 and not failed
    roles = [m["role"] for m in sft[0]["messages"]]
    # 一问 → 一次带 tool_calls 的 assistant → 一条 tool 返回 → 最终回答
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert sft[0]["messages"][1]["tool_calls"][0]["function"]["name"] == "get_order"
    assert "899" in sft[0]["messages"][2]["content"]
    assert extra["dropped"] == {}


def test_event_only_trace_still_yields_tool_calls():
    """Turn.calls 为空、调用只在 Event 里时，样本仍必须带 tool_calls。

    修复前 _to_messages 只遍历 turn.calls，产出的是
    "user → assistant 最终回答"，整条工具调用示范被静默丢掉——
    样本看着合法，实际没有可学的内容。
    """
    sft, failed, extra = distill_traces([make_event_trace()], {"T1": make_task()})

    assert not failed, "这条轨迹判分应通过"
    assert len(sft) == 1
    assert sft[0]["tools"] == ["get_order"]
    messages = sft[0]["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "tool", "assistant"]
    assert messages[1]["tool_calls"][0]["function"]["name"] == "get_order"
    assert "899" in messages[2]["content"]
    assert extra["dropped"] == {}


def test_event_tool_error_is_dropped():
    """事件里失败过的调用也要触发 tool_error 闸门，否则错误示范会进 SFT。"""
    sft, failed, extra = distill_traces([make_event_trace(ok=False)], {"T1": make_task()})

    assert not sft
    assert failed == []
    assert extra["dropped"] == {"tool_error": 1}


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


# ---------------------------------------------------------------- 直接喂轨迹文件
#
# Workflow 的轨迹落在 runs/*.jsonl，不在 runs/<stamp>/{fit,holdout-*}.jsonl 的
# 目录约定里。--trace 是给它们开的口子，不加就只能先手工搬文件。


def test_distill_accepts_trace_files_directly(tmp_path):
    golden = tmp_path / "g.jsonl"
    golden.write_text(
        json.dumps(
            {
                "id": "T1",
                "input": "1001 那单花了多少钱",
                "split": "fit",
                "assertions": {"expect_tools": ["get_order"], "must_have": ["899"]},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    trace_path = tmp_path / "wf.jsonl"
    save_traces([make_event_trace()], trace_path)

    out = tmp_path / "out"
    rc = distill_main(
        ["--trace", str(trace_path), "--golden", str(golden), "--out", str(out)]
    )

    assert rc == 0
    rows = [
        json.loads(line)
        for line in (out / "sft.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    # workflow 形态：Turn.calls 是空的，tool_calls 必须从事件投影出来
    assert rows[0]["tools"] == ["get_order"]
    assert any(m.get("tool_calls") for m in rows[0]["messages"])
    assert (out / "manifest.json").exists()


def test_distill_without_run_or_trace_returns_error():
    assert distill_main([]) == 1


def test_sample_check_flags_broken_tool_pairing():
    """产物检查要能挑出坏样本，否则"条数对"会被当成"内容对"。"""
    good = {
        "messages": [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_order",
                            "arguments": '{"order_id": 1001}',
                        },
                    }
                ],
            },
            {"role": "tool", "name": "get_order", "content": "ok"},
            {"role": "assistant", "content": "答"},
        ]
    }
    assert _check_sample(good) == []

    broken = {
        "messages": [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"type": "function", "function": {"name": "get_order", "arguments": "{bad"}}
                ],
            },
            {"role": "tool", "name": "get_logistics", "content": ""},
        ]
    }
    problems = _check_sample(broken)
    assert any("arguments" in p for p in problems)
    assert any("不匹配" in p for p in problems)
    assert any("末条" in p for p in problems)
