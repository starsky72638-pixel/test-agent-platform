"""SFT 质量闸门的离线测试。

十种情况各一条，全部是纯函数级别的判断，不读真实数据文件：
1 精确重复  2 答案冲突  3 holdout 隔离  4 unknown 隔离  5 缺元信息
6 非法 tool_calls  7 无 tool_calls 的纯文本  8 输出稳定  9 schema 齐全
10 工具调用与响应不匹配
"""

from __future__ import annotations

import json

from pipeline import sft


def make_sample(
    user: str,
    answer: str,
    *,
    agent: str | None = "shop-cold",
    split: str | None = "fit",
    task_id: str | None = "S01",
    sid: str | None = None,
    calls: list[tuple[str, dict]] | None = None,
    tool_names: list[str] | None = None,
):
    """造一条 SFT 样本。tool_names 用来故意制造调用/响应不匹配。"""
    messages: list[dict] = [{"role": "user", "content": user}]
    for name, args in calls or []:
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)},
                    }
                ],
            }
        )
    for name in tool_names if tool_names is not None else [n for n, _ in (calls or [])]:
        messages.append({"role": "tool", "name": name, "content": "status=200"})
    messages.append({"role": "assistant", "content": answer})

    raw: dict = {"messages": messages}
    if task_id is not None:
        raw["task_id"] = task_id
    if agent is not None:
        raw["agent"] = agent
    if split is not None:
        raw["split"] = split
    if sid is not None:
        raw["id"] = sid
    return (raw, {"file": "t.jsonl", "line": 1})


def build(*items):
    return sft.build(list(items))


def test_exact_duplicate_is_kept_once():
    result = build(
        make_sample("订单 1002 什么状态", "已发货，429 元", sid="sft-b"),
        make_sample("订单 1002 什么状态", "已发货，429 元", sid="sft-a"),
    )

    assert len(result["train"]) == 1
    assert result["report"]["counts"]["exact_duplicates"] == 1
    assert result["train"][0]["sample_id"] == "sft-a"
    dup = [r for r in result["quarantine"] if r["reason"] == "exact_duplicate"]
    assert [r["sample_id"] for r in dup] == ["sft-b"]


def test_same_input_different_answer_goes_to_quarantine():
    """不随机留一条：我们并不知道哪个答案对。"""
    result = build(
        make_sample("订单 1002 什么状态", "已发货", sid="sft-1"),
        make_sample("订单 1002 什么状态", "已签收", sid="sft-2"),
    )

    assert result["train"] == []
    counts = result["report"]["counts"]
    assert counts["conflicting_inputs"] == 1
    assert counts["conflict_samples"] == 2
    assert {r["reason"] for r in result["quarantine"]} == {"answer_conflict"}
    assert result["report"]["conflicts"][0]["answers"] == 2


def test_holdout_never_enters_train():
    result = build(make_sample("订单 1002 什么状态", "已发货", split="holdout"))

    assert result["train"] == []
    assert len(result["holdout"]) == 1
    assert result["quarantine"] == []


def test_unknown_split_is_quarantined():
    result = build(make_sample("订单 1001 金额", "899 元", split="validation"))

    assert result["train"] == []
    assert result["quarantine"][0]["reason"] == "unknown_split"
    assert result["report"]["counts"]["unknown_samples"] == 1


def test_missing_metadata_is_marked_but_trainable():
    """缺 agent / task_id 用 unknown 占位并计数；split 明确为 fit 时仍可训练。"""
    result = build(
        make_sample("订单 1001 金额", "899 元", agent=None, task_id=None, split="fit")
    )

    row = result["train"][0]
    assert row["agent"] == "unknown"
    assert row["task_id"] == "unknown"
    assert row["missing_fields"] == ["task_id", "agent"]
    assert result["report"]["counts"]["missing_field_samples"] == 1


def test_bad_tool_arguments_are_quarantined():
    raw, source = make_sample("订单 1002 什么状态", "已发货", calls=[("get_order", {"order_id": 1002})])
    raw["messages"][1]["tool_calls"][0]["function"]["arguments"] = "{order_id: 1002"

    result = sft.build([(raw, source)])

    assert result["train"] == []
    assert result["quarantine"][0]["reason"] == "tool_call_bad_arguments"
    assert result["report"]["counts"]["bad_tool_calls"] == 1


def test_plain_text_sample_without_tool_calls_is_trainable():
    result = build(make_sample("你们几点下班", "工作日到 18 点"))

    row = result["train"][0]
    assert row["tool_calls"] == []
    assert row["messages"][-1]["content"] == "工作日到 18 点"


def test_tool_response_mismatch_is_quarantined():
    result = build(
        make_sample(
            "订单 1002 什么状态",
            "已发货",
            calls=[("get_order", {"order_id": 1002})],
            tool_names=["get_logistics"],
        )
    )

    assert result["quarantine"][0]["reason"] == "tool_response_mismatch"
    assert result["report"]["counts"]["bad_tool_calls"] == 1


def test_output_is_stable_across_runs():
    items = [
        make_sample("订单 1002 什么状态", "已发货", sid="sft-b"),
        make_sample("订单 1001 金额", "899 元", sid="sft-a"),
        make_sample("订单 1002 什么状态", "已发货", sid="sft-c"),
    ]

    first = build(*items)
    second = build(*items)

    assert json.dumps(first["train"], ensure_ascii=False) == json.dumps(
        second["train"], ensure_ascii=False
    )
    assert json.dumps(first["quarantine"], ensure_ascii=False) == json.dumps(
        second["quarantine"], ensure_ascii=False
    )
    assert [r["sample_id"] for r in first["train"]] == ["sft-a", "sft-b"]


def test_sample_carries_required_schema_fields():
    result = build(make_sample("订单 1002 什么状态", "已发货", sid="sft-x"))
    row = result["train"][0]

    required = {
        "sample_id",
        "task_id",
        "agent",
        "split",
        "source_trace",
        "tool_calls",
        "messages",
    }
    assert required <= set(row)
    # 旧文件没有 source_trace，必须写 null，不能编一个
    assert row["source_trace"] is None
    assert row["source"] == {"file": "t.jsonl", "line": 1}


# ---------------------------------------------------------------- CLI 接线


def test_build_sft_cli_writes_outputs(tmp_path):
    """真跑一次 CLI：导入路径、参数解析、落盘、报告都要通。

    这条是补的：纯函数测试再怎么全，也照不到 `scripts/` 里的导入写错
    （曾经把 pipeline.provenance 写成根目录模块，单测全绿、脚本一跑就崩）。
    """
    from scripts import build_sft

    src = tmp_path / "sft.jsonl"
    src.write_text(
        json.dumps(
            {
                "id": "sft-x",
                "task_id": "S01",
                "agent": "shop-cold",
                "split": "fit",
                "messages": [
                    {"role": "user", "content": "订单 1002 什么状态"},
                    {"role": "assistant", "content": "已发货"},
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"

    assert build_sft.main(["--in", str(src), "--out", str(out)]) == 0

    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["counts"]["train_samples"] == 1
    assert report["counts"]["holdout_samples"] == 0
    assert report["inputs"][0]["rows"] == 1
    assert len((out / "train.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    assert (out / "holdout.jsonl").exists()
    assert (out / "quarantine.jsonl").exists()


def test_merge_memory_script_imports():
    """同类风险：归并脚本也曾按根目录模块写导入。"""
    from scripts import merge_memory

    assert merge_memory.DEFAULT_STORE_PATH.name == "experiences.jsonl"
