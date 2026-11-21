"""SFT 质量闸门的离线测试。

覆盖四类判定与它们之间的边界：
duplicate / rerun_variance / true_conflict / unresolved，
以及 split 隔离、canonical 选取顺序、输出稳定性、CLI 接线。

全部是纯函数级别的判断，不读真实数据文件。
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
    file: str = "t.jsonl",
    line: int = 1,
):
    """造一条 SFT 样本。file/line 决定 canonical 选取顺序。"""
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
    return (raw, {"file": file, "line": line})


def build(*items):
    return sft.build(list(items))


def reasons_of(result) -> list[str]:
    return [r["reason"] for r in result["quarantine"]]


# ---------------------------------------------------------------- duplicate


def test_same_origin_same_answer_is_duplicate():
    result = build(
        make_sample("订单 1002 什么状态", "已发货", sid="sft-b", file="b.jsonl", line=1),
        make_sample("订单 1002 什么状态", "已发货", sid="sft-a", file="a.jsonl", line=1),
    )

    assert [r["sample_id"] for r in result["train"]] == ["sft-a"]
    assert reasons_of(result) == ["duplicate"]
    assert result["report"]["counts"]["duplicate_samples"] == 1
    assert result["quarantine"][0]["canonical_sample_id"] == "sft-a"


# ---------------------------------------------------------------- rerun


def test_same_origin_different_answer_is_rerun_variance():
    """同 task_id + agent + split 的两次运行：留一条 canonical，另一条记 rerun。"""
    result = build(
        make_sample("订单 1002 什么状态", "已发货，429 元", sid="sft-1", file="a.jsonl", line=1),
        make_sample("订单 1002 什么状态", "已发货", sid="sft-2", file="b.jsonl", line=7),
    )

    assert [r["sample_id"] for r in result["train"]] == ["sft-1"]
    counts = result["report"]["counts"]
    assert counts["rerun_variance_groups"] == 1
    assert counts["rerun_variance_samples"] == 2
    assert counts["true_conflict_groups"] == 0
    assert reasons_of(result) == ["rerun_variance"]
    assert result["quarantine"][0]["canonical_sample_id"] == "sft-1"

    reruns = result["report"]["reruns"]
    assert reruns[0]["canonical_sample_id"] == "sft-1"
    assert reruns[0]["rerun_sample_ids"] == ["sft-2"]


def test_canonical_follows_source_order_not_sample_id():
    """canonical 由源文件路径 + 行号决定，与 sample_id 字母序无关。"""
    result = build(
        make_sample("同一个问题", "答案 B", sid="sft-aaa", file="b.jsonl", line=5),
        make_sample("同一个问题", "答案 A", sid="sft-zzz", file="a.jsonl", line=9),
        make_sample("同一个问题", "答案 C", sid="sft-mmm", file="c.jsonl", line=1),
    )

    assert [r["sample_id"] for r in result["train"]] == ["sft-zzz"]
    assert result["report"]["reruns"][0]["rerun_sample_ids"] == ["sft-aaa", "sft-mmm"]


# ---------------------------------------------------------------- true conflict


def test_different_agent_is_true_conflict():
    """不同 Agent 不自动合并，也不随机留一条。"""
    result = build(
        make_sample("订单 1002 什么状态", "已发货", agent="shop-cold", sid="sft-1"),
        make_sample("订单 1002 什么状态", "已签收", agent="main", sid="sft-2"),
    )

    assert result["train"] == []
    counts = result["report"]["counts"]
    assert counts["true_conflict_groups"] == 1
    assert counts["true_conflict_samples"] == 2
    assert set(reasons_of(result)) == {"true_conflict"}
    entry = result["report"]["groups"][0]
    assert entry["kind"] == "true_conflict"
    assert {r["agent"] for r in entry["rows"]} == {"shop-cold", "main"}


def test_different_task_id_same_input_is_true_conflict():
    result = build(
        make_sample("同一个问题", "答案甲", task_id="S01", sid="sft-1"),
        make_sample("同一个问题", "答案乙", task_id="S02", sid="sft-2"),
    )

    assert result["train"] == []
    assert result["report"]["groups"][0]["kind"] == "true_conflict"


def test_unresolved_when_origin_is_unknown():
    """来源里有 unknown，就无法判断是否同源，不猜。"""
    result = build(
        make_sample("同一个问题", "答案甲", agent=None, sid="sft-1"),
        make_sample("同一个问题", "答案乙", agent="shop-cold", sid="sft-2"),
    )

    assert result["train"] == []
    assert set(reasons_of(result)) == {"unresolved"}
    assert result["report"]["counts"]["unresolved_groups"] == 1


# ---------------------------------------------------------------- split 隔离


def test_holdout_never_enters_train():
    result = build(make_sample("订单 1002 什么状态", "已发货", split="holdout"))

    assert result["train"] == []
    assert len(result["holdout"]) == 1
    assert result["quarantine"] == []


def test_unknown_never_enters_train():
    result = build(make_sample("订单 1001 金额", "899 元", split="validation"))

    assert result["train"] == []
    assert result["quarantine"][0]["reason"] == "unknown_split"
    assert result["report"]["counts"]["unknown_samples"] == 1


def test_unknown_sample_cannot_block_a_trainable_one():
    """unknown 不参与训练侧判定，没有资格否掉同问句的 fit 样本。"""
    result = build(
        make_sample("订单 1002 什么状态", "已发货", split="fit", sid="sft-fit"),
        make_sample("订单 1002 什么状态", "查不到", split=None, sid="sft-unknown"),
    )

    assert [r["sample_id"] for r in result["train"]] == ["sft-fit"]
    assert result["report"]["counts"]["true_conflict_groups"] == 0
    assert reasons_of(result) == ["unknown_split"]


def test_same_origin_across_splits_is_not_merged():
    """同 task_id + agent 但 split 不同：各在各池，不跨池选 canonical。"""
    result = build(
        make_sample("订单 1002 什么状态", "已发货", split="fit", sid="sft-fit"),
        make_sample("订单 1002 什么状态", "已签收", split="holdout", sid="sft-hold"),
    )

    assert [r["sample_id"] for r in result["train"]] == ["sft-fit"]
    assert [r["sample_id"] for r in result["holdout"]] == ["sft-hold"]
    assert result["report"]["groups"] == []


# ---------------------------------------------------------------- 结构与 schema


def test_missing_metadata_is_marked_but_trainable():
    result = build(
        make_sample("订单 1001 金额", "899 元", agent=None, task_id=None, split="fit")
    )

    row = result["train"][0]
    assert row["agent"] == "unknown"
    assert row["task_id"] == "unknown"
    assert row["missing_fields"] == ["task_id", "agent"]
    assert result["report"]["counts"]["missing_field_samples"] == 1


def test_bad_tool_arguments_are_quarantined():
    raw, source = make_sample(
        "订单 1002 什么状态", "已发货", calls=[("get_order", {"order_id": 1002})]
    )
    raw["messages"][1]["tool_calls"][0]["function"]["arguments"] = "{order_id: 1002"

    result = sft.build([(raw, source)])

    assert result["train"] == []
    assert result["quarantine"][0]["reason"] == "tool_call_bad_arguments"
    assert result["report"]["counts"]["bad_tool_calls"] == 1


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


def test_tool_call_id_style_is_accepted():
    """OpenAI 形态：tool 消息没有 name，只有 tool_call_id，不该被判成结构异常。"""
    messages = [
        {"role": "user", "content": "预算 200 元以内有什么推荐"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_0c2d",
                    "type": "function",
                    "function": {
                        "name": "search_products",
                        "arguments": '{"max_price": 200}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_0c2d", "content": "status=200"},
        {"role": "assistant", "content": "只找到一款 159 元的充电宝"},
    ]

    result = sft.build(
        [
            (
                {
                    "task_id": "S01",
                    "agent": "shop-cold",
                    "split": "fit",
                    "messages": messages,
                },
                {"file": "t", "line": 1},
            )
        ]
    )

    assert result["report"]["counts"]["bad_tool_calls"] == 0
    assert len(result["train"]) == 1
    assert result["train"][0]["tool_calls"] == [
        {"name": "search_products", "arguments": '{"max_price": 200}'}
    ]


def test_plain_text_sample_without_tool_calls_is_trainable():
    result = build(make_sample("你们几点下班", "工作日到 18 点"))

    row = result["train"][0]
    assert row["tool_calls"] == []
    assert row["messages"][-1]["content"] == "工作日到 18 点"


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
    assert row["source_trace"] is None  # 旧文件没有，写 null 不编造
    assert row["source"] == {"file": "t.jsonl", "line": 1}


def test_output_is_stable_across_runs():
    items = [
        make_sample("订单 1002 什么状态", "已发货", sid="sft-b", file="b.jsonl", line=2),
        make_sample("订单 1001 金额", "899 元", sid="sft-a", file="a.jsonl", line=1),
        make_sample("订单 1002 什么状态", "已发货了", sid="sft-c", file="c.jsonl", line=3),
    ]

    first = build(*items)
    second = build(*items)

    for key in ("train", "holdout", "quarantine"):
        assert json.dumps(first[key], ensure_ascii=False) == json.dumps(
            second[key], ensure_ascii=False
        )
    assert json.dumps(first["report"], ensure_ascii=False) == json.dumps(
        second["report"], ensure_ascii=False
    )


# ---------------------------------------------------------------- CLI 接线


def test_build_sft_cli_writes_outputs(tmp_path):
    """真跑一次 CLI：导入路径、参数解析、落盘、报告都要通。"""
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
