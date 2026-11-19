"""feedback.py 的单测：三道闸门必须挡住脏线索。

这三个断言最要紧：
- 提示词版本不一致 → 不许配对（否则线索里写的是"在旧规则下这样答是错的"）
- 多轮 → 不许配对（配对要的是同一个输入、两种回答）
- 失败轨迹 → 不许进 golden 候选
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.feedback import (  # noqa: E402
    count_rules,
    diff_reports,
    explain_failure,
    make_golden,
    make_hint,
    split_prompt,
)

PROMPT_A = "你是客服。\n\n规则：\n1. 先查再答。\n2. 不许编。"
PROMPT_B = "你是客服。\n\n规则：\n1. 先查再答。\n2. 不许编。\n3. 状态查全。"


def make_case(
    cid: str = "S01",
    passed: bool = True,
    answer: str = "已查到答案",
    prompt: str = PROMPT_A,
    turns: list[str] | None = None,
    **extra,
) -> dict:
    """造一条最小可用的 case，形状和 replay 报告里的 results 一致。"""
    turns = turns or ["问题"]
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": turns[0]},
    ]
    if answer:
        messages.append({"role": "assistant", "content": answer})

    case = {
        "id": cid,
        "input": turns[0],
        "turns": turns,
        "note": "",
        "passed": passed,
        "answer": answer,
        "failure_type": "" if passed else "missing_tool",
        # 字段名沿用 replay 落盘的历史命名，内容就是完整轨迹消息
        "train_messages": messages,
        "rounds": [{"turn": i, "answer": answer} for i in range(1, len(turns) + 1)],
        "trace_id": f"trace-{cid}",
    }
    case.update(extra)
    return case


# =========================
# 闸门一：golden 候选只收 passed
# =========================

def test_golden_skips_failed_case():
    assert make_golden(make_case(passed=False), source="r.json") is None


def test_golden_keeps_passed_case_and_tags_prompt_version():
    sample = make_golden(make_case(passed=True), source="r.json")

    assert sample is not None
    assert sample["meta"]["case_id"] == "S01"
    assert sample["meta"]["source"] == "r.json"
    # PROMPT_A 里是 2 条规则，这个数字用来判断这批候选有没有过期
    assert sample["meta"]["prompt_rules"] == 2
    assert sample["messages"][0]["role"] == "system"
    # 候选必须留 trace，才能回到报告里查它当时到底调了什么工具
    assert sample["meta"]["trace_id"] == "trace-S01"


def test_golden_marks_multi_turn_case():
    sample = make_golden(make_case(turns=["第一问", "第二问"]), source="r.json")

    assert sample is not None
    assert sample["meta"].get("multi_turn") is True


# =========================
# 闸门二：prompt 必须一致
# =========================

def test_hint_rejects_prompt_mismatch():
    base = make_case(passed=False, prompt=PROMPT_A, answer="只查了订单")
    cur = make_case(passed=True, prompt=PROMPT_B, answer="订单和物流都查了")

    hint, why = make_hint(base, cur)

    assert hint is None
    assert "提示词" in why


def test_hint_builds_when_prompt_matches():
    base = make_case(passed=False, answer="只查了订单", tool_msg="缺 get_logistics")
    cur = make_case(passed=True, answer="订单和物流都查了")

    hint, why = make_hint(base, cur)

    assert why == ""
    assert hint is not None
    assert hint["wrong_answer"] == "只查了订单"
    assert hint["correct_answer"] == "订单和物流都查了"
    # 上下文是 system + user，不含 assistant
    assert [m["role"] for m in hint["prompt"]] == ["system", "user"]
    assert hint["failure_type"] == "missing_tool"
    assert "缺 get_logistics" in hint["reason"]
    # 两个 trace 都要留，才能回到各自的报告里核对
    assert hint["wrong_trace"] == "trace-S01"
    assert hint["correct_trace"] == "trace-S01"


# =========================
# 闸门三：多轮不配对
# =========================

def test_hint_rejects_multi_turn():
    base = make_case(passed=False, turns=["第一问", "第二问"])
    cur = make_case(passed=True, turns=["第一问", "第二问"])

    hint, why = make_hint(base, cur)

    assert hint is None
    assert "多轮" in why


def test_hint_rejects_when_baseline_already_passed():
    base = make_case(passed=True)
    cur = make_case(passed=True)

    hint, why = make_hint(base, cur)

    assert hint is None
    assert "本来就是通过" in why


def test_hint_rejects_when_still_failing():
    base = make_case(passed=False)
    cur = make_case(passed=False)

    hint, why = make_hint(base, cur)

    assert hint is None
    assert "仍然失败" in why


# =========================
# 对比与工具函数
# =========================

def test_diff_reports_splits_four_ways():
    base = {"results": [
        make_case("S01", passed=True),
        make_case("S02", passed=False),
        make_case("S03", passed=False),
        make_case("S04", passed=True),
    ]}
    cur = {"results": [
        make_case("S01", passed=True),
        make_case("S02", passed=True),
        make_case("S03", passed=False),
        make_case("S04", passed=False),
    ]}

    diff = diff_reports(base, cur)

    assert diff["fixed"] == ["S02"]
    assert diff["regressed"] == ["S04"]
    assert diff["still_fail"] == ["S03"]
    assert diff["still_pass"] == ["S01"]


def test_diff_reports_marks_uncomparable_case():
    base = {"results": [make_case("S01")]}
    cur = {"results": [make_case("S02")]}

    diff = diff_reports(base, cur)

    assert diff["missing"] == ["S01", "S02"]
    assert diff["fixed"] == []


def test_explain_failure_ignores_hit_marker():
    case = make_case(passed=False, tool_msg="命中", param_msg="命中", order_msg="未校验")

    reason = explain_failure(case)

    # "命中" 不是问题，不能混进失败原因；"未校验"也不该算错
    assert "命中" not in reason
    assert reason.startswith("missing_tool")


def test_split_prompt_stops_at_first_assistant():
    messages = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
        {"role": "assistant", "content": "A"},
        {"role": "tool", "content": "T"},
    ]

    prompt = split_prompt(messages)

    assert [m["role"] for m in prompt] == ["system", "user"]


def test_count_rules_only_counts_numbered_lines():
    assert count_rules(PROMPT_B) == 3
    # 正文里出现数字不算规则
    assert count_rules("规则：\n1. 见 2.5 节\n2. 别编") == 2
