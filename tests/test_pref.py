"""偏好对构造的规则测试。

每条测试钉一条规则，规则错了就说明配对逻辑有问题——不是"预期值不合适"。
"""

from __future__ import annotations

import json

from pipeline import pref as P


def cand(
    task_id="S15",
    agent="shop-cold",
    split="fit",
    users=("订单 9999 什么状态",),
    answer="查不到该订单",
    file="a.jsonl",
    line=1,
    outcome=P.PASS,
    failure_type="",
    trace=None,
):
    messages = []
    for u in users:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": answer})
    return P.Candidate(
        task_id=task_id,
        agent=agent,
        split=split,
        messages=messages,
        source={"file": file, "line": line},
        outcome=outcome,
        failure_type=failure_type,
        source_trace=trace,
    )


def test_same_task_same_agent_makes_one_pair():
    good = cand(answer="查不到该订单，请确认单号", file="sft.jsonl", line=3)
    bad = cand(answer="订单已发货，金额 429 元", file="failed.jsonl", line=1, outcome=P.FAIL, failure_type="wrong_fact")

    result = P.build([good], [bad])

    assert len(result["pairs"]) == 1
    pair = result["pairs"][0]
    assert pair["task_id"] == "S15"
    assert pair["agent"] == "shop-cold"
    assert pair["split"] == "fit"
    assert pair["failure_type"] == "wrong_fact"
    assert pair["basis"] == "same_agent_rerun"
    assert pair["prompt"] == ["订单 9999 什么状态"]
    assert pair["chosen"]["source"] == {"file": "sft.jsonl", "line": 3}
    assert pair["rejected"]["source"] == {"file": "failed.jsonl", "line": 1}


def test_different_agent_is_not_paired_by_default():
    """两侧答案不同、问法相同，唯一差别是 Agent —— 仍然不配对。

    刻意让答案不同：否则这条测试会因为"答案相同"而通过，测不到 agent 判定。
    """
    good = cand(agent="wf:shop-demo", answer="中通快递，运单 ZT987", file="sft.jsonl", line=2)
    bad = cand(
        agent="wf:canvas-demo",
        answer="我只能查到订单，没有物流信息",
        file="failed.jsonl",
        line=1,
        outcome=P.FAIL,
        failure_type="missing_tool",
    )

    result = P.build([good], [bad])

    assert result["pairs"] == []
    assert result["report"]["skipped"].get("agent_mismatch") == 1
    assert result["report"]["counts"]["cross_agent_pairs"] == 0


def test_fail_side_without_agent_is_marked_unverified():
    """历史 failed 产物没有 agent 字段：允许配对，但必须标出来，不能装作验证过。"""
    good = cand(agent="shop-cold", answer="查不到该订单", file="sft.jsonl", line=1)
    bad = cand(
        agent="",
        answer="订单已发货，金额 429 元",
        file="failed.jsonl",
        line=1,
        outcome=P.FAIL,
        failure_type="wrong_fact",
    )

    result = P.build([good], [bad])

    assert len(result["pairs"]) == 1
    pair = result["pairs"][0]
    assert pair["agent"] == "shop-cold"
    assert pair["agent_verified"] is False
    assert pair["basis"] == "agent_unverified"
    assert result["report"]["counts"]["unverified_agent_pairs"] == 1


def test_cross_agent_can_be_enabled_explicitly():
    """跨 Agent 只作为显式选项：默认关，开了要能标出来。

    两侧答案必须不同——答案一样就没有优劣，不是偏好对。
    """
    good = cand(agent="wf:shop-demo", answer="中通快递，运单 ZT987", file="sft.jsonl", line=2)
    bad = cand(
        agent="wf:canvas-demo",
        answer="我只能查到订单，没有物流信息",
        file="failed.jsonl",
        line=1,
        outcome=P.FAIL,
        failure_type="missing_tool",
    )

    result = P.build([good], [bad], allow_cross_agent=True)

    assert len(result["pairs"]) == 1
    assert result["pairs"][0]["basis"] == "cross_agent"
    assert result["report"]["counts"]["cross_agent_pairs"] == 1


def test_pass_without_fail_makes_no_pair():
    result = P.build([cand()], [])

    assert result["pairs"] == []
    assert result["report"]["counts"]["tasks_with_both"] == 0


def test_prompt_mismatch_is_rejected():
    """同 task 但用户问法不同，不是同一道题，不能配对。"""
    good = cand(users=("订单 1002 什么状态",), file="sft.jsonl", line=1)
    bad = cand(users=("订单 1001 什么状态",), file="failed.jsonl", line=1, outcome=P.FAIL, failure_type="wrong_fact")

    result = P.build([good], [bad])

    assert result["pairs"] == []
    assert result["report"]["skipped"].get("prompt_mismatch") == 1


def test_same_answer_is_not_a_preference():
    """两侧答案一样就没有优劣，不能当偏好对。"""
    good = cand(answer="查不到该订单", file="sft.jsonl", line=1)
    bad = cand(answer="查不到该订单", file="failed.jsonl", line=1, outcome=P.FAIL, failure_type="wrong_fact")

    result = P.build([good], [bad])

    assert result["pairs"] == []
    assert result["report"]["skipped"].get("same_answer") == 1


def test_split_must_match():
    """holdout 的失败不能拿 train 的成功做对照，否则验证集泄漏进训练。"""
    good = cand(split="fit", file="sft.jsonl", line=1)
    bad = cand(split="holdout", file="failed.jsonl", line=1, outcome=P.FAIL, failure_type="wrong_fact")

    result = P.build([good], [bad])

    assert result["pairs"] == []
    assert result["report"]["skipped"].get("split_mismatch") == 1


def test_chosen_is_stable_by_source_order():
    """同 task 多条 PASS 时，按文件路径 + 行号取第一条，不看答案长度。"""
    long_one = cand(answer="很长的正确答案" * 20, file="z.jsonl", line=9)
    short_one = cand(answer="短答案", file="a.jsonl", line=2)
    bad = cand(answer="错误答案", file="f.jsonl", line=1, outcome=P.FAIL, failure_type="wrong_fact")

    result = P.build([long_one, short_one], [bad])

    assert len(result["pairs"]) == 1
    assert result["pairs"][0]["chosen"]["source"] == {"file": "a.jsonl", "line": 2}


def test_output_is_stable_across_runs():
    good = cand(file="sft.jsonl", line=5)
    bad = cand(answer="另一个答案", file="failed.jsonl", line=2, outcome=P.FAIL, failure_type="wrong_fact")

    first = P.build([good], [bad])
    second = P.build([good], [bad])

    assert json.dumps(first, ensure_ascii=False, sort_keys=True) == json.dumps(
        second, ensure_ascii=False, sort_keys=True
    )


def test_cli_writes_pairs_and_report(tmp_path):
    """接线测试：脚本能导入、能落盘、报告里有计数。"""
    from scripts import build_pref

    src = tmp_path / "pass.jsonl"
    src.write_text(
        json.dumps(
            {
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
    bad = tmp_path / "fail.jsonl"
    bad.write_text(
        json.dumps(
            {
                "task_id": "S01",
                "agent": "shop-cold",
                "split": "fit",
                "failure_type": "wrong_fact",
                "messages": [
                    {"role": "user", "content": "订单 1002 什么状态"},
                    {"role": "assistant", "content": "已签收"},
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "pref"
    build_pref.PASS_INPUTS = (src,)
    build_pref.FAIL_INPUTS = (bad,)

    assert build_pref.main(["--out", str(out)]) == 0

    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["counts"]["pairs"] == 1
    assert report["by_failure_type"] == {"wrong_fact": 1}
    rows = (out / "pairs.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0])["chosen"]["answer"] == "已发货"
