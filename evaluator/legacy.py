"""把历史报告里的 case 记录还原成 Trace，用于新旧判分器对拍。

为什么需要它：
reports/*.json 是旧判分器跑出来的 42 条真实轨迹，里面每条都带
rounds[].tool_results（每次调用与它自己那次返回的配对）。
把这些轨迹重新喂给新 evaluator，比较 passed / failure_type / fail_turn，
就能在不调一次 API 的前提下证明"新实现没有改变判据"。

为什么优先用 tool_results 而不是 outputs：
outputs 是按工具名合并的文本，同名多次调用会糊成一段，无法恢复逐次对齐。
tool_results 保留了 call_id 与各自的返回，是唯一无损的来源。

这一层不参与任何真实执行，只服务于对拍与回归。
"""

from __future__ import annotations

from core.trace import STATUS_OK, ToolCall, Trace, Turn


def trace_from_case(case: dict) -> Trace:
    """把报告里的一条 result 还原成 Trace。

    注意：不根据旧结论猜 status。infra_error 这类在轨迹里本来就空，
    用旧结论回填会让对拍变成自我循环。这类条目应在对拍集合里单独排除。
    """
    turns: list[Turn] = []

    for r in case.get("rounds") or []:
        turn_no = int(r.get("turn") or len(turns) + 1)
        results = r.get("tool_results") or []

        calls: list[ToolCall] = []
        if results:
            for item in results:
                calls.append(
                    ToolCall(
                        name=str(item.get("name") or ""),
                        args=dict(item.get("args") or {}),
                        call_id=str(item.get("call_id") or ""),
                        output=str(item.get("output") or ""),
                    )
                )
        else:
            # 退化路径：没有逐次记录时，只能把按名合并的返回挂回同名调用。
            merged = r.get("outputs") or {}
            for item in r.get("calls") or []:
                name = str(item.get("name") or "")
                calls.append(
                    ToolCall(
                        name=name,
                        args=dict(item.get("args") or {}),
                        output=str(merged.get(name) or ""),
                    )
                )

        turns.append(
            Turn(
                index=turn_no - 1,
                user_input=str(r.get("input") or ""),
                calls=calls,
                answer=str(r.get("answer") or ""),
                steps=int(r.get("steps") or 0),
                meta={
                    "status": r.get("status") or "ok",
                    "msg_count": r.get("msg_count"),
                },
            )
        )

    return Trace(
        task_id=str(case.get("id") or ""),
        agent="legacy_report",
        turns=turns,
        status=STATUS_OK,
        meta={
            "source": "report",
            "legacy_trace_id": case.get("trace_id"),
            "legacy_passed": case.get("passed"),
            "legacy_failure_type": case.get("failure_type"),
        },
    )


def trace_from_report(report: dict) -> list[Trace]:
    """整份报告 -> 轨迹列表。"""
    return [trace_from_case(c) for c in (report.get("results") or [])]
