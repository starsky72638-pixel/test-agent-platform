"""把跑过的轨迹变成训练数据。

输入是一次实验目录（runs/<stamp>/），输出三份东西：

    sft.jsonl       通过的轨迹 → 工具调用格式的 SFT 样本
    failed.jsonl    没通过的轨迹 → 连同失败类型和失败轮次留档
    manifest.json   条数、去重数、各类丢弃原因

━━ 为什么要有这一步 ━━

Trace 现在只被用来看和判分：跑完、判完、写在磁盘上，然后就没人再读。
对 Agent 系统来说这是浪费——一次成功的多轮工具调用，本身就是
"这类任务该怎么拆、该调哪个工具、拿到什么再答"的示范。
失败的那一半也不能删：它带着 failure_type 和 fail_turn，
是后训练里做对比、做 reward 的原始信号。

━━ 三道闸门，宁少不多 ━━

一、工具失败过的轨迹不进 SFT。
    一条轨迹里有过 ERROR 或重试，它示范的是"怎么出错"，不是"怎么做对"。
    这类放进失败集，标 reason=tool_error。

二、判分没过的轨迹不进 SFT。
    这条最容易被忽略：模型答得流畅、看着像对的，但判分说它漏了工具调用。
    只看文本收进来的话，就是在教模型"漏查也能答"。

三、去重按 (用户输入, 工具与参数序列, 答案) 三段的规范化哈希。
    同一道题跑两次常常只差几个标点，不合并的话
    训练集会被少数题刷满权重。

━━ 一个必须说清的边界 ━━

这里产出的是"轨迹格式"的样本，不是可以直接喂给某个训练框架的定稿。
字段按 OpenAI 的 tool_calls 形态组织（assistant.tool_calls + role=tool），
换框架时改一层转换即可，不用重跑实验。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

# 允许直接运行：python scripts/distill.py
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.task import load_many
from core.trace import STATUS_OK, load_traces
from evaluator.verdict import calls_of_turn, evaluate

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_GOLDENS = (
    PROJECT_ROOT / "data" / "shop_eval.jsonl",
    PROJECT_ROOT / "data" / "shop_eval_multi.jsonl",
    PROJECT_ROOT / "data" / "shop_eval_hard.jsonl",
)

# 实验目录里三种轨迹文件，分别对应三个阶段。
GROUPS = {
    "fit": "fit.jsonl",
    "baseline": "holdout-baseline.jsonl",
    "injected": "holdout-injected.jsonl",
}

# 答案太短的样本没有示范价值，多半是"好的""已为您查询"这类空话。
MIN_ANSWER_CHARS = 6

_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    """规范化：去掉空白差异，避免同一句话因排版不同算成两条。"""
    return _WS.sub("", str(text or "")).strip().lower()


def _signature(trace) -> str:
    """一条轨迹的指纹：问了什么、调了什么（含参数）、答了什么。"""
    parts = []
    for turn in trace.turns:
        parts.append(_norm(turn.user_input))
        for call in calls_of_turn(trace, turn):
            parts.append(f"{call.name}:{json.dumps(call.args, ensure_ascii=False, sort_keys=True)}")
        parts.append(_norm(turn.answer))
    raw = "|".join(parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _has_tool_error(trace) -> bool:
    return any(not call.ok for call in trace.all_calls())


def _to_messages(trace) -> list[dict]:
    """把一条多轮轨迹摊成对话。

    一次调用一条 assistant + 一条 tool：
    trace 里同一轮的多个调用是并行的，但训练数据里扁平化更稳，
    换框架时也不用处理并行分支。
    """
    messages: list[dict] = []
    for turn in trace.turns:
        messages.append({"role": "user", "content": turn.user_input})
        for call in calls_of_turn(trace, turn):
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.args, ensure_ascii=False),
                            },
                        }
                    ],
                }
            )
            messages.append(
                {"role": "tool", "name": call.name, "content": str(call.output or "")}
            )
        messages.append({"role": "assistant", "content": turn.answer})
    return messages


def distill_traces(traces, tasks: dict, *, min_answer: int = MIN_ANSWER_CHARS):
    """把轨迹分成 (sft 样本, 失败样本, 丢弃统计)。纯函数，不碰磁盘。"""
    sft: list[dict] = []
    failed: list[dict] = []
    dropped: dict[str, int] = {}
    seen: set[str] = set()
    duplicates = 0

    def drop(reason: str) -> None:
        dropped[reason] = dropped.get(reason, 0) + 1

    for trace in traces:
        task = tasks.get(trace.task_id)
        if task is None:
            drop("no_task")
            continue

        verdict = evaluate(trace, task)

        if not verdict.passed:
            # 失败轨迹不丢：它带着失败类型和失败轮次，是后训练的负样本来源。
            failed.append(
                {
                    "task_id": trace.task_id,
                    "split": trace.split,
                    "failure_type": verdict.failure_type,
                    "fail_turn": verdict.fail_turn,
                    "turns": len(trace.turns),
                    "messages": _to_messages(trace),
                }
            )
            continue

        if trace.status != STATUS_OK:
            drop("status_not_ok")
            continue
        if _has_tool_error(trace):
            drop("tool_error")
            continue
        if any(not str(t.answer or "").strip() for t in trace.turns):
            drop("empty_answer")
            continue
        if len(str(trace.turns[-1].answer or "").strip()) < min_answer:
            drop("answer_too_short")
            continue

        sig = _signature(trace)
        if sig in seen:
            duplicates += 1
            continue
        seen.add(sig)

        sft.append(
            {
                "id": f"sft-{sig}",
                "task_id": trace.task_id,
                "split": trace.split,
                "agent": trace.agent,
                "turns": len(trace.turns),
                "tools": sorted({c.name for c in trace.all_calls()}),
                "messages": _to_messages(trace),
            }
        )

    return sft, failed, {"dropped": dropped, "duplicates": duplicates}


def _load_group(run_dir: Path, group: str):
    path = run_dir / GROUPS[group]
    if not path.exists():
        return []
    return load_traces(path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="把实验轨迹蒸馏成 SFT / 失败样本")
    p.add_argument("--run", required=True, help="实验目录，例如 runs/full-cold/20261118-000730")
    p.add_argument(
        "--group",
        action="append",
        choices=sorted(GROUPS) + ["all"],
        default=None,
        help="用哪一组轨迹，可重复；默认 fit（学习阶段，没有注入经验，最干净）",
    )
    p.add_argument("--out", default=str(PROJECT_ROOT / "data" / "distill"))
    p.add_argument("--golden", action="append", default=None, help="任务文件，可重复传")
    p.add_argument("--min-answer", type=int, default=MIN_ANSWER_CHARS)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    if not run_dir.exists():
        print(f"[错误] 实验目录不存在：{run_dir}")
        return 1

    groups = args.group or ["fit"]
    if "all" in groups:
        groups = list(GROUPS)

    goldens = [Path(g) for g in (args.golden or DEFAULT_GOLDENS)]
    if not all(g.exists() for g in goldens):
        print("[错误] 任务集不全，缺少：" + "、".join(str(g) for g in goldens if not g.exists()))
        return 1
    tasks = {t.id: t for t in load_many(goldens)}

    out_dir = Path(args.out)
    total_sft: list[dict] = []
    total_failed: list[dict] = []
    stats: dict[str, dict] = {}

    for group in groups:
        traces = _load_group(run_dir, group)
        if not traces:
            print(f"[跳过] {group}：{run_dir / GROUPS[group]} 不存在或为空")
            continue
        sft, failed, extra = distill_traces(traces, tasks, min_answer=args.min_answer)
        total_sft.extend(sft)
        total_failed.extend(failed)
        stats[group] = {
            "traces": len(traces),
            "sft": len(sft),
            "failed": len(failed),
            **extra,
        }
        print(
            f"[{group}] 轨迹 {len(traces)} 条 → SFT {len(sft)} 条 / 失败 {len(failed)} 条"
            f" | 去重丢弃 {extra['duplicates']} 条 | 其它丢弃 {extra['dropped'] or '无'}"
        )

    if not stats:
        print("[错误] 一条轨迹都没读到，检查 --run 是否指到了带 jsonl 的那层目录")
        return 1

    # 跨组去重：同一道题在 baseline 和 injected 里都通过时，只留一条。
    uniq: dict[str, dict] = {}
    for row in total_sft:
        uniq.setdefault(row["id"], row)
    cross_group_dupes = len(total_sft) - len(uniq)
    total_sft = list(uniq.values())

    write_jsonl(out_dir / "sft.jsonl", total_sft)
    write_jsonl(out_dir / "failed.jsonl", total_failed)

    manifest = {
        "run": str(run_dir),
        "groups": groups,
        "min_answer": args.min_answer,
        "tasks": len(tasks),
        "total": {
            "sft": len(total_sft),
            "failed": len(total_failed),
            "duplicates": sum(s["duplicates"] for s in stats.values()) + cross_group_dupes,
        },
        "by_group": stats,
        "failure_types": {},
    }
    for row in total_failed:
        key = row.get("failure_type") or "unknown"
        manifest["failure_types"][key] = manifest["failure_types"].get(key, 0) + 1

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n合计 SFT {len(total_sft)} 条 / 失败 {len(total_failed)} 条")
    if manifest["failure_types"]:
        print("失败类型分布：" + "  ".join(f"{k}={v}" for k, v in manifest["failure_types"].items()))
    print(f"输出：{out_dir / 'sft.jsonl'}")
    print(f"      {out_dir / 'failed.jsonl'}")
    print(f"      {out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
