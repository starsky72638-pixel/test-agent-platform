"""统一入口：按名字加载一个 Agent，跑一条问题，落一条 Trace。

为什么单独一个脚本：
    run_shop.py 直接 import 了 shop（第 43 行）、probe.py 的 --agent 只认 cold/tuned，
    两者都是"某次实验的入口"，不是"框架的入口"。
    这里只做一件事：名字 -> Agent -> 一条任务 -> 一条轨迹，
    不判分、不学经验、不写报告，所以换任何 Agent 都不用改这个文件。

执行路径全部复用现有部件：
    builtin.make(name, max_steps=...)  接入层按名字造 Agent
    core.runner.run_task(agent, task)  崩溃隔离、split 校正、耗时兜底
    core.trace.save_traces([trace])    落盘为可直接回读的 jsonl

用法：
    python scripts/run_agent.py --list
    python scripts/run_agent.py --agent shop-cold --question "预算 500 以内有什么推荐"
    python scripts/run_agent.py --agent main --question "..." --max-steps 8

退出码：0 执行成功（status=ok）；1 轨迹存在但状态异常（crashed / empty_trace）；
        2 参数或 Agent 名字有问题，没有发起执行。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# 允许直接运行：python scripts/run_agent.py
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agents.builtin as builtin
from config import PROJECT_ROOT
from core.registry import UnknownAgentError
from core.runner import run_task
from core.task import SPLIT_UNASSIGNED, Task
from core.trace import (
    KIND_ERROR,
    KIND_LLM_CALL,
    KIND_NODE_END,
    KIND_NODE_START,
    KIND_SLOT_REUSE,
    KIND_TOOL_CALL,
    save_traces,
)

RUNS_DIR = PROJECT_ROOT / "runs" / "adhoc"
SEPARATOR = "-" * 74


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="按名字加载一个 Agent，跑一条问题并落 Trace",
    )
    p.add_argument("--list", action="store_true", help="列出可接入的 Agent 后退出")
    p.add_argument(
        "--agent",
        default=builtin.DEFAULT_AGENT,
        help=f"Agent 名字（默认 {builtin.DEFAULT_AGENT}），用 --list 查看全部",
    )
    p.add_argument("--question", default="", help="单轮用户问题")
    p.add_argument(
        "--workflow",
        default="",
        help="Workflow id 或 json 路径，配合 --agent workflow 使用",
    )
    p.add_argument("--max-steps", type=int, default=6, help="单轮步数上限（默认 6）")
    p.add_argument(
        "--out",
        default=None,
        help="Trace 输出根目录，默认 runs/adhoc",
    )
    return p.parse_args(argv)


def build_task(question: str, *, task_id: str | None = None, split: str = SPLIT_UNASSIGNED) -> Task:
    """把一条临时问题包成 Task。

    split 固定 unassigned：临时问题不属于任何评测集，
    不能让它的轨迹混进 fit/holdout 的统计里。
    """
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return Task(id=task_id or f"adhoc-{stamp}", input=question, split=split)


def print_calls(trace) -> None:
    """按发生顺序打印工具调用，只做展示，不影响任何判定。"""
    calls = trace.all_calls()
    print(f"工具调用 {len(calls)} 次")
    for i, call in enumerate(calls, 1):
        try:
            args = json.dumps(call.args, ensure_ascii=False)
        except (TypeError, ValueError):
            args = str(call.args)
        if len(args) > 70:
            args = args[:69] + "…"
        flag = "ok" if call.ok else f"ERROR {call.error[:40]}"
        print(f"  {i}. {call.name} {args} {flag}")


def print_events(trace) -> None:
    """按发生顺序打印节点级执行过程。

    与 print_calls 的区别：calls 只说"调了什么工具"，
    events 还说"哪个节点、什么时候、花了多久、有没有分支或异常"。
    没有事件的轨迹（ReAct Agent）不打印，保持原输出不变。
    """
    if not trace.events:
        return
    nodes = sum(1 for e in trace.events if e.kind == KIND_NODE_END)
    print(f"节点执行 {nodes} 个 | 事件 {len(trace.events)} 条")
    for e in trace.events:
        if e.kind == KIND_NODE_START:
            print(f"  [{e.seq}] > {e.node_id} ({e.node_type})")
        elif e.kind == KIND_TOOL_CALL:
            try:
                args = json.dumps(e.args, ensure_ascii=False)
            except (TypeError, ValueError):
                args = str(e.args)
            if len(args) > 60:
                args = args[:59] + "…"
            flag = "ok" if e.ok else f"ERROR {(e.error or '')[:40]}"
            print(f"  [{e.seq}]   tool {e.name} {args} {flag} {e.elapsed_ms}ms")
        elif e.kind == KIND_LLM_CALL:
            head = (e.output or "").replace("\n", " ")[:50]
            print(f"  [{e.seq}]   llm  {e.name} {e.elapsed_ms}ms -> {head}")
        elif e.kind == KIND_NODE_END:
            print(f"  [{e.seq}]   < {e.node_id} {e.elapsed_ms}ms")
        elif e.kind == KIND_SLOT_REUSE:
            # 只在这一轮真的沿用了上一轮的值时出现，不是每次都打。
            print(f"  [{e.seq}]   slot {e.name}={e.output} ← t{e.args.get('from_turn')}")
        elif e.kind == KIND_ERROR:
            print(f"  [{e.seq}]   ERROR {(e.error or '')[:80]}")


def run_once(
    agent,
    question: str,
    *,
    max_steps: int = 6,
    out_dir=None,
    task_id: str | None = None,
):
    """跑一条问题并落盘。返回 (trace, trace 文件路径)。

    不判分：这一层只负责执行与记录。要看对错请走 run_shop / run_loop。
    """
    task = build_task(question, task_id=task_id)
    trace = run_task(agent, task, max_steps=max_steps)

    root = Path(out_dir) if out_dir else RUNS_DIR
    path = save_traces([trace], root / task.id / f"{task.id}.jsonl")
    return trace, path


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.list:
        print("可接入的 Agent：")
        for name in builtin.names():
            mark = "（默认）" if name == builtin.DEFAULT_AGENT else ""
            print(f"  {name}{mark}")
        if builtin.ALIASES:
            pairs = ", ".join(f"{k} -> {v}" for k, v in sorted(builtin.ALIASES.items()))
            print(f"别名：{pairs}")
        return 0

    question = (args.question or "").strip()
    if not question:
        print('[错误] 缺少 --question。用法：--agent <名字> --question "..."')
        print("       可先用 --list 查看可用 Agent。")
        return 2

    try:
        kwargs = {"max_steps": args.max_steps}
        # 只有真要用 workflow 时才传这个参数：别的工厂不认识它，
        # 无脑透传会让 --workflow 配 --agent customer 直接变成 TypeError。
        if args.workflow and builtin.ALIASES.get(args.agent, args.agent) == "workflow":
            kwargs["workflow"] = args.workflow
        agent = builtin.make(args.agent, **kwargs)
    except UnknownAgentError as exc:
        # 这条异常的信息里已经带了"当前可用"的名字，原样打出即可。
        print(f"[错误] {exc.args[0] if exc.args else exc}")
        return 2
    except (TypeError, ValueError) as exc:
        print(f"[错误] {type(exc).__name__}: {exc}")
        return 2

    print(f"Agent {agent.name} | 请求名 {args.agent} | 步数上限 {args.max_steps}")
    print(SEPARATOR)

    started = time.perf_counter()
    trace, path = run_once(
        agent,
        question,
        max_steps=args.max_steps,
        out_dir=args.out,
    )
    elapsed = int((time.perf_counter() - started) * 1000)

    print(f"状态 {trace.status} | 耗时 {elapsed}ms | 轮次 {len(trace.turns)}")
    if trace.error:
        print(f"错误 {trace.error}")
    if trace.events:
        print_events(trace)
    else:
        print_calls(trace)

    print(SEPARATOR)
    print("回答：")
    print(trace.final_answer() or "（空）")
    print(SEPARATOR)
    print(f"Trace：{path}")
    return 0 if trace.status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
