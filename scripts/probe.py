"""定点对照：同一道题、同一份提示词，只切换注入的经验文本。

它不跑学习、不碰经验库、不写报告，只回答一个问题：
    ctx.experience 传进去之后，Agent 的工具调用和回答到底变没变？

为什么需要这个脚本：
    run_loop 的对比里，同一道题两次结果经常自己翻转（temperature=0 也不是确定性的）。
    在那样的数据上，一次 FAIL -> PASS 既可能是经验起作用，也可能是抖动。
    要判断因果，必须把变量压到只剩一个：除了注入文本，其它全不动；
    并且每组重复跑，看"稳定出现的差异"，而不是看单次结果。

用法（两个窗口）：

    窗口 A  .venv\\Scripts\\python.exe mock_server.py
    窗口 B  .venv\\Scripts\\python.exe scripts\\probe.py --task H04 --repeat 2 --experience "回答签收状态前必须先调用 get_order 和 get_logistics，以工具返回为准，不能顺着用户的话说。"

    --experience 留空时，两组都不注入，用来量"同一条件下的抖动幅度"。
    这个数是判读的基准：如果空注入组自己就能翻来翻去，那注入组的差异要更大才算数。

输出三块：
    1. 无注入组：每次的判定、工具序列、注入字数
    2. 注入组：同题重跑，唯一变化是那段文本
    3. 对照：工具序列集合是否变化、判定是否变化、meta 是否记录到注入
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# 允许直接运行：python scripts/probe.py
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agents.builtin as builtin
from config import PROJECT_ROOT, SERVICE_BASE
from core.runner import run_task
from core.registry import UnknownAgentError
from core.task import load_many
from evaluator import evaluate

DEFAULT_GOLDENS = (
    PROJECT_ROOT / "data" / "shop_eval.jsonl",
    PROJECT_ROOT / "data" / "shop_eval_multi.jsonl",
    PROJECT_ROOT / "data" / "shop_eval_hard.jsonl",
)

SEPARATOR = "-" * 78
ANSWER_PREVIEW = 90


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="定点对照：只切换注入的经验文本")
    p.add_argument("--task", action="append", required=True, help="任务 id，可重复传")
    p.add_argument("--experience", default="", help="注入的经验文本；留空则两组都不注入")
    p.add_argument("--repeat", type=int, default=2, help="每组跑几次（默认 2）")
    p.add_argument("--agent", default="cold")
    p.add_argument("--max-steps", type=int, default=6)
    p.add_argument("--golden", action="append", default=None, help="任务文件，可重复传")
    p.add_argument("--no-reset", action="store_true", help="不在每次执行前重置 mock 状态")
    return p.parse_args(argv)


def check_server() -> None:
    """先确认被测服务活着，否则会跑出一堆 infra_error，看起来像经验有害。"""
    import httpx

    try:
        httpx.get(f"{SERVICE_BASE}/products/search", timeout=3.0)
    except httpx.HTTPError as exc:
        raise SystemExit(
            f"[错误] 连不上被测服务 {SERVICE_BASE}（{type(exc).__name__}）\n"
            f"       先另开一个窗口执行：.venv\\Scripts\\python.exe mock_server.py"
        ) from exc


def reset_server() -> None:
    """每次执行前把 mock 的退款记录清空，保证两组的初始状态相同。"""
    import httpx

    try:
        httpx.post(f"{SERVICE_BASE}/admin/reset", timeout=3.0)
    except httpx.HTTPError:
        pass


def one_line(text: str, limit: int = ANSWER_PREVIEW) -> str:
    flat = " ".join(str(text or "").split())
    return flat[:limit] + ("…" if len(flat) > limit else "")


def tools_of(trace) -> list[str]:
    return [c.name for c in trace.all_calls()]


def run_group(agent, task, *, experience: str, repeat: int, label: str, reset: bool):
    """跑一组，返回每次的记录。"""
    rows = []
    for i in range(repeat):
        if reset:
            reset_server()
        started = time.perf_counter()
        trace = run_task(agent, task, experience=experience)
        verdict = evaluate(trace, task)
        elapsed = int((time.perf_counter() - started) * 1000)
        row = {
            "trace": trace,
            "verdict": verdict,
            "elapsed": elapsed,
            "tools": tools_of(trace),
            "chars": int((trace.meta or {}).get("experience_chars") or 0),
            "applied": bool((trace.meta or {}).get("experience_applied")),
            # 恢复与计划都要能被直接看见：这两件事的证据不该只存在于
            # 最终回答里，否则"到底重试了没有"只能靠调用次数反推。
            "retries": sum(
                max(0, int(m.get("attempts") or 0) - 1)
                for t in trace.turns
                for m in (t.meta.get("recovery") or [])
            ),
            "plan": sum(len(t.meta.get("plan") or []) for t in trace.turns),
            "missed": sum(len(t.meta.get("plan_uncovered") or []) for t in trace.turns),
            # 最后一轮累积到的槽位数。要证明的是"跨轮事实没丢"，
            # 所以取最后一次出现的那条，不是各轮相加。
            "slots": next(
                (
                    len(t.meta.get("slots") or {})
                    for t in reversed(trace.turns)
                    if t.meta.get("slots")
                ),
                0,
            ),
            # 门开了却没拆出待办的轮次。和"计划=0项"必须分开：
            # 前者是计划调用失败，后者是这道题本来不需要拆。
            "plan_failed": [
                str(t.meta.get("plan_error"))
                for t in trace.turns
                if t.meta.get("plan_error")
            ],
        }
        rows.append(row)

        extra = ""
        if trace.status != "ok":
            flag = f"异常 {trace.status}"
            err = (getattr(trace, "error", "") or "").strip()
            if err:
                # 上一次 6 次 crashed 就是因为不打印这行，白查了一轮。
                extra = f"\n        err: {one_line(err, 160)}"
        elif verdict.passed:
            flag = "PASS"
        else:
            flag = f"FAIL {verdict.failure_type}@t{verdict.fail_turn}"

        tags = f"  注入={row['chars']}字"
        if row["retries"]:
            tags += f"  重试={row['retries']}次"
        if row["plan"]:
            tags += f"  计划={row['plan']}项/漏{row['missed']}"
        if row["plan_failed"]:
            tags += f"  计划失败={len(row['plan_failed'])}轮({row['plan_failed'][0]})"
        if row["slots"]:
            tags += f"  槽位={row['slots']}项"
        print(
            f"[{label}{i + 1}] {flag:<24} {elapsed:>6}ms  "
            f"tools={row['tools'] or '-'}{tags}"
        )
        if extra:
            print(f"        {extra.strip()}")
        answers = [t.answer for t in trace.turns if t.answer]
        if answers:
            print(f"        答: {one_line(answers[-1])}")
    return rows


def summarize(rows) -> dict:
    seqs = [tuple(r["tools"]) for r in rows]
    return {
        "seqs": sorted({s for s in seqs}),
        "seq_list": seqs,
        "passed": sum(1 for r in rows if r["verdict"].passed),
        "total": len(rows),
        "chars": sorted({r["chars"] for r in rows}),
    }


def compare(name: str, base, inj, injected: bool = True) -> None:
    print(f"\n{SEPARATOR}\n对照：{name}\n{SEPARATOR}")
    b, j = summarize(base), summarize(inj)

    print(f"无注入  {b['passed']}/{b['total']} 通过  工具序列 {b['seq_list']}")
    print(f"注入    {j['passed']}/{j['total']} 通过  工具序列 {j['seq_list']}")
    print(f"注入字数 无注入 {b['chars']} / 注入 {j['chars']}")

    if set(j["seqs"]) == set(b["seqs"]):
        print("结论  工具序列没有变化")
    else:
        only_inj = set(j["seqs"]) - set(b["seqs"])
        only_base = set(b["seqs"]) - set(j["seqs"])
        print("结论  工具序列发生变化")
        if only_inj:
            print(f"      只在注入组出现: {sorted(only_inj)}")
        if only_base:
            print(f"      只在无注入组出现: {sorted(only_base)}")

    if b["total"] and j["total"]:
        if not injected:
            # 两组传的都是空串，唯一变量没被切换：通过数差异只可能是抖动。
            # 以前这里照样打「注入组通过数更低（经验有害，或抖动）」，
            # 结果 0 字注入的一次 3/3 vs 2/3 被读成"经验有害"，属于把噪声当因果。
            # 这种运行只能当抖动基准用，不能当经验效果对照写进报告。
            print("判定  两组都未注入（0 字），条件相同，通过数差异即抖动，不构成经验效果对照")
        elif b["passed"] == b["total"] and j["passed"] == j["total"]:
            print("判定  两组都全过，这道题没有提升空间，只能看工具序列")
        elif j["passed"] > b["passed"]:
            print("判定  注入组通过数更高")
        elif j["passed"] < b["passed"]:
            print("判定  注入组通过数更低（经验有害，或抖动）")
        else:
            print("判定  通过数相同")

    if injected and all(c == 0 for c in j["chars"]):
        print("警告  注入组 meta.experience_chars 全是 0：文本没进 Agent，先查该 Agent 是否把经验写进 meta")


def main(argv=None) -> int:
    args = parse_args(argv)

    goldens = [Path(g) for g in (args.golden or DEFAULT_GOLDENS)]
    missing = [str(g) for g in goldens if not g.exists()]
    if missing:
        print(f"[错误] 找不到任务文件：{', '.join(missing)}")
        return 1

    tasks = {t.id: t for t in load_many(goldens)}
    picked = []
    for tid in args.task:
        task = tasks.get(tid)
        if task is None:
            print(f"[错误] 没有这个任务：{tid}（可选示例：{', '.join(sorted(tasks)[:8])} …）")
            return 1
        picked.append(task)

    # 去掉 argparse 的 choices 之后，名字写错不再由 argparse 拦住，
    # 这里补回同样干净的报错（否则会掉一整屏 traceback）。
    # 顺序上先解析名字再探活：名字错了不该被"服务没起"的消息盖住。
    try:
        agent = builtin.make(args.agent, max_steps=args.max_steps)
    except UnknownAgentError as exc:
        print(f"[错误] {exc.args[0] if exc.args else exc}")
        return 2

    check_server()

    print(f"Agent {agent.name} | 请求名 {args.agent} | 步数上限 {args.max_steps}")
    print(f"每组重复 {args.repeat} 次 | 注入文本 {len(args.experience)} 字")
    if args.experience.strip():
        print(f"注入内容: {one_line(args.experience, 160)}")
    else:
        print("注入内容: （空）——两组条件相同，用来量抖动幅度")

    for task in picked:
        print(f"\n{SEPARATOR}\n任务 {task.id} | {task.turn_count()} 轮 | {one_line(task.input, 80)}\n{SEPARATOR}")
        base = run_group(agent, task, experience="", repeat=args.repeat, label="A", reset=not args.no_reset)
        inj = run_group(
            agent,
            task,
            experience=args.experience,
            repeat=args.repeat,
            label="B",
            reset=not args.no_reset,
        )
        compare(task.id, base, inj, injected=bool(args.experience.strip()))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
