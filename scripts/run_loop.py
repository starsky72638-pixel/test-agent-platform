"""经验闭环的真跑入口：学习 -> 入库 -> 验证 -> 出报告。

这个脚本不进 pytest。它要连 mock_server、要花模型额度、同一道题两次可能
结果不同。它是"人工验收"用的，不是 CI 里的断言。

用法（两个窗口）：

    窗口 A  .venv\\Scripts\\python.exe mock_server.py
    窗口 B  .venv\\Scripts\\python.exe scripts\\run_loop.py --agent cold

    先小样本试通管道：
    窗口 B  .venv\\Scripts\\python.exe scripts\\run_loop.py --agent cold --fit-limit 8 --holdout-limit 6

它做四件事：
    1. 跑 fit 侧、判分、把失败提炼成经验、过五道闸门后入库；
    2. 用同一批 holdout 跑两遍：一遍不带经验，一遍带检索到的经验；
    3. 打印逐题翻转表、归因变化、每条 holdout 注入了什么；
    4. 把本次实验落成一个自包含目录：

        runs/<时间戳>/
            fit.jsonl                学习阶段的轨迹（证据的原始来源）
            holdout-baseline.jsonl   基线组轨迹
            holdout-injected.jsonl   注入组轨迹
            experiences.jsonl        本次用的经验库（每次实验一份，不共用）
            report.json              全部账目

为什么每次实验一个新库：
    record_outcome 会累加反证，同一份库跑第三次时经验可能已被自动退休，
    结果无法复现。要复现或接续，用 --store 指向已有文件。

一个必须说清的边界：
    学习阶段只跑 fit；验证阶段只跑 holdout。两者不交叉。
    这条边由 pipeline.learn 内部的 split 过滤保证，不是靠调用方自觉。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# 允许直接运行：python scripts/run_loop.py
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agents.builtin as builtin
from config import PROJECT_ROOT
from core.task import load_many
from core.trace import save_traces
from memory.store import ExperienceStore
from pipeline import learn, verify

# 被测服务的地址。与 agents/*/tools 读同一个环境变量，
# 所以改地址只需要设一次 SHOP_BASE_URL，不需要改脚本。
SHOP_BASE = (os.getenv("SHOP_BASE_URL") or "http://127.0.0.1:8000").rstrip("/")

DEFAULT_GOLDENS = (
    PROJECT_ROOT / "data" / "shop_eval.jsonl",
    PROJECT_ROOT / "data" / "shop_eval_multi.jsonl",
    PROJECT_ROOT / "data" / "shop_eval_hard.jsonl",
)

RUNS_DIR = PROJECT_ROOT / "runs"
SEPARATOR = "-" * 78


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="跑一遍经验闭环：学习 -> 验证 -> 报告")
    p.add_argument(
        "--agent",
        default=builtin.DEFAULT_AGENT,
        help=f"Agent 名字，可选：{', '.join(builtin.names())}（默认 {builtin.DEFAULT_AGENT}）",
    )
    p.add_argument("--list-agents", action="store_true", help="列出可接入的 Agent 后退出")
    p.add_argument("--fit-limit", type=int, default=0, help="只跑前 N 条 fit，0 表示不限")
    p.add_argument("--holdout-limit", type=int, default=0, help="只跑前 N 条 holdout")
    p.add_argument("--max-steps", type=int, default=6)
    p.add_argument("--top-k", type=int, default=6, help="每条任务最多注入几条经验")
    p.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="检索阈值，不给则用 memory.retrieve 的默认值",
    )
    p.add_argument("--golden", action="append", default=None, help="任务文件，可重复传")
    p.add_argument("--store", default=None, help="复用已有经验库（不给则新建）")
    p.add_argument("--out", default=None, help="输出根目录，默认 runs/")
    p.add_argument("--no-record", action="store_true", help="不回填效果，不改动经验库")
    return p.parse_args(argv)


def check_server() -> None:
    """先确认被测服务活着，否则整批会跑成 infra_error，看起来像能力全面崩盘。"""
    import httpx

    try:
        httpx.get(f"{SHOP_BASE}/products/search", timeout=3.0)
        httpx.post(f"{SHOP_BASE}/admin/reset", timeout=3.0)
    except httpx.HTTPError as exc:
        raise SystemExit(
            f"[错误] 连不上被测服务 {SHOP_BASE}（{type(exc).__name__}）\n"
            f"       先另开一个窗口执行：.venv\\Scripts\\python.exe mock_server.py"
        ) from exc


def tool_names_of(agent) -> set[str]:
    """从 Agent 自己身上取工具名，不从某个具体实现里 import。

    这个集合用作经验闸门的一部分（禁止经验指向不存在的工具）。
    如果这里写死 shop 的工具名，接第二个 Agent 时闸门就在拿别人的工具表判它。
    """
    return {
        getattr(t, "name", "")
        for t in (getattr(agent, "tools", None) or [])
        if getattr(t, "name", "")
    }


def _flag(verdict) -> str:
    if verdict.passed:
        return "PASS"
    turn = f"@t{verdict.fail_turn}" if verdict.fail_turn else ""
    return f"FAIL {verdict.failure_type}{turn}"


def _tools(trace) -> str:
    return ",".join(c.name for c in trace.all_calls()) or "-"


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.list_agents:
        print("可接入的 Agent：")
        for name in builtin.names():
            print(f"  {name}")
        return 0

    goldens = [Path(g) for g in (args.golden or DEFAULT_GOLDENS)]
    for gp in goldens:
        if not gp.exists():
            print(f"[错误] 找不到任务文件：{gp}")
            return 1

    all_tasks = load_many(goldens)
    fit = [t for t in all_tasks if t.split == "fit"]
    holdout = [t for t in all_tasks if t.split == "holdout"]
    if args.fit_limit > 0:
        fit = fit[: args.fit_limit]
    if args.holdout_limit > 0:
        holdout = holdout[: args.holdout_limit]
    selected = fit + holdout

    if not fit:
        print("[错误] 没有 fit 侧任务可跑。")
        return 1
    if not holdout:
        print("[错误] 没有 holdout 侧任务可跑。")
        return 1

    check_server()

    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = (Path(args.out) if args.out else RUNS_DIR) / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    store_path = Path(args.store) if args.store else run_dir / "experiences.jsonl"
    store = ExperienceStore(store_path)
    existing = len(store.active())

    agent = builtin.make(args.agent, max_steps=args.max_steps)
    allowed = tool_names_of(agent)

    print(f"Agent {agent.name} | 请求名 {args.agent} | 步数上限 {args.max_steps}")
    print(f"fit {len(fit)} 条 / holdout {len(holdout)} 条 | 经验库 {store_path}")
    print(f"工具 {sorted(allowed) or '无'} | 上下文 {getattr(agent, 'keep_turns', '-')}")
    if existing:
        print(f"经验库已有 {existing} 条可用经验（本次会接续使用）")

    # ---------------- 学习 ----------------
    print(f"\n{SEPARATOR}\n学习阶段：只跑 fit 侧，不注入经验\n{SEPARATOR}")
    counter = {"n": 0}

    def on_fit(task, trace, verdict) -> None:
        counter["n"] += 1
        print(
            f"[{counter['n']:02d}/{len(fit)}] {task.id:<5} {_flag(verdict):<26} "
            f"{trace.elapsed_ms:>6}ms  {trace.status}  tools=[{_tools(trace)}]"
        )

    started = time.perf_counter()
    learned = learn(
        agent,
        selected,
        store,
        allowed_tools=allowed,
        on_judged=on_fit,
        max_steps=args.max_steps,
    )
    learn_sec = int(time.perf_counter() - started)

    print(SEPARATOR)
    for line in learned.lines():
        print(line)
    print(f"耗时 {learn_sec}s")

    if learned.errors:
        print("\n判定异常（未参与提炼）：")
        for err in learned.errors:
            print(f"  ! {err}")

    if learned.rejected:
        print("\n被闸门拒绝：")
        for report in learned.rejected:
            print(f"  x {report.exp_id}  {'; '.join(report.reasons)}")

    if learned.kept:
        print("\n入库经验：")
        for i, exp in enumerate(learned.kept, 1):
            print(f"  {i}. {exp.exp_id} [{exp.failure_type}] {exp.action[:70]}")
    else:
        print(
            "\n没有经验入库。可能原因：fit 侧全过（没有可学的）、失败都属于环境问题、"
            "或提炼出的动作没过闸门。"
        )

    save_traces(learned.traces, run_dir / "fit.jsonl")

    # ---------------- 验证 ----------------
    print(f"\n{SEPARATOR}\n验证阶段：同一批 holdout 跑两遍（基线 / 注入）\n{SEPARATOR}")

    side_counter = {"n": 0}

    def on_side(label, task, trace, verdict) -> None:
        side_counter["n"] += 1
        tag = "基线" if label == "baseline" else "注入"
        name = f"[{tag}]"
        print(
            f"[{side_counter['n']:02d}/{len(holdout) * 2}] {name:<6} {task.id:<5} "
            f"{_flag(verdict):<26} {trace.elapsed_ms:>6}ms  tools=[{_tools(trace)}]"
        )

    started = time.perf_counter()
    compare = verify(
        agent,
        selected,
        store,
        allowed_tools=allowed,
        on_judged=on_side,
        top_k=args.top_k,
        min_score=args.min_score,
        max_steps=args.max_steps,
        record=not args.no_record,
    )
    verify_sec = int(time.perf_counter() - started)

    base, inj = compare.baseline, compare.injected
    flips = compare.flips()
    delta = compare.delta()

    save_traces(base.traces, run_dir / "holdout-baseline.jsonl")
    save_traces(inj.traces, run_dir / "holdout-injected.jsonl")

    print(f"\n{SEPARATOR}\n对比\n{SEPARATOR}")
    print(f"基线   {base.passed}/{base.total}（{base.rate:.0%}）  {base.by_failure()}")
    print(f"注入   {inj.passed}/{inj.total}（{inj.rate:.0%}）  {inj.by_failure()}")
    print(f"差值   {delta['rate_delta']:+.0%}（{delta['injected_passed'] - delta['baseline_passed']:+d} 条）")
    # 链路故障（限流/连接/超时）不是能力问题，混进分母会让差值虚高或虚低。
    # 两套口径都打出来，报告里用"可判"那一套，别挑好看的那个。
    if base.infra_failed or inj.infra_failed:
        print(
            f"可判   基线 {base.passed}/{base.evaluable}（{base.rate_evaluable:.0%}）"
            f"  注入 {inj.passed}/{inj.evaluable}（{inj.rate_evaluable:.0%}）"
            f"  —— 已剔除链路故障 {base.infra_failed}/{inj.infra_failed} 条"
        )
    print(f"耗时 {verify_sec}s")

    if not compare.injected_tasks():
        print(
            "\n注意：没有任何 holdout 任务检索到经验。注入组等同于基线，"
            "这不是「经验无效」，而是「根本没注入」。"
        )

    print("\n翻转：")
    for label, ids in (
        ("FAIL -> PASS", flips["up"]),
        ("PASS -> FAIL", flips["down"]),
        ("一直通过", flips["same_pass"]),
        ("一直未过", flips["same_fail"]),
    ):
        print(f"  {label:<14} {len(ids):>2} 条  {', '.join(ids) or '-'}")

    print(f"\n检索覆盖：{compare.injected_tasks()}/{inj.total} 条任务注入了经验")
    counts = compare.hit_counts()
    if counts:
        print("每条经验被注入到几条任务：")
        for exp_id, n in counts.items():
            action = ""
            found = store.get(exp_id)
            if found is not None:
                action = f"  {found.action[:56]}"
            print(f"  {exp_id}  x{n}{action}")

    print("\n逐题注入明细：")
    for row in compare.coverage():
        ids = ", ".join(row["exp_ids"]) if row["injected"] else "（无）"
        print(f"  {row['task_id']:<5} {row['chars']:>4} 字  {ids}")

    if compare.retired:
        print(f"\n注意：本次有 {compare.retired} 条经验被自动退休（反证压过支持）。")

    # ---------------- 落盘 ----------------
    report = {
        "run": {
            "stamp": stamp,
            "agent": agent.name,
            "agent_requested": args.agent,
            "max_steps": args.max_steps,
            "top_k": args.top_k,
            "min_score": args.min_score,
            "fit_limit": args.fit_limit,
            "holdout_limit": args.holdout_limit,
            "fit_tasks": [t.id for t in fit],
            "holdout_tasks": [t.id for t in holdout],
            "store": str(store_path),
            "record_outcome": not args.no_record,
        },
        "learn": learned.to_dict(),
        "verify": compare.to_dict(),
        "store_after": store.stats(),
    }
    report_path = run_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n本次实验目录：{run_dir}")
    print(f"报告：{report_path}")
    print(f"经验库：{store_path}（{len(store.active())} 条可用）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
