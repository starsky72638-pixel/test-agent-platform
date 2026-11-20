"""真跑入口：把一个真 Agent 放进 Runtime 跑一遍，并和历史报告对口径。

这个脚本不进 pytest，因为它要连 mock_server、要花模型额度、
同一道题两次跑结果可能不同。它是一次"人工验收"用的工具，
不是 CI 里的断言。

用法（两个窗口）：

    窗口 A  .venv\\Scripts\\python.exe mock_server.py
    窗口 B  .venv\\Scripts\\python.exe scripts\\run_shop.py --agent cold --split holdout --limit 3

常用组合：

    --agent tuned              用现有 11 条规则跑（历史基线）
    --agent cold               只用 3 条底线规则跑（对比起点）
    --split holdout            只跑没参与过经验提炼的题
    --golden data/shop_eval.jsonl   只跑某一份数据，可重复传

它做四件事：
    1. 逐题跑，每题打一行（状态 / 工具序列 / 判定）；
    2. 用 evaluator 判分并汇总；
    3. 把轨迹写进 runs/（jsonl，可被后续经验提炼直接读取）；
    4. 按 id 与 reports/ 下同名历史报告比 passed / failure_type / fail_turn。

第 4 步是"口径抽查"，不是断言：
    历史报告是旧测评脚本跑出来的，模型输出与服务端状态都可能变过，
    不一致不一定说明新判定错了。判定本身有没有退化，由
    tests/test_evaluator.py 里那 42 条历史轨迹对拍负责（那个不调 API）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# 允许直接运行：python scripts/run_shop.py
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agents.builtin as builtin
from config import PROJECT_ROOT
from core.runner import run_tasks
from core.registry import UnknownAgentError
from core.task import load_many
from core.trace import save_traces
from evaluator.verdict import evaluate

DEFAULT_GOLDENS = (
    PROJECT_ROOT / "data" / "shop_eval.jsonl",
    PROJECT_ROOT / "data" / "shop_eval_multi.jsonl",
    PROJECT_ROOT / "data" / "shop_eval_hard.jsonl",
)

RUNS_DIR = PROJECT_ROOT / "runs"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="把真 Agent 放进 Runtime 跑一遍")
    p.add_argument("--agent", default="tuned")
    p.add_argument("--split", default="all", choices=("all", "fit", "holdout"))
    p.add_argument("--limit", type=int, default=0, help="只跑前 N 条，0 表示不限")
    p.add_argument("--max-steps", type=int, default=6)
    p.add_argument(
        "--golden",
        action="append",
        default=None,
        help="任务文件，可重复传；默认三份全读",
    )
    p.add_argument("--out", default=None, help="轨迹输出目录，默认 runs/")
    p.add_argument("--no-compare", action="store_true", help="跳过与历史报告的对比")
    return p.parse_args(argv)


def check_server() -> None:
    """先确认被测服务活着。

    不做这一步的话，17 道题会全部跑成 infra_error，
    看起来像"Agent 全面失败"，实际只是忘了开 mock_server。
    """
    import httpx

    from config import SERVICE_BASE

    try:
        httpx.get(f"{SERVICE_BASE}/products/search", timeout=3.0)
    except httpx.HTTPError as exc:
        raise SystemExit(
            f"[错误] 连不上被测服务 {SERVICE_BASE}（{type(exc).__name__}）\n"
            f"       先另开一个窗口执行：.venv\\Scripts\\python.exe mock_server.py"
        ) from exc


def reset_mock() -> None:
    """清空退款记录，保证每次开跑的初始状态一致。

    从 check_server 拆出来：探活回答"服务在不在"，reset 是实验初始条件，
    两者的失败原因不同，混在一起时提示只能挑一个说。
    """
    import httpx

    from config import SERVICE_BASE

    try:
        httpx.post(f"{SERVICE_BASE}/admin/reset", timeout=3.0)
    except httpx.HTTPError as exc:
        raise SystemExit(
            f"[错误] 无法重置被测服务 {SERVICE_BASE}/admin/reset（{type(exc).__name__}）\n"
            f"       服务可能起来了但接口异常，看 mock_server 窗口的输出确认"
        ) from exc


def load_old_results(golden_paths) -> dict[str, dict]:
    """读历史报告，返回 {task_id: 旧记录}。

    报告名与数据文件同名（shop_eval.jsonl -> reports/shop_eval.json）。
    """
    old: dict[str, dict] = {}
    for gp in golden_paths:
        rp = PROJECT_ROOT / "reports" / (Path(gp).stem + ".json")
        if not rp.exists():
            continue
        try:
            data = json.loads(rp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for row in data.get("results") or []:
            if row.get("id"):
                old[row["id"]] = row
    return old


def compare(verdicts: list, old: dict[str, dict]) -> None:
    """逐条比对本次判定与历史报告。只比 id 有交集的题。"""
    common = [v for v in verdicts if v.task_id in old]
    if not common:
        print("\n历史报告里没有可比对的题（可能这份数据还没跑过旧脚本）。")
        return

    same = 0
    diffs = []
    for v in common:
        o = old[v.task_id]
        o_ft = o.get("failure_type") or ""
        if bool(o.get("passed")) == bool(v.passed) and o_ft == v.failure_type:
            same += 1
        else:
            diffs.append((v.task_id, o.get("passed"), o_ft, v.passed, v.failure_type))

    print(f"\n{'=' * 74}")
    print(f"与历史报告比对（按 id 交集 {len(common)} 条）：一致 {same}，不一致 {len(diffs)}")
    for tid, o_pass, o_ft, n_pass, n_ft in diffs:
        print(
            f"  {tid}  旧 {'PASS' if o_pass else 'FAIL ' + (o_ft or '?')}"
            f"  ->  新 {'PASS' if n_pass else 'FAIL ' + (n_ft or '?')}"
        )
    if diffs:
        print(
            "  不一致不一定说明判定错了：模型输出和服务端状态都可能变过。\n"
            "  判定本身有没有退化，看 tests/test_evaluator.py 里那 42 条离线对拍。"
        )


def tool_names_of(agent) -> set[str]:
    """从 Agent 自己身上取工具名，不从某个具体实现里 import。

    这个集合是判定器的 allowed_tools。写死 shop 的工具表时，
    换成别的 Agent 就会把它的正确调用判成 invalid_tool。
    与 run_loop.py 同名函数刻意各存一份：两个入口互不依赖，删一个不影响另一个。
    """
    return {
        getattr(t, "name", "")
        for t in (getattr(agent, "tools", None) or [])
        if getattr(t, "name", "")
    }


def main(argv=None) -> int:
    args = parse_args(argv)

    goldens = [Path(g) for g in (args.golden or DEFAULT_GOLDENS)]
    for gp in goldens:
        if not gp.exists():
            print(f"[错误] 找不到任务文件：{gp}")
            return 1

    tasks = load_many(goldens)
    if args.split != "all":
        tasks = [t for t in tasks if t.split == args.split]
    if args.limit > 0:
        tasks = tasks[: args.limit]

    if not tasks:
        print("过滤之后没有任务可跑。")
        return 1

    # 去掉 argparse 的 choices 之后，名字写错不再由 argparse 拦住，
    # 这里补回同样干净的报错（否则会掉一整屏 traceback）。
    # 顺序上先解析名字再探活：名字错了不该被"服务没起"的消息盖住。
    try:
        agent = builtin.make(args.agent, max_steps=args.max_steps)
    except UnknownAgentError as exc:
        print(f"[错误] {exc.args[0] if exc.args else exc}")
        return 2
    allowed = tool_names_of(agent)

    check_server()
    reset_mock()

    print(
        f"Agent {agent.name} | 请求名 {args.agent} | 任务 {len(tasks)} 条 | "
        f"split {args.split} | 步数上限 {args.max_steps}"
    )
    print("-" * 74)

    by_id = {t.id: t for t in tasks}
    verdicts: list = []
    started = time.perf_counter()

    def on_trace(trace) -> None:
        verdict = evaluate(trace, by_id[trace.task_id], allowed_tools=allowed)
        verdicts.append(verdict)

        flag = "PASS" if verdict.passed else f"FAIL {verdict.failure_type}"
        turn = f"@t{verdict.fail_turn}" if verdict.fail_turn else ""
        names = [c.name for c in trace.all_calls()]
        print(
            f"[{len(verdicts):02d}/{len(tasks)}] {trace.task_id:<5} "
            f"{flag + turn:<24} {trace.elapsed_ms:>6}ms  "
            f"{trace.status}  tools={names}"
        )

    result = run_tasks(agent, tasks, max_steps=args.max_steps, on_trace=on_trace)
    elapsed = int(time.perf_counter() - started)

    # ---------- 汇总 ----------
    passed = sum(1 for v in verdicts if v.passed)
    infra = [v for v in verdicts if v.failure_type == "infra_error"]
    types: dict[str, int] = {}
    for v in verdicts:
        if not v.passed:
            types[v.failure_type] = types.get(v.failure_type, 0) + 1

    valid = len(verdicts) - len(infra)
    print("-" * 74)
    print(
        f"跑完 {len(verdicts)} 条，有效 {valid} 条（剔除链路故障 {len(infra)} 条），"
        f"耗时 {elapsed}s"
    )
    if valid:
        print(f"通过 {passed}/{valid} ({round(100 * passed / valid)}%)")
    if types:
        print("未通过归因：" + "  ".join(f"{k} {v}" for k, v in sorted(types.items())))
    print(f"轨迹：{result.summary()}")

    # ---------- 落盘 ----------
    if result.traces:
        out_dir = Path(args.out) if args.out else RUNS_DIR
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out_path = out_dir / f"{args.agent}-{stamp}.jsonl"
        save_traces(result.traces, out_path)
        print(f"轨迹已写入：{out_path}")

    # ---------- 与历史报告对口径 ----------
    if not args.no_compare:
        compare(verdicts, load_old_results(goldens))

    return 0


if __name__ == "__main__":
    sys.exit(main())
