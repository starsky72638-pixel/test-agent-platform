"""P5-c 一次性度量：产物统计、Workflow 轨迹蒸馏检查、Memory 检索统计。

三个子命令，互相独立：

    stats        经验库与 SFT 文件的实际数量、分布、异常样本
    wf-distill   把 runs/*.jsonl 里的 Workflow 轨迹喂进 distill，并逐条检查产物内容
    memory       用真实经验库对真实任务集跑检索，统计命中；可选跑一次带注入的执行

只读原则：
    - 不改任何经验库、评测集、已有 SFT 文件；
    - 新产物一律写到 data/distill-wf/、runs/p5c/、reports/p5c-*.json，
      与原始数据分开，删掉不影响任何既有实验。

为什么单独一个脚本而不是塞进 run_shop/run_loop：
    这两个脚本的职责是"跑 Agent"，不是"数产物"。把统计混进去会让
    "跑一次实验"顺手改统计口径，下次对不上就分不清是数据变了还是代码变了。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PROJECT_ROOT
from core.task import load_many
from core.trace import KIND_LLM_CALL, KIND_MEMORY_CALL, save_traces
from memory.retrieve import retrieve
from memory.schema import STATUS_ACTIVE
from memory.store import ExperienceStore

REPORT_DIR = PROJECT_ROOT / "reports"
EXP_GLOB = "runs/**/experiences.jsonl"

SFT_FILES = (
    Path("data/distill/sft.jsonl"),
    Path("data/distill-main-a2/sft.jsonl"),
    Path("data/train_sft.jsonl"),
)

TASK_SETS = {
    "shop_eval": Path("data/shop_eval.jsonl"),
    "shop_multi": Path("data/shop_eval_multi.jsonl"),
    "shop_hard": Path("data/shop_eval_hard.jsonl"),
    "customer": Path("data/customer_eval.jsonl"),
    "wf": Path("data/wf_eval.jsonl"),
}

WF_TRACES = (
    "runs/wf-branch-demo-20261120-210214.jsonl",
    "runs/wf-canvas-demo-20261120-210241.jsonl",
)


# ---------------------------------------------------------------- 工具函数


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(name: str, payload: dict) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / name
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


def exp_stores() -> list[tuple[Path, ExperienceStore]]:
    out: list[tuple[Path, ExperienceStore]] = []
    for path in sorted(PROJECT_ROOT.glob(EXP_GLOB)):
        out.append((path, ExperienceStore(path)))
    return out


# ---------------------------------------------------------------- stats


def scan_experiences() -> dict:
    files: list[dict] = []
    all_ids: set[str] = set()

    for path, store in exp_stores():
        items = store.all()
        all_ids.update(e.exp_id for e in items)
        files.append(
            {
                "path": rel(path),
                "records": count_lines(path),
                "unique": len(items),
                "active": sum(
                    1 for e in items if e.status == STATUS_ACTIVE and not e.is_expired()
                ),
                "retired": sum(1 for e in items if e.status != STATUS_ACTIVE),
                "expired": sum(1 for e in items if e.is_expired()),
                "failure_types": dict(
                    sorted(Counter(e.failure_type for e in items).items())
                ),
                "shapes": dict(sorted(Counter(e.trigger.shape for e in items).items())),
                "support": {e.exp_id: e.support for e in items},
                "contradict": {e.exp_id: e.contradict for e in items},
            }
        )

    def total(key: str) -> int:
        return sum(f[key] for f in files)

    failure_types: Counter = Counter()
    for f in files:
        failure_types.update(f["failure_types"])

    return {
        "files": len(files),
        "records": total("records"),
        "unique_per_file": total("unique"),
        "unique_global": len(all_ids),
        "active": total("active"),
        "retired": total("retired"),
        "expired": total("expired"),
        "failure_types": dict(sorted(failure_types.items())),
        "detail": files,
    }


def _sft_task_key(row: dict) -> str:
    return str(
        row.get("task_id")
        or (row.get("meta") or {}).get("case_id")
        or row.get("id")
        or ""
    )


def _has_tool_calls(row: dict) -> bool:
    return any(m.get("tool_calls") for m in (row.get("messages") or []))


def _empty_reason(row: dict) -> str:
    msgs = row.get("messages") or []
    if not msgs:
        return "no_messages"
    if not any(
        m.get("role") == "assistant" and str(m.get("content") or "").strip()
        for m in msgs
    ):
        return "no_assistant_answer"
    return ""


def scan_sft() -> dict:
    files: list[dict] = []
    for relpath in SFT_FILES:
        path = PROJECT_ROOT / relpath
        rows = load_jsonl(path)
        empties = [r for r in rows if _empty_reason(r)]
        files.append(
            {
                "path": rel(relpath),
                "exists": path.exists(),
                "samples": len(rows),
                "agents": dict(
                    sorted(Counter(str(r.get("agent") or "unknown") for r in rows).items())
                ),
                "tasks": len({_sft_task_key(r) for r in rows if _sft_task_key(r)}),
                "splits": dict(
                    sorted(Counter(str(r.get("split") or "unknown") for r in rows).items())
                ),
                "with_tool_calls": sum(1 for r in rows if _has_tool_calls(r)),
                "empty_or_invalid": len(empties),
                "empty_reasons": dict(
                    sorted(Counter(_empty_reason(r) for r in empties).items())
                ),
            }
        )
    return {"samples": sum(f["samples"] for f in files), "files": files}


def print_stats(exp: dict, sft: dict) -> None:
    print("=" * 78)
    print("一、经验库")
    print("-" * 78)
    print(
        f"文件 {exp['files']} 个 | 记录 {exp['records']} 行 | "
        f"唯一 exp_id（全局去重）{exp['unique_global']} | "
        f"各库唯一数合计 {exp['unique_per_file']}"
    )
    print(
        f"active {exp['active']} / retired {exp['retired']} / expired {exp['expired']}"
    )
    print("failure_type：" + (json.dumps(exp["failure_types"], ensure_ascii=False) if exp["failure_types"] else "无"))
    print()
    header = f"{'库':<52}{'行':>4}{'唯一':>5}{'active':>7}  类型"
    print(header)
    for f in exp["detail"]:
        types = ",".join(f"{k}:{v}" for k, v in f["failure_types"].items()) or "-"
        print(f"{f['path']:<52}{f['records']:>4}{f['unique']:>5}{f['active']:>7}  {types}")

    print()
    print("=" * 78)
    print("二、SFT 文件")
    print("-" * 78)
    print(f"合计样本 {sft['samples']} 条")
    for f in sft["files"]:
        print()
        print(f"{f['path']}  （存在={f['exists']}）")
        print(f"  样本 {f['samples']} | 来源任务 {f['tasks']} | 含 tool_calls {f['with_tool_calls']}")
        print(f"  来源 agent：{json.dumps(f['agents'], ensure_ascii=False)}")
        print(f"  split：{json.dumps(f['splits'], ensure_ascii=False)}")
        print(
            f"  空或无效：{f['empty_or_invalid']}"
            + (f" {json.dumps(f['empty_reasons'], ensure_ascii=False)}" if f['empty_or_invalid'] else "")
        )


def cmd_stats(args) -> int:
    exp = scan_experiences()
    sft = scan_sft()
    print_stats(exp, sft)
    p1 = write_json("p5c-experiences.json", exp)
    p2 = write_json("p5c-sft.json", sft)
    print()
    print(f"已写出：{rel(p1)}、{rel(p2)}")
    return 0


# ---------------------------------------------------------------- wf-distill


def _check_sample(row: dict) -> list[str]:
    """逐条检查一条 SFT 样本的结构问题，返回问题列表（空表示通过）。"""
    problems: list[str] = []
    msgs = row.get("messages") or []
    if not msgs:
        return ["messages 为空"]
    if msgs[0].get("role") != "user":
        problems.append("首条不是 user")

    pending: list[str] = []
    for i, m in enumerate(msgs):
        role = m.get("role")
        if role == "assistant":
            for tc in m.get("tool_calls") or []:
                fn = (tc or {}).get("function") or {}
                name = str(fn.get("name") or "")
                raw = fn.get("arguments")
                if not name:
                    problems.append(f"#{i} tool_call 缺 name")
                try:
                    json.loads(raw if isinstance(raw, str) else "null")
                except (TypeError, ValueError):
                    problems.append(f"#{i} tool_call arguments 不是合法 JSON")
                pending.append(name)
            if not (m.get("tool_calls") or []) and not str(m.get("content") or "").strip():
                problems.append(f"#{i} assistant 既无 tool_calls 也无内容")
        elif role == "tool":
            if not pending:
                problems.append(f"#{i} tool 消息前面没有对应的 tool_call")
            else:
                expected = pending.pop(0)
                if str(m.get("name") or "") != expected:
                    problems.append(
                        f"#{i} tool.name={m.get('name')!r} 与调用 {expected!r} 不匹配"
                    )
            if not str(m.get("content") or "").strip():
                problems.append(f"#{i} tool 消息内容为空")
        elif role == "user" and i > 0:
            if pending:
                problems.append(f"#{i} 上一轮还有 {len(pending)} 个 tool_call 未配对")

    if pending:
        problems.append(f"结尾有 {len(pending)} 个 tool_call 没有 tool 返回")
    if msgs[-1].get("role") != "assistant":
        problems.append("末条不是 assistant")
    elif not str(msgs[-1].get("content") or "").strip():
        problems.append("末条 assistant 内容为空")
    return problems


def cmd_wf_distill(args) -> int:
    from scripts.distill import main as distill_main

    out_dir = PROJECT_ROOT / "data" / "distill-wf"
    argv: list[str] = []
    for item in WF_TRACES:
        path = PROJECT_ROOT / item
        if path.exists():
            argv += ["--trace", item]
        else:
            print(f"[警告] 轨迹不存在，跳过：{item}")
    if not argv:
        print("[错误] 一条 Workflow 轨迹都没找到，先跑 run_shop 生成")
        return 1

    print("=" * 78)
    print("三、Workflow 轨迹 → Distill")
    print("-" * 78)
    print("输入轨迹：" + "、".join(argv[1::2]))
    rc = distill_main(
        argv + ["--golden", "data/wf_eval.jsonl", "--out", str(out_dir)]
    )
    if rc != 0:
        return rc

    sft_rows = load_jsonl(out_dir / "sft.jsonl")
    failed_rows = load_jsonl(out_dir / "failed.jsonl")
    print()
    print(f"产物：{rel(out_dir)}/sft.jsonl {len(sft_rows)} 条、failed.jsonl {len(failed_rows)} 条")

    checked: list[dict] = []
    for row in sft_rows:
        problems = _check_sample(row)
        checked.append(
            {
                "id": row.get("id"),
                "task_id": row.get("task_id"),
                "agent": row.get("agent"),
                "tools": row.get("tools"),
                "messages": len(row.get("messages") or []),
                "problems": problems,
            }
        )

    sigs = [row.get("id") for row in sft_rows]
    dupes = len(sigs) - len(set(sigs))
    bad = [c for c in checked if c["problems"]]

    print()
    print("逐条内容检查：")
    for c in checked:
        flag = "OK" if not c["problems"] else "问题: " + "; ".join(c["problems"])
        print(f"  {c['task_id']:<6} agent={c['agent']:<16} msgs={c['messages']:<3} tools={c['tools']}  {flag}")
    print()
    print(f"重复样本 {dupes} 条 | 结构问题 {len(bad)} 条")
    if failed_rows:
        print("失败样本：" + "、".join(
            f"{r.get('task_id')}({r.get('failure_type')})" for r in failed_rows
        ))

    payload = {
        "traces": argv[1::2],
        "sft": len(sft_rows),
        "failed": len(failed_rows),
        "duplicates": dupes,
        "problem_samples": len(bad),
        "checked": checked,
        "failed_detail": [
            {"task_id": r.get("task_id"), "failure_type": r.get("failure_type"), "fail_turn": r.get("fail_turn")}
            for r in failed_rows
        ],
    }
    print(f"\n已写出：{rel(write_json('p5c-wf-distill.json', payload))}")
    return 0


# ---------------------------------------------------------------- memory


def cmd_memory(args) -> int:
    stores = [(p, s) for p, s in exp_stores() if len(s)]
    task_sets = {
        # load_many 收的是路径列表（内部 for path in paths），单文件要包成 [..]
        name: load_many([PROJECT_ROOT / rel_path])
        for name, rel_path in TASK_SETS.items()
        if (PROJECT_ROOT / rel_path).exists()
    }

    rows: list[dict] = []
    for path, store in stores:
        for name, tasks in task_sets.items():
            hit_tasks = 0
            hit_total = 0
            detail: list[dict] = []
            for task in tasks:
                kwargs = {} if args.min_score is None else {"min_score": args.min_score}
                hits = retrieve(task, store.active(), top_k=args.top_k, **kwargs)
                if hits:
                    hit_tasks += 1
                    hit_total += len(hits)
                detail.append(
                    {"task": task.id, "hits": [h.exp.exp_id for h in hits]}
                )
            rows.append(
                {
                    "store": rel(path),
                    "store_size": len(store),
                    "task_set": name,
                    "tasks": len(tasks),
                    "hit_tasks": hit_tasks,
                    "hit_total": hit_total,
                    "avg_per_task": round(hit_total / len(tasks), 2) if tasks else 0.0,
                    "detail": detail,
                }
            )

    print("=" * 78)
    print("四、Memory 检索统计")
    print("-" * 78)
    print(f"经验库 {len(stores)} 个 | 任务集 {len(task_sets)} 个 | top_k={args.top_k}")
    print()
    current = ""
    for r in rows:
        if r["store"] != current:
            current = r["store"]
            print(f"[库] {current}（{r['store_size']} 条）")
        print(
            f"  {r['task_set']:<12} 任务 {r['tasks']:>3} | 命中任务 {r['hit_tasks']:>3} | "
            f"命中经验 {r['hit_total']:>3} | 平均 {r['avg_per_task']}"
        )

    totals = {
        "pairs": len(rows),
        "hit_pairs": sum(1 for r in rows if r["hit_tasks"]),
        "max_avg": max((r["avg_per_task"] for r in rows), default=0.0),
    }
    print()
    print(
        f"共 {totals['pairs']} 个「库×任务集」组合，其中 {totals['hit_pairs']} 个有命中；"
        f"单组合最高平均命中 {totals['max_avg']}"
    )

    proof = None
    if args.inject_run:
        store_arg = Path(args.store) if args.store else _best_store(rows, stores)
        if store_arg is not None and not store_arg.is_absolute():
            store_arg = PROJECT_ROOT / store_arg
        proof = inject_proof(store_arg)

    payload = {"rows": rows, "totals": totals, "inject_proof": proof,
               "top_k": args.top_k, "min_score": args.min_score}
    print(f"\n已写出：{rel(write_json('p5c-memory.json', payload))}")
    return 0


def _best_store(rows: list[dict], stores) -> Path | None:
    """命中最多的那个库；没有命中就不注入，免得跑一次空的还当成证据。"""
    best = max(rows, key=lambda r: r["hit_total"], default=None)
    if not best or best["hit_total"] == 0:
        return None
    for path, _store in stores:
        if rel(path) == best["store"]:
            return path
    return None


def inject_proof(store_path: Path | None) -> dict:
    """真跑一次 Workflow，检查经验文本是否出现在 llm_call 的 prompt 里。

    这是"命中 ≠ 用上"的唯一硬证据：只看 memory_call 的 hit_count 不够，
    必须能在下游 llm_call 的 args.prompt 里找到同一段经验文本。
    """
    if store_path is None:
        print("\n[跳过注入验证] 所有组合命中为 0，没有可注入的经验")
        return {"skipped": "no_hits"}

    import agents.builtin as builtin
    from agentplatform.adapter import WorkflowAgent
    from agentplatform.definition import WorkflowDefinition
    from core.task import Task

    out_dir = PROJECT_ROOT / "runs" / "p5c"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"inject-{stamp}.jsonl"

    try:
        store = ExperienceStore(store_path)
        defn = WorkflowDefinition.load(PROJECT_ROOT / "data" / "workflows" / "branch-demo.json")
        base = builtin.make("shop-cold")
        agent = WorkflowAgent(defn, tools=base.tools, memory_store=store)
        trace = agent.run(Task(id="W01", input="订单 1002 现在是什么状态，金额多少"))
        save_traces([trace], out_path)
    except Exception as exc:  # 模型/服务没起也算一种结果，如实记，不伪装成 0 命中
        print(f"\n[注入验证失败] {type(exc).__name__}: {exc}")
        print(traceback.format_exc(limit=2))
        return {"error": f"{type(exc).__name__}: {exc}", "store": rel(store_path)}

    mem_events = [e for e in trace.events if e.kind == KIND_MEMORY_CALL]
    llm_events = [e for e in trace.events if e.kind == KIND_LLM_CALL]
    hit_ids = mem_events[0].args.get("hits", []) if mem_events else []
    actions = {}
    for exp_id in hit_ids:
        for exp in store.all():
            if exp.exp_id == exp_id:
                actions[exp_id] = exp.text()
                break

    prompts = [str(e.args.get("prompt") or "") for e in llm_events]
    joined = "\n".join(prompts)
    in_prompt = {exp_id: (text[:24] in joined) for exp_id, text in actions.items()}

    print()
    print("[注入验证] 轨迹：" + rel(out_path))
    print(f"  memory_call 事件 {len(mem_events)} 条 | 命中 {len(hit_ids)} 条：{hit_ids}")
    print(f"  llm_call 事件 {len(llm_events)} 条 | prompt 总字数 {len(joined)}")
    print(f"  含【历史经验】表头：{'【历史经验' in joined}")
    for exp_id, ok in in_prompt.items():
        print(f"  经验 {exp_id} 文本进入 prompt：{ok}")

    return {
        "trace": rel(out_path),
        "store": rel(store_path),
        "hit_count": len(hit_ids),
        "hits": hit_ids,
        "llm_calls": len(llm_events),
        "prompt_chars": len(joined),
        "prompt_has_header": "【历史经验" in joined,
        "action_in_prompt": in_prompt,
    }


# ---------------------------------------------------------------- 入口


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="P5-c 闭环贯通与产物验证")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("stats", help="统计经验库与 SFT 产物")

    sub.add_parser("wf-distill", help="Workflow 轨迹进 distill 并检查产物")

    mem = sub.add_parser("memory", help="Memory 检索统计")
    mem.add_argument("--top-k", type=int, default=6)
    mem.add_argument("--min-score", type=float, default=None)
    mem.add_argument("--store", default=None, help="指定注入验证用的经验库路径")
    mem.add_argument(
        "--inject-run",
        action="store_true",
        help="真跑一次 Workflow 并检查经验是否进入 prompt（需要 mock_server 与模型 key）",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.cmd == "stats":
        return cmd_stats(args)
    if args.cmd == "wf-distill":
        return cmd_wf_distill(args)
    if args.cmd == "memory":
        return cmd_memory(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
