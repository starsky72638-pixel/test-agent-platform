"""Golden Set 回放：把 Agent 的行为变成可比较的数字。

三层断言，对应后训练里的三层 reward：
  1 工具选择  该用哪个工具、有没有漏
  2 参数      关键参数取值对不对
  3 终态事实  该有的有、该没有的没有（must_not_have 专抓编造）

跑法（项目根目录，先另开窗口 python mock_server.py）：
    python -m test_agent.replay

产物 reports/shop_eval.json 里每条都带失败归因，
失败样本可直接拿去当 DPO 的负例。
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import time
from collections import Counter
from pathlib import Path

import httpx

from config import PROJECT_ROOT, build_llm, load_settings
from main_agent.core.agent import run_agent
from main_agent.tools import SHOP_SYSTEM_PROMPT, get_registry, register_default_tools

SHOP_BASE = "http://127.0.0.1:8000"
GOLDEN_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "data" / "shop_eval.jsonl"
REPORT_FILE = PROJECT_ROOT / "reports" / (GOLDEN_FILE.stem + ".json")


def load_cases(path: Path) -> list[dict]:
    cases: list[dict] = []
    for no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            cases.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name} 第 {no} 行不是合法 JSON：{exc}") from exc
    return cases


def probe_env() -> None:
    """被测环境连不上时直接退出，避免把环境故障算成模型失败。"""
    try:
        httpx.get(f"{SHOP_BASE}/products/search", timeout=2.0)
        httpx.post(f"{SHOP_BASE}/admin/reset", timeout=2.0)
    except httpx.HTTPError as exc:
        raise SystemExit(
            f"[错误] 连不上被测环境 {SHOP_BASE}（{type(exc).__name__}）\n"
            f"       先另开一个窗口执行：python mock_server.py"
        ) from exc


def same(a, b) -> bool:
    """参数比较：200 和 200.0 算相等，1002 和 '1002' 也算相等。"""
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return str(a) == str(b)


def score(case: dict, calls: list[dict], answer: str) -> dict:
    names = [c["name"] for c in calls]

    expect = case.get("expect_tools")
    if expect is None:
        tool_ok, tool_msg = True, "未校验"
    else:
        missing = [t for t in expect if t not in names]
        tool_ok = not missing
        tool_msg = f"缺 {','.join(missing)}" if missing else "命中"

    expect_args = case.get("expect_args") or {}
    if not expect_args:
        param_ok, param_msg = True, "未校验"
    else:
        bad = [
            f"{k}={v}" for k, v in expect_args.items()
            if not any(same(c["args"].get(k), v) for c in calls)
        ]
        param_ok = not bad
        param_msg = f"错 {','.join(bad)}" if bad else "命中"

    miss = [s for s in case.get("must_have", []) if s not in answer]
    any_miss = [g for g in case.get("must_have_any", []) if not any(s in answer for s in g)]
    forbid = [s for s in case.get("must_not_have", []) if s in answer]
    answer_ok = not (miss or any_miss or forbid)

    # 归因优先级：编造 > 漏工具 > 错参数 > 漏事实
    if tool_ok and param_ok and answer_ok:
        ftype = ""
    elif forbid:
        ftype = "hallucination"
    elif not tool_ok:
        ftype = "missing_tool"
    elif not param_ok:
        ftype = "wrong_arg"
    else:
        ftype = "wrong_fact"

    return {
        "tool_ok": tool_ok, "tool_msg": tool_msg,
        "param_ok": param_ok, "param_msg": param_msg,
        "answer_ok": answer_ok,
        "missing_fact": miss, "missing_any": [g[0] for g in any_miss], "forbidden_hit": forbid,
        "passed": ftype == "", "failure_type": ftype,
    }


def pct(results: list[dict], key: str) -> str:
    n = len(results)
    ok = sum(1 for r in results if r[key])
    return f"{ok}/{n} ({round(100 * ok / n)}%)" if n else "0/0"


def main() -> int:
    if not GOLDEN_FILE.exists():
        print(f"[错误] 找不到用例集：{GOLDEN_FILE}")
        return 1

    probe_env()
    settings = load_settings()
    llm = build_llm(settings, streaming=False).bind(temperature=0)
    register_default_tools()
    tools = get_registry().list()
    cases = load_cases(GOLDEN_FILE)

    print(f"Golden Set {len(cases)} 条 | 模型 {settings.model} | 温度 0")
    print("提醒：S13/S14 会改动服务端状态，每次回放前请重启 mock_server\n")

    results: list[dict] = []
    started = time.perf_counter()

    for i, case in enumerate(cases, 1):
        t0 = time.perf_counter()
        answer, messages, err = "", [], ""
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                answer, messages = run_agent(llm, tools, SHOP_SYSTEM_PROMPT, case["input"])
        except Exception as exc:  # 单条炸掉不能中断整批
            err = f"{type(exc).__name__}: {exc}"

        elapsed = int((time.perf_counter() - t0) * 1000)
        calls = [
            {"name": c["name"], "args": c["args"]}
            for m in messages for c in (getattr(m, "tool_calls", None) or [])
        ]
        row = {
            "id": case["id"], "input": case["input"], "note": case.get("note", ""),
            "answer": answer, "calls": calls, "elapsed_ms": elapsed,
            "msg_count": len(messages), "error": err,
            **score(case, calls, answer),
        }
        results.append(row)

        flag = "PASS" if row["passed"] else f"FAIL {row['failure_type']}"
        print(f"[{i:02d}/{len(cases)}] {row['id']} {flag:<22} {elapsed:>5}ms  "
              f"tools={[c['name'] for c in calls]}")

    total = int(time.perf_counter() - started)
    failures = [r for r in results if not r["passed"]]
    types = Counter(r["failure_type"] for r in failures)
    avg_ms = round(sum(r["elapsed_ms"] for r in results) / len(results))
    avg_calls = round(sum(len(r["calls"]) for r in results) / len(results), 1)

    print("\n" + "=" * 70)
    print(f"回放完毕：{len(results)} 条，总耗时 {total}s，平均 {avg_ms}ms/条，平均调用 {avg_calls} 次")
    print(f"通过      {pct(results, 'passed')}    工具选择 {pct(results, 'tool_ok')}")
    print(f"参数      {pct(results, 'param_ok')}    终态事实 {pct(results, 'answer_ok')}")
    print("=" * 70)

    if failures:
        print("\n失败明细：")
        for r in failures:
            print(f"  {r['id']} [{r['failure_type']}] {r['input']}")
            print(f"     工具：{[c['name'] for c in r['calls']]}  {r['tool_msg']} / {r['param_msg']}")
            print(f"     回答：{r['answer'][:110]}")
            if r["forbidden_hit"]:
                print(f"     编造命中：{r['forbidden_hit']}")
            if r["missing_fact"] or r["missing_any"]:
                print(f"     缺失事实：{r['missing_fact'] + r['missing_any']}")
        print("\n归因：" + "  ".join(f"{k} {v}" for k, v in types.most_common()))
    else:
        print("\n全部通过——看看是不是 golden 写得太松了，加两条边界再跑。")

    REPORT_FILE.parent.mkdir(exist_ok=True)
    REPORT_FILE.write_text(
        json.dumps(
            {
                "model": settings.model,
                "summary": {
                    "total": len(results),
                    "passed": len(results) - len(failures),
                    "avg_ms": avg_ms,
                    "avg_calls": avg_calls,
                    "failure_types": dict(types),
                },
                "results": results,
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n报告已写入：{REPORT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
