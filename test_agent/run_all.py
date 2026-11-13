"""一条命令跑完：读文档 → 生成用例 → 执行 → 出报告。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from config import PROJECT_ROOT, build_llm, load_settings
from test_agent.create.generator import generate_cases
from test_agent.create.runner import run_cases, summarize


BASE_DIR = Path(__file__).resolve().parent
SPEC_FILE = BASE_DIR / "spec" / "openapi.json"
REPORT_DIR = PROJECT_ROOT / "reports"
BASE_URL = "http://127.0.0.1:8000"


def main() -> int:
    if not SPEC_FILE.exists():
        print(f"[错误] 找不到文档：{SPEC_FILE}")
        return 1

    settings = load_settings()
    llm = build_llm(settings, streaming=False).bind(temperature=0)
    spec = json.loads(SPEC_FILE.read_text(encoding="utf-8"))

    print(f"[1/3] 读文档：{len(spec.get('paths', {}))} 个接口")

    cases, dropped = generate_cases(llm, spec)
    print(f"[2/3] 生成用例 {len(cases)} 条（格式非法丢弃 {dropped} 条）")
    for c in cases:
        print(f"      {c['id']} [{c['category']:<8}] {c['method']:<5} "
              f"{c['path']:<14} expect={c['expect_status']}  {c['title']}")

    print(f"[3/3] 执行：{BASE_URL}")
    try:
        results = run_cases(cases, BASE_URL)
    except Exception as exc:
        print(f"[错误] 执行失败：{type(exc).__name__}: {exc}")
        return 1

    s = summarize(results)
    print()
    print("=" * 62)
    print(f"通过 {s['passed']}/{s['total']}   通过率 {s['pass_rate']}%   "
          f"总耗时 {s['elapsed_ms']}ms   疑似产品缺陷 {s['suspect_defect']}")
    print("按类别：" + "  ".join(
        f"{k} {v['passed']}/{v['total']}" for k, v in sorted(s["by_category"].items())
    ))
    print("=" * 62)

    failures = [r for r in results if not r["passed"]]
    if failures:
        print("失败明细：")
        for r in failures:
            print(f"  {r['id']} [{r['category']}] {r['method']} {r['path']} "
                  f"query={r['query']} → {r['verdict']}  {r['detail']}")
            print(f"        响应：{r['response'][:120]}")
    else:
        print("全部通过——如果被测服务是带 bug 的那份，说明用例没打到点上。")

    if failures:
        groups: dict[tuple, list[dict]] = {}
        for r in failures:
            key = (r["method"], r["path"], r["expect_status"], r["actual_status"])
            groups.setdefault(key, []).append(r)
        print(f"\n失败 {len(failures)} 条 → 归并为 {len(groups)} 个缺陷")
        for i, (key, items) in enumerate(
            sorted(groups.items(), key=lambda kv: -len(kv[1])), 1
        ):
            method, path, exp, act = key
            ids = ", ".join(x["id"] for x in items)
            print(f"  D{i} {method} {path} 期望{exp} 实际{act}  "
                  f"命中{len(items)}条({ids})  {items[0]['verdict']}")

    REPORT_DIR.mkdir(exist_ok=True)
    out = REPORT_DIR / "report.json"
    out.write_text(
        json.dumps({"base_url": BASE_URL, "summary": s, "results": results},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n报告已写入：{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
