"""从已有 SFT / 失败样本构造偏好对（chosen vs rejected）。

默认输入（存在几个算几个）：
    PASS 侧：data/distill/sft.jsonl、data/distill-main-a2/sft.jsonl
    FAIL 侧：data/distill/failed.jsonl、data/distill-main-a2/failed.jsonl

配对规则见 pipeline/pref.py：同 task_id + 同 agent + 同 split + 用户轮逐字一致。

输出：
    data/pref/pairs.jsonl    偏好对（prompt / chosen / rejected / failure_type / 来源）
    data/pref/report.json    计数、按失败类型分层、跳过原因

用法：
    .venv\\Scripts\\python.exe scripts\\build_pref.py
    .venv\\Scripts\\python.exe scripts\\build_pref.py --out data\\pref-probe
    .venv\\Scripts\\python.exe scripts\\build_pref.py --allow-cross-agent
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import pref as P
from pipeline import provenance, sft

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PASS_INPUTS = (
    PROJECT_ROOT / "data" / "distill" / "sft.jsonl",
    PROJECT_ROOT / "data" / "distill-main-a2" / "sft.jsonl",
)

FAIL_INPUTS = (
    PROJECT_ROOT / "data" / "distill" / "failed.jsonl",
    PROJECT_ROOT / "data" / "distill-main-a2" / "failed.jsonl",
)

DEFAULT_OUT = PROJECT_ROOT / "data" / "pref"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_side(paths: tuple[Path, ...], outcome: str, inputs_meta: list[dict]) -> list[P.Candidate]:
    out: list[P.Candidate] = []
    for path in paths:
        if not path.is_file():
            print(f"  跳过（不存在）{provenance.rel_path(path)}")
            continue
        rows = sft.read_jsonl(path)
        inputs_meta.append({**provenance.file_entry(path), "rows": len(rows), "side": outcome})
        rel = provenance.rel_path(path)
        for raw, lineno in rows:
            out.append(
                P.Candidate(
                    task_id=raw.get("task_id") or "",
                    agent=raw.get("agent") or "",
                    split=raw.get("split") or "",
                    messages=raw.get("messages") or [],
                    source={"file": rel, "line": lineno},
                    outcome=outcome,
                    failure_type=str(raw.get("failure_type") or ""),
                    source_trace=raw.get("source_trace"),
                )
            )
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="从已有轨迹构造偏好对（chosen / rejected）")
    p.add_argument("--out", default=str(DEFAULT_OUT), help="输出目录")
    p.add_argument(
        "--allow-cross-agent",
        action="store_true",
        help="允许跨 Agent 配对（默认禁止：配置差异不等于模型行为差异）",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    inputs_meta: list[dict] = []

    passes = load_side(PASS_INPUTS, P.PASS, inputs_meta)
    fails = load_side(FAIL_INPUTS, P.FAIL, inputs_meta)

    if not passes or not fails:
        print(f"[错误] 缺样本：PASS {len(passes)} 条 / FAIL {len(fails)} 条，两侧都要有才能配对")
        return 1

    result = P.build(passes, fails, allow_cross_agent=args.allow_cross_agent)
    pairs = result["pairs"]
    report = result["report"]
    counts = report["counts"]

    out_dir = Path(args.out)
    write_jsonl(out_dir / "pairs.jsonl", pairs)
    payload = {
        "schema_version": P.SCHEMA_VERSION,
        "generated_at": provenance.now_iso(),
        "inputs": inputs_meta,
        **report,
    }
    (out_dir / "report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"PASS 样本 {counts['pass_samples']} 条 / FAIL 样本 {counts['fail_samples']} 条")
    print(f"两侧都有的 task {counts['tasks_with_both']} 个 → 配对 {counts['pairs']} 对")
    if counts["cross_agent_pairs"]:
        print(f"  其中跨 Agent {counts['cross_agent_pairs']} 对（--allow-cross-agent 显式开启）")
    if counts["unverified_agent_pairs"]:
        print(
            f"  其中 {counts['unverified_agent_pairs']} 对未验证 agent"
            "（历史 failed 产物没有该字段；distill.py 现在会写，旧文件未重生成）"
        )
    if report["by_failure_type"]:
        print("按失败类型：" + "  ".join(f"{k}={v}" for k, v in report["by_failure_type"].items()))
    if report["skipped"]:
        print("跳过：" + "  ".join(f"{k}={v}" for k, v in report["skipped"].items()))

    print(f"\n输出：{provenance.rel_path(out_dir / 'pairs.jsonl')}（{len(pairs)} 对）")
    print(f"      {provenance.rel_path(out_dir / 'report.json')}")
    if not pairs:
        print("\n一对都没配上。原因看上面『跳过』那一行，常见是只有单侧样本或用户轮不一致。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
