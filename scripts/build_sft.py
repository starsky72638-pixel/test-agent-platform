"""把已有的 SFT 文件过一遍质量闸门，产出可训练 / 评估 / 隔离三份数据。

它不生成样本、不训练模型、不改 messages —— 只做"这条该进哪一档"的判断，
并把判断依据写成机器可读的报告。

默认输入（存在几个算几个）：
    data/distill/sft.jsonl            一次实验产物
    data/distill-main-a2/sft.jsonl    另一次实验产物
    data/distill-wf/sft.jsonl         Workflow 轨迹产物
    data/train_sft.jsonl              人工整理，来源与上面三份不同

输出：
    data/sft/train.jsonl       可训练
    data/sft/holdout.jsonl     只评估
    data/sft/quarantine.jsonl  冲突 / 未知来源 / 结构异常，待人处理
    data/sft/report.json       计数、原因、冲突清单、输入哈希

用法：
    .venv\\Scripts\\python.exe scripts\\build_sft.py
    .venv\\Scripts\\python.exe scripts\\build_sft.py --in data\\distill\\sft.jsonl --out data\\sft-probe
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import provenance, sft

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUTS = (
    PROJECT_ROOT / "data" / "distill" / "sft.jsonl",
    PROJECT_ROOT / "data" / "distill-main-a2" / "sft.jsonl",
    PROJECT_ROOT / "data" / "distill-wf" / "sft.jsonl",
    PROJECT_ROOT / "data" / "train_sft.jsonl",
)

DEFAULT_OUT = PROJECT_ROOT / "data" / "sft"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="SFT 数据质量闸门：去重、冲突隔离、split 隔离")
    p.add_argument(
        "--in",
        dest="inputs",
        action="append",
        default=None,
        help="输入 SFT 文件，可重复；不给则用默认四份（存在几个算几个）",
    )
    p.add_argument("--out", default=str(DEFAULT_OUT), help="输出目录")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    paths = [Path(x) for x in args.inputs] if args.inputs else list(DEFAULT_INPUTS)

    items: list[tuple[dict, dict]] = []
    inputs_meta: list[dict] = []
    for path in paths:
        if not path.is_file():
            print(f"  跳过（不存在）{provenance.rel_path(path)}")
            continue
        rows = sft.read_jsonl(path)
        inputs_meta.append({**provenance.file_entry(path), "rows": len(rows)})
        rel = provenance.rel_path(path)
        for raw, lineno in rows:
            items.append((raw, {"file": rel, "line": lineno}))

    if not items:
        print("[错误] 一个输入文件都没读到")
        return 1

    result = sft.build(items)
    counts = result["report"]["counts"]

    out_dir = Path(args.out)
    write_jsonl(out_dir / "train.jsonl", result["train"])
    write_jsonl(out_dir / "holdout.jsonl", result["holdout"])
    write_jsonl(out_dir / "quarantine.jsonl", result["quarantine"])

    report = {
        "schema_version": sft.SCHEMA_VERSION,
        "generated_at": provenance.now_iso(),
        "inputs": inputs_meta,
        **result["report"],
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"输入 {counts['input_samples']} 条（{len(inputs_meta)} 个文件）")
    print(
        f"有效 {counts['valid_samples']}"
        f" | duplicate {counts['duplicate_samples']}"
        f" | rerun_variance {counts['rerun_variance_groups']} 组"
        f"（{counts['rerun_variance_samples']} 条）"
        f" | true_conflict {counts['true_conflict_groups']} 组"
        f"（{counts['true_conflict_samples']} 条）"
        f" | unresolved {counts['unresolved_groups']} 组"
    )
    print(
        f"缺字段 {counts['missing_field_samples']}"
        f" | tool_calls 异常 {counts['bad_tool_calls']}"
    )
    print(
        f"canonical {counts['canonical_samples']}"
        f" | train {counts['train_samples']} / holdout {counts['holdout_samples']}"
        f" / unknown {counts['unknown_samples']} / 隔离 {counts['isolation_samples']}"
    )
    if report["reasons"]:
        print("隔离原因：" + "  ".join(f"{k}={v}" for k, v in report["reasons"].items()))
    if report["reruns"]:
        print(
            f"同源重跑 {len(report['reruns'])} 组："
            "canonical 与 rerun 的对应见 report.json 的 reruns"
        )
    pending = [g for g in report["groups"] if g["kind"] in ("true_conflict", "unresolved")]
    if pending:
        print(f"待人工审查 {len(pending)} 组（true_conflict / unresolved），逐组清单见 report.json 的 groups")

    print(f"\n输出：{provenance.rel_path(out_dir / 'train.jsonl')}")
    print(f"      {provenance.rel_path(out_dir / 'holdout.jsonl')}")
    print(f"      {provenance.rel_path(out_dir / 'quarantine.jsonl')}")
    print(f"      {provenance.rel_path(out_dir / 'report.json')}")
    print(f"可训练样本：{counts['trainable_samples']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
