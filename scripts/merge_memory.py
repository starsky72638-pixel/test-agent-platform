"""把散在各次实验目录里的经验库合成一份。

背景：每跑一次闭环就在 runs/<时间戳>/ 下新建一个 experiences.jsonl（这是故意的，
单次实验的库要能独立复现）。代价是同一个失败模式会在多个目录里各存一份，
统计"平台到底有多少条经验"时只能靠人肉去重。

这个脚本只做一件事：按 exp_id 取最新快照，写到单一路径。
不做语义合并（support/contradict 相加会重复计数），不删除、不改写任何输入文件。

用法：

    # 先看会合成多少条，不写盘
    .venv\\Scripts\\python.exe scripts\\merge_memory.py --dry-run

    # 合成到默认路径 data/memory/experiences.jsonl
    .venv\\Scripts\\python.exe scripts\\merge_memory.py

    # 只合指定的几个库，输出到别处
    .venv\\Scripts\\python.exe scripts\\merge_memory.py --from runs\\full-cold\\20261118-000730\\experiences.jsonl --out data\\memory\\probe.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory.store import DEFAULT_STORE_PATH, ExperienceStore, merge_files

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def find_sources() -> list[Path]:
    """runs/ 下所有经验库。按路径排序，保证两次运行结果一致。"""
    return sorted(PROJECT_ROOT.joinpath("runs").rglob("experiences.jsonl"))


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except (ValueError, OSError):
        return str(path)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="把多个经验库合成一份（按 exp_id 取最新快照）")
    p.add_argument(
        "--from",
        dest="sources",
        action="append",
        default=None,
        help="只合指定的库，可重复；不给则扫描 runs/ 下全部 experiences.jsonl",
    )
    p.add_argument("--out", default=str(DEFAULT_STORE_PATH), help="输出文件")
    p.add_argument("--dry-run", action="store_true", help="只报告条数，不写盘")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    paths = [Path(item) for item in args.sources] if args.sources else find_sources()

    existing = [p for p in paths if p.is_file()]
    if not existing:
        print("[错误] 没有找到任何经验库；检查路径或先跑一次 run_loop")
        return 1

    for path in paths:
        if not path.is_file():
            print(f"  跳过（不存在）{_rel(path)}")
            continue
        stats = ExperienceStore(path).stats()
        print(f"  {_rel(path):<58} 行 {stats['lines']:>3}  唯一 {stats['total']:>2}")

    result = merge_files(paths, args.out, write=not args.dry_run)

    print(f"\n来源 {result['sources']} 个库 | 去重后 {result['unique']} 条")
    if args.dry_run:
        print(f"[dry-run] 未写盘；将会写到 {_rel(Path(args.out))}")
    else:
        print(f"已写出：{_rel(Path(result['out']))}")
        print("原库未改动；接入层与 run_loop 的 --store default 都读这个路径。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
