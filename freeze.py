"""把一次运行的用例冻结成 Golden Set，之后改 prompt 都用它做回归。"""

from __future__ import annotations

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REPORT = BASE_DIR / "reports" / "report.json"
OUT = BASE_DIR / "data" / "golden.jsonl"

KEYS = ("id", "title", "path", "method", "query",
        "path_params", "body", "category", "expect_status")


def main() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    cases = report["results"]

    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps({k: c[k] for k in KEYS}, ensure_ascii=False) + "\n")

    print(f"已冻结 {len(cases)} 条 → {OUT}")
    print("下一步：逐行看 expect_status 是否正确，这是你唯一的人工标注工作。")


if __name__ == "__main__":
    main()
