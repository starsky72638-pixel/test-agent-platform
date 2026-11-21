"""经验库：追加写落盘、版本留痕、可失效。

四个机制对应四条硬要求：

1. 存储：一行一条 JSON（jsonl）。同一 exp_id 更新时不覆盖，而是追加一条 version+1 的新行。
   读的时候取每个 exp_id 的最大 version。好处是任何一次修改都有历史可查，
   "这条经验上周说的是什么"不需要靠备份找。

2. 来源追踪：来源信息在 Experience.evidence 里（trace_id / task_id / 轮次 / 原文引用）。
   库里每条经验都能回到具体一次执行，不需要额外索引表。

3. 版本管理：version 随每次变更递增，updated_at 同步更新；snapshot() 可以导出当前视图，
   用于对比"注入前 / 注入后"的库内容。

4. 失效：两条路径。
   - 时间：超过 ttl_days 自动退休（默认 30 天，防止旧经验长期主导提示词）；
   - 效果：注入后仍失败记 contradict，反证数量压过支持且确实被注入过，自动退休。
   退休是粘性的：不因为"又失败了一次"就复活，否则一条有毒经验会靠反复失败给自己刷支持度。
   要复活必须人工把 status 改回 active 并重写一条新版本。

为什么不用数据库：库的规模是几十条，jsonl 足够，而且便于 git diff 和人工审核。
换成数据库的唯一动机是并发，这个项目没有并发。
"""

from __future__ import annotations

import json
from pathlib import Path

from core.trace import now_ms

from memory.schema import (
    STATUS_ACTIVE,
    STATUS_RETIRED,
    Experience,
)

# 自动退休阈值：至少被注入过这么多次，且反证明显多于支持，才判定无效。
RETIRE_MIN_INJECTED = 2
RETIRE_CONTRADICT_MARGIN = 1

# 归并后的单一库。放在 data/ 而不是 runs/：
# runs/ 是"某一次实验的产物"，data/ 是"当前在用的数据"。
# 路径在这里定义一次，接入层与脚本都从这里取，避免又长出一份平行定义。
DEFAULT_STORE_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "memory" / "experiences.jsonl"
)


class ExperienceStore:
    """经验库。所有变更都立即落盘，进程挂掉不丢已写入的内容。"""

    def __init__(self, path: str | Path, *, auto_load: bool = True) -> None:
        self.path = Path(path)
        self._by_id: dict[str, Experience] = {}
        self._lines = 0
        if auto_load and self.path.exists():
            self.load()

    # -- 读 -------------------------------------------------------

    def load(self) -> "ExperienceStore":
        """读整个文件，同名取最大版本。返回自身便于链式调用。"""
        self._by_id.clear()
        self._lines = 0
        if not self.path.exists():
            return self
        with self.path.open(encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                self._lines += 1
                try:
                    exp = Experience.from_dict(json.loads(line))
                except Exception as exc:
                    raise ValueError(f"{self.path}:{lineno} 经验解析失败: {exc}") from exc
                current = self._by_id.get(exp.exp_id)
                if current is None or exp.version >= current.version:
                    self._by_id[exp.exp_id] = exp
        return self

    def all(self) -> list[Experience]:
        return sorted(self._by_id.values(), key=lambda e: (e.created_at, e.exp_id))

    def get(self, exp_id: str) -> Experience | None:
        return self._by_id.get(exp_id)

    def active(self, now: int | None = None) -> list[Experience]:
        """当前可用经验：未退休、未过期。不修改库内容。"""
        return [
            e
            for e in self.all()
            if e.status == STATUS_ACTIVE and not e.is_expired(now)
        ]

    def __len__(self) -> int:
        return len(self._by_id)

    # -- 写 -------------------------------------------------------

    def add(self, exp: Experience, *, now: int | None = None) -> str:
        """入库。返回 "new" / "merged"。

        同 exp_id 视为同一失败模式的再次出现：合并触发条件与证据，support+1，版本+1。
        """
        stamp = int(now or now_ms())
        current = self._by_id.get(exp.exp_id)
        if current is None:
            exp.version = max(1, int(exp.version))
            exp.created_at = int(exp.created_at or stamp)
            exp.updated_at = stamp
            self._by_id[exp.exp_id] = exp
            self._append(exp)
            return "new"

        current.merge_from(exp, now=stamp)
        self._append(current)
        return "merged"

    def add_many(self, exps: list[Experience], *, now: int | None = None) -> dict[str, int]:
        result = {"new": 0, "merged": 0}
        for exp in exps or []:
            result[self.add(exp, now=now)] += 1
        return result

    def record_injection(self, exp_ids: list[str], *, now: int | None = None) -> int:
        """记一次注入。这是计算"注入后成功率"的分母，不能省。"""
        stamp = int(now or now_ms())
        count = 0
        for exp_id in exp_ids or []:
            exp = self._by_id.get(exp_id)
            if exp is None:
                continue
            exp.injected += 1
            exp.version += 1
            exp.updated_at = stamp
            self._append(exp)
            count += 1
        return count

    def record_outcome(
        self, exp_ids: list[str], *, passed: bool, now: int | None = None
    ) -> dict[str, int]:
        """回填"注入之后这条任务过没过"。

        通过 -> support+1；失败 -> contradict+1。随后自动检查退休条件。
        """
        stamp = int(now or now_ms())
        touched = 0
        for exp_id in exp_ids or []:
            exp = self._by_id.get(exp_id)
            if exp is None:
                continue
            if passed:
                exp.support += 1
            else:
                exp.contradict += 1
            exp.meta["last_outcome"] = "passed" if passed else "failed"
            exp.version += 1
            exp.updated_at = stamp
            self._append(exp)
            touched += 1
        retired = self.refresh_status(now=stamp)
        return {"touched": touched, "retired": len(retired)}

    def retire(self, exp_id: str, reason: str, *, now: int | None = None) -> bool:
        exp = self._by_id.get(exp_id)
        if exp is None or exp.status == STATUS_RETIRED:
            return False
        stamp = int(now or now_ms())
        exp.status = STATUS_RETIRED
        exp.meta["retired_reason"] = reason
        exp.meta["retired_at"] = stamp
        exp.version += 1
        exp.updated_at = stamp
        self._append(exp)
        return True

    def refresh_status(self, *, now: int | None = None) -> list[str]:
        """按 TTL 与反证情况退休一批经验，返回本次退休的 id。"""
        stamp = int(now or now_ms())
        retired: list[str] = []
        for exp in self.all():
            if exp.status != STATUS_ACTIVE:
                continue
            if exp.is_expired(stamp):
                if self.retire(exp.exp_id, "ttl_expired", now=stamp):
                    retired.append(exp.exp_id)
                continue
            if (
                exp.injected >= RETIRE_MIN_INJECTED
                and exp.contradict - exp.support >= RETIRE_CONTRADICT_MARGIN
            ):
                if self.retire(exp.exp_id, "contradicted", now=stamp):
                    retired.append(exp.exp_id)
        return retired

    # -- 落盘 -----------------------------------------------------

    def _append(self, exp: Experience) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(exp.to_line() + "\n")
        self._lines += 1

    def snapshot(self, path: str | Path) -> Path:
        """导出当前视图（每个 exp_id 一行），用于对比或交给别人看。"""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as f:
            for exp in self.all():
                f.write(exp.to_line() + "\n")
        return target

    # -- 统计 -----------------------------------------------------

    def stats(self) -> dict:
        exps = self.all()
        by_failure: dict[str, int] = {}
        for exp in exps:
            by_failure[exp.failure_type] = by_failure.get(exp.failure_type, 0) + 1
        return {
            "file": str(self.path),
            "lines": self._lines,
            "total": len(exps),
            "active": sum(1 for e in exps if e.status == STATUS_ACTIVE),
            "retired": sum(1 for e in exps if e.status == STATUS_RETIRED),
            "by_failure": by_failure,
            "injected_total": sum(e.injected for e in exps),
            "support_total": sum(e.support for e in exps),
            "contradict_total": sum(e.contradict for e in exps),
        }


def merge_files(
    paths, out_path: str | Path | None = None, *, write: bool = True
) -> dict:
    """把多个经验库合成一份视图：每个 exp_id 取 (updated_at, version) 最新的那条。

    为什么不做"语义合并"（support / contradict 相加）：
        同一批实验会被重复跑进多个目录，相加等于把"发生过一次"记成"发生过五次"，
        之后任何按 support 做的判断都会偏。取最新快照是唯一不会凭空造数字的选择。
        代价是：某个 exp_id 在不同库里各自演进过时，只保留演进得最靠后的那一份，
        其余版本留在原文件里，不丢——本函数只读输入，绝不改它们。

    为什么先写临时文件再 replace：
        中途失败留下半个库，比没有库更糟：下次 load 会把它当完整数据读。
    """
    best: dict[str, Experience] = {}
    inputs: list[str] = []

    for item in paths or []:
        p = Path(item)
        if not p.is_file():
            continue
        inputs.append(str(p))
        for exp in ExperienceStore(p).all():
            current = best.get(exp.exp_id)
            if current is None or (exp.updated_at, exp.version) > (
                current.updated_at,
                current.version,
            ):
                best[exp.exp_id] = exp

    out = Path(out_path) if out_path else DEFAULT_STORE_PATH
    if write:
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for exp in sorted(best.values(), key=lambda e: (e.created_at, e.exp_id)):
                f.write(exp.to_line() + "\n")
        tmp.replace(out)

    return {
        "sources": len(inputs),
        "inputs": inputs,
        "unique": len(best),
        "out": str(out) if write else None,
    }
