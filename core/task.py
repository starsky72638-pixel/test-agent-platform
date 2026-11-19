"""任务集的数据契约，以及写死的 fit / holdout 划分。

两个职责，都很克制：
1. 把一份 jsonl 读成 Task 列表，断言内容原样保存，不解析语义。
2. 决定一条任务属于训练侧（fit）还是验证侧（holdout）。

为什么断言不解析：
一旦核心层认识 expect_tools / expect_order 这些关键字，它就与某个评测口径绑死，
换一种评判方式就得改核心。核心只负责搬，解读交给 evaluator。

划分的优先级（高到低）：
1. 调用方显式传入的 split —— 供测试与特殊流程覆盖，保留旧调用方式；
2. 数据文件里每条任务自己的 "split" 字段 —— 新数据集不必再改核心代码，
   换一个业务场景（客服/旅行/金融）时划分跟着数据走；
3. FIT_IDS 常量表 —— 老数据集没有该字段，仍按这张表判定。

第 3 条是兜底，也是默认安全方向：不在 FIT_IDS 里的任务一律进 holdout。
新任务、新数据集忘了声明，最坏是白跑一遍，绝不会悄悄泄进训练侧。
划分不随机：随机划分每次结果不同，等于给了“事后挑一个好划分”的机会。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

SPLIT_FIT = "fit"
SPLIT_HOLDOUT = "holdout"
SPLIT_UNASSIGNED = "unassigned"

SPLITS = (SPLIT_FIT, SPLIT_HOLDOUT, SPLIT_UNASSIGNED)

# 训练侧：只用于提炼经验。验证侧：只用于验收，绝不许参与提炼。
# 划分依据是题型覆盖：单轮/多轮/难集三套各自都留了代表题在 holdout 里。
FIT_IDS = frozenset(
    {
        # 单轮 20 条 -> 12
        "S01", "S02", "S03", "S05", "S06", "S07", "S08",
        "S10", "S11", "S13", "S15", "S17",
        # 多轮 10 条 -> 6
        "M01", "M02", "M04", "M06", "M08", "M10",
        # 难集 13 条 -> 8
        "H01", "H03", "H05", "H07", "H09", "H11", "H12", "H13",
    }
)


def assign_split(task_id: str) -> str:
    """任务 -> 划分。未登记的 id 一律 holdout。"""
    return SPLIT_FIT if task_id in FIT_IDS else SPLIT_HOLDOUT


@dataclass
class Task:
    """一条评测任务。

    turns 为空时视为单轮任务，用户输入取 input。
    assertions 是原始字典，可能包含整体断言与逐轮断言，核心层不解读。
    """

    id: str
    input: str = ""
    turns: list[str] = field(default_factory=list)
    assertions: dict = field(default_factory=dict)
    split: str = SPLIT_HOLDOUT
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Task.id 不能为空")
        if self.split not in SPLITS:
            raise ValueError(f"未知 split: {self.split!r}，可选 {SPLITS}")
        self.turns = [str(t) for t in (self.turns or [])]
        self.assertions = dict(self.assertions or {})
        self.meta = dict(self.meta or {})

    @property
    def is_multi_turn(self) -> bool:
        return len(self.turns) > 1

    def user_turns(self) -> list[str]:
        """实际的用户输入序列。多轮取 turns，单轮取 input。"""
        return list(self.turns) if self.turns else [self.input]

    def turn_count(self) -> int:
        return len(self.user_turns())

    def turn_assertions(self) -> list[dict]:
        """逐轮断言，长度与轮数对齐；未配的轮补空 dict。

        仅做对齐，不解释字段含义。
        """
        raw = self.assertions.get("turn_checks") or []
        n = self.turn_count()
        out: list[dict] = []
        for i in range(n):
            item = raw[i] if i < len(raw) and isinstance(raw[i], dict) else {}
            out.append(dict(item))
        return out

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "input": self.input,
            "turns": list(self.turns),
            "assertions": dict(self.assertions),
            "split": self.split,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, raw: dict, split: str | None = None) -> "Task":
        if not isinstance(raw, dict):
            raise TypeError(f"Task 需要 dict，收到 {type(raw).__name__}")
        if not raw.get("id"):
            raise ValueError("任务缺少 id 字段")
        task_id = str(raw["id"])
        # 断言 = 除去结构性字段以外的全部内容，原样保留。
        structural = {"id", "input", "turns", "note", "split"}
        assertions = {k: v for k, v in raw.items() if k not in structural}
        meta = {"note": raw.get("note") or ""}
        # 数据文件可以自己声明划分；显式传入的 split 仍然优先（兼容旧调用方）。
        # 老数据集没有这个字段，自动回落到 FIT_IDS 表，行为不变。
        declared = raw.get("split")
        if split is None and isinstance(declared, str) and declared.strip():
            split = declared.strip()
        return cls(
            id=task_id,
            input=str(raw.get("input") or ""),
            turns=[str(t) for t in (raw.get("turns") or [])],
            assertions=assertions,
            split=split or assign_split(task_id),
            meta=meta,
        )


def load_tasks(path: str | Path) -> list[Task]:
    """读一份任务 jsonl。路径由调用方给，核心层不猜数据集在哪。"""
    p = Path(path)
    tasks: list[Task] = []
    seen: set[str] = set()
    with p.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{p}:{lineno} 不是合法 JSON: {exc}") from exc
            task = Task.from_dict(raw)
            if task.id in seen:
                raise ValueError(f"{p}:{lineno} 任务 id 重复: {task.id}")
            seen.add(task.id)
            tasks.append(task)
    return tasks


def load_many(paths: Iterable[str | Path]) -> list[Task]:
    """读多份任务 jsonl 并合并。id 跨文件重复直接报错。"""
    all_tasks: list[Task] = []
    for path in paths:
        all_tasks.extend(load_tasks(path))
    ids = [t.id for t in all_tasks]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        raise ValueError(f"合并后任务 id 重复: {sorted(dup)}")
    return all_tasks


def select(tasks: Iterable[Task], split: str) -> list[Task]:
    """按划分筛选任务。"""
    if split not in SPLITS:
        raise ValueError(f"未知 split: {split!r}")
    return [t for t in tasks if t.split == split]


def split_counts(tasks: Iterable[Task]) -> dict[str, int]:
    items = list(tasks)
    return {s: sum(1 for t in items if t.split == s) for s in SPLITS}
