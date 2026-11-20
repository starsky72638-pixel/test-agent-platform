"""执行轨迹的数据契约。

这一层刻意不知道任何 Agent 框架、任何业务域、任何判分规则。
它只回答一个问题：一次任务的执行过程，能不能被完整、保序、可往返地记下来。

三条硬要求（都有对应单测）：
1. 保序：第几轮、这一轮第几次调用，顺序不能丢。
2. 一对一：一次调用对应它自己那次的返回，同一个工具被调两次也不能串。
3. 可往返：to_dict -> json -> from_dict 之后，内容完全一致。

约定：Turn.index 从 0 开始，与列表下标一致；ToolCall.call_id 形如 "t0c1"。

Turn 与 Event 是两套语义，不能互相替代：
    Turn  回答"用户说了什么、答了什么"，数量由对话轮次决定；
    Event 回答"这一步做了什么"，数量由执行路径决定。
    一个 5 节点 workflow 跑一条单轮任务：turns 是 1，events 是十几个。
    把节点塞进 Turn 会让判分、蒸馏、界面三处都拿不到真实结构——
    所以事件是独立的一层，Turn 保持原样不动。
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# 轨迹格式版本。字段语义发生不兼容变化时必须递增，读旧文件时据此判断。
SCHEMA_VERSION = "1.0"

STATUS_OK = "ok"
STATUS_INFRA = "infra_error"
STATUS_MAX_STEPS = "max_steps"
STATUS_CRASHED = "crashed"
STATUS_EMPTY = "empty_trace"

STATUSES = (
    STATUS_OK,
    STATUS_INFRA,
    STATUS_MAX_STEPS,
    STATUS_CRASHED,
    STATUS_EMPTY,
)

# 这些状态下的轨迹不参与能力评估，只用于排障。
NON_EVALUABLE_STATUSES = frozenset({STATUS_INFRA, STATUS_CRASHED, STATUS_EMPTY})


def now_ms() -> int:
    """当前时间戳（毫秒）。抽出来便于测试替换。"""
    return int(time.time() * 1000)


@dataclass
class ToolCall:
    """一次工具调用：调了谁、带什么参数、返回了什么、是否成功。"""

    name: str
    args: dict = field(default_factory=dict)
    call_id: str = ""
    output: str = ""
    ok: bool = True
    error: str = ""
    elapsed_ms: int = 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "args": dict(self.args or {}),
            "call_id": self.call_id,
            "output": self.output,
            "ok": bool(self.ok),
            "error": self.error,
            "elapsed_ms": int(self.elapsed_ms or 0),
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "ToolCall":
        if not isinstance(raw, dict):
            raise TypeError(f"ToolCall 需要 dict，收到 {type(raw).__name__}")
        return cls(
            name=str(raw.get("name") or ""),
            args=dict(raw.get("args") or {}),
            call_id=str(raw.get("call_id") or ""),
            output=str(raw.get("output") or ""),
            ok=bool(raw.get("ok", True)),
            error=str(raw.get("error") or ""),
            elapsed_ms=int(raw.get("elapsed_ms") or 0),
        )

    def brief(self) -> str:
        """给日志和报告用的一行摘要，不做截断以外的改写。"""
        try:
            args = json.dumps(self.args or {}, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            args = str(self.args)
        if len(args) > 120:
            args = args[:117] + "..."
        return f"{self.name}({args})"


# ---- 执行事件 ----------------------------------------------------
# 与 Turn 的分工见模块 docstring。这里只定义"发生了什么"，不定义"算不算对"。
KIND_NODE_START = "node_start"
KIND_NODE_END = "node_end"
KIND_TOOL_CALL = "tool_call"
KIND_LLM_CALL = "llm_call"
KIND_BRANCH = "branch"
KIND_ERROR = "error"

EVENT_KINDS = (
    KIND_NODE_START,
    KIND_NODE_END,
    KIND_TOOL_CALL,
    KIND_LLM_CALL,
    KIND_BRANCH,
    KIND_ERROR,
)


@dataclass
class Event:
    """一次执行步骤：哪个节点、做了什么、花了多久、成没成。

    seq 是全局执行序号，跨节点单调递增，用来还原真实先后。
    不用时间戳排序：同一毫秒内产生的多条事件时间戳会相等，顺序就丢了。

    parent 记父子关系（子流程、嵌套 Agent），空串表示顶层节点。
    不在这里做树结构：事件是一条有序流，树由 parent 在消费端重建，
    这样落盘格式保持扁平，回放时不需要递归。
    """

    kind: str
    seq: int = -1
    node_id: str = ""
    node_type: str = ""
    parent: str = ""
    # 归属的对话轮次，与 Turn.index 对应。多轮 workflow 靠它把事件分回各轮。
    turn: int = 0
    name: str = ""
    args: dict = field(default_factory=dict)
    output: str = ""
    ok: bool = True
    error: str = ""
    started_at: int = 0
    elapsed_ms: int = 0

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "seq": int(self.seq),
            "node_id": self.node_id,
            "node_type": self.node_type,
            "parent": self.parent,
            "turn": int(self.turn or 0),
            "name": self.name,
            "args": dict(self.args or {}),
            "output": self.output,
            "ok": bool(self.ok),
            "error": self.error,
            "started_at": int(self.started_at or 0),
            "elapsed_ms": int(self.elapsed_ms or 0),
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "Event":
        if not isinstance(raw, dict):
            raise TypeError(f"Event 需要 dict，收到 {type(raw).__name__}")
        # kind 不做白名单校验：新版本可能新增事件类型，
        # 读旧代码时应该照原样保留，而不是拒绝整条轨迹。
        return cls(
            kind=str(raw.get("kind") or ""),
            seq=int(raw.get("seq", -1)),
            node_id=str(raw.get("node_id") or ""),
            node_type=str(raw.get("node_type") or ""),
            parent=str(raw.get("parent") or ""),
            turn=int(raw.get("turn") or 0),
            name=str(raw.get("name") or ""),
            args=dict(raw.get("args") or {}),
            output=str(raw.get("output") or ""),
            ok=bool(raw.get("ok", True)),
            error=str(raw.get("error") or ""),
            started_at=int(raw.get("started_at") or 0),
            elapsed_ms=int(raw.get("elapsed_ms") or 0),
        )

    def as_toolcall(self) -> ToolCall:
        """投影成 ToolCall，供判分器复用。

        判分器只认 ToolCall（evaluator.verdict.judge_round），不认 Event。
        这里做一次投影，Workflow 的调用序列就能直接进五层判定，
        不必再写一套并行判分逻辑。
        """
        return ToolCall(
            name=self.name,
            args=dict(self.args or {}),
            call_id=f"e{self.seq}" if self.seq >= 0 else "",
            output=self.output,
            ok=bool(self.ok),
            error=self.error,
            elapsed_ms=int(self.elapsed_ms or 0),
        )


@dataclass
class Turn:
    """一轮用户交互：用户说了什么、agent 做了什么、最后答了什么。"""

    index: int
    user_input: str = ""
    calls: list[ToolCall] = field(default_factory=list)
    answer: str = ""
    steps: int = 0
    elapsed_ms: int = 0
    meta: dict = field(default_factory=dict)

    def call_names(self) -> list[str]:
        return [c.name for c in self.calls]

    def outputs(self) -> list[str]:
        """与 calls 逐位对齐的返回原文。顺序即调用顺序。"""
        return [c.output for c in self.calls]

    def to_dict(self) -> dict:
        return {
            "index": int(self.index),
            "user_input": self.user_input,
            "calls": [c.to_dict() for c in self.calls],
            "answer": self.answer,
            "steps": int(self.steps or 0),
            "elapsed_ms": int(self.elapsed_ms or 0),
            "meta": dict(self.meta or {}),
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "Turn":
        if not isinstance(raw, dict):
            raise TypeError(f"Turn 需要 dict，收到 {type(raw).__name__}")
        calls = [ToolCall.from_dict(c) for c in (raw.get("calls") or [])]
        return cls(
            index=int(raw.get("index") or 0),
            user_input=str(raw.get("user_input") or ""),
            calls=calls,
            answer=str(raw.get("answer") or ""),
            steps=int(raw.get("steps") or 0),
            elapsed_ms=int(raw.get("elapsed_ms") or 0),
            meta=dict(raw.get("meta") or {}),
        )


@dataclass
class Trace:
    """一条完整轨迹 = 一个任务的一次执行。

    task_id 指向任务，trace_id 唯一标识这一次执行，两者不能混。
    同一任务跑多次会得到多条 Trace，这是对比实验的基础。
    """

    task_id: str
    agent: str
    turns: list[Turn] = field(default_factory=list)
    split: str = "unassigned"
    trace_id: str = ""
    status: str = STATUS_OK
    error: str = ""
    started_at: int = 0
    elapsed_ms: int = 0
    schema_version: str = SCHEMA_VERSION
    meta: dict = field(default_factory=dict)
    # 执行事件流。旧轨迹没有这个键，读出来是空列表，
    # 语义等价于"这条轨迹没有节点级信息"——不是不兼容变化，故不递增 SCHEMA_VERSION。
    events: list[Event] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.trace_id:
            self.trace_id = f"{self.task_id}-{uuid.uuid4().hex[:8]}"
        if not self.started_at:
            self.started_at = now_ms()
        if self.status not in STATUSES:
            raise ValueError(f"未知 status: {self.status!r}，可选 {STATUSES}")

    # -- 便捷视图 -------------------------------------------------

    @property
    def evaluable(self) -> bool:
        """是否具备被评估的前提：状态正常且至少有一轮。"""
        return self.status == STATUS_OK and bool(self.turns)

    def all_calls(self) -> list[ToolCall]:
        """按执行顺序展开的全部调用。

        有 event 时以 event 为准：workflow 的调用发生在节点里，Turn.calls 是空的。
        不在这里统一，Memory 溯源、质量闸门、蒸馏、CLI 打印每个消费方
        都要各自判断一次事件，迟早漏一处——漏的那处会静默地看到 0 次调用。
        没有 event 的老轨迹行为逐字不变。
        """
        if self.events:
            return [e.as_toolcall() for e in self.events if e.kind == KIND_TOOL_CALL]
        return [c for t in self.turns for c in t.calls]

    def call_names(self) -> list[str]:
        return [c.name for c in self.all_calls()]

    def outputs_by_call(self) -> list[tuple[str, dict, str]]:
        """(工具名, 参数, 该次返回) 三元组列表，逐次严格对齐。

        需要“第 N 次调用的返回”时用这个，不要用 outputs_by_name()。
        """
        return [(c.name, dict(c.args or {}), c.output) for c in self.all_calls()]

    def outputs_by_name(self) -> dict[str, str]:
        """工具名 -> 返回原文。**同名多次时只保留最后一次**。

        这个视图只适合“我只要该工具最新一次结果”的场景。
        要做依据溯源、逐次核对时请用 outputs_by_call()。
        """
        out: dict[str, str] = {}
        for c in self.all_calls():
            out[c.name] = c.output
        return out

    def answers(self) -> list[str]:
        return [t.answer for t in self.turns]

    def final_answer(self) -> str:
        return self.turns[-1].answer if self.turns else ""

    # -- 执行事件 -------------------------------------------------

    def add_event(self, kind: str, **kwargs) -> Event:
        """追加一个事件，编号由这里统一分配。

        为什么不让调用方自己写 seq：同一批事件由不同节点产生，
        各自记号一定会撞号；集中分配，"谁先谁后"才有唯一来源。
        注意它不改 turns——事件与对话轮次是两条独立的线。
        """
        event = Event(kind=kind, **kwargs)
        event.seq = len(self.events)
        self.events.append(event)
        return event

    def events_of(self, kind: str) -> list[Event]:
        """按类型取事件，保持原顺序。"""
        return [e for e in self.events if e.kind == kind]

    def node_span(self) -> list[tuple[str, int]]:
        """(node_id, elapsed_ms) 列表，供报告统计节点耗时。"""
        return [(e.node_id, e.elapsed_ms) for e in self.events if e.kind == KIND_NODE_END]

    # -- 序列化 ---------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "trace_id": self.trace_id,
            "task_id": self.task_id,
            "agent": self.agent,
            "split": self.split,
            "status": self.status,
            "error": self.error,
            "started_at": int(self.started_at),
            "elapsed_ms": int(self.elapsed_ms or 0),
            "turns": [t.to_dict() for t in self.turns],
            "events": [e.to_dict() for e in self.events],
            "meta": dict(self.meta or {}),
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "Trace":
        if not isinstance(raw, dict):
            raise TypeError(f"Trace 需要 dict，收到 {type(raw).__name__}")
        return cls(
            task_id=str(raw.get("task_id") or ""),
            agent=str(raw.get("agent") or ""),
            turns=[Turn.from_dict(t) for t in (raw.get("turns") or [])],
            events=[Event.from_dict(e) for e in (raw.get("events") or [])],
            split=str(raw.get("split") or "unassigned"),
            trace_id=str(raw.get("trace_id") or ""),
            status=str(raw.get("status") or STATUS_OK),
            error=str(raw.get("error") or ""),
            started_at=int(raw.get("started_at") or 0),
            elapsed_ms=int(raw.get("elapsed_ms") or 0),
            schema_version=str(raw.get("schema_version") or SCHEMA_VERSION),
            meta=dict(raw.get("meta") or {}),
        )

    # -- 落盘 -----------------------------------------------------

    def save(self, path: str | Path, append: bool = False) -> Path:
        """写单条轨迹。append=True 时按 jsonl 追加一行。"""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(self.to_dict(), ensure_ascii=False)
        mode = "a" if append else "w"
        with p.open(mode, encoding="utf-8") as f:
            f.write(line + "\n")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "Trace":
        """读文件里的第一条轨迹（单条 json 或 jsonl 首行）。"""
        p = Path(path)
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    return cls.from_dict(json.loads(line))
        raise ValueError(f"文件里没有轨迹: {p}")


def load_traces(path: str | Path) -> list[Trace]:
    """读一份 jsonl，返回全部轨迹。空行跳过。"""
    p = Path(path)
    traces: list[Trace] = []
    with p.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                traces.append(Trace.from_dict(json.loads(line)))
            except Exception as exc:  # 定位到行号，别只报"解析失败"
                raise ValueError(f"{p}:{lineno} 解析失败: {exc}") from exc
    return traces


def save_traces(traces: Iterable[Trace], path: str | Path) -> Path:
    """整批覆盖写入 jsonl。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")
    return p


def make_trace(trace_id: str) -> Trace:
    """按 id 建一条空轨迹，用于极简演示与测试。"""
    return Trace(task_id=trace_id, agent="unknown")


def summarize(traces: Iterable[Trace]) -> dict[str, Any]:
    """轨迹集合的粗统计，给 CLI 打印用，不含任何判分语义。"""
    items = list(traces)
    total_calls = sum(len(t.all_calls()) for t in items)
    return {
        "traces": len(items),
        "evaluable": sum(1 for t in items if t.evaluable),
        "turns": sum(len(t.turns) for t in items),
        "events": sum(len(t.events) for t in items),
        "calls": total_calls,
        "by_status": {
            s: sum(1 for t in items if t.status == s) for s in STATUSES if any(t.status == s for t in items)
        },
    }
