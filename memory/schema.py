"""经验条目的数据契约。

一条经验回答三个问题，缺一不可：
1. 什么时候适用（Trigger）——由用户输入的关键词、任务形态、涉及的工具决定；
2. 该怎么做（action / guard）——可执行的指令，不是对失败的自然语言总结；
3. 凭什么这么说（Evidence）——哪条轨迹、第几轮、哪种失败、原文是什么。

三条硬约束：
- 动作文本里不允许出现任何具体数字或编号。一旦出现，就等于把某道题的答案
  写进了提示词，换一道题就变成误导，而且会把 holdout 的答案泄漏进训练产物。
  证据文本不受此限：它是给人查账用的，不进入提示词。
- exp_id 由失败模式（pattern）决定，不由题目决定。同一类错误在不同任务上重复出现，
  应聚合成一条并累加 support，而不是攒出几十条一题一条的碎片。
- 计数语义分离：support 是"提炼时被观察到的次数 + 注入后成功的次数"，
  contradict 是"注入后仍然失败"，injected 是"被注入过多少次"。
  没有 injected 这个分母，contradict 就无从判断是否该退休。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from core.trace import now_ms

SCHEMA_VERSION = "1.0"

STATUS_ACTIVE = "active"
STATUS_RETIRED = "retired"
STATUSES = (STATUS_ACTIVE, STATUS_RETIRED)

SHAPE_SINGLE = "single"
SHAPE_MULTI = "multi"
SHAPES = (SHAPE_SINGLE, SHAPE_MULTI)

DEFAULT_TTL_DAYS = 30.0
MAX_EVIDENCE = 3
MAX_TAGS = 32


def make_exp_id(pattern: str) -> str:
    """失败模式 -> 稳定的经验 id。

    只取模式、不取题目，所以同类错误会落到同一个 id 上，由 store 合并。
    """
    digest = hashlib.sha1((pattern or "").encode("utf-8")).hexdigest()
    return f"exp-{digest[:12]}"


def shape_of_turn_count(turn_count: int) -> str:
    return SHAPE_MULTI if int(turn_count or 1) > 1 else SHAPE_SINGLE


@dataclass
class Trigger:
    """适用条件。只用可匹配的浅层特征，不存任何答案。

    tags：从用户原话抽的关键词（中文二元组 / 英文词），不含数字。
    tools：这条经验涉及的工具名，用于人工核对，不参与打分。
    shape：单轮 / 多轮，多轮的缺口和单轮不一样，混用会互相误导。
    """

    tags: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    shape: str = SHAPE_SINGLE

    def __post_init__(self) -> None:
        self.tags = sorted({str(t) for t in (self.tags or []) if str(t).strip()})[:MAX_TAGS]
        self.tools = sorted({str(t) for t in (self.tools or []) if str(t).strip()})
        if self.shape not in SHAPES:
            raise ValueError(f"未知 shape: {self.shape!r}，可选 {SHAPES}")

    def union(self, other: "Trigger") -> "Trigger":
        """合并触发条件：标签与工具取并集，形态以自己为准（形态已进 pattern）。"""
        merged = sorted(set(self.tags) | set(other.tags))[:MAX_TAGS]
        return Trigger(tags=merged, tools=sorted(set(self.tools) | set(other.tools)), shape=self.shape)

    def to_dict(self) -> dict:
        return {"tags": list(self.tags), "tools": list(self.tools), "shape": self.shape}

    @classmethod
    def from_dict(cls, raw: dict) -> "Trigger":
        raw = raw or {}
        return cls(
            tags=list(raw.get("tags") or []),
            tools=list(raw.get("tools") or []),
            shape=str(raw.get("shape") or SHAPE_SINGLE),
        )


@dataclass
class Evidence:
    """一条经验的来源。字段全部可回查，不接受"凭感觉"。

    quote 必须能在该条轨迹里原样找到（回答或某次工具返回的片段），
    唯一例外是空回答类失败，此时允许为空。
    """

    trace_id: str
    task_id: str
    turn: int
    failure_type: str
    quote: str = ""
    agent: str = ""
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "task_id": self.task_id,
            "turn": int(self.turn),
            "failure_type": self.failure_type,
            "quote": self.quote,
            "agent": self.agent,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "Evidence":
        raw = raw or {}
        return cls(
            trace_id=str(raw.get("trace_id") or ""),
            task_id=str(raw.get("task_id") or ""),
            turn=int(raw.get("turn") or 0),
            failure_type=str(raw.get("failure_type") or ""),
            quote=str(raw.get("quote") or ""),
            agent=str(raw.get("agent") or ""),
            detail=str(raw.get("detail") or ""),
        )

    def key(self) -> tuple[str, int]:
        return (self.trace_id, int(self.turn))


@dataclass
class Experience:
    """一条可检索、可注入、可失效的经验。"""

    exp_id: str
    failure_type: str
    pattern: str
    action: str
    trigger: Trigger = field(default_factory=Trigger)
    guard: str = ""
    evidence: list[Evidence] = field(default_factory=list)
    version: int = 1
    status: str = STATUS_ACTIVE
    support: int = 1
    contradict: int = 0
    injected: int = 0
    ttl_days: float = DEFAULT_TTL_DAYS
    created_at: int = 0
    updated_at: int = 0
    origin_split: str = "fit"
    schema_version: str = SCHEMA_VERSION
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"未知 status: {self.status!r}，可选 {STATUSES}")
        now = now_ms()
        self.created_at = int(self.created_at or now)
        self.updated_at = int(self.updated_at or now)
        self.evidence = list(self.evidence or [])
        self.meta = dict(self.meta or {})
        if self.ttl_days is None:
            self.ttl_days = DEFAULT_TTL_DAYS

    # -- 视图 -----------------------------------------------------

    @property
    def confidence(self) -> float:
        """注入后成功率。没有注入记录时退回"被观察到的次数"做先验。"""
        if self.injected > 0:
            total = self.support + self.contradict
            if total > 0:
                return round(self.support / total, 3)
        return round(min(1.0, 0.5 + 0.1 * max(0, self.support - 1)), 3)

    def text(self) -> str:
        """注入用文本：动作 + 负向约束。不含任何证据细节。"""
        body = self.action.strip()
        if self.guard.strip():
            body = f"{body} {self.guard.strip()}"
        return body

    def is_expired(self, now: int | None = None) -> bool:
        if self.ttl_days <= 0:
            return False
        age_ms = (now if now is not None else now_ms()) - int(self.created_at or 0)
        return age_ms > self.ttl_days * 86_400_000

    def age_days(self, now: int | None = None) -> float:
        delta = (now if now is not None else now_ms()) - int(self.created_at or 0)
        return round(delta / 86_400_000, 2)

    # -- 合并 -----------------------------------------------------

    def merge_from(self, other: "Experience", now: int | None = None) -> "Experience":
        """把同一模式的新证据并进来。版本 +1，旧版本仍留在落盘文件里。

        退休状态是粘性的：不因为又出现一次失败就自动复活。
        否则一条被判定无效的经验会靠"反复失败"给自己刷支持度。
        """
        if other.exp_id != self.exp_id:
            raise ValueError(f"只能合并同一模式的经验: {self.exp_id} != {other.exp_id}")
        self.support += max(1, int(other.support or 1))
        self.trigger = self.trigger.union(other.trigger)
        seen = {e.key() for e in self.evidence}
        for ev in other.evidence:
            if ev.key() not in seen:
                self.evidence.append(ev)
                seen.add(ev.key())
        if len(self.evidence) > MAX_EVIDENCE:
            self.evidence = self.evidence[-MAX_EVIDENCE:]
        self.version += 1
        self.updated_at = int(now or now_ms())
        if self.status == STATUS_RETIRED:
            self.meta["reobserved_while_retired"] = int(self.meta.get("reobserved_while_retired", 0)) + 1
        return self

    # -- 序列化 ---------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "exp_id": self.exp_id,
            "failure_type": self.failure_type,
            "pattern": self.pattern,
            "action": self.action,
            "guard": self.guard,
            "trigger": self.trigger.to_dict(),
            "evidence": [e.to_dict() for e in self.evidence],
            "version": int(self.version),
            "status": self.status,
            "support": int(self.support),
            "contradict": int(self.contradict),
            "injected": int(self.injected),
            "ttl_days": float(self.ttl_days),
            "created_at": int(self.created_at),
            "updated_at": int(self.updated_at),
            "origin_split": self.origin_split,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "Experience":
        if not isinstance(raw, dict):
            raise TypeError(f"Experience 需要 dict，收到 {type(raw).__name__}")
        if not raw.get("exp_id"):
            raise ValueError("经验缺少 exp_id")
        return cls(
            exp_id=str(raw["exp_id"]),
            failure_type=str(raw.get("failure_type") or ""),
            pattern=str(raw.get("pattern") or ""),
            action=str(raw.get("action") or ""),
            trigger=Trigger.from_dict(raw.get("trigger") or {}),
            guard=str(raw.get("guard") or ""),
            evidence=[Evidence.from_dict(e) for e in (raw.get("evidence") or [])],
            version=int(raw.get("version") or 1),
            status=str(raw.get("status") or STATUS_ACTIVE),
            support=int(raw.get("support") or 0),
            contradict=int(raw.get("contradict") or 0),
            injected=int(raw.get("injected") or 0),
            ttl_days=float(raw.get("ttl_days") if raw.get("ttl_days") is not None else DEFAULT_TTL_DAYS),
            created_at=int(raw.get("created_at") or 0),
            updated_at=int(raw.get("updated_at") or 0),
            origin_split=str(raw.get("origin_split") or "fit"),
            schema_version=str(raw.get("schema_version") or SCHEMA_VERSION),
            meta=dict(raw.get("meta") or {}),
        )

    def to_line(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def brief(self) -> str:
        return f"[{self.exp_id} v{self.version} {self.status}] {self.failure_type}: {self.text()}"
