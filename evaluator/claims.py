"""事实溯源：回答里的一个事实，到底来自哪一次工具调用。

旧实现用 outputs[工具名] 一段按名合并的文本做判断，只能回答"这个工具返回里有"。
这里改成逐次调用扫描，回答"第几轮的第几次调用、命中的是哪一段"。

为什么要精确到调用：
- 多轮里同名工具会被调多次，合并文本会让人误以为依据来自最近一次；
- 经验提炼需要知道"这条经验对应的是哪一次动作"，否则只能记"该调这个工具"。

语义与旧实现严格一致：
- 回答没提到这个值 → mentioned=False, grounded=True（不负责判缺失）；
- 回答了、且在指定工具的某次返回里找得到 → grounded=True，带证据；
- 回答了、但指定工具的回答里没有 → grounded=False。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.trace import ToolCall, Trace


@dataclass(frozen=True)
class Evidence:
    """一处事实依据：第几轮的第几次调用，命中了返回文本的哪个区间。"""

    turn: int
    call_index: int
    call_id: str
    tool: str
    span: tuple[int, int]
    snippet: str = ""

    def to_dict(self) -> dict:
        return {
            "turn": self.turn,
            "call_index": self.call_index,
            "call_id": self.call_id,
            "tool": self.tool,
            "span": list(self.span),
            "snippet": self.snippet,
        }


@dataclass
class Claim:
    """一条被检查的事实断言。"""

    value: str
    source_tool: str
    mentioned: bool = False
    grounded: bool = True
    evidence: list[Evidence] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "source_tool": self.source_tool,
            "mentioned": self.mentioned,
            "grounded": self.grounded,
            "evidence": [e.to_dict() for e in self.evidence],
        }


def flatten(trace: Trace) -> list[tuple[int, ToolCall]]:
    """把整条轨迹展平成 (轮次下标, 调用)，顺序即真实调用顺序。"""
    out: list[tuple[int, ToolCall]] = []
    for turn in trace.turns:
        for call in turn.calls:
            out.append((turn.index, call))
    return out


def turn_slice(trace: Trace, upto: int) -> list[tuple[int, ToolCall]]:
    """截至第 upto 轮（含）的所有调用。

    依据溯源是唯一允许跨轮取数的一层：用户在第二轮引用第一轮查到的价格是合法的。
    工具选择与参数校验不能这样做，否则"这一轮什么都没查"也会因为上一轮查过而通过。
    """
    out: list[tuple[int, ToolCall]] = []
    for turn in trace.turns:
        if turn.index > upto:
            continue
        for call in turn.calls:
            out.append((turn.index, call))
    return out


def resolve(
    claim_cfg: dict,
    answer: str,
    history: list[tuple[int, ToolCall]],
) -> Claim:
    """检查一条事实断言。

    history 是"截至本轮"的调用序列，由调用方控制范围。
    """
    value = str((claim_cfg or {}).get("value", ""))
    source = str((claim_cfg or {}).get("from", ""))
    text = "" if answer is None else str(answer)

    # 回答没提到这个值，本函数不负责判它缺失。
    if value not in text:
        return Claim(value=value, source_tool=source, mentioned=False, grounded=True)

    evidence: list[Evidence] = []
    for idx, (turn_no, call) in enumerate(history):
        if call.name != source:
            continue
        body = call.output or ""
        pos = body.find(value)
        if pos < 0:
            continue
        evidence.append(
            Evidence(
                turn=turn_no,
                call_index=idx,
                call_id=call.call_id,
                tool=call.name,
                span=(pos, pos + len(value)),
                snippet=_snippet(body, pos, len(value)),
            )
        )

    return Claim(
        value=value,
        source_tool=source,
        mentioned=True,
        grounded=bool(evidence),
        evidence=evidence,
    )


def resolve_all(
    claim_cfgs: list[dict] | None,
    answer: str,
    history: list[tuple[int, ToolCall]],
) -> list[Claim]:
    return [resolve(cfg, answer, history) for cfg in (claim_cfgs or [])]


def ungrounded_values(claims: list[Claim]) -> list[str]:
    return [c.value for c in claims if not c.grounded]


def _snippet(body: str, start: int, length: int, pad: int = 18) -> str:
    """截取命中位置附近的一小段，供报告展示。不改动原文。"""
    lo = max(0, start - pad)
    hi = min(len(body), start + length + pad)
    return body[lo:hi].replace("\n", " ")
