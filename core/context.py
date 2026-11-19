"""上下文管理原语：跨轮不丢的结构化槽位。

为什么放在 core，而不是某个 Agent 的 view 目录里：
    "窗口裁掉之后模型想不起订单号/预算"不是电商问题，是所有多轮 Agent 的问题。
    机制（按工具实参反填槽位、渲染成提示）与业务无关；
    业务那一半是「哪个工具的哪个参数填哪个槽」的映射表，由 Agent 传进来。
    这样接一个新 Agent 只写一张映射表，不用把槽位逻辑重写一遍，
    Runtime 也不会长出一个叫 search_products 的字段名。

为什么只认工具实参，不从用户原话抽：
    参数是结构化的、零歧义的、可离线单测的。
    从"三百块""不超过 900"里写正则要处理中文数字、口语副词，
    错一次就污染后面整轮，而且出错时没有任何人能看见。
    代价是：用户说了但模型没传的参数进不了槽位——
    这条边界由 Agent 侧的提示词负责，不是这一层的职责。

为什么空值不覆盖：
    一次没传 max_price 的搜索不能把上一轮的预算清掉，
    否则"用户没重复说"反而变成"用户取消了限制"。
"""

from __future__ import annotations

__all__ = ["render_slots", "update_slots"]


def update_slots(slots: dict, calls, mapping) -> dict:
    """从工具调用实参反填槽位，原地修改并返回。

    slots:   要维护的字典，跨轮保持同一个对象即可。
    calls:   [{"name": ..., "args": {...}}, ...]，只包含本轮真的执行过的调用。
    mapping: {(工具名, 参数名): 槽位名}，由 Agent 提供。

    未知工具、未知参数、空值一律跳过，不抛异常。
    """
    for call in calls or []:
        name = (call or {}).get("name", "")
        args = (call or {}).get("args") or {}
        for (tool, arg), slot in (mapping or {}).items():
            if tool != name:
                continue
            value = args.get(arg)
            if value in (None, "", []):
                continue
            slots[slot] = value
    return slots


def render_slots(slots: dict, labels: dict) -> str:
    """把槽位渲染成 system prompt 末尾的一段；全空时返回空串。

    labels: {槽位名: 给模型看的说明}。只渲染 labels 里有的键，
    顺序也按 labels 走，保证同一个 handler 渲染出来的提示是稳定的。
    """
    lines = []
    for key, label in (labels or {}).items():
        value = (slots or {}).get(key)
        if value in (None, "", []):
            continue
        lines.append(f"- {label}：{value}")
    if not lines:
        return ""
    return (
        "当前会话已确认的槽位（用户这一轮没重复说，也直接沿用，不要反问）：\n"
        + "\n".join(lines)
    )
