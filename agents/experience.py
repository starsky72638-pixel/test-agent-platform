"""接入层的经验挂载约定：任何 Agent 拿到的经验块，措辞必须逐字一致。

为什么单独抽出来：
    对比实验只允许有一个变量——Agent 本身。如果 shop 用一套措辞、
    main_agent 用另一套，"注入后结果不同"就分不清是 Agent 的差别
    还是提示词措辞的差别。所以经验块由接入层统一渲染，
    Agent 只决定"挂在哪"（各自都是 system prompt 末尾）。

它不检索、不筛选、不读经验库——那些在 memory/。
这里只做一件事：把已经渲染好的经验文本贴到提示词后面，并加一句边界声明。

边界声明为什么必须写：
    经验讲的是做事的次序和检查点，不提供价格、状态这类事实值。
    不写清楚，模型会把经验里出现的示例当成可以照抄的答案——
    那就从"没有依据的编造"变成"有依据但已过期的编造"，更难查。
"""

from __future__ import annotations

# 经验块前面的固定说明。措辞改动会影响所有 Agent，改之前先想清楚。
EXPERIENCE_RULE = (
    "---\n"
    "以下是过往同类任务失败后沉淀的经验，讲的是做事的次序和检查点。\n"
    "它们不提供任何业务数据（价格、状态、单号一律以工具返回为准），\n"
    "与工具返回冲突时一律以工具返回为准。\n"
)


def attach(prompt: str, experience: str) -> str:
    """把经验文本挂到提示词末尾。经验为空时原样返回，不留空段落。"""
    body = (experience or "").strip()
    if not body:
        return prompt
    return f"{prompt}\n\n{EXPERIENCE_RULE}\n{body}"


__all__ = ["attach", "EXPERIENCE_RULE"]
