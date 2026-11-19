"""上下文层。

回答"这个 Agent 看到的提示词是什么、经验挂在哪"。
不含任何工具定义、不含任何模型调用。
"""

from agents.shop.view.context import (
    ALL_RULES,
    COLD_PROMPT,
    COLD_RULE_COUNT,
    SHOP_PROMPT,
    TUNED_PROMPT,
    build_prompt,
    prompt_for,
    split_rules,
    with_experience,
)

__all__ = [
    "ALL_RULES",
    "COLD_PROMPT",
    "COLD_RULE_COUNT",
    "SHOP_PROMPT",
    "TUNED_PROMPT",
    "build_prompt",
    "prompt_for",
    "split_rules",
    "with_experience",
]
