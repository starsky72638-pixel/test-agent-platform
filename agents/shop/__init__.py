"""电商客服 Agent，一套完整实现在这个目录里。

形状与 agents/main_agent 同构，但代码完全独立——删掉 agents/main_agent，
这个包照样 import、照样跑、照样过单测。

    core/agent.py        循环怎么转、轨迹怎么记
    tools/shop_skill.py  能调哪 5 个接口
    view/context.py      模型看到什么提示词
    llm.py               用哪个模型、什么参数

分三个子目录而不是平铺四个文件，是为了回答"这次对比实验到底改了什么"：
改了提示词就是 view，换了模型就是 llm，加了 skill 就是 tools，
不用读一个几百行的文件去翻。

对外只有一个入口：make_shop_agent("tuned" | "cold")。
"""

from agents.shop.core import planning
from agents.shop.core.agent import ShopAgent, calls_from_messages, run_loop
from agents.shop.tools import allowed_tool_names, shop_tools
from agents.shop.view.context import (
    COLD_PROMPT,
    SHOP_PROMPT,
    TUNED_PROMPT,
    prompt_for,
    with_experience,
)

VARIANTS = ("tuned", "cold")


def make_shop_agent(variant: str = "tuned", **kwargs) -> ShopAgent:
    """按变体造一个 ShopAgent。

    name 默认带上变体名，这样同一个注册表里放 cold 和 tuned 时，
    轨迹上的 agent 字段能直接区分是谁跑的，事后筛选不用再翻 meta。

    计划开关由 SHOP_PLAN 环境变量决定（默认开，写 0 关），
    和 SHOP_RETRY_LIMIT 同一套做法：对照实验只切环境变量，不回滚代码。
    """
    kwargs.setdefault("name", f"shop-{variant}")
    kwargs.setdefault("plan", planning.enabled())
    return ShopAgent(variant, **kwargs)


__all__ = [
    "ShopAgent",
    "make_shop_agent",
    "VARIANTS",
    "SHOP_PROMPT",
    "COLD_PROMPT",
    "TUNED_PROMPT",
    "prompt_for",
    "with_experience",
    "shop_tools",
    "allowed_tool_names",
    "calls_from_messages",
    "run_loop",
]
