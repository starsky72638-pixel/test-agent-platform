"""售后客服 Agent：订单 / 物流 / 退款三件事。

形状与 agents/shop 同构，但代码完全独立——删掉 agents/shop 整个目录，
这个包照样 import、照样跑、照样过单测（由 tests/test_customer_agent.py 守着）。

    tools.py   能调哪 3 个接口
    agent.py   循环怎么转、轨迹怎么记、模型看到什么提示词

没有再分 core/ 与 view/ 两个子目录：客服只有一套提示词、一个循环，
拆成四个文件只会让人多点开三次。等这里真的出现"要对比两套提示词"时再拆。

对外只有一个入口：make_customer_agent()。
"""

from agents.customer.agent import COLD_PROMPT, CustomerAgent, calls_from_messages, run_turn
from agents.customer.tools import (
    COLD_TOOLS,
    CUSTOMER_TOOLS,
    allowed_tool_names,
    cold_tools,
    customer_tools,
)


def make_customer_agent(**kwargs) -> CustomerAgent:
    """按名字造一个 CustomerAgent。kwargs 原样传给构造函数。

    不做 variant 参数：客服只有一套提示词，没有"cold / tuned"那种对照需求。
    真要做对照时，注入 prompt= 即可，不必改工厂签名。
    """
    return CustomerAgent(**kwargs)


__all__ = [
    "CustomerAgent",
    "make_customer_agent",
    "COLD_PROMPT",
    "CUSTOMER_TOOLS",
    "COLD_TOOLS",
    "customer_tools",
    "cold_tools",
    "allowed_tool_names",
    "calls_from_messages",
    "run_turn",
]
