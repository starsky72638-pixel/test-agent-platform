"""工具层：这个 Agent 能用哪些 skill。

装配和查询都在这里，工具定义在 shop_skill.py。
分两个文件的理由：装配方式（绑几个、白名单怎么算）是会随评测需求变的，
工具本身（怎么调后端）很少变，混在一起改一处要看两件事。

为什么不走全局 ToolRegistry：
    registry.register() 是 append，不幂等。
    同一进程里先后构造 cold 和 tuned 两个 Agent，注册两遍会得到 10 个重名工具。
    这里自己持有一份元组，各 Agent 各拿一份 list，互不影响，
    也让"这个 Agent 到底绑了哪些工具"变成能直接读的事实。
"""

from __future__ import annotations

from agents.shop.tools.shop_skill import (
    SHOP_BASE,
    apply_refund,
    get_logistics,
    get_order,
    get_product_detail,
    search_products,
)

# 顺序固定。判分的 allowed_tools 从同一份元组派生，两边不可能不一致。
SHOP_TOOLS = (
    search_products,
    get_product_detail,
    get_order,
    get_logistics,
    apply_refund,
)


def shop_tools() -> list:
    """本次绑定给模型的工具列表。每次返回新 list，避免调用方原地改到共享元组。"""
    return list(SHOP_TOOLS)


def allowed_tool_names() -> set[str]:
    """允许出现的工具名。判分的 allowed_tools 参数用它。

    从工具对象自己取名字，不另写一份字符串表——手写白名单最容易在改名之后忘记同步，
    结果是正确调用被判成 invalid_tool。
    """
    return {t.name for t in SHOP_TOOLS}


__all__ = ["SHOP_TOOLS", "SHOP_BASE", "shop_tools", "allowed_tool_names"]
