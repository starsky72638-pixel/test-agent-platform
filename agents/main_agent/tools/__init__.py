"""工具层：5 个电商 skill 的注册入口。

为什么拆成 5 个语义化工具而不是 1 个万能 http_request：
  评测里有一层 reward 叫"工具选择正确性"。只有一个万能工具时，
  模型永远不会选错，这一层直接失效——"该查物流却查了订单"这种错误
  根本无从暴露。工具必须语义化，错误才有形状。
"""

from agents.main_agent.tools.registry import ToolRegistry, get_registry
from agents.main_agent.tools.shop_skill import (
    SHOP_SYSTEM_PROMPT,
    apply_refund,
    get_logistics,
    get_order,
    get_product_detail,
    search_products,
)

__all__ = [
    "ToolRegistry",
    "get_registry",
    "register_default_tools",
    "SHOP_SYSTEM_PROMPT",
]

_SHOPPING_TOOLS = (search_products, get_product_detail, get_order, get_logistics, apply_refund)


def register_default_tools() -> None:
    """显式注册内置工具。由入口调用，避免 import 时产生副作用。"""
    registry = get_registry()
    for tool in _SHOPPING_TOOLS:
        registry.register(tool)
