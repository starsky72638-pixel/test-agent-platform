"""工具层。"""

from main_agent.tools.registry import ToolRegistry, get_registry
from main_agent.tools.http_tool import http_request
from main_agent.tools.shop_skill import (
    SHOP_SYSTEM_PROMPT,
    apply_refund,
    get_logistics,
    get_order,
    get_product_detail,
    search_products,
)

__all__ = ["ToolRegistry", "get_registry", "register_default_tools","SHOP_SYSTEM_PROMPT"]
_SHOPPING_TOOLS = (search_products, get_product_detail, get_order, get_logistics, apply_refund)

def register_default_tools() -> None:
    """显式注册内置工具。由 main 调用，避免 import 时产生副作用。"""
    registry=get_registry()
    registry.register(http_request)
    for tool in _SHOPPING_TOOLS:
        registry.register(tool)
