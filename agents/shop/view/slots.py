"""shop 的槽位规则：哪个工具的哪个参数，填进哪个槽。

机制在 core/context.py，这张表是业务的一半，所以留在 Agent 层。
换一个电商以外的 Agent 只需要换这张表和 labels，core 不用动。

注意 search_products 的关键词参数名是 q，不是 keyword。
写错的话 category 槽永远不会被填，是静默失效——所以下面有单测钉住它。
"""

from __future__ import annotations

# (工具名, 实参名) -> 槽位名
ARG_TO_SLOT = {
    ("search_products", "max_price"): "budget_max",
    ("search_products", "q"): "category",
    ("get_order", "order_id"): "last_order_id",
    ("get_logistics", "order_id"): "last_order_id",
    ("apply_refund", "order_id"): "last_order_id",
    ("get_product_detail", "product_id"): "last_product_id",
}

# 槽位名 -> 渲染给模型看的说明
SLOT_LABELS = {
    "budget_max": "用户给的预算上限（元）",
    "category": "用户要的品类或商品名",
    "last_order_id": "最近涉及的订单号",
    "last_product_id": "最近涉及的商品 id",
}

__all__ = ["ARG_TO_SLOT", "SLOT_LABELS"]
