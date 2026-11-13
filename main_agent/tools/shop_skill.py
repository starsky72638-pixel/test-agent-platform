"""电商客服工具集：售前 2 个 + 售后 3 个，共 5 个 skill。

为什么不复用通用的 http_request：
  Agent 评测里有一条 reward 叫"工具选择正确性"。只有一个万能工具时，
  模型永远不会选错，这个维度直接消失。拆成语义化工具后，
  "该查物流却去查订单"才能被记成一次错误轨迹。

约束与 http_tool 一致：返回字符串、永不抛异常、status 单独成行。
"""

from __future__ import annotations

import os

import httpx
from langchain_core.tools import tool

SHOP_BASE = (os.getenv("SHOP_BASE_URL") or "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = 5.0
MAX_BODY = 800

SHOP_SYSTEM_PROMPT = """你是电商平台客服助手，只能依据工具返回的数据回答。

规则：
1. 涉及商品、订单、物流、退款的事实，必须先调工具查，不许凭印象作答。
2. 工具返回什么就用什么。字段为 null、空列表、压根没有该字段，就如实说"暂无"，
   禁止推测、补全或给出示例值。
3. 状态值原样复述：订单状态是中文（已签收/已发货/待支付），
   物流状态是英文（delivered/shipped/pending），不要翻译或改写。
4. 先说查了什么、查到什么，再给结论。查不到就说查不到，不要换个接口瞎猜。
5. 中文回答，不超过 150 字。"""


def _request(method: str, path: str, params: dict | None = None, body: dict | None = None) -> str:
    url = f"{SHOP_BASE}{path}"
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            resp = client.request(method, url, params=params or None, json=body)
    except httpx.TimeoutException:
        return f"ERROR 请求超时（超过 {TIMEOUT}s）：{url}"
    except httpx.HTTPError as exc:
        return f"ERROR 请求失败 {type(exc).__name__}: {exc}"

    text = resp.text or ""
    if len(text) > MAX_BODY:
        text = text[:MAX_BODY] + f" ...(已截断，原始长度 {len(resp.text)} 字符)"
    return f"status={resp.status_code}\nbody={text}"


@tool
def search_products(q: str = "", min_price: float | None = None, max_price: float | None = None) -> str:
    """按关键词和价格区间搜索商品，返回精简列表（id / 标题 / 价格 / 库存 / 评分），按评分从高到低。

    Args:
        q: 关键词，空字符串表示不按关键词过滤。会匹配标题、适用场景、描述。
        min_price: 最低价（含），None 表示不限。
        max_price: 最高价（含），None 表示不限。
    """
    params: dict = {}
    if q:
        params["q"] = q
    if min_price is not None:
        params["min_price"] = min_price
    if max_price is not None:
        params["max_price"] = max_price
    return _request("GET", "/products/search", params=params)


@tool
def get_product_detail(product_id: int) -> str:
    """查单个商品的完整信息，含规格、适用场景、描述、库存。

    Args:
        product_id: 商品 id，正整数。只在 search_products 已给出 id 后使用。
    """
    return _request("GET", f"/products/{int(product_id)}")


@tool
def get_order(order_id: int) -> str:
    """查订单状态、金额、下单日期。不包含物流轨迹。

    Args:
        order_id: 订单号，如 1001 / 1002 / 1003。
    """
    return _request("GET", f"/orders/{int(order_id)}")


@tool
def get_logistics(order_id: int) -> str:
    """查订单对应的物流：承运商、运单号、物流状态、预计送达时间、轨迹明细。

    Args:
        order_id: 订单号。物流按订单号查，不是运单号。
    """
    return _request("GET", f"/logistics/{int(order_id)}")


@tool
def apply_refund(order_id: int, reason: str) -> str:
    """为订单提交退款申请。仅已支付订单可退；同一订单重复申请会返回 409。

    Args:
        order_id: 要退款的订单号。
        reason: 退款原因，必填，用用户的原话概括。
    """
    return _request("POST", "/refund", body={"order_id": int(order_id), "reason": reason})
