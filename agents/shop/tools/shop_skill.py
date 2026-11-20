"""电商客服的 5 个 skill：售前 2 个 + 售后 3 个。

这一份属于 agents/shop 自己，不从 main_agent 导入。
两个 Agent 各有一份指向同一后端的工具定义，是刻意接受的重复：
    agents/main_agent 是被历史报告验证过的老工程，它随时可能被删；
    agents/shop 要能在它消失之后照常跑，就不能在 import 上依赖它。

为什么拆成 5 个语义化工具而不是 1 个万能 http_request：
    评测里有一层叫"工具选择正确性"。只有一个万能工具时，
    模型永远不会选错，这一层直接失效——"该查物流却查了订单"
    这种错误根本无从暴露。工具必须语义化，错误才有形状。

约束：返回字符串、永不抛异常、status 单独成行。
"""

from __future__ import annotations

import httpx
from langchain_core.tools import tool

from config.settings import SERVICE_BASE

# 服务地址的单一来源是 config.settings.SERVICE_BASE。
# 保留 SHOP_BASE 这个名字，是因为 tools/__init__.py 的 __all__ 与脚本都在用它，
# 改名会扩散到 5 个文件，而收益只是"名字更好看"。
SHOP_BASE = SERVICE_BASE
TIMEOUT = 5.0
MAX_BODY = 800


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
    # 参数缺失属于调用方错误，本地就拦下来，
    # 不要发出去让服务端把"缺参数"和"重复申请"混成同一个错误码。
    reason = (reason or "").strip()
    if not reason:
        return "ERROR 参数错误：reason 必填，请用用户原话概括退款原因"

    return _request("POST", "/refund", body={"order_id": int(order_id), "reason": reason})
