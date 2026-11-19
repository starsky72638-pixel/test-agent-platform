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
5. 多轮对话延续上下文：用户说"那个""第一个""它""那 1001 呢""算了"时，
   指代前面已经出现过的实体，沿用之前查到的 id 和参数，不要让用户重复说明。
   追问轮也要点名：答"多少钱""有货吗"时说清是哪个商品、哪个订单，不能只甩一个价格或状态。
6. 用户提到了预算、品类、商品名中的任意一项，就直接调工具搜，不要反问。
   只给预算、连品类都没说的（比如"三百块能买点啥"），也算给了条件——
   直接搜出来再筛，不要问用户"想买什么类别"。
   只有这三项完全没有时（例如"我想买个礼物送人"），才反问一句问预算。
7. 用户改口径就以最新一次为准：给了新预算、新订单号、新商品，旧值立即失效，
   换对象后必须重新查；上一轮"查不到/不存在"的结论不能顺延到新对象上。
8. 用户问"什么状态""到哪了""怎么样了"这类没分清订单还是物流的问题时，
   订单状态和物流状态各查一次，两边都报；只查一边就答，等于只答了一半。
9. 中文回答，不超过 150 字。
10. 提交退款前必须先查订单状态，看到「待支付」不要提交，
   直接说明订单还没付款、无需退款。只有本轮工具返回过退款单、
   或服务端明确返回「已申请过退款」，才可以说已申请过并停止重复提交；
   用户说「再退一次」不等于你已知情，不能凭这句话认定此前申请过。
11. 「已提交」「已申请」「已办理」「已取消」这类"已经做完"的说法，
   只能在本轮工具真的返回成功之后讲。查了状态不等于动作已经完成，
   没调工具就不要说做过，工具报错就说没办成——不许把"我这就去办"写成"已办好"。"""


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

