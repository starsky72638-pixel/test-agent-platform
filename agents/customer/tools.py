"""售后客服的 3 个 skill：查订单、查物流、申请退款。

为什么不复用 agents/shop/tools：
    与 agents/main_agent/tools/shop_skill.py 同一个理由——两个 Agent 各持一份
    指向同一后端的工具定义，是刻意接受的重复。客服 Agent 要能在 shop 包
    被删掉之后照常跑，就不能在 import 上依赖它。

为什么只要 3 个而不是 5 个：
    售前那两个（搜索、详情）属于导购场景。客服只处理"我的订单怎么了"。
    多绑一个用不上的工具，就多一次模型选错的机会，也会把"该查物流却查了商品"
    这类错误掩盖掉——它压根调不到搜索。

约束（与 shop 工具一致，判分器按这个格式读 status）：
    返回字符串、永不抛异常、status 单独成行。
"""

from __future__ import annotations

import httpx
from langchain_core.tools import tool

from config.settings import SERVICE_BASE

TIMEOUT = 5.0
MAX_BODY = 800


def _request(method: str, path: str, params: dict | None = None, body: dict | None = None) -> str:
    url = f"{SERVICE_BASE}{path}"
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

    调用前先 get_order 确认订单状态：待支付订单申请退款会被服务端拒绝（422）。

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


# 顺序固定：提示词里的处理次序与这里一致（先查状态，再查物流，最后才动退款）。
CUSTOMER_TOOLS = (
    get_order,
    get_logistics,
    apply_refund,
)


def customer_tools() -> list:
    """本次绑定给模型的工具列表。每次返回新 list，避免调用方原地改到共享元组。"""
    return list(CUSTOMER_TOOLS)


# -------------------------
# 冷启动对照用的"初稿工具"
# -------------------------
# 为什么需要这一份：
#   行为约束不只写在 prompt 里，也写在工具描述里。apply_refund 的
#   "调用前先 get_order 确认订单状态"就是一条完整的业务规则，而工具描述
#   每一轮都会随 schema 进上下文。上一轮只删 prompt 的 cold 变体因此
#   仍然拿到全部约束，6/6 全过——那个 null 是自变量没删干净造成的，
#   不是"模型太强"。这一份把描述也降到初稿水平：一个开发者照着接口
#   随手写了三个工具、没写文档，这是真实的起点。
#
# 边界：只换描述。函数体、参数签名、返回格式全部复用上面三个，
#   否则"注入后通过"就分不清是经验带来的还是工具行为变了带来的。
COLD_DESCRIPTIONS = {
    "get_order": "查订单。",
    "get_logistics": "查物流。",
    "apply_refund": "提交退款申请。",
}


def _terse(original):
    """同名、同参数、同函数体，只把描述换成一句话。"""
    clone = tool(original.func)
    clone.name = original.name
    clone.description = COLD_DESCRIPTIONS[original.name]
    return clone


COLD_TOOLS = tuple(_terse(t) for t in CUSTOMER_TOOLS)


def cold_tools() -> list:
    """冷变体绑定的工具列表。同样每次返回新 list。"""
    return list(COLD_TOOLS)


def allowed_tool_names() -> set[str]:
    """允许出现的工具名。判分的 allowed_tools 参数用它。

    从工具对象自己取名字，不另写一份字符串表——手写白名单最容易在改名之后忘记同步。
    """
    return {t.name for t in CUSTOMER_TOOLS}


__all__ = [
    "CUSTOMER_TOOLS",
    "customer_tools",
    "allowed_tool_names",
    "COLD_TOOLS",
    "COLD_DESCRIPTIONS",
    "cold_tools",
]
