"""从用户输入里抽结构化参数，供节点模板 `{{extract.<字段>}}` 引用。

为什么用规则而不是模型：
    抽取要做的事只有一件——把用户已经说出口的值搬到正确位置。
    正则对同一句话永远给出同一个结果，评测才有可比性；
    接一次模型抽取，同一道题两次可能不一样，通过率的变化就分不清
    是流程改了还是抽签抽到了好结果。

为什么抽不到要抛错，而不是返回空串：
    返回空串会让工具带着 order_id="" 去打接口，拿到 404，
    最后记成"模型答错了"。抛错会落到 infra_error，评测阶段当链路故障
    剔除——失败出现在该出现的地方，才查得动。

为什么不支持从节点输出里抽：
    节点输出是工具返回的原始报文，要抽就得先约定格式。
    第一版只抽用户输入，够用且边界清楚；需要更多就走 memory/condition 节点。
"""

from __future__ import annotations

import re


class ExtractError(ValueError):
    """输入里找不到要抽的字段，或字段名不在白名单里。"""


_ORDER_IN_CONTEXT = re.compile(
    r"(?:订单|单号|order)\s*(?:号|id)?\s*[:：]?\s*(?<!\d)(\d{4})(?!\d)",
    re.IGNORECASE,
)
_ORDER_ANY_4 = re.compile(r"(?<!\d)(\d{4})(?!\d)")
_AMOUNT = re.compile(r"(\d+(?:\.\d+)?)\s*(?:元|块|rmb)", re.IGNORECASE)
_PRODUCT_ID = re.compile(r"(?<![A-Za-z0-9])([Pp]\d{2,6})(?![A-Za-z0-9])")

# 字段名 -> (按顺序尝试的正则, 抽不到时的提示)
# 顺序有意义：先带上下文的（"订单 1001"），再退到裸数字。
# 反例是"预算 1001 元"这种既有金额又有四位数金额的句子——
# 裸数字规则会把金额当订单号，所以上下文规则必须排在前面。
_RULES: dict[str, tuple[tuple[re.Pattern, ...], str]] = {
    "order_id": (
        (_ORDER_IN_CONTEXT, _ORDER_ANY_4),
        "输入里没有 4 位订单号",
    ),
    "amount": ((_AMOUNT,), "输入里没有以元为单位的金额"),
    "product_id": ((_PRODUCT_ID,), "输入里没有形如 P1001 的商品编号"),
}

EXTRACT_FIELDS = tuple(_RULES)


def extract_field(name: str, text: str) -> str:
    """从 text 抽出 name 对应的值。抽不到抛 ExtractError，绝不返回空串。"""
    key = str(name or "").strip()
    rule = _RULES.get(key)
    if rule is None:
        raise ExtractError(
            f"未知抽取字段 {key!r}，可用：{list(EXTRACT_FIELDS)}"
        )

    patterns, hint = rule
    body = str(text or "")
    for pattern in patterns:
        match = pattern.search(body)
        if match:
            return match.group(1)
    raise ExtractError(f"抽不到 {key}：{hint}")
