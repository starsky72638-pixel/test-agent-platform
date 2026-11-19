"""会话槽位记忆 + 上下文视图裁剪。

为什么需要槽位，而不是继续靠 messages：
    多轮对话里模型只能翻 messages 找"刚才那个订单"。消息一多就必须裁，
    agent.py 每轮只把最近 keep_turns 轮发给模型——被裁掉那几轮里的订单号、
    预算、品类就彻底消失了，模型只能反问"您说的是哪个订单"。
    槽位是跨轮不丢的最小事实集：窗口裁到多小，这几个值都还在。

槽位存在哪里：
    挂在 history[0]（系统消息）的 additional_kwargs 上，随 history 一起传递。
    这样做的好处是调用方（main.py / replay.py）不需要额外参数，
    多轮之间天然接续；已确认 additional_kwargs 不会被发进请求体。

事实卡片与槽位的分工：
    槽位只放"下一次调用要用的参数"（订单号、预算、品类）；
    事实卡片放"查过哪些对象、各自是什么"（订单→商品、商品→名称）。
    两者都只解决"该查谁"，不解决"答案是多少"——金额、状态、库存必须重新调工具拿，
    否则模型会拿旧值当现状，评测也会把"背下来的答案"算成能力。

设计取舍：
    只记"能从工具调用实参反填"的字段，不从用户原话做自然语言抽取。
    参数结构化、零歧义、可离线单测；从"300 以内"写正则要处理"三百块"，
    错一次就污染整轮。
"""

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from core.context import render_slots as _core_render_slots
from core.context import update_slots as _core_update_slots

# 槽位在 additional_kwargs 里的键名
SLOTS_KEY = "slots"

# 事实表在 additional_kwargs 里的键名
FACTS_KEY = "facts"

# 事实表最多保留多少条。上限的作用不是省内存（几条 dict 无所谓），
# 而是防提示词无限膨胀：工具每查一次就加一条，聊二十轮后系统提示会盖过对话本身。
MAX_FACTS = 8

# 槽位名 -> 注入提示时给模型看的说明
SLOT_LABELS = {
    "budget_max": "用户给的预算上限（元）",
    "category": "用户要的品类或商品名",
    "last_order_id": "最近涉及的订单号",
    "last_product_id": "最近涉及的商品 id",
}

# (工具名, 实参名) -> 槽位名
# 注意：search_products 的关键词参数名是 q，不是 keyword（见 shop_skill.py:56），
# 写错的话 category 槽永远不会被填，属于静默失效。
_ARG_TO_SLOT = {
    ("search_products", "max_price"): "budget_max",
    ("search_products", "q"): "category",
    ("get_order", "order_id"): "last_order_id",
    ("get_logistics", "order_id"): "last_order_id",
    ("apply_refund", "order_id"): "last_order_id",
    ("get_product_detail", "product_id"): "last_product_id",
}


def update_slots(slots: dict, calls) -> dict:
    """从工具调用的实参反向填槽，原地修改并返回 slots。

    实现委托给 core.context：机制只有一份，两个 Agent 共用，
    这里只保留「哪个参数填哪个槽」这张业务表。
    """
    return _core_update_slots(slots, calls, _ARG_TO_SLOT)


def render_slots(slots: dict) -> str:
    """把槽位渲染成一段提示；全空时返回空字符串。

    渲染格式同样由 core.context 统一定义，
    避免两个 Agent 各自长出一套措辞、对比实验里多一个变量。
    """
    return _core_render_slots(slots, SLOT_LABELS)


# =========================
# 事实卡片：把工具返回里"以后还要用"的字段记下来
# =========================

def _parse_tool_output(output):
    """把工具返回的 "status=200\\nbody={json}" 拆成 (状态码, 解析后的对象)。

    解析不出对象时第二个值给 None，三种情况都算：
    - 超时、参数错误返回的是 "ERROR ..."，压根没有 status 行；
    - body 被 MAX_BODY 截断，JSON 不完整；
    - 状态码不是 200，body 是错误信息。
    """
    if not isinstance(output, str):
        return 0, None
    head, sep, body = output.partition("\nbody=")
    match = re.match(r"status=(\d+)", head.strip())
    if not match:
        return 0, None
    code = int(match.group(1))
    if not sep:
        return code, None
    try:
        return code, json.loads(body)
    except (TypeError, ValueError):
        return code, None


def _product_fact(item: dict) -> dict:
    """把一条商品记录转成事实表条目。"""
    product_id = item.get("id")
    if product_id is None:
        return {}
    return {
        f"product:{product_id}": {
            "商品名": item.get("title"),
            "价格": item.get("price"),
            "库存": item.get("stock"),
        }
    }


def extract_facts(tool_name: str, args: dict, output: str) -> dict:
    """从一次工具返回里抽出可以跨轮引用的字段，返回 {key: fact}。

    只认 status=200：404/422 的意思是"这个对象查不到"，
    记成事实会让下一轮换的那个对象背上上一轮的结论
    （提示词第 7 条专门禁了这件事）。
    搜索类工具一次返回多条商品，所以返回值是字典而不是单条。
    """
    code, data = _parse_tool_output(output)
    if code != 200 or not isinstance(data, dict):
        return {}

    if tool_name == "get_order":
        order_id = data.get("order_id")
        if order_id is None:
            return {}
        return {
            f"order:{order_id}": {
                "订单号": order_id,
                "订单状态": data.get("status"),
                "金额": data.get("amount"),
                "下单日期": data.get("created_at"),
                "商品id": data.get("product_id"),
            }
        }

    if tool_name == "get_logistics":
        order_id = data.get("order_id")
        if order_id is None:
            return {}
        # 和 get_order 共用同一个 key：两边字段不重名，合并后才是"这个订单的全貌"
        return {
            f"order:{order_id}": {
                "订单号": order_id,
                "承运商": data.get("carrier"),
                "运单号": data.get("tracking_no"),
                "物流状态": data.get("status"),
                "预计送达": data.get("eta"),
            }
        }

    if tool_name == "get_product_detail":
        return _product_fact(data)

    if tool_name == "search_products":
        facts = {}
        for item in data.get("items") or []:
            if isinstance(item, dict):
                facts.update(_product_fact(item))
        return facts

    if tool_name == "apply_refund":
        order_id = data.get("order_id")
        if order_id is None:
            return {}
        return {
            f"refund:{order_id}": {
                "订单号": order_id,
                "退款单": data.get("refund_id"),
                "退款状态": data.get("status"),
                "退款金额": data.get("amount"),
                "到账天数": data.get("eta_days"),
            }
        }

    return {}


def update_facts(facts: dict, tool_name: str, args: dict, output: str) -> dict:
    """把一次工具返回并进事实表，原地修改并返回。

    同 key 合并（订单信息和物流信息拼成一条），空值不覆盖已有值；
    合并时先删再插，借 dict 的插入顺序把它标成"最近用过"，
    超过上限就淘汰最早的条目。
    """
    for key, fact in extract_facts(tool_name, args, output).items():
        merged = dict(facts.get(key) or {})
        merged.update({k: v for k, v in fact.items() if v not in (None, "", [])})
        facts.pop(key, None)
        facts[key] = merged
    while len(facts) > MAX_FACTS:
        facts.pop(next(iter(facts)))
    return facts


def render_facts(facts: dict) -> str:
    """把事实表渲染成提示；空表返回空字符串，不占系统提示的位置。"""
    lines = []
    for fact in (facts or {}).values():
        body = "，".join(
            f"{k}：{v}" for k, v in fact.items() if v not in (None, "", [])
        )
        if body:
            lines.append(f"- {body}")
    if not lines:
        return ""
    return (
        "之前几轮查过的记录，只能当线索用——它的作用是你据此认出用户说的是哪个商品、哪个订单，"
        "不是替你回答。价格、金额、状态、库存、物流一律以最新的工具返回为准："
        "用户再问同一件事，仍要按规则调工具查一遍，不要拿这里的旧值当答案。"
        "和最新一次工具返回冲突时以最新为准。\n" + "\n".join(lines)
    )


def build_view(messages, system_prompt: str, slots: dict, keep_turns: int = 3,
               facts: dict | None = None):
    """把完整 history 压成发给模型的视图：系统提示（含槽位和事实）+ 最近 keep_turns 轮。

    只影响"发给模型的消息"，不改 history 本身——history 要完整保留，
    报告里的逐轮切片依赖它。

    按 HumanMessage 切轮：每一轮从一条用户消息开始，到最终回答结束。
    只在轮与轮之间切，不会把 AIMessage(tool_calls) 和它的 ToolMessage 拆开。
    """
    notes = [render_slots(slots), render_facts(facts or {})]
    note = "\n\n".join(n for n in notes if n)
    head = SystemMessage(system_prompt + ("\n\n" + note if note else ""))

    rest = (
        messages[1:]
        if messages and isinstance(messages[0], SystemMessage)
        else list(messages)
    )

    groups: list[list] = []
    for msg in rest:
        if isinstance(msg, HumanMessage) or not groups:
            groups.append([msg])
        else:
            groups[-1].append(msg)

    if keep_turns and len(groups) > keep_turns:
        groups = groups[-keep_turns:]

    return [head] + [m for g in groups for m in g]


def load_slots(messages) -> dict:
    """从会话系统消息里取回槽位；首次对话时返回空字典。"""
    if messages and isinstance(messages[0], SystemMessage):
        return dict(messages[0].additional_kwargs.get(SLOTS_KEY) or {})
    return {}


def save_slots(messages, slots: dict) -> None:
    """把槽位写回会话系统消息，跟着 history 一起传给下一轮。"""
    if messages and isinstance(messages[0], SystemMessage):
        messages[0].additional_kwargs[SLOTS_KEY] = dict(slots)


def load_facts(messages) -> dict:
    """从会话系统消息里取回事实表；首次对话时返回空字典。"""
    if messages and isinstance(messages[0], SystemMessage):
        return dict(messages[0].additional_kwargs.get(FACTS_KEY) or {})
    return {}


def save_facts(messages, facts: dict) -> None:
    """把事实表写回会话系统消息，和槽位放在一起随 history 传递。"""
    if messages and isinstance(messages[0], SystemMessage):
        messages[0].additional_kwargs[FACTS_KEY] = dict(facts)
