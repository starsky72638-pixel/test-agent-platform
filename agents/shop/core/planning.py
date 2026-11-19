"""结构化计划层：先列待办，再执行，收尾前检查有没有漏项。

为什么要有这一层：
    纯 ReAct 只有"下一步调什么"，没有"这一轮一共要做几件事"。
    用户一次问两件事（M08 第 5 轮"那两个订单加起来一共多少钱"），
    模型查完一个就作答，判定写 missing_tool，而人看那段回答像是答完了。

这一层做三件事，都是能被观察到的执行逻辑，不是提示词里的一句要求：

    1. 进循环前先拆待办：让模型输出结构化 JSON（工具名限定在真实工具集内）；
    2. 待办渲染进 system prompt，每次决策都看得到还剩几项；
    3. 模型想收尾时，若待办里还有没被任何一次调用抵掉的项，
       就把它顶回去继续做——最多顶一次，避免和模型无限拉锯。

待办与调用的对应关系按"一个调用只能抵一条待办"算：
同一个工具要查两个对象，必须调两次；查一次只抵得掉一条。
这条规则是 M08 那种"两个订单"场景能不能被兜住的关键，
如果按工具名计数，get_order 调过一次就算覆盖，兜底等于没写。

失败必须安静：模型不给 JSON、给了非法 JSON、或者列了不存在的工具，
一律当成"没有计划"，退回原来的 ReAct 行为。
计划是增强，不是新的崩溃点。
"""

from __future__ import annotations

import json
import os
import re

# 计划开关。调成 0 就退化成改动前的行为，对照实验只切这一个变量。
PLAN_ENV = "SHOP_PLAN"

# 待办里能区分对象的那几个参数。其余参数（比如 sort）不参与匹配，
# 否则模型把 sort 写成 "price" 还是 "价格" 都会让一条待办永远匹配不上。
KEY_ARGS = ("order_id", "product_id", "q", "keyword", "category", "max_price")

# 计划最多写几条。超过这个数说明模型在罗列，不是真拆任务。
MAX_ITEMS = 5

# 最多顶回去几次。1 次足够覆盖"漏了第二件事"，再多就是在替模型做任务。
MAX_NUDGES = 1

# 只有用户这一句里出现并列/汇总信号时才拆计划。
#
# 为什么必须加这道门：S08 的输入是"1002 现在什么状态"——一个并列词都没有，
# 但它隐含需要查两次（订单状态 + 物流状态），原来靠提示词规则 8 兜住。
# 一旦对它开计划，模型只把"查订单"写进清单，答完就收尾，
# 反而把规则 8 覆盖掉了。实测 S08、S13 都是这么从 PASS 掉成 missing_tool 的。
#
# 计划层的定位是"多意图别漏"，不是"所有任务都先写清单"。
# 「还有」不在表里：语料里它只出现在 S04「智能手表 S2 还有货吗」，
# 那里"还有货"是动词短语，不是并列。列进来等于白花一次计划调用。
# 宁可漏拆（退回 ReAct）也不能多拆——多拆的代价已经用 S08/S13/H12 付过一次了。
MULTI_MARKERS = (
    "各", "分别", "两个", "两件", "两笔", "三种", "三个",
    "加起来", "一共", "总共", "合计",
    "以及", "顺便", "、", "和",
)

# 解析失败时最多再问一次。
#
# 实测 M08 第 5 轮连跑 6 次：3 次拆出 2 项待办，3 次拆不出，
# 而通过与否和"拆没拆出来"6/6 完全吻合（有计划 3/3 PASS，没计划 0/3 PASS）。
# 失败有三种形态，都不是模型"判断这题不用拆"，而是输出格式跑偏：
#   1. {"steps": []}                    直接给空
#   2. {"steps":[{"action":...,"params":...}]}  键名不按 schema
#   3. 回了一大段分析文字，压根没有 JSON
# 第 2 类靠下面的键名别名就地认掉，不花额外调用；
# 第 1、3 类只能重发一次，把上次的错回复带回去再要求一遍。
# 只重试一次：两次都拆不出来的题，再多问就是在替模型做任务。
PLAN_RETRIES = 1

# 键名别名。模型经常把 args 写成 params/arguments，把 tool 写成 name/function。
# 这类偏差不该用一次额外的模型往返来修。
_ARG_ALIASES = ("args", "params", "arguments", "parameters")
_TOOL_ALIASES = ("tool", "tool_name", "name", "function")


def needed(question: str) -> bool:
    """这一句是不是多意图，需不需要拆计划。

    只看字面信号，不调模型：判断本身必须便宜、确定、可复现。
    认不出来的就当成单意图——宁可少拆一次，也不能替模型把任务缩小。
    """
    text = str(question or "")
    return any(mark in text for mark in MULTI_MARKERS)

PLAN_INSTRUCTION = """先把这一轮要做的事列成待办清单，再动手做。

只输出一个 JSON 对象，不要解释，不要调用工具：
{{"steps": [{{"goal": "这件事要查什么（一句话）", "tool": "工具名", "args": {{"参数": "值"}}}}]}}

规则：
- 用户一次问了几件事，就拆成几条；只有一件事就写一条。
- tool 只能从这些里选：{tools}
- args 只填能确定的值，不确定就写空对象。
- 同一件事要查两个对象（比如两个订单），必须写成两条，各带自己的 order_id。
- 实在拆不出来就输出 {{"steps": []}}。"""

_PLAN_HEADER = (
    "\n\n━━ 本轮待办（最低要求，不是全部）━━\n"
    "清单没列到、但用户隐含需要的信息，照样要查；系统提示词里的规则仍然适用。"
)

PLAN_RETRY_HINT = """上面的回复不能当待办用。只要一个 JSON 对象，键名固定是 steps / tool / args。

现在只输出 JSON，不要任何解释：
{{"steps": [{{"goal": "要查什么", "tool": "工具名", "args": {{"参数": "值"}}}}]}}

tool 只能从这些里选：{tools}
要查两个不同对象（比如两个订单）就必须写成两条，各带自己的 order_id。"""

_REMINDER = (
    "待办里还有没做的：{items}。"
    "先用工具把这{count}项查完，再回答用户；不要把没查的部分当成查过了。"
)


def enabled() -> bool:
    """读开关。只有明确写成 0/false/no 才算关闭。"""
    raw = (os.getenv(PLAN_ENV) or "").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _first_json(text: str) -> dict | None:
    """从模型输出里取出第一个 JSON 对象。

    模型经常包一层 ```json，或者在前后补一句人话，
    所以先按代码块剥一次，再从第一个 { 扫到配平的 }。
    """
    body = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", body, re.S)
    if fenced:
        body = fenced.group(1)

    start = body.find("{")
    if start < 0:
        return None

    depth = 0
    for i in range(start, len(body)):
        ch = body[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(body[start : i + 1])
                except json.JSONDecodeError:
                    return None
                return obj if isinstance(obj, dict) else None
    return None


def _first_text(item: dict, keys) -> str:
    """按别名顺序取第一个非空字符串。取不到返回空串。"""
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _first_dict(item: dict, keys) -> dict:
    """按别名顺序取第一个 dict。取不到返回空 dict。"""
    for key in keys:
        value = item.get(key)
        if isinstance(value, dict):
            return dict(value)
    return {}


def parse_plan(text: str, allowed: set[str] | None = None) -> list[dict]:
    """把模型输出解析成待办列表。解析不出来就返回空列表，不抛异常。"""
    obj = _first_json(text)
    if not obj:
        return []

    raw_steps = obj.get("steps")
    if not isinstance(raw_steps, list):
        return []

    steps: list[dict] = []
    for item in raw_steps:
        if not isinstance(item, dict):
            continue
        goal = str(item.get("goal") or item.get("desc") or "").strip()
        tool = _first_text(item, _TOOL_ALIASES)
        args = _first_dict(item, _ARG_ALIASES)
        # 认不出的工具名直接丢掉：留着会让"覆盖"永远算不成，
        # 结果是每条待办都漏，兜底逻辑把模型顶回去做一件不存在的事。
        if allowed is not None and tool and tool not in allowed:
            continue
        # 没有工具名的待办一律丢掉，两种坏结果都不划算：
        # 一是 tool 为空时任何一次调用都算抵掉它（假覆盖，等于没兜底），
        # 二是永远匹配不上（永久漏项，白顶模型一次）。
        if not tool:
            continue
        if not goal:
            goal = tool
        steps.append({"goal": goal, "tool": tool, "args": args})
        if len(steps) >= MAX_ITEMS:
            break
    return steps


def render_plan(steps: list[dict]) -> str:
    """把待办渲染成 system prompt 末尾的一段。空计划返回空串。"""
    if not steps:
        return ""
    lines = [_PLAN_HEADER]
    for i, step in enumerate(steps, 1):
        tool = step.get("tool") or "（未指定工具）"
        args = step.get("args") or {}
        arg_text = "，".join(f"{k}={v}" for k, v in args.items())
        suffix = f"（参数：{arg_text}）" if arg_text else ""
        lines.append(f"{i}. {step.get('goal') or '（未命名）'} → {tool}{suffix}")
    return "\n".join(lines)


def _same(a, b) -> bool:
    """参数比对：数字与字符串混着来也要判等（1001 vs "1001"）。"""
    return str(a).strip().lower() == str(b).strip().lower()


def _matches(step: dict, call: dict) -> bool:
    """一次调用能不能抵掉这条待办。"""
    tool = step.get("tool") or ""
    if tool and tool != call.get("name"):
        return False

    step_args = step.get("args") or {}
    call_args = call.get("args") or {}
    for key in KEY_ARGS:
        if key in step_args and not _same(step_args[key], call_args.get(key, "\0")):
            return False
    return True


def uncovered(steps: list[dict], calls: list[dict]) -> list[dict]:
    """还差哪几条。一个调用只能抵一条待办，所以先匹配到的先消耗掉。"""
    if not steps:
        return []
    pool = [dict(c) for c in calls if c.get("name")]
    missed: list[dict] = []
    for step in steps:
        hit = next((i for i, call in enumerate(pool) if _matches(step, call)), None)
        if hit is None:
            missed.append(step)
        else:
            pool.pop(hit)
    return missed


def make_plan(
    llm,
    system_prompt: str,
    question: str,
    tools,
    history=None,
    report: dict | None = None,
) -> list[dict]:
    """让模型先把这一轮拆成待办。

    历史只带最近几条：计划只针对本轮，不需要整段对话，
    而且计划调用本身不该成为一笔大的额外开销。

    report 是出参，用来把"为什么没有计划"讲清楚。
    没有它的话，「空计划」会有两种完全不同的原因混在一起：
    (a) 这道题不需要拆（门没开，压根没调模型）；
    (b) 门开了但模型没给出可解析的 JSON / 调用本身失败。
    这两件事在实验里的含义完全相反——(a) 是设计，(b) 是缺陷。
    实际观测到过 M08 第 5 轮同一题干两次运行一次有计划一次没有，
    没有 report 就只能干看着，无法判断是不是计划调用静默失败。
    """
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    names = [getattr(t, "name", "") for t in (tools or [])]
    allowed = {n for n in names if n}
    tool_list = "、".join(sorted(allowed))

    if report is not None:
        report["called"] = True

    base = [SystemMessage(system_prompt)]
    if history:
        base.extend(list(history)[-6:])
    base.append(
        HumanMessage(PLAN_INSTRUCTION.format(tools=tool_list) + "\n\n用户这一轮说：" + question)
    )

    attempts = 0
    last_error = "empty"
    messages = list(base)

    while True:
        attempts += 1
        try:
            raw = llm.invoke(messages)
        except Exception as exc:
            # 计划调用自己失败（限流、超时）不该让整条任务崩掉，
            # 但必须留下痕迹，否则和"这道题不需要计划"分不开。
            if report is not None:
                report["error"] = f"invoke:{type(exc).__name__}"
                report["attempts"] = attempts
            return []

        text = str(getattr(raw, "content", "") or "")
        steps = parse_plan(text, allowed)
        if steps:
            if report is not None:
                report["attempts"] = attempts
            return steps

        # 模型回了话但拆不出待办：把原文前 60 字带上，
        # 事后能区分「回了空的 steps」和「回了一段解释性文字」。
        last_error = "no_steps:" + " ".join(text.split())[:60]
        if attempts > PLAN_RETRIES:
            break

        # 把上次的错回复原样带回去，再明确一遍键名。
        # 空 steps 和"跑题成分析文字"这两类，重发一次基本都能纠回来。
        messages = list(base) + [
            AIMessage(content=text or "(空)"),
            HumanMessage(PLAN_RETRY_HINT.format(tools=tool_list)),
        ]

    if report is not None:
        report["error"] = last_error
        report["attempts"] = attempts
    return []


def reminder(steps: list[dict]) -> str:
    """把未完成的待办写成一句催促，回灌给模型。"""
    items = "；".join(
        f"第{i}项 {s.get('goal') or s.get('tool')}" for i, s in enumerate(steps, 1)
    )
    return _REMINDER.format(items=items, count=len(steps))


__all__ = [
    "KEY_ARGS",
    "MAX_ITEMS",
    "MAX_NUDGES",
    "MULTI_MARKERS",
    "PLAN_ENV",
    "PLAN_INSTRUCTION",
    "PLAN_RETRIES",
    "PLAN_RETRY_HINT",
    "enabled",
    "make_plan",
    "needed",
    "parse_plan",
    "reminder",
    "render_plan",
    "uncovered",
]
