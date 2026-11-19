"""核心层：ReAct 循环本体，以及把它接进 Runtime 的适配。

这个文件回答两个问题：
    1. 一轮对话里，模型和工具怎么来回（run_loop）；
    2. 整个过程怎么变成 Runtime 认的 Trace（ShopAgent）。

它不认识 main_agent、不认识 config、不认识任何评测口径。

━━ 与 main_agent 版本的差别，以及为什么这么选 ━━

main_agent 的循环带三层上下文机制：窗口裁剪（只发最近 keep_turns 轮）、
槽位（从工具实参反填的订单号/预算/品类）、事实卡片（查过的对象摘要）。
这三层是为了解决"消息越滚越长"。

这里不裁剪，全部 history 每轮都发。
理由：本题集最多 3-4 轮，一轮最多 6 步，完整 history 装得下；
不裁剪就不需要槽位和事实卡片——那两样存在的唯一目的是"补回被裁掉的信息"。
少一层机制，少一处出错的地方，也少一份和 main_agent 相同的代码。

代价要说清楚：如果以后出现十几轮的长会话，这里会先撞上下文上限，
而 main_agent 不会。这不是 bug，是刻意的取舍——真到那天再加裁剪。

━━ 三处错了不会报错、只会悄悄判歪的地方 ━━

一、工具返回按 tool_call_id 一对一配对。
    不按工具名合并。同名工具调两次时合并会串，第一次调用拿到两次返回拼起来的内容。

二、Turn.meta["status"] 必须带上循环的结束状态。
    evaluator 判"步数耗尽"只看这个字段。漏了，一次 max_steps 的轨迹会被
    按回答内容猜成"缺事实"，归因错一层，后面提炼的经验也跟着错。

三、经验挂在 system_prompt 上，不塞进 history。
    每轮重建 SystemMessage，history 本身不动，轨迹才干净。
"""

from __future__ import annotations

import contextlib
import io
import time

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from core.protocol import RunContext
from core.task import Task
from core.trace import ToolCall, Trace, Turn, now_ms

from agents.shop.core import planning
from agents.shop.core.recovery import HINTS, classify, retry_limit
from agents.shop.llm import build_shop_llm
from agents.shop.tools import shop_tools
from agents.shop.view.context import prompt_for, with_experience
from agents.shop.view.slots import ARG_TO_SLOT, SLOT_LABELS
from core.context import render_slots, update_slots

# 出现这些字样时认为失败是链路问题，值得重试一次。
# 一次 502 不该被记成"模型能力差"。
INFRA_HINTS = (
    "Connection",
    "Timeout",
    "RateLimit",
    "APIConnection",
    "InternalServer",
    "502",
    "503",
)

RETRY_WAIT_SEC = 3.0

# 步数用尽时给模型的收尾指令。要求它停下来说已经查到什么，而不是再要一次工具。
CLOSING_HINT = (
    "工具调用次数已经用完。不要再调用任何工具，"
    "直接用上面已经拿到的结果回答用户；确实没查到的部分，明确说没查到。"
)


def _invoke_tool(fn, args: dict) -> tuple[str, dict]:
    """执行一次工具调用，按返回类型决定是否自动重试。

    返回 (最终输出, 执行记录)。记录会进 Turn.meta，
    让"这次到底重试了没有、为什么重试"可观测，
    而不是只能从最终回答里反推。
    """
    limit = retry_limit()
    attempts = 0
    kind = "ok"
    out = ""

    while attempts <= limit:
        attempts += 1
        try:
            out = str(fn.invoke(args))
        except Exception as exc:
            # 工具层承诺不抛异常，这里兜的是它没守住的意外。
            out = f"ERROR 工具执行失败 {type(exc).__name__}: {exc}"
        kind = classify(out)
        if kind != "retry":
            break

    if attempts > 1:
        out = (
            f"{out}\n[运行时] 上一次调用返回服务端错误，"
            f"已用同一参数自动重试 {attempts - 1} 次，最终判定 {kind}。"
        )
    elif kind in HINTS:
        out = f"{out}\n[运行时] {HINTS[kind]}"

    return out, {"tool": getattr(fn, "name", ""), "attempts": attempts, "kind": kind}


def _closing_answer(llm, messages, system_prompt: str) -> str:
    """步数用尽时的收尾回答。

    先让模型基于已有结果说一句人话；模型不配合（还想要工具、或返回空），
    就退到"最后一条工具结果"——至少让轨迹里有东西可判、可读。
    返回空字符串表示确实拿不出任何内容，那种情况下判 max_steps 是应当的。
    """
    try:
        closing = llm.invoke(
            compose_view(list(messages) + [HumanMessage(CLOSING_HINT)], system_prompt)
        )
        text = str(getattr(closing, "content", "") or "").strip()
        if text and not getattr(closing, "tool_calls", None):
            return text
    except Exception:
        pass

    for msg in reversed(list(messages)):
        if getattr(msg, "type", "") != "tool":
            continue
        body = str(getattr(msg, "content", "") or "").strip()
        if body:
            return "（本次可用的工具调用次数已用完，以下是最后一次查到的结果）\n" + body[:300]
    return ""


def compose_view(messages, system_prompt: str) -> list:
    """把 history 拼成这一轮要发给模型的消息。

    每轮都用传入的 system_prompt 重建首条系统消息，
    而不是沿用 messages[0]——经验是逐轮挂上去的，
    沿用旧的首条会让第二轮之后经验失效。
    """
    rest = (
        messages[1:]
        if messages and isinstance(messages[0], SystemMessage)
        else list(messages)
    )
    return [SystemMessage(system_prompt)] + list(rest)


def _compose_prompt(system_prompt: str, slots, plan_steps) -> str:
    """把系统提示、槽位、本轮待办拼成这一次真正发出去的系统提示。

    三段的顺序是有意的：槽位在最前面（它是跨轮的事实），
    待办在后面（它只作用于本轮）。两者都空时与原始提示逐字相同，
    保证关掉这两个功能后与改动前完全一致。
    """
    parts = [system_prompt]
    held = render_slots(slots or {}, SLOT_LABELS)
    if held:
        parts.append(held)
    plan_text = planning.render_plan(plan_steps)
    if plan_text:
        parts.append(plan_text)
    return "\n\n".join(parts)


def run_loop(
    llm,
    tools,
    system_prompt: str,
    question: str,
    *,
    max_steps: int = 6,
    history=None,
    use_plan: bool = False,
    slots: dict | None = None,
) -> dict:
    """跑一轮：模型决定调哪个工具，代码执行，结果回灌，直到模型不再调工具。

    返回结构与 main_agent 的 run_agent 对齐（answer / messages / status / steps），
    这样上层适配不需要为两个 Agent 写两套解析。

    use_plan 打开时多两步（见 planning 模块）：
        进循环前先拆待办；模型想收尾但待办没做完时，顶回去一次。
    关掉就与改动前完全一致，对照实验只切这一个开关。
    """
    bound = llm.bind_tools(tools) if tools else llm
    tool_map = {t.name: t for t in tools}

    plan_report: dict = {}
    plan_steps: list[dict] = []
    if use_plan and planning.needed(question):
        plan_steps = planning.make_plan(
            llm, system_prompt, question, tools, history, report=plan_report
        )
    held_slots = slots if slots is not None else {}
    active_prompt = _compose_prompt(system_prompt, held_slots, plan_steps)

    messages = list(history) if history is not None else [SystemMessage(system_prompt)]
    messages.append(HumanMessage(question))
    recovery: list[dict] = []
    # 已经真的执行过的调用，用来算待办覆盖。
    executed: list[dict] = []
    nudges = 0

    for step in range(1, max_steps + 1):
        ai = bound.invoke(compose_view(messages, active_prompt))
        messages.append(ai)

        if not ai.tool_calls:
            missed = planning.uncovered(plan_steps, executed)
            # 想收尾但还有待办没做：顶回去一次。
            # 只在还有步数可用时顶，否则等于把 max_steps 的语义改掉。
            if missed and nudges < planning.MAX_NUDGES and step < max_steps:
                nudges += 1
                messages.append(HumanMessage(planning.reminder(missed)))
                continue

            return {
                "answer": ai.content or "",
                "messages": messages,
                "status": "ok",
                "steps": step,
                "recovery": recovery,
                "plan": plan_steps,
                "plan_report": plan_report,
                "slots": dict(held_slots),
                "uncovered": missed,
                "nudges": nudges,
            }

        for call in ai.tool_calls:
            fn = tool_map.get(call["name"])
            if fn is None:
                out = f"未知工具 {call['name']}，可用工具：{list(tool_map)}"
                recovery.append(
                    {"tool": call["name"], "attempts": 0, "kind": "unknown"}
                )
            else:
                out, record = _invoke_tool(fn, call["args"])
                recovery.append(record)
                executed.append({"name": call["name"], "args": dict(call["args"] or {})})
            messages.append(ToolMessage(content=out, tool_call_id=call["id"]))

        # 同一轮里也可能先查 A 再查 B，后面的决策要看得到刚填进来的槽位；
        # 跨轮则由调用方持有同一个 slots 字典继续传下来。
        update_slots(held_slots, executed, ARG_TO_SLOT)
        active_prompt = _compose_prompt(system_prompt, held_slots, plan_steps)

    # 步数用尽：不让模型交白卷。
    # 状态仍然记 max_steps——判定口径不变，兜底只保证轨迹里有可用信息，
    # 不能把一次"没做完"粉饰成"做完了"。
    return {
        "answer": _closing_answer(llm, messages, active_prompt),
        "messages": messages,
        "status": "max_steps",
        "steps": max_steps,
        "recovery": recovery,
        "plan": plan_steps,
        "plan_report": plan_report,
        "slots": dict(held_slots),
        "uncovered": planning.uncovered(plan_steps, executed),
        "nudges": nudges,
    }


def calls_from_messages(messages, turn_index: int) -> list[ToolCall]:
    """把一轮新增的消息切成按发生顺序排列的 ToolCall。

    两遍扫描：
        第一遍按出现顺序建出 ToolCall，用 tool_call_id 建索引；
        第二遍把 ToolMessage 的返回填回它自己那次调用。

    必须两遍的原因：一条 AIMessage 可能带多个 tool_calls，
    它们的返回要等后续若干条 ToolMessage 才齐全，边扫边填会留下空返回。
    """
    index: dict[str, ToolCall] = {}
    ordered: list[ToolCall] = []

    for msg in messages:
        for raw in (getattr(msg, "tool_calls", None) or []):
            call = ToolCall(
                name=str(raw.get("name") or ""),
                args=dict(raw.get("args") or {}),
                # 按 core 的约定重编号为 t{轮}c{序}，不用模型给的 id：
                # 那串格式因服务端而异，而轨迹要能跨 Agent 读同一套编号。
                call_id=f"t{turn_index}c{len(ordered)}",
            )
            # 空 id 不能让两个调用互相覆盖，补一个占位键。
            index[str(raw.get("id") or f"__pos{len(ordered)}")] = call
            ordered.append(call)

    for msg in messages:
        if getattr(msg, "type", "") != "tool":
            continue
        call = index.get(str(getattr(msg, "tool_call_id", "") or ""))
        if call is None:
            # 找不到对应调用的返回不猜、不补，直接丢掉。
            continue
        output = str(getattr(msg, "content", "") or "")
        call.output = output
        if output.startswith("ERROR"):
            call.ok = False
            call.error = output[:200]

    return ordered


class ShopAgent:
    """电商客服 Agent 在 Runtime 里的实现。

    variant 决定用哪套提示词：
        "tuned" 11 条规则，对应历史报告那条基线；
        "cold"  前 3 条底线规则，作为"还没积累经验"的起点。

    llm / tools / prompt 都可注入，单测用一个假模型就能跑完整个 run。
    """

    name = "shop"

    def __init__(
        self,
        variant: str = "tuned",
        *,
        prompt: str | None = None,
        tools: list | None = None,
        llm=None,
        max_steps: int = 6,
        retries: int = 1,
        plan: bool = False,
        name: str | None = None,
    ) -> None:
        self.variant = variant
        self.prompt = prompt if prompt is not None else prompt_for(variant)
        self.tools = list(tools) if tools is not None else shop_tools()
        self._llm = llm
        self.max_steps = max_steps
        self.retries = retries
        # 默认关。真实入口（make_shop_agent）按 SHOP_PLAN 环境变量打开，
        # 单测直接构造 ShopAgent 时保持旧行为，不因新功能改动既有断言。
        self.plan = bool(plan)
        if name:
            self.name = name

    def _ensure_llm(self):
        """延迟到真要跑时才建模型：缺 API Key 不该在构造对象时就炸。"""
        if self._llm is None:
            self._llm = build_shop_llm()
        return self._llm

    def _run_turns(self, task: Task, prompt: str, budget: int) -> tuple[list[Turn], str]:
        """逐轮执行，返回 (轮次列表, 错误信息)。错误信息为空表示成功。"""
        llm = self._ensure_llm()
        last = ""

        for attempt in range(self.retries + 1):
            try:
                # 屏蔽循环里的中间打印，否则一屏都是工具日志，看不到判定结果。
                with contextlib.redirect_stdout(io.StringIO()):
                    history = None
                    prev_len = 0
                    turns: list[Turn] = []
                    # 跨轮槽位：同一个字典在每一轮之间传下去，
                    # run_loop 会就地补全它。这样第 3 轮即使把第 1、2 轮的消息
                    # 全带在 history 里，模型也能在系统提示里直接看到"预算 900"，
                    # 不用自己去几十条消息里翻。
                    slots: dict = {}

                    for i, text in enumerate(task.user_turns()):
                        turn_started = now_ms()
                        result = run_loop(
                            llm,
                            self.tools,
                            prompt,
                            text,
                            max_steps=budget,
                            history=history,
                            use_plan=self.plan,
                            slots=slots,
                        )
                        history = result["messages"]
                        # 只取本轮新增的消息，上一轮的工具返回不能算进这一轮。
                        new_msgs = history[prev_len:]
                        prev_len = len(history)

                        # max_steps 判定的唯一依据，见模块文档第二点。
                        meta = {"status": result.get("status") or "ok"}
                        # 只在真的发生过重试/参数纠偏时写，避免每条轨迹都背一个空数组。
                        if result.get("recovery"):
                            meta["recovery"] = list(result["recovery"])
                        # 计划只在拆出来非空时落盘；落盘是为了事后能查
                        # "这次到底计划了什么、哪项没做完"，而不是只看最终回答。
                        if result.get("plan"):
                            meta["plan"] = [dict(s) for s in result["plan"]]
                            meta["plan_uncovered"] = [
                                dict(s) for s in (result.get("uncovered") or [])
                            ]
                            meta["plan_nudges"] = int(result.get("nudges") or 0)
                        elif (result.get("plan_report") or {}).get("called"):
                            # 门开了但没拆出待办 = 计划失败，不是"不需要计划"。
                            # 两者在 meta 里都没有 plan 字段，含义却相反：
                            # 前者是待修的缺陷，后者是设计。必须能分开看，
                            # 否则只能对着"这次怎么又没过"猜。
                            meta["plan_error"] = str(
                                (result.get("plan_report") or {}).get("error") or "empty"
                            )
                        # 槽位落盘才能事后验证"第 3 轮确实看到了第 2 轮的预算"，
                        # 而不是只能从最终回答里猜它有没有用上。
                        if result.get("slots"):
                            meta["slots"] = dict(result["slots"])

                        turns.append(
                            Turn(
                                index=i,
                                user_input=text,
                                calls=calls_from_messages(new_msgs, i),
                                answer=str(result.get("answer") or ""),
                                steps=int(result.get("steps") or 0),
                                elapsed_ms=max(0, now_ms() - turn_started),
                                meta=meta,
                            )
                        )
                return turns, ""

            except Exception as exc:
                last = f"{type(exc).__name__}: {exc}"
                if attempt < self.retries and any(k in last for k in INFRA_HINTS):
                    time.sleep(RETRY_WAIT_SEC)
                    continue
                return [], last

        return [], last

    def run(self, task: Task, ctx: RunContext | None = None) -> Trace:
        started = now_ms()
        experience = ctx.experience if ctx else ""
        prompt = with_experience(self.prompt, experience)

        # 预算以 Runtime 给的为准。Agent 自己的 max_steps 只是
        # "不经过 runner、直接调 run()"时的兜底，不是一个平行的第二来源。
        budget = ctx.max_steps if ctx is not None and ctx.max_steps else self.max_steps

        turns, err = self._run_turns(task, prompt, budget)

        if err:
            # 抛出去让 runner 兜成 status=crashed。
            # 不自己造一条"状态异常但看起来成功"的轨迹——
            # 那会让 evaluator 把链路故障当成能力问题记进报告。
            raise RuntimeError(err)

        return Trace(
            task_id=task.id,
            agent=self.name,
            turns=turns,
            split=task.split,
            elapsed_ms=max(0, now_ms() - started),
            meta={
                "variant": self.variant,
                "plan": self.plan,
                "experience_applied": bool(experience and experience.strip()),
                "experience_chars": len(experience or ""),
                "hits": list(ctx.hits) if ctx else [],
                "max_steps": budget,
            },
        )

    def __repr__(self) -> str:
        return (
            f"<ShopAgent {self.name!r} variant={self.variant!r} "
            f"tools={len(self.tools)} max_steps={self.max_steps}>"
        )


__all__ = ["ShopAgent", "run_loop", "calls_from_messages", "compose_view"]
