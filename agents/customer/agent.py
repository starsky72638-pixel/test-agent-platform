"""售后客服 Agent：ReAct 循环 + Trace 组装，全部自包含。

━━ 为什么这里又写了一遍循环 ━━

agents/shop/core/agent.py 里有一份完整循环（带 plan、recovery、slots）。
这里刻意不 import 它，因为"可插拔"必须可验证：
    如果第二个业务 Agent 必须依赖第一个 Agent 的实现，
    那"换 Agent 不改核心"就只是口号——shop 改一处，customer 跟着崩。

tools 层已经确立过同样的做法（main_agent 与 shop 各持一份 shop_skill.py），
这里只是把同一条原则从工具延伸到循环。重复的是通用模式，不是业务逻辑。

代价说清楚：shop 那份有自动重试、步数耗尽兜底、槽位裁剪、待办检查，
这里只保留前两个。没有 plan 与 slots——客服场景最多 3 轮，
那两层机制解决的是"消息越滚越长"，这里撞不到。

━━ 两处错了不会报错、只会悄悄判歪的地方 ━━

一、工具返回必须按 tool_call_id 一对一配对。
    一条 AIMessage 可能带多个 tool_calls，按工具名合并会把两次返回拼在一起。

二、Turn.meta["status"] 必须写循环结束状态。
    evaluator 判"步数耗尽"只看这个字段；漏了就按回答内容猜，归因错一层。
"""

from __future__ import annotations

import contextlib
import io
import time

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from agents.customer.tools import customer_tools
from core.protocol import RunContext
from core.task import Task
from core.trace import ToolCall, Trace, Turn, now_ms

DEFAULT_PROMPT = """你是电商售后客服，只处理三类问题：订单状态、物流进度、退款申请。

规则：
1. 涉及订单事实的问题必须先调用工具查，不能凭印象或推测回答。
2. 查物流要用订单号；用户没说清订单号时先问清楚，不要猜。
3. 申请退款前必须先 get_order 确认订单状态：待支付订单无法退款，要如实说明原因。
4. 工具没返回的信息就是没有：不要编造预计送达时间、承运商或物流轨迹。
5. 回答先给结论，再给依据，简洁不啰嗦。"""

# 冷启动对照专用：只有角色和工具，没有任何行为准则。
# 为什么要它：DEFAULT_PROMPT 把"退款前先查订单""不编造 ETA"都写死了，
# 在那个 Agent 上注入经验测不出差别（6/6 全过，是 prompt 的功劳，不是经验）。
# 而经验只从失败轮提炼，基线不失败就学不到任何东西——这是上一轮闭环
# 得出 null 的真实原因。冷变体存在的唯一目的，就是给出一个会失败的起点。
COLD_PROMPT = """你是电商售后客服，可以调用工具查询订单、查询物流、提交退款申请。

根据用户的问题调用合适的工具，然后回答。
"""

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

# 步数用尽时给模型的收尾指令：要求它停下来说已经查到什么，而不是再要一次工具。
CLOSING_HINT = (
    "工具调用次数已经用完。不要再调用任何工具，"
    "直接用上面已经拿到的结果回答用户；确实没查到的部分，明确说没查到。"
)


def build_customer_llm(settings=None):
    """构造客服 Agent 的模型客户端。

    与 shop 保持同一组参数（streaming=False、temperature=0）：
    温度不为 0 时同一道题两次结果不同，就说不清"没通过"是能力还是抖动。
    延迟到真正要跑时才读配置，保证构造 Agent 对象不会因为缺密钥而失败。
    """
    from config import build_llm, load_settings

    cfg = settings or load_settings()
    return build_llm(cfg, streaming=False).bind(temperature=0)


def with_experience(prompt: str, experience: str) -> str:
    """把经验拼到提示词尾部。

    明确写"以工具返回为准"，是因为经验里可能带示例值（比如某个订单号），
    模型很容易把它当成事实照抄——那正好是我们要抑制的幻觉。
    """
    text = (experience or "").strip()
    if not text:
        return prompt
    return (
        f"{prompt}\n\n【历史经验】\n{text}\n"
        "（经验仅供参考，事实一律以工具返回为准。）"
    )


def _view(system_prompt: str, messages) -> list:
    """拼这一轮要发给模型的消息，首条系统消息每轮重建。

    history 里存的第一条是上一轮用过的 SystemMessage；沿用它，
    本轮调整过的提示词就进不去。
    """
    rest = (
        messages[1:]
        if messages and isinstance(messages[0], SystemMessage)
        else list(messages)
    )
    return [SystemMessage(system_prompt)] + list(rest)


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


def _closing_answer(llm, messages, system_prompt: str) -> str:
    """步数用尽时的收尾回答。

    先让模型基于已有结果说一句人话；模型不配合（还想要工具、或返回空），
    就退到"最后一条工具结果"——至少让轨迹里有东西可判、可读。
    返回空字符串表示确实拿不出任何内容，那种情况下判 max_steps 是应当的。
    """
    try:
        closing = llm.invoke(_view(system_prompt, list(messages) + [HumanMessage(CLOSING_HINT)]))
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


def run_turn(
    llm,
    tools,
    system_prompt: str,
    question: str,
    *,
    max_steps: int = 6,
    history=None,
) -> dict:
    """跑一轮：模型决定调哪个工具，代码执行，结果回灌，直到模型不再调工具。

    返回 answer / messages / status / steps，与 shop 版同名同义，
    上层不需要为两个 Agent 写两套解析。
    """
    bound = llm.bind_tools(tools) if tools else llm
    tool_map = {t.name: t for t in tools}

    messages = list(history) if history is not None else [SystemMessage(system_prompt)]
    messages.append(HumanMessage(question))

    for step in range(1, max_steps + 1):
        ai = bound.invoke(_view(system_prompt, messages))
        messages.append(ai)

        if not ai.tool_calls:
            return {
                "answer": ai.content or "",
                "messages": messages,
                "status": "ok",
                "steps": step,
            }

        for call in ai.tool_calls:
            fn = tool_map.get(call["name"])
            if fn is None:
                out = f"未知工具 {call['name']}，可用工具：{list(tool_map)}"
            else:
                try:
                    out = str(fn.invoke(call.get("args") or {}))
                except Exception as exc:
                    # 工具层承诺不抛异常，这里兜的是它没守住的意外。
                    out = f"ERROR 工具执行失败 {type(exc).__name__}: {exc}"
            messages.append(ToolMessage(content=out, tool_call_id=call["id"]))

    # 步数用尽：不让模型交白卷。
    # 状态仍然记 max_steps——判定口径不变，兜底只保证轨迹里有可用信息，
    # 不能把一次"没做完"粉饰成"做完了"。
    return {
        "answer": _closing_answer(llm, messages, system_prompt),
        "messages": messages,
        "status": "max_steps",
        "steps": max_steps,
    }


class CustomerAgent:
    """售后客服 Agent 在 Runtime 里的实现。

    llm / tools / prompt 都可注入，单测用一个假模型就能跑完整个 run。
    """

    name = "customer"

    def __init__(
        self,
        *,
        prompt: str | None = None,
        tools: list | None = None,
        llm=None,
        max_steps: int = 6,
        retries: int = 1,
        name: str | None = None,
    ) -> None:
        self.prompt = DEFAULT_PROMPT if prompt is None else prompt
        self.tools = list(tools) if tools is not None else customer_tools()
        self._llm = llm
        self.max_steps = max_steps
        self.retries = retries
        if name:
            self.name = name

    def _ensure_llm(self):
        """延迟到真要跑时才建模型：缺 API Key 不该在构造对象时就炸。"""
        if self._llm is None:
            self._llm = build_customer_llm()
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

                    for i, text in enumerate(task.user_turns()):
                        turn_started = now_ms()
                        result = run_turn(
                            llm,
                            self.tools,
                            prompt,
                            text,
                            max_steps=budget,
                            history=history,
                        )
                        history = result["messages"]
                        # 只取本轮新增的消息，上一轮的工具返回不能算进这一轮。
                        new_msgs = history[prev_len:]
                        prev_len = len(history)

                        # max_steps 判定的唯一依据，见模块文档第二点。
                        turns.append(
                            Turn(
                                index=i,
                                user_input=text,
                                calls=calls_from_messages(new_msgs, i),
                                answer=str(result.get("answer") or ""),
                                steps=int(result.get("steps") or 0),
                                elapsed_ms=max(0, now_ms() - turn_started),
                                meta={"status": result.get("status") or "ok"},
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
        # "不经过 runner、直接调 run()"时的兜底，不是平行的第二来源。
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
                "experience_applied": bool(experience and experience.strip()),
                "experience_chars": len(experience or ""),
                "hits": list(ctx.hits) if ctx else [],
                "max_steps": budget,
            },
        )

    def __repr__(self) -> str:
        return (
            f"<CustomerAgent {self.name!r} "
            f"tools={len(self.tools)} max_steps={self.max_steps}>"
        )


__all__ = [
    "CustomerAgent",
    "DEFAULT_PROMPT",
    "run_turn",
    "calls_from_messages",
    "with_experience",
]
