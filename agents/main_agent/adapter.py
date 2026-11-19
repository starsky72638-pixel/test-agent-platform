"""接入层：把 main_agent 这个真实 Agent 接到 Runtime 协议上。

它和 agents/shop 是两条独立实现，差别不在代码风格，在上下文机制：

    shop       不裁剪，全量 history 每轮都发
    main_agent 窗口裁剪 + 槽位反填 + 事实卡片，三层（见 view/memory.py）

同一份任务集、同一个 Runtime，跑出两种不同的上下文策略——
这是"Runtime 与 Agent 解耦"的可执行证据，而不是一句结构说明。
如果两个 Agent 只是同一个循环换套提示词，接入第二个就证明不了任何事。

━━ 这里只做三件事，不重写循环 ━━

一、逐轮调 run_agent()，把每轮新增的消息切成 Turn。
    循环本体、窗口裁剪、槽位、事实卡片全部沿用原实现。
    抄一份过来最省事，但那等于把 main_agent 的优势抹掉，
    最后两个 Agent 退化成"同一个实现的两种配置"。

二、把 LangChain 消息形状翻成 core 的 ToolCall。
    core/trace.py 刻意不认 langchain（core 不许依赖任何框架），
    所以转换只能发生在接入层。

三、把经验挂到 system_prompt 上。
    build_view 用的是传参 prompt（view/memory.py:253），不是 messages[0]，
    所以每轮传 prompt + 经验就生效，history 本身不被污染。
    槽位和事实卡片挂在 messages[0].additional_kwargs 上，与这条路径互不干扰。

━━ 两处错了不会报错、只会悄悄判歪的地方 ━━

一、Turn.meta["status"] 必须带循环的结束状态。
    evaluator 判"步数耗尽"只看这个字段。漏了，一次 max_steps 会被
    按回答内容猜成"缺事实"，归因错一层，后面提炼的经验跟着错。

二、run_agent 会往 stdout 打工具日志。
    一批跑 25 条、每条 6 步，就是一屏几百行的 [tool]/[result]，
    判定结果会被冲掉。所以整段调用要屏蔽 stdout。
"""

from __future__ import annotations

import contextlib
import io
import time

from core.protocol import RunContext
from core.task import Task
from core.trace import ToolCall, Trace, Turn, now_ms

from agents.experience import attach as with_experience

# 出现这些字样时认为失败是链路问题，值得重试一次。
# 与 agents/shop 保持同一组，否则两个 Agent 在"网络抖一下"面前表现不同，
# 对比实验就会把重试策略的差别算成能力差别。
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


def calls_from_messages(messages, turn_index: int) -> list[ToolCall]:
    """把一轮新增的消息切成按发生顺序排列的 ToolCall。

    两遍扫描：先按出现顺序建出调用、用 tool_call_id 建索引，
    再把 ToolMessage 的返回填回它自己那一次调用。

    必须两遍：一条 AIMessage 可能带多个 tool_calls，
    它们的返回要等后续若干条 ToolMessage 才齐全，边扫边填会留下空返回。

    这里与 agents/shop 的同名函数逻辑一致，但各存一份——
    两个 Agent 要能互相删除而不影响对方，就不能共用一个转换函数。
    转换的是 LangChain 的消息形状，不是业务逻辑，重复的代价是可控的。
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


def default_tools() -> list:
    """取 main_agent 自己那套工具。

    register_default_tools 是 append 语义，重复调用会把同一批工具注册两遍，
    tools 列表变成 10 项、模型看到两份同名工具。所以先看表里有没有。
    """
    from agents.main_agent.tools import get_registry, register_default_tools

    registry = get_registry()
    if registry.is_empty():
        register_default_tools()
    return registry.list()


def build_default_llm():
    """构造 main_agent 用的模型客户端。

    streaming=False、temperature=0，与 agents/shop 的 llm.py 同一组参数。
    温度不为 0 时同一道题两次结果不同，注入前后就没法归因。
    """
    from config import build_llm, load_settings

    return build_llm(load_settings(), streaming=False).bind(temperature=0)


class MainAgent:
    """把 main_agent 的 ReAct 循环包成 Runtime 认的 Agent。

    prompt / tools / llm 都可注入，离线单测用假模型就能跑完整个 run。
    """

    name = "main"

    def __init__(
        self,
        *,
        prompt: str | None = None,
        tools: list | None = None,
        llm=None,
        max_steps: int = 6,
        keep_turns: int = 3,
        retries: int = 1,
        name: str | None = None,
    ) -> None:
        if prompt is None:
            from agents.main_agent.tools import SHOP_SYSTEM_PROMPT

            prompt = SHOP_SYSTEM_PROMPT
        self.prompt = prompt
        self.tools = list(tools) if tools is not None else default_tools()
        self.keep_turns = keep_turns
        self.max_steps = max_steps
        self.retries = retries
        self._llm = llm
        if name:
            self.name = name

    def _ensure_llm(self):
        """延迟到真要跑时才建模型：缺 API Key 不该在构造对象时就炸。"""
        if self._llm is None:
            self._llm = build_default_llm()
        return self._llm

    def _run_turns(self, task: Task, prompt: str, budget: int) -> tuple[list[Turn], str]:
        """逐轮执行，返回 (轮次列表, 错误信息)。错误信息为空表示成功。"""
        from agents.main_agent.core.agent import run_agent

        llm = self._ensure_llm()
        last = ""

        for attempt in range(self.retries + 1):
            try:
                # 屏蔽循环里的工具日志，否则判定结果会被几百行中间输出冲掉。
                with contextlib.redirect_stdout(io.StringIO()):
                    history = None
                    prev_len = 0
                    turns: list[Turn] = []

                    for i, text in enumerate(task.user_turns()):
                        turn_started = now_ms()
                        result = run_agent(
                            llm,
                            self.tools,
                            prompt,
                            text,
                            max_steps=budget,
                            history=history,
                            keep_turns=self.keep_turns,
                        )
                        history = result["messages"]
                        # 只取本轮新增的消息，上一轮的工具返回不能算进这一轮。
                        new_msgs = history[prev_len:]
                        prev_len = len(history)

                        turns.append(
                            Turn(
                                index=i,
                                user_input=text,
                                calls=calls_from_messages(new_msgs, i),
                                answer=str(result.get("answer") or ""),
                                steps=int(result.get("steps") or 0),
                                elapsed_ms=max(0, now_ms() - turn_started),
                                # max_steps 判定的唯一依据，见模块文档第二点。
                                meta={
                                    "status": result.get("status") or "ok",
                                    # 这一轮实际发给模型几条消息。窗口裁剪生效时它小于 history 长度，
                                    # 这是事后核对"槽位/事实卡片有没有真的起作用"的唯一观测点。
                                    "sent_msgs": int(result.get("sent_msgs") or 0),
                                },
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
                "variant": "main_agent",
                "context": "windowed",  # 窗口裁剪 + 槽位 + 事实卡片，与 shop 的 "full" 相对
                "keep_turns": self.keep_turns,
                "experience_applied": bool(experience and experience.strip()),
                "experience_chars": len(experience or ""),
                "hits": list(ctx.hits) if ctx else [],
                "max_steps": budget,
            },
        )

    def __repr__(self) -> str:
        return (
            f"<MainAgent {self.name!r} tools={len(self.tools)} "
            f"max_steps={self.max_steps} keep_turns={self.keep_turns}>"
        )


__all__ = [
    "MainAgent",
    "calls_from_messages",
    "default_tools",
    "build_default_llm",
]
