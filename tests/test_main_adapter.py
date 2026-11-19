"""第二个真实 Agent（main_agent）接入 Runtime 的离线测试。

全部不连模型、不连 HTTP、不花额度。要证明三件事：

1. 它符合协议，而且跑的是自己的循环。
   main_agent 带窗口裁剪 + 槽位 + 事实卡片，shop 一层都没有。
   如果两个 Agent 只是同一个循环换套提示词，"接入第二个 Agent"就证明不了任何事。
   所以这里的断言不是"能跑"，而是"裁剪后第 1 轮的信息还在 system prompt 里"。

2. 预算由 Runtime 说了算。
   ctx.max_steps 优先于构造参数，且两者都只应有一个生效。
   这条不写死，报告里"步数上限 6"就只是打印出来的一个数。

3. CLI 与具体 Agent 解耦。
   scripts/run_loop.py 里不许出现任何 agents.shop / agents.main_agent 字面量。
   加一个 Agent 只加一行注册，不改 CLI——这是可插拔的可执行定义。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import agents.builtin as builtin
from agents.experience import attach
from agents.main_agent.adapter import MainAgent, calls_from_messages
from core.protocol import RunContext, validate_agent
from core.runner import run_task
from core.task import Task
from core.trace import STATUS_CRASHED, STATUS_OK, Trace
from evaluator.verdict import evaluate

ROOT = Path(__file__).resolve().parents[1]


# =========================
# 假对象：不连服务也能跑完整链路
# =========================


class FakeTool:
    def __init__(self, name: str, output: str) -> None:
        self.name = name
        self.output = output

    def invoke(self, args):
        return self.output


class ScriptedLLM:
    """按脚本逐次返回消息的模型替身。

    每次 invoke 都把收到的消息列表存下来，这样"经验进没进提示词"
    "裁剪有没有生效"都能被直接检查，而不是靠推断。
    """

    def __init__(self, steps) -> None:
        self.steps = list(steps)
        self.calls = 0
        self.seen: list[list] = []
        self.bound_tools = None

    def bind_tools(self, tools):
        self.bound_tools = list(tools)
        return self

    def invoke(self, messages):
        self.seen.append(list(messages))
        step = self.steps[self.calls] if self.calls < len(self.steps) else ""
        self.calls += 1
        if isinstance(step, AIMessage):
            return step
        return AIMessage(content=str(step))


def tool_call(name: str, args: dict, call_id: str = "call_a") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def tool_msg(text: str, call_id: str) -> ToolMessage:
    return ToolMessage(content=text, tool_call_id=call_id)


def one_turn_task(task_id: str = "T1") -> Task:
    return Task(id=task_id, input="我的订单 1001 到哪了", split="holdout")


def make_agent(steps, *, tools=None, **kwargs) -> tuple[MainAgent, ScriptedLLM]:
    llm = ScriptedLLM(steps)
    agent = MainAgent(
        prompt="P",
        tools=tools if tools is not None else [FakeTool("get_order", "已签收")],
        llm=llm,
        **kwargs,
    )
    return agent, llm


# =========================
# 一、协议与轨迹
# =========================


def test_agent_satisfies_protocol():
    validate_agent(MainAgent(prompt="P", tools=[]))


def test_run_produces_valid_trace():
    agent, _ = make_agent([tool_call("get_order", {"order_id": 1001}), "订单 1001 已签收"])
    trace = agent.run(one_turn_task())

    assert isinstance(trace, Trace)
    assert trace.status == STATUS_OK
    assert trace.agent == "main"
    assert len(trace.turns) == 1
    assert trace.turns[0].calls[0].name == "get_order"
    assert trace.turns[0].calls[0].output == "已签收"
    assert trace.final_answer() == "订单 1001 已签收"


def test_call_ids_follow_turn_convention():
    msgs = [
        tool_call("get_order", {"order_id": 1}, "x"),
        tool_call("get_logistics", {"order_id": 1}, "y"),
    ]
    assert [c.call_id for c in calls_from_messages(msgs, 3)] == ["t3c0", "t3c1"]


def test_calls_pair_by_id_not_by_name():
    """同名工具调两次，各自拿各自的返回——按名字合并会串。"""
    msgs = [
        tool_call("get_order", {"order_id": 1001}, "a"),
        tool_call("get_order", {"order_id": 1002}, "b"),
        tool_msg("第一次返回", "a"),
        tool_msg("第二次返回", "b"),
    ]
    calls = calls_from_messages(msgs, 0)
    assert [c.output for c in calls] == ["第一次返回", "第二次返回"]


def test_multi_turn_slices_calls_per_turn():
    """第二轮只算第二轮自己的调用，第一轮的不能重复计进来。"""
    agent, _ = make_agent(
        [
            tool_call("get_order", {"order_id": 1001}, "a"),
            "订单 1001 已签收",
            tool_call("get_logistics", {"order_id": 1001}, "b"),
            "已由中通送达",
        ]
    )
    task = Task(id="M1", turns=["查订单 1001", "物流呢"], split="fit")
    trace = agent.run(task)

    assert [t.index for t in trace.turns] == [0, 1]
    assert trace.turns[0].call_names() == ["get_order"]
    assert trace.turns[1].call_names() == ["get_logistics"]
    assert trace.turns[1].calls[0].call_id == "t1c0"


def test_max_steps_status_lands_in_meta():
    """max_steps 必须写进 meta，否则 evaluator 会当成正常结束。"""
    agent, _ = make_agent([tool_call("get_order", {"order_id": 1001})], max_steps=1)
    trace = agent.run(one_turn_task())

    assert trace.turns[0].meta["status"] == "max_steps"
    assert trace.turns[0].answer == ""


def test_llm_failure_becomes_crashed_trace():
    class BoomLLM:
        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            raise ConnectionError("Connection refused")

    agent = MainAgent(prompt="P", tools=[], llm=BoomLLM(), retries=0)
    trace = run_task(agent, one_turn_task())

    assert trace.status == STATUS_CRASHED
    assert "Connection refused" in trace.error


def test_trace_can_be_judged():
    agent, _ = make_agent([tool_call("get_order", {"order_id": 1001}), "订单 1001 已签收"])
    task = Task(
        id="T1",
        input="订单 1001 什么状态",
        split="holdout",
        assertions={
            "expect_tools": ["get_order"],
            "expect_args": {"order_id": 1001},
            "must_have": ["已签收"],
        },
    )
    verdict = evaluate(agent.run(task), task)

    assert verdict.passed is True
    assert verdict.failure_type == ""


# =========================
# 二、它真的在跑另一套上下文策略
# =========================


def test_context_window_really_trims():
    """keep_turns=1 时，第 3 轮发给模型的消息里不该再有前两轮的用户话。

    这是"两个 Agent 结构不同"的硬证据：
    shop 每轮发全量 history，永远不会出现这个现象。
    """
    agent, llm = make_agent(
        ["第一轮回答", "第二轮回答", "第三轮回答"],
        keep_turns=1,
    )
    task = Task(id="M3", turns=["查订单 1001", "物流呢", "什么时候到"], split="fit")
    trace = agent.run(task)

    assert len(trace.turns) == 3
    last_view = llm.seen[-1]
    human_texts = [m.content for m in last_view if isinstance(m, HumanMessage)]
    assert human_texts == ["什么时候到"], human_texts


def test_slots_survive_after_history_is_trimmed():
    """第 1 轮拿到的订单号被裁掉了，但槽位把它留在 system prompt 里。

    这正是 main_agent 三层机制里"槽位"存在的唯一理由：
    窗口裁到多小，参数都还在。裁掉之后模型还能接着查，靠的就是这个。
    """
    agent, llm = make_agent(
        [
            tool_call("get_order", {"order_id": 1001}, "a"),
            "订单 1001 已发货",
            "物流已更新",
        ],
        keep_turns=1,
    )
    task = Task(id="M2", turns=["订单 1001 到哪了", "还有别的吗"], split="fit")
    agent.run(task)

    last_view = llm.seen[-1]
    system_text = last_view[0].content
    assert "1001" in system_text, "槽位没生效，裁剪后订单号彻底丢了"

    human_texts = [m.content for m in last_view if isinstance(m, HumanMessage)]
    assert "订单 1001 到哪了" not in human_texts, "第 1 轮没被裁掉，这条测试就测不到槽位"


def test_sent_msgs_recorded_per_turn():
    """每轮实际发了几条消息要留在 meta 里，否则事后无法核对裁剪是否生效。"""
    agent, _ = make_agent(["答一", "答二"], keep_turns=1)
    task = Task(id="M4", turns=["第一问", "第二问"], split="fit")
    trace = agent.run(task)

    assert trace.turns[0].meta["sent_msgs"] > 0
    assert trace.turns[1].meta["sent_msgs"] > 0


# =========================
# 三、预算与经验
# =========================


def test_runtime_budget_overrides_agent_default():
    """ctx.max_steps 生效，构造参数只做兜底。

    agent 自带 1 步，Runtime 给 3 步；脚本里第 2 步就给出答案。
    若错误地用了 agent 自己的 1 步，结果会是 max_steps。
    """
    agent, _ = make_agent(
        [tool_call("get_order", {"order_id": 1}), "查到了"],
        max_steps=1,
    )
    task = one_turn_task()
    trace = agent.run(task, RunContext(task=task, max_steps=3))

    assert trace.turns[0].meta["status"] == "ok"
    assert trace.turns[0].steps == 2
    assert trace.meta["max_steps"] == 3


def test_runtime_budget_can_lower_agent_default():
    agent, _ = make_agent(
        [tool_call("get_order", {"order_id": 1}), "查到了"],
        max_steps=5,
    )
    task = one_turn_task()
    trace = agent.run(task, RunContext(task=task, max_steps=1))

    assert trace.turns[0].meta["status"] == "max_steps"


def test_experience_reaches_system_prompt():
    agent, llm = make_agent(["好的"])
    task = one_turn_task()
    trace = agent.run(task, RunContext(task=task, experience="退款前先查订单状态", hits=["e1"]))

    assert "退款前先查订单状态" in llm.seen[0][0].content
    assert trace.meta["experience_applied"] is True
    assert trace.meta["hits"] == ["e1"]
    assert trace.meta["experience_chars"] == len("退款前先查订单状态")


def test_no_experience_leaves_prompt_untouched():
    agent, llm = make_agent(["好的"])
    task = one_turn_task()
    agent.run(task, RunContext(task=task))

    assert llm.seen[0][0].content == "P"


# =========================
# 四、接入层：名字 -> Agent，CLI 不认识具体实现
# =========================


def test_builtin_lists_both_real_agents():
    assert "shop-cold" in builtin.names()
    assert "shop-tuned" in builtin.names()
    assert "main" in builtin.names()


def test_builtin_alias_keeps_old_commands_working():
    """之前几轮真跑用的是 --agent cold / tuned，那两个名字必须还能用。"""
    assert builtin.ALIASES["cold"] == "shop-cold"
    assert builtin.ALIASES["tuned"] == "shop-tuned"


def test_builtin_make_by_name():
    agent = builtin.make("main")
    assert agent.name == "main"
    agent2 = builtin.make("cold")
    assert agent2.name == "shop-cold"


def test_builtin_unknown_name_lists_available():
    with pytest.raises(KeyError, match="当前可用"):
        builtin.make("nope")


def test_builtin_does_not_pollute_core_default_registry():
    """造 Agent 不许往 core.registry.DEFAULT 里写。

    那张表的契约是"默认必须为空"——删掉整个 agents/ 与 demos/，
    core 仍能 import 和测试，靠的就是它。接入层往里塞实例，
    等于让业务 Agent 反向污染核心全局状态；测试文件的执行顺序一变，
    core 那条空表守卫就会被牵连成红。
    """
    from core.registry import DEFAULT

    builtin.make("cold")
    assert DEFAULT.names() == []


def test_builtin_import_does_not_load_langchain():
    """import agents.builtin 是查名字用的，不该顺带把 langchain 拖进来。

    用子进程验证：当前进程早就导过 langchain 了，查 sys.modules 看不出差别。
    """
    code = (
        "import sys, agents.builtin; "
        "bad=[m for m in sys.modules if m.startswith(('langchain','httpx','agents.shop','agents.main_agent.adapter'))]; "
        "print(','.join(bad))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "", out.stdout


def test_run_loop_does_not_name_any_concrete_agent():
    """CLI 里不许出现具体 Agent 的模块路径。

    一旦有人图省事写回 from agents.shop import ...，
    "接一个新 Agent 不用改 CLI"就成了空话。这条会立刻红。
    """
    src = (ROOT / "scripts" / "run_loop.py").read_text(encoding="utf-8")
    for token in ("agents.shop", "agents.main_agent"):
        assert token not in src, f"scripts/run_loop.py 出现了具体 Agent：{token}"


def test_main_adapter_does_not_import_shop():
    """两个 Agent 必须能互相删除。main_agent 不许借 shop 的任何东西。"""
    offenders = []
    for path in (ROOT / "agents" / "main_agent").rglob("*.py"):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            s = line.strip()
            if s.startswith(("import ", "from ")) and "agents.shop" in s:
                offenders.append(f"{path.relative_to(ROOT)}:{lineno} {s}")
    assert not offenders, offenders


# =========================
# 五、经验挂载只有一份措辞
# =========================


def test_experience_attach_is_shared_between_agents():
    """两个 Agent 拿到的经验块必须逐字相同。

    否则"注入后结果不同"分不清是 Agent 的差别还是措辞的差别，
    整个对比实验就多了一个变量。
    """
    from agents.shop import with_experience

    assert with_experience("P", "先查订单") == attach("P", "先查订单")


def test_experience_attach_empty_is_noop():
    assert attach("P", "") == "P"
    assert attach("P", "   \n  ") == "P"
