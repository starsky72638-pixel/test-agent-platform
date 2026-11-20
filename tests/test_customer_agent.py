"""售后客服 Agent 的离线测试：不连模型、不连 HTTP、不花额度。

盯五件事：
1. 只绑 3 个售后工具，售前的搜索/详情不在里面；
2. run() 能产出合法 Trace，工具调用按 core 的编号约定落盘；
3. 经验能进系统提示词，且明确要求"以工具返回为准"；
4. 注册表里确实有 customer，别名 cs 指向它；
5. agents/customer 不依赖 agents/shop —— "删掉 shop 照样跑"的可执行证据。
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage

from agents.customer import CustomerAgent, allowed_tool_names, customer_tools
from core.protocol import RunContext, validate_agent
from core.task import Task
from core.trace import STATUS_OK, Trace

ROOT = Path(__file__).resolve().parents[1]


class FakeTool:
    """只认 name 和 invoke 的工具替身，行为完全由构造时给定。"""

    def __init__(self, name: str, output: str) -> None:
        self.name = name
        self.output = output

    def invoke(self, args):
        return self.output


class ScriptedLLM:
    """按脚本逐次返回消息的模型替身。

    每次 invoke 都把自己收到的消息列表存下来，
    这样"经验有没有真的进到提示词"可以被直接检查，而不是靠推断。
    """

    def __init__(self, steps) -> None:
        self.steps = list(steps)
        self.calls = 0
        self.seen: list[list] = []

    def bind_tools(self, tools):
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


def make_agent(steps, tools=None, **kwargs) -> tuple[CustomerAgent, ScriptedLLM]:
    llm = ScriptedLLM(steps)
    agent = CustomerAgent(
        prompt="P",
        tools=tools if tools is not None else [FakeTool("get_order", "status=200")],
        llm=llm,
        **kwargs,
    )
    return agent, llm


# =========================
# 一、工具集与协议
# =========================


def test_only_three_after_sales_tools():
    """客服只绑售后三件套。售前工具绑上来会让"工具选择"这一层失效。"""
    names = {t.name for t in customer_tools()}
    assert names == {"get_order", "get_logistics", "apply_refund"}
    assert names == allowed_tool_names()
    assert "search_products" not in names
    assert "get_product_detail" not in names


def test_agent_satisfies_protocol():
    validate_agent(CustomerAgent(prompt="P", tools=[]))


def test_tools_list_is_not_shared_between_agents():
    a, b = CustomerAgent(prompt="P"), CustomerAgent(prompt="P")
    a.tools.append("junk")
    assert "junk" not in b.tools


# =========================
# 二、run() 的完整链路
# =========================


def test_run_produces_valid_trace():
    agent, _ = make_agent([tool_call("get_order", {"order_id": 1002}), "订单 1002 已发货"])
    task = Task(id="C1", input="1002 到哪了", split="holdout")
    trace = agent.run(task)

    assert isinstance(trace, Trace)
    assert trace.status == STATUS_OK
    assert trace.agent == "customer"
    assert len(trace.turns) == 1
    assert trace.turns[0].calls[0].name == "get_order"
    # core 的编号约定：t{轮}c{序}
    assert trace.turns[0].calls[0].call_id == "t0c0"
    assert trace.turns[0].answer == "订单 1002 已发货"
    assert trace.final_answer() == "订单 1002 已发货"


def test_max_steps_status_lands_in_meta():
    """max_steps 必须写进 meta，否则 evaluator 会把它当成正常结束。"""
    agent, _ = make_agent([tool_call("get_order", {"order_id": 1002})], max_steps=1)
    trace = agent.run(Task(id="C2", input="1002 到哪了", split="holdout"))

    assert trace.turns[0].meta["status"] == "max_steps"


def test_experience_reaches_system_prompt():
    agent, llm = make_agent(["好的"])
    task = Task(id="C3", input="我要退款", split="holdout")
    ctx = RunContext(task=task, experience="退款前先查订单状态", hits=["e1"])
    trace = agent.run(task, ctx)

    system_text = llm.seen[0][0].content
    assert "退款前先查订单状态" in system_text
    # 经验里可能带示例值，必须明确它不能覆盖工具返回
    assert "以工具返回为准" in system_text
    assert trace.meta["experience_applied"] is True
    assert trace.meta["hits"] == ["e1"]


def test_no_experience_leaves_prompt_untouched():
    agent, llm = make_agent(["好的"])
    agent.run(Task(id="C4", input="1002 到哪了", split="holdout"))

    assert llm.seen[0][0].content == "P"


# =========================
# 三、接入与分层边界
# =========================


def test_registered_in_builtin():
    """加一行注册就能按名字构造，入口脚本不需要认识 CustomerAgent。"""
    import agents.builtin as builtin

    assert "customer" in builtin.names()
    assert builtin.ALIASES["cs"] == "customer"

    # 真的走一次工厂：构造不读配置（llm 延迟到 run 时才建），离线安全
    agent = builtin.make("customer")
    assert agent.name == "customer"


def test_cold_variant_differs_in_prompt_and_tool_descriptions_only():
    """冷变体只改两样：系统提示词、工具描述。

    为什么是两样：业务约束同时存在于这两处。apply_refund 的描述里写着
    "调用前先 get_order 确认订单状态"，工具描述每轮都随 schema 进上下文，
    只删 prompt 等于没删约束（上一轮就是这么得出假 null 的）。

    为什么不许改别的：函数体、工具名、参数名必须与常态完全一致，
    否则"基线失败、注入后通过"就分不清是经验带来的还是工具行为变了带来的。
    """
    from agents.customer import COLD_PROMPT, COLD_TOOLS

    cold = CustomerAgent(prompt=COLD_PROMPT, tools=list(COLD_TOOLS), name="customer-cold")
    warm = CustomerAgent()

    assert cold.prompt != warm.prompt
    assert cold.max_steps == warm.max_steps
    assert [t.name for t in cold.tools] == [t.name for t in warm.tools]

    for c, w in zip(cold.tools, warm.tools):
        # 同一个函数对象：工具的行为一行没改
        assert c.func is w.func
        assert list(c.args.keys()) == list(w.args.keys())
        # 描述必须真的更短，且不含任何行为准则
        assert len(c.description) < len(w.description)
        for banned in ("必须先", "先 get_order", "不要", "不能", "必填", "再查"):
            assert banned not in c.description

    # 冷提示词里也不许再有行为准则，否则等于把答案提前写进去
    for banned in ("必须先", "不能编造", "先问清楚"):
        assert banned not in COLD_PROMPT


def test_cold_variant_is_registered():
    import agents.builtin as builtin

    from agents.customer.agent import DEFAULT_PROMPT

    assert "customer-cold" in builtin.names()
    assert builtin.ALIASES["cs-cold"] == "customer-cold"

    agent = builtin.make("cs-cold")
    assert agent.name == "customer-cold"
    assert agent.prompt != DEFAULT_PROMPT
    # 描述必须真的是初稿水平，否则对照实验的自变量又没删干净
    warm = builtin.make("customer")
    assert all(len(t.description) < len(w.description)
               for t, w in zip(agent.tools, warm.tools))
    # 工具名不变：判分的 allowed_tools 与 expect_tools 才继续对得上
    assert [t.name for t in agent.tools] == [t.name for t in warm.tools]


def test_customer_does_not_import_shop():
    """agents/customer 必须独立于 agents/shop。

    这是"删掉 shop 照样跑"的可执行证据，也是 T5-b 的核心目的：
    第二个业务 Agent 接入时，既不需要复制第一个 Agent 的实现，
    也不允许反过来依赖它。tools 与循环都各有一份，正是为此。

    与 test_shop_does_not_import_main_agent 同一种写法，只换被扫的目录。
    """
    offenders = []
    for path in (ROOT / "agents" / "customer").rglob("*.py"):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            s = line.strip()
            if s.startswith(("import ", "from ")) and "agents.shop" in s:
                offenders.append(f"{path.relative_to(ROOT)}:{lineno} {s}")
    assert not offenders, offenders
