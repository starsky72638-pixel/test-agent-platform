"""接入层的离线测试。全部不连模型、不连 HTTP 服务、不花额度。

覆盖三件事：

1. 提示词拆分是无损的，cold 是 tuned 的真子集。
   这是对比实验成立的前提：如果两套提示词的差异不止"少了几条规则"，
   后面所有"经验让 cold 变好"的结论都不成立。

2. 工具返回按 call_id 一对一配对，同一个工具调两次不会串。
   这是 core/trace.py 契约的硬要求，也是 replay 旧实现的一个已知缺陷。

3. 真 Agent 的 run() 能产出合法 Trace，且能被 evaluator 判分。
   用假模型跑完整链路，毫秒级验证"接入层 → Runtime → 判定"是通的。
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from agents.shop import (
    COLD_PROMPT,
    SHOP_PROMPT,
    TUNED_PROMPT,
    ShopAgent,
    calls_from_messages,
    make_shop_agent,
    with_experience,
)
from agents.shop.tools import allowed_tool_names, shop_tools
from agents.shop.view.context import ALL_RULES, COLD_RULE_COUNT, build_prompt, split_rules
from core.protocol import RunContext, validate_agent
from core.runner import run_task
from core.task import Task
from core.trace import STATUS_CRASHED, STATUS_OK, Trace
from evaluator.verdict import evaluate

ROOT = Path(__file__).resolve().parents[1]


def _history_prompt():
    """历史那一份提示词副本（main_agent 里的）。

    main_agent 被搬进 agents/ 或整目录删除后，这里返回 None，
    相关测试自动跳过——它守的是"两份文本别漂移"，
    那份副本不存在了，这条约束自然也就不存在了。
    """
    for mod in ("agents.main_agent.tools", "main_agent.tools"):
        try:
            return getattr(importlib.import_module(mod), "SHOP_SYSTEM_PROMPT")
        except (ImportError, AttributeError):
            continue
    return None


# =========================
# 一、假对象：不连服务也能跑完整链路
# =========================


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


def single_turn_task(task_id: str = "T1") -> Task:
    return Task(id=task_id, input="我的订单 1001 到哪了", split="holdout")


# =========================
# 二、提示词拆分
# =========================


def test_prompt_split_is_lossless():
    """切完再拼回去必须与原文逐字符相同。

    这条一旦红了，说明规则块的排版变了（比如编号格式换了），
    cold 的切分会跟着错位——所以它同时充当格式守卫。
    """
    assert build_prompt(ALL_RULES) == SHOP_PROMPT


def test_prompt_matches_history_copy():
    """本实现里的提示词，必须与 main_agent 那份历史副本逐字相同。

    历史报告里的 train_messages[0].content 记的就是那份文本。
    两边一旦漂移，拿新轨迹去和历史报告对口径就毫无意义。

    失败时刻意只报第一处差异的前后 30 字，而不是把两整段 900 字的提示词
    全打出来——那玩意儿在终端里根本没法比对。
    """
    hist = _history_prompt()
    if hist is None:
        pytest.skip("历史副本已不可用（main_agent 已移除）")
    if SHOP_PROMPT == hist:
        return

    limit = min(len(SHOP_PROMPT), len(hist))
    at = next((i for i in range(limit) if SHOP_PROMPT[i] != hist[i]), None)
    if at is None:
        pytest.fail(f"长度不同：新 {len(SHOP_PROMPT)} 字，历史 {len(hist)} 字")
    pytest.fail(
        f"第 {at} 个字符起不同\n"
        f"  新  : {SHOP_PROMPT[max(0, at - 30):at + 30]!r}\n"
        f"  历史: {hist[max(0, at - 30):at + 30]!r}"
    )


def test_rule_count_is_eleven():
    assert len(ALL_RULES) == 11


def test_split_rejects_prompt_without_rules():
    with pytest.raises(ValueError, match="规则"):
        split_rules("你是一个助手，没有规则块")


def test_tuned_is_the_original_text():
    """tuned 必须原样引用，不做重组——它要跟历史报告同源。"""
    assert TUNED_PROMPT is SHOP_PROMPT


def test_cold_is_a_true_prefix_of_tuned():
    """最强的子集证明：tuned 以 cold 开头。

    这意味着两套提示词的差异只可能是"后面还有 8 条"，不可能有
    "第 2 条被改写了"这种悄悄发生的偏移。
    """
    assert TUNED_PROMPT.startswith(COLD_PROMPT)
    tail = TUNED_PROMPT[len(COLD_PROMPT):]
    assert tail.lstrip("\n") == "\n".join(ALL_RULES[COLD_RULE_COUNT:])


def test_cold_rules_all_appear_in_tuned_verbatim():
    for rule in ALL_RULES[:COLD_RULE_COUNT]:
        assert rule in TUNED_PROMPT


def test_cold_keeps_only_baseline_rules():
    """cold 只保留"必须查工具 / 禁止补全 / 状态原样复述"这三条底线。"""
    assert COLD_RULE_COUNT == 3
    assert COLD_PROMPT.endswith(ALL_RULES[COLD_RULE_COUNT - 1])
    for later in ALL_RULES[COLD_RULE_COUNT:]:
        assert later not in COLD_PROMPT


# =========================
# 三、经验挂载
# =========================


def test_with_experience_empty_returns_prompt_unchanged():
    assert with_experience("P", "") == "P"
    assert with_experience("P", "   \n  ") == "P"


def test_with_experience_appends_after_prompt():
    out = with_experience("P", "先查订单再退款")
    assert out.startswith("P")
    assert out.rstrip().endswith("先查订单再退款")
    assert "先查订单再退款" in out


def test_with_experience_warns_not_to_override_tool_output():
    """经验里可能带示例值，必须明确它不能覆盖工具返回。"""
    out = with_experience("P", "x")
    assert "以工具返回为准" in out


# =========================
# 四、工具返回的配对
# =========================


def test_calls_pair_by_id_not_by_name():
    """同一个工具调两次，各自拿各自的返回。

    这正是 replay 里 outputs 按工具名合并会出错的地方：
    合并之后两次调用都会看到两段返回拼在一起的内容。
    """
    msgs = [
        tool_call("get_order", {"order_id": 1001}, "a"),
        tool_call("get_order", {"order_id": 1002}, "b"),
        _tool_msg("第一次返回", "a"),
        _tool_msg("第二次返回", "b"),
    ]
    calls = calls_from_messages(msgs, 0)
    assert [c.args["order_id"] for c in calls] == [1001, 1002]
    assert [c.output for c in calls] == ["第一次返回", "第二次返回"]


def _tool_msg(text: str, call_id: str):
    from langchain_core.messages import ToolMessage

    return ToolMessage(content=text, tool_call_id=call_id)


def test_call_ids_follow_turn_convention():
    msgs = [
        tool_call("get_order", {"order_id": 1}, "x"),
        tool_call("get_logistics", {"order_id": 1}, "y"),
    ]
    calls = calls_from_messages(msgs, 3)
    assert [c.call_id for c in calls] == ["t3c0", "t3c1"]


def test_empty_call_id_does_not_collide():
    msgs = [tool_call("get_order", {}, ""), tool_call("get_order", {}, "")]
    assert len(calls_from_messages(msgs, 0)) == 2


def test_tool_error_marks_call_not_ok():
    msgs = [tool_call("get_order", {}, "a"), _tool_msg("ERROR 请求超时", "a")]
    calls = calls_from_messages(msgs, 0)
    assert calls[0].ok is False
    assert "超时" in calls[0].error


def test_no_calls_returns_empty():
    assert calls_from_messages([AIMessage(content="直接回答")], 0) == []


def test_orphan_tool_message_is_dropped():
    """找不到对应调用的返回不猜、不补，直接丢掉。"""
    assert calls_from_messages([_tool_msg("孤儿返回", "nope")], 0) == []


# =========================
# 五、接入协议
# =========================


def test_agent_satisfies_protocol():
    validate_agent(ShopAgent("tuned", prompt="P", tools=[]))


def test_factory_names_include_variant():
    assert make_shop_agent("cold").name == "shop-cold"
    assert make_shop_agent("tuned").name == "shop-tuned"


def test_unknown_variant_raises():
    with pytest.raises(KeyError):
        ShopAgent("nonexistent")


def test_tools_and_whitelist_agree():
    names = {t.name for t in shop_tools()}
    assert names == allowed_tool_names()
    assert len(names) == 5


def test_shop_tools_does_not_share_list_between_agents():
    a, b = make_shop_agent("tuned"), make_shop_agent("tuned")
    a.tools.append("junk")
    assert "junk" not in b.tools


# =========================
# 六、run() 的完整链路
# =========================


def _agent_with(steps, tools=None, **kwargs) -> tuple[ShopAgent, ScriptedLLM]:
    llm = ScriptedLLM(steps)
    agent = ShopAgent(
        "tuned",
        prompt="P",
        tools=tools if tools is not None else [FakeTool("get_order", "已签收")],
        llm=llm,
        **kwargs,
    )
    return agent, llm


def test_run_produces_valid_trace():
    agent, _ = _agent_with([tool_call("get_order", {"order_id": 1001}), "订单 1001 已签收"])
    trace = agent.run(single_turn_task())

    assert isinstance(trace, Trace)
    assert trace.status == STATUS_OK
    assert trace.agent == "shop"
    assert len(trace.turns) == 1
    assert trace.turns[0].calls[0].name == "get_order"
    assert trace.turns[0].calls[0].output == "已签收"
    assert trace.turns[0].answer == "订单 1001 已签收"
    assert trace.final_answer() == "订单 1001 已签收"


def test_run_multi_turn_keeps_index_and_slices_calls():
    """第二轮只算第二轮自己的调用，第一轮的不能重复计进来。"""
    agent, llm = _agent_with(
        [
            tool_call("get_order", {"order_id": 1001}, "a"),
            "订单 1001 已签收",
            tool_call("get_logistics", {"order_id": 1001}, "b"),
            "已由顺丰送达",
        ]
    )
    task = Task(id="M1", turns=["查订单 1001", "物流呢"], split="fit")
    trace = agent.run(task)

    assert [t.index for t in trace.turns] == [0, 1]
    assert trace.turns[0].call_names() == ["get_order"]
    assert trace.turns[1].call_names() == ["get_logistics"]
    assert trace.turns[1].calls[0].call_id == "t1c0"


def test_max_steps_status_lands_in_meta():
    """max_steps 必须写进 meta，否则 evaluator 会把它当成正常结束。

    步数用尽时不再交白卷：会拿已有的工具结果收个尾。
    这条断言跟着行为一起改——旧的 answer == "" 正是本次要修的问题，
    但状态仍必须是 max_steps，不能靠一段兜底话把"没做完"粉饰成"做完了"。
    """
    agent, _ = _agent_with([tool_call("get_order", {"order_id": 1001})], max_steps=1)
    trace = agent.run(single_turn_task())

    assert trace.turns[0].meta["status"] == "max_steps"
    assert trace.turns[0].answer.strip() != ""


def test_tool_error_is_retried_once(monkeypatch):
    """服务端 5xx 要用同一参数自动重试一次，并把重试记进 meta。

    没有这条时，模型看到 status=500 只能自己猜要不要再试，
    实测里它有时直接放弃、有时换个工具瞎查。
    """
    monkeypatch.setenv("SHOP_RETRY_LIMIT", "1")

    class FlakyTool:
        name = "get_order"

        def __init__(self) -> None:
            self.n = 0

        def invoke(self, args):
            self.n += 1
            if self.n == 1:
                return 'status=500\nbody={"error": "boom"}'
            return "status=200\nbody={\"status\": \"已签收\"}"

    tool = FlakyTool()
    agent, _ = _agent_with(
        [tool_call("get_order", {"order_id": 1001}), "订单 1001 已签收"],
        tools=[tool],
    )
    trace = agent.run(single_turn_task())

    assert tool.n == 2
    assert trace.turns[0].answer == "订单 1001 已签收"
    record = trace.turns[0].meta["recovery"][0]
    assert record["attempts"] == 2 and record["kind"] == "ok"


def test_retry_limit_zero_keeps_single_attempt(monkeypatch):
    """SHOP_RETRY_LIMIT=0 必须退回"只试一次"，对照实验靠它。

    只切这一个环境变量就能对比"有恢复/无恢复"，
    不需要回滚代码——否则两次运行之间改的是两处，结论不成立。
    """
    monkeypatch.setenv("SHOP_RETRY_LIMIT", "0")

    class DeadTool:
        name = "get_order"

        def __init__(self) -> None:
            self.n = 0

        def invoke(self, args):
            self.n += 1
            return 'status=503\nbody={"error": "unavailable"}'

    tool = DeadTool()
    agent, _ = _agent_with(
        [tool_call("get_order", {"order_id": 1001}), "暂时查不到"],
        tools=[tool],
    )
    trace = agent.run(single_turn_task())

    assert tool.n == 1
    assert trace.turns[0].meta["recovery"][0]["attempts"] == 1


def test_experience_reaches_system_prompt():
    agent, llm = _agent_with(["好的"])
    ctx = RunContext(task=single_turn_task(), experience="退款前先查订单状态", hits=["e1"])
    trace = agent.run(single_turn_task(), ctx)

    system_text = llm.seen[0][0].content
    assert "退款前先查订单状态" in system_text
    assert trace.meta["experience_applied"] is True
    assert trace.meta["hits"] == ["e1"]


def test_no_experience_leaves_prompt_untouched():
    agent, llm = _agent_with(["好的"])
    agent.run(single_turn_task(), RunContext(task=single_turn_task()))

    assert llm.seen[0][0].content == "P"


def test_llm_failure_becomes_crashed_trace():
    """模型/网络挂了必须落成 crashed，而不是一条看起来成功的空轨迹。"""

    class BoomLLM:
        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            raise ConnectionError("Connection refused")

    agent = ShopAgent("tuned", prompt="P", tools=[], llm=BoomLLM(), retries=0)
    trace = run_task(agent, single_turn_task())

    assert trace.status == STATUS_CRASHED
    assert "Connection refused" in trace.error


def test_crashed_trace_is_not_evaluable():
    class BoomLLM:
        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            raise RuntimeError("boom")

    agent = ShopAgent("tuned", prompt="P", tools=[], llm=BoomLLM(), retries=0)
    task = single_turn_task()
    verdict = evaluate(run_task(agent, task), task)

    assert verdict.passed is False
    assert verdict.failure_type == "infra_error"


# =========================
# 七、接入之后判分链路是通的
# =========================


def test_trace_from_agent_can_be_judged():
    """假 Agent 跑出来的轨迹能走完五层判定，结果符合断言配置。"""
    agent, _ = _agent_with([tool_call("get_order", {"order_id": 1001}), "订单 1001 已签收"])
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


def test_agent_trace_reports_missing_tool():
    agent, _ = _agent_with(["订单已签收"])
    task = Task(
        id="T2",
        input="订单 1001 什么状态",
        split="holdout",
        assertions={"expect_tools": ["get_order"]},
    )
    verdict = evaluate(agent.run(task), task)

    assert verdict.passed is False
    assert verdict.failure_type == "missing_tool"


# =========================
# 八、分层边界
# =========================


CORE_DIRS = ("core", "evaluator", "memory")


def test_core_modules_do_not_import_agents():
    """核心不能被接入层反向依赖——否则删掉 agents/ 核心就跑不起来。"""
    offenders = []
    for name in CORE_DIRS:
        for path in (ROOT / name).glob("*.py"):
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                s = line.strip()
                if s.startswith(("import ", "from ")) and "agents" in s:
                    offenders.append(f"{path.name}:{lineno} {s}")
    assert not offenders, offenders


def test_agents_package_does_not_import_demos():
    """接入层不该依赖示例包，它们是两个独立方向。"""
    for path in (ROOT / "agents").rglob("*.py"):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            s = line.strip()
            if s.startswith(("import ", "from ")):
                assert "demos" not in s, f"{path}:{lineno} {s}"


def test_shop_does_not_import_main_agent():
    """agents/shop 必须独立于 agents/main_agent。

    这是"删掉 main_agent 照样跑"的可执行证据，也是这一轮重构的核心目的。
    之前那一版把循环和工具都从 main_agent 借过来，结构上分了层、
    实质上还是绑死的——这条测试就是防止退回那种写法。
    """
    offenders = []
    for path in (ROOT / "agents" / "shop").rglob("*.py"):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            s = line.strip()
            if s.startswith(("import ", "from ")) and "main_agent" in s:
                offenders.append(f"{path.relative_to(ROOT)}:{lineno} {s}")
    assert not offenders, offenders


def test_agents_root_package_stays_empty():
    """agents/__init__.py 是空入口，不该 import 具体 Agent。

    一旦这里写了 from agents.shop import ...，`import agents` 就会连带把
    langchain 与 httpx 拉进内存——核心侧做一次最简检查也被拖慢，
    分层也就白分了。
    """
    src = (ROOT / "agents" / "__init__.py").read_text(encoding="utf-8")
    for line in src.splitlines():
        s = line.strip()
        if s.startswith(("import ", "from ")):
            assert "agents.shop" not in s, s
