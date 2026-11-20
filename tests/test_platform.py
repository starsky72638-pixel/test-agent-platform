"""Workflow 引擎的离线测试。

三件事必须钉住，因为它们是这个阶段唯一的产出：
1. 坏定义在执行前就被拒绝，且错误能定位到具体节点；
2. 合法定义能真跑，事件来自真实执行而不是拼出来的；
3. Turn 语义不被破坏——单轮 workflow 仍然是 1 个 Turn，工具调用只在事件里。

全程不联网：模型与工具都是注入的假对象。
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import agents.builtin as builtin
from agentplatform import (
    NODE_CONDITION,
    NODE_END,
    NODE_LLM,
    NODE_MEMORY,
    NODE_START,
    NODE_TOOL,
    DefinitionError,
    ExtractError,
    WorkflowAgent,
    WorkflowDefinition,
    compare,
    extract_field,
    topo_sort,
    validate,
)
from agentplatform.executor import Executor, TemplateError, render
from agentplatform.topology import check as check_topology
from core.protocol import RunContext, validate_agent
from core.task import Task
from core.trace import (
    KIND_BRANCH,
    KIND_LLM_CALL,
    KIND_MEMORY_CALL,
    KIND_NODE_END,
    KIND_NODE_START,
    KIND_TOOL_CALL,
    STATUS_INFRA,
    STATUS_OK,
    Trace,
)
from evaluator.verdict import calls_of_turn
from memory.extract import task_tags
from memory.schema import Experience, Trigger
from memory.store import ExperienceStore

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "data" / "workflows" / "demo.json"


# ---------------------------------------------------------------- 假对象


class FakeTool:
    """模拟 LangChain 工具：有 name、用 invoke，可指定返回值或抛错。"""

    def __init__(self, name: str = "get_order", output: str = "status=200", error: str = "") -> None:
        self.name = name
        self.output = output
        self.error = error
        self.received: list[dict] = []

    def invoke(self, args: dict):
        self.received.append(dict(args))
        if self.error:
            raise RuntimeError(self.error)
        return self.output


class FakeLLM:
    def __init__(self, text: str = "订单已发货，金额 899 元") -> None:
        self.text = text
        self.prompts: list[str] = []

    def invoke(self, prompt: str):
        self.prompts.append(prompt)
        return SimpleNamespace(content=self.text)


def make_definition(*, extra_nodes=None, workflow_id="t1", tools=None) -> WorkflowDefinition:
    nodes = [
        {"id": "start", "type": NODE_START, "next": ["lookup"]},
        {
            "id": "lookup",
            "type": NODE_TOOL,
            "config": {"tool": "get_order", "args": {"order_id": 1002}},
            "next": ["explain"],
        },
        {
            "id": "explain",
            "type": NODE_LLM,
            "config": {"prompt": "用户问：{{input}}；接口返回：{{nodes.lookup.output}}"},
            "next": ["end"],
        },
        {"id": "end", "type": NODE_END, "next": []},
    ]
    raw = {"workflow_id": workflow_id, "entry": "start", "nodes": nodes}
    if tools is not None:
        raw["tools"] = tools
    if extra_nodes:
        raw["nodes"] = nodes + list(extra_nodes)
    return WorkflowDefinition.from_dict(raw)


def make_agent(**kwargs) -> WorkflowAgent:
    tool = kwargs.pop("tool", None) or FakeTool()
    llm = kwargs.pop("llm", None) or FakeLLM()
    return WorkflowAgent(
        kwargs.pop("definition", None) or make_definition(),
        tools=kwargs.pop("tools", None) or [tool],
        llm=llm,
        **kwargs,
    )


# ---------------------------------------------------------------- 定义


def test_demo_definition_loads():
    defn = WorkflowDefinition.load(DEMO)

    assert defn.workflow_id == "demo"
    assert defn.entry == "start"
    assert defn.node_ids() == ["start", "lookup", "explain", "end"]
    assert defn.tools == ["get_order"]
    assert defn.node("lookup").config["tool"] == "get_order"


def test_definition_roundtrip():
    defn = make_definition()
    assert WorkflowDefinition.from_dict(defn.to_dict()).to_dict() == defn.to_dict()


def test_definition_rejects_broken_json(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text("{ not json", encoding="utf-8")
    with pytest.raises(DefinitionError, match="不是合法 JSON"):
        WorkflowDefinition.load(bad)


# ---------------------------------------------------------------- 校验


def test_schema_reports_unknown_node_type():
    raw = {
        "workflow_id": "t",
        "entry": "start",
        "nodes": [
            {"id": "start", "type": NODE_START, "next": ["x"]},
            {"id": "x", "type": "magic", "next": ["end"]},
            {"id": "end", "type": NODE_END},
        ],
    }
    errors = validate(WorkflowDefinition.from_dict(raw))
    assert any("magic" in e and "不支持" in e for e in errors)


def test_schema_reports_entry_not_start():
    raw = {
        "workflow_id": "t",
        "entry": "end",
        "nodes": [
            {"id": "start", "type": NODE_START, "next": ["end"]},
            {"id": "end", "type": NODE_END},
        ],
    }
    errors = validate(WorkflowDefinition.from_dict(raw))
    assert any("必须是 start" in e for e in errors)


def test_schema_reports_dangling_next():
    raw = {
        "workflow_id": "t",
        "entry": "start",
        "nodes": [
            {"id": "start", "type": NODE_START, "next": ["ghost"]},
            {"id": "end", "type": NODE_END},
        ],
    }
    errors = validate(WorkflowDefinition.from_dict(raw))
    assert any("ghost" in e for e in errors)


def test_schema_reports_llm_without_prompt():
    defn = make_definition()
    defn.node("explain").config = {}
    errors = validate(defn)
    assert any("config.prompt" in e for e in errors)


def test_schema_reports_tool_not_registered():
    defn = make_definition()
    errors = validate(defn, available_tools={"search_products"})
    assert any("get_order" in e and "未注册" in e for e in errors)


def test_schema_reports_declared_tool_never_used():
    defn = make_definition(tools=["get_order", "get_logistics"])
    errors = validate(defn, available_tools={"get_order", "get_logistics"})
    assert any("get_logistics" in e and "没有任何 tool 节点使用" in e for e in errors)


# ---------------------------------------------------------------- 拓扑


def test_topology_detects_cycle():
    defn = make_definition()
    # 让 explain 指回 lookup：lookup -> explain -> lookup，构成环
    defn.node("explain").next = ["lookup"]
    errors = check_topology(defn)
    assert any("环" in e for e in errors)


def test_topology_detects_unreachable():
    raw = {
        "workflow_id": "t",
        "entry": "start",
        "nodes": [
            {"id": "start", "type": NODE_START, "next": ["end"]},
            {"id": "end", "type": NODE_END},
            {"id": "orphan", "type": NODE_END},
        ],
    }
    errors = check_topology(WorkflowDefinition.from_dict(raw))
    assert any("orphan" in e and "不可达" in e for e in errors)


def test_topo_order_respects_edges_and_is_stable():
    defn = make_definition()
    assert [n.id for n in topo_sort(defn)] == ["start", "lookup", "explain", "end"]
    # 同样输入排两次，顺序必须一致，否则两次轨迹无法逐条对比
    assert [n.id for n in topo_sort(defn)] == [n.id for n in topo_sort(defn)]


def test_agent_rejects_bad_definition_at_construction():
    """坏定义在构造时就该炸，而不是等用户点运行。"""
    defn = make_definition()
    defn.node("explain").config = {}
    with pytest.raises(DefinitionError, match="config.prompt"):
        WorkflowAgent(defn, tools=[FakeTool()])


# ---------------------------------------------------------------- 执行


def test_executor_emits_real_events_in_order():
    tool = FakeTool(output="status=200 amount=899")
    llm = FakeLLM("已发货，899 元")
    agent = make_agent(tool=tool, llm=llm)
    task = Task(id="T1", input="订单 1002 到哪了", split="fit")

    trace = agent.run(task)

    assert trace.status == STATUS_OK
    assert trace.meta["workflow"]["id"] == "t1"
    # 单轮 workflow 只能有 1 个 Turn，节点数不能变成轮次
    assert len(trace.turns) == 1
    assert trace.turns[0].answer == "已发货，899 元"
    # 工具调用只在事件里；写进 Turn.calls 会让判分的 call_count 翻倍
    assert trace.turns[0].calls == []

    kinds = [e.kind for e in trace.events]
    assert kinds.count(KIND_NODE_START) == 4
    assert kinds.count(KIND_NODE_END) == 4
    assert kinds.count(KIND_TOOL_CALL) == 1
    assert kinds.count(KIND_LLM_CALL) == 1
    # seq 必须连续递增，这是"事件保序"的唯一依据
    assert [e.seq for e in trace.events] == list(range(len(trace.events)))
    # started_at 必须落盘：否则轨迹只有相对耗时，
    # 无法把节点耗时归到绝对时间轴（并行分支、总耗时归因都要靠它）
    assert all(e.started_at > 0 for e in trace.events)

    tool_event = next(e for e in trace.events if e.kind == KIND_TOOL_CALL)
    assert tool_event.name == "get_order"
    assert tool_event.args == {"order_id": 1002}
    assert "899" in tool_event.output

    # 真跑了：假工具收到了真实参数，假模型收到了渲染后的 prompt
    assert tool.received == [{"order_id": 1002}]
    assert "status=200 amount=899" in llm.prompts[0]


def test_all_calls_sees_workflow_tool_calls():
    """判分、蒸馏、Memory 都走 all_calls，它必须看得见事件里的调用。"""
    trace = make_agent().run(Task(id="T1", input="q"))
    assert [c.name for c in trace.all_calls()] == ["get_order"]


def test_tool_failure_marks_infra_and_keeps_event():
    agent = make_agent(tool=FakeTool(error="连不上服务"))
    trace = agent.run(Task(id="T1", input="q"))

    assert trace.status == STATUS_INFRA
    assert "lookup" in trace.error
    failed = [e for e in trace.events if e.kind == KIND_TOOL_CALL and not e.ok]
    assert failed and "连不上服务" in failed[0].error
    # 崩了也不能丢轮次：否则评测会当成"没有执行"
    assert len(trace.turns) == 1


def test_multi_turn_produces_one_turn_per_input():
    agent = make_agent()
    trace = agent.run(Task(id="T1", turns=["第一轮", "第二轮"]))

    assert len(trace.turns) == 2
    assert [t.index for t in trace.turns] == [0, 1]
    assert {e.turn for e in trace.events} == {0, 1}
    assert all(t.calls == [] for t in trace.turns)


# ---------------------------------------------------------------- 跨轮上下文


def test_turns_share_one_context_and_archive_each_turn():
    """多轮共用一个 ctx；历史按轮回档，当前轮只装本轮的产出。"""
    executor = Executor(
        make_definition(), tools={"get_order": FakeTool()}, llm=FakeLLM()
    )
    ctx = executor.new_context()
    trace = Trace(task_id="T1", agent="t1")

    executor.run(trace, turn=0, user_input="第一轮", ctx=ctx)
    executor.run(trace, turn=1, user_input="第二轮", ctx=ctx)

    assert ctx["input"] == "第二轮"
    assert [h["turn"] for h in ctx["history"]] == [0, 1]
    assert [h["input"] for h in ctx["history"]] == ["第一轮", "第二轮"]
    assert sorted(ctx["nodes"]) == ["end", "explain", "lookup", "start"]
    # 历史是快照，不是当前轮那块可变 dict 的引用：
    # 改当前轮的产出，已归档的那一轮必须原封不动。
    ctx["nodes"]["lookup"]["output"] = "被改过"
    assert ctx["history"][0]["nodes"]["lookup"]["output"] == "status=200"


def test_current_turn_does_not_see_previous_turn_outputs():
    """上一轮的产出若留在 ctx["nodes"]，本轮的 {{nodes.x.output}} 会读到旧值。"""
    executor = Executor(
        make_definition(), tools={"get_order": FakeTool()}, llm=FakeLLM()
    )
    ctx = executor.new_context()
    ctx["nodes"] = {"lookup": {"output": "上一轮的旧值"}}

    executor.run(Trace(task_id="T1", agent="t1"), turn=1, user_input="第二轮", ctx=ctx)

    # lookup 是这一轮真跑出来的，不是那份"旧值"
    assert ctx["nodes"]["lookup"]["output"] == "status=200"


def test_two_tasks_do_not_share_context():
    """换一条 Task 必须换 ctx：否则上一条的轮次会算进这一条的上下文。"""
    llm = FakeLLM()
    agent = make_agent(tool=FakeTool(), llm=llm)

    first = agent.run(Task(id="A", turns=["订单 1002 的状态", "订单 1002 的金额"]))
    second = agent.run(Task(id="B", input="订单 1003 的金额"))

    assert first.meta["context_turns"] == 2
    assert second.meta["context_turns"] == 1
    assert [t.index for t in second.turns] == [0]
    assert {e.turn for e in second.events} == {0}
    # 第二条任务的 prompt 里不能出现上一条的输入
    assert "订单 1003" in llm.prompts[-1]
    assert "订单 1002" not in llm.prompts[-1]


def test_multi_turn_calls_are_judged_per_turn():
    """事件按轮归属，判分器按轮取调用——两轮的调用不能混成一份。"""
    agent = make_agent()
    trace = agent.run(Task(id="T1", turns=["订单 1002 的状态", "订单 1002 的金额"]))

    assert [c.name for c in calls_of_turn(trace, trace.turns[0])] == ["get_order"]
    assert [c.name for c in calls_of_turn(trace, trace.turns[1])] == ["get_order"]
    # 每轮 4 个节点（start/lookup/explain/end），步数不会跨轮累加
    assert trace.turns[0].steps == 4
    assert trace.turns[1].steps == 4


def test_single_turn_context_shape_is_unchanged():
    """单轮路径对外形状不变：1 个 Turn、1 条上下文轮次、事件全归 turn 0。"""
    trace = make_agent().run(Task(id="T1", input="订单 1002 的状态"))

    assert len(trace.turns) == 1
    assert trace.meta["context_turns"] == 1
    assert {e.turn for e in trace.events} == {0}


def test_missing_order_id_still_fails_instead_of_reusing_previous_turn():
    """P8-a 不做槽位回退：第 2 轮没单号就报错，不偷偷沿用第 1 轮的值。"""
    defn = WorkflowDefinition.load(ROOT / "data" / "workflows" / "report-demo.json")
    agent = WorkflowAgent(
        defn,
        tools=[FakeTool(name="get_order"), FakeTool(name="get_logistics")],
        llm=FakeLLM(),
    )

    trace = agent.run(Task(id="T1", turns=["订单 1002 的状态", "那物流呢"]))

    assert trace.status == STATUS_INFRA
    assert "抽不到 order_id" in trace.error
    # 崩溃轮仍然留一个 Turn（meta 标 error），但不会被归档进上下文
    assert len(trace.turns) == 2
    assert trace.turns[1].meta["status"] == "error"
    assert trace.meta["context_turns"] == 1


def test_render_rejects_unknown_variable():
    with pytest.raises(TemplateError, match="ghost"):
        render("{{nodes.ghost.output}}", {"input": "", "nodes": {}})


def test_workflow_agent_satisfies_agent_protocol():
    validate_agent(make_agent())


def test_workflow_registered_in_builtin():
    assert "workflow" in builtin.names()
    assert builtin.ALIASES["wf"] == "workflow"
    # 走真实注册链：能按名字造出来，说明接入层不依赖 platform 的内部细节
    agent = builtin.make("workflow", workflow="demo")
    assert agent.name == "wf:demo"
    assert "get_order" in [t.name for t in agent.tools]


# ---------------------------------------------------------------- 分支
#
# condition 第一版只做互斥两支：不并行、不做 AND 汇聚。
# 这三条钉住的是"未选中分支不留任何痕迹"，它比"走对了哪支"更重要——
# 半条轨迹会让人以为那个节点跑过，事后所有归因都跟着偏。

BRANCH_RAW = {
    "workflow_id": "br",
    "entry": "start",
    "nodes": [
        {"id": "start", "type": NODE_START, "next": ["lookup"]},
        {
            "id": "lookup",
            "type": NODE_TOOL,
            "config": {"tool": "get_order", "args": {"order_id": 1002}},
            "next": ["check"],
        },
        {
            # next 故意不写：condition 的出边由 branches 派生，少写一份真相。
            "id": "check",
            "type": NODE_CONDITION,
            "config": {
                "field": "{{nodes.lookup.output}}",
                "op": "contains",
                "value": "已发货",
                "branches": {"true": ["ship"], "false": ["other"]},
            },
        },
        {
            "id": "ship",
            "type": NODE_LLM,
            "config": {"prompt": "已发货：{{input}}"},
            "next": ["end_ship"],
        },
        {"id": "end_ship", "type": NODE_END},
        {
            "id": "other",
            "type": NODE_LLM,
            "config": {"prompt": "未发货：{{input}}"},
            "next": ["end_other"],
        },
        {"id": "end_other", "type": NODE_END},
    ],
}

MEMORY_RAW = {
    "workflow_id": "mem",
    "entry": "start",
    "nodes": [
        {"id": "start", "type": NODE_START, "next": ["recall"]},
        {
            "id": "recall",
            "type": NODE_MEMORY,
            "config": {"query": "{{input}}", "top_k": 3},
            "next": ["answer"],
        },
        {
            "id": "answer",
            "type": NODE_LLM,
            "config": {"prompt": "经验：{{nodes.recall.output}}；问题：{{input}}"},
            "next": ["end"],
        },
        {"id": "end", "type": NODE_END},
    ],
}


def branch_agent(output: str) -> tuple[WorkflowAgent, FakeLLM]:
    llm = FakeLLM("已发货，429 元")
    agent = WorkflowAgent(
        WorkflowDefinition.from_dict(BRANCH_RAW),
        tools=[FakeTool(output=output)],
        llm=llm,
    )
    return agent, llm


def test_condition_takes_true_branch_and_leaves_no_trace_of_the_other():
    agent, llm = branch_agent("status=200 body=已发货")
    trace = agent.run(Task(id="B1", input="订单 1002 什么状态"))

    assert trace.status == STATUS_OK
    nodes = {e.node_id for e in trace.events}
    assert {"check", "ship", "end_ship"} <= nodes
    # 未选中的分支连 node_start 都不该有
    assert "other" not in nodes and "end_other" not in nodes
    assert len(llm.prompts) == 1
    assert llm.prompts[0].startswith("已发货")
    assert trace.turns[0].answer == "已发货，429 元"


def test_condition_takes_false_branch():
    agent, llm = branch_agent("status=200 body=待付款")
    trace = agent.run(Task(id="B2", input="订单 1002 什么状态"))

    assert trace.status == STATUS_OK
    nodes = {e.node_id for e in trace.events}
    assert {"check", "other", "end_other"} <= nodes
    assert "ship" not in nodes and "end_ship" not in nodes
    assert llm.prompts[0].startswith("未发货")


def test_condition_emits_one_branch_event_with_result():
    agent, _ = branch_agent("status=200 body=已发货")
    trace = agent.run(Task(id="B3", input="订单 1002 什么状态"))

    events = [e for e in trace.events if e.kind == KIND_BRANCH]
    assert len(events) == 1
    ev = events[0]
    assert ev.node_id == "check"
    assert ev.args["result"] == "true"
    assert ev.output == "true"


@pytest.mark.parametrize(
    "left, op, right, expected",
    [
        ("已发货", "contains", "发货", True),
        ("已发货", "not_contains", "退款", True),
        ("1002", "eq", 1002, True),  # 数值优先，不做字符串比较
        ("10", "gt", "9", True),  # 数值序；字符串序会给出 False
        ("abc", "gt", "9", False),  # 非数字不猜大小，直接判否
        ("", "empty", None, True),
        (" ", "not_empty", None, False),
        ("order-1002", "regex", r"\d{4}", True),
    ],
)
def test_compare_ops(left, op, right, expected):
    assert compare(left, op, right) is expected


def test_condition_next_must_match_branches():
    """next 与 branches 不一致时必须在加载期报错，不能挑一个用。"""
    raw = {
        "workflow_id": "t",
        "entry": "start",
        "nodes": [
            {"id": "start", "type": NODE_START, "next": ["c"]},
            {
                "id": "c",
                "type": NODE_CONDITION,
                "config": {
                    "field": "{{input}}",
                    "op": "contains",
                    "value": "x",
                    "branches": {"true": ["a"], "false": ["b"]},
                },
                "next": ["a"],
            },
            {"id": "a", "type": NODE_END},
            {"id": "b", "type": NODE_END},
        ],
    }
    with pytest.raises(DefinitionError, match="不一致"):
        WorkflowDefinition.from_dict(raw)


def test_schema_reports_bad_condition_and_memory_config():
    raw = {
        "workflow_id": "t",
        "entry": "start",
        "nodes": [
            {"id": "start", "type": NODE_START, "next": ["c"]},
            {"id": "c", "type": NODE_CONDITION, "config": {"op": "maybe"}, "next": ["m"]},
            {"id": "m", "type": NODE_MEMORY, "config": {"top_k": 0}, "next": ["end"]},
            {"id": "end", "type": NODE_END},
        ],
    }
    errors = validate(WorkflowDefinition.from_dict(raw))
    assert any("config.field" in e for e in errors)
    assert any("config.op" in e for e in errors)
    assert any("config.branches" in e for e in errors)
    assert any("config.query" in e for e in errors)
    assert any("config.top_k" in e for e in errors)


# ---------------------------------------------------------------- 记忆


def make_exp(exp_id: str, action: str, query: str) -> Experience:
    """造一条与 query 关键词完全重合的经验，避开检索门槛的不确定性。"""
    return Experience(
        exp_id=exp_id,
        failure_type="missing_tool",
        pattern="问快递却只查了订单",
        action=action,
        trigger=Trigger(tags=task_tags(Task(id="t", input=query))),
    )


QUERY = "订单 1002 用的哪家快递"


def memory_events(trace) -> list:
    return [e for e in trace.events if e.kind == KIND_MEMORY_CALL]


def test_memory_node_with_empty_store_keeps_prompt_clean(tmp_path: Path):
    llm = FakeLLM("查不到")
    agent = WorkflowAgent(
        WorkflowDefinition.from_dict(MEMORY_RAW),
        tools=[],
        llm=llm,
        memory_store=ExperienceStore(tmp_path / "empty.jsonl"),
    )
    trace = agent.run(Task(id="M1", input=QUERY))

    assert trace.status == STATUS_OK
    events = memory_events(trace)
    assert len(events) == 1
    assert events[0].args["hit_count"] == 0
    assert events[0].args["hits"] == []
    assert events[0].output == ""
    # 空库不能往 prompt 里塞一段"历史经验"表头
    assert "【历史经验" not in llm.prompts[0]


def test_memory_node_injects_hits_into_llm_prompt(tmp_path: Path):
    """检索到不等于用上了：必须验证经验文本真的进了 LLM 的 prompt。"""
    store = ExperienceStore(tmp_path / "mem.jsonl")
    store.add(make_exp("exp-1", "先调 get_logistics 查快递再回答", QUERY))

    llm = FakeLLM("中通快递")
    agent = WorkflowAgent(
        WorkflowDefinition.from_dict(MEMORY_RAW),
        tools=[],
        llm=llm,
        memory_store=store,
    )
    trace = agent.run(Task(id="M2", input=QUERY))

    event = memory_events(trace)[0]
    assert event.args["hit_count"] == 1
    assert event.args["hits"] == ["exp-1"]
    assert event.args["injected_chars"] > 0
    assert event.elapsed_ms >= 0
    assert "get_logistics" in llm.prompts[0]


def test_memory_and_branch_events_are_not_tool_calls(tmp_path: Path):
    """memory_call / branch 不是 ToolCall：判分器不能把它们算成工具调用。"""
    from evaluator import evaluate

    store = ExperienceStore(tmp_path / "mem.jsonl")
    store.add(make_exp("exp-1", "先查快递再回答", QUERY))
    agent = WorkflowAgent(
        WorkflowDefinition.from_dict(MEMORY_RAW),
        tools=[],
        llm=FakeLLM(),
        memory_store=store,
    )
    trace = agent.run(Task(id="M3", input=QUERY))

    assert trace.all_calls() == []
    verdict = evaluate(
        trace,
        Task(id="M3", input=QUERY, assertions={"expect_tools": []}),
    )
    assert verdict.rounds[0].call_count == 0
    assert verdict.passed is True


def test_two_workflows_do_not_share_experience(tmp_path: Path):
    """两个 Workflow 各用各的库：检索只读，不写回，也不会互相看见。"""
    store_a = ExperienceStore(tmp_path / "a.jsonl")
    store_a.add(make_exp("exp-a", "A 的做法", QUERY))
    store_b = ExperienceStore(tmp_path / "b.jsonl")
    store_b.add(make_exp("exp-b", "B 的做法", QUERY))

    defn = WorkflowDefinition.from_dict(MEMORY_RAW)
    trace_a = WorkflowAgent(defn, tools=[], llm=FakeLLM(), memory_store=store_a).run(
        Task(id="A", input=QUERY)
    )
    trace_b = WorkflowAgent(defn, tools=[], llm=FakeLLM(), memory_store=store_b).run(
        Task(id="B", input=QUERY)
    )

    assert memory_events(trace_a)[0].args["hits"] == ["exp-a"]
    assert memory_events(trace_b)[0].args["hits"] == ["exp-b"]
    # 检索是只读的：跑完两个库各自还是 1 条
    assert len(ExperienceStore(tmp_path / "a.jsonl")) == 1
    assert len(ExperienceStore(tmp_path / "b.jsonl")) == 1


def test_branch_demo_definition_runs_end_to_end():
    """示例定义要能真跑：带 memory 节点、未注入经验库时也不能崩。"""
    defn = WorkflowDefinition.load(ROOT / "data" / "workflows" / "branch-demo.json")
    llm = FakeLLM("已发货，429 元")
    agent = WorkflowAgent(
        defn,
        tools=[FakeTool(name="get_order", output="status=200 body=已发货")],
        llm=llm,
    )
    trace = agent.run(Task(id="BD1", input="订单 1002 什么状态"))

    assert trace.status == STATUS_OK
    nodes = {e.node_id for e in trace.events}
    assert {"recall", "check_status", "llm_ship"} <= nodes
    assert "llm_other" not in nodes
    assert trace.turns[0].answer == "已发货，429 元"


# ---------------------------------------------------------------- 参数抽取


def extract_definition(args: dict) -> WorkflowDefinition:
    """最小可跑的取参流程：start → tool(args) → end。"""
    return WorkflowDefinition.from_dict(
        {
            "workflow_id": "t-extract",
            "entry": "start",
            "nodes": [
                {"id": "start", "type": NODE_START, "next": ["lookup"]},
                {
                    "id": "lookup",
                    "type": NODE_TOOL,
                    "config": {"tool": "get_order", "args": args},
                    "next": ["end"],
                },
                {"id": "end", "type": NODE_END, "next": []},
            ],
        }
    )


def test_extract_field_rules_prefer_context_over_bare_digits():
    """演示 why 顺序重要：先带"订单"上下文的，再退到裸四位数。"""
    assert extract_field("order_id", "订单 1001 的金额，预算 1002 元") == "1001"
    assert extract_field("order_id", "1002") == "1002"
    assert extract_field("amount", "预算 899 元以内") == "899"
    assert extract_field("product_id", "商品 P1001 怎么样") == "P1001"
    # 名字写错与抽不到是两类问题，必须给不同的话术
    with pytest.raises(ExtractError, match="未知抽取字段"):
        extract_field("ghost", "1001")
    with pytest.raises(ExtractError, match="抽不到"):
        extract_field("order_id", "帮我查下订单的金额")


def test_tool_args_render_extract_from_input():
    """模板参数按输入取，而不是设计期写死——这是 P6 的全部意义。"""
    tool = FakeTool()
    agent = WorkflowAgent(
        extract_definition({"order_id": "{{extract.order_id}}"}),
        tools=[tool],
        llm=FakeLLM(),
    )
    trace = agent.run(Task(id="W03", input="帮我查下订单 1001 的金额"))

    assert trace.status == STATUS_OK
    assert tool.received == [{"order_id": "1001"}]


def test_extract_failure_marks_infra_instead_of_calling_with_empty_arg():
    """抽不到就别调工具：带空参数打接口会拿到 404，最后被记成"模型答错"。"""
    tool = FakeTool()
    agent = WorkflowAgent(
        extract_definition({"order_id": "{{extract.order_id}}"}),
        tools=[tool],
        llm=FakeLLM(),
    )
    trace = agent.run(Task(id="W99", input="帮我查下订单的金额"))

    assert trace.status == STATUS_INFRA
    assert "抽不到" in trace.error
    assert tool.received == []
    assert len(trace.turns) == 1


def test_schema_rejects_unknown_extract_field():
    """名字写错要在加载期报出来，不能跑到一半才抛。"""
    errors = validate(extract_definition({"order_id": "{{extract.order}}"}))
    assert any("未知抽取字段" in e and "'order'" in e for e in errors)


# ---------------------------------------------------------------- 双工具对比


def test_shop_demo_runs_both_tools_with_one_extracted_id():
    """抽一次、两处用：两个节点拿到同一个订单号，不是各抽各的。"""
    defn = WorkflowDefinition.load(ROOT / "data" / "workflows" / "shop-demo.json")
    order = FakeTool(name="get_order", output="status=200 body=已签收 金额 899.0")
    logistics = FakeTool(name="get_logistics", output="status=200 body=carrier=中通")
    agent = WorkflowAgent(defn, tools=[order, logistics], llm=FakeLLM("已签收，899 元"))

    trace = agent.run(Task(id="S1", input="订单 1001 现在什么状态，顺便看下物流"))

    assert trace.status == STATUS_OK
    assert order.received == [{"order_id": "1001"}]
    assert logistics.received == [{"order_id": "1001"}]
    nodes = [e.node_id for e in trace.events if e.kind == KIND_NODE_END]
    assert nodes == ["start", "get_order", "get_logistics", "llm1", "end"]


def test_wf_eval_tasks_are_well_formed():
    """评测集是数据文件，节点名写错只会在 20 秒的真跑里暴露。这里先钉结构。"""
    text = (ROOT / "data" / "wf_eval.jsonl").read_text(encoding="utf-8")
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    assert [r["id"] for r in rows] == ["W01", "W02", "W03", "W04", "W05", "W06"]

    for row in rows:
        assert row["input"].strip()
        assert row["split"] in ("fit", "holdout")
        assert isinstance(row.get("expect_tools"), list)

    by_id = {r["id"]: r for r in rows}
    # W05 刻意只做负向断言：不许出现别的订单的真实状态与金额
    assert by_id["W05"]["must_not_have"]
    assert not by_id["W05"]["must_have"]


# ---------------------------------------------------------------- 经验注入


MEMO = (
    "【历史经验（来自过去失败任务的教训，只说明该怎么做，不是当前事实）】\n"
    "1. 只用本轮工具返回的内容作答，别的订单的金额不能带进来。"
)


def report_agent() -> tuple[WorkflowAgent, FakeLLM]:
    defn = WorkflowDefinition.load(ROOT / "data" / "workflows" / "report-demo.json")
    llm = FakeLLM("待支付，299 元")
    agent = WorkflowAgent(
        defn,
        tools=[FakeTool(name="get_order"), FakeTool(name="get_logistics")],
        llm=llm,
    )
    return agent, llm


def test_workflow_injects_runtime_experience_into_prompt():
    """Runtime 递进来的经验必须真的进 prompt——只记字数等于没注入。"""
    agent, llm = report_agent()
    task = Task(id="E1", input="订单 1003 的金额是多少")

    trace = agent.run(task, RunContext(task=task, experience=MEMO, hits=["exp-x"]))

    assert trace.status == STATUS_OK
    assert llm.prompts[0].startswith(MEMO), "经验要拼在最前面，截断记录时才看得见"
    assert trace.meta["experience_chars"] == len(MEMO)
    event = [e for e in trace.events if e.kind == KIND_LLM_CALL][0]
    assert event.args["experience_chars"] == len(MEMO)


def test_workflow_without_experience_leaves_prompt_clean():
    """基线组不能出现空的经验表头：那是往提示词里塞一句空承诺。"""
    agent, llm = report_agent()
    task = Task(id="E2", input="订单 1003 的金额是多少")

    trace = agent.run(task)

    assert trace.status == STATUS_OK
    assert "【历史经验" not in llm.prompts[0]
    assert "experience_chars" not in trace.meta
    event = [e for e in trace.events if e.kind == KIND_LLM_CALL][0]
    assert "experience_chars" not in event.args


def test_schema_rejects_non_bool_inject_experience():
    defn = WorkflowDefinition.from_dict(
        {
            "workflow_id": "t-inject",
            "entry": "start",
            "nodes": [
                {"id": "start", "type": NODE_START, "next": ["llm1"]},
                {
                    "id": "llm1",
                    "type": NODE_LLM,
                    "config": {"prompt": "回答：{{input}}", "inject_experience": "yes"},
                    "next": ["end"],
                },
                {"id": "end", "type": NODE_END, "next": []},
            ],
        }
    )
    errors = validate(defn)
    assert any("inject_experience" in e for e in errors)
