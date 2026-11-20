"""Workflow 引擎的离线测试。

三件事必须钉住，因为它们是这个阶段唯一的产出：
1. 坏定义在执行前就被拒绝，且错误能定位到具体节点；
2. 合法定义能真跑，事件来自真实执行而不是拼出来的；
3. Turn 语义不被破坏——单轮 workflow 仍然是 1 个 Turn，工具调用只在事件里。

全程不联网：模型与工具都是注入的假对象。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import agents.builtin as builtin
from agentplatform import (
    NODE_END,
    NODE_LLM,
    NODE_START,
    NODE_TOOL,
    DefinitionError,
    WorkflowAgent,
    WorkflowDefinition,
    topo_sort,
    validate,
)
from agentplatform.executor import TemplateError, render
from agentplatform.topology import check as check_topology
from core.protocol import validate_agent
from core.task import Task
from core.trace import (
    KIND_LLM_CALL,
    KIND_NODE_END,
    KIND_NODE_START,
    KIND_TOOL_CALL,
    STATUS_INFRA,
    STATUS_OK,
)

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
