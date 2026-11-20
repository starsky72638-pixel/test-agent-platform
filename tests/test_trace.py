"""core 层的数据契约测试。

三个目标：
1. 轨迹能完整、保序、可往返地记录（这是后面一切的前提）。
2. 任务集能被读成 Task，训练/验证划分互斥且不漏。
3. core 真的与业务、与框架解耦——用静态检查断言，不靠口头保证。

本文件不调用任何网络、不连服务、不读模型配置，纯本地可跑。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from core.protocol import AgentContractError, RunContext, validate_agent
from core.registry import DuplicateAgentError, Registry, UnknownAgentError
from core.task import (
    FIT_IDS,
    SPLIT_FIT,
    SPLIT_HOLDOUT,
    Task,
    assign_split,
    load_many,
    load_tasks,
    split_counts,
)
from core.trace import (
    KIND_NODE_END,
    KIND_NODE_START,
    KIND_TOOL_CALL,
    SCHEMA_VERSION,
    STATUS_INFRA,
    STATUS_OK,
    Event,
    ToolCall,
    Trace,
    Turn,
    load_traces,
    save_traces,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TASK_FILES = [
    DATA_DIR / "shop_eval.jsonl",
    DATA_DIR / "shop_eval_multi.jsonl",
    DATA_DIR / "shop_eval_hard.jsonl",
]


# ---------------------------------------------------------------- 轨迹契约


def test_tool_call_roundtrip():
    call = ToolCall(name="search", args={"q": "机械键盘", "max_price": 900}, output="200 ok")
    again = ToolCall.from_dict(call.to_dict())
    assert again == call
    assert again.args == {"q": "机械键盘", "max_price": 900}


def test_event_roundtrip():
    event = Event(
        kind=KIND_TOOL_CALL,
        seq=3,
        node_id="t1",
        node_type="tool",
        turn=1,
        name="get_order",
        args={"order_id": 1002},
        output="status=200",
        elapsed_ms=12,
    )
    assert Event.from_dict(event.to_dict()) == event


def test_add_event_assigns_monotonic_seq():
    """编号由 Trace 统一分配：不同节点各自记 seq 一定会撞号。"""
    trace = Trace(task_id="T1", agent="wf")
    a = trace.add_event(KIND_NODE_START, node_id="start", node_type="start")
    b = trace.add_event(KIND_TOOL_CALL, node_id="t1", name="get_order")
    assert (a.seq, b.seq) == (0, 1)
    assert [e.seq for e in trace.events] == [0, 1]
    # 事件不改 turns：两条线互不干扰，这是 Turn/Event 分开的意义
    assert trace.turns == []


def test_legacy_trace_without_events_field():
    """旧 jsonl 没有 events 键，读出来必须是空列表，且往返后完全一致。"""
    raw = {"task_id": "T1", "agent": "shop", "turns": [], "status": "ok"}
    trace = Trace.from_dict(raw)
    assert trace.events == []
    assert Trace.from_dict(trace.to_dict()).to_dict() == trace.to_dict()


def test_all_calls_sees_events_when_turn_has_none():
    """Memory 溯源、质量闸门、蒸馏、CLI 打印都走 all_calls。

    workflow 轨迹的 Turn.calls 是空的，如果这里也返回空，
    那些消费方会静默看到"0 次调用"——比报错更难查。
    """
    trace = Trace(task_id="T1", agent="wf", turns=[Turn(index=0, user_input="q", answer="a")])
    trace.add_event(KIND_NODE_START, node_id="start", node_type="start")
    trace.add_event(
        KIND_TOOL_CALL, node_id="t1", name="get_order", args={"order_id": 1002}, output="ok"
    )
    trace.add_event(KIND_NODE_END, node_id="end", node_type="end")

    assert [c.name for c in trace.all_calls()] == ["get_order"]
    assert trace.outputs_by_call() == [("get_order", {"order_id": 1002}, "ok")]


def test_turn_preserves_call_order():
    turn = Turn(
        index=0,
        user_input="1001 到哪了",
        calls=[
            ToolCall(name="a", call_id="t0c0"),
            ToolCall(name="b", call_id="t0c1"),
            ToolCall(name="c", call_id="t0c2"),
        ],
        answer="已签收",
    )
    again = Turn.from_dict(turn.to_dict())
    assert again.call_names() == ["a", "b", "c"]
    assert [c.call_id for c in again.calls] == ["t0c0", "t0c1", "t0c2"]


def test_same_tool_twice_keeps_separate_outputs():
    """同一工具调两次，各自的返回不能互相覆盖。旧实现按工具名存 map，这里必须不同。"""
    turn = Turn(
        index=0,
        calls=[
            ToolCall(name="q", args={"n": 1}, call_id="t0c0", output="第一次返回"),
            ToolCall(name="q", args={"n": 2}, call_id="t0c1", output="第二次返回"),
        ],
    )
    trace = Trace(task_id="T1", agent="fake", turns=[turn])

    aligned = trace.outputs_by_call()
    assert [o for _, _, o in aligned] == ["第一次返回", "第二次返回"]
    assert [a for _, a, _ in aligned] == [{"n": 1}, {"n": 2}]

    # 按名字取只保留最后一次，这是有意为之的语义，文档里写明了
    assert trace.outputs_by_name() == {"q": "第二次返回"}


def test_trace_roundtrip_keeps_everything():
    trace = Trace(
        task_id="S01",
        agent="demo",
        split=SPLIT_FIT,
        turns=[
            Turn(index=0, user_input="你好", answer="你好，需要什么"),
            Turn(
                index=1,
                user_input="键盘多少钱",
                calls=[ToolCall(name="search", args={"q": "键盘"}, output="429")],
                answer="429 元",
                elapsed_ms=1200,
            ),
        ],
        elapsed_ms=3000,
        meta={"tag": "多轮"},
    )
    again = Trace.from_dict(trace.to_dict())
    assert again.to_dict() == trace.to_dict()
    assert again.trace_id == trace.trace_id
    assert again.final_answer() == "429 元"
    assert again.call_names() == ["search"]
    assert again.evaluable is True


def test_trace_id_and_timestamp_auto_filled():
    trace = Trace(task_id="S02", agent="x")
    assert trace.trace_id.startswith("S02-")
    assert trace.started_at > 0
    assert trace.schema_version == SCHEMA_VERSION
    assert trace.turns == []
    assert trace.evaluable is False


def test_infra_trace_is_not_evaluable():
    trace = Trace(task_id="S03", agent="x", status=STATUS_INFRA, error="403")
    assert trace.evaluable is False
    assert trace.to_dict()["status"] == STATUS_INFRA


def test_unknown_status_rejected():
    with pytest.raises(ValueError):
        Trace(task_id="S04", agent="x", status="whatever")


def test_trace_jsonl_roundtrip(tmp_path: Path):
    traces = [
        Trace(task_id="A", agent="x", turns=[Turn(index=0, answer="一")]),
        Trace(task_id="B", agent="x", turns=[Turn(index=0, answer="二")]),
    ]
    path = tmp_path / "nested" / "traces.jsonl"
    save_traces(traces, path)
    back = load_traces(path)
    assert [t.task_id for t in back] == ["A", "B"]
    assert [t.to_dict() for t in back] == [t.to_dict() for t in traces]


def test_boundary_chars_survive_serialization():
    """换行、引号、中文、U+2028 都必须能安全往返。"""
    tricky = '第一行\n第二行 "引号" 和 \u2028 行分隔符'
    trace = Trace(task_id="C", agent="x", turns=[Turn(index=0, answer=tricky)])
    again = Trace.from_dict(json.loads(json.dumps(trace.to_dict(), ensure_ascii=False)))
    assert again.final_answer() == tricky


# ---------------------------------------------------------------- 任务契约


def test_task_keeps_assertions_verbatim():
    raw = {
        "id": "X01",
        "input": "键盘多少钱",
        "expect_tools": ["search"],
        "must_have": ["429"],
        "note": "说明不该进断言",
    }
    task = Task.from_dict(raw)
    assert task.id == "X01"
    assert task.assertions["expect_tools"] == ["search"]
    assert task.assertions["must_have"] == ["429"]
    assert "note" not in task.assertions
    assert task.meta["note"] == "说明不该进断言"


def test_task_turn_assertions_aligned():
    raw = {
        "id": "X02",
        "turns": ["一", "二", "三"],
        "turn_checks": [{"must_have": ["a"]}, {"must_have": ["b"]}],
    }
    task = Task.from_dict(raw)
    checks = task.turn_assertions()
    assert len(checks) == 3
    assert checks[0] == {"must_have": ["a"]}
    assert checks[2] == {}


def test_task_single_turn_uses_input():
    task = Task(id="X03", input="你好")
    assert task.user_turns() == ["你好"]
    assert task.turn_count() == 1
    assert task.is_multi_turn is False


def test_duplicate_task_id_rejected(tmp_path: Path):
    p = tmp_path / "dup.jsonl"
    p.write_text(
        '{"id":"D1","input":"a"}\n{"id":"D1","input":"b"}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError):
        load_tasks(p)


def test_assign_split_defaults_to_holdout():
    assert assign_split("S01") == SPLIT_FIT
    assert assign_split("UNSEEN-999") == SPLIT_HOLDOUT


def test_task_split_from_data_field():
    """数据文件自己声明划分，不必改 core 的 FIT_IDS 表（新场景接入用）。"""
    task = Task.from_dict({"id": "NEW-1", "input": "a", "split": "fit"})
    assert task.split == SPLIT_FIT


def test_task_split_defaults_when_absent():
    """老数据集没有该字段：未登记的 id 仍进 holdout，不许悄悄变 fit。"""
    task = Task.from_dict({"id": "NEW-1", "input": "a"})
    assert task.split == SPLIT_HOLDOUT


def test_task_split_explicit_arg_wins():
    """显式参数优先于文件字段，旧调用方式不被改写。"""
    task = Task.from_dict({"id": "NEW-1", "input": "a", "split": "fit"}, split="holdout")
    assert task.split == SPLIT_HOLDOUT


def test_task_rejects_unknown_split():
    """写错的值必须当场报错，而不是被静默降级成 holdout。"""
    with pytest.raises(ValueError):
        Task.from_dict({"id": "NEW-1", "input": "a", "split": "train"})


@pytest.mark.skipif(
    not all(p.exists() for p in TASK_FILES), reason="真实任务集不在仓库里"
)
def test_real_task_set_is_readable_and_split_is_disjoint():
    tasks = load_many(TASK_FILES)
    assert len(tasks) == 43, f"任务集应为 43 条，实际 {len(tasks)}"

    counts = split_counts(tasks)
    assert counts[SPLIT_FIT] + counts[SPLIT_HOLDOUT] == len(tasks)
    assert counts["unassigned"] == 0

    fit = {t.id for t in tasks if t.split == SPLIT_FIT}
    holdout = {t.id for t in tasks if t.split == SPLIT_HOLDOUT}
    assert fit & holdout == set()
    assert fit == (FIT_IDS & {t.id for t in tasks})
    assert fit | holdout == {t.id for t in tasks}

    # 每一条都必须真的读到内容，空任务是数据问题
    for t in tasks:
        assert t.user_turns(), f"{t.id} 没有用户输入"


# 多轮场景集里允许存在的单轮任务：M05 是“澄清策略”用例，
# 用户说“我想买个礼物送人”，期望 agent 反问预算与对象，本身就不该有第二轮。
SINGLE_TURN_ALLOWED_IN_MULTI = {"M05"}


@pytest.mark.skipif(
    not all(p.exists() for p in TASK_FILES), reason="真实任务集不在仓库里"
)
def test_multi_turn_task_keeps_turns():
    tasks = load_many(TASK_FILES)
    multi = [t for t in tasks if t.id.startswith("M")]
    assert len(multi) == 10, f"多轮集应有 10 条，实际 {len(multi)}"

    true_multi = [t for t in multi if t.turn_count() >= 2]
    single = {t.id for t in multi if t.turn_count() == 1}

    # 单轮只允许登记过的澄清类用例；新增单轮任务必须在这里显式登记，
    # 否则多半是 turns 写漏了。
    assert single <= SINGLE_TURN_ALLOWED_IN_MULTI, (
        f"多轮集里出现未登记的单轮任务: {sorted(single - SINGLE_TURN_ALLOWED_IN_MULTI)}"
    )
    assert len(true_multi) == len(multi) - len(single)

    for t in true_multi:
        assert t.turn_assertions(), f"{t.id} 是多轮但没有逐轮断言"
        assert len(t.turn_assertions()) == t.turn_count(), (
            f"{t.id} 断言 {len(t.turn_assertions())} 条，轮数 {t.turn_count()}"
        )


# ---------------------------------------------------------------- 协议与注册


class _FakeAgent:
    name = "fake"

    def run(self, task, ctx=None):
        return Trace(
            task_id=task.id,
            agent=self.name,
            turns=[Turn(index=0, user_input=task.input, answer="ok")],
        )


def test_validate_agent_accepts_minimal_implementation():
    validate_agent(_FakeAgent())  # 不抛异常即为通过


def test_validate_agent_rejects_missing_name():
    class Nameless:
        def run(self, task, ctx=None):
            return Trace(task_id="X", agent="x")

    with pytest.raises(AgentContractError):
        validate_agent(Nameless())


def test_validate_agent_rejects_bad_signature():
    class BadSig:
        name = "bad"

        def run(self):
            return None

    with pytest.raises(AgentContractError):
        validate_agent(BadSig())


def test_registry_register_and_get():
    reg = Registry("t")
    agent = _FakeAgent()
    reg.register(agent)
    assert reg.names() == ["fake"]
    assert reg.get("fake") is agent
    assert "fake" in reg
    assert len(reg) == 1


def test_registry_decorator_form():
    reg = Registry("t")

    @reg.register
    class Another(_FakeAgent):
        name = "another"

    assert reg.get("another").name == "another"


def test_registry_duplicate_rejected_unless_replace():
    reg = Registry("t")
    reg.register(_FakeAgent())
    with pytest.raises(DuplicateAgentError):
        reg.register(_FakeAgent())
    reg.register(_FakeAgent(), replace=True)
    assert len(reg) == 1


def test_registry_unknown_lists_available():
    reg = Registry("t")
    reg.register(_FakeAgent())
    with pytest.raises(UnknownAgentError) as exc:
        reg.get("missing")
    assert "fake" in str(exc.value)


def test_registry_default_is_empty():
    """默认表必须为空：核心包不许偷偷注册任何 demo。"""
    from core.registry import DEFAULT

    assert DEFAULT.names() == []


# ---------------------------------------------------------------- 解耦断言

FORBIDDEN_IMPORTS = {
    "langchain",
    "langchain_core",
    "langchain_openai",
    "openai",
    "httpx",
    "requests",
    "aiohttp",
}
FORBIDDEN_WORDS = re.compile(
    r"\b(shop|order|product|logistics|refund|cart|customer)\b", re.IGNORECASE
)
IMPORT_LINE = re.compile(r"^\s*(?:from|import)\s+([A-Za-z_][\w.]*)", re.MULTILINE)


def _core_files() -> list[Path]:
    return sorted((ROOT / "core").glob("*.py"))


def test_core_depends_only_on_stdlib():
    allowed = {
        "__future__",
        "json",
        "time",
        "uuid",
        "dataclasses",
        "pathlib",
        "typing",
        "inspect",
        "re",
        "sys",
        "os",
        "enum",
        "math",
        "hashlib",
        "copy",
        "collections",
        "itertools",
        "functools",
        "textwrap",
        "traceback",
        "core",
    }
    offenders: list[str] = []
    for path in _core_files():
        src = path.read_text(encoding="utf-8")
        for mod in IMPORT_LINE.findall(src):
            top = mod.split(".")[0]
            if top in FORBIDDEN_IMPORTS:
                offenders.append(f"{path.name}: 禁止依赖 {mod}")
            elif top not in allowed:
                offenders.append(f"{path.name}: 未登记依赖 {mod}")
    assert not offenders, "核心层依赖不干净:\n" + "\n".join(offenders)


def test_core_has_no_business_words():
    offenders: list[str] = []
    for path in _core_files():
        src = path.read_text(encoding="utf-8")
        for m in FORBIDDEN_WORDS.finditer(src):
            line = src[: m.start()].count("\n") + 1
            offenders.append(f"{path.name}:{line} 出现业务词 {m.group(0)!r}")
    assert not offenders, "核心层混入业务词汇:\n" + "\n".join(offenders)


def test_core_never_imports_demos():
    """核心层不许 import 任何 demo，否则删掉 demos 就跑不了。

    这里做源码级断言，而不是靠 sys.modules 差异——
    模块一旦被别的测试加载过，导入副作用就测不出来了，那种写法等于假过。
    """
    offenders: list[str] = []
    for path in _core_files():
        src = path.read_text(encoding="utf-8")
        for mod in IMPORT_LINE.findall(src):
            if mod.split(".")[0] == "demos":
                offenders.append(f"{path.name}: import {mod}")
    assert not offenders, "核心层引用了 demo:\n" + "\n".join(offenders)
