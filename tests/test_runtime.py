"""驱动层与接入协议测试。

要证明的四件事：
1. 一个类 + 一个 run 就能接进 Runtime，换 Agent 不改核心代码。
2. 单条崩溃不拖垮整批，错误被如实记进轨迹。
3. 任务自带的划分与 id 不会被 Agent 篡改。
4. 经验能被送到 Agent 手上（送达即行为变化）。

全部本地执行：不调模型、不连服务、不读配置。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.registry import Registry
from core.runner import run_suite, run_task, run_tasks
from core.task import SPLIT_FIT, SPLIT_HOLDOUT, Task, load_many
from core.trace import STATUS_CRASHED, STATUS_OK, Trace, Turn, load_traces
from demos.echo import BrokenAgent, EchoAgent, SilentAgent, WrongTypeAgent

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TASK_FILES = [
    DATA_DIR / "shop_eval.jsonl",
    DATA_DIR / "shop_eval_multi.jsonl",
    DATA_DIR / "shop_eval_hard.jsonl",
]


def make_task(task_id: str = "T1", turns: list[str] | None = None) -> Task:
    texts = turns or ["你好"]
    return Task(id=task_id, input=texts[0], turns=texts)


# ---------------------------------------------------------------- 基本接入


def test_echo_agent_is_runnable_through_runner():
    agent = EchoAgent()
    trace = run_task(agent, make_task())
    assert isinstance(trace, Trace)
    assert trace.status == STATUS_OK
    assert trace.agent == "echo"
    assert [t.answer for t in trace.turns] == ["[echo] 你好"]
    assert trace.all_calls()[0].name == "echo"


def test_second_agent_only_needs_one_class():
    """换一个全新 Agent 只写一个类，注册后走同一条执行路径。"""

    class SecondAgent:
        name = "second"

        def run(self, task, ctx=None):
            texts = task.user_turns()
            turns = [
                Turn(index=i, user_input=t, calls=[], answer=f"second:{t}")
                for i, t in enumerate(texts)
            ]
            return Trace(task_id=task.id, agent=self.name, turns=turns)

    reg = Registry("t")
    reg.register(EchoAgent())
    reg.register(SecondAgent())
    assert reg.names() == ["echo", "second"]

    task = make_task("T2", ["a", "b"])
    first = run_task(reg.get("echo"), task)
    second = run_task(reg.get("second"), task)

    # 结构完全一致：同一个 runner，不因 Agent 不同而有分支
    assert len(first.turns) == len(second.turns) == 2
    assert first.task_id == second.task_id == "T2"
    assert first.split == second.split
    # 内容各自不同，说明确实是两个 Agent
    assert first.agent == "echo" and second.agent == "second"
    assert second.final_answer() == "second:b"


def test_runner_does_not_know_any_agent_name():
    """核心驱动层里不许出现任何具体 Agent 类名或 demos 引用。"""
    src = (ROOT / "core" / "runner.py").read_text(encoding="utf-8")
    for banned in ("Echo", "Silent", "Broken", "WrongType", "demos"):
        assert banned not in src, f"core/runner.py 不该出现 {banned!r}"


# ---------------------------------------------------------------- 轨迹与划分


def test_split_comes_from_task_not_agent():
    class LyingAgent:
        name = "liar"

        def run(self, task, ctx=None):
            # 故意声称自己在训练侧
            return Trace(task_id=task.id, agent=self.name, split=SPLIT_FIT,
                         turns=[Turn(index=0, answer="x")])

    task = Task(id="T9", input="hi", split=SPLIT_HOLDOUT)
    trace = run_task(LyingAgent(), task)
    assert trace.split == SPLIT_HOLDOUT, "划分必须以任务为准"


def test_task_id_mismatch_is_corrected_and_warned():
    class ConfusedAgent:
        name = "confused"

        def run(self, task, ctx=None):
            return Trace(task_id="WRONG", agent=self.name, turns=[Turn(index=0, answer="x")])

    trace = run_task(ConfusedAgent(), make_task("T3"))
    assert trace.task_id == "T3"
    assert any("task_id" in w for w in trace.meta.get("warnings", []))


def test_multi_turn_calls_are_aligned():
    task = make_task("T4", ["一", "二", "三"])
    trace = run_task(EchoAgent(), task)
    assert [t.index for t in trace.turns] == [0, 1, 2]
    assert [c.call_id for c in trace.all_calls()] == ["t0c0", "t1c0", "t2c0"]
    assert [o for _, _, o in trace.outputs_by_call()] == ["echo:一", "echo:二", "echo:三"]


# ---------------------------------------------------------------- 崩溃隔离


def test_broken_agent_produces_crashed_trace():
    trace = run_task(BrokenAgent("接口挂了"), make_task("T5"))
    assert trace.status == STATUS_CRASHED
    assert "RuntimeError" in trace.error
    assert "接口挂了" in trace.error
    assert trace.evaluable is False


def test_crash_does_not_stop_the_batch():
    class FlakyAgent:
        name = "flaky"

        def __init__(self):
            self.seen: list[str] = []

        def run(self, task, ctx=None):
            self.seen.append(task.id)
            if task.id == "T2":
                raise ValueError("第二条炸了")
            return Trace(task_id=task.id, agent=self.name, turns=[Turn(index=0, answer="ok")])

    agent = FlakyAgent()
    tasks = [make_task(i) for i in ("T1", "T2", "T3")]
    result = run_tasks(agent, tasks)

    assert agent.seen == ["T1", "T2", "T3"], "崩溃后必须继续跑"
    assert len(result.traces) == 3
    assert result.by_status() == {STATUS_OK: 2, STATUS_CRASHED: 1}
    assert [t.task_id for t in result.crashed()] == ["T2"]


def test_wrong_return_type_is_crashed_not_swallowed():
    trace = run_task(WrongTypeAgent(), make_task("T6"))
    assert trace.status == STATUS_CRASHED
    assert "Trace" in trace.error


def test_silent_agent_is_recorded_as_is():
    """什么都不说也如实记录——runner 不做内容判断，判断留给 evaluator。"""
    trace = run_task(SilentAgent(), make_task("T7"))
    assert trace.status == STATUS_OK
    assert trace.final_answer() == ""
    assert trace.all_calls() == []


# ---------------------------------------------------------------- 经验送达


def test_experience_reaches_agent_and_changes_behavior():
    task = make_task("T8")
    plain = run_task(EchoAgent(), task)
    with_exp = run_task(EchoAgent(), task, experience="- 问状态时两个工具都查", hits=["exp-1"])

    assert plain.final_answer().startswith("[echo] ")
    assert with_exp.final_answer().startswith("[echo+exp] ")
    assert with_exp.meta["experience_applied"] is True
    assert with_exp.meta["hits"] == ["exp-1"]
    assert plain.meta["experience_applied"] is False


def test_provider_called_once_per_task():
    calls: list[str] = []

    def provider(task):
        calls.append(task.id)
        return (f"经验给 {task.id}", [f"e-{task.id}"])

    tasks = [make_task(i) for i in ("T1", "T2")]
    result = run_tasks(EchoAgent(), tasks, experience_provider=provider)

    assert calls == ["T1", "T2"]
    assert [t.meta["hits"] for t in result.traces] == [["e-T1"], ["e-T2"]]


def test_provider_failure_is_recorded_and_run_continues():
    def broken_provider(task):
        raise RuntimeError("经验库读不了")

    tasks = [make_task("T1"), make_task("T2")]
    result = run_tasks(EchoAgent(), tasks, experience_provider=broken_provider)

    assert len(result.traces) == 2
    assert all(t.status == STATUS_OK for t in result.traces), "经验失败不该影响任务执行"
    for t in result.traces:
        assert any("经验提供者失败" in w for w in t.meta.get("warnings", []))


# ---------------------------------------------------------------- 批量与落盘


def test_run_suite_filters_by_split():
    tasks = [
        Task(id="F1", input="a", split=SPLIT_FIT),
        Task(id="H1", input="b", split=SPLIT_HOLDOUT),
        Task(id="H2", input="c", split=SPLIT_HOLDOUT),
    ]
    result = run_suite(EchoAgent(), tasks, split=SPLIT_HOLDOUT)
    assert [t.task_id for t in result.traces] == ["H1", "H2"]


def test_run_suite_writes_jsonl(tmp_path: Path):
    out = tmp_path / "out" / "traces.jsonl"
    tasks = [make_task("T1"), make_task("T2")]
    result = run_suite(EchoAgent(), tasks, out_path=out)

    assert out.exists()
    back = load_traces(out)
    assert [t.task_id for t in back] == ["T1", "T2"]
    assert [t.to_dict() for t in back] == [t.to_dict() for t in result.traces]


def test_empty_batch_is_not_an_error():
    result = run_tasks(EchoAgent(), [])
    assert result.traces == []
    assert result.summary()["tasks"] == 0


def test_progress_hook_sees_every_trace():
    seen: list[str] = []
    tasks = [make_task(i) for i in ("T1", "T2", "T3")]
    run_tasks(EchoAgent(), tasks, on_trace=lambda t: seen.append(t.task_id))
    assert seen == ["T1", "T2", "T3"]


def test_summary_counts_are_consistent():
    tasks = [make_task("T1", ["a", "b"]), make_task("T2")]
    result = run_tasks(EchoAgent(), tasks)
    s = result.summary()
    assert s["tasks"] == 2
    assert s["turns"] == 3
    assert s["calls"] == 3, "echo 每轮记一次调用"
    assert s["evaluable"] == 2
    assert s["elapsed_ms"] >= 0


@pytest.mark.skipif(
    not all(p.exists() for p in TASK_FILES), reason="真实任务集不在仓库里"
)
def test_all_real_tasks_run_through_runtime():
    """43 条真实任务在 Runtime 上全跑一遍，不调模型。"""
    tasks = load_many(TASK_FILES)
    result = run_tasks(EchoAgent(), tasks)

    assert len(result.traces) == 43
    assert all(t.status == STATUS_OK for t in result.traces)
    assert all(t.evaluable for t in result.traces)

    s = result.summary()
    assert s["calls"] == s["turns"], "echo 每轮恰好一次调用，两者必须相等"

    # 划分跟任务走，43 条里两侧都有
    fit = [t for t in result.traces if t.split == SPLIT_FIT]
    holdout = [t for t in result.traces if t.split == SPLIT_HOLDOUT]
    assert len(fit) == 26 and len(holdout) == 17
