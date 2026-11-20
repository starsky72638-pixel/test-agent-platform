"""闭环编排层的离线测试。全部不调 API、不需要 mock_server。

这里用一个测试替身 FollowerAgent：冷启动一律不调工具，读到经验文本里
点名了哪个工具就调哪个。它真的按工具名去匹配经验文本，不是只看"有没有
拿到经验"——后者只能证明管道连通，证明不了经验内容起作用。

答案内容直接取自断言的 must_have，这是为了让判分能过。要验的是
pipeline 的编排，不是模型能力；真模型的能力由 scripts/run_loop.py 验。

六个必测的面：
1. 学习阶段只碰 fit（这是防泄漏的第一道也是最后一道）；
2. 提炼出的经验证据能回查到真实轨迹（不是凭空生成）；
3. 空库时两组完全相同（证明"有变化"不是判分自己抖出来的）；
4. 注入后 holdout 真的翻转（闭环的核心断言）；
5. 注入计数与效果回填都被记下（否则经验永远不会退休）；
6. pipeline 不依赖任何具体 Agent（可插拔的硬证据）。
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from core.protocol import RunContext
from core.task import Task
from core.trace import STATUS_OK, ToolCall, Trace, Turn
from memory.inject import EXPERIENCE_HEADER
from memory.store import ExperienceStore
from pipeline import learn, verify

ROOT = Path(__file__).resolve().parents[1]

ALLOWED = {"search_products", "get_order", "get_logistics"}

IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+([A-Za-z_][\w.]*)", re.MULTILINE)


class FollowerAgent:
    """测试替身：没有经验就不动手，经验里点名了哪个工具才调哪个。"""

    name = "follower"

    def __init__(self) -> None:
        self.seen: list[str] = []
        self.experience_texts: list[str] = []

    def run(self, task: Task, ctx: RunContext | None = None) -> Trace:
        self.seen.append(task.id)
        exp = (ctx.experience if ctx else "") or ""
        self.experience_texts.append(exp)

        expect = list(task.assertions.get("expect_tools") or [])
        calls = [name for name in expect if exp and name in exp]

        must = list(task.assertions.get("must_have") or [])
        answer = ("查询结果：" + "，".join(must)) if calls else "暂时没查到"

        turn = Turn(
            index=0,
            user_input=task.user_turns()[0],
            calls=[
                ToolCall(name=name, args={}, call_id=f"t0c{i}", output="ok")
                for i, name in enumerate(calls)
            ],
            answer=answer,
            steps=1,
            meta={"status": "ok"},
        )
        return Trace(task_id=task.id, agent=self.name, turns=[turn], split=task.split)


class BrokenAgent:
    """每次执行都抛异常，用来验证崩溃不会变成经验。"""

    name = "broken"

    def run(self, task: Task, ctx: RunContext | None = None) -> Trace:
        raise RuntimeError("boom")


def make_tasks() -> list[Task]:
    """两组互不重合的题：库存类（F1/H1）与物流类（F2/H2）。

    刻意让两组的关键词零交集，这样"H1 只命中库存那条经验"是被测出来的，
    而不是碰巧。
    """
    return [
        Task(
            id="F1",
            input="这个蓝牙耳机还有现货吗",
            assertions={"expect_tools": ["search_products"], "must_have": ["有货"]},
            split="fit",
        ),
        Task(
            id="F2",
            input="订单的物流到哪里了",
            assertions={
                "expect_tools": ["get_order", "get_logistics"],
                "must_have": ["已签收"],
            },
            split="fit",
        ),
        Task(
            id="H1",
            input="这个蓝牙耳机有现货吗",
            assertions={"expect_tools": ["search_products"], "must_have": ["有货"]},
            split="holdout",
        ),
        Task(
            id="H2",
            input="订单物流到哪里了",
            assertions={
                "expect_tools": ["get_order", "get_logistics"],
                "must_have": ["已签收"],
            },
            split="holdout",
        ),
    ]


def make_agent() -> FollowerAgent:
    return FollowerAgent()


def make_store(tmp_path) -> ExperienceStore:
    return ExperienceStore(tmp_path / "experiences.jsonl")


# ---------------------------------------------------------------- 学习阶段


def test_learn_only_runs_fit(tmp_path):
    """学习阶段一个 holdout 任务都不许碰。"""
    tasks = make_tasks()
    agent = make_agent()
    result = learn(agent, tasks, make_store(tmp_path), allowed_tools=ALLOWED)

    assert agent.seen == ["F1", "F2"]
    assert {t.task_id for t in result.traces} == {"F1", "F2"}


def test_learn_extracts_one_experience_per_pattern(tmp_path):
    tasks = make_tasks()
    result = learn(make_agent(), tasks, make_store(tmp_path), allowed_tools=ALLOWED)

    assert len(result.candidates) == 2
    assert {e.failure_type for e in result.kept} == {"missing_tool"}
    patterns = {e.pattern for e in result.kept}
    assert "missing_tool|search_products|single" in patterns
    assert "missing_tool|get_logistics/get_order|single" in patterns


def test_learn_action_names_the_missing_tools(tmp_path):
    tasks = make_tasks()
    result = learn(make_agent(), tasks, make_store(tmp_path), allowed_tools=ALLOWED)

    by_pattern = {e.pattern: e for e in result.kept}
    stock = by_pattern["missing_tool|search_products|single"]
    assert "search_products" in stock.action
    assert not re.search(r"\d", stock.text())


def test_learn_evidence_points_to_real_trace(tmp_path):
    tasks = make_tasks()
    result = learn(make_agent(), tasks, make_store(tmp_path), allowed_tools=ALLOWED)

    real_ids = {t.trace_id for t in result.traces}
    for exp in result.kept:
        assert exp.evidence, "入库经验必须有证据"
        assert exp.evidence[0].trace_id in real_ids
        assert exp.evidence[0].task_id in {"F1", "F2"}


def test_learn_persists_to_disk(tmp_path):
    path = tmp_path / "experiences.jsonl"
    store = ExperienceStore(path)
    learn(make_agent(), make_tasks(), store, allowed_tools=ALLOWED)
    assert path.exists()

    reloaded = ExperienceStore(path)
    assert len(reloaded.active()) == 2


def test_learn_without_fit_raises(tmp_path):
    only_holdout = [t for t in make_tasks() if t.split == "holdout"]
    with pytest.raises(ValueError, match="fit"):
        learn(make_agent(), only_holdout, make_store(tmp_path), allowed_tools=ALLOWED)


def test_crashed_trace_becomes_no_experience(tmp_path):
    """崩溃属于环境问题，不许沉淀成能力经验。"""
    result = learn(BrokenAgent(), make_tasks(), make_store(tmp_path), allowed_tools=ALLOWED)

    assert result.traces and all(t.status != STATUS_OK for t in result.traces)
    assert result.candidates == []
    assert result.kept == []


def test_learn_result_lines_report_counts(tmp_path):
    result = learn(make_agent(), make_tasks(), make_store(tmp_path), allowed_tools=ALLOWED)
    text = "\n".join(result.lines())
    assert "fit 2 条" in text
    assert "提炼 2 条" in text


# ---------------------------------------------------------------- 验证阶段


def test_verify_flips_failed_to_passed(tmp_path):
    """闭环的核心断言：没有经验过不了，注入经验之后过了。"""
    tasks = make_tasks()
    store = make_store(tmp_path)
    learn(make_agent(), tasks, store, allowed_tools=ALLOWED)

    compare = verify(make_agent(), tasks, store, allowed_tools=ALLOWED)

    assert compare.baseline.passed == 0
    assert compare.injected.passed == 2
    assert sorted(compare.flips()["up"]) == ["H1", "H2"]
    assert compare.flips()["down"] == []


def test_baseline_runs_without_experience(tmp_path):
    tasks = make_tasks()
    store = make_store(tmp_path)
    learn(make_agent(), tasks, store, allowed_tools=ALLOWED)

    agent = make_agent()
    verify(agent, tasks, store, allowed_tools=ALLOWED)

    # 顺序是 [H1 基线, H2 基线, H1 注入, H2 注入]
    assert agent.seen == ["H1", "H2", "H1", "H2"]
    assert agent.experience_texts[0] == ""
    assert agent.experience_texts[1] == ""
    assert agent.experience_texts[2].startswith(EXPERIENCE_HEADER)


def test_injection_text_mentions_only_matching_experience(tmp_path):
    """H1 是库存题，不该被注入物流那条经验。"""
    tasks = make_tasks()
    store = make_store(tmp_path)
    learn(make_agent(), tasks, store, allowed_tools=ALLOWED)

    agent = make_agent()
    compare = verify(agent, tasks, store, allowed_tools=ALLOWED)

    h1_text = agent.experience_texts[2]
    assert "search_products" in h1_text
    assert "get_logistics" not in h1_text

    cov = {row["task_id"]: row for row in compare.coverage()}
    assert cov["H1"]["injected"] is True
    assert cov["H2"]["injected"] is True


def test_verify_records_injection_and_outcome(tmp_path):
    tasks = make_tasks()
    store = make_store(tmp_path)
    learn(make_agent(), tasks, store, allowed_tools=ALLOWED)

    before = store.stats()
    compare = verify(make_agent(), tasks, store, allowed_tools=ALLOWED)
    after = store.stats()

    # 两条经验各被注入到一条任务上
    assert after["injected_total"] - before["injected_total"] == 2
    # 注入之后都通过了，support 各 +1
    assert after["support_total"] - before["support_total"] == 2
    assert compare.touched == 2
    assert compare.retired == 0


def test_outcome_skips_tasks_that_passed_without_experience(tmp_path):
    """基线本来就能过的题，不能把"注入后仍然通过"记成经验的功劳。

    旧口径只要注入组通过就 support+1，于是"经验根本没起作用"也会涨分。
    这里加一条不做任何断言的 H3：它必然 PASS，又因为与 H1 同关键词会被注入，
    正好构成"基线过 + 注入过"这一格。support 只该涨 H1/H2 的两分。
    """
    tasks = make_tasks()
    already_ok = Task(
        id="H3",
        input="这个蓝牙耳机还有现货吗",
        assertions={},
        split="holdout",
    )
    store = make_store(tmp_path)
    learn(make_agent(), tasks, store, allowed_tools=ALLOWED)

    before = store.stats()
    compare = verify(make_agent(), tasks + [already_ok], store, allowed_tools=ALLOWED)
    after = store.stats()

    assert compare.baseline.verdicts["H3"].passed is True
    assert compare.injected.verdicts["H3"].passed is True
    # H1/H2 是"失败 -> 通过"，各 +1；H3 基线就过，不计
    assert after["support_total"] - before["support_total"] == 2


def test_verify_resets_between_sides(tmp_path):
    """每一侧开跑前都要重置被测服务，且顺序是 基线 -> 注入。

    为什么这条必须有：mock 的退款记录是全局状态，同一订单二次退款返回 409。
    基线的写操作会污染注入组，让它在同样的订单上失败——那会产出一个
    假的"经验有害"。pipeline 不认识 HTTP，所以这里只验钩子被正确调用，
    真正发请求的是入口脚本。
    """
    tasks = make_tasks()
    store = make_store(tmp_path)
    learn(make_agent(), tasks, store, allowed_tools=ALLOWED)

    calls: list[str] = []
    compare = verify(
        make_agent(),
        tasks,
        store,
        allowed_tools=ALLOWED,
        before_side=calls.append,
    )

    assert calls == ["baseline", "injected"]
    # 钩子只负责重置，不该改变判定结果
    assert compare.baseline.total == compare.injected.total


def test_verify_without_hook_keeps_old_behaviour(tmp_path):
    """不传钩子的调用方（旧代码、测试）行为不变：不需要任何回调也能跑完。"""
    tasks = make_tasks()
    store = make_store(tmp_path)
    learn(make_agent(), tasks, store, allowed_tools=ALLOWED)

    compare = verify(make_agent(), tasks, store, allowed_tools=ALLOWED, record=False)

    assert compare.injected.total > 0


def test_verify_creates_no_new_experience(tmp_path):
    """验证阶段不许往库里加新经验——加了就是拿验证集训练。"""
    tasks = make_tasks()
    store = make_store(tmp_path)
    learn(make_agent(), tasks, store, allowed_tools=ALLOWED)

    before = len(store.all())
    verify(make_agent(), tasks, store, allowed_tools=ALLOWED)
    assert len(store.all()) == before


def test_record_false_leaves_store_untouched(tmp_path):
    tasks = make_tasks()
    store = make_store(tmp_path)
    learn(make_agent(), tasks, store, allowed_tools=ALLOWED)

    before = store.stats()
    verify(make_agent(), tasks, store, allowed_tools=ALLOWED, record=False)
    after = store.stats()

    assert after["injected_total"] == before["injected_total"]
    assert after["support_total"] == before["support_total"]


def test_empty_store_makes_both_sides_identical(tmp_path):
    """空库时两组必须完全一样。不一样就说明差异来自别处，不是经验。"""
    tasks = make_tasks()
    compare = verify(make_agent(), tasks, make_store(tmp_path), allowed_tools=ALLOWED)

    assert compare.baseline.passed == 0
    assert compare.injected.passed == 0
    assert compare.flips()["up"] == []
    assert compare.flips()["down"] == []
    assert compare.injected_tasks() == 0
    assert all(not row["injected"] for row in compare.coverage())


def test_verify_without_holdout_raises(tmp_path):
    only_fit = [t for t in make_tasks() if t.split == "fit"]
    with pytest.raises(ValueError, match="holdout"):
        verify(make_agent(), only_fit, make_store(tmp_path), allowed_tools=ALLOWED)


def test_compare_dict_is_json_serializable(tmp_path):
    tasks = make_tasks()
    store = make_store(tmp_path)
    learn(make_agent(), tasks, store, allowed_tools=ALLOWED)
    compare = verify(make_agent(), tasks, store, allowed_tools=ALLOWED)

    text = json.dumps(compare.to_dict(), ensure_ascii=False)
    payload = json.loads(text)

    assert payload["delta"]["flips_up"] == 2
    assert payload["delta"]["coverage_rate"] == 1.0
    assert payload["baseline"]["per_task"]["H1"]["passed"] is False
    assert payload["injected"]["per_task"]["H1"]["passed"] is True
    assert payload["injected"]["per_task"]["H1"]["calls"] == ["search_products"]


def test_learn_dict_is_json_serializable(tmp_path):
    result = learn(make_agent(), make_tasks(), make_store(tmp_path), allowed_tools=ALLOWED)
    payload = json.loads(json.dumps(result.to_dict(), ensure_ascii=False))
    assert payload["kept"] == 2
    assert payload["candidates"] == 2
    assert len(payload["experiences"]) == 2


# ---------------------------------------------------------------- 解耦守护


def _imported_modules(path: Path) -> set[str]:
    src = path.read_text(encoding="utf-8")
    return {m.split(".")[0] for m in IMPORT_RE.findall(src)}


def test_pipeline_imports_nothing_business_specific():
    """源码级扫描：pipeline 只许依赖 core / evaluator / memory 与标准库。"""
    allowed = {"core", "evaluator", "memory", "pipeline"}
    stdlib_ok = {
        "dataclasses", "typing", "json", "re", "time", "pathlib", "collections",
    }
    for path in (ROOT / "pipeline").glob("*.py"):
        for mod in _imported_modules(path):
            if mod in allowed or mod in stdlib_ok or mod.startswith("__"):
                continue
            pytest.fail(f"{path.name} 依赖了 {mod}，编排层不该认识业务或框架")


def test_pipeline_imports_without_agents_or_langchain():
    """子进程验证：import pipeline 不会把 agents / demos / langchain 拖进来。"""
    code = (
        "import sys, pipeline; "
        "bad = [m for m in sys.modules "
        "if m.split('.')[0] in ('agents', 'demos', 'langchain', 'langchain_core')]; "
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
