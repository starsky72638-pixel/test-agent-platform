"""经验库归并与默认库注入的离线测试。

三件事必须钉住：
1. 归并只取"最新快照"，不把 support/contradict 相加（那会凭空造数字）；
2. 归并不修改任何输入文件；
3. `wf:*` 真的拿到了默认库 —— 否则"线上 memory 节点恒 0 命中"会继续存在。

全程不联网：默认库用 monkeypatch 指到 tmp_path，工作流里不放 llm 节点。
"""

from __future__ import annotations

import json
from pathlib import Path

import agents.builtin as builtin
import memory.store as store_mod
from core.task import Task
from core.trace import KIND_MEMORY_CALL, STATUS_OK
from memory.extract import task_tags
from memory.schema import Experience, Trigger
from memory.store import DEFAULT_STORE_PATH, ExperienceStore, merge_files

QUERY = "订单 1002 用的哪家快递"


def make_exp(exp_id: str, action: str, *, version: int = 1, updated_at: int = 100) -> Experience:
    return Experience(
        exp_id=exp_id,
        failure_type="missing_tool",
        pattern="问快递却只查了订单",
        action=action,
        trigger=Trigger(tags=["快递", "订单"]),
        version=version,
        created_at=1000,
        updated_at=updated_at,
    )


def write_lib(path: Path, exps: list[Experience]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(e.to_line() + "\n" for e in exps), encoding="utf-8")
    return path


def read_lib(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# ---------------------------------------------------------------- 归并


def test_merge_keeps_newest_version_per_exp_id(tmp_path):
    older = write_lib(
        tmp_path / "a" / "experiences.jsonl",
        [make_exp("exp-x", "旧版本", version=1, updated_at=100)],
    )
    newer = write_lib(
        tmp_path / "b" / "experiences.jsonl",
        [
            make_exp("exp-x", "新版本", version=2, updated_at=200),
            make_exp("exp-y", "另一条", version=1, updated_at=150),
        ],
    )
    out = tmp_path / "merged.jsonl"

    result = merge_files([older, newer], out)

    assert result["sources"] == 2
    assert result["unique"] == 2
    rows = {r["exp_id"]: r for r in read_lib(out)}
    assert set(rows) == {"exp-x", "exp-y"}
    assert rows["exp-x"]["action"] == "新版本"
    assert rows["exp-x"]["version"] == 2
    # support 不累加：取快照，不做语义合并
    assert rows["exp-x"]["support"] == 1


def test_merge_does_not_touch_inputs(tmp_path):
    a = write_lib(tmp_path / "a.jsonl", [make_exp("exp-a", "动作")])
    b = write_lib(tmp_path / "b.jsonl", [make_exp("exp-b", "动作")])
    before = (a.read_bytes(), b.read_bytes())

    merge_files([a, b], tmp_path / "out.jsonl")

    assert (a.read_bytes(), b.read_bytes()) == before


def test_merge_skips_missing_file(tmp_path):
    a = write_lib(tmp_path / "a.jsonl", [make_exp("exp-a", "动作")])

    result = merge_files([a, tmp_path / "ghost.jsonl"], tmp_path / "out.jsonl")

    assert result["sources"] == 1
    assert result["unique"] == 1
    assert [r["exp_id"] for r in read_lib(tmp_path / "out.jsonl")] == ["exp-a"]


def test_dry_run_reports_without_writing(tmp_path):
    a = write_lib(tmp_path / "a.jsonl", [make_exp("exp-a", "动作")])
    out = tmp_path / "out.jsonl"

    result = merge_files([a], out, write=False)

    assert result["unique"] == 1
    assert result["out"] is None
    assert not out.exists()


def test_default_store_path_is_a_single_known_file():
    assert DEFAULT_STORE_PATH.name == "experiences.jsonl"
    assert DEFAULT_STORE_PATH.parent.name == "memory"
    assert DEFAULT_STORE_PATH.parent.parent.name == "data"


# ---------------------------------------------------------------- 默认库注入


def test_workflow_agent_gets_default_store(tmp_path, monkeypatch):
    target = tmp_path / "default.jsonl"
    monkeypatch.setattr(store_mod, "DEFAULT_STORE_PATH", target)

    agent = builtin.make("wf:branch-demo")

    assert agent.memory_store is not None
    assert Path(agent.memory_store.path) == target


def test_workflow_memory_node_hits_default_store(tmp_path, monkeypatch):
    """注入进来的默认库必须真的被 memory 节点读到，否则等于没接。"""
    target = tmp_path / "default.jsonl"
    store = ExperienceStore(target)
    store.add(
        Experience(
            exp_id="exp-merge-1",
            failure_type="missing_tool",
            pattern="问快递却只查了订单",
            action="先查快递再回答",
            trigger=Trigger(tags=task_tags(Task(id="t", input=QUERY))),
        )
    )
    monkeypatch.setattr(store_mod, "DEFAULT_STORE_PATH", target)

    definition = {
        "workflow_id": "mem-default",
        "entry": "start",
        "nodes": [
            {"id": "start", "type": "start", "next": ["recall"]},
            {
                "id": "recall",
                "type": "memory",
                "config": {"query": "{{input}}", "top_k": 3},
                "next": ["end"],
            },
            {"id": "end", "type": "end", "next": []},
        ],
    }
    path = tmp_path / "mem-default.json"
    path.write_text(json.dumps(definition, ensure_ascii=False), encoding="utf-8")

    agent = builtin.make("workflow", workflow=str(path))
    trace = agent.run(Task(id="M1", input=QUERY))

    assert trace.status == STATUS_OK
    events = [e for e in trace.events if e.kind == KIND_MEMORY_CALL]
    assert len(events) == 1
    assert events[0].args["hit_count"] == 1
    assert events[0].args["hits"] == ["exp-merge-1"]
    assert "先查快递再回答" in trace.turns[0].answer


def test_run_loop_store_default_is_opt_in(tmp_path):
    """默认仍是"每次新建"；只有显式写 default 才用归并库。"""
    from scripts.run_loop import resolve_store_path

    run_dir = tmp_path / "run"
    assert resolve_store_path(None, run_dir) == run_dir / "experiences.jsonl"
    # 与 store 模块当前常量比，避免 run_loop 里那份 import 绑定过期
    assert resolve_store_path("default", run_dir) == store_mod.DEFAULT_STORE_PATH
    assert resolve_store_path("runs/x/e.jsonl", run_dir) == Path("runs/x/e.jsonl")
