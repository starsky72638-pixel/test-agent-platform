"""Agent 定义的存储与按名寻址。

三件事必须钉住：
1. 存下去能原样读回来（含描述与版本）；
2. 覆盖保存时版本递增，而不是默默覆盖成同一版；
3. 一个坏文件不能让整个列表变空——那会让人以为"什么都没存"。

全程用临时目录，不碰仓库里的 data/workflows。
"""

from __future__ import annotations

from pathlib import Path

import pytest

import agentplatform.store as store_module
from agentplatform import AgentStore, DefinitionError, WorkflowDefinition

import agents.builtin as builtin


def make_defn(workflow_id: str = "t1", version: int = 1) -> WorkflowDefinition:
    return WorkflowDefinition.from_dict(
        {
            "workflow_id": workflow_id,
            "version": version,
            "name": "测试用",
            "description": "一句话说明",
            "entry": "start",
            "nodes": [
                {"id": "start", "type": "start", "next": ["end"]},
                {"id": "end", "type": "end"},
            ],
        }
    )


@pytest.fixture()
def store(tmp_path: Path, monkeypatch) -> AgentStore:
    # 必须在调用时替换：AgentStore 在 __init__ 里读这个值
    monkeypatch.setattr(store_module, "DEFAULT_ROOT", tmp_path)
    return AgentStore()


def test_save_and_load_roundtrip(store):
    path = store.save(make_defn())

    assert path.name == "t1.json"
    back = store.load("t1")
    assert back.workflow_id == "t1"
    assert back.description == "一句话说明"
    assert back.node_ids() == ["start", "end"]


def test_save_bumps_version_on_overwrite(store):
    store.save(make_defn(version=1))
    store.save(make_defn(version=1))

    # 版本由存储层递增：调用方传什么都不该把历史压回 v1
    assert store.load("t1").version == 2


def test_save_rejects_invalid_definition(store):
    bad = make_defn()
    bad.nodes = [n for n in bad.nodes if n.id != "end"]

    with pytest.raises(DefinitionError, match="end"):
        store.save(bad)
    assert not store.exists("t1"), "校验失败不该留下半个文件"


def test_list_agents_reports_broken_file(store, tmp_path: Path):
    store.save(make_defn())
    (tmp_path / "broken.json").write_text("{oops", encoding="utf-8")

    rows = {s.workflow_id: s for s in store.list_agents()}

    assert rows["t1"].error == ""
    assert rows["t1"].nodes == 2
    assert "不是合法 JSON" in rows["broken"].error


def test_ids_tolerates_missing_dir(tmp_path: Path):
    empty = AgentStore(root=tmp_path / "nope")

    assert empty.ids() == []
    assert empty.list_agents() == []


def test_names_includes_disk_workflows(store):
    store.save(make_defn("alpha"))

    listing = builtin.names()
    # 新建的定义立刻可寻址
    assert "wf:alpha" in listing
    # 静态类型不受影响
    assert "workflow" in listing
    assert "customer" in listing


def test_make_resolves_wf_prefix(store):
    """存下去就能按名字取出来，不需要注册、不需要重启进程。"""
    store.save(make_defn("alpha"))

    agent = builtin.make("wf:alpha")

    assert agent.name == "wf:alpha"
    assert agent.definition.workflow_id == "alpha"
