"""scripts/run_agent.py 的离线测试：不联网、不花额度、不读密钥。

用 demos.echo.EchoAgent 作为注入的假 Agent，
证明"统一入口 + 落盘 + 回读"这条链在没有模型的情况下也是通的。
"""

from __future__ import annotations

from pathlib import Path

from core.trace import load_traces
from demos.echo import EchoAgent
from scripts.run_agent import build_task, main, run_once


def test_build_task_is_adhoc_and_unassigned():
    task = build_task("你好")
    assert task.id.startswith("adhoc-")
    # 临时问题不许混进 fit/holdout 的统计口径
    assert task.split == "unassigned"
    assert task.user_turns() == ["你好"]


def test_run_once_saves_reloadable_trace(tmp_path: Path):
    agent = EchoAgent()
    trace, path = run_once(agent, "你好", out_dir=tmp_path, task_id="adhoc-test")

    assert trace.status == "ok"
    assert trace.final_answer().startswith("[echo]")
    assert path.exists()

    back = load_traces(path)
    assert len(back) == 1
    assert back[0].task_id == "adhoc-test"
    assert back[0].agent == "echo"


def test_main_rejects_empty_question():
    """没有问题时不该构造 Agent，更不该发起执行。"""
    assert main(["--agent", "shop-cold", "--question", "   "]) == 2


def test_main_lists_agents(capsys):
    assert main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "shop-cold" in out
    assert "main" in out
