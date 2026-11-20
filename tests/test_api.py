"""api/app.py 的离线测试：TestClient + EchoAgent，不联网、不花额度、不读密钥。

这一层要证明的是"路由 -> Runtime -> 落盘 -> 回读"整条链是通的，
与模型能力无关，所以用 EchoAgent 让它在毫秒级可验证。

一处刻意的安排：unknown-agent 那条**不**用 fixture 里打了补丁的 make，
否则永远构造不出"名字不存在"的场景，测试会变成自证。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import agents.builtin as builtin
import api.app as api_app
from demos.echo import EchoAgent


@pytest.fixture()
def client(tmp_path: Path, monkeypatch) -> TestClient:
    # 轨迹写进临时目录，不污染仓库的 runs/api
    monkeypatch.setattr(api_app, "RUNS_DIR", tmp_path)
    # 不构造真 Agent：缺密钥的机器也要能跑这套测试
    monkeypatch.setattr(
        builtin, "make", lambda name, **kwargs: EchoAgent(name=f"fake-{name}")
    )
    return TestClient(api_app.app)


def test_health_reports_agent_count(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["agents"] >= 4


def test_agents_lists_registered_names(client):
    body = client.get("/agents").json()
    assert body["default"] == builtin.DEFAULT_AGENT
    assert [a["name"] for a in body["agents"]] == builtin.names()
    assert "customer" in [a["name"] for a in body["agents"]]
    assert body["aliases"]["cs"] == "customer"


def test_unknown_agent_returns_400():
    # 不用 client fixture：那条链路把 make 换了，造不出"名字不存在"
    bare = TestClient(api_app.app)
    r = bare.post("/runs", json={"agent": "no-such-agent", "question": "你好"})
    assert r.status_code == 400
    assert "customer" in r.json()["detail"]


def test_post_run_returns_summary_and_saves_trace(client):
    r = client.post("/runs", json={"agent": "customer", "question": "订单 1002 到哪了"})
    assert r.status_code == 200
    body = r.json()

    assert body["status"] == "ok"
    assert body["answer"].startswith("[echo]")
    assert body["turns"][0]["calls"][0]["name"] == "echo"
    assert body["turns"][0]["calls"][0]["args"] == {"text": "订单 1002 到哪了"}

    # 落盘的文件与 GET 回读必须是同一条：不另存一份，也不重新构造
    path = Path(body["trace_path"])
    assert path.exists()
    back = client.get(f"/runs/{body['run_id']}")
    assert back.status_code == 200
    assert back.json()["task_id"] == body["run_id"]
    assert back.json()["agent"] == "fake-customer"


def test_post_run_accepts_multi_turn(client):
    r = client.post(
        "/runs",
        json={"agent": "customer", "turns": ["1001 到哪了", "那算了吧，帮我退了"]},
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["turns"]) == 2
    assert body["turns"][1]["input"] == "那算了吧，帮我退了"


def test_run_id_is_validated_and_missing_run_is_404(client):
    # 非法字符直接拦住：run_id 会被拼进文件路径，不做校验等于开了目录穿越
    assert client.get("/runs/bad%20id").status_code == 400
    assert client.get("/runs/nope-123").status_code == 404


def test_ui_page_is_mounted():
    """界面必须与 API 同源挂载。

    file:// 打开再 fetch 127.0.0.1 会跨域，所以界面不能靠双击 html 用，
    必须由 /ui 提供；这条测试盯住挂载点不要被后来的路由挤掉。
    """
    bare = TestClient(api_app.app)
    r = bare.get("/ui/")
    assert r.status_code == 200
    assert "Agent Runtime" in r.text
