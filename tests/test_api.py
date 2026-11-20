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


# ---------------------------------------------------------------- 定义管理


WF_BODY = {
    "workflow_id": "demo-t",
    "version": 1,
    "name": "接口创建的 Agent",
    "description": "只用 start/end，不联网",
    "entry": "start",
    "nodes": [
        {"id": "start", "type": "start", "next": ["end"]},
        {"id": "end", "type": "end"},
    ],
}


@pytest.fixture()
def wf_root(tmp_path: Path, monkeypatch) -> Path:
    """定义写进临时目录，绝不碰仓库的 data/workflows。"""
    import agentplatform.store as store_module

    monkeypatch.setattr(store_module, "DEFAULT_ROOT", tmp_path / "workflows")
    return tmp_path / "workflows"


def test_workflow_save_list_get(client, wf_root):
    created = client.post("/workflows", json=WF_BODY)
    assert created.status_code == 200
    assert created.json()["version"] == 1

    listing = client.get("/workflows").json()["agents"]
    assert [a["workflow_id"] for a in listing] == ["demo-t"]
    assert listing[0]["description"] == "只用 start/end，不联网"

    detail = client.get("/workflows/demo-t").json()
    assert detail["workflow_id"] == "demo-t"
    assert len(detail["nodes"]) == 2


def test_workflow_save_rejects_invalid_definition(client, wf_root):
    bad = dict(WF_BODY, nodes=[{"id": "start", "type": "start", "next": ["ghost"]}])

    r = client.post("/workflows", json=bad)

    # 结构问题属于调用方输入错误，用 400 而不是 500
    assert r.status_code == 400
    assert "ghost" in r.json()["detail"]


def test_workflow_id_is_validated(client, wf_root):
    # id 会拼成文件名：不卡字符集就等于开了目录穿越
    bad = dict(WF_BODY, workflow_id="../evil")
    assert client.post("/workflows", json=bad).status_code == 400
    assert client.get("/workflows/bad%20id").status_code == 400


def test_workflow_missing_is_404(client, wf_root):
    assert client.get("/workflows/nope-123").status_code == 404


# ---------------------------------------------------------------- 画布前端


def test_tools_endpoint_matches_runtime_toolbox(client):
    """/tools 必须等于 Workflow 工厂真正注入的工具池。

    两处各写一份名单的话，画布上能选到、运行时报"未注册"，
    而那种错误只在点"运行"那一刻才出现。
    """
    from agents.customer.tools import CUSTOMER_TOOLS

    body = client.get("/tools").json()
    assert body["tools"] == sorted(t.name for t in CUSTOMER_TOOLS)
    assert body["tools"] == ["apply_refund", "get_logistics", "get_order"]

    # params 让画布能画出参数行（名字/类型/必填/说明）：
    # 用户不必手写 JSON，也不会漏掉 reason 这种必填项。
    params = body["params"]
    assert [p["name"] for p in params["get_order"]] == ["order_id"]
    assert params["get_order"][0]["type"] == "integer"
    assert params["get_order"][0]["required"] is True
    # LangChain 生成的 schema 不带参数级说明，这里是 docstring 兜底的结果；
    # 断言具体字样，防止哪天兜底失效只返回空串还"看起来通过"
    assert "订单号" in params["get_order"][0]["description"]

    refund = {p["name"]: p for p in params["apply_refund"]}
    assert refund["order_id"]["required"] is True
    # reason 是真实踩过的坑：漏了就 ValidationError → infra_error + 空回答
    assert refund["reason"]["required"] is True
    assert refund["reason"]["type"] == "string"
    assert "退款原因" in refund["reason"]["description"]

    assert {p["name"] for p in params["get_logistics"]} == {"order_id"}

    # 默认必须是只读工具：写操作有副作用、必填参数更多，
    # 拿它当新节点的默认值，第一次拖拽就必然报错。
    assert body["default"] == "get_order"


def test_builder_page_is_served_by_ui_mount():
    """画布挂在 /ui 这条既有静态挂载下，不需要为它再写一条路由。

    若哪天把 web/ 目录挪走，这条会红——那时应该发现"界面没了"，
    而不是等打开浏览器看到 404。
    """
    bare = TestClient(api_app.app)
    page = bare.get("/ui/builder/")
    assert page.status_code == 200
    assert 'id="canvas"' in page.text

    script = bare.get("/ui/builder/app.js")
    assert script.status_code == 200
    assert "POST /workflows" in script.text
