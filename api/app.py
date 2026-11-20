"""最小 HTTP API：执行、回读、以及 Agent 定义的管理。

四类端点：
    执行与回读  GET /health、GET /agents、POST /runs、GET /runs/{id}
    定义管理    GET /workflows、GET /workflows/{id}、POST /workflows

存的是定义而不是运行中的对象：保存后立刻可以用 agent="wf:<id>" 跑起来，
不需要重启服务。本层不判分、不学经验，只执行与记录。

为什么不放进 core/：core 的契约是"删掉 agents/ 仍可 import 和测试"，
而这里必须先知道有哪些 Agent 才能列出来，属于接入层之上的薄壳。
它只调用三样东西：builtin.make、core.runner.run_task、core.trace.save_traces。

执行是同步的：请求内跑完再返回，单条约 10–50 秒。
代价是请求会等；收益是没有内存状态机，进程重启不丢任何东西。
真要做长任务，再引入队列，届时只需换这一层，core 不用动。

只监听 127.0.0.1，没有鉴权：这是本地开发工具，不做公网假设。
"""

from __future__ import annotations

import re
import secrets
import sys
import time
from pathlib import Path

# 允许直接运行：python api\app.py
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import Body, FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import agents.builtin as builtin
from config import PROJECT_ROOT
from core.registry import UnknownAgentError
from core.runner import run_task
from core.task import SPLIT_UNASSIGNED, Task
from core.trace import load_traces, save_traces

RUNS_DIR = PROJECT_ROOT / "runs" / "api"

# run_id 直接拼进文件路径，必须先卡死字符集，否则 /runs/.. 能穿越目录。
RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# workflow_id 会变成 data/workflows/<id>.json 的文件名，同样必须先卡字符集。
WORKFLOW_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# 8000 被 mock_server 占；8010 留给本机的另一个项目，这里避开。
DEFAULT_PORT = 8020


class RunRequest(BaseModel):
    agent: str = Field(default=builtin.DEFAULT_AGENT, description="Agent 名字，可用别名胜")
    question: str = Field(default="", description="单轮问题；与 turns 至少给一个")
    turns: list[str] = Field(default_factory=list, description="多轮输入，非空时优先")
    max_steps: int = Field(default=6, ge=1, le=20, description="单轮步数上限")


def new_run_id() -> str:
    """时间戳 + 3 字节随机，防止同一秒并发撞名。"""
    return f"api-{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"


def trace_path(run_id: str) -> Path:
    return RUNS_DIR / run_id / f"{run_id}.jsonl"


def check_workflow_id(workflow_id: str) -> None:
    """id 拼成文件名之前必须过这一关，否则 ../ 能写到目录外。"""
    if not WORKFLOW_ID_RE.fullmatch(str(workflow_id or "")):
        raise HTTPException(
            status_code=400,
            detail="workflow_id 只允许字母、数字、下划线与短横线，长度 1-64",
        )


def workflow_store():
    """每次请求新建一个：测试会替换 store 的默认根目录，缓存住会读不到。"""
    from agentplatform.store import AgentStore

    return AgentStore()


def _required_args(tool_obj) -> list[str]:
    """工具的必填参数名。

    走 args_schema 的 JSON Schema，而不是直接读 pydantic 字段：
    读字段要区分 model_fields 与 __fields__，两种 pydantic 版本写法不同，
    而 JSON Schema 两边都只有同一个键 'required'。
    """
    schema = getattr(tool_obj, "args_schema", None)
    if schema is None:
        return []
    try:
        data = (
            schema.model_json_schema()
            if hasattr(schema, "model_json_schema")
            else schema.schema()
        )
    except Exception:  # 取不到就当没有必填项，不要因为它让 /tools 整个 500
        return []
    return [str(x) for x in (data.get("required") or [])]


def summarize(trace, path: Path) -> dict:
    """执行摘要：够 UI 直接渲染，完整内容走 GET /runs/{id}。

    工具返回的 output 故意不在这里返回——它可能很长，
    单条轨迹回读一次就能拿全，不必让每次 POST 都背着。
    """
    return {
        "run_id": trace.task_id,
        "agent": trace.agent,
        "status": trace.status,
        "error": trace.error,
        "answer": trace.final_answer(),
        "elapsed_ms": trace.elapsed_ms,
        "turns": [
            {
                "index": t.index,
                "input": t.user_input,
                "answer": t.answer,
                "calls": [
                    {"name": c.name, "args": c.args, "ok": c.ok, "error": c.error}
                    for c in t.calls
                ],
            }
            for t in trace.turns
        ],
        "trace_path": str(path),
    }


def create_app() -> FastAPI:
    app = FastAPI(
        title="test_agent API",
        version="0.1.0",
        description="按名字跑一个 Agent，拿回逐轮轨迹。只执行与记录，不判分。",
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "agents": len(builtin.names())}

    @app.get("/agents")
    def list_agents() -> dict:
        return {
            "default": builtin.DEFAULT_AGENT,
            "agents": [
                {"name": n, "default": n == builtin.DEFAULT_AGENT}
                for n in builtin.names()
            ],
            "aliases": dict(builtin.ALIASES),
        }

    @app.get("/tools")
    def list_tools() -> dict:
        """Workflow 能用的工具池，附带每个工具的必填参数。

        必须与 agents/builtin.py:_workflow_factory 注入的是同一份 CUSTOMER_TOOLS。
        前端若自己写死一份名单，画布上选得到、运行时却报"工具未注册"——
        那种错误只在点"运行"那一刻才暴露，最难查。

        required 是给画布做保存前校验用的：缺必填参数时后端能跑，
        但会在节点里抛 ValidationError，结果是空回答 + infra_error。
        那种失败看起来像"模型不会回答"，其实是参数没填。
        """
        from agents.customer.tools import CUSTOMER_TOOLS

        names = sorted(t.name for t in CUSTOMER_TOOLS)
        required = {t.name: _required_args(t) for t in CUSTOMER_TOOLS}
        # 默认给只读工具：写操作有副作用、必填参数更多，
        # 拿它当新节点的默认值，等于让第一次拖拽就必然报错。
        #
        # 这里必须显式点名 get_order，不能写 read_only[0]：
        # 字典序下 "get_logistics" < "get_order"，取第一个会拿到物流。
        # 而任意订单都有状态与金额，物流对未发货订单没意义——
        # 默认值该选"对任何输入都有话说"的那个。
        read_only = [n for n in names if n.startswith("get_")]
        if "get_order" in names:
            default = "get_order"
        elif read_only:
            default = read_only[0]
        else:
            default = names[0] if names else ""
        return {"tools": names, "required": required, "default": default}

    @app.post("/runs")
    def create_run(req: RunRequest) -> dict:
        turns = [str(t) for t in (req.turns or []) if str(t).strip()]
        question = (req.question or "").strip()
        if not turns and not question:
            raise HTTPException(status_code=422, detail="question 与 turns 至少给一个")

        try:
            agent = builtin.make(req.agent, max_steps=req.max_steps)
        except UnknownAgentError as exc:
            # 400 而不是 404：路径是对的，是请求体里的名字错了。
            raise HTTPException(
                status_code=400,
                detail=str(exc.args[0]) if exc.args else str(exc),
            ) from exc
        except (TypeError, ValueError) as exc:
            # 缺密钥、参数非法这类构造期错误，属于调用方配置问题，不是服务端 500。
            raise HTTPException(
                status_code=400, detail=f"{type(exc).__name__}: {exc}"
            ) from exc

        run_id = new_run_id()
        task = Task(
            id=run_id,
            input=question or turns[0],
            turns=turns,
            split=SPLIT_UNASSIGNED,
        )
        trace = run_task(agent, task, max_steps=req.max_steps)
        path = save_traces([trace], trace_path(run_id))
        return summarize(trace, path)

    @app.get("/runs/{run_id}")
    def get_run(run_id: str) -> dict:
        if not RUN_ID_RE.fullmatch(run_id):
            raise HTTPException(
                status_code=400, detail="run_id 只允许字母、数字、下划线与短横线"
            )
        path = trace_path(run_id)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"没有这次执行：{run_id}")
        traces = load_traces(path)
        if not traces:
            raise HTTPException(status_code=500, detail="轨迹文件存在但没有内容")
        return traces[0].to_dict()

    # -- Agent 管理 ----------------------------------------------
    # 存的是定义，不是运行中的对象：保存后立刻能用 POST /runs 的
    # agent="wf:<id>" 跑起来，不需要重启服务。

    @app.get("/workflows")
    def list_workflows() -> dict:
        return {"agents": [s.to_dict() for s in workflow_store().list_agents()]}

    @app.get("/workflows/{workflow_id}")
    def get_workflow(workflow_id: str) -> dict:
        from agentplatform.definition import DefinitionError

        check_workflow_id(workflow_id)
        try:
            return workflow_store().load(workflow_id).to_dict()
        except DefinitionError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/workflows")
    def save_workflow(payload: dict = Body(...)) -> dict:
        from agentplatform.definition import DefinitionError, WorkflowDefinition

        try:
            definition = WorkflowDefinition.from_dict(payload)
        except DefinitionError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        check_workflow_id(definition.workflow_id)
        try:
            path = workflow_store().save(definition)
        except DefinitionError as exc:
            # 结构/图不合法属于调用方输入错误，不是服务端 500。
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return {
            "workflow_id": definition.workflow_id,
            "version": definition.version,
            "name": definition.name,
            "path": str(path),
        }

    # 同源挂载界面：避免 CORS，也不必把页面和 API 拆成两个进程。
    # 只挂 /ui，不动根路径，/docs 与上面四条路由都不受影响。
    # 目录不存在时跳过，保证裁剪部署（只留 API）仍能启动。
    web_dir = PROJECT_ROOT / "web"
    if web_dir.is_dir():
        app.mount("/ui", StaticFiles(directory=web_dir, html=True), name="ui")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=DEFAULT_PORT)
