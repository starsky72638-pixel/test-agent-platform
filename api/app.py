"""最小 HTTP API：把"按名字跑一条"暴露成 4 个端点。

定位与 scripts/run_agent.py 完全一致——只执行与记录，不判分、不学经验。
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

from fastapi import FastAPI, HTTPException
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

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=DEFAULT_PORT)
