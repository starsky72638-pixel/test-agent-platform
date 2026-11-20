"""HTTP 接入层：把 Runtime 暴露成 4 个端点。

只依赖 agents.builtin 与 core 三件套（runner / task / trace），
不碰 evaluator、memory、pipeline——那些属于"跑完之后"的事。
"""

from api.app import create_app

__all__ = ["create_app"]
