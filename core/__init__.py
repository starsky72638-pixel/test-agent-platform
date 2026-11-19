"""核心包：执行轨迹、任务契约、Agent 协议、注册表。

约束（由 tests/test_trace.py 的纯净性用例守着）：
- 只依赖标准库；
- 不出现任何业务域词汇；
- 不感知任何 Agent 框架的消息对象。

因此删掉 demos/ 之后，本包仍可独立 import 与测试。
"""

from core.protocol import Agent, AgentContractError, RunContext, validate_agent
from core.registry import (
    DEFAULT,
    DuplicateAgentError,
    Registry,
    UnknownAgentError,
)
from core.task import (
    FIT_IDS,
    SPLIT_FIT,
    SPLIT_HOLDOUT,
    SPLIT_UNASSIGNED,
    Task,
    assign_split,
    load_many,
    load_tasks,
    split_counts,
)
from core.trace import (
    SCHEMA_VERSION,
    STATUS_CRASHED,
    STATUS_EMPTY,
    STATUS_INFRA,
    STATUS_MAX_STEPS,
    STATUS_OK,
    ToolCall,
    Trace,
    Turn,
    load_traces,
    save_traces,
    summarize,
)

__all__ = [
    "Agent",
    "AgentContractError",
    "RunContext",
    "validate_agent",
    "Registry",
    "DEFAULT",
    "DuplicateAgentError",
    "UnknownAgentError",
    "Task",
    "FIT_IDS",
    "SPLIT_FIT",
    "SPLIT_HOLDOUT",
    "SPLIT_UNASSIGNED",
    "assign_split",
    "load_tasks",
    "load_many",
    "split_counts",
    "ToolCall",
    "Turn",
    "Trace",
    "SCHEMA_VERSION",
    "STATUS_OK",
    "STATUS_INFRA",
    "STATUS_MAX_STEPS",
    "STATUS_CRASHED",
    "STATUS_EMPTY",
    "load_traces",
    "save_traces",
    "summarize",
]
