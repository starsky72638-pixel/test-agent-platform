"""Mini Agent Platform：Workflow 的定义、校验与执行。

为什么包名是 agentplatform 而不是 platform：
    platform 是标准库模块。openai SDK 构造 User-Agent 时会 `import platform`，
    项目自带同名包会把标准库遮蔽，结果是每次模型调用都崩，
    而报错信息里完全看不出根因。改一个词省掉一整类事故。

分层，各自只依赖下面一层：
    definition  定义与 JSON 读写，不认识执行
    schema      字段校验
    topology    图校验与拓扑排序
    executor    按序执行节点，把过程写成 Event
    adapter     包成 Agent Protocol，交给 core.runner
"""

from agentplatform.adapter import WorkflowAgent, build_workflow_agent, compile_workflow
from agentplatform.definition import (
    BRANCH_KEYS,
    CONDITION_OPS,
    NODE_CONDITION,
    NODE_END,
    NODE_LLM,
    NODE_MEMORY,
    NODE_START,
    NODE_TOOL,
    NODE_TYPES,
    DefinitionError,
    NodeDef,
    WorkflowDefinition,
)
from agentplatform.executor import ExecutionError, Executor, compare, render, render_obj
from agentplatform.extractor import EXTRACT_FIELDS, ExtractError, extract_field
from agentplatform.schema import validate
from agentplatform.store import AgentStore, AgentSummary
from agentplatform.topology import TopologyError, topo_sort

__all__ = [
    "WorkflowAgent",
    "build_workflow_agent",
    "compile_workflow",
    "AgentStore",
    "AgentSummary",
    "WorkflowDefinition",
    "NodeDef",
    "DefinitionError",
    "NODE_START",
    "NODE_LLM",
    "NODE_TOOL",
    "NODE_CONDITION",
    "NODE_MEMORY",
    "NODE_END",
    "NODE_TYPES",
    "BRANCH_KEYS",
    "CONDITION_OPS",
    "Executor",
    "ExecutionError",
    "compare",
    "render",
    "render_obj",
    "extract_field",
    "EXTRACT_FIELDS",
    "ExtractError",
    "validate",
    "topo_sort",
    "TopologyError",
]
