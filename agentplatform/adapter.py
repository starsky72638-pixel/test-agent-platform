"""WorkflowAgent：把一份定义包成符合 Agent Protocol 的执行对象。

它只做三件事，其余全部交给已有的 Runtime：
    1. 校验定义（结构 + 拓扑 + 工具存在性），不合法就直接拒绝，
       不让一个坏定义跑到一半才崩；
    2. 逐轮调用 Executor；
    3. 把结果组装成 Trace——1 个 Turn 承载对话，N 个 Event 承载执行过程。

刻意不做的事：
    - 不把节点写成 Turn：Turn 的数量由用户轮次决定，与节点数无关；
    - 不碰 core.runner：崩溃隔离、split 校正、耗时兜底都由 Runtime 负责；
    - 不 import 任何业务 Agent：工具池由接入层（agents/builtin.py）注入，
      这一层不知道有 shop 或 customer。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from core.trace import KIND_NODE_END, STATUS_INFRA, Trace, Turn, now_ms

from agentplatform.definition import DefinitionError, WorkflowDefinition
from agentplatform.executor import ExecutionError, Executor
from agentplatform.schema import validate
from agentplatform.topology import check as check_topology


def _tool_map(tools) -> dict[str, Callable]:
    """{工具名: 可调用对象}。

    同时兼容 LangChain 工具（有 name 属性、用 invoke 调）和普通函数
    （用 __name__、直接调用），这样测试不必为了注入假工具去装一个 LC 工具。
    """
    out: dict[str, Callable] = {}
    for item in tools or []:
        name = getattr(item, "name", None) or getattr(item, "__name__", "")
        if not name:
            raise ValueError(f"工具缺少名字，无法注册：{item!r}")
        out[str(name)] = item
    return out


def compile_workflow(
    definition: WorkflowDefinition,
    *,
    tools: dict[str, Callable] | None = None,
) -> None:
    """执行前的全量校验。不合法直接抛，错误一次列全。"""
    available = set((tools or {}).keys())
    errors = validate(definition, available_tools=available) + check_topology(definition)
    if errors:
        raise DefinitionError(
            "Workflow 定义不合法：\n" + "\n".join(f"  - {e}" for e in errors)
        )


class WorkflowAgent:
    """按名字接入 Runtime 的 Workflow 执行体。"""

    def __init__(
        self,
        definition: WorkflowDefinition,
        *,
        tools=None,
        llm: Any = None,
        name: str = "",
        memory_store: Any = None,
    ) -> None:
        self.definition = definition
        self.name = name or f"wf:{definition.workflow_id}"
        # 暴露原始工具列表：run_loop / run_shop 的 tool_names_of 会读它。
        self.tools = list(tools or [])
        self._llm = llm
        # 经验库显式注入：不传就是没有，memory 节点输出空串。
        # 不设默认路径，避免 A 的 Workflow 悄悄读到 B 实验留下的经验。
        self.memory_store = memory_store
        self._tool_map = _tool_map(self.tools)
        # 构造时就校验：一个坏定义不该等到有人点"运行"才暴露。
        compile_workflow(definition, tools=self._tool_map)

    # -- Agent Protocol ------------------------------------------

    def run(self, task, ctx=None) -> Trace:
        trace = Trace(task_id=task.id, agent=self.name, split=task.split)
        trace.meta["workflow"] = {
            "id": self.definition.workflow_id,
            "version": self.definition.version,
            "name": self.definition.name,
            "nodes": len(self.definition.nodes),
            "entry": self.definition.entry,
        }
        # Runtime 递进来的经验只记字数：Workflow 取经验走 memory 节点，
        # 读的是构造时注入的经验库。两条来源不合并——
        # 合并会出现"同一条经验插两遍"，且说不清是哪条起了作用。
        # 键名保持 experience_chars：probe 与 runbook 都按它统计注入量。
        experience = getattr(ctx, "experience", "") if ctx is not None else ""
        if experience and experience.strip():
            trace.meta["experience_chars"] = len(experience)

        executor = Executor(
            self.definition,
            tools=self._tool_map,
            llm=self._llm,
            memory_store=self.memory_store,
        )

        started = now_ms()
        for index, text in enumerate(task.user_turns()):
            turn_started = now_ms()
            try:
                answer = executor.run(trace, turn=index, user_input=text)
            except ExecutionError as exc:
                # 节点崩了属于链路问题，不是"答错了"。
                # 标成 infra_error，评测阶段会把它当链路故障剔除，
                # 而不是记成模型能力问题。
                trace.status = STATUS_INFRA
                trace.error = str(exc)
                trace.turns.append(
                    Turn(
                        index=index,
                        user_input=text,
                        calls=[],
                        answer="",
                        elapsed_ms=max(0, now_ms() - turn_started),
                        meta={"status": "error", "error": str(exc)},
                    )
                )
                break

            steps = sum(
                1
                for event in trace.events
                if event.kind == KIND_NODE_END and event.turn == index
            )
            # calls 刻意留空：工具调用记在 events 里，由 all_calls() 统一投影。
            # 两边都写会让判分的 call_count 翻倍。
            trace.turns.append(
                Turn(
                    index=index,
                    user_input=text,
                    calls=[],
                    answer=answer,
                    steps=steps,
                    elapsed_ms=max(0, now_ms() - turn_started),
                    meta={"status": "ok"},
                )
            )

        trace.elapsed_ms = max(0, now_ms() - started)
        return trace


def build_workflow_agent(
    ref: str | None = None,
    *,
    tools=None,
    llm: Any = None,
    name: str = "",
    memory_store: Any = None,
    **_ignored: Any,
) -> WorkflowAgent:
    """按 id 或路径装配一个 WorkflowAgent。

    tools 必填：工具池属于接入层的知识，放在这里会让 platform 反向依赖业务。
    """
    import os

    target = str(ref or os.environ.get("WORKFLOW_ID") or "").strip()
    if not target:
        raise DefinitionError(
            "需要指定 workflow：给 --workflow <id|路径>，或设置环境变量 WORKFLOW_ID"
        )

    # 明确给 .json 路径时按路径读；只给 id 时一律走 store。
    # 两者必须共用一个根目录，否则会出现"存进去的文件"和"读出来的文件"
    # 不是同一个——那种 bug 只在换了工作目录时才暴露，极难查。
    if Path(target).suffix == ".json":
        definition = WorkflowDefinition.load(target)
    else:
        from agentplatform.store import AgentStore

        definition = AgentStore().load(target)

    return WorkflowAgent(
        definition,
        tools=tools,
        llm=llm,
        name=name,
        memory_store=memory_store,
    )
