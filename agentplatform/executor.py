"""节点执行器：按拓扑序跑节点，把每一步写进 Trace 的 events。

三条边界，都是刻意的：
1. 只写 events，不碰 turns。Turn 由 adapter 组装，语义是"一次用户交互"；
   把节点塞进 Turn 会让判分、蒸馏、界面三处都拿不到真实结构。
2. 出错不吞：先记一条 error 事件，再抛 ExecutionError。
   吞掉会让整条轨迹看起来"跑完了但答案为空"，那是更难查的失败。
3. 不认识业务：工具与模型都是构造时注入的，这一层不知道谁在用。

为什么事件由 Trace 编号：
    同一轮里多个节点各自记 seq 一定撞号。集中分配，"谁先谁后"才有唯一来源。
"""

from __future__ import annotations

import re
from typing import Any, Callable

from core.trace import (
    KIND_ERROR,
    KIND_LLM_CALL,
    KIND_NODE_END,
    KIND_NODE_START,
    KIND_TOOL_CALL,
    Trace,
    now_ms,
)

from agentplatform.definition import (
    NODE_END,
    NODE_LLM,
    NODE_START,
    NODE_TOOL,
    NodeDef,
    WorkflowDefinition,
)
from agentplatform.topology import topo_sort

_TEMPLATE = re.compile(r"\{\{\s*([A-Za-z0-9_.\-]+)\s*\}\}")

# 节点输入/输出进事件时截断，避免整段 prompt 把 jsonl 撑大。
_TEXT_LIMIT = 400


class TemplateError(ValueError):
    """模板引用了不存在或尚未产出的变量。"""


class ExecutionError(RuntimeError):
    """某个节点执行失败。带上节点 id 与类型，便于直接定位。"""

    def __init__(self, node_id: str, node_type: str, message: str) -> None:
        self.node_id = node_id
        self.node_type = node_type
        self.message = message
        super().__init__(f"{node_type}@{node_id} 失败：{message}")


def _lookup(key: str, ctx: dict) -> Any:
    if key == "input":
        return ctx.get("input", "")
    parts = key.split(".")
    if len(parts) == 3 and parts[0] == "nodes":
        produced = (ctx.get("nodes") or {}).get(parts[1])
        if produced is None:
            raise TemplateError(f"引用了尚未执行的节点 {parts[1]!r}")
        if parts[2] not in produced:
            raise TemplateError(f"节点 {parts[1]!r} 没有字段 {parts[2]!r}")
        return produced[parts[2]]
    raise TemplateError(f"未知模板变量 {key!r}，可用：input、nodes.<节点id>.output")


def render(text: str, ctx: dict) -> str:
    """把 {{input}} / {{nodes.<id>.output}} 替换成实际值。

    引用不存在时抛错而不是替换成空串：空串会让 prompt 少一段上下文，
    模型照样给出一个像样的答案，没人发现自己写错了变量名。
    """

    def _sub(match: re.Match) -> str:
        return str(_lookup(match.group(1), ctx))

    return _TEMPLATE.sub(_sub, str(text))


def render_obj(value: Any, ctx: dict) -> Any:
    """对 dict/list 里的字符串逐个渲染，其余类型原样返回。"""
    if isinstance(value, str):
        return render(value, ctx)
    if isinstance(value, dict):
        return {k: render_obj(v, ctx) for k, v in value.items()}
    if isinstance(value, list):
        return [render_obj(v, ctx) for v in value]
    return value


def _truncate(text: str, limit: int = _TEXT_LIMIT) -> str:
    text = str(text or "")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _last_output(ctx: dict) -> str:
    produced = ctx.get("nodes") or {}
    for value in reversed(list(produced.values())):
        out = str(value.get("output") or "")
        if out.strip():
            return out
    return ""


class Executor:
    """一个 Workflow 的执行实例。构造时即排好序，图不合法当场抛。"""

    def __init__(
        self,
        definition: WorkflowDefinition,
        *,
        tools: dict[str, Callable] | None = None,
        llm: Any = None,
    ) -> None:
        self.definition = definition
        self._tools: dict[str, Callable] = dict(tools or {})
        # llm 不在这里构造：读配置需要密钥，而"造一个 Agent 对象"
        # 不该因为当前机器没有 key 就失败。延迟到 _get_llm。
        self._llm = llm
        # 构造即排序：让"图有环"在跑之前暴露，而不是跑到一半才发现。
        self._order = topo_sort(definition)

    # -- 对外 -----------------------------------------------------

    def tool_names(self) -> list[str]:
        return sorted(self._tools)

    def run(self, trace: Trace, *, turn: int, user_input: str) -> str:
        """执行一轮，事件写进 trace，返回最终答案。

        只沿拓扑序前进，遇到 end 就停：DAG 里可能存在与本次无关的分支，
        不提前停会把不该跑的节点也跑掉，事件也就跟着虚增。
        """
        ctx: dict = {"input": str(user_input or ""), "nodes": {}}
        answer = ""
        last_output = ""

        for node in self._order:
            started = now_ms()
            trace.add_event(
                KIND_NODE_START,
                node_id=node.id,
                node_type=node.type,
                turn=turn,
                args={"input": _truncate(ctx["input"])},
            )
            try:
                output = self._exec(node, trace, turn, ctx)
            except Exception as exc:  # 任何节点实现的问题都在这里收口
                message = f"{type(exc).__name__}: {exc}"
                trace.add_event(
                    KIND_ERROR,
                    node_id=node.id,
                    node_type=node.type,
                    turn=turn,
                    ok=False,
                    error=message,
                    elapsed_ms=max(0, now_ms() - started),
                )
                raise ExecutionError(node.id, node.type, message) from exc

            elapsed = max(0, now_ms() - started)
            ctx["nodes"][node.id] = {"output": output}
            last_output = output
            trace.add_event(
                KIND_NODE_END,
                node_id=node.id,
                node_type=node.type,
                turn=turn,
                output=_truncate(output),
                elapsed_ms=elapsed,
            )
            if node.type == NODE_END:
                answer = output
                break

        return answer if answer else last_output

    # -- 节点实现 -------------------------------------------------

    def _exec(self, node: NodeDef, trace: Trace, turn: int, ctx: dict) -> str:
        if node.type == NODE_START:
            return ctx["input"]
        if node.type == NODE_LLM:
            return self._run_llm(node, trace, turn, ctx)
        if node.type == NODE_TOOL:
            return self._run_tool(node, trace, turn, ctx)
        if node.type == NODE_END:
            template = node.config.get("answer")
            return render(str(template), ctx) if template else _last_output(ctx)
        raise ValueError(f"未知节点类型 {node.type!r}")

    def _run_tool(self, node: NodeDef, trace: Trace, turn: int, ctx: dict) -> str:
        name = str(node.config.get("tool") or "")
        fn = self._tools.get(name)
        if fn is None:
            raise ValueError(f"工具 {name!r} 未注册，当前可用：{self.tool_names()}")
        args = render_obj(node.config.get("args") or {}, ctx)
        if not isinstance(args, dict):
            raise ValueError("config.args 渲染后必须是对象")

        started = now_ms()
        try:
            output = fn.invoke(args) if hasattr(fn, "invoke") else fn(**args)
        except Exception as exc:
            # 工具失败也要留一条事件：否则轨迹里看不出"调过但挂了"，
            # 只剩下一个空节点，事后无法判断是没调还是调崩了。
            trace.add_event(
                KIND_TOOL_CALL,
                node_id=node.id,
                node_type=NODE_TOOL,
                turn=turn,
                name=name,
                args=args,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                elapsed_ms=max(0, now_ms() - started),
            )
            raise

        text = str(output if output is not None else "")
        trace.add_event(
            KIND_TOOL_CALL,
            node_id=node.id,
            node_type=NODE_TOOL,
            turn=turn,
            name=name,
            args=args,
            output=_truncate(text),
            elapsed_ms=max(0, now_ms() - started),
        )
        return text

    def _run_llm(self, node: NodeDef, trace: Trace, turn: int, ctx: dict) -> str:
        prompt = render(str(node.config.get("prompt") or ""), ctx)
        llm = self._get_llm()
        model = str(
            node.config.get("model")
            or getattr(llm, "model_name", "")
            or getattr(llm, "model", "")
            or "llm"
        )

        started = now_ms()
        response = llm.invoke(prompt)
        text = str(getattr(response, "content", response) or "")
        trace.add_event(
            KIND_LLM_CALL,
            node_id=node.id,
            node_type=NODE_LLM,
            turn=turn,
            name=model,
            args={"prompt": _truncate(prompt)},
            output=_truncate(text),
            elapsed_ms=max(0, now_ms() - started),
        )
        return text

    def _get_llm(self):
        if self._llm is None:
            from config import build_llm, load_settings

            # 与 shop / customer 保持同一组参数：温度 0，同一道题两次结果一致。
            self._llm = build_llm(load_settings(), streaming=False).bind(temperature=0)
        return self._llm
