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
    KIND_BRANCH,
    KIND_ERROR,
    KIND_LLM_CALL,
    KIND_MEMORY_CALL,
    KIND_NODE_END,
    KIND_NODE_START,
    KIND_SLOT_REUSE,
    KIND_TOOL_CALL,
    Trace,
    now_ms,
)

from agentplatform.definition import (
    CONDITION_OPS,
    NODE_CONDITION,
    NODE_END,
    NODE_LLM,
    NODE_MEMORY,
    NODE_START,
    NODE_TOOL,
    NodeDef,
    WorkflowDefinition,
)
from agentplatform.extractor import EXTRACT_FIELDS, ExtractError, extract_field
from agentplatform.topology import topo_sort

_TEMPLATE = re.compile(r"\{\{\s*([A-Za-z0-9_.\-]+)\s*\}\}")

# 节点输入/输出进事件时截断，避免整段 prompt 把 jsonl 撑大。
_TEXT_LIMIT = 400


def _scan_extract_fields(definition: WorkflowDefinition) -> tuple[str, ...]:
    """定义里出现过哪些 `{{extract.X}}`，按首次出现顺序返回。

    只扫 config 不扫别的：模板只会写在 config 里。
    扫描结果用于"每轮开头把用得到的字段解析一遍"，
    所以它必须覆盖全部节点，而不只是本轮会被激活的那些——
    漏一个字段，那一轮就会退化成渲染期才报错。
    """
    found: list[str] = []

    def walk(value):
        if isinstance(value, str):
            for key in _TEMPLATE.findall(value):
                if key.startswith("extract."):
                    field = key.split(".", 1)[1]
                    if field not in found:
                        found.append(field)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    for node in definition.nodes:
        walk(node.config)
    return tuple(found)


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
    # extract.<字段>：本轮开头已经解析过（_resolve_slots），这里只取值。
    # 取不到说明"本轮没抽到、也没有历史值"，退回 extract_field 抛原来的错——
    # 不返回空串、也不猜默认值：猜出来的单号会让工具去查一张不存在的订单，
    # 最后被记成"模型答错了"。
    # 只从用户输入抽、不从节点输出抽——节点输出是工具原始报文，
    # 要抽就得先约定格式，那是下一版的事。
    if len(parts) == 2 and parts[0] == "extract":
        field = parts[1]
        slot = (ctx.get("slots") or {}).get(field)
        if slot is not None:
            return slot["value"]
        return extract_field(field, str(ctx.get("input", "")))
    raise TemplateError(
        f"未知模板变量 {key!r}，可用：input、nodes.<节点id>.output、"
        f"extract.<字段>（{'/'.join(EXTRACT_FIELDS)}）"
    )


def render(text: str, ctx: dict) -> str:
    """把 {{input}} / {{nodes.<id>.output}} / {{extract.<字段>}} 替换成实际值。

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


def _as_number(text: str) -> float | None:
    try:
        return float(str(text).strip())
    except (TypeError, ValueError):
        return None


def compare(left, op: str, right=None) -> bool:
    """condition 的判定，只做单条比较。

    数值优先：两边都能转成数字时按数值比，"1002" 与 1002 视为相等；
    任一边不是数字就退回字符串比较（去首尾空白）。
    大小比较（gt/lt/gte/lte）遇到非数字一律判否——不猜，"9" > "10" 这种
    字符串序对比数字序的错，比判否更难发现。
    """
    a = str(left if left is not None else "")
    b = str(right if right is not None else "")

    if op == "empty":
        return not a.strip()
    if op == "not_empty":
        return bool(a.strip())
    if op == "contains":
        return b in a
    if op == "not_contains":
        return b not in a
    if op == "regex":
        return re.search(b, a) is not None
    if op in ("eq", "ne"):
        x, y = _as_number(a), _as_number(b)
        same = (x == y) if (x is not None and y is not None) else (a.strip() == b.strip())
        return same if op == "eq" else not same
    if op in ("gt", "lt", "gte", "lte"):
        x, y = _as_number(a), _as_number(b)
        if x is None or y is None:
            return False
        return {"gt": x > y, "lt": x < y, "gte": x >= y, "lte": x <= y}[op]
    raise ValueError(f"未知条件运算符 {op!r}，可选 {list(CONDITION_OPS)}")


def _last_output(ctx: dict) -> str:
    produced = ctx.get("nodes") or {}
    for value in reversed(list(produced.values())):
        out = str(value.get("output") or "")
        if out.strip():
            return out
    return ""


def _render_experience(hits, *, top_k: int, max_chars: int) -> str:
    """把命中经验渲染成可拼进 prompt 的文本。没有命中就返回空串。

    空串是关键：空库时下游 prompt 里 {{nodes.recall.output}} 渲染成空，
    不该出现"【历史经验】"表头，否则等于往提示词里塞了一段空承诺。
    延迟 import，不用 memory 节点的 Workflow 不必加载整个 memory 包。
    """
    if not hits:
        return ""
    from memory.inject import render as render_experience

    return render_experience(hits, max_items=top_k, max_chars=max_chars)


class Executor:
    """一个 Workflow 的执行实例。构造时即排好序，图不合法当场抛。"""

    def __init__(
        self,
        definition: WorkflowDefinition,
        *,
        tools: dict[str, Callable] | None = None,
        llm: Any = None,
        memory_store: Any = None,
    ) -> None:
        self.definition = definition
        self._tools: dict[str, Callable] = dict(tools or {})
        # llm 不在这里构造：读配置需要密钥，而"造一个 Agent 对象"
        # 不该因为当前机器没有 key 就失败。延迟到 _get_llm。
        self._llm = llm
        # 经验库由调用方注入。刻意不设"默认路径"：
        # 一旦默认去读某个全局文件，A 的实验就可能吃到 B 留下的经验，
        # 结果对了也说不清是谁的功劳。没注入就是空，宁可不注入。
        self._memory_store = memory_store
        # 构造即排序：让"图有环"在跑之前暴露，而不是跑到一半才发现。
        self._order = topo_sort(definition)
        # 定义里出现过的抽取字段，每轮开头统一解析一遍。
        self._extract_refs = _scan_extract_fields(definition)

    # -- 对外 -----------------------------------------------------

    def tool_names(self) -> list[str]:
        return sorted(self._tools)

    def new_context(self) -> dict:
        """一条 Task 共享的执行上下文。

        五个键各有归属，读代码时不用猜：
        - input       当前轮用户原话，每轮覆盖
        - nodes       仅当前轮已执行节点的产出，每轮清空
        - history     已结束轮次的快照，按发生顺序追加，只增不改
        - experience  Runtime 递进来的经验文本，整条任务共用
        - slots       已解析的抽取字段 -> {"value", "turn"}，跨轮沿用

        history 存的是拷一份的值，不是引用：后面某轮再往 ctx["nodes"] 里写，
        不能把已归档的那一轮改掉——那等于"历史被当前轮改写"，
        事后对比两轮产出时会完全对不上。

        slots 与 history 分开：history 是"跑过什么的记录"，只读；
        slots 是"当前有效的参数值"，会被后续轮次覆盖。
        """
        return {
            "input": "",
            "nodes": {},
            "history": [],
            "experience": "",
            "slots": {},
        }

    def run(
        self,
        trace: Trace,
        *,
        turn: int,
        user_input: str,
        experience: str = "",
        ctx: dict | None = None,
    ) -> str:
        """执行一轮，事件写进 trace，返回最终答案。

        ctx 由调用方按 Task 建一次、逐轮传进来（见 new_context）：多轮之间
        共享历史，但当前轮只看得到本轮的产出。传 None 就现建一个，
        单轮调用（测试、脚本直调）因此与从前逐字一致。

        experience 是 Runtime 递进来的经验文本（已由 memory.inject 渲染好），
        只有显式开了 config.inject_experience 的 llm 节点才会用它。
        默认空串＝不注入，基线组因此与"没有 runtime 经验"的老行为逐字一致。

        从 entry 出发，只跑被激活的节点：普通节点把 next 全部激活；
        condition 只激活命中的那一支，另一支整体跳过，连 node_start 都不发——
        "没被选中的分支留下半条轨迹"会让人以为它跑过。

        汇聚用 OR：任一入边激活就执行，不要求所有入边都到（不做 AND 汇聚）。
        第一版不做并行：两条边同时激活时谁先谁后没有确定答案，
        事件顺序不稳定，两次轨迹就无法逐条对比。

        遇到 end 就停：不提前停会把与本次无关的分支也跑掉，事件跟着虚增。
        """
        if ctx is None:
            ctx = self.new_context()
        # 每轮只留本轮的东西：input 覆盖成本轮原话，nodes 清空后只装本轮产出。
        # 不清空的话，第 2 轮的 {{nodes.<id>.output}} 会读到第 1 轮的值——
        # 看起来跑通了，其实是拿旧数据答新问题，比直接报错难查得多。
        ctx["input"] = str(user_input or "")
        ctx["nodes"] = {}
        ctx["experience"] = str(experience or "")
        ctx.setdefault("slots", {})
        # 槽位在本轮任何节点执行前定下来：事件顺序才是"先定参数、再跑节点"。
        self._resolve_slots(trace, turn, ctx)
        answer = ""
        last_output = ""
        active: set[str] = {self.definition.entry} if self.definition.entry else set()

        for node in self._order:
            if node.id not in active:
                continue

            started = now_ms()
            trace.add_event(
                KIND_NODE_START,
                node_id=node.id,
                node_type=node.type,
                turn=turn,
                args={"input": _truncate(ctx["input"])},
                started_at=started,
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
                    started_at=started,
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
                started_at=started,
                elapsed_ms=elapsed,
            )
            active.update(self._next_of(node, output))
            if node.type == NODE_END:
                answer = output
                break

        # 本轮跑完才归档。中途抛错的那一轮不归档——半轮的产出混进 history，
        # 下一轮读到它就会把"跑了一半"当成"跑完了"。
        ctx["history"].append(
            {
                "turn": turn,
                "input": ctx["input"],
                "nodes": {nid: dict(value) for nid, value in ctx["nodes"].items()},
            }
        )
        return answer if answer else last_output

    def _resolve_slots(self, trace: Trace, turn: int, ctx: dict) -> None:
        """把本轮用得上的抽取字段定下来：先抽本轮输入，抽不到才沿用旧值。

        放在每轮开头只做一次，而不是在渲染时懒解析：
        1. 复用必须留一条事件证据；懒解析会在每个引用点各留一条，
           "同一轮沿用了两次"看起来就像两个不同来源；
        2. 一轮内多次引用同一字段必须拿到同一个值，集中解析天然成立。

        抽不到、也没有旧值时什么都不做：留给渲染期抛 ExtractError。
        这样"没有历史值"的报错位置与文案跟没有槽位这层时完全一致，
        不会因为加了槽位就把错误从"缺参数"变成"缺参数但静默用了空值"。
        """
        for field in self._extract_refs:
            try:
                value = extract_field(field, str(ctx["input"]))
            except ExtractError:
                slot = ctx["slots"].get(field)
                if slot is None:
                    continue
                # 沿用旧值时不改 slot["turn"]：from_turn 要指向值真正产生的轮次，
                # 否则"第 3 轮沿用了第 1 轮"会被写成"沿用了第 3 轮"，证据就假了。
                trace.add_event(
                    KIND_SLOT_REUSE,
                    turn=turn,
                    name=field,
                    args={
                        "field": field,
                        "value": str(slot["value"]),
                        "source": "previous_turn",
                        "from_turn": int(slot["turn"]),
                        "reused_from_previous_turn": True,
                    },
                    output=str(slot["value"]),
                )
                continue
            ctx["slots"][field] = {"value": value, "turn": turn}

    def _next_of(self, node: NodeDef, output: str) -> list[str]:
        """这个节点执行完之后，哪些节点被激活。

        condition 按判定结果只取一支；其余节点走 next（无条件）。
        """
        if node.type != NODE_CONDITION:
            return list(node.next)
        key = "true" if str(output).strip().lower() == "true" else "false"
        seq = (node.config.get("branches") or {}).get(key) or []
        return [str(x) for x in seq if str(x)]

    # -- 节点实现 -------------------------------------------------

    def _exec(self, node: NodeDef, trace: Trace, turn: int, ctx: dict) -> str:
        if node.type == NODE_START:
            return ctx["input"]
        if node.type == NODE_LLM:
            return self._run_llm(node, trace, turn, ctx)
        if node.type == NODE_TOOL:
            return self._run_tool(node, trace, turn, ctx)
        if node.type == NODE_CONDITION:
            return self._run_condition(node, trace, turn, ctx)
        if node.type == NODE_MEMORY:
            return self._run_memory(node, trace, turn, ctx)
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
                started_at=started,
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
            started_at=started,
            elapsed_ms=max(0, now_ms() - started),
        )
        return text

    def _run_llm(self, node: NodeDef, trace: Trace, turn: int, ctx: dict) -> str:
        prompt = render(str(node.config.get("prompt") or ""), ctx)
        # 经验只在显式开启的节点上拼，且原样拼在 prompt 之前：
        # memory.inject 已经把表头、边界声明、尾注都拼好了，这里不再包一层，
        # 否则原生 Agent 与 Workflow 的注入形态就不一致，报告里没法对比。
        # 空经验完全不拼——基线组不能出现一个空的"【历史经验】"表头，
        # 那是往提示词里塞了一句空承诺。
        experience = str(ctx.get("experience") or "")
        injected = 0
        if node.config.get("inject_experience") and experience.strip():
            prompt = f"{experience}\n\n{prompt}"
            injected = len(experience)

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
        args: dict = {"prompt": _truncate(prompt)}
        if injected:
            # 只在真的注入了才加这个键：老轨迹的 llm_call.args 保持原样。
            args["experience_chars"] = injected
        trace.add_event(
            KIND_LLM_CALL,
            node_id=node.id,
            node_type=NODE_LLM,
            turn=turn,
            name=model,
            args=args,
            output=_truncate(text),
            started_at=started,
            elapsed_ms=max(0, now_ms() - started),
        )
        return text

    def _run_condition(self, node: NodeDef, trace: Trace, turn: int, ctx: dict) -> str:
        """判定并返回 "true"/"false"。

        branch 事件单独记一条：轨迹里要能看出"这一步选了哪支"，
        否则只能从后续节点反推，一旦两支的下游节点相同就完全看不出来。
        """
        field = render(str(node.config.get("field") or ""), ctx)
        op = str(node.config.get("op") or "")
        value = node.config.get("value")
        if isinstance(value, str):
            value = render(value, ctx)

        result = compare(field, op, value)
        key = "true" if result else "false"
        trace.add_event(
            KIND_BRANCH,
            node_id=node.id,
            node_type=NODE_CONDITION,
            turn=turn,
            name=op,
            args={
                "field": _truncate(field),
                "value": _truncate(value),
                "result": key,
            },
            output=key,
            started_at=now_ms(),
        )
        return key

    def _run_memory(self, node: NodeDef, trace: Trace, turn: int, ctx: dict) -> str:
        """检索经验并渲染成可注入的文本。

        事件里必须留下命中数量、命中 ID、耗时、注入内容摘要：
        以后判断"这条经验有没有用"，靠的就是这几个字段，不是感觉。
        """
        cfg = node.config
        query = render(str(cfg.get("query") or ""), ctx)
        top_k = int(cfg.get("top_k") or 3)
        max_chars = int(cfg.get("max_chars") or 1200)
        min_score = cfg.get("min_score")

        started = now_ms()
        hits = self._recall(query, top_k=top_k, min_score=min_score)
        text = _render_experience(hits, top_k=top_k, max_chars=max_chars)
        ids = [h.exp.exp_id for h in hits]
        trace.add_event(
            KIND_MEMORY_CALL,
            node_id=node.id,
            node_type=NODE_MEMORY,
            turn=turn,
            name="memory",
            args={
                "query": _truncate(query),
                "top_k": top_k,
                "hit_count": len(hits),
                "hits": ids,
                "injected_chars": len(text),
            },
            output=_truncate(text),
            started_at=started,
            elapsed_ms=max(0, now_ms() - started),
        )
        return text

    def _recall(self, query: str, *, top_k: int, min_score):
        """从注入的经验库里检索。没注入或 query 为空时返回空列表。"""
        if self._memory_store is None or not str(query or "").strip():
            return []
        from core.task import Task
        from memory.retrieve import retrieve_from_store

        task = Task(id=f"wf:{self.definition.workflow_id}", input=str(query))
        kwargs = {}
        if min_score is not None:
            kwargs["min_score"] = float(min_score)
        return retrieve_from_store(self._memory_store, task, top_k=top_k, **kwargs)

    def _get_llm(self):
        if self._llm is None:
            from config import build_llm, load_settings

            # 与 shop / customer 保持同一组参数：温度 0，同一道题两次结果一致。
            self._llm = build_llm(load_settings(), streaming=False).bind(temperature=0)
        return self._llm
