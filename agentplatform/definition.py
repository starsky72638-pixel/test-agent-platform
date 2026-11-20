"""Workflow 定义：一份 JSON -> 一个对象。

这一层只描述"有哪些节点、谁指向谁、每个节点配了什么"，不含执行逻辑，
也不认识任何业务域。校验在 schema/topology，执行在 executor，三者互不依赖。

为什么不存 edges：
    节点上的 next 已经是完整的边信息。同时存 edges 就有两份真相，
    改一处忘另一处时，校验通过但实际跑的是另一条路。可视化编辑器
    生成 JSON 时直接把连线写成 next 即可。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

NODE_START = "start"
NODE_LLM = "llm"
NODE_TOOL = "tool"
NODE_CONDITION = "condition"
NODE_MEMORY = "memory"
NODE_END = "end"

NODE_TYPES = (NODE_START, NODE_LLM, NODE_TOOL, NODE_CONDITION, NODE_MEMORY, NODE_END)

# condition 能用的比较运算符。第一版只做单条比较：
# 不做 and/or 组合、不做正则替换、不做表达式求值——
# 需要组合就串两个 condition 节点，这样每一步都能被 schema 逐项校验，
# 也不会为了"灵活"引入一个 eval 后门。
CONDITION_OPS = (
    "eq",
    "ne",
    "contains",
    "not_contains",
    "gt",
    "lt",
    "gte",
    "lte",
    "empty",
    "not_empty",
    "regex",
)

# 第一版只支持互斥的 true/false 两支。并行与 AND 汇聚都不做：
# 两条边同时激活时谁是终态、事件怎么排序，都没有确定答案。
BRANCH_KEYS = ("true", "false")

WORKFLOW_DIR = Path(__file__).resolve().parents[1] / "data" / "workflows"


class DefinitionError(ValueError):
    """定义本身不合法：JSON 坏了、字段类型不对、文件不存在。"""


def _branch_targets(node_id: str, config: dict, declared: list[str]) -> list[str]:
    """condition 的出边：branches 是权威，next 只是它的展开。

    两份信息不一致时必须报错，不能挑一个用：校验按 next 认为"能到"、
    执行按 branches 走不到，轨迹里就会有个节点凭空消失，事后无从解释。
    只写 branches 不写 next 是允许的，这里补齐。
    """
    raw = config.get("branches")
    if raw is None:
        return declared
    if not isinstance(raw, dict):
        raise DefinitionError(f"节点 {node_id} 的 config.branches 必须是对象")
    unknown = sorted(set(raw) - set(BRANCH_KEYS))
    if unknown:
        raise DefinitionError(
            f"节点 {node_id} 的 branches 只支持 {list(BRANCH_KEYS)}，收到 {unknown}"
        )

    targets: list[str] = []
    for key in BRANCH_KEYS:
        seq = raw.get(key)
        if seq is None:
            continue
        if not isinstance(seq, list):
            raise DefinitionError(f"节点 {node_id} 的 branches.{key} 必须是数组")
        for item in seq:
            target = str(item)
            if target and target not in targets:
                targets.append(target)

    if declared and set(declared) != set(targets):
        raise DefinitionError(
            f"节点 {node_id} 的 next 与 branches 不一致："
            f"next={declared}，branches 并集={targets}"
        )
    return targets


@dataclass
class NodeDef:
    """一个节点。config 内容按 type 解释，这一层不解析。"""

    id: str
    type: str
    config: dict = field(default_factory=dict)
    next: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "config": dict(self.config or {}),
            "next": list(self.next),
        }

    @classmethod
    def from_dict(cls, raw) -> "NodeDef":
        if not isinstance(raw, dict):
            raise DefinitionError(f"节点必须是对象，收到 {type(raw).__name__}")
        node_id = str(raw.get("id") or "").strip()
        node_type = str(raw.get("type") or "").strip()
        config = raw.get("config") or {}
        if not isinstance(config, dict):
            raise DefinitionError(f"节点 {node_id or '?'} 的 config 必须是对象")
        next_ids = raw.get("next") or []
        if not isinstance(next_ids, list):
            raise DefinitionError(f"节点 {node_id or '?'} 的 next 必须是数组")
        targets = [str(x) for x in next_ids]
        if node_type == NODE_CONDITION:
            targets = _branch_targets(node_id, config, targets)
        return cls(
            id=node_id,
            type=node_type,
            config=dict(config),
            next=targets,
        )


@dataclass
class WorkflowDefinition:
    workflow_id: str
    nodes: list[NodeDef] = field(default_factory=list)
    entry: str = ""
    version: int = 1
    name: str = ""
    description: str = ""
    tools: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def node(self, node_id: str) -> NodeDef | None:
        for item in self.nodes:
            if item.id == node_id:
                return item
        return None

    def node_ids(self) -> list[str]:
        return [n.id for n in self.nodes]

    def to_dict(self) -> dict:
        return {
            "workflow_id": self.workflow_id,
            "version": int(self.version),
            "name": self.name,
            "description": self.description,
            "entry": self.entry,
            "tools": list(self.tools),
            "nodes": [n.to_dict() for n in self.nodes],
            "meta": dict(self.meta or {}),
        }

    @classmethod
    def from_dict(cls, raw) -> "WorkflowDefinition":
        if not isinstance(raw, dict):
            raise DefinitionError(f"定义必须是对象，收到 {type(raw).__name__}")
        nodes_raw = raw.get("nodes") or []
        if not isinstance(nodes_raw, list):
            raise DefinitionError("nodes 必须是数组")
        try:
            version = int(raw.get("version") or 1)
        except (TypeError, ValueError) as exc:
            raise DefinitionError(f"version 必须是整数，收到 {raw.get('version')!r}") from exc
        return cls(
            # agent_id 是早期草案里的字段名，一并接受，避免旧文件直接失效。
            workflow_id=str(raw.get("workflow_id") or raw.get("agent_id") or "").strip(),
            nodes=[NodeDef.from_dict(x) for x in nodes_raw],
            entry=str(raw.get("entry") or "").strip(),
            version=version,
            name=str(raw.get("name") or ""),
            description=str(raw.get("description") or ""),
            tools=[str(t) for t in (raw.get("tools") or [])],
            meta=dict(raw.get("meta") or {}),
        )

    @classmethod
    def load(cls, path) -> "WorkflowDefinition":
        p = Path(path)
        try:
            text = p.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise DefinitionError(f"找不到 Workflow 定义：{p}") from exc
        except OSError as exc:
            raise DefinitionError(f"读取 {p} 失败：{exc}") from exc
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise DefinitionError(f"{p.name} 不是合法 JSON：{exc}") from exc
        return cls.from_dict(raw)

    def save(self, path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return p
