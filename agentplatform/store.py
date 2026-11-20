"""Agent 定义的持久化：一个 id 一个 JSON 文件。

为什么不用数据库：
    现在的规模是一个人几个 Agent。目录 + 文件能直接 diff、能进 git、
    坏了能手改一行；引入 SQLite 只会多一层没人验证的中间件，
    还多出一个"库里和磁盘上哪份才是真的"的问题。
    等到出现"上百个 Agent、要按条件检索"再谈。

版本策略（说清楚，不假装有版本树）：
    覆盖保存时 version+1，旧内容不保留。真正的版本历史交给 git——
    一个提交就是一个版本，`git log -p data/workflows/demo.json` 能看全改动。
    自己再造一套版本表只会比 git 更难用，而且没人会去维护。

校验到哪一步：
    保存时只做结构与图校验（类型、连线、环、可达）。
    工具是否存在不在这里查——它取决于运行时注入的工具池，
    存的时候还不知道将来由谁来跑。执行前由 WorkflowAgent 构造时再查。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agentplatform.definition import WORKFLOW_DIR, DefinitionError, WorkflowDefinition
from agentplatform.schema import validate
from agentplatform.topology import check as check_topology

# 测试会替换这个值来指向临时目录，所以 AgentStore 里必须在调用时读它，
# 不能在模块导入时就绑死成默认参数。
DEFAULT_ROOT = WORKFLOW_DIR


@dataclass
class AgentSummary:
    """列表页需要的最小信息。坏文件也能出现在列表里，只是带 error。"""

    workflow_id: str
    name: str = ""
    version: int = 0
    description: str = ""
    nodes: int = 0
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "version": int(self.version),
            "description": self.description,
            "nodes": int(self.nodes),
            "error": self.error,
        }


class AgentStore:
    def __init__(self, root=None) -> None:
        self.root = Path(root) if root else Path(DEFAULT_ROOT)

    # -- 查询 -----------------------------------------------------

    def path_of(self, workflow_id: str) -> Path:
        return self.root / f"{workflow_id}.json"

    def ids(self) -> list[str]:
        """磁盘上现有的 id。目录不存在就当空，不让调用方（--list）挂掉。"""
        try:
            return sorted(p.stem for p in self.root.glob("*.json"))
        except OSError:
            return []

    def exists(self, workflow_id: str) -> bool:
        return self.path_of(workflow_id).is_file()

    def list_agents(self) -> list[AgentSummary]:
        out: list[AgentSummary] = []
        for wid in self.ids():
            try:
                defn = self.load(wid)
            except DefinitionError as exc:
                # 一个坏文件不该让整个列表变空：那样用户会以为"什么都没存"。
                out.append(AgentSummary(workflow_id=wid, error=str(exc)))
                continue
            out.append(
                AgentSummary(
                    # 文件名与内部 id 不一致时以文件名为准来定位：
                    # 否则列表里点进去会打开另一个文件。
                    workflow_id=wid,
                    name=defn.name,
                    version=defn.version,
                    description=defn.description,
                    nodes=len(defn.nodes),
                )
            )
        return out

    def load(self, workflow_id: str) -> WorkflowDefinition:
        return WorkflowDefinition.load(self.path_of(workflow_id))

    # -- 写入 -----------------------------------------------------

    def save(self, definition: WorkflowDefinition, *, bump: bool = True) -> Path:
        """校验后落盘。已存在且 bump 时版本号自动 +1。"""
        errors = validate(definition) + check_topology(definition)
        if errors:
            raise DefinitionError(
                "Workflow 定义不合法：\n" + "\n".join(f"  - {e}" for e in errors)
            )
        path = self.path_of(definition.workflow_id)
        if bump and path.is_file():
            try:
                definition.version = self.load(definition.workflow_id).version + 1
            except DefinitionError:
                # 旧文件坏了：从给定版本继续，不要因为读不出来就卡住保存。
                definition.version = max(1, int(definition.version or 1))
        return definition.save(path)


__all__ = ["AgentStore", "AgentSummary", "DEFAULT_ROOT"]
