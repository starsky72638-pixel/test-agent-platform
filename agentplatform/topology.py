"""图结构：拓扑排序、环检测、可达性。

单独一层是因为它只依赖"节点 id 与边"，与节点类型、配置、业务都无关。
把图的问题和字段的问题分开报，用户才知道是改连线还是改参数。
"""

from __future__ import annotations

from agentplatform.definition import NodeDef, WorkflowDefinition


class TopologyError(ValueError):
    """图结构不合法，无法确定执行顺序。"""


def topo_sort(defn: WorkflowDefinition) -> list[NodeDef]:
    """Kahn 算法。返回执行顺序；有环抛 TopologyError。

    顺序稳定：入度为 0 的节点按定义里的出现顺序处理，
    同一个 Workflow 每次跑出来的节点顺序一致——否则两次轨迹无法逐条对比。
    """
    nodes = {n.id: n for n in defn.nodes if n.id}
    indeg = {nid: 0 for nid in nodes}
    for node in nodes.values():
        for target in node.next:
            if target in indeg:
                indeg[target] += 1

    ready = [nid for nid in nodes if indeg[nid] == 0]
    order: list[str] = []
    while ready:
        current = ready.pop(0)
        order.append(current)
        for target in nodes[current].next:
            if target not in indeg:
                continue
            indeg[target] -= 1
            if indeg[target] == 0:
                ready.append(target)

    if len(order) != len(nodes):
        stuck = sorted(nid for nid, deg in indeg.items() if deg > 0)
        raise TopologyError(f"存在环，无法确定执行顺序；涉及节点：{stuck}")
    return [nodes[nid] for nid in order]


def reachable(defn: WorkflowDefinition) -> set[str]:
    """从 entry 出发能走到的节点集合。"""
    if not defn.entry or defn.node(defn.entry) is None:
        return set()
    seen: set[str] = set()
    stack = [defn.entry]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        node = defn.node(current)
        if node is None:
            continue
        stack.extend(node.next)
    return seen


def unreachable(defn: WorkflowDefinition) -> list[str]:
    """从 entry 走不到的节点。

    这类节点不会执行，写在定义里只会让人以为它生效了。
    """
    if not defn.entry or defn.node(defn.entry) is None:
        return []
    hit = reachable(defn)
    return sorted(n.id for n in defn.nodes if n.id and n.id not in hit)


def check(defn: WorkflowDefinition) -> list[str]:
    """返回全部图结构问题；空列表表示通过。"""
    errors: list[str] = []
    for node_id in unreachable(defn):
        errors.append(f"节点 {node_id} 从 entry 不可达")
    try:
        topo_sort(defn)
    except TopologyError as exc:
        errors.append(str(exc))
    return errors
