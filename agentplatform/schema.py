"""结构校验：只回答"这份定义能不能跑"，不执行、不读盘、不联网。

为什么返回错误列表而不是抛第一个异常：
    用户改一处、跑一次、再报下一个，来回十遍才能跑通。
    一次报全，改一轮就够。

为什么和 topology 分开：
    这一层看字段（类型对不对、参数全不全），那一层看图（有没有环、可不可达）。
    混在一起时，报错顺序会随节点顺序变化，测试也就钉不住。
"""

from __future__ import annotations

from agentplatform.definition import (
    NODE_END,
    NODE_LLM,
    NODE_START,
    NODE_TOOL,
    NODE_TYPES,
    WorkflowDefinition,
)


def validate(
    defn: WorkflowDefinition,
    *,
    available_tools: set[str] | None = None,
) -> list[str]:
    """返回全部问题；空列表表示通过。

    available_tools 为 None 时跳过工具存在性检查——那是"只验结构"的用法，
    真正执行前必须传工具池，否则会在跑到一半时才发现工具不存在。
    """
    errors: list[str] = []

    if not str(defn.workflow_id or "").strip():
        errors.append("workflow_id 不能为空")

    if not defn.nodes:
        errors.append("nodes 不能为空")
        return errors

    seen: set[str] = set()
    for node in defn.nodes:
        if not node.id:
            errors.append("存在 id 为空的节点")
            continue
        if node.id in seen:
            errors.append(f"节点 id 重复：{node.id}")
        seen.add(node.id)

    for node in defn.nodes:
        if not node.id:
            continue
        if node.type not in NODE_TYPES:
            errors.append(
                f"节点 {node.id} 的类型 {node.type!r} 不支持，可选 {NODE_TYPES}"
            )
        for target in node.next:
            if target not in seen:
                errors.append(f"节点 {node.id} 的 next 指向不存在的节点 {target!r}")

        if node.type == NODE_LLM:
            prompt = node.config.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                errors.append(f"节点 {node.id} 是 llm，必须配置非空 config.prompt")

        if node.type == NODE_TOOL:
            tool = node.config.get("tool")
            if not isinstance(tool, str) or not tool.strip():
                errors.append(f"节点 {node.id} 是 tool，必须配置 config.tool")
            elif available_tools is not None and tool not in available_tools:
                errors.append(
                    f"节点 {node.id} 引用了未注册的工具 {tool!r}，"
                    f"当前可用：{sorted(available_tools)}"
                )
            args = node.config.get("args")
            if args is not None and not isinstance(args, dict):
                errors.append(f"节点 {node.id} 的 config.args 必须是对象")

    starts = [n for n in defn.nodes if n.type == NODE_START]
    ends = [n for n in defn.nodes if n.type == NODE_END]
    if len(starts) != 1:
        errors.append(f"必须恰好有一个 start 节点，当前 {len(starts)} 个")
    if not ends:
        errors.append("至少需要一个 end 节点")

    if not defn.entry:
        errors.append("entry 不能为空")
    else:
        entry = defn.node(defn.entry)
        if entry is None:
            errors.append(f"entry 指向不存在的节点 {defn.entry!r}")
        elif entry.type != NODE_START:
            errors.append(f"entry {defn.entry!r} 的类型是 {entry.type}，必须是 start")

    # 声明了 tools 就必须真的用得上：声明了却没人用，多半是节点写错了工具名。
    if available_tools is not None and defn.tools:
        used = {
            str(n.config.get("tool"))
            for n in defn.nodes
            if n.type == NODE_TOOL and n.config.get("tool")
        }
        for name in defn.tools:
            if name not in used:
                errors.append(f"tools 里声明了 {name!r}，但没有任何 tool 节点使用它")

    return errors
