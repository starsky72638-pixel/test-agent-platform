"""结构校验：只回答"这份定义能不能跑"，不执行、不读盘、不联网。

为什么返回错误列表而不是抛第一个异常：
    用户改一处、跑一次、再报下一个，来回十遍才能跑通。
    一次报全，改一轮就够。

为什么和 topology 分开：
    这一层看字段（类型对不对、参数全不全），那一层看图（有没有环、可不可达）。
    混在一起时，报错顺序会随节点顺序变化，测试也就钉不住。
"""

from __future__ import annotations

import re

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
    WorkflowDefinition,
)
from agentplatform.extractor import EXTRACT_FIELDS

# 与 executor 的模板语法保持一致：{{ 名字 }}，名字允许点号分段。
_TEMPLATE_REF = re.compile(r"\{\{\s*([A-Za-z0-9_.\-]+)\s*\}\}")


def _positive_int(value) -> bool:
    # bool 是 int 的子类，True 不该被当成 top_k=1。
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _iter_strings(value):
    """把 config 里所有字符串摊平，用于扫模板引用。"""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_strings(item)


def _check_extract_refs(node) -> list[str]:
    """`{{extract.X}}` 的 X 必须在白名单里。

    写在 from_dict 里拦不住（它不解析 config 内容），放到执行期拦又会
    跑到一半才炸。放在校验层，加载时就能一次报全。
    """
    errors: list[str] = []
    for text in _iter_strings(node.config):
        for ref in _TEMPLATE_REF.findall(text):
            if not ref.startswith("extract."):
                continue
            name = ref.split(".", 1)[1]
            if name not in EXTRACT_FIELDS:
                errors.append(
                    f"节点 {node.id} 引用了未知抽取字段 {name!r}，"
                    f"可用：{list(EXTRACT_FIELDS)}"
                )
    return errors


def _check_condition(node) -> list[str]:
    """condition 的配置问题。分支语义在第一版是互斥两支，不做并行/AND 汇聚。"""
    errors: list[str] = []
    cfg = node.config

    field = cfg.get("field")
    if not isinstance(field, str) or not field.strip():
        errors.append(f"节点 {node.id} 是 condition，必须配置非空 config.field")

    op = cfg.get("op")
    if op not in CONDITION_OPS:
        errors.append(
            f"节点 {node.id} 是 condition，config.op 必须是 {list(CONDITION_OPS)} 之一，"
            f"收到 {op!r}"
        )
    elif op not in ("empty", "not_empty") and "value" not in cfg:
        # empty/not_empty 只判空，强行要求 value 会让人写一个没用的字段。
        errors.append(f"节点 {node.id} 的 op={op} 需要 config.value")

    branches = cfg.get("branches")
    if not isinstance(branches, dict) or not branches:
        errors.append(f"节点 {node.id} 是 condition，必须配置 config.branches")
        return errors

    unknown = sorted(set(branches) - set(BRANCH_KEYS))
    if unknown:
        errors.append(
            f"节点 {node.id} 的 branches 只支持 {list(BRANCH_KEYS)}，收到 {unknown}"
        )
    union: list[str] = []
    for key in BRANCH_KEYS:
        if key not in branches:
            continue
        seq = branches[key]
        if not isinstance(seq, list):
            errors.append(f"节点 {node.id} 的 branches.{key} 必须是数组")
            continue
        for item in seq:
            target = str(item)
            if target and target not in union:
                union.append(target)

    # from_dict 已经拦过一次；这里再查，是为了拦住手工拼的 NodeDef。
    if node.next and set(node.next) != set(union):
        errors.append(
            f"节点 {node.id} 的 next 与 branches 不一致："
            f"next={list(node.next)}，branches 并集={union}"
        )
    return errors


def _check_memory(node) -> list[str]:
    errors: list[str] = []
    cfg = node.config

    query = cfg.get("query")
    if not isinstance(query, str) or not query.strip():
        errors.append(f"节点 {node.id} 是 memory，必须配置非空 config.query")

    for key in ("top_k", "max_chars"):
        if key in cfg and not _positive_int(cfg[key]):
            errors.append(
                f"节点 {node.id} 的 config.{key} 必须是正整数，收到 {cfg[key]!r}"
            )

    if "min_score" in cfg:
        val = cfg["min_score"]
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            errors.append(f"节点 {node.id} 的 config.min_score 必须是数字，收到 {val!r}")
        elif not 0 <= float(val) <= 1:
            errors.append(f"节点 {node.id} 的 config.min_score 应在 0~1，收到 {val!r}")
    return errors


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
            if "inject_experience" in node.config and not isinstance(
                node.config["inject_experience"], bool
            ):
                errors.append(
                    f"节点 {node.id} 的 config.inject_experience 必须是 true/false，"
                    f"收到 {node.config['inject_experience']!r}"
                )

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

        if node.type == NODE_CONDITION:
            errors.extend(_check_condition(node))

        if node.type == NODE_MEMORY:
            errors.extend(_check_memory(node))

        # 所有节点类型都可能写模板，统一扫一遍。
        errors.extend(_check_extract_refs(node))

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
