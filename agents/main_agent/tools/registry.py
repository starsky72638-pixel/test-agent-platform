from __future__ import annotations

from typing import Any, Callable, List

try:  # langchain-core 标准工具基类，缺失时降级为普通 callable
    from langchain_core.tools import BaseTool
except ImportError:  # pragma: no cover
    BaseTool = Any  # type: ignore[assignment,misc]


class ToolRegistry:
    """工具注册表。"""

    def __init__(self) -> None:
        self._tools: List[Any] = []

    def register(self, tool: Any) -> Any:
        """注册一个工具（支持装饰器用法）。"""
        self._tools.append(tool)
        return tool

    def list(self) -> List[Any]:
        return list(self._tools)

    def is_empty(self) -> bool:
        return not self._tools

    def bind(self, llm: Any) -> Any:
        """把工具绑定给模型；无工具时原样返回，不产生额外请求参数。"""
        if self.is_empty():
            return llm
        return llm.bind_tools(self._tools)

#全局工具注册表
_REGISTRY: ToolRegistry | None = None

def get_registry() -> ToolRegistry:
    """返回全局工具注册表（当前为空）。"""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = ToolRegistry()
    return _REGISTRY
