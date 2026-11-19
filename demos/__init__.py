"""接入示例包。

这里只放“怎么接一个 Agent”的示例，核心模块一律不 import 它。
删掉整个目录，core / evaluator / memory 仍应可 import 与测试。
"""

from demos.echo import EchoAgent

__all__ = ["EchoAgent"]
