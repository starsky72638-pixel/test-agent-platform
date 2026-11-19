"""核心层：循环本体与 Runtime 适配。

只回答"一次任务怎么跑完、怎么记成轨迹"。
不含提示词文本、不含工具定义、不含模型参数——换任何一层都不用动这里。
"""

from agents.shop.core.agent import ShopAgent, calls_from_messages, compose_view, run_loop

__all__ = ["ShopAgent", "run_loop", "calls_from_messages", "compose_view"]
