"""模型层：只负责把 llm 造出来。

参数与 tests/replay.py 保持同一组（streaming=False、temperature=0），
这不是随手抄的：温度不为 0 时同一道题两次运行结果不同，
与历史报告的对拍就失去意义，也说不清"这次没通过"是能力波动还是真的退步。
"""

from __future__ import annotations

from config import build_llm, load_settings


def build_shop_llm(settings=None):
    """构造被测 Agent 用的模型客户端。

    没有显式传 settings 时才去读环境变量：
    load_settings() 在缺 API Key 时会直接抛错，
    所以调用方（agent.py）把这个动作推迟到真正要跑的时候，
    保证 import 本模块、构造 Agent 对象都不会因为缺密钥而失败。
    """
    cfg = settings or load_settings()
    return build_llm(cfg, streaming=False).bind(temperature=0)
