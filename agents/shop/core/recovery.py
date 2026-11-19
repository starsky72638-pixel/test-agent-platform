"""工具返回的分类，以及"要不要重试"的判断。

没有这一层时，工具失败只是把一段错误字符串塞回对话：
模型看到 status=500 之后做什么全凭运气——有时原样再试一次，
有时换个工具瞎查，有时直接把"服务出错了"当成答案交上去。
这里把这件事变成显式判断，循环只负责按判断执行。

四类，只有第一类允许自动重试：

    retry    5xx / 429 / 连接超时  同参数再试一次可能就通了
    arg      400/401/403/409/422   同参数重试必然还错，只能让模型改参数
    missing  404                   对象不存在，重复查同一个标识没有意义
    ok       其它                  不干预，原样交给模型

重试上限由环境变量 SHOP_RETRY_LIMIT 控制，默认 1。
调成 0 就退化成改动前的行为（只试一次），
对照实验只切这一个变量，不需要回滚代码。
"""

from __future__ import annotations

import os
import re

# 只认行首的 status=NNN，工具层保证状态码单独成行。
_CODE = re.compile(r"^\s*status=(\d{3})")

RETRYABLE_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})
ARG_CODES = frozenset({400, 401, 403, 409, 422})

DEFAULT_RETRY_LIMIT = 1

# 给模型的说明。只陈述"这次调用的性质"，不替模型决定下一步做什么。
HINTS = {
    "arg": "这次调用被服务端按参数错误拒绝，同参数重试不会有不同结果，先修正参数再调用。",
    "missing": "这次调用查的对象不存在，不要用同一个标识重复查，换对象或如实告知用户。",
}


def retry_limit() -> int:
    """读重试上限。环境变量非法时回落到默认值，不让配置错误变成崩溃。"""
    raw = (os.getenv("SHOP_RETRY_LIMIT") or "").strip()
    if not raw:
        return DEFAULT_RETRY_LIMIT
    try:
        return max(0, int(raw))
    except ValueError:
        return DEFAULT_RETRY_LIMIT


def classify(output: str) -> str:
    """把一次工具返回归到 retry / arg / missing / ok 四类之一。"""
    text = (output or "").lstrip()
    match = _CODE.match(text)
    if match:
        code = int(match.group(1))
        if code == 404:
            return "missing"
        if code in RETRYABLE_CODES:
            return "retry"
        if code in ARG_CODES:
            return "arg"
        return "ok"
    # 工具层把超时/连接失败写成 "ERROR ..."，这类值得再试一次。
    if text.startswith("ERROR"):
        return "retry"
    return "ok"


__all__ = ["ARG_CODES", "HINTS", "RETRYABLE_CODES", "classify", "retry_limit"]
