"""文本匹配原子。

四个函数，各自只做一件事，互不依赖：
- matches：断言命中（子串，或 re: 前缀走正则）
- same：参数宽松比较（200 与 200.0、1002 与 "1002"）
- looks_non_answer：这条回答是不是压根没答（拒答模板 / 整段外文）
- looks_off_topic：查到了数据但一个字段都没引用

这些判断全是有缺陷的启发式，价值在于口径稳定、可复现、可被单测钉住，
不在于"判断准确"。词表判语义是死路，这一点写在 README 的局限里。
"""

from __future__ import annotations

import re
from typing import Iterable

# 断言里以这个前缀开头的按正则匹配，其余按子串。
REGEX_PREFIX = "re:"

# 拒答模板：模型怀疑自己被注入，于是拒绝执行任务。
# 这类回答的共性是"不复述任何业务数据，只解释自己为什么不干"。
NON_ANSWER_PATTERNS = (
    "prompt injection",
    "embedded instructions",
    "i can't follow",
    "i cannot follow",
    "i won't act on",
    "i will not follow",
    "无法遵从消息中的指令",
    "疑似提示词注入",
)


def matches(pattern, text) -> bool:
    """断言命中。

    以 re: 开头走正则，其余走子串。正则写错时退回子串匹配——
    最坏结果是这条断言查得不准，不会把正确回答判成错误，也不会让整场评测崩掉。
    """
    if not isinstance(pattern, str):
        return False

    body_text = "" if text is None else str(text)

    if pattern.startswith(REGEX_PREFIX):
        body = pattern[len(REGEX_PREFIX):]
        try:
            return re.search(body, body_text) is not None
        except re.error:
            return body in body_text

    return pattern in body_text


def same(a, b) -> bool:
    """参数宽松比较，避免 200 与 200.0、1002 与 "1002" 被当成不同。"""
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return str(a) == str(b)


def looks_non_answer(answer: str) -> bool:
    """这条回答是不是"根本没在答"。

    只认两种很窄的情况，避免误伤正常回答：
    1. 命中拒答模板；
    2. 长度 >= 30 且中文字符占比不到 20%（这套题全中文提问，长段外文本身违规）。
    """
    text = (answer or "").strip()
    if not text:
        return False

    low = text.lower()
    if any(p in low for p in NON_ANSWER_PATTERNS):
        return True

    # 短答不判：正常回答里也可能夹几个英文词。
    if len(text) < 30:
        return False

    zh = len(re.findall(r"[\u4e00-\u9fff]", text))
    return zh / len(text) < 0.2


def looks_off_topic(answer: str, outputs: Iterable[str]) -> bool:
    """回答有没有一处引用工具返回的数据。

    只在断言已判失败时调用，用来区分两种情况：
    1. 查到了数据，回答里一个字段都没引用 —— 答非所问；
    2. 引用了数据但引错或缺了 —— 缺事实 / 错事实。

    判据故意保守：只要回答里出现返回中的任意一个值，就不判答非所问。
    """
    blob = "\n".join(str(o or "") for o in outputs)
    if not blob:
        return False

    values = set(re.findall(r'"([^"]{2,40})"', blob))
    values |= set(re.findall(r":\s*(-?\d{2,}(?:\.\d+)?)", blob))
    values |= {
        v.split(".")[0]
        for v in values
        if re.fullmatch(r"-?\d+(?:\.\d+)?", v)
    }

    return not any(v in answer for v in values)
