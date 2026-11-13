"""HTTP 工具：Agent 唯一的手脚，负责真实发出请求。

设计约束：
- 返回值永远是字符串，且永远不抛异常。工具失败是正常事件，不该打断对话。
- 状态码单独成行，方便模型稳定解析，也方便后面做断言。
"""

from __future__ import annotations

import httpx
from langchain_core.tools import tool

TIMEOUT = 5.0
MAX_BODY = 800


@tool
def http_request(
    method: str,
    url: str,
    body: dict | None = None,
    headers: dict | None = None,
) -> str:
    """对指定 URL 发起 HTTP 请求，返回状态码与响应体，用于验证接口的真实行为。

    Args:
        method: HTTP 方法，GET / POST / PUT / PATCH / DELETE。
        url: 完整地址，必须以 http:// 或 https:// 开头。
        body: 可选，JSON 请求体，POST / PUT 时使用。
        headers: 可选，自定义请求头，如鉴权 token。
    """
    if not url.startswith(("http://", "https://")):
        return f"ERROR 非法 URL：{url}，必须以 http:// 或 https:// 开头"

    try:
        with httpx.Client(timeout=TIMEOUT, follow_redirects=True) as client:
            resp = client.request(
                method.upper(),
                url,
                json=body if body else None,
                headers=headers or None,
            )
    except httpx.TimeoutException:
        return f"ERROR 请求超时（超过 {TIMEOUT}s）：{url}"
    except httpx.HTTPError as exc:
        return f"ERROR 请求失败 {type(exc).__name__}: {exc}"

    text = resp.text or ""
    if len(text) > MAX_BODY:
        text = text[:MAX_BODY] + f" ...(已截断，原始长度 {len(resp.text)} 字符)"
    return f"status={resp.status_code}\nbody={text}"
