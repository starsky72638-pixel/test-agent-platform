"""模型客户端：Agent 与评测共用。
这部分为配置模型
"""
from __future__ import annotations

from typing import Callable, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from config.settings import Settings


def build_llm(settings: Settings, streaming: bool = True) -> ChatOpenAI:
    """按配置创建模型客户端。"""
    return ChatOpenAI(
        model=settings.model,
        api_key=settings.api_key,
        base_url=settings.base_url,
        temperature=settings.temperature,
        streaming=streaming,
        max_retries=2,
    )


def build_chain(prompt: ChatPromptTemplate, llm: Runnable) -> Runnable:
    """组装 LCEL 链：prompt | llm | StrOutputParser。"""
    return prompt | llm | StrOutputParser()


class LLMEngine:
    """对上层暴露最小接口：给一句话，拿一句回答。"""

    def __init__(self, chain: Runnable) -> None:
        self._chain = chain

    def answer(self, user_input: str, on_token: Optional[Callable[[str], None]] = None) -> str:
        """执行一次推理。on_token 非空时按 token 流式回调。"""
        if on_token is None:
            return self._chain.invoke({"input": user_input})

        chunks = []
        for chunk in self._chain.stream({"input": user_input}):
            if not chunk:
                continue
            chunks.append(chunk)
            on_token(chunk)
        return "".join(chunks)
