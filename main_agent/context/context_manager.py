from __future__ import annotations

from typing import List

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate


class ContextManager:
    """上下文构造器。"""

    def __init__(self, system_prompt: str) -> None:
        self._system_prompt = system_prompt

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    def build_prompt(self) -> ChatPromptTemplate:
        """返回可直接接入 LCEL 链的提示词模板。"""
        return ChatPromptTemplate.from_messages(
            [
                ("system", self._system_prompt),
                ("human", "{input}"),
            ]
        )

    def build_messages(self, user_input: str) -> List[BaseMessage]:
        """预览本轮实际发送给模型的消息列表（调试用，不含历史记忆）。"""
        messages: List[BaseMessage] = self.build_prompt().format_messages(input=user_input)
        return messages
