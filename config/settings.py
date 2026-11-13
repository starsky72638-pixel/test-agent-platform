"""
这部分为从.env中读取变量注入在服务中
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR
ENV_FILE = BASE_DIR / ".env"

# 只加载一次，不覆盖已存在的真实环境变量
load_dotenv(dotenv_path=ENV_FILE, override=False)

DEFAULT_SYSTEM_PROMPT = "你是一个简洁、准确的中文 AI 助手。回答直接，不啰嗦。"


@dataclass(frozen=True)
class Settings:
    """运行期配置快照。"""

    api_key: str
    model: str
    base_url: Optional[str]
    temperature: float
    system_prompt: str

    def validate(self) -> None:
        """缺少必要配置时给出可操作的报错，而不是抛出底层堆栈。"""
        if not self.api_key:
            raise ValueError(
                f"未读取到 API Key。请在 {ENV_FILE} 中配置 OPENAI_API_KEY=你的密钥\n"
                f"（可参考同目录下的 .env.example）"
            )


def load_settings() -> Settings:
    """从环境变量组装配置。"""
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    base_url = (os.getenv("OPENAI_BASE_URL") or "").strip() or None

    raw_temp = (os.getenv("TEMPERATURE") or "").strip()
    try:
        temperature = float(raw_temp) if raw_temp else 0.7
    except ValueError:
        temperature = 0.7

    settings = Settings(
        api_key=api_key,
        model=(os.getenv("MODEL_NAME") or "gpt-4o-mini").strip(),
        base_url=base_url,
        temperature=temperature,
        system_prompt=(os.getenv("SYSTEM_PROMPT") or DEFAULT_SYSTEM_PROMPT).strip(),
    )
    settings.validate()
    return settings
