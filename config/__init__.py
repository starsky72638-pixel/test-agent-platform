"""配置层。"""

"""共享基础设施：配置与模型客户端。"""

from config.settings import PROJECT_ROOT, Settings, load_settings
from config.llm_client import build_llm

__all__ = ["PROJECT_ROOT", "Settings", "load_settings", "build_llm"]
