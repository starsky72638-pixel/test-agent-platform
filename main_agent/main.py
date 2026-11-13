"""Agent 控制台入口（多轮会话）。

运行（在项目根目录）：
    python -m main_agent.main

依赖方向（自下而上，任何一层都不认识上层）：
    config  ->  tools  ->  core  ->  main
"""

from __future__ import annotations

import sys

from config import build_llm, load_settings
from main_agent.core.agent import run_agent
from main_agent.tools import get_registry, register_default_tools,SHOP_SYSTEM_PROMPT

EXIT_WORDS = {"exit", "quit", "q", ":q", "bye"}


def main() -> int:
    # 配置、模型、工具都是运行期不变量，只建一次
    try:
        settings = load_settings()
    except ValueError as exc:
        print(f"[配置错误] {exc}")
        return 1

    llm = build_llm(settings, streaming=False)
    register_default_tools()
    tools = get_registry().list()
    tool_names = [t.name for t in tools]

    print(" Test Agent  |  输入 exit / quit 退出")
    print(f" 模型：{settings.model}")
    print(f" 工具：{tool_names or '无'}")

    history = None  # 会话级上下文，跨轮累积

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            return 0

        if not user_input:
            continue
        if user_input.lower() in EXIT_WORDS:
            print("再见。")
            return 0

        try:
            #后续可将SHOP_SYSTEM_PROMPT改为系统提示词指向的md文件中内容
            answer, history = run_agent(
                llm, tools, SHOP_SYSTEM_PROMPT, user_input, history=history
            )
        except Exception as exc:  # 网络/鉴权/额度等异常，不打断会话
            print(f"[调用失败] {type(exc).__name__}: {exc}")
            continue

        print(f"AI: {answer}")
        print(f"  （上下文累计 {len(history)} 条消息）")


if __name__ == "__main__":
    sys.exit(main())
