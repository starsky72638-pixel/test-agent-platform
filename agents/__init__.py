"""真实 Agent 的接入层。每个 Agent 一套完整实现，一个目录。

目录形状统一，看目录名就知道有哪些 Agent 可插：

    agents/main_agent/     第一个 Agent（原 main_agent/ 整体搬入）
        core/agent.py      ReAct 循环
        tools/             5 个 skill
        view/memory.py     上下文（窗口裁剪 + 槽位 + 事实卡片）
        main.py            交互式控制台入口

    agents/shop/           第二个 Agent，代码完全独立于上面那个
        core/agent.py      循环 + Runtime 适配
        tools/shop_skill.py
        view/context.py    提示词 + 规则切分 + 经验挂载
        llm.py

为什么不是"一个通用接入层包一层皮"：
    如果 shop 只是 import main_agent 的循环和工具，那它就不是第二个 Agent，
    只是第一个 Agent 换了个提示词。main_agent 一改，shop 立刻断，
    "可插拔"就只剩结构上的说法。所以 agents/shop 自带循环与工具，
    删掉 agents/main_agent 之后它照样跑（test_shop_does_not_import_main_agent 守着这条）。

和 demos/ 的分工：
    demos/    最小协议示例：不连模型、不连服务，只证明"一个类加一个 run 就能接进来"。
    agents/   真实 Agent：会调模型、会连 HTTP 服务，依赖比核心重得多。
    删掉 demos/ 与 agents/ 两个目录，core / evaluator / memory 仍可 import、仍能跑单测。

为什么每个 Agent 都拆 core / tools / view 三个子目录：
    这三样是"换 Agent"时真正会变的东西。混在一个文件里就分不清
    "这次对比实验改了模型、改了工具集，还是改了提示词"。
    拆开后看目录名就知道。组合它们的那个文件只做编排。

这里刻意不在包顶层 import 任何具体 Agent：
    import agents 不应该顺带把 langchain 拖进内存。
    要哪个 Agent，显式 from agents.shop import ShopAgent。
"""

__all__: list[str] = []
