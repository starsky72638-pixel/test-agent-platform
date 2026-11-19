"""经验闭环的编排层：把 core / evaluator / memory 串成一条能真跑的链。

两个函数，一条单向链，没有第三个入口：

    learn(agent, tasks, store)    跑 fit -> 判分 -> 提炼 -> 过闸门 -> 入库
    verify(agent, tasks, store)   holdout 跑两遍（无经验 / 有经验）-> 对比

为什么单独一层而不是塞进 memory：
memory 是"经验怎么存、怎么找、怎么判质量"，它不该知道"先跑哪批任务"。
编排是流程知识，流程会变（今天一轮学习，明天可能要加人工复核），
存储不该跟着变。反过来 core 也不能知道 memory——那会成环。
所以这一层依赖 core + evaluator + memory 三者，三者互不知道对方的编排方式。

依赖方向（只向下，不向上）：

    pipeline -> memory -> evaluator -> core
    pipeline -> evaluator -> core
    pipeline -> core

它不认识任何具体 Agent，也不 import agents/ 与 demos/。
"塞哪个 Agent 进来"由 CLI 决定，不由这一层决定。
"""

from pipeline.learn import JudgedHook, LearnResult, learn
from pipeline.verify import CompareResult, SideHook, SideResult, verify

__all__ = [
    "learn",
    "verify",
    "LearnResult",
    "CompareResult",
    "SideResult",
    "JudgedHook",
    "SideHook",
]
