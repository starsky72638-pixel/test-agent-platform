"""名字 -> Agent 工厂。CLI 只认名字，不 import 任何具体 Agent。

为什么需要这一层：
    scripts/run_loop.py 原来写死 `from agents.shop import make_shop_agent`。
    那意味着"接入一个新 Agent"要改 CLI——CLI 是 Runtime 的入口，
    改它就等于改核心。加一个 Agent 应该只加一行注册。

为什么存工厂而不是实例：
    Agent 构造要读环境变量（API Key）。在 import 时就构造，
    缺密钥的机器连 `--help` 都打不开。工厂把构造推迟到真要跑的时候。

为什么 import 写在函数里：
    `import agents.builtin` 不该顺带把 langchain 和 httpx 拖进内存，
    更不该在缺密钥时失败。这一点由 tests/test_main_adapter.py 守着。

与 core.registry 的分工：
    这里存的是"能造哪些"（工厂）；自建的 INSTANCES 记"这个进程里造过谁"，只用于自省。
    刻意**不**往 core.registry.DEFAULT 里写：那张表的契约是"默认必须为空"，
    是"删掉整个 agents/ 与 demos/，core 仍可 import 和测试"的运行时护栏。
    接入层往里塞实例，等于让业务 Agent 反向污染核心全局状态。
"""

from __future__ import annotations

from typing import Any, Callable

from core.registry import Registry, UnknownAgentError

# 本进程造过的实例，纯自省用。不碰 core.registry.DEFAULT。
INSTANCES = Registry("builtin")

_FACTORIES: dict[str, Callable[..., Any]] = {}

# 旧名字保持能用：之前几轮真跑用的就是 --agent cold / --agent tuned。
# cs 是 customer 的短别名，给命令行省点打字。
ALIASES = {
    "cold": "shop-cold",
    "tuned": "shop-tuned",
    "shop": "shop-tuned",
    "main_agent": "main",
    "cs": "customer",
    "cs-cold": "customer-cold",
    "wf": "workflow",
}


def register(name: str, factory: Callable[..., Any]) -> None:
    """登记一个工厂。同名重复登记直接报错，不静默覆盖。"""
    if name in _FACTORIES:
        raise ValueError(f"Agent 名字重复: {name!r}")
    _FACTORIES[name] = factory


# 磁盘上某个 Workflow 的可寻址前缀，形如 wf:demo。
# 不把每个 id 都塞进 _FACTORIES：那张表回答"能造哪些类型"，
# 而 workflow 是同一类型的多个实例，混进去会让"类型列表"失去意义。
WORKFLOW_PREFIX = "wf:"


def workflow_ids() -> list[str]:
    """磁盘上现有的 Workflow id。读不到就当没有——列个表不该让入口崩。"""
    from agentplatform.store import AgentStore

    return AgentStore().ids()


def names() -> list[str]:
    """全部可寻址的名字：静态类型 + 磁盘上的 workflow 实例。

    --list、/agents、/health 都读这个函数，所以三处不会各说一套。
    """
    dynamic = {f"{WORKFLOW_PREFIX}{wid}" for wid in workflow_ids()}
    return sorted(set(_FACTORIES) | dynamic)


def make(name: str, **kwargs) -> Any:
    """按名字造一个 Agent。kwargs 原样传给工厂。

    wf:<id> 会被拆成 workflow 工厂 + 对应定义，所以刚存下的 Workflow
    不用注册、不用重启就能按名字跑（run_shop --agent wf:demo 直接可用）。
    """
    key = ALIASES.get(name, name)
    if key.startswith(WORKFLOW_PREFIX):
        kwargs.setdefault("workflow", key[len(WORKFLOW_PREFIX):])
        key = "workflow"
    try:
        factory = _FACTORIES[key]
    except KeyError:
        available = ", ".join(names()) or "（空）"
        raise UnknownAgentError(
            f"没有名为 {name!r} 的 Agent，当前可用: {available}"
        ) from None

    agent = factory(**kwargs)
    # 登记实例只用于自省（谁被造过、叫什么），失败不该影响执行。
    INSTANCES.register(agent, replace=True)
    return agent


def _shop_factory(variant: str):
    def factory(**kwargs):
        from agents.shop import make_shop_agent

        return make_shop_agent(variant, **kwargs)

    return factory


def _main_factory(**kwargs):
    from agents.main_agent.adapter import MainAgent

    return MainAgent(**kwargs)


def _customer_factory(**kwargs):
    from agents.customer import make_customer_agent

    return make_customer_agent(**kwargs)


def _customer_cold_factory(**kwargs):
    """冷启动变体：只有角色，没有行为准则，工具描述也压到初稿水平。

    它存在的目的是给经验学习一个"真的会失败"的起点。
    两条约束都要删：prompt 里的规则、工具描述里的规则。上一版只删了前者，
    而 apply_refund 的描述里写着"调用前先 get_order"，等于约束还在，
    于是 6/6 全过、学不到任何东西。

    不改工厂签名，只注入 prompt= 与 tools=——按 agents/customer/__init__.py
    里已经写好的约定（要做对照就传参，不必动 make_customer_agent）。
    """
    from agents.customer import COLD_PROMPT, cold_tools, make_customer_agent

    kwargs.setdefault("prompt", COLD_PROMPT)
    kwargs.setdefault("tools", cold_tools())
    kwargs.setdefault("name", "customer-cold")
    return make_customer_agent(**kwargs)


def _default_memory_store():
    """`wf:*` 默认读归并库 data/memory/experiences.jsonl。

    库不存在就是空库：memory 节点输出空串，prompt 里不会多出空表头，
    与"没有经验"完全等价。刻意不猜别的路径——猜错一个，实验里就会混进
    另一个目录留下的经验，结果对了也说不清是谁的功劳。

    import 写在函数里，理由同本模块其它工厂：`import agents.builtin`
    不该顺带把 memory 整包拉进来。
    """
    from memory.store import DEFAULT_STORE_PATH, ExperienceStore

    return ExperienceStore(DEFAULT_STORE_PATH)


def _workflow_factory(**kwargs):
    """按 JSON 定义执行的 Agent。

    工具池与经验库都在这里注入，不在 agentplatform 里 import 业务模块——
    平台层不该认识具体业务。默认给客服域三个工具，
    工作流声明了没注册的工具会在构造时被 schema 校验拦下，
    不会跑到一半才发现工具不存在。

    经验库默认指向归并库（DEFAULT_STORE_PATH）。要显式关掉就传
    memory_store=None，要做对照实验就传自己的库——两条路都不需要改平台层。
    """
    from agentplatform.adapter import build_workflow_agent
    from agents.customer.tools import CUSTOMER_TOOLS

    ref = kwargs.pop("workflow", None)
    kwargs.setdefault("tools", list(CUSTOMER_TOOLS))
    kwargs.setdefault("memory_store", _default_memory_store())
    return build_workflow_agent(ref, **kwargs)


register("shop-cold", _shop_factory("cold"))
register("shop-tuned", _shop_factory("tuned"))
register("main", _main_factory)
register("customer", _customer_factory)
register("customer-cold", _customer_cold_factory)
register("workflow", _workflow_factory)

DEFAULT_AGENT = "shop-cold"

__all__ = ["register", "names", "make", "ALIASES", "DEFAULT_AGENT"]
