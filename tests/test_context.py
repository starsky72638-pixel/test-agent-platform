"""上下文槽位的测试：机制在 core，规则在 shop。

守三件事：
1. 槽位只从结构化工具实参填，空值不覆盖旧值；
2. 渲染出来的提示只含 labels 里有的键，且顺序稳定；
3. 跨轮真的传下去——第 2 轮的预算要出现在第 3 轮发给模型的系统提示里。
   第 3 条是这个模块存在的理由，前两条只是它的前提。
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from agents.shop.core.agent import run_loop
from agents.shop.view.slots import ARG_TO_SLOT, SLOT_LABELS
from core.context import render_slots, update_slots


class FakeTool:
    def __init__(self, name: str, output: str = "ok") -> None:
        self.name = name
        self.output = output

    def invoke(self, args):
        return self.output


class ScriptedLLM:
    def __init__(self, steps) -> None:
        self.steps = list(steps)
        self.calls = 0
        self.seen: list[list] = []

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.seen.append(list(messages))
        step = self.steps[self.calls] if self.calls < len(self.steps) else "结束"
        self.calls += 1
        if isinstance(step, AIMessage):
            return step
        return AIMessage(content=str(step))


def _call(name: str, args: dict, cid: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": cid}])


def test_update_slots_fills_from_tool_args():
    slots: dict = {}
    update_slots(
        slots,
        [{"name": "search_products", "args": {"q": "耳机", "max_price": 900}}],
        ARG_TO_SLOT,
    )

    assert slots == {"category": "耳机", "budget_max": 900}


def test_empty_value_does_not_wipe_existing_slot():
    """一次没传 max_price 的搜索，不能把上一轮的预算清掉。"""
    slots = {"budget_max": 900}
    update_slots(slots, [{"name": "search_products", "args": {"q": "耳机"}}], ARG_TO_SLOT)

    assert slots["budget_max"] == 900


def test_unknown_tool_and_arg_are_ignored():
    slots: dict = {}
    update_slots(slots, [{"name": "get_weather", "args": {"city": "北京"}}], ARG_TO_SLOT)

    assert slots == {}


def test_render_slots_empty_returns_empty_string():
    assert render_slots({}, SLOT_LABELS) == ""


def test_render_slots_follows_label_order():
    text = render_slots({"last_order_id": 1001, "budget_max": 900}, SLOT_LABELS)

    assert text.index("预算上限") < text.index("订单号")
    assert "900" in text and "1001" in text


def test_slots_carry_across_turns_into_prompt():
    """第 2 轮传了 max_price=900，第 3 轮的系统提示里必须能看到它。

    这就是"多轮上下文能力"和"靠经验注入打补丁"的分界线：
    前者关掉 Memory 仍然成立，后者一关就退回去。
    """
    llm = ScriptedLLM(
        [
            _call("search_products", {"q": "耳机", "max_price": 900}, "c1"),
            "预算 900 以内有 1 款",
            "沿用 900 的预算，只有 1 款",
        ]
    )
    slots: dict = {}
    run_loop(
        llm,
        [FakeTool("search_products")],
        "系统提示",
        "预算提到 900",
        max_steps=2,
        slots=slots,
    )

    assert slots["budget_max"] == 900

    run_loop(
        llm,
        [FakeTool("search_products")],
        "系统提示",
        "再帮我搜个耳机",
        max_steps=2,
        slots=slots,
    )

    system_text = str(llm.seen[-1][0].content)
    assert "900" in system_text
    assert "预算上限" in system_text


def test_prompt_is_byte_identical_when_no_slots_and_no_plan():
    """两个功能都关掉时，系统提示必须与改动前逐字相同。

    否则"关闭开关"就不是真的回到旧行为，对照实验不成立。
    """
    llm = ScriptedLLM(["好的"])
    run_loop(llm, [FakeTool("search_products")], "系统提示原文", "随便问问", max_steps=2)

    assert str(llm.seen[0][0].content) == "系统提示原文"
