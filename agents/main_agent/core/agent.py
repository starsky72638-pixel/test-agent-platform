"""ReAct 循环：模型决定调哪个工具，代码执行并把结果回灌到上下文。

history 传入时为多轮续聊（在同一个列表上追加），为 None 时新建会话。

返回结构：
    {
        "answer":    最后一轮模型说的话（步数耗尽时为空字符串）
        "messages":  完整对话历史，下一轮继续传进来
        "status":    "ok" 正常结束 / "max_steps" 步数耗尽
        "steps":     实际用掉的模型调用轮数
        "slots":     会话槽位（预算/品类/最近订单/最近商品）
        "sent_msgs": 这一轮真正发给模型的消息条数
    }

上下文怎么组织的：
    发给模型的消息 = 系统提示（含槽位 + 事实表）+ 最近 keep_turns 轮，见 memory.build_view。
    history 本身不裁剪，仍完整保留——replay 的逐轮切片依赖它，动了就错。

槽位和事实表为什么不走函数参数：
    都挂在 history[0].additional_kwargs 上，随 history 自然传递，
    调用方（main.py / replay.py）不需要改任何一行，多轮之间也不会丢。
"""

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from agents.main_agent.view.memory import (
    build_view,
    load_facts,
    load_slots,
    save_facts,
    save_slots,
    update_facts,
    update_slots,
)


def run_agent(
    llm,
    tools,
    system_prompt,
    question,
    max_steps=6,
    history=None,
    keep_turns=3,
):
    bound = llm.bind_tools(tools)
    tool_map = {t.name: t for t in tools}
    messages = history if history is not None else [SystemMessage(system_prompt)]

    # 从上一轮留下的系统消息里取回槽位和事实表；首次对话都是空字典。
    slots = load_slots(messages)
    facts = load_facts(messages)

    messages.append(HumanMessage(question))

    for step in range(1, max_steps + 1):
        # 只把"系统提示 + 槽位 + 事实表 + 最近 keep_turns 轮"发给模型。
        view = build_view(messages, system_prompt, slots, keep_turns, facts)
        ai = bound.invoke(view)
        messages.append(ai)

        # 把这一轮的工具实参反填进槽位，供后续轮次（含被裁掉的轮次）使用。
        update_slots(slots, ai.tool_calls)
        save_slots(messages, slots)

        if not ai.tool_calls:
            return {
                "answer": ai.content or "",
                "messages": messages,
                "status": "ok",
                "steps": step,
                "slots": slots,
                "sent_msgs": len(view),
            }

        for call in ai.tool_calls:
            fn = tool_map.get(call["name"])
            print(f"  [tool] {call['name']} {call['args']}", flush=True)

            out: str = ""
            if fn is None:
                out = f"未知工具 {call['name']}，可用工具：{list(tool_map)}"
            else:
                try:
                    out = str(fn.invoke(call["args"]))
                except Exception as exc:
                    out = f"工具执行失败 {type(exc).__name__}: {exc}"

            print(f"  [result] {out[:120]}", flush=True)
            messages.append(ToolMessage(content=out, tool_call_id=call["id"]))

            # 把结果里"以后还要用"的字段记进事实表：
            # 下一轮引用一个已被窗口裁掉的对象时，靠的就是这里存下的东西。
            update_facts(facts, call["name"], call["args"], out)

        # 一批工具跑完再写回系统消息，不用每个 call 都改一次。
        save_facts(messages, facts)

    return {
        "answer": "",
        "messages": messages,
        "status": "max_steps",
        "steps": max_steps,
        "slots": slots,
        "sent_msgs": 0,
    }
