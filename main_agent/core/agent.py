"""ReAct 循环：模型决定调哪个工具，代码执行并把结果回灌到上下文。

history 传入时为多轮续聊（在同一个列表上追加），为 None 时新建会话。
"""

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage


def run_agent(llm, tools, system_prompt, question, max_steps=6, history=None):
    bound = llm.bind_tools(tools)
    tool_map = {t.name: t for t in tools}
    messages = history if history is not None else [SystemMessage(system_prompt)]
    messages.append(HumanMessage(question))

    for _ in range(max_steps):
        ai = bound.invoke(messages)
        messages.append(ai)
        if not ai.tool_calls:
            return ai.content, messages

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

    return f"达到最大步数 {max_steps}，已停止", messages
