"""上下文层：提示词原文、规则切分、经验挂载。

这一层只有文本，没有模型、没有 HTTP、没有循环。

三件事，按重要性排：

一、提示词原文在这里，是这份实现的唯一源头。
    main_agent 那边有一份历史副本（它先存在），两边必须逐字一致，
    否则历史报告的对拍就失去意义。test_shop_prompt_matches_history 负责盯住这件事。
    main_agent 删掉之后那条测试会自动跳过，不影响本模块运行。

二、cold 不是另写的，是从这份原文里切出来的。
    手写两份，两份就会各自漂移，改了一边忘了另一边，
    对比实验的差异就不再是"少了那几条规则"。
    所以这里用 split_rules 切、build_prompt 拼，
    test_prompt_split_is_lossless 保证"切完拼回去与原文逐字符相同"。

三、经验挂在提示词末尾，不塞进对话历史。
    塞进 history 的话，轨迹里就分不清哪句是模型说的、哪句是注入的经验；
    挂在提示词上，每轮重新拼一次，history 永远干净。
"""

from __future__ import annotations

import re

# 经验块由接入层统一渲染，这里只转发。
# 不在这里另写一份措辞：两个 Agent 若各写一套，注入实验就多了一个变量。
from agents.experience import attach as _attach

# 提示词原文。11 条规则，与历史报告 reports/*.json 里
# train_messages[0].content 记录的那份必须逐字相同。
SHOP_PROMPT = """你是电商平台客服助手，只能依据工具返回的数据回答。

规则：
1. 涉及商品、订单、物流、退款的事实，必须先调工具查，不许凭印象作答。
2. 工具返回什么就用什么。字段为 null、空列表、压根没有该字段，就如实说"暂无"，
   禁止推测、补全或给出示例值。
3. 状态值原样复述：订单状态是中文（已签收/已发货/待支付），
   物流状态是英文（delivered/shipped/pending），不要翻译或改写。
4. 先说查了什么、查到什么，再给结论。查不到就说查不到，不要换个接口瞎猜。
5. 多轮对话延续上下文：用户说"那个""第一个""它""那 1001 呢""算了"时，
   指代前面已经出现过的实体，沿用之前查到的 id 和参数，不要让用户重复说明。
   追问轮也要点名：答"多少钱""有货吗"时说清是哪个商品、哪个订单，不能只甩一个价格或状态。
6. 用户提到了预算、品类、商品名中的任意一项，就直接调工具搜，不要反问。
   只给预算、连品类都没说的（比如"三百块能买点啥"），也算给了条件——
   直接搜出来再筛，不要问用户"想买什么类别"。
   只有这三项完全没有时（例如"我想买个礼物送人"），才反问一句问预算。
7. 用户改口径就以最新一次为准：给了新预算、新订单号、新商品，旧值立即失效，
   换对象后必须重新查；上一轮"查不到/不存在"的结论不能顺延到新对象上。
8. 用户问"什么状态""到哪了""怎么样了"这类没分清订单还是物流的问题时，
   订单状态和物流状态各查一次，两边都报；只查一边就答，等于只答了一半。
9. 中文回答，不超过 150 字。
10. 提交退款前必须先查订单状态，看到「待支付」不要提交，
   直接说明订单还没付款、无需退款。只有本轮工具返回过退款单、
   或服务端明确返回「已申请过退款」，才可以说已申请过并停止重复提交；
   用户说「再退一次」不等于你已知情，不能凭这句话认定此前申请过。
11. 「已提交」「已申请」「已办理」「已取消」这类"已经做完"的说法，
   只能在本轮工具真的返回成功之后讲。查了状态不等于动作已经完成，
   没调工具就不要说做过，工具报错就说没办成——不许把"我这就去办"写成"已办好"。"""

# 规则块的起始标记，split_rules 靠它把"人设说明"与"规则列表"分开。
RULE_HEADER = "规则："

# 规则条目的行首形态，例如 "1. "、"11. "。
RULE_PATTERN = r"(?m)^(?=\d+\.)"

# cold 保留几条规则。
#
# 为什么是 3：
#   第 1-3 条是"必须查工具 / 禁止补全字段 / 状态原样复述"，
#   任何工具型 Agent 的底线，不算"学来的"；
#   第 4-11 条每条都能溯到一次真实失败（第 8 条来自 S08，第 11 条来自 S13），
#   属于"被评测逼出来的经验"。
#   cold = 1-3，tuned = 1-11，两者的差正好是那 8 条。
COLD_RULE_COUNT = 3


def split_rules(prompt: str) -> tuple[str, list[str]]:
    """把提示词拆成 (不变前缀, 规则列表)。

    prefix 保留"人设 + 规则："这段原文，重建时原样放回，
    所以 build_prompt(split_rules(p)[1]) 能还原出 p 本身。
    """
    head, sep, body = prompt.partition(RULE_HEADER)
    if not sep:
        raise ValueError(f"提示词里找不到 {RULE_HEADER!r}，无法拆分规则")

    prefix = head + sep + "\n"
    if body.startswith("\n"):
        body = body[1:]

    parts = re.split(RULE_PATTERN, body)
    rules = [p.strip() for p in parts if p.strip()]
    if not rules:
        raise ValueError("没切出任何规则条目，检查规则是否以 '1. ' 这类行首编号开头")
    return prefix, rules


def build_prompt(rules: list[str] | tuple[str, ...], *, prefix: str | None = None) -> str:
    """把前缀和规则拼回一份提示词。"""
    head = _PREFIX if prefix is None else prefix
    return head + "\n".join(rules)


_PREFIX, ALL_RULES = split_rules(SHOP_PROMPT)

# tuned 原样使用，不做重组。
TUNED_PROMPT = SHOP_PROMPT

# cold 只保留底线规则。
COLD_PROMPT = build_prompt(ALL_RULES[:COLD_RULE_COUNT], prefix=_PREFIX)

PROMPTS = {
    "tuned": TUNED_PROMPT,
    "cold": COLD_PROMPT,
}


def prompt_for(variant: str) -> str:
    """按变体名取提示词。未登记的变体直接报错，不做静默兜底。"""
    try:
        return PROMPTS[variant]
    except KeyError:
        raise KeyError(f"未知变体 {variant!r}，可选 {sorted(PROMPTS)}") from None


def with_experience(prompt: str, experience: str) -> str:
    """把经验文本挂到提示词末尾。经验为空时原样返回，不加任何空段落。

    实现已移到 agents/experience.py，这里保留同名函数是为了
    让上下文的调用方（core/agent.py、测试）不必关心它住在哪一层。
    """
    return _attach(prompt, experience)


__all__ = [
    "SHOP_PROMPT",
    "TUNED_PROMPT",
    "COLD_PROMPT",
    "PROMPTS",
    "ALL_RULES",
    "COLD_RULE_COUNT",
    "split_rules",
    "build_prompt",
    "prompt_for",
    "with_experience",
]
