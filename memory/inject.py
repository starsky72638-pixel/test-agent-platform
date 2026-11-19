"""把检索到的经验渲染成提示词片段，并包成 Runtime 认的经验提供者。

渲染有三条纪律：

1. 顶部必须有边界声明：经验说的是"该怎么做"，不是"现在是什么"。
   没有这句话，模型会把经验里的旧结论当成本轮事实——这正是之前出过的错
   （把上一轮卡片里的旧金额当成新答案）。经验越像事实，这个风险越大。

2. 只放动作，不放证据。证据里含真实数字和单号（那是查账用的），
   一旦进提示词就变成"答案泄漏"。quality 只保证经验文本无数字，
   但证据字段本身是有数字的，所以这里必须只取 exp.text()。

3. 有条数上限与字符上限。注入不是越多越好：十几条经验全塞进去会挤掉
   真正的对话上下文，而且互相冲突时模型无从取舍。超限时按分数从高到低截断。

提供的接口与 core.runner.ExperienceProvider 对齐：
    provider(task) -> (rendered_text, [exp_id, ...])
runner 拿到 ids 后可以在本轮结束后回填"注入过谁"，这是失效机制的分母。
"""

from __future__ import annotations

from typing import Iterable, Sequence

from core.task import Task

from memory.retrieve import Hit, retrieve_from_store
from memory.schema import Experience

EXPERIENCE_HEADER = "【历史经验（来自过去失败任务的教训，只说明该怎么做，不是当前事实）】"
# 尾部两句各管一件事，缺一件都会出真问题：
#   前半句防"把经验当事实"——经验里的旧结论不能顶替本轮工具返回；
#   后半句防"经验越权"——经验是从别的任务提炼的，它不知道这一轮用户要什么。
#   实测过反例：用户明确要求执行某个动作，注入的却是一条禁止该动作的经验，
#   模型没去少调工具，而是把回答说错了（PASS -> wrong_fact）。
#   经验与当前请求冲突时，必须让位给用户，否则注入会从帮忙变成使坏。
EXPERIENCE_FOOTER = (
    "以上只是做事方式：具体价格、金额、状态、单号一律以本轮工具返回为准。"
    "若某条经验与用户本轮的明确请求冲突——例如经验说不要调用某个动作，"
    "而用户正是要求执行它——以用户请求为准，照常执行并如实汇报，"
    "不要因为经验而拒绝、拖延或改口。"
)

# 兼容短名：早期代码用的是 HEADER / FOOTER，保留别名避免外部引用断掉。
HEADER = EXPERIENCE_HEADER
FOOTER = EXPERIENCE_FOOTER

DEFAULT_MAX_ITEMS = 6
DEFAULT_MAX_CHARS = 1200


def _as_items(items: Iterable[Hit | Experience]) -> list[tuple[Experience, float]]:
    out: list[tuple[Experience, float]] = []
    seen: set[str] = set()
    for item in items or []:
        if isinstance(item, Hit):
            exp, score = item.exp, item.score
        else:
            exp, score = item, item.confidence
        if exp.exp_id in seen:
            continue
        seen.add(exp.exp_id)
        out.append((exp, score))
    out.sort(key=lambda pair: (pair[1], pair[0].support), reverse=True)
    return out


def render(
    items: Iterable[Hit | Experience],
    *,
    max_items: int = DEFAULT_MAX_ITEMS,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> str:
    """渲染成一段可直接拼进系统提示的文本。没有可用经验时返回空串。"""
    picked = _as_items(items)
    if not picked:
        return ""

    body: list[str] = []
    used = 0
    for exp, _score in picked:
        if len(body) >= max_items:
            break
        line = f"{len(body) + 1}. {exp.text()}"
        if used + len(line) > max_chars:
            break
        body.append(line)
        used += len(line)

    if not body:
        return ""
    return "\n".join([EXPERIENCE_HEADER, *body, EXPERIENCE_FOOTER])


def render_for_task(
    store,
    task: Task,
    *,
    top_k: int = DEFAULT_MAX_ITEMS,
    min_score: float | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> tuple[str, list[str]]:
    """检索 + 渲染一步到位。返回 (文本, 命中的 exp_id 列表)。"""
    kwargs = {}
    if min_score is not None:
        kwargs["min_score"] = min_score
    hits: Sequence[Hit] = retrieve_from_store(store, task, top_k=top_k, **kwargs)
    text = render(hits, max_items=top_k, max_chars=max_chars)
    ids = [h.exp.exp_id for h in hits] if text else []
    return text, ids


def make_provider(
    store,
    *,
    top_k: int = DEFAULT_MAX_ITEMS,
    min_score: float | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
):
    """包成 core.runner 需要的 experience_provider。

    runner 会把 ids 留在上下文里，本轮结束后可用 store.record_outcome(ids, passed=...)
    回填效果——这是"经验是否真的有帮助"的唯一证据来源。
    """

    def provider(task: Task) -> tuple[str, list[str]]:
        return render_for_task(
            store, task, top_k=top_k, min_score=min_score, max_chars=max_chars
        )

    return provider
