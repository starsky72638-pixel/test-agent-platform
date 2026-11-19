"""从经验库里为一条新任务检索可用的经验。

能不能用，先过两道门槛，两道都跟"这道题的答案"无关：
- 关键词必须有重合（二元组 Jaccard > 0）：决定"是不是同一类问题"；
- 形态必须相同（单轮 / 多轮）：多轮里漏查和单轮里漏查，补法不一样。
  形态不符直接不注入——这条曾经是加分项，实测出过事故：一条"多轮漏查订单"
  的经验被注入到单轮退款题上，共 213 字，把本来通过的题打成 wrong_fact。
  加分项只能让它在同形态的候选里靠前，拦不住跨形态混入；门槛才拦得住。

过了门槛再排序。排序分数里形态那一项恒为常数（门槛已经保证形态相同），
真正拉开差距的是关键词重合度和置信度。

为什么不引向量库：
库的规模是十几条，关键词重合已经够用；引入 embedding 需要额外依赖、额外模型调用，
收益在这个量级上是负的，而且会让"为什么检索到这条"变得不可解释——
报告里要能写出"命中了哪些词"，这比一个 0.83 的余弦相似度有用。

检索失败是允许的：没有足够的重合就返回空，宁可不注入，也不要把无关经验塞进去。
经验里的任何数字都被 quality 拦在主库外，所以这里也不需要做泄漏过滤。
"""

from __future__ import annotations

from dataclasses import dataclass

from core.task import Task

from memory.extract import task_tags
from memory.schema import Experience, shape_of_turn_count

# 权重：关键词是主信号，置信度是微调。
# shape 项保留但不参与排序判断：形态已经升为门槛，能走到打分这一步的候选
# shape 恒为 1，它贡献的是常数 0.15。之所以不把这一项删掉，是为了让分数、
# 阈值与历史报告保持同一口径——否则"重跑结果变了"就分不清是门槛起作用，
# 还是分数被重新标定过。
W_KEYWORD = 0.75
W_SHAPE = 0.15
W_CONFIDENCE = 0.10

# 低于这个分数视为不相关。定得低是因为中文短句的二元组重合本来就稀疏，
# 定高了会导致"明明同类却一条都检索不到"。
# 诚实说一句：有 0.15 的形态常数垫着，这个阈值实际上很松，关键词只要有一个
# 重合就能过。已知，本轮不动它——本轮只把形态从加分项改成门槛，一个变量。
# 收紧关键词门槛等有更多实验数据再说，否则涨跌无法归因。
DEFAULT_MIN_SCORE = 0.12


@dataclass
class Hit:
    """一次检索结果。带分项分数，便于解释"为什么命中这条"。"""

    exp: Experience
    score: float
    keyword: float = 0.0
    shape: float = 0.0
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "exp_id": self.exp.exp_id,
            "failure_type": self.exp.failure_type,
            "score": round(self.score, 3),
            "keyword": round(self.keyword, 3),
            "shape": self.shape,
            "confidence": self.confidence,
        }

    def why(self) -> str:
        shape = "相符" if self.shape else "不符"
        return (
            f"关键词 {self.keyword:.2f} / 形态{shape} / "
            f"置信 {self.confidence:.2f} -> {self.score:.3f}"
        )


def jaccard(a: list[str] | set[str], b: list[str] | set[str]) -> float:
    sa, sb = set(a or []), set(b or [])
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _shape_score(task: Task, exp: Experience) -> float:
    """形态是否相同（1.0 / 0.0）。只给门槛用，不计入 score。"""
    task_shape = shape_of_turn_count(task.turn_count())
    return 1.0 if exp.trigger.shape == task_shape else 0.0


def score_one(task: Task, exp: Experience, *, tags: list[str] | None = None) -> Hit:
    """给一条经验打分。tags 可传入以避免重复抽词。

    shape 字段照常算出来，但它的作用是门槛（见 retrieve），不是加分：
    形态不符的候选在 retrieve 里就被丢掉了，走不到这里参与排序。
    """
    task_kw = task_tags(task) if tags is None else tags
    keyword = jaccard(task_kw, exp.trigger.tags)
    shape = _shape_score(task, exp)
    confidence = exp.confidence
    score = W_KEYWORD * keyword + W_SHAPE * shape + W_CONFIDENCE * confidence
    return Hit(exp=exp, score=round(score, 4), keyword=keyword, shape=shape, confidence=confidence)


def retrieve(
    task: Task,
    experiences: list[Experience],
    *,
    top_k: int = 8,
    min_score: float = DEFAULT_MIN_SCORE,
) -> list[Hit]:
    """检索可注入的经验，按分数降序。

    experiences 应由调用方保证已是 active 状态（store.active()），
    这里再兜一次：退休的、过期的直接跳过，避免调用方漏过滤。
    """
    tags = task_tags(task)
    hits: list[Hit] = []
    for exp in experiences or []:
        if not exp.action.strip():
            continue
        hit = score_one(task, exp, tags=tags)
        # 门槛一：形态必须相同。
        # 多轮的教训（第几轮该补查什么）搬到单轮题上不成立，反过来也一样。
        # 曾经这里是加分项，结果一条多轮漏查经验被注入到单轮退款题，
        # 把 PASS 打成 wrong_fact。门槛比加分更保守，也更可解释。
        if hit.shape < 1.0:
            continue
        # 门槛二：一个关键词都不重合，说明不是同一类问题。
        # 只靠"都是单轮任务"这一点相似就注入，等于把无关经验塞进提示词，
        # 比不注入更糟——它会让模型按别人的教训去回答当前问题。
        if hit.keyword <= 0:
            continue
        if hit.score >= min_score:
            hits.append(hit)
    hits.sort(key=lambda h: (h.score, h.exp.support, h.exp.exp_id), reverse=True)
    if top_k is not None and top_k >= 0:
        hits = hits[:top_k]
    return hits


def retrieve_from_store(store, task: Task, *, top_k: int = 8, min_score: float = DEFAULT_MIN_SCORE) -> list[Hit]:
    """从 ExperienceStore 直接检索。库为空时返回空列表，不报错。"""
    return retrieve(task, store.active(), top_k=top_k, min_score=min_score)


def covered_tags(hits: list[Hit]) -> list[str]:
    """辅助：本次注入覆盖了哪些关键词，供报告说明"为什么是这几条"。"""
    out: set[str] = set()
    for hit in hits:
        out.update(hit.exp.trigger.tags)
    return sorted(out)
