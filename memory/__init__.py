"""经验记忆层：把"做错过什么"变成"下次该怎么做"。

闭环里的位置：

    执行 -> Trace -> 判定(Verdict) -> extract -> quality -> store
                                        ^                   |
                                        |                   v
                                    retriever  <-- 新任务 -> inject -> Agent 再执行
                                                                            |
                                                     store.record_outcome <--

对外入口只有四个：
    from memory import extract_from_run, filter_batch, ExperienceStore
    from memory import make_provider

依赖方向：memory -> core / evaluator。反过来不允许，所以这一层
不认识任何具体 Agent、任何具体业务域，也不读任何 Demo 的实现。
"""

from memory.extract import extract_from_run, extract_one, keywords, task_tags
from memory.inject import EXPERIENCE_FOOTER, EXPERIENCE_HEADER
from memory.inject import make_provider, render, render_for_task
from memory.quality import QualityReport, check, filter_batch, holdout_answer_values
from memory.retrieve import Hit, jaccard, retrieve, retrieve_from_store, score_one
from memory.schema import (
    DEFAULT_TTL_DAYS,
    STATUS_ACTIVE,
    STATUS_RETIRED,
    Evidence,
    Experience,
    Trigger,
    make_exp_id,
)
from memory.store import ExperienceStore

__all__ = [
    "Experience",
    "Trigger",
    "Evidence",
    "make_exp_id",
    "STATUS_ACTIVE",
    "STATUS_RETIRED",
    "DEFAULT_TTL_DAYS",
    "extract_from_run",
    "extract_one",
    "keywords",
    "task_tags",
    "check",
    "filter_batch",
    "holdout_answer_values",
    "QualityReport",
    "ExperienceStore",
    "retrieve",
    "retrieve_from_store",
    "score_one",
    "jaccard",
    "Hit",
    "render",
    "render_for_task",
    "make_provider",
    "EXPERIENCE_HEADER",
    "EXPERIENCE_FOOTER",
]
