"""判定层：五层判定、事实溯源、归因。

对外只需要两个入口：
    from evaluator import evaluate
    verdict = evaluate(trace, task)

约束：不依赖任何 Agent 框架、任何业务实现。换一个领域、换一个策略的 Agent，
这一层一行都不用改——它只看轨迹和任务，不看是谁跑出来的。
"""

from evaluator.claims import Claim, Evidence, resolve, resolve_all, ungrounded_values
from evaluator.legacy import trace_from_case, trace_from_report
from evaluator.matching import (
    NON_ANSWER_PATTERNS,
    REGEX_PREFIX,
    looks_non_answer,
    looks_off_topic,
    matches,
    same,
)
from evaluator.verdict import (
    FAILURE_PRIORITY,
    RoundVerdict,
    Verdict,
    calls_of_turn,
    evaluate,
    evaluate_all,
    judge_round,
)

__all__ = [
    "Claim",
    "Evidence",
    "resolve",
    "resolve_all",
    "ungrounded_values",
    "matches",
    "same",
    "looks_non_answer",
    "looks_off_topic",
    "NON_ANSWER_PATTERNS",
    "REGEX_PREFIX",
    "Verdict",
    "RoundVerdict",
    "evaluate",
    "evaluate_all",
    "judge_round",
    "calls_of_turn",
    "FAILURE_PRIORITY",
    "trace_from_case",
    "trace_from_report",
]
