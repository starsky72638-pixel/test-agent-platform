"""从已有轨迹构造偏好对（chosen / rejected）。

**它不做什么**：不调模型、不生成新文本、不做训练、不改原始样本。
只在已存在的样本之间配对——这是它能被复现、且不会夹带模型幻觉的原因。

配对前提（三个都必须满足）：
1. 同一 `task_id`
2. 同一 `agent`
3. 同一 `split`（不跨 split 配对，否则验证集泄漏进训练）

再加一条硬校验：两侧的**用户轮序列必须逐字一致**，否则不是同一道题。

为什么默认不跨 Agent 配对：
`wf:canvas-demo` 挂 `missing_tool` 是因为**画布没画那个节点**，不是模型不肯调。
拿它做 rejected，等于教模型"去调一个不存在的工具"——那是有害样本。
跨 Agent 的差异属于**配置差异**，不是**模型行为差异**；数量记进报告，不默认进 pairs。

`failure_type` 保留在 pair 上，用于按失败类型分层统计。
"""

from __future__ import annotations

import hashlib

SCHEMA_VERSION = "1.0"

# 两侧的标记。配对只看"谁通过、谁失败"，不关心具体判定细节——细节在 failure_type 里。
PASS = "pass"
FAIL = "fail"


def _norm(text: str) -> str:
    return " ".join(str(text or "").split())


def answer_of(messages: list) -> str:
    """最后一轮 assistant 的正文。空串表示这条样本没有可用回答。"""
    for msg in reversed(messages or []):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            return str(msg.get("content") or "")
    return ""


def user_turns(messages: list) -> list[str]:
    return [
        str(m.get("content") or "")
        for m in messages or []
        if isinstance(m, dict) and m.get("role") == "user"
    ]


class Candidate:
    """一条可参与配对的样本。source 用于追溯（文件 + 行号）。"""

    __slots__ = ("task_id", "agent", "split", "messages", "source", "outcome", "failure_type", "source_trace")

    def __init__(
        self,
        *,
        task_id: str,
        agent: str,
        split: str,
        messages: list[dict],
        source: dict,
        outcome: str,
        failure_type: str = "",
        source_trace: str | None = None,
    ) -> None:
        self.task_id = str(task_id or "")
        self.agent = str(agent or "")
        self.split = str(split or "")
        self.messages = list(messages or [])
        self.source = dict(source or {})
        self.outcome = outcome
        self.failure_type = str(failure_type or "")
        self.source_trace = source_trace

    @property
    def sample_key(self) -> str:
        return f"{self.source.get('file')}:{self.source.get('line')}"

    def order_key(self) -> tuple:
        # 稳定取第一条：文件路径 → 行号 → task_id。不看答案长度、不掷随机数。
        return (
            str(self.source.get("file") or ""),
            int(self.source.get("line") or 0),
            self.task_id,
        )


def _pair_id(task_id: str, agent: str, chosen: Candidate, rejected: Candidate) -> str:
    raw = "|".join(
        [
            task_id,
            agent,
            str(chosen.source.get("file")),
            str(chosen.source.get("line")),
            str(rejected.source.get("file")),
            str(rejected.source.get("line")),
        ]
    )
    return "pref-" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def make_pair(
    task_id: str,
    chosen: Candidate,
    rejected: Candidate,
    basis: str,
    *,
    verified: bool = True,
) -> dict:
    return {
        "pair_id": _pair_id(task_id, chosen.agent or rejected.agent, chosen, rejected),
        "task_id": task_id,
        "agent": chosen.agent or rejected.agent,
        "agent_verified": bool(verified),
        "split": chosen.split,
        "failure_type": rejected.failure_type or "",
        "basis": basis,
        "prompt": user_turns(chosen.messages),
        "chosen": {
            "answer": answer_of(chosen.messages),
            "messages": chosen.messages,
            "source": dict(chosen.source),
            "source_trace": chosen.source_trace,
        },
        "rejected": {
            "answer": answer_of(rejected.messages),
            "messages": rejected.messages,
            "source": dict(rejected.source),
            "source_trace": rejected.source_trace,
        },
    }


def _same_prompt(a: Candidate, b: Candidate) -> bool:
    return [_norm(x) for x in user_turns(a.messages)] == [
        _norm(x) for x in user_turns(b.messages)
    ]


def _agent_ok(cand: Candidate, rej: Candidate) -> bool:
    """任一侧 agent 缺失时不阻断配对，但结果会被标成 agent_verified=false。

    历史 failed.jsonl 是 P10-c 之前的产物，没有 agent 字段；
    `distill.py` 现在会写，但重生成会改掉已入库的数据与哈希。
    这里选择"允许配对 + 如实标注"，而不是拿旧字段当不存在。
    """
    if not cand.agent or not rej.agent:
        return True
    return cand.agent == rej.agent


def _try_pair(rej: Candidate, pool: list[Candidate], used: set[str]) -> tuple[Candidate | None, str]:
    """在 pool 里为 rej 找配对；返回 (样本, 失败原因)。"""
    for cand in pool:
        if cand.sample_key in used:
            continue
        if not _agent_ok(cand, rej):
            continue
        if cand.split != rej.split:
            continue
        if not _same_prompt(cand, rej):
            continue
        return cand, ""
    if not pool:
        return None, "no_pass_sample"
    if all(not _agent_ok(c, rej) for c in pool):
        return None, "agent_mismatch"
    if all(c.split != rej.split for c in pool):
        return None, "split_mismatch"
    return None, "prompt_mismatch"


def build(
    passes: list[Candidate],
    fails: list[Candidate],
    *,
    allow_cross_agent: bool = False,
) -> dict:
    """配对并返回 {pairs, report}。纯函数：不读盘、不写盘、不打印。"""
    skipped: dict[str, int] = {}

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    by_task_pass: dict[str, list[Candidate]] = {}
    by_task_fail: dict[str, list[Candidate]] = {}
    for c in passes:
        by_task_pass.setdefault(c.task_id, []).append(c)
    for c in fails:
        by_task_fail.setdefault(c.task_id, []).append(c)
    for bucket in (by_task_pass, by_task_fail):
        for rows in bucket.values():
            rows.sort(key=lambda c: c.order_key())

    pairs: list[dict] = []
    tasks_with_both = 0
    used_chosen: set[str] = set()
    used_rejected: set[str] = set()

    for task_id in sorted(set(by_task_pass) & set(by_task_fail)):
        tasks_with_both += 1
        for rej in by_task_fail[task_id]:
            chosen, why = _try_pair(rej, by_task_pass[task_id], used_chosen)
            if chosen is None:
                skip(why)
                continue
            if _norm(answer_of(chosen.messages)) == _norm(answer_of(rej.messages)):
                skip("same_answer")
                continue
            used_chosen.add(chosen.sample_key)
            used_rejected.add(rej.sample_key)
            verified = bool(chosen.agent and rej.agent)
            pairs.append(
                make_pair(
                    task_id,
                    chosen,
                    rej,
                    basis="same_agent_rerun" if verified else "agent_unverified",
                    verified=verified,
                )
            )

    if allow_cross_agent:
        for task_id in sorted(set(by_task_pass) & set(by_task_fail)):
            for rej in by_task_fail[task_id]:
                if rej.sample_key in used_rejected:
                    continue
                for cand in by_task_pass[task_id]:
                    if cand.split != rej.split or not _same_prompt(cand, rej):
                        continue
                    if _norm(answer_of(cand.messages)) == _norm(answer_of(rej.messages)):
                        continue
                    used_rejected.add(rej.sample_key)
                    pairs.append(make_pair(task_id, cand, rej, basis="cross_agent"))
                    break

    pairs.sort(key=lambda p: p["pair_id"])

    by_failure: dict[str, int] = {}
    for p in pairs:
        key = p["failure_type"] or "unknown"
        by_failure[key] = by_failure.get(key, 0) + 1

    report = {
        "schema_version": SCHEMA_VERSION,
        "counts": {
            "pass_samples": len(passes),
            "fail_samples": len(fails),
            "tasks_with_both": tasks_with_both,
            "pairs": len(pairs),
            "cross_agent_pairs": sum(1 for p in pairs if p["basis"] == "cross_agent"),
            "unverified_agent_pairs": sum(1 for p in pairs if not p["agent_verified"]),
        },
        "by_failure_type": dict(sorted(by_failure.items())),
        "skipped": dict(sorted(skipped.items())),
    }
    return {"pairs": pairs, "report": report}
