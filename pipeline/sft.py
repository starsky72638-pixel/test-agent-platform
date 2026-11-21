"""SFT 数据的质量闸门：去重、冲突隔离、split 隔离、schema 统一、质量报告。

为什么单独一层，而不在 distill 里顺手做掉：
    distill 只看得见本次跑的这一批，它的去重是按"一条轨迹的完整指纹"算的。
    但重复与冲突是**跨批次**才出现的——两批样本各自内部都干净，合到一起
    才有"同一句话两个答案"。判断必须发生在合并这一层，否则永远发现不了。

四条规则，都是"宁可少收，不可错收"：

1. 同一输入 + 同一答案 -> 只留一条。它们信息量相同，多留只会给这道题加权。
2. 同一输入 + 不同答案 -> 全部隔离，记冲突。不随机挑一条：
   挑中的那条会被当成"标准答案"，而实际上我们并不知道哪条对。
   不同 Agent 的差异也走这条路——那是待判断的差异，不是可删的重复。
3. holdout 不进训练集。数量少不是并入的理由；少就少，训练集缺样本是
   另一个问题，不能靠污染验证集来补。
4. split / agent / task_id 缺失一律写 unknown 并计数；unknown 默认不进训练集。
   猜一个 fit 出来，等于把来路不明的数据混进训练。

输出三份，互相不重叠：train（可训练）、holdout（只评估）、quarantine（待处理）。
每条被排除的样本都带 reason，报告里另有一份按原因聚合的计数——不允许静默丢弃。
"""

from __future__ import annotations

import hashlib
import json
import re

SCHEMA_VERSION = "1.0"

# 训练侧 / 评估侧 / 未知。accept 两种写法：distill 产出 fit/holdout，人工数据可能写 train/test。
TRAIN_SPLITS = ("fit", "train")
EVAL_SPLITS = ("holdout", "test")
UNKNOWN = "unknown"

_WS = re.compile(r"\s+")


def norm_text(text) -> str:
    """规范化：去空白、去首尾、转小写。

    只做这三样。再往下做（去标点、同义词）就开始猜了：
    把"已发货"和"已发出"当同一句，会把一次真实的表述差异抹平。
    """
    return _WS.sub("", str(text or "")).strip().lower()


def _messages(raw: dict) -> list:
    value = (raw or {}).get("messages")
    return list(value) if isinstance(value, list) else []


def user_turns(messages) -> list[str]:
    return [
        str(m.get("content") or "")
        for m in messages
        if isinstance(m, dict) and m.get("role") == "user"
    ]


def final_answer(messages) -> str:
    """最后一条有内容的 assistant 消息。带 tool_calls 的那条 content 是空的，不算。"""
    for m in reversed(list(messages)):
        if isinstance(m, dict) and m.get("role") == "assistant":
            if str(m.get("content") or "").strip():
                return str(m["content"])
    return ""


def input_key(messages) -> str:
    """输入指纹：所有 user 轮拼起来规范化后取哈希。多轮也按整串比。"""
    joined = "\n".join(user_turns(messages))
    return hashlib.sha1(norm_text(joined).encode("utf-8")).hexdigest()[:16]


def answer_key(messages) -> str:
    return hashlib.sha1(norm_text(final_answer(messages)).encode("utf-8")).hexdigest()[:16]


def flatten_tool_calls(messages) -> list[dict]:
    """把 messages 里的 tool_calls 摊平成 [{name, arguments}]。

    arguments 统一成字符串：模型侧和工具侧本来就都按 JSON 字符串传递，
    这里转成 dict 再序列化会让"原样保留"这件事失真。
    """
    calls: list[dict] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        for item in m.get("tool_calls") or []:
            fn = (item or {}).get("function") or {}
            args = fn.get("arguments")
            if not isinstance(args, str):
                args = json.dumps(args if args is not None else {}, ensure_ascii=False)
            calls.append({"name": str(fn.get("name") or ""), "arguments": args})
    return calls


def _call_ids(messages) -> tuple[list[str], list[str]]:
    """(调用侧 id 列表, 响应侧 tool_call_id 列表)。没有 id 的数据两边都是空串。"""
    call_ids: list[str] = []
    resp_ids: list[str] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        for item in m.get("tool_calls") or []:
            call_ids.append(str((item or {}).get("id") or ""))
        if m.get("role") == "tool":
            resp_ids.append(str(m.get("tool_call_id") or ""))
    return call_ids, resp_ids


def tool_structure_errors(messages, calls) -> list[str]:
    """tool_calls 的结构问题。返回问题列表，空列表表示没发现问题。

    配对方式有两种，不能混着判：
    - OpenAI 风格：assistant.tool_calls[].id 与 tool.tool_call_id 对应，tool 消息没有 name；
    - distill 产出的精简风格：没有 id，只能按 name 顺序对。
    只按 name 判会把第一种全部误判成"调了 A 回了 B"（实测踩过：17 条全红）。
    """
    errors: list[str] = []

    if any(not c["name"] for c in calls):
        errors.append("tool_call_missing_name")

    for c in calls:
        try:
            parsed = json.loads(c["arguments"])
        except (TypeError, ValueError):
            errors.append("tool_call_bad_arguments")
            break
        if not isinstance(parsed, dict):
            errors.append("tool_call_bad_arguments")
            break

    # 一次调用必须有一条对应的 tool 响应，顺序也要对得上。
    # 少了响应，样本教的是"调完不用看结果"，那是最坏的一种示范。
    called = [c["name"] for c in calls]
    if called:
        call_ids, resp_ids = _call_ids(messages)
        if any(call_ids) or any(resp_ids):
            if not all(call_ids) or resp_ids != call_ids:
                errors.append("tool_response_mismatch")
        else:
            answered = [
                str(m.get("name") or "")
                for m in messages
                if isinstance(m, dict) and m.get("role") == "tool"
            ]
            if answered != called:
                errors.append("tool_response_mismatch")

    return errors


def normalize_sample(raw: dict, *, source: dict | None = None) -> tuple[dict, list[str]]:
    """统一外部元信息。messages 原样保留，一个字段都不改。

    拿不到的字段写 unknown / null，不猜——猜出来的 agent 会让
    "这条经验是谁的"永远查不清。
    """
    raw = raw or {}
    messages = _messages(raw)

    issues: list[str] = []
    if not messages:
        issues.append("no_messages")

    calls = flatten_tool_calls(messages)
    issues.extend(tool_structure_errors(messages, calls))

    if messages and not final_answer(messages).strip():
        issues.append("no_final_answer")

    task_id = str(raw.get("task_id") or "").strip() or UNKNOWN
    agent = str(raw.get("agent") or "").strip() or UNKNOWN
    split = str(raw.get("split") or "").strip().lower() or UNKNOWN

    sample_id = str(raw.get("sample_id") or raw.get("id") or "").strip()
    if not sample_id:
        sample_id = f"sft-{input_key(messages)}"

    missing = [
        name
        for name, value in (("task_id", task_id), ("agent", agent), ("split", split))
        if value == UNKNOWN
    ]

    sample = {
        "sample_id": sample_id,
        "task_id": task_id,
        "agent": agent,
        "split": split,
        "source_trace": raw.get("source_trace") or None,
        "tool_calls": calls,
        "messages": messages,
        "missing_fields": missing,
        "source": dict(source or {}),
    }
    return sample, issues


def _source_order(sample: dict) -> tuple:
    """canonical 的唯一选取依据：源文件路径、行号，最后拿 sample_id 兜底。

    刻意不看答案长度、不问模型偏好、不掷随机数——那些都不可复现，
    也不代表质量。文件与行号是"数据里真实存在的先后"，谁都能复核。
    """
    source = sample.get("source") or {}
    return (
        str(source.get("file") or ""),
        int(source.get("line") or 0),
        sample["sample_id"],
    )


def _row_ref(sample: dict) -> dict:
    """报告里的单条索引，够人直接翻到源文件第几行核对。"""
    source = sample.get("source") or {}
    return {
        "sample_id": sample["sample_id"],
        "task_id": sample["task_id"],
        "agent": sample["agent"],
        "split": sample["split"],
        "file": source.get("file"),
        "line": source.get("line"),
    }


def _resolve_pool(
    pool: list[dict],
    label: str,
    quarantine_row,
    groups: list[dict],
    reruns: list[dict],
) -> list[dict]:
    """一个池子内的分类，返回可保留的样本（每个输入至多一条 canonical）。

    四类判定，按证据强弱从上往下，先信任最强的一条：

    1. duplicate      —— 答案完全一致。同一句话留一份，其余记 duplicate。
    2. rerun_variance —— 答案不同，但来源三元组 (task_id, agent, split) 只有一种。
                         这是同一次实验的两次运行，留第一条为 canonical，
                         其余记为 rerun（带 canonical_sample_id），不删原样本。
    3. unresolved     —— 来源里出现 unknown，无法判断谁和谁同源，不猜。
    4. true_conflict  —— 其余：多来源之间答案不一致（不同 Agent、不同 task_id、
                         手工集与生成集），全部隔离等人工裁决。

    判定依据是 task_id/agent/split 三个字段，不是输入文本相似度：
    相似度只能说明"像不像"，说明不了"是不是同一来源"。
    """
    buckets: dict[str, list[dict]] = {}
    for sample in pool:
        buckets.setdefault(input_key(sample["messages"]), []).append(sample)

    kept: list[dict] = []
    for key, rows in sorted(buckets.items()):
        rows = sorted(rows, key=_source_order)
        answers = {answer_key(r["messages"]) for r in rows}
        origins = {(r["task_id"], r["agent"]) for r in rows}
        unknown_origin = any(
            r["task_id"] == UNKNOWN or r["agent"] == UNKNOWN for r in rows
        )

        entry = {
            "input_key": key,
            "split": label,
            "answers": len(answers),
            "rows": [_row_ref(r) for r in rows],
        }

        if len(answers) == 1:
            canonical = rows[0]
            kept.append(canonical)
            for extra in rows[1:]:
                quarantine_row(
                    extra,
                    "duplicate",
                    {
                        "split": label,
                        "input_key": key,
                        "canonical_sample_id": canonical["sample_id"],
                    },
                )
            if len(rows) > 1:
                entry.update(
                    kind="duplicate", canonical_sample_id=canonical["sample_id"]
                )
                groups.append(entry)
            continue

        if len(origins) == 1 and not unknown_origin:
            canonical = rows[0]
            kept.append(canonical)
            rerun_ids: list[str] = []
            for extra in rows[1:]:
                quarantine_row(
                    extra,
                    "rerun_variance",
                    {
                        "split": label,
                        "input_key": key,
                        "canonical_sample_id": canonical["sample_id"],
                    },
                )
                rerun_ids.append(extra["sample_id"])
            entry.update(
                kind="rerun_variance", canonical_sample_id=canonical["sample_id"]
            )
            groups.append(entry)
            reruns.append(
                {
                    "input_key": key,
                    "split": label,
                    "canonical_sample_id": canonical["sample_id"],
                    "rerun_sample_ids": rerun_ids,
                }
            )
            continue

        kind = "unresolved" if unknown_origin else "true_conflict"
        for row in rows:
            quarantine_row(row, kind, {"split": label, "input_key": key})
        entry.update(kind=kind, canonical_sample_id=None)
        groups.append(entry)

    return kept


def build(items) -> dict:
    """items 是 [(raw_sample, source_dict), ...]，返回三份输出 + 报告。

    纯函数：不读盘、不写盘、不打印。所有判断都能被单测逐条钉住。

    顺序是刻意的：结构校验 -> split 分流 -> 池内去重/冲突。
    split 必须在冲突之前：unknown 样本不许进训练集，也就没有资格
    把同问句的 fit 样本一起拖进隔离（这一版之前就是这么错的）。
    """
    prepared: list[tuple[dict, list[str]]] = []
    for raw, source in items:
        prepared.append(normalize_sample(raw, source=source))

    good = [(s, i) for s, i in prepared if not i]
    bad = [(s, i) for s, i in prepared if i]

    missing_fields = sum(1 for s, _ in prepared if s["missing_fields"])
    bad_tool_calls = sum(
        1
        for _, issues in prepared
        if any(x.startswith("tool_call") or x == "tool_response_mismatch" for x in issues)
    )

    quarantine: list[dict] = []
    reasons: dict[str, int] = {}

    def quarantine_row(sample: dict, reason: str, extra: dict | None = None) -> None:
        row = dict(sample)
        row["reason"] = reason
        if extra:
            row.update(extra)
        quarantine.append(row)
        reasons[reason] = reasons.get(reason, 0) + 1

    for sample, issues in bad:
        quarantine_row(sample, issues[0], {"issues": list(issues)})

    # ---- 先按 split 分流 ----
    train_pool: list[dict] = []
    eval_pool: list[dict] = []
    for sample, _ in good:
        split = sample["split"]
        if split in TRAIN_SPLITS:
            train_pool.append(sample)
        elif split in EVAL_SPLITS:
            eval_pool.append(sample)
        else:
            quarantine_row(sample, "unknown_split")

    # ---- 再在各池内部判同源重跑 / 冲突 ----
    groups: list[dict] = []
    reruns: list[dict] = []
    train = _resolve_pool(train_pool, "train", quarantine_row, groups, reruns)
    holdout = _resolve_pool(eval_pool, "holdout", quarantine_row, groups, reruns)

    # 输出顺序固定，两次运行结果逐字节一致
    groups.sort(key=lambda g: (g["split"], g["kind"], g["input_key"]))
    reruns.sort(key=lambda r: (r["split"], r["input_key"]))
    train.sort(key=lambda r: r["sample_id"])
    holdout.sort(key=lambda r: r["sample_id"])
    quarantine.sort(key=lambda r: (r["reason"], r["sample_id"]))

    def kind_groups(kind: str) -> int:
        return sum(1 for g in groups if g["kind"] == kind)

    def kind_rows(kind: str) -> int:
        return sum(len(g["rows"]) for g in groups if g["kind"] == kind)

    counts = {
        "input_samples": len(prepared),
        "valid_samples": len(good),
        "duplicate_samples": reasons.get("duplicate", 0),
        "rerun_variance_groups": kind_groups("rerun_variance"),
        "rerun_variance_samples": kind_rows("rerun_variance"),
        "true_conflict_groups": kind_groups("true_conflict"),
        "true_conflict_samples": kind_rows("true_conflict"),
        "unresolved_groups": kind_groups("unresolved"),
        "unresolved_samples": kind_rows("unresolved"),
        "canonical_samples": len(train) + len(holdout),
        "train_samples": len(train),
        "holdout_samples": len(holdout),
        "unknown_samples": reasons.get("unknown_split", 0),
        "isolation_samples": len(quarantine),
        "missing_field_samples": missing_fields,
        "bad_tool_calls": bad_tool_calls,
        "trainable_samples": len(train),
    }

    return {
        "train": train,
        "holdout": holdout,
        "quarantine": quarantine,
        "report": {
            "schema_version": SCHEMA_VERSION,
            "counts": counts,
            "reasons": dict(sorted(reasons.items())),
            "groups": groups,
            "reruns": reruns,
        },
    }


def read_jsonl(path):
    """读 jsonl，跳过空行。行号从 1 开始，用于 source 定位。"""
    rows = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rows.append((json.loads(line), lineno))
    return rows
