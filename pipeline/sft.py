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


def _split_conflicts(
    pool: list[dict],
    label: str,
    quarantine_row,
    conflicts: list[dict],
) -> list[dict]:
    """在一个池子里做"同输入"的去重与冲突判定，返回可保留的样本。

    必须按池子分开做，不能把所有样本混在一起判：
    unknown 来源的样本本来就不许进训练集，它没有资格否掉一条 fit 样本；
    同理 holdout 的冲突只影响评估集，不该把训练样本一起拖走。
    """
    groups: dict[str, list[dict]] = {}
    for sample in pool:
        groups.setdefault(input_key(sample["messages"]), []).append(sample)

    kept: list[dict] = []
    for key, rows in sorted(groups.items()):
        answers = {answer_key(r["messages"]) for r in rows}
        if len(answers) > 1:
            ids = sorted(r["sample_id"] for r in rows)
            conflicts.append(
                {
                    "input_key": key,
                    "split": label,
                    "answers": len(answers),
                    "sample_ids": ids,
                    "files": sorted({str(r["source"].get("file") or UNKNOWN) for r in rows}),
                }
            )
            for row in rows:
                quarantine_row(row, "answer_conflict", {"input_key": key, "split": label})
            continue

        # 同一答案只留一条：按 sample_id 排序取第一条，保证两次运行结果一致
        rows = sorted(rows, key=lambda r: r["sample_id"])
        kept.append(rows[0])
        for extra in rows[1:]:
            quarantine_row(extra, "exact_duplicate", {"split": label})
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

    # ---- 再在各池内部去重 / 判冲突 ----
    conflicts: list[dict] = []
    train = _split_conflicts(train_pool, "train", quarantine_row, conflicts)
    holdout = _split_conflicts(eval_pool, "holdout", quarantine_row, conflicts)

    # 输出顺序固定，两次运行结果逐字节一致
    train.sort(key=lambda r: r["sample_id"])
    holdout.sort(key=lambda r: r["sample_id"])
    quarantine.sort(key=lambda r: (r["reason"], r["sample_id"]))

    counts = {
        "input_samples": len(prepared),
        "valid_samples": len(good),
        "exact_duplicates": reasons.get("exact_duplicate", 0),
        "conflicting_inputs": len(conflicts),
        "conflict_samples": reasons.get("answer_conflict", 0),
        "train_samples": len(train),
        "holdout_samples": len(holdout),
        "unknown_samples": reasons.get("unknown_split", 0),
        "missing_field_samples": missing_fields,
        "bad_tool_calls": bad_tool_calls,
        "quarantined_samples": len(quarantine),
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
            "conflicts": conflicts,
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
