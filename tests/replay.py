from __future__ import annotations

# =========================
# 一、导入需要的工具
# =========================

import contextlib       # 临时改变程序的运行环境，例如屏蔽某些输出
import io               # 创建内存中的文本缓冲区
import json             # 读取和保存 JSON 数据
import re               # 从工具返回里抽取字段值，判断回答有没有用到数据
import sys              # 获取命令行参数，以及退出程序
import time             # 计算程序运行时间
import uuid             # 生成唯一编号
from collections import Counter  # 统计各种失败原因出现的次数
from pathlib import Path          # 处理文件路径

import httpx             # 发送 HTTP 请求，检查 Mock Server 是否启动

# 允许直接运行：python tests/replay.py
# 脚本模式下 sys.path[0] 是 tests/，项目根不在里面，下面的 from config 会导入失败。
# -m tests.replay 与 pytest 场景下 __package__ 非空，这里不会重复插入。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 项目配置：项目根目录、创建模型、读取配置
from config import PROJECT_ROOT, build_llm, load_settings

# Agent 的实际执行入口
from agents.main_agent.core.agent import run_agent

# 商城 Agent 的系统提示词、工具注册和工具列表
from agents.main_agent.tools import (
    SHOP_SYSTEM_PROMPT,
    get_registry,
    register_default_tools,
)


# =========================
# 二、全局配置
# =========================

# Mock Server 的地址。
# replay 会通过这个地址检查服务是否启动，并重置服务端数据。
SHOP_BASE = "http://127.0.0.1:8000"

# 默认读取的考试题目文件。
# 如果运行时没有指定其他 JSONL 文件，就使用它。
DEFAULT_GOLDEN = PROJECT_ROOT / "data" / "shop_eval_multi.jsonl"

# 一条考试题至少要有一个评分要求。
# 如果一条题没有任何评分要求，就无法判断 Agent 是否做对。
ASSERT_KEYS = (
    "expect_tools",
    "expect_args",
    "expect_order",
    "max_calls",
    "must_have",
    "must_have_any",
    "must_not_have",
    "grounded_claims",
    "forbidden_tools",
)

# 出现这些错误关键词时，优先认为是网络或服务链路故障。
# 这类故障不应该直接算成模型能力差。
INFRA_HINTS = (
    "Connection",
    "Timeout",
    "RateLimit",
    "APIConnection",
    "InternalServer",
    "502",
    "503",
)


# =========================
# 三、检查考试题目是否合法
# =========================

def validate_case(case: dict) -> str:
    """
    检查一条考试题是否写完整。

    返回空字符串：说明题目合法。
    返回错误文字：说明题目缺少必要信息，不能正常评分。
    """

    # 每道题都必须有编号，方便定位失败用例。
    if not case.get("id"):
        return "缺少 id"

    # 支持两种题目格式：
    # 1. 单轮：input
    # 2. 多轮：turns
    turns = case.get("turns") or (
        [case["input"]] if case.get("input") else []
    )

    if not turns:
        return "缺少 input / turns"

    # 如果题目配置了逐轮评分标准，就检查每轮是否都有评分要求。
    checks = case.get("turn_checks")

    if checks:
        # 评分标准的数量必须与对话轮数一致。
        if len(checks) != len(turns):
            return (
                f"turn_checks {len(checks)} 条，"
                f"与 turns {len(turns)} 条不一致"
            )

        # 检查每一轮是否至少有一项评分要求。
        empty = [
            i
            for i, c in enumerate(checks, 1)
            if not any(c.get(k) for k in ASSERT_KEYS)
        ]

        if empty:
            return f"第 {empty} 轮 turn_checks 没有任何断言"

        return ""

    # 如果没有逐轮评分标准，就检查整道题有没有评分要求。
    if not any(case.get(k) for k in ASSERT_KEYS):
        return "整条用例没有任何断言字段"

    return ""


# =========================
# 四、读取考试题目
# =========================

def load_cases(path: Path) -> list[dict]:
    """
    从 JSONL 文件中读取所有考试题。

    JSONL 的特点：
    每一行都是一个独立的 JSON 对象。
    """

    cases: list[dict] = []

    # 逐行读取文件。
    for no, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        line = line.strip()

        # 空行不参与读取。
        if not line:
            continue

        try:
            # 把 JSON 字符串转换成 Python 字典。
            cases.append(json.loads(line))

        except json.JSONDecodeError as exc:
            # 如果某一行 JSON 格式错误，指出具体行号。
            raise ValueError(
                f"{path.name} 第 {no} 行不是合法 JSON：{exc}"
            ) from exc

    # 检查读取出来的每一道题是否有完整的评分标准。
    bad = [
        (c.get("id") or "?", validate_case(c))
        for c in cases
    ]

    # 只保留检查不通过的题目。
    bad = [
        (cid, why)
        for cid, why in bad
        if why
    ]

    if bad:
        detail = "\n".join(
            f"    {cid}: {why}"
            for cid, why in bad
        )

        raise ValueError(
            f"{path.name} 有 {len(bad)} 条用例不可判分，先修用例：\n"
            f"{detail}"
        )

    return cases


# =========================
# 五、确定读取哪份考试题
# =========================

def golden_path(argv: list[str] | None = None) -> Path:
    """
    决定本次评测使用哪份 JSONL 文件。

    运行：
        python replay.py
    使用默认 shop_eval.jsonl。

    运行：
        python replay.py data/shop_eval_hard.jsonl
    使用指定的 shop_eval_hard.jsonl。

    把参数读取放在函数里，避免导入 replay.py 时就执行参数解析。
    """

    # 如果外部没有传入参数，就读取当前程序的命令行参数。
    args = sys.argv if argv is None else argv

    # 如果用户指定了一个以 .jsonl 结尾的文件，就使用它。
    if len(args) > 1 and args[1].endswith(".jsonl"):
        return Path(args[1])

    # 否则使用默认考试题目。
    return DEFAULT_GOLDEN


# =========================
# 六、检查 Mock Server 是否启动
# =========================

def probe_env() -> None:
    """
    检查商城 Mock Server 是否能正常连接。

    如果服务没启动，就直接报错。
    避免把服务器连接失败误认为 Agent 能力不足。
    """

    try:
        # 请求商品搜索接口，检查服务是否能连接。
        httpx.get(
            f"{SHOP_BASE}/products/search",
            timeout=2.0,
        )

        # 重置 Mock Server 数据。
        # 让每次评测尽量从相同的初始状态开始。
        httpx.post(
            f"{SHOP_BASE}/admin/reset",
            timeout=2.0,
        )

    except httpx.HTTPError as exc:
        raise SystemExit(
            f"[错误] 连不上被测环境 {SHOP_BASE} "
            f"（{type(exc).__name__}）\n"
            f"       先另开一个窗口执行：python mock_server.py"
        ) from exc


# =========================
# 七、比较工具参数
# =========================

def same(a, b) -> bool:
    """
    比较两个参数是否相等。

    例如：
    200 和 200.0 认为相等。
    1002 和 "1002" 也认为相等。

    这样可以避免因为数字和字符串类型不同而误判。
    """

    try:
        return float(a) == float(b)

    except (TypeError, ValueError):
        return str(a) == str(b)


# =========================
# 八、收集 Agent 的工具调用和工具返回结果
# =========================

def collect_outputs(
    messages,
) -> tuple[list[dict], dict[str, str], list[dict]]:
    """
    从 Agent 的对话记录中提取工具调用信息。

    返回三个结果：

    calls：
        Agent 调用了什么工具、传了什么参数。

    outputs：
        每种工具返回了什么内容，按工具名称合并。

    tool_results：
        每一次工具调用及其返回结果的一对一记录。

    为什么要保留 tool_call_id？
    因为同一个工具可能被调用多次。
    需要知道每次调用具体对应哪次返回结果。
    """

    # 用工具调用 ID 保存每次调用的完整记录。
    id2rec: dict[str, dict] = {}

    # 保存工具调用列表，后面检查工具选择和参数时使用。
    calls: list[dict] = []

    # 第一次遍历：收集 Agent 发起的工具调用。
    for m in messages:

        # 一条 AI 消息可能包含多个工具调用。
        for c in (getattr(m, "tool_calls", None) or []):

            rec = {
                "call_id": c["id"],
                "name": c["name"],
                "args": c["args"],
                "output": "",
            }

            # 先建立调用记录，返回结果稍后再填入。
            id2rec[c["id"]] = rec

            calls.append({
                "name": c["name"],
                "args": c["args"],
            })

    # 第二次遍历：找到工具执行后的返回消息。
    for m in messages:

        if getattr(m, "type", "") == "tool":

            # 根据 tool_call_id 找到对应的工具调用记录。
            rec = id2rec.get(m.tool_call_id)

            if rec is not None:
                # 把工具返回内容放回对应的调用记录。
                rec["output"] = str(m.content)

    # 保留每一次调用与返回结果的一对一关系。
    tool_results = list(id2rec.values())

    # 同时按工具名称汇总返回结果。
    buckets: dict[str, list[str]] = {}

    for r in tool_results:
        buckets.setdefault(r["name"], []).append(r["output"])

    outputs = {
        k: "\n".join(v)
        for k, v in buckets.items()
    }

    return calls, outputs, tool_results


# =========================
# 九、检查回答中的事实是否有工具依据
# =========================

def is_grounded(
    claim: dict,
    answer: str,
    outputs: dict[str, str],
) -> bool:
    """
    检查回答中某个指定事实是否有工具返回作为依据。

    例如：
    Agent 回答价格 199 元。
    如果工具返回里也有 199，就认为这个指定事实有依据。

    注意：
    这里不是检查回答是否包含这个事实。
    回答没提到它时，本函数返回 True。
    是否必须提到，由 must_have 等规则负责检查。
    """

    # 取出要检查的事实值。
    value = str(claim.get("value", ""))

    # 如果 Agent 没有提到这个值，本项不负责判它缺失。
    if value not in answer:
        return True

    # 找到这个事实要求对应的来源工具。
    raw = outputs.get(claim.get("from", ""), "")

    # 回答提到了这个值，就要求指定工具确实返回过这个值。
    return bool(raw) and value in raw


def looks_off_topic(answer: str, tool_results: list[dict]) -> bool:
    """
    判断回答是不是根本没在用工具返回的数据（答非所问）。

    只在"断言已经判失败"的前提下调用，用来区分两种完全不同的失败：

    1. 查到了数据，回答里一个字段都没引用 —— 答非所问（off_topic）
    2. 引用了数据，但引错或缺了 —— 缺事实 / 错事实

    判据故意保守：只要回答里出现了返回中的任意一个值，
    就认为它用到了数据，不判 off_topic。
    这样"已发货""899 元"这类正常回答不会被误伤。
    """

    # 把本轮所有工具返回拼成一段文本。
    blob = "\n".join(
        str(t.get("output") or "")
        for t in tool_results
    )

    if not blob:
        return False

    # 从返回的 JSON 里抽出所有字符串值和数字值。
    # 字符串既包含字段名也包含字段值，对"有没有引用"这个判断已经够用。
    # 数字只取两位以上：像 "total": 2 这种单字符数字在回答里太容易误命中。
    values = set(re.findall(r'"([^"]{2,40})"', blob))
    values |= set(re.findall(r":\s*(-?\d{2,}(?:\.\d+)?)", blob))

    # 价格这类值在返回里是 159.0，回答里常写成 159，两种都算引用到了。
    values |= {
        v.split(".")[0]
        for v in values
        if re.fullmatch(r"-?\d+(?:\.\d+)?", v)
    }

    return not any(v in answer for v in values)


# =========================
# 断言匹配：子串 or 正则
# =========================

# 断言字段里以这个前缀开头的字符串，按正则匹配，其余一律按子串匹配。
REGEX_PREFIX = "re:"


def hit(pattern: str, text: str) -> bool:
    """
    判断回答里有没有命中某条断言。

    默认按子串匹配；以 "re:" 开头的按正则匹配。

    为什么要加正则：
        否定表达是开放集合——"查不到 / 不存在 / 没有这个订单 / 查无此单 / not found"，
        词表永远补不全。H13 就因为答案从"查不到"换成"没有这个订单号"被误判过一次。
        写成 "re:(查不到|不存在|没有(这个|该)?订单|not ?found)" 一条覆盖所有变体。
    """

    if not isinstance(pattern, str):
        return False

    if pattern.startswith(REGEX_PREFIX):

        body = pattern[len(REGEX_PREFIX):]

        try:
            return re.search(body, text) is not None

        except re.error:
            # 正则写错不能把整场评测带崩：退回按原文子串匹配。
            # 最坏结果只是这条断言查得不准，不会把正确回答判成错误。
            return body in text

    return pattern in text


# 拒答模板：模型怀疑自己被注入，于是拒绝执行任务。
# 这类回答的共性是"不复述任何业务数据，只解释自己为什么不干"。
NON_ANSWER_PATTERNS = (
    "prompt injection",
    "embedded instructions",
    "i can't follow",
    "i cannot follow",
    "i won't act on",
    "i will not follow",
    "无法遵从消息中的指令",
    "疑似提示词注入",
)


def looks_non_answer(answer: str) -> bool:
    """
    判断这条回答是不是"根本没在答这道题"。

    与 off_topic 的区别：
        off_topic 是"答了别的东西"，例如问价格却聊人生；
        non_answer 是"压根没答"，例如直接拒绝执行、或者用外文回一段说明。

    只认两种很窄的情况，避免误伤正常回答：

    1. 命中拒答模板（模型声称用户消息里有注入指令，拒绝执行）；
    2. 几乎全是外文：中文字符占比不到 20%。这套 golden 全部是中文提问，
       系统提示也要求中文回答，长段英文本身就是违规。

    为什么单独归一类：
        这类回答会被 must_have 判失败，落进 wrong_fact，
        看起来像"答错了内容"，实际是"一句业务数据都没给"。
        归因错了，后面改提示词的方向也会跟着错。
    """

    text = (answer or "").strip()

    if not text:
        return False

    low = text.lower()

    if any(p in low for p in NON_ANSWER_PATTERNS):
        return True

    # 短答不判：正常回答里也可能夹几个英文单词（如 delivered）。
    if len(text) < 30:
        return False

    zh = len(re.findall(r"[\u4e00-\u9fff]", text))

    return zh / len(text) < 0.2


# =========================
# 十、把对话记录转换成训练数据格式
# =========================

def to_train_messages(messages) -> list[dict]:
    """
    把 Agent 的 LangChain 对话记录转换成训练数据格式。

    例如：
    LangChain 的 human 对应训练数据里的 user。
    LangChain 的 ai 对应训练数据里的 assistant。

    转换后的数据可以进一步用于整理 SFT 训练样本。
    但转换完成不代表已经进行了模型训练。
    """

    # LangChain 消息类型与训练数据角色的对应关系。
    role_map = {
        "system": "system",
        "human": "user",
        "ai": "assistant",
        "tool": "tool",
    }

    out: list[dict] = []

    for m in messages:

        # 先保存角色和文本内容。
        item = {
            "role": role_map.get(
                getattr(m, "type", ""),
                "user",
            ),
            "content": m.content or "",
        }

        # 如果 AI 消息中包含工具调用，也要转换并保存。
        tcs = getattr(m, "tool_calls", None) or []

        if tcs:
            item["tool_calls"] = [
                {
                    "id": c["id"],
                    "type": "function",
                    "function": {
                        "name": c["name"],
                        "arguments": json.dumps(
                            c["args"],
                            ensure_ascii=False,
                        ),
                    },
                }
                for c in tcs
            ]

        # 工具返回消息要保留它对应的工具调用 ID。
        if getattr(m, "type", "") == "tool":
            item["tool_call_id"] = m.tool_call_id

        out.append(item)

    return out


# =========================
# 十一、对 Agent 的表现进行评分
# =========================

def score(
    case: dict,
    calls: list[dict],
    answer: str,
    outputs: dict[str, str],
    error: str = "",
    allowed_tools: set[str] | None = None,
) -> dict:
    """
    给 Agent 的一条回答或一轮对话打分。

    主要检查五个方面：

    1. 工具选择：有没有调用要求的工具。
    2. 参数正确：工具参数有没有填对。
    3. 调用顺序：该先做的有没有先做（例如退款前先查订单）。
    4. 事实依据：回答中的指定事实有没有工具依据。
    5. 回答内容：有没有包含必需内容、出现禁止内容。

    如果发生执行错误、回答为空或调用未知工具，
    会优先归类为对应的失败类型。
    """

    # 默认评分结果。
    # 如果某一项没有通过，就在这个基础上更新结果。
    blank = {
        "tool_ok": False,
        "tool_msg": "",
        "param_ok": False,
        "param_msg": "",
        "order_ok": False,
        "order_msg": "",
        "ground_ok": False,
        "ungrounded": [],
        "answer_ok": False,
        "missing_fact": [],
        "missing_any": [],
        "forbidden_hit": [],
        "forbidden_tools": [],
        "invalid_tools": [],
        # 拒绝执行 / 外文作答：单独打标，报告里能看出"没答"而不是"答错"。
        "non_answer": False,
        # 调用次数超限：只上报，不影响 passed（正确性优先于效率）。
        "over_calls": [],
        "passed": False,
        "failure_type": "",
    }

    # 如果 Agent 执行时发生错误，就不再检查后面的评分项。
    if error:
        return {
            **blank,
            "tool_msg": "未执行",
            "param_msg": "未执行",
            "failure_type": "infra_error",
        }

    # 如果 Agent 没有返回任何回答，直接判定失败。
    if not str(answer or "").strip():
        return {
            **blank,
            "tool_msg": "无回答",
            "param_msg": "无回答",
            "failure_type": "empty_answer",
        }

    # 提取 Agent 实际调用过的工具名称。
    names = [c["name"] for c in calls]

    # 检查 Agent 是否调用了不存在或不允许的工具。
    unknown = [
        n
        for n in names
        if allowed_tools is not None and n not in allowed_tools
    ]

    if unknown:
        return {
            **blank,
            "invalid_tools": unknown,
            "tool_msg": f"未知工具 {','.join(unknown)}",
            "param_msg": "未校验",
            "failure_type": "invalid_tool",
        }

    # 检查这一轮禁止调用的工具。
    # 与 invalid_tool 的区别：那些工具存在、平时也能用，
    # 只是这一轮（例如用户已改口说不退）不允许调。
    # 只认真正发起过的调用，回答里写了什么不算。
    banned = [
        t
        for t in (case.get("forbidden_tools") or [])
        if t in names
    ]

    if banned:
        return {
            **blank,
            "forbidden_tools": banned,
            "tool_msg": f"禁止调用 {','.join(banned)}",
            "param_msg": "未校验",
            "failure_type": "forbidden_action",
        }

    # ---------- 评分项 1：工具选择 ----------

    expect = case.get("expect_tools")

    if expect is None:
        # 如果题目没有要求具体工具，就不检查这一项。
        tool_ok, tool_msg = True, "未校验"

    else:
        # 检查要求的工具有没有出现在实际调用列表中。
        missing = [
            t
            for t in expect
            if t not in names
        ]

        tool_ok = not missing

        tool_msg = (
            f"缺 {','.join(missing)}"
            if missing
            else "命中"
        )

    # ---------- 评分项 2：工具参数 ----------

    expect_args = case.get("expect_args") or {}

    if not expect_args:
        # 没有配置预期参数，就不检查这一项。
        param_ok, param_msg = True, "未校验"

    else:
        # 每个预期参数都要能在某次实际工具调用中找到对应值。
        bad = [
            f"{k}={v}"
            for k, v in expect_args.items()
            if not any(
                same(c["args"].get(k), v)
                for c in calls
            )
        ]

        param_ok = not bad

        param_msg = (
            f"错 {','.join(bad)}"
            if bad
            else "命中"
        )

    # ---------- 评分项 3：调用顺序 ----------

    # expect_order 写的是"必须按这个先后出现"的工具名，
    # 中间插入别的工具没关系（查物流、查商品都算正常动作），
    # 但这一串的先后不能乱。例：["get_order", "apply_refund"]
    # 意思是退款前必须先查订单，先退后查就算违规。
    expect_order = case.get("expect_order") or []

    if not expect_order:
        order_ok, order_msg = True, "未校验"

    else:
        # 按实际调用先后走一遍，看能不能对上这个子序列。
        pos = 0

        for name in names:

            if pos < len(expect_order) and name == expect_order[pos]:
                pos += 1

        order_ok = pos == len(expect_order)

        if order_ok:
            order_msg = "命中"

        elif pos == 0:
            # 第一个该调的工具压根没出现，说"顺序错"会误导。
            order_msg = f"顺序错，{expect_order[0]} 一次都没调"

        else:
            # pos 就是"对到第几个断了"，用它说清是哪个工具来早了。
            order_msg = f"顺序错，{expect_order[pos]} 没在 {expect_order[pos - 1]} 之后"

    # ---------- 评分项 4：事实依据 ----------

    claims = case.get("grounded_claims") or []

    # 找出回答中提到了、但指定工具返回结果里找不到的事实。
    ungrounded = [
        c["value"]
        for c in claims
        if not is_grounded(c, answer, outputs)
    ]

    ground_ok = not ungrounded

    # ---------- 评分项 5：最终回答内容 ----------

    # must_have：这些内容必须全部出现在回答中。
    miss = [
        s
        for s in case.get("must_have", [])
        if not hit(s, answer)
    ]

    # must_have_any：每组备选内容至少要出现一个。
    any_miss = [
        g
        for g in case.get("must_have_any", [])
        if not any(hit(s, answer) for s in g)
    ]

    # must_not_have：这些内容不允许出现在回答中。
    forbid = [
        s
        for s in case.get("must_not_have", [])
        if hit(s, answer)
    ]

    answer_ok = not (miss or any_miss or forbid)

    # ---------- 评分项 6：调用次数（只上报，不判失败） ----------

    # max_calls 写法：{"get_order": 1}，表示这一轮里同一工具最多调 1 次。
    # 刻意不参与 passed：正确性优先于效率。
    # "答对了但多查两遍"该优化，"答错了"该修——两者混在一起会把真问题盖掉。
    limits = case.get("max_calls") or {}

    over_calls = [
        [name, names.count(name), int(limit)]
        for name, limit in limits.items()
        if names.count(name) > int(limit)
    ]

    # 这一轮整体是否通过。下面两处都用它，避免同一个条件写两遍、改一处漏一处。
    round_ok = tool_ok and param_ok and order_ok and answer_ok and ground_ok

    # 拒答 / 外文作答：只要这一轮整体没过，就标出来。
    # 不能只看 answer_ok——像"既没调工具、又回了整段英文拒绝"这种，
    # 因为用例没配 must_have，answer_ok 反而是 True，标记就漏了。
    # 反过来说，整体通过的轮次一律不标，避免正常回答夹几个英文单词被打标。
    non_answer = bool(not round_ok and looks_non_answer(answer))

    # ---------- 汇总评分结果 ----------

    # 五项全部通过，才算这轮通过。
    if round_ok:
        ftype = ""

    # 按优先级确定本轮失败原因。
    elif ungrounded:
        ftype = "ungrounded"

    elif forbid:
        ftype = "hallucination"

    elif not tool_ok:
        ftype = "missing_tool"

    # 顺序排在参数前面：先动手后核实属于流程违规，
    # 哪怕参数后来都补对了，动作本身已经发出去了。
    elif not order_ok:
        ftype = "wrong_order"

    elif not param_ok:
        ftype = "wrong_arg"

    # 排在 wrong_fact 前面：工具和参数都对，但回答不是"答错"而是"没答"。
    # 这两种失败的修法完全不同，必须分开记。
    elif non_answer:
        ftype = "non_answer"

    else:
        ftype = "wrong_fact"

    return {
        "tool_ok": tool_ok,
        "tool_msg": tool_msg,
        "param_ok": param_ok,
        "param_msg": param_msg,
        "order_ok": order_ok,
        "order_msg": order_msg,
        "ground_ok": ground_ok,
        "ungrounded": ungrounded,
        "answer_ok": answer_ok,
        "missing_fact": miss,
        "missing_any": [g[0] for g in any_miss],
        "forbidden_hit": forbid,
        "non_answer": non_answer,
        "over_calls": over_calls,
        "invalid_tools": [],
        "passed": ftype == "",
        "failure_type": ftype,
    }


# =========================
# 十二、让 Agent 实际完成一道考试题
# =========================

def run_case(
    llm,
    tools,
    case: dict,
    retries: int = 1,
) -> tuple[list[dict], list, str]:
    """
    把一道题交给 Agent 执行，并记录每一轮的过程。

    返回：
        rounds：每一轮的回答、工具调用和工具返回。
        messages：完整的对话历史。
        error：执行失败时的错误信息。

    retries=1 表示第一次失败后，符合重试条件时最多再试一次。
    """

    # 获取题目中的多轮对话。
    # 如果没有 turns，就使用单轮 input。
    turns = case.get("turns") or [case["input"]]

    last = ""

    # 根据 retries 设置最多执行次数。
    for attempt in range(retries + 1):

        try:
            # history 保存 Agent 的完整对话历史。
            history = None

            # 记录上一次处理结束时的消息数量。
            # 用于区分每轮新增的消息。
            prev_len = 0

            rounds: list[dict] = []

            # 暂时屏蔽 Agent 运行时的标准输出，
            # 避免大量中间信息干扰 replay 的评测报告。
            with contextlib.redirect_stdout(io.StringIO()):

                # 逐轮把用户输入交给 Agent。
                for text in turns:
                    result = run_agent(
                        llm,
                        tools,
                        SHOP_SYSTEM_PROMPT,
                        text,
                        history=history,
                    )

                    # 回答和历史都从结构化结果里取。
                    answer = result["answer"]
                    history = result["messages"]
                    # 只拿当前这一轮新产生的消息。
                    new_msgs = history[prev_len:]

                    # 更新消息数量，供下一轮使用。
                    prev_len = len(history)

                    # 从本轮消息中提取工具调用和工具返回结果。
                    calls, outputs, trs = collect_outputs(new_msgs)

                    # 保存本轮的完整评测记录。
                    rounds.append({
                        "turn": len(rounds) + 1,
                        "input": text,
                        "answer": answer,
                        "calls": calls,
                        "outputs": outputs,
                        "tool_results": trs,
                        "msg_count": prev_len,
                        "status": result["status"],
                        "steps": result["steps"],
                        "sent_msgs": result.get("sent_msgs", 0),
                    })

            return rounds, history or [], ""

        except Exception as exc:

            # 保存异常类型和错误内容。
            last = f"{type(exc).__name__}: {exc}"

            # 只有看起来像链路故障时，才等待后重试。
            if (
                attempt < retries
                and any(k in last for k in INFRA_HINTS)
            ):
                time.sleep(3)
                continue

            # 执行失败时，把错误信息交给上层处理。
            return [], [], last

    return [], [], last


# =========================
# 十三、对一道题的所有轮次进行评分和汇总
# =========================

def evaluate_case(
    case: dict,
    rounds: list[dict],
    err: str,
    allowed_tools: set[str],
) -> dict:
    """
    汇总一道题所有轮次的评分结果。

    多轮对话不能只检查最后一轮。
    例如第一轮应该搜索却没有搜索，就应该在第一轮判失败。
    """

    # 每次执行生成一个短 ID，方便区分不同的执行轨迹。
    trace_id = uuid.uuid4().hex[:12]

    # 获取题目配置的逐轮评分标准。
    checks = case.get("turn_checks") or []

    judged: list[dict] = []

    # ---------- 如果 Agent 执行失败，直接记录失败 ----------

    # 原代码接收了 err，却没有使用它。
    # 这会导致 rounds 为空时，后面可能错误地把题目判为 PASS。
    if err:
        return {
            "trace_id": trace_id,
            "rounds": [],
            "answer": "",
            "calls": [],
            "outputs": {},
            "tool_results": [],
            "msg_count": 0,
            "passed": False,
            "failure_type": "infra_error",
            "fail_turn": None,
            "tool_ok": False,
            "param_ok": False,
            "order_ok": False,
            "ground_ok": False,
            "answer_ok": False,
            "tool_msg": "未执行",
            "param_msg": "未执行",
            "order_msg": "未执行",
            "ungrounded": [],
            "forbidden_hit": [],
            "forbidden_tools": [],
            "missing_fact": [],
            "missing_any": [],
            "invalid_tools": [],
            "over_calls": [],
        }

    # 没有执行出任何轮次，也不能当成通过。
    if not rounds:
        return {
            "trace_id": trace_id,
            "rounds": [],
            "answer": "",
            "calls": [],
            "outputs": {},
            "tool_results": [],
            "msg_count": 0,
            "passed": False,
            "failure_type": "no_execution",
            "fail_turn": None,
            "tool_ok": False,
            "param_ok": False,
            "order_ok": False,
            "ground_ok": False,
            "answer_ok": False,
            "tool_msg": "没有执行轮次",
            "param_msg": "没有执行轮次",
            "order_msg": "没有执行轮次",
            "ungrounded": [],
            "forbidden_hit": [],
            "forbidden_tools": [],
            "missing_fact": [],
            "missing_any": [],
            "invalid_tools": [],
            "over_calls": [],
        }

    # ---------- 逐轮判分 ----------

    # 累积"截至当前轮为止"的工具返回，专门给「依据溯源」用。
    #
    # 为什么只有这一层要跨轮：
    # 用户在第二轮引用第一轮查到的价格是合法的，
    # 但如果溯源也只翻本轮的返回，就会把这些正确引用判成"无依据"。
    #
    # 为什么工具选择和参数不能跨轮：
    # 第二轮该查却没查，必须判缺失；若改成"整个会话查过就算"，
    # 那第二轮什么都不做也会因为第一轮调过而通过，逐轮判定就失效了。
    seen_outputs: dict[str, str] = {}

    for i, r in enumerate(rounds):

        for k, v in (r.get("outputs") or {}).items():
            seen_outputs[k] = (
                f"{seen_outputs[k]}\n{v}" if k in seen_outputs else v
            )

        if checks:
            # 如果有 turn_checks，就使用对应轮次的评分标准。
            ck = checks[i]

        elif i == len(rounds) - 1:
            # 如果没有逐轮标准，顶层评分标准只检查最后一轮。
            ck = case

        else:
            # 其他轮次没有独立评分标准，不参与判分。
            ck = None

        if ck is None:
            judged.append({
                **r,
                "judged": False,
                "passed": None,
                "failure_type": "",
            })
            continue

        # 使用 score() 检查当前轮次的表现。
        # 依据溯源用累积返回（seen_outputs），其余三项在 score 内部只看本轮 calls。
        s = score(
            ck,
            r["calls"],
            r["answer"],
            seen_outputs,
            allowed_tools=allowed_tools,
        )

        # 步数耗尽说明这一轮根本没得出结论，
        # 直接记成 max_steps，而不是按回答内容猜是缺事实还是空回答。
        if r.get("status") == "max_steps":
            s = {
                **s,
                "passed": False,
                "failure_type": "max_steps",
                "answer_ok": False,
            }

        # 查到了数据，回答里却一个字段都没用上。
        # 这种情况按"缺事实"记录会把真因藏起来（例如拿到订单却回了一句问候），
        # 所以单独归为 off_topic。只在断言已经判失败时才检查，避免误伤正常回答。
        elif (
            not s["answer_ok"]
            # 拒答/外文作答比 off_topic 更具体：两种都会表现为"没引用数据"，
            # 但前者是模型拒绝执行，后者是答了别的，不能互相覆盖。
            # 这里看 non_answer 字段而不是 failure_type：即使本轮因缺工具先归了
            # missing_tool，只要答话本身是拒答，也不该再被标成"答错题"。
            and not s.get("non_answer")
            and s["failure_type"]
            not in {
                "infra_error",
                "empty_answer",
                "invalid_tool",
                "forbidden_action",
            }
            and r.get("tool_results")
            and looks_off_topic(r["answer"], r["tool_results"])
        ):
            s = {
                **s,
                "failure_type": "off_topic",
            }

        # 保存当前轮次的原始记录和评分结果。
        judged.append({
            **r,
            "judged": True,
            **s,
        })

    # 只取真正参与评分的轮次。
    graded = [
        j
        for j in judged
        if j["judged"]
    ]

    # 找出所有没有通过的轮次。
    failed = [
        j
        for j in graded
        if not j["passed"]
    ]

    # 记录最早失败的轮次，作为这道题的主要失败原因。
    first = failed[0] if failed else None

    # ---------- 合并所有轮次的工具返回 ----------

    merged: dict[str, str] = {}

    for r in rounds:
        for k, v in r["outputs"].items():

            if k in merged:
                merged[k] = f"{merged[k]}\n{v}"

            else:
                merged[k] = v

    # ---------- 定义两个汇总小工具 ----------

    def agg(key: str) -> bool:
        """
        汇总某项评分是否全部通过。

        例如：
        所有被评分的轮次 tool_ok 都为 True，
        那么最终 tool_ok 就为 True。
        """
        return all(j[key] for j in graded) if graded else False

    def union(key: str) -> list:
        """
        把每轮产生的同类问题合并成一个列表。

        例如把各轮的无依据事实合并起来。
        """
        return [
            x
            for j in graded
            for x in (j.get(key) or [])
        ]

    # 没有任何轮次真正参与评分时，也不能判定为通过。
    no_judged_turn = not graded

    return {
        "trace_id": trace_id,

        # 保存每轮的详细评分。
        "rounds": judged,

        # 最后一轮的回答。
        "answer": rounds[-1]["answer"] if rounds else "",

        # 合并所有轮次的工具调用。
        "calls": [
            c
            for r in rounds
            for c in r["calls"]
        ],

        # 合并所有轮次的工具返回。
        "outputs": merged,

        # 保留每次工具调用与返回的一对一记录。
        "tool_results": [
            t
            for r in rounds
            for t in r["tool_results"]
        ],

        # 最后一轮结束时的消息总数。
        "msg_count": rounds[-1]["msg_count"] if rounds else 0,

        # 必须至少有一轮参与评分，且没有失败，才能通过。
        "passed": bool(graded) and first is None,

        # 如果没有可评分轮次，单独标记出来。
        "failure_type": (
            first["failure_type"]
            if first
            else ("no_judged_turn" if no_judged_turn else "")
        ),

        # 第一次失败发生在哪一轮。
        "fail_turn": first["turn"] if first else None,

        # 汇总四项评分。
        "tool_ok": agg("tool_ok"),
        "param_ok": agg("param_ok"),
        "order_ok": agg("order_ok"),
        "ground_ok": agg("ground_ok"),
        "answer_ok": agg("answer_ok"),

        # 汇总工具和参数错误说明。
        "tool_msg": "; ".join(
            j["tool_msg"]
            for j in graded
            if j["tool_msg"] and not j["tool_ok"]
        ),

        "param_msg": "; ".join(
            j["param_msg"]
            for j in graded
            if j["param_msg"] and not j["param_ok"]
        ),

        "order_msg": "; ".join(
            j["order_msg"]
            for j in graded
            if j["order_msg"] and not j["order_ok"]
        ),

        # 汇总各类回答问题。
        "ungrounded": union("ungrounded"),
        "forbidden_hit": union("forbidden_hit"),
        "forbidden_tools": union("forbidden_tools"),
        "missing_fact": union("missing_fact"),
        "missing_any": union("missing_any"),
        "invalid_tools": union("invalid_tools"),

        # 调用次数超限（只上报，不影响 passed）。
        # 每项带上轮次：[轮次, 工具名, 实际次数, 上限]，方便定位是哪一轮在重复查。
        "over_calls": [
            [j["turn"], *oc]
            for j in graded
            for oc in (j.get("over_calls") or [])
        ],
    }


# =========================
# 十四、计算某项评分的通过率
# =========================

def pct(rows: list[dict], key: str) -> str:
    """
    计算某个评分指标的通过率。

    例如：
    20 道有效题中有 15 道通过，
    就显示 15/20 (75%)。

    infra_error 是链路故障，不计入能力评分的分母。
    """

    # 过滤掉基础设施故障的题目。
    valid = [
        r
        for r in rows
        if r["failure_type"] != "infra_error"
    ]

    n = len(valid)

    # 统计指定指标为 True 的题目数量。
    ok = sum(
        1
        for r in valid
        if r[key]
    )

    if not n:
        return "0/0"

    return f"{ok}/{n} ({round(100 * ok / n)}%)"

# =========================
# 十五、
# =========================
def pct_ground(rows: list[dict]) -> str:
    """只统计真正配了 grounded_claims 的题目。

    没配溯源断言的题目 ground_ok 恒为 True，算进分母会得到虚高的 100%。
    """
    valid = [r for r in rows if r["failure_type"] != "infra_error"]
    covered = [r for r in valid if r.get("has_claims")]

    if not covered:
        return f"0/0 覆盖 0/{len(valid)}"

    ok = sum(1 for r in covered if r["ground_ok"])
    return (
        f"{ok}/{len(covered)} ({round(100 * ok / len(covered))}%) "
        f"覆盖 {len(covered)}/{len(valid)}"
    )

# =========================
# 十六、主程序：执行完整的 Agent 评测
# =========================

def main() -> int:
    """
    整个 replay.py 的执行入口。

    执行顺序：

    1. 确定考试题目文件。
    2. 检查 Mock Server。
    3. 加载模型和工具。
    4. 读取考试题目。
    5. 逐题运行 Agent。
    6. 对 Agent 的表现评分。
    7. 输出统计结果。
    8. 保存 JSON 报告。
    """

    # ---------- 第一步：确定考试题目 ----------

    golden = golden_path()

    if not golden.exists():
        print(f"[错误] 找不到用例集：{golden}")
        return 1

    # 报告文件名跟随考试题目文件名。
    # 例如 shop_eval_hard.jsonl 对应 shop_eval_hard.json。
    report_file = (
        PROJECT_ROOT
        / "reports"
        / (golden.stem + ".json")
    )

    # ---------- 第二步：检查 Mock Server ----------

    probe_env()

    # ---------- 第三步：加载模型 ----------

    settings = load_settings()

    # 创建 LLM，关闭流式输出，并将温度设置为 0。
    # 温度为 0 是为了尽量减少重复评测时的随机差异。
    llm = build_llm(
        settings,
        streaming=False,
    ).bind(temperature=0)

    # ---------- 第四步：注册 Agent 工具 ----------

    register_default_tools()

    # 获取当前注册的所有工具。
    tools = get_registry().list()

    # 读取所有考试题目。
    cases = load_cases(golden)

    # 记录所有允许调用的工具名称。
    allowed = {
        t.name
        for t in tools
    }

    print(
        f"Golden Set {len(cases)} 条 | "
        f"模型 {settings.model} | "
        f"温度 0 | "
        f"工具 {len(tools)} 个"
    )

    print(
        "启动时已自动 reset 服务端状态，"
        "S13/S14 的退款链路可复现\n"
    )

    # ---------- 第五步：逐题运行 Agent ----------

    results: list[dict] = []

    # 记录整个评测的开始时间。
    started = time.perf_counter()

    for i, case in enumerate(cases, 1):

        # 记录当前题目开始时间。
        t0 = time.perf_counter()

        # 让 Agent 完成当前题目。
        rounds, messages, err = run_case(
            llm,
            tools,
            case,
        )

        # 计算当前题目耗时，单位为毫秒。
        elapsed = int(
            (time.perf_counter() - t0) * 1000
        )

        # 对当前题目进行评分和汇总。
        ev = evaluate_case(
            case,
            rounds,
            err,
            allowed,
        )

        # 组合当前题目的完整结果。
        row = {
            "id": case["id"],
            "input": case["input"],
            "turns": case.get("turns") or [case["input"]],
            "note": case.get("note", ""),
            "elapsed_ms": elapsed,
            "error": err,
            # 本题是否配了溯源断言，只有配了才进依据可溯的分母
            "has_claims": any(
                c.get("grounded_claims")
                for c in (case.get("turn_checks") or [case])
            ),

            # 把完整对话转换成可用于后续训练整理的数据格式。
            "train_messages": (
                to_train_messages(messages)
                if messages
                else []
            ),

            # 加入本题的评分和执行记录。
            **ev,
        }

        results.append(row)

        # 在终端显示当前题目的通过或失败状态。
        flag = (
            "PASS"
            if row["passed"]
            else f"FAIL {row['failure_type']}"
        )

        turn = (
            f" @turn{row['fail_turn']}"
            if row["fail_turn"]
            else ""
        )

        print(
            f"[{i:02d}/{len(cases)}] "
            f"{row['id']} "
            f"{flag + turn:<26}"
            f"{elapsed:>6}ms  "
            f"tools={[c['name'] for c in row['calls']]}"
        )

    # ---------- 第六步：统计整个评测结果 ----------

    # 计算整个评测的耗时，单位为秒。
    total = int(
        time.perf_counter() - started
    )

    # 单独统计基础设施故障。
    infra = [
        r
        for r in results
        if r["failure_type"] == "infra_error"
    ]

    # 统计真正的 Agent 能力失败。
    # 基础设施故障不算在这里。
    failures = [
        r
        for r in results
        if (
            not r["passed"]
            and r["failure_type"] != "infra_error"
        )
    ]

    # 统计每种失败原因出现的次数。
    types = Counter(
        r["failure_type"]
        for r in failures
    )

    # 有效题目 = 总题目 - 基础设施故障。
    valid = [
        r
        for r in results
        if r["failure_type"] != "infra_error"
    ]

    # 计算有效题目的平均耗时。
    avg_ms = (
        round(
            sum(r["elapsed_ms"] for r in valid)
            / len(valid)
        )
        if valid
        else 0
    )

    # 计算每道有效题平均调用多少次工具。
    avg_calls = (
        round(
            sum(len(r["calls"]) for r in valid)
            / len(valid),
            1,
        )
        if valid
        else 0
    )

    # 统计调用次数超限的地方（只上报，不影响判定）。
    over = [
        (r["id"], *oc)
        for r in results
        for oc in (r.get("over_calls") or [])
    ]

    # ---------- 第七步：打印评测总报告 ----------

    print("\n" + "=" * 74)

    print(
        f"回放完毕：{len(results)} 条，有效 {len(valid)} 条"
        f"（剔除链路故障 {len(infra)} 条），"
        f"总耗时 {total}s，"
        f"平均 {avg_ms}ms/条，"
        f"平均调用 {avg_calls} 次"
    )

    print(
        f"通过      {pct(results, 'passed')}     "
        f"工具选择 {pct(results, 'tool_ok')}"
    )

    print(
        f"参数      {pct(results, 'param_ok')}     "
        f"依据可溯 {pct_ground(results)}"
    )

    print(
        f"顺序      {pct(results, 'order_ok')}     "
        f"终态事实  {pct(results, 'answer_ok')}"
    )

    print(
        f"重复调用  {len(over)} 处（只上报，不影响判定）"
    )

    print("=" * 74)

    # ---------- 打印调用次数超限明细 ----------

    if over:
        print("\n重复调用（同一轮内同一工具查了多次）：")

        for cid, turn, name, actual, limit in over[:12]:
            print(
                f"  {cid} 第 {turn} 轮  "
                f"{name} ×{actual}（上限 {limit}）"
            )

    # ---------- 第八步：打印基础设施故障 ----------

    if infra:

        print(
            f"\n链路故障（不计入成绩，建议重跑）："
            f"{[r['id'] for r in infra]}"
        )

        for r in infra:
            print(
                f"  {r['id']}  {r['error'][:90]}"
            )

    # ---------- 第九步：打印 Agent 失败明细 ----------

    if failures:

        print("\n失败明细：")

        for r in failures:

            print(
                f"  {r['id']} "
                f"[{r['failure_type']}] "
                f"{r['input']}"
            )

            print(
                f"     工具："
                f"{[c['name'] for c in r['calls']]}  "
                f"{r['tool_msg']} / {r['param_msg']}"
                + (f" / {r['order_msg']}" if not r["order_ok"] else "")
            )

            print(
                f"     回答：{r['answer'][:110]}"
            )

            if r["ungrounded"]:
                print(
                    f"     无依据断言：{r['ungrounded']}"
                )

            if r["forbidden_hit"]:
                print(
                    f"     编造命中：{r['forbidden_hit']}"
                )

            if r["missing_fact"] or r["missing_any"]:
                print(
                    f"     缺失事实："
                    f"{r['missing_fact'] + r['missing_any']}"
                )

        # 汇总不同失败原因出现的次数。
        print(
            "\n归因："
            + "  ".join(
                f"{k} {v}"
                for k, v in types.most_common()
            )
        )

    elif not infra:

        print(
            "\n全部通过——看看是不是 golden 写得太松了，"
            "加两条边界再跑。"
        )

    # ---------- 第十步：保存 JSON 评测报告 ----------

    # 如果 reports 文件夹不存在，就自动创建。
    report_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 组装最终报告。
    report = {
        "model": settings.model,

        # 记录本次实际使用的考试题目文件名。
        "golden": golden.name,

        # 保存总体统计数据。
        "summary": {
            "total": len(results),
            "valid": len(valid),
            "infra_error": len(infra),
            "passed": len(valid) - len(failures),
            "avg_ms": avg_ms,
            "avg_calls": avg_calls,
            "failure_types": dict(types),

            # 保存发生失败的题目编号。
            "fail_turns": [
                r["id"]
                for r in failures
                if r["fail_turn"]
            ],
        },

        # 保存每一道题的详细执行和评分记录。
        "results": results,
    }

    # 将 Python 字典转换成 JSON，并写入报告文件。
    report_file.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"\n报告已写入：{report_file}"
    )

    return 0


# =========================
# 十六、程序入口
# =========================

# 只有直接运行 replay.py 时，才启动完整评测。
# 如果其他文件 import replay.py，只会加载函数，不会自动开始考试。
if __name__ == "__main__":
    sys.exit(main())