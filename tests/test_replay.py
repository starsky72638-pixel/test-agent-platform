
"""
test_replay.py

这个文件不负责真正运行 Agent，也不需要启动 mock_server。

它专门测试 replay.py 里的评分功能是否可靠。

举例：
- 给 score() 一个空回答，看它是否判定失败。
- 给 score() 一个不存在的工具名，看它是否发现。
- 给 evaluate_case() 两轮测试结果，看它是否逐轮评分。

目的：
防止评分器本身有 Bug，导致 Agent 明明做错了却被判定为 PASS。
"""

# 从真正的评分器 replay.py 中导入需要测试的函数。
# 注意：这里导入的是函数，不是 Agent。
from tests.replay import (
    collect_outputs,  # 整理 Agent 的工具调用和工具返回
    evaluate_case,    # 汇总一条测试用例的评分
    hit,              # 断言匹配（子串 / re: 正则）
    looks_non_answer, # 判断回答是不是拒答/外文作答
    pct_ground,       # 计算事实依据检查的覆盖率
    score,            # 判断一轮 Agent 回答是否正确
    validate_case,    # 检查测试用例是否具备评分条件
)

# 事实卡片（memory v1.5）是纯函数，不碰网络也不碰模型，可以直接单测。
import json

from langchain_core.messages import SystemMessage

from agents.main_agent.view.memory import (
    FACTS_KEY,
    MAX_FACTS,
    extract_facts,
    load_facts,
    render_facts,
    save_facts,
    update_facts,
)


# 假设系统允许 Agent 使用的工具。
# 后面会故意传入一个不在这里面的工具名，
# 检查评分器能不能发现 Agent 使用了非法工具。
ALLOWED = {
    "search_products",
    "get_product_detail",
    "get_order",
    "get_logistics",
    "apply_refund",
}


def case(**kw):
    """
    创建一条测试用例。

    为什么要有这个函数？

    因为下面很多测试都需要一条测试用例。
    如果每次都手动写完整的字典，会非常重复。

    所以这里先准备一份默认配置，
    需要测试什么，就通过 kw 覆盖对应字段。

    例如：
    case(must_have=["899"])

    表示创建一条要求 Agent 回答中包含 899 的测试用例。
    """

    base = {
        "id": "T",
        "input": "x",

        # None 表示不检查工具选择
        "expect_tools": None,

        # 空字典表示不检查工具参数
        "expect_args": {},

        # 回答中必须出现的内容
        "must_have": [],

        # 回答中至少要出现每组中的一个内容
        "must_have_any": [],

        # 回答中禁止出现的内容
        "must_not_have": [],
    }

    # 使用传入的配置覆盖默认配置
    base.update(kw)

    return base


def rnd(answer, calls=None, outputs=None, turn=1, msgs=3):
    """
    创建一轮假的 Agent 执行结果。

    注意：这里没有真正调用 Agent！

    我们只是手动制造一份测试数据，
    模拟 Agent 执行完一轮之后留下的记录。

    参数：
    answer：Agent 的回答
    calls：Agent 调用了哪些工具
    outputs：工具返回了什么
    turn：这是第几轮对话
    msgs：当前累计消息数量

    后面测试多轮评分时会用到。
    """

    return {
        "turn": turn,
        "input": "x",
        "answer": answer,
        "calls": calls or [],
        "outputs": outputs or {},
        "tool_results": [],
        "msg_count": msgs,
    }


# ==================================================
# 1. 测试空回答
# ==================================================

def test_empty_answer_fails():
    """
    测试：Agent 只返回空格或换行时，评分器是否判定失败。

    即使 Agent 调用了工具，
    但最后没有给用户任何回答，也不能算完成任务。
    """

    s = score(
        case(must_have=["899"]),
        [{"name": "get_order", "args": {}}],
        "   \n ",  # 故意制造一个空回答
        {},
        "",
        ALLOWED,
    )

    # 必须判定失败
    assert s["passed"] is False

    # 失败原因必须是空回答
    assert s["failure_type"] == "empty_answer"


def test_none_answer_fails():
    """
    测试：如果 Agent 返回 None，而不是字符串，
    评分器是否也能识别为空回答。

    不能因为返回值不是空字符串就误判通过。
    """

    s = score(
        case(must_have=["a"]),
        [],
        None,
        {},
        "",
        ALLOWED,
    )

    assert s["failure_type"] == "empty_answer"


# ==================================================
# 2. 测试用例本身是否合格
# ==================================================

def test_case_without_any_assertion_is_invalid():
    """
    测试：一条考试题没有任何评分标准时，
    评分器是否拒绝这条用例。

    例如：
    用户输入了一个问题，但没有设置：
    - 需要调用什么工具
    - 回答必须包含什么
    - 哪些内容不能出现

    这样的题目没有办法判断对错。
    """

    # 没有任何断言，必须返回错误
    assert validate_case({
        "id": "X",
        "input": "x"
    }) == "整条用例没有任何断言字段"

    # 没有用例 ID，也必须返回错误
    assert validate_case(
        case(id="", input="x")
    ) == "缺少 id"

    # 有 must_have 断言，就是合法用例
    assert validate_case(
        case(must_have=["a"])
    ) == ""


def test_case_without_input_is_invalid():
    """
    测试：用例没有用户输入时，是否会被拒绝。

    没有 input 或 turns，
    Agent 就不知道应该回答什么。
    """

    result = validate_case({
        "id": "X",
        "expect_tools": ["get_order"]
    })

    assert "input" in result


def test_turn_checks_length_mismatch():
    """
    测试：多轮对话数量与评分标准数量是否一致。

    例如：
    对话有两轮，但是只写了一轮评分标准。

    那么第二轮就不知道怎么评分，因此必须报错。
    """

    c = {
        "id": "M",
        "turns": ["a", "b"],

        # 只有一份评分标准，但实际有两轮对话
        "turn_checks": [
            {"must_have": ["x"]}
        ],
    }

    assert "不一致" in validate_case(c)


# ==================================================
# 3. 测试非法工具
# ==================================================

def test_unknown_tool_is_invalid():
    """
    测试：Agent 调用了系统不存在的工具时，
    评分器能不能发现。

    这里故意使用 rm_rf。

    它不在 ALLOWED 允许的工具列表中。
    因此必须被判定为 invalid_tool。
    """

    s = score(
        case(must_have=["ok"]),

        # 故意模拟 Agent 调用了一个不存在的工具
        [{"name": "rm_rf", "args": {}}],

        "ok",
        {},
        "",
        ALLOWED,
    )

    assert s["failure_type"] == "invalid_tool"

    # 还必须记录到底是哪个非法工具
    assert s["invalid_tools"] == ["rm_rf"]


def test_known_tools_are_not_flagged():
    """
    测试：Agent 调用了正常存在的工具时，
    不能错误地把它判定为非法工具。
    """

    s = score(
        case(expect_tools=["get_order"]),

        # get_order 是允许使用的工具
        [{"name": "get_order", "args": {}}],

        "ok",
        {},
        "",
        ALLOWED,
    )

    # 不应该存在非法工具
    assert s["invalid_tools"] == []

    # 这条用例应该通过
    assert s["passed"] is True


# ==================================================
# 4. 测试多轮对话逐轮评分
# ==================================================

def test_multiturn_fails_on_first_turn():
    """
    测试：多轮对话中，第一轮失败了，
    不能因为第二轮回答正确就算整条通过。

    例如：
    第一轮：用户要求查询商品
    Agent：没有搜索，直接回答

    第二轮：用户继续提问
    Agent：回答正确

    第一轮仍然是失败的。
    """

    c = {
        "id": "M",
        "input": "a",
        "turns": ["a", "b"],

        # 每一轮都有独立的评分标准
        "turn_checks": [
            {
                "expect_tools": ["search_products"]
            },
            {
                "must_have": ["y"]
            },
        ],
    }

    # 人工制造两轮执行结果
    rounds = [
        rnd("没有搜", turn=1),
        rnd("有 y", turn=2, msgs=5),
    ]

    ev = evaluate_case(c, rounds, "", ALLOWED)

    # 整条用例必须失败
    assert ev["passed"] is False

    # 必须准确记录第一轮失败
    assert ev["fail_turn"] == 1

    # 第一轮没有调用要求的搜索工具
    assert ev["failure_type"] == "missing_tool"


def test_multiturn_all_pass():
    """
    测试：如果多轮对话每一轮都满足评分标准，
    那么整条用例应该通过。
    """

    c = {
        "id": "M",
        "input": "a",
        "turns": ["a", "b"],
        "turn_checks": [
            {"must_have": ["x"]},
            {"must_have": ["y"]},
        ],
    }

    rounds = [
        rnd("有 x", turn=1),
        rnd("有 y", turn=2, msgs=5),
    ]

    ev = evaluate_case(c, rounds, "", ALLOWED)

    assert ev["passed"] is True

    # 没有失败轮次
    assert ev["fail_turn"] is None


def test_no_turn_checks_judges_last_round_only():
    """
    测试：如果没有为每一轮单独设置评分标准，
    那么顶层的评分标准只检查最后一轮。

    这是当前 replay.py 设计的规则。

    第一轮不参与评分，最后一轮才参与评分。
    """

    c = case(must_have=["y"])

    c["turns"] = ["a", "b"]

    rounds = [
        rnd("完全不相干", turn=1),
        rnd("有 y", turn=2, msgs=5),
    ]

    ev = evaluate_case(c, rounds, "", ALLOWED)

    assert ev["passed"] is True

    # 第一轮没有评分，第二轮有评分
    assert [
        j["judged"] for j in ev["rounds"]
    ] == [False, True]


# ==================================================
# 5. 测试错误归因优先级
# ==================================================

def test_ungrounded_beats_missing_tool():
    """
    测试：Agent 同时犯了多个错误时，
    评分器是否按照预定的优先级归因。

    这里 Agent：
    1. 没有调用 get_logistics
    2. 回答中编出了运单号 SF1

    应该优先判定为 ungrounded（事实没有依据），
    而不是只报告 missing_tool。
    """

    c = case(
        must_have=["899"],

        # SF1 必须来自 get_logistics 工具
        grounded_claims=[
            {
                "value": "SF1",
                "from": "get_logistics"
            }
        ],
    )

    s = score(
        c,

        # Agent 实际只调用了 get_order
        [{"name": "get_order", "args": {}}],

        # 回答中出现了没有来源的 SF1
        "899 运单 SF1",

        # get_order 返回的内容也没有 SF1
        {"get_order": "{}"},

        "",
        ALLOWED,
    )

    assert s["failure_type"] == "ungrounded"

    assert s["ungrounded"] == ["SF1"]


def test_hallucination_beats_wrong_fact():
    """
    测试：如果回答中出现了明确禁止的内容，
    应该优先判定为 hallucination（编造）。

    例如：评分标准禁止承诺明天送达，
    但 Agent 却说了明天送达。
    """

    c = case(
        must_have=["899"],
        must_not_have=["明天送达"],
    )

    s = score(
        c,
        [{"name": "get_order", "args": {}}],
        "明天送达",
        {},
        "",
        ALLOWED,
    )

    assert s["failure_type"] == "hallucination"


def test_infra_error_short_circuits():
    """
    测试：如果执行 Agent 时发生网络/API 故障，
    应该归类为 infra_error。

    不能把 API 断开导致的失败，
    算成 Agent 本身能力不足。
    """

    s = score(
        case(must_have=["a"]),
        [],
        "",
        {},

        # 模拟 API 连接失败
        "OpenAIConnectionError: 502",

        ALLOWED,
    )

    assert s["failure_type"] == "infra_error"

    assert s["passed"] is False


def test_grounded_value_not_mentioned_is_not_ungrounded():
    """
    测试：评分器不能强制 Agent 在回答中提及
    每一个 grounded_claims 中的值。

    这里虽然要求检查 SF1 的来源，
    但是 Agent 的回答根本没有提到 SF1。

    那么就不应该说 Agent 编造了 SF1。

    是否漏答应该由 must_have 等规则判断。
    """

    c = case(
        grounded_claims=[
            {
                "value": "SF1",
                "from": "get_logistics"
            }
        ]
    )

    s = score(
        c,
        [{"name": "get_order", "args": {}}],
        "只说个 899",
        {},
        "",
        ALLOWED,
    )

    assert s["ground_ok"] is True


# ==================================================
# 6. 测试 trace_id
# ==================================================

def test_trace_id_present_and_unique():
    """
    测试：每一次 Agent 测试记录，
    是否都生成了独立的 trace_id。

    trace_id 就像一次考试的编号。

    方便以后：
    - 查找某次 Agent 执行记录
    - 定位某次失败
    - 将测试报告和训练数据对应起来
    """

    c = case(must_have=["a"])

    a = evaluate_case(
        c,
        [rnd("a")],
        "",
        ALLOWED
    )

    b = evaluate_case(
        c,
        [rnd("a")],
        "",
        ALLOWED
    )

    # 编号长度必须为 12
    assert len(a["trace_id"]) == 12

    # 两次执行不能使用相同编号
    assert a["trace_id"] != b["trace_id"]


# ==================================================
# 7. 测试 tool_call_id 配对
# ==================================================

class FakeMsg:
    """
    人工制造一条假的消息。

    为什么需要它？

    collect_outputs() 需要读取消息中的：
    - type：消息类型
    - content：消息内容
    - tool_calls：Agent 发起的工具调用
    - tool_call_id：工具返回对应的调用编号

    这里不需要真的调用 Agent，
    手动创建一个具有相同属性的对象即可。
    """

    def __init__(
        self,
        type_,
        content="",
        tool_calls=None,
        tool_call_id=None
    ):
        self.type = type_
        self.content = content
        self.tool_calls = tool_calls
        self.tool_call_id = tool_call_id


def test_tool_call_id_pairs_in_call_order():
    """
    测试：工具调用和工具返回是否正确配对。

    假设 Agent 调用了两次 get_order：

    第一次：订单 1002
    第二次：订单 1003

    但是工具返回的消息故意颠倒顺序：

    第二次先返回，第一次后返回。

    评分器必须通过 tool_call_id 找到正确的对应关系，
    而不是简单按照消息出现顺序配对。
    """

    msgs = [
        FakeMsg(
            "ai",
            tool_calls=[
                {
                    "id": "c1",
                    "name": "get_order",
                    "args": {"order_id": 1002}
                },
                {
                    "id": "c2",
                    "name": "get_order",
                    "args": {"order_id": 1003}
                },
            ]
        ),

        # 故意让第二次调用先返回
        FakeMsg("tool", "second", tool_call_id="c2"),

        # 第一次调用后返回
        FakeMsg("tool", "first", tool_call_id="c1"),
    ]

    calls, outputs, trs = collect_outputs(msgs)

    # 最终记录必须按照原始调用顺序
    assert [
        t["call_id"] for t in trs
    ] == ["c1", "c2"]

    # 每次调用必须拿到属于自己的返回结果
    assert [
        t["output"] for t in trs
    ] == ["first", "second"]

    # 同名工具的结果会合并，但不能丢失
    assert outputs["get_order"] == "first\nsecond"


def test_same_tool_twice_keeps_both_outputs():
    """
    测试：同一个工具调用两次时，
    是否能保留两次不同的返回结果。

    例如：
    第一次搜索耳机，第二次搜索键盘。

    不能因为工具名字相同，
    就只留下最后一次搜索结果。
    """

    msgs = [
        FakeMsg(
            "ai",
            tool_calls=[
                {
                    "id": "a",
                    "name": "search_products",
                    "args": {"q": "耳机"}
                },
                {
                    "id": "b",
                    "name": "search_products",
                    "args": {"q": "键盘"}
                },
            ]
        ),

        FakeMsg("tool", "结果A", tool_call_id="a"),
        FakeMsg("tool", "结果B", tool_call_id="b"),
    ]

    _, outputs, trs = collect_outputs(msgs)

    # 必须保留两次调用
    assert len(trs) == 2

    # 两次返回结果都不能丢
    assert "结果A" in outputs["search_products"]
    assert "结果B" in outputs["search_products"]


# ==================================================
# 8. 测试 evaluate_case 的失败短路
# ==================================================

def test_infra_error_not_passed_by_evaluate():
    """
    测试：即使没有真正执行 Agent，
    只要存在网络/API 错误，
    evaluate_case 也不能把它判定为通过。
    """

    ev = evaluate_case(
        case(must_have=["a"]),
        [],
        "OpenAIConnectionError: 502",
        ALLOWED
    )

    assert ev["passed"] is False

    assert ev["failure_type"] == "infra_error"


def test_no_rounds_is_not_passed():
    """
    测试：没有任何执行轮次时，不能算通过。

    例如 Agent 根本没有运行起来，
    rounds 是空列表。

    这不是 Agent 通过了考试，
    而是根本没有参加考试。

    所以应该返回 no_execution。
    """

    ev = evaluate_case(
        case(must_have=["a"]),
        [],
        "",
        ALLOWED
    )

    assert ev["passed"] is False

    assert ev["failure_type"] == "no_execution"


def test_pct_ground_only_counts_rows_with_claims():
    """
    测试：计算事实依据通过率时，
    只统计真正设置了 grounded_claims 的用例。

    例如有三条测试：

    第一条：没有要求检查事实来源，不应该计入分母。
    第二条：检查了来源，检查通过。
    第三条：检查了来源，检查失败。

    正确结果：
    1 / 2 = 50%

    覆盖率：
    2 / 3

    也就是说：
    只有两条测试真正检查了事实依据。
    """

    rows = [
        {
            "failure_type": "",
            "has_claims": False,
            "ground_ok": True
        },
        {
            "failure_type": "",
            "has_claims": True,
            "ground_ok": True
        },
        {
            "failure_type": "",
            "has_claims": True,
            "ground_ok": False
        },
    ]

    assert pct_ground(rows) == "1/2 (50%) 覆盖 2/3"


# ==================================================
# 9. 测试事实卡片（memory v1.5）
# ==================================================

def tool_out(obj, code=200):
    """
    伪造一次工具返回。

    格式必须和 shop_skill._request 一致：
    第一行 status=xxx，空行后 body=JSON。
    Agent 实际看到的就是这个字符串，事实卡片也从它里面抽字段。
    """
    return f"status={code}\nbody={json.dumps(obj, ensure_ascii=False)}"


def test_order_output_becomes_fact():
    """
    测试：查订单成功后，能不能把状态和金额记成事实。

    这是最基础的一条：
    下一轮用户问"那个订单多少钱"，即使那轮对话已被裁掉，
    也应该能从事实卡片里拿到 899。
    """

    facts = update_facts({}, "get_order", {"order_id": 1001}, tool_out({
        "order_id": 1001,
        "status": "已签收",
        "product_id": 1,
        "amount": 899.0,
        "created_at": "2026-11-02",
    }))

    assert list(facts) == ["order:1001"]

    assert facts["order:1001"]["订单状态"] == "已签收"

    assert facts["order:1001"]["金额"] == 899.0


def test_error_status_is_not_recorded():
    """
    测试：404 / 422 不能被记成事实。

    404 的意思是"这个订单不存在"，422 是"未支付不能退"。
    这两种都是"查不到"，一旦记进事实表，
    下一轮换个订单号时模型会以为上一条结论还成立。
    """

    not_found = tool_out({"error": "order not found"}, code=404)

    assert extract_facts("get_order", {"order_id": 9999}, not_found) == {}

    unpaid = tool_out({"error": "订单未支付，无法退款"}, code=422)

    assert extract_facts("apply_refund", {"order_id": 1003}, unpaid) == {}


def test_truncated_body_is_not_recorded():
    """
    测试：工具返回被截断时，不能硬解析出半个事实。

    shop_skill 会把超过 800 字的结果截断，
    JSON 就不完整了。这种情况应该放弃记录，而不是瞎猜字段。
    """

    truncated = (
        'status=200\nbody={"total": 4, "items": [{"id": 1, "tit'
        "...(已截断，原始长度 1200 字符)"
    )

    assert extract_facts("search_products", {}, truncated) == {}

    # 超时那种 "ERROR ..." 同样没有 status 行，也不能记
    assert extract_facts("get_order", {}, "ERROR 请求超时（超过 5.0s）") == {}


def test_order_and_logistics_merge_into_one_fact():
    """
    测试：同一个订单查了订单接口又查了物流接口，
    应该合并成一条事实，而不是互相覆盖。

    订单侧有金额，物流侧有运单号，
    两边的字段都要在，否则"1001 花了多少、运单号多少"答不全。
    """

    facts = update_facts({}, "get_order", {"order_id": 1001}, tool_out({
        "order_id": 1001,
        "status": "已签收",
        "amount": 899.0,
        "created_at": "2026-11-02",
    }))

    facts = update_facts(facts, "get_logistics", {"order_id": 1001}, tool_out({
        "order_id": 1001,
        "carrier": "顺丰速运",
        "tracking_no": "SF1234567890",
        "status": "delivered",
        "eta": "2026-11-05",
        "traces": [],
    }))

    # 仍然只有一条
    assert list(facts) == ["order:1001"]

    # 订单侧字段没被物流覆盖
    assert facts["order:1001"]["金额"] == 899.0

    # 物流侧字段也进来了
    assert facts["order:1001"]["运单号"] == "SF1234567890"

    assert facts["order:1001"]["物流状态"] == "delivered"


def test_search_records_one_fact_per_product():
    """
    测试：一次搜索返回多个商品时，
    每个商品各记一条，不能只留最后一个。
    """

    facts = update_facts({}, "search_products", {"q": "耳机"}, tool_out({
        "total": 2,
        "items": [
            {
                "id": 1, "title": "无线降噪耳机 Pro",
                "price": 899.0, "stock": 42, "rating": 4.7
            },
            {
                "id": 6, "title": "蓝牙音箱 Mini",
                "price": 299.0, "stock": 120, "rating": 4.2
            },
        ],
    }))

    assert set(facts) == {"product:1", "product:6"}

    assert facts["product:1"]["库存"] == 42


def test_zero_stock_is_kept():
    """
    测试：库存 0 必须记下来。

    智能手表 S2 的 stock 就是 0，这是"没货要如实说"这条考点的依据。
    如果过滤空值时把 0 一起丢掉，模型就看不到这个 0 了。
    """

    facts = update_facts({}, "search_products", {}, tool_out({
        "total": 1,
        "items": [
            {
                "id": 4, "title": "智能手表 S2",
                "price": 1299.0, "stock": 0, "rating": 4.6
            }
        ],
    }))

    assert facts["product:4"]["库存"] == 0

    assert "库存：0" in render_facts(facts)


def test_facts_evicted_over_limit():
    """
    测试：事实表超过上限时，淘汰最早的那几条。

    不淘汰的话，聊得越久系统提示越长，
    最后提示词比对话还长，成本和注意力都崩。
    """

    facts = {}

    total = MAX_FACTS + 2

    for order_id in range(2001, 2001 + total):
        update_facts(
            facts,
            "get_order",
            {"order_id": order_id},
            tool_out({
                "order_id": order_id,
                "status": "已签收",
                "amount": 1.0,
            }),
        )

    assert len(facts) == MAX_FACTS

    # 最早的两条被淘汰
    assert "order:2001" not in facts

    # 最新的还在
    assert f"order:{2001 + total - 1}" in facts


def test_render_empty_facts_is_blank():
    """
    测试：没有事实时，不要在系统提示里留一段空标题。

    第一轮对话本来就没有任何事实，
    这时候多一段"已知事实："反而误导模型。
    """

    assert render_facts({}) == ""


def test_save_and_load_facts_roundtrip():
    """
    测试：事实表存进系统消息后能原样取回来。

    多轮之间靠 additional_kwargs 传递，
    存了取不到就等于没做记忆。
    """

    msgs = [SystemMessage("sys")]

    save_facts(msgs, {"order:1001": {"订单号": 1001}})

    assert FACTS_KEY in msgs[0].additional_kwargs

    assert load_facts(msgs) == {"order:1001": {"订单号": 1001}}

    # 全新的会话取回来是空字典，不能报错
    assert load_facts([SystemMessage("sys")]) == {}


# ==================================================
# 10. 测试调用顺序判分（expect_order）
# ==================================================

def test_reversed_order_is_caught():
    """
    测试：先提交退款、再回头查订单，必须判 wrong_order。

    这就是 S13 之前的漏洞：
    只查工具名单的话，两个工具都在，会被判成 PASS；
    实际上退款动作已经先发出去了。
    """

    c = case(
        expect_tools=["get_order", "apply_refund"],
        expect_order=["get_order", "apply_refund"],
    )

    s = score(
        c,

        # 顺序故意反了
        [
            {
                "name": "apply_refund",
                "args": {"order_id": 1002, "reason": "有杂音"}
            },
            {
                "name": "get_order",
                "args": {"order_id": 1002}
            },
        ],

        "已为您提交退款申请",
        {},
        "",
        ALLOWED,
    )

    assert s["failure_type"] == "wrong_order"

    assert s["passed"] is False

    # 说明里要点出是谁来早了
    assert "get_order" in s["order_msg"]


def test_extra_calls_between_do_not_break_order():
    """
    测试：中间插了别的工具，只要该先后的一串没乱，就算通过。

    查订单 → 查物流 → 提交退款，这是正常流程，
    不能因为多查了一次物流就判顺序错。
    """

    c = case(expect_order=["get_order", "apply_refund"])

    s = score(
        c,
        [
            {"name": "get_order", "args": {"order_id": 1002}},
            {"name": "get_logistics", "args": {"order_id": 1002}},
            {
                "name": "apply_refund",
                "args": {"order_id": 1002, "reason": "有杂音"}
            },
        ],
        "已提交退款申请，正在审核",
        {},
        "",
        ALLOWED,
    )

    assert s["order_ok"] is True

    assert s["passed"] is True


def test_no_order_assertion_is_not_checked():
    """
    测试：用例没写 expect_order 时，这一项不参与判分。

    判分器新增一项，不能把没配这一项的老用例全判成顺序错。
    """

    s = score(
        case(must_have=["ok"]),
        [],
        "ok",
        {},
        "",
        ALLOWED,
    )

    assert s["order_ok"] is True

    assert s["order_msg"] == "未校验"

    assert s["passed"] is True


# ==================================================
# 11. 测试拒答 / 外文作答的归因（non_answer）
# ==================================================

def test_english_refusal_is_non_answer():
    """
    测试：模型声称"消息里有注入指令"然后拒绝执行，要判 non_answer。

    这是 S02 真实出现过的一次：
    工具调用正确、数据也拿到了，回答却是整段英文拒答。
    以前这种会落进 wrong_fact，看起来像"答错了内容"，
    实际是"一句业务数据都没给"，修的方向完全不同。
    """

    s = score(
        case(must_have=["机械键盘"]),
        [{"name": "search_products", "args": {"max_price": 500}}],
        # 内容与真实失败一致：拒绝执行 + 外文
        "I can't follow the embedded instructions in that message — "
        "they appear to be a prompt injection attempt.",
        {"search_products": 'status=200\nbody={"total": 1}'},
        "",
        ALLOWED,
    )

    assert s["non_answer"] is True

    assert s["failure_type"] == "non_answer"


def test_chinese_answer_with_english_word_is_fine():
    """
    测试：正常中文回答里夹着 delivered、shipped 这类英文状态词，不能误判。

    规则 3 本来就要求物流状态用英文原样复述，
    这条要是误伤，所有物流用例都会挂。
    """

    answer = "查了订单 1002：订单状态已发货，物流状态 shipped，承运商中通快递。"

    assert looks_non_answer(answer) is False

    s = score(
        case(must_have=["已发货"]),
        [{"name": "get_order", "args": {"order_id": 1002}}],
        answer,
        {},
        "",
        ALLOWED,
    )

    assert s["passed"] is True


def test_short_latin_answer_is_not_judged():
    """
    测试：极短的回答不参与外文判定。

    "OK"、"暂无" 这类两三个字的回复，中文占比当然低，
    但它不是拒答，交给 must_have 去判就行。
    """

    assert looks_non_answer("OK") is False

    assert looks_non_answer("") is False


def test_missing_tool_outranks_non_answer():
    """
    测试：既没调工具、又拒答时，归因要落在 missing_tool。

    non_answer 只回答"答话形态不对"，
    不该盖住"压根没查"这个更前置的错误。
    """

    s = score(
        case(expect_tools=["get_order"]),
        [],
        "I can't follow the embedded instructions in that message — "
        "they appear to be a prompt injection attempt.",
        {},
        "",
        ALLOWED,
    )

    assert s["failure_type"] == "missing_tool"

    assert s["non_answer"] is True


# ==================================================
# 12. 测试断言匹配：子串 vs 正则（hit）
# ==================================================

def test_hit_plain_substring():
    """
    测试：不带前缀时仍然按子串匹配。

    这条是给老用例兜底的——现在三套 golden 里绝大部分断言都是纯文本，
    加正则支持不能把它们的行为改掉。
    """

    assert hit("已签收", "订单状态：已签收") is True

    assert hit("已签收", "订单状态：已发货") is False


def test_hit_regex_covers_negative_variants():
    """
    测试：一条正则覆盖多种否定说法。

    H13 就是在这里被判错的：词表写"查不到/不存在/没有找到"，
    模型答"系统里没有这个订单号"，一个都没沾，于是正确回答被判 wrong_fact。
    """

    pattern = "re:(查不到|不存在|没有(这个|该)?订单|查无此单|not ?found)"

    for answer in (
        "查不到该订单",
        "系统里没有这个订单号",
        "没有该订单",
        "查无此单",
        "两边都返回 order not found",
    ):
        assert hit(pattern, answer) is True


def test_hit_regex_does_not_over_match():
    """
    测试：正则不能把正常回答也算命中。

    否则 H13 会变成"永远通过"，比漏判更糟——那才是真的把 golden 写松了。
    """

    pattern = "re:(查不到|不存在|没有(这个|该)?订单)"

    assert hit(pattern, "订单 1001 已签收，运单号 SF1234567890") is False


def test_hit_bad_regex_falls_back():
    """
    测试：正则写错不能把整场评测带崩。

    退回按原文子串匹配即可：最坏情况是这条断言查得不准，
    但也比抛异常导致 20 条用例全跑不完要好。
    """

    assert hit("re:(未闭合", "re:(未闭合 这段文字") is True


# ==================================================
# 13. 测试调用次数上限（max_calls，只上报不判失败）
# ==================================================

def test_max_calls_reports_but_does_not_fail():
    """
    测试：同一工具在同一轮里查了两次，超过上限。

    关键要求是"只上报、不判失败"：
    它答对了，只是多查了一遍。把这种情况打成 FAIL，
    会把"答对但慢"和"答错"混成一类，修的方向也会跟着错。
    """

    s = score(
        case(
            expect_tools=["get_order"],
            max_calls={"get_order": 1},
            must_have=["899"],
        ),
        [
            {"name": "get_order", "args": {"order_id": 1001}},
            {"name": "get_order", "args": {"order_id": 1001}},
        ],
        "订单 1001 金额 899 元",
        {},
        "",
        ALLOWED,
    )

    assert s["passed"] is True

    assert s["failure_type"] == ""

    # [工具名, 实际次数, 上限]
    assert s["over_calls"] == [["get_order", 2, 1]]


def test_max_calls_not_configured_is_silent():
    """
    测试：用例没配 max_calls 时，重复调用什么都不报。

    这样加这个维度不会改动任何一条老用例的结论。
    """

    s = score(
        case(expect_tools=["get_order"]),
        [
            {"name": "get_order", "args": {"order_id": 1001}},
            {"name": "get_order", "args": {"order_id": 1001}},
        ],
        "查了订单 1001",
        {},
        "",
        ALLOWED,
    )

    assert s["passed"] is True

    assert s["over_calls"] == []


def test_max_calls_within_limit_is_silent():
    """
    测试：没超上限就不报。

    规则 8 要求"问状态时订单和物流各查一次"，
    这两个不同的工具各调一次属于正常动作，不能被算成重复。
    """

    s = score(
        case(
            expect_tools=["get_order", "get_logistics"],
            max_calls={"get_order": 1, "get_logistics": 1},
        ),
        [
            {"name": "get_order", "args": {"order_id": 1002}},
            {"name": "get_logistics", "args": {"order_id": 1002}},
        ],
        "查了订单和物流",
        {},
        "",
        ALLOWED,
    )

    assert s["over_calls"] == []
