# Agent 评测考场

**这个仓库的重点不是那个客服 Agent，而是判断它答得对不对的那套东西。**

LLM Agent 的失败很难复现，也很难归因：同一版提示词、`temperature=0`，同一条用例跑两次可以一次正常、一次拒答。所以“它变好了”这句话，没有评测系统就只是感觉。

这里做的是一套可复现的考场：42 条 golden 用例、五层判分、带优先级的失败归因、每轮轨迹落盘成人能读的报告。被测的购物客服 Agent 是第一个考生。

| 集 | 用例 | 最近一次 | 平均耗时 | 平均调用 |
|---|---|---|---|---|
| `shop_eval.jsonl` | 20 单轮 | 20/20 | 8119 ms | 1.6 次 |
| `shop_eval_multi.jsonl` | 9 多轮 | 9/9 | 17993 ms | 3.9 次 |
| `shop_eval_hard.jsonl` | 13 难集 | 13/13 | 7427 ms | 1.6 次 |

模型 `qwen3.8-flash`，`temperature=0`，5 个工具。**这是一次运行的结果，不是稳定性结论** —— 见文末「已知局限」。

---

## 一、三层结构

```
被测 Agent          5 个工具 · 11 条规则 · ReAct 多轮 · 记忆(槽位+事实卡片)
     ↓ 每轮完整轨迹
判分器(阅卷)        五层断言 · 11 类归因 · 严格优先级
     ↓ 逐题逐轮的 calls / outputs / answer / 五层判定
报告                 reports/*.json，任意两轮可逐题 diff
     ↓ 通过轨迹 / 失败轨迹
沉淀                 golden 候选 · 规则线索 · 待定位缺口
```

**被测 Agent**（`main_agent/`）
`search_products` / `get_product_detail` / `get_order` / `get_logistics` / `apply_refund` 五个工具，背后是一个按真实接口语义造的 `mock_server.py`（含 404、422、409、待支付不可退等边界）。记忆分两层：槽位（`budget_max` / `category` / `last_order_id` / `last_product_id`，只从工具实参反填）+ 事实卡片（`MAX_FACTS=8` 的 LRU，只当线索，价格金额一律以最新工具返回为准）。

**判分器**（`tests/replay.py`）
给定一条轨迹和一份断言，输出五层判定 + 一个归因。

**报告**
每轮的 `calls`（工具名+参数+返回）、`answer`、五层判定全部落盘。`tests/feedback.py` 把报告转成下一轮要用的三份产物。

---

## 二、五层判分：每一层在拦什么

| 层 | 判什么 | 典型反例 |
|---|---|---|
| L1 工具选择 | 该调的调了没有；不该调的调了没有 | 用户问“订单什么状态”，只查了订单没查物流，答一半 |
| L2 参数 | 工具实参填对没有 | 预算 300 却传 `max_price=3000` |
| L3 依据溯源 | 回答里的事实有没有工具返回作依据 | 工具返回 `null`，回答却给了个“预计明天到” |
| L4 终态事实 | 必需内容在不在、禁止内容在不在 | 1003 是待支付，回答却说“已为您提交退款” |
| L5 调用顺序 | 该先做的有没有先做 | 没查订单状态就调 `apply_refund` |

**L3 是唯一允许跨轮取数的一层。** 多轮里“上一轮查到的事实”是合法上下文；但如果 L1 也跨轮，就会出现“这轮不查、复用上轮调用”的作弊路径。所以 L1 只认本轮 `tool_calls`，L3 用累积的 `seen_outputs`。

---

## 三、归因为什么必须有优先级

一条轨迹常常同时踩几个坑。不定义优先级，同一条轨迹换个人归到不同类，报告就不可比。

```
infra_error → empty_answer → invalid_tool → forbidden_action
  → ungrounded → hallucination → missing_tool
  → wrong_order → wrong_arg → non_answer → wrong_fact
```

几处刻意的排序：

- **`infra_error` 最前**：403、超时不是模型的问题，必须从成绩里剔除，否则“通过率”被基础设施污染。
- **`wrong_order` 排在 `wrong_arg` 前面**（`replay.py:863`）：先动手后核实属于流程违规，哪怕参数后来都补对了，动作本身已经发出去了。
- **`non_answer` 和 `wrong_fact` 分开**：“没答”和“答错”的修法完全不同 —— 前者调提示词或温度，后者查工具或数据。混在一起就会优化错的东西。
- **`redundant_call` 只上报，不影响 `passed`**：正确性优先于效率。把“多查两次”和“答错”混进同一个结论，真问题会被盖掉。

---

## 四、迭代记录：每一轮改动都是被一次真实失败推着走的

| 观察到的失败 | 归因 | 补了什么 | 前后 |
|---|---|---|---|
| S08 只查订单不查物流，答了一半 | `missing_tool` | 规则 8：状态类问题订单+物流各查一次 | 19/20 → 20/20 |
| S13 查完订单直接答“已提交退款”，`apply_refund` 一次没调 | `missing_tool` | 规则 11：动作类断言必须有本轮工具成功返回 | 规则 11 后轨迹里真的出现 `apply_refund` |
| S14 被判失败，实际拿到 200 并如实报了单号 | 判分器误判 | 用例改成两轮自足（第1轮退拿 200，第2轮再退撞 409），不再依赖别的用例的副作用 | 假失败消除 |
| S02 只给预算就反问“想买什么类别” | `missing_tool` | 规则 6 重写为“预算/品类/商品名任一项即可搜” | 19/20 → 20/20 |
| H13 答“没有这个订单号”被判答错 | 判分器误判 | 词表补了 4 次仍未穷尽 → 改 `re:` 前缀正则 | 12/13 → 13/13 |
| M09 单题 9 次调用、139 秒仍 PASS | 无约束 | 加 `max_calls`，**只上报不判失败** | 报告里可见 |

两件事值得单独讲：

**1. 判分器自己也会错，而且更危险。** S14 那次，Agent 完全正确，是判分器依赖了别的用例留下的退款状态。Agent 的 bug 影响一条回答，判分器的 bug 篡改结论。所以“改判分器必须重跑全部”，旧报告在新判分器下没有证明力。

**2. 提示词里的例句不能和 golden 原句重合。** 规则 6 曾经把 S02 的原话写进提示词，模型把那段当成 prompt injection 直接英文拒答。测试用例和提示词共用一句话，会造成自指污染。

---

## 五、结果怎么读（分母比分子重要）

| 数字 | 真实含义 |
|---|---|
| 通过 20/20 | 20 条有效，0 条链路故障 |
| 顺序 N/N | 只有 S13、S14 配了 `expect_order`，其余 18 条是默认通过。**真实分母是 2** |
| 依据可溯 3/3 覆盖 3/13 | hard 只有 3 条配了正向事实溯源，那 10 条本质是“不许编造”，没有正向事实可溯 |
| 重复调用 0 处 | 只有 M02、M09 配了 `max_calls`。0 表示“检查过没超”，不是“全站无重复” |

不标分母的 100% 是自欺。

---

## 六、已知局限

1. **单次全绿不等于稳定。** `temperature=0` 不等于确定性：S02 五次里出现过三种表现（正常 / 反问 / 英文拒答）。42 条跑一次全绿，只能说这次没翻车。
2. **词表判语义是死路。** H13 的否定表达补了 4 次词（查不到 → 不存在 → 没有这个订单号 → 正则），每次都是补上这次漏的。`re:` 只是把枚举写紧凑，不是语义理解。要真正解决得上 LLM-as-judge，但那就引入新的“判官也会错”。
3. **规模小。** 42 条、单域、5 个工具、无并发、无真实流量。这套东西的价值在方法，不在规模。
4. **没有训练。** 模型是 API 上的 `qwen3.8-flash`，没有权重，不存在微调这一环。仓库里所有“数据”产物都是给下一轮评测用的，不是给梯度用的。
5. **顺序与重复调用两个开关覆盖得薄。** 只有少数用例配了断言，机制本身有单测，但实跑里触发次数很少。

---

## 七、跑起来

```powershell
# 1. 起假后端（另开一个窗口常驻）
python mock_server.py

# 2. 跑三套
python -m tests.replay data\shop_eval.jsonl
python -m tests.replay data\shop_eval_multi.jsonl
python -m tests.replay data\shop_eval_hard.jsonl

# 3. 单测
python -m pytest tests -q

# 4. 从报告沉淀下一轮（可选）
python tests\feedback.py --current reports\shop_eval.json reports\shop_eval_multi.json reports\shop_eval_hard.json
python tests\feedback.py --baseline reports\shop_eval_before.json --current reports\shop_eval.json --dry-run
```

`.env` 需要 `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `MODEL_NAME` / `TEMPERATURE` 四项（见 `config/settings.py`）。replay 启动时会自动 reset 服务端状态。

---

## 八、目录

```
main_agent/         被测 Agent：core/agent.py 主循环，tools/shop_skill.py 工具+提示词，view/memory.py 记忆
tests/replay.py     判分器：五层断言 + 归因 + 报告落盘
tests/feedback.py   报告 → golden 候选 / 规则线索 / 待定位缺口
tests/test_*.py     单测：判分器的边界、数据管道的三道闸门
data/               三套 golden + 历史产物
reports/            历次评测报告（保留 19/20 的基线，用于逐题 diff）
mock_server.py      假后端，含 404/422/409/待支付等边界
```

`reports/shop_eval_before.json` 是加规则 8 之前的 19/20 基线，故意留着 —— 它是“定位缺口 → 补能力 → 证明变好”这条链在仓库里的证据。
