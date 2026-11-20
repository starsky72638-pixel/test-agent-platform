# 平台能力与产物报告（P5-c）

生成时间：2026-11-20
口径：只记录能从磁盘文件或真实执行输出中核对的事实。标注「待实测」的条目尚未取得数据，不用估算值填充。
配套度量脚本：`scripts/p5c_probe.py`（`stats` / `wf-distill` / `memory` 三个子命令）。

---

## 1. Agent Runtime 能力

| 能力 | 实现位置 | 状态 |
|---|---|---|
| 统一执行入口 | `core/runner.py:138` `run_tasks(agent, tasks, *, experience_provider, max_steps, on_trace)` | 已实现、已实测 |
| 单条执行 | `core/runner.py` `run_task(agent, task)` | 已实现、已实测 |
| Agent 协议 | `core/protocol.py:65` — `name` + `tools` + `run(task, ctx) -> Trace` | 已实现、已实测 |
| 崩溃隔离 | runner 捕获异常，单条失败不影响后续任务，异常轨迹标 `status=crashed/infra` | 已实现、已实测 |
| 经验注入通道 | `ExperienceProvider`：`provider(task) -> (text, exp_ids)`，ids 回填给调用方 | 已实现、已实测 |
| 动态注册 | `core/registry.py` + `agents/builtin.py`，`make(name, **kwargs)` | 已实现、已实测 |

Runtime **不埋点**：轨迹由 Agent 自己产出（`trace = agent.run(task, ctx)`）。这是「外部黑盒 Agent 接不进」的根因，也是 Workflow 必须自己写 Event 的原因。

## 2. Agent 接入情况

6 个注册工厂：

```
shop-cold / shop-tuned / main / customer / customer-cold / workflow
```

别名：`cold→shop-cold`、`tuned→shop-tuned`、`shop→shop-tuned`、`main_agent→main`、`cs→customer`、`cs-cold→customer-cold`、`wf→workflow`。

磁盘 Workflow 通过 `wf:<id>` 前缀动态可寻址，当前 3 个：

```
wf:demo          start → tool(get_order) → llm → end
wf:canvas-demo   画布产出，单工具节点（P5-c 时参数写死 1002；P6 起改为 {{extract.order_id}}）
wf:branch-demo   start → tool → memory → condition → 两支 llm → 各自 end
```

实现差异：

| 类型 | 轨迹形态 | 工具调用来源 |
|---|---|---|
| 原生 Agent（shop / customer / main） | 1 Turn = 1 轮对话，`Turn.calls` 有值 | `Turn.calls` |
| Workflow（`wf:*`） | 1 Turn 承载问答 + N 个 Event 承载执行 | `Event(kind=tool_call)` 投影 |

## 3. Trace / Event 结构

`core/trace.py`，`schema_version=1.0`。

**Turn**（对话语义）：`index` / `user_input` / `calls` / `answer` / `steps` / `elapsed_ms` / `meta`

**Event**（执行语义，与 Turn 平级）：`kind` / `seq` / `node_id` / `node_type` / `parent` / `turn` / `name` / `args` / `output` / `ok` / `error` / `started_at` / `elapsed_ms`

7 种 kind：

```
node_start / node_end / tool_call / llm_call / branch / memory_call / error
```

三条硬约定：

1. 节点不进 Turn。Workflow 的 `turns` 恒等于用户轮次数，节点只影响 `events`。
2. `calls_of_turn()` 有 events 时从 events 取，否则走 `turn.calls`，**两者不相加**（相加会让 `call_count` 虚高、顺序断言错乱）。
3. `memory_call` / `branch` 不参与工具判分——判分器只认 `KIND_TOOL_CALL`。

`ToolCall.node_id`（P5-a 新增）让失败能定位到节点：空值不写进 JSON，历史轨迹逐字节不变。

## 4. Evaluator 评测能力

入口：`evaluator/verdict.py` `evaluate(trace, task, *, allowed_tools=None) -> Verdict`。

五层判定：工具选择 → 参数 → 依据溯源 → 终态事实 → 调用顺序。失败归因优先级固定，共 11 类（`missing_tool` / `wrong_arg` / `wrong_order` / `ungrounded` / `hallucination` / `forbidden_action` / `invalid_tool` / `non_answer` / `wrong_fact` / `off_topic` / `max_steps`）。

同一判分器同时判两类 Agent，实测证据（`wf:canvas-demo` × `data/wf_eval.jsonl`）：

```
[01/3] W01   PASS                             4619ms  ok  tools=['get_order']
[02/3] W02   FAIL missing_tool@t1             5510ms  ok  tools=['get_order']
[03/3] W03   FAIL wrong_arg@t1[get_order]     6118ms  ok  tools=['get_order']
通过 1/3 (33%)
未通过归因：missing_tool 1  wrong_arg 1
失败节点：get_order
```

Verdict 携带结构化 `fail_nodes`（P5-a），消费方直接取字段，不用从文案里正则抠节点名。

已知缺口：**跨轮 `expect_order` 不可表达**。`judge_round` 只吃本轮 calls，C06 注入侧缺 `get_order` 仍判 PASS 的情况曾出现过。

## 5. Memory 经验生产与检索

**存储**：`runs/<实验目录>/experiences.jsonl`，append-only，同名 `exp_id` 取最大 version。刻意没有全局默认库，避免不同实验互相串用。

**实测数量**（`p5c_probe.py stats`，原始输出见 `reports/p5c-experiences.json`）：

| 库 | 记录行 | 唯一 exp_id | active | 类型分布 |
|---|---|---|---|---|
| `runs/20261117-001110/` | 12 | 2 | 2 | forbidden_action 1, missing_tool 1 |
| `runs/20261117-193648/` | 3 | 1 | 1 | missing_tool 1 |
| `runs/20261117-200124/` | 7 | 3 | 3 | missing_tool 1, wrong_fact 1, wrong_order 1 |
| `runs/20261117-230554/` | 8 | 2 | 2 | missing_tool 2 |
| `runs/full-cold/20261118-000730/` | 14 | 4 | 4 | forbidden_action 1, missing_tool 1, wrong_arg 1, wrong_fact 1 |
| `runs/full-plan/20261118-231325/` | 19 | 5 | 5 | forbidden_action 1, missing_tool 2, wrong_arg 1, wrong_fact 1 |
| `runs/m10-search/20261117-233252/` | 8 | 2 | 2 | missing_tool 1, wrong_arg 1 |
| `runs/main-a2/20261119-194423/` | 5 | 1 | 1 | missing_tool 1 |
| **合计** | **76** | **20**（各库相加）/ **9**（全局去重） | **20** | 见下 |

全局 failure_type 分布：`missing_tool 10` / `forbidden_action 3` / `wrong_arg 3` / `wrong_fact 3` / `wrong_order 1`。`retired 0`、`expired 0`。

「各库相加 20」与「全局去重 9」的差值说明：**同一条经验在 8 个实验库里被反复写入**（同一个 `exp_id` 出现在多个实验目录）。这不影响单次实验的正确性，但统计"平台共产出多少条不同经验"时只能用 9 这个数。

**字段**：`schema_version` / `exp_id` / `failure_type` / `pattern` / `action` / `guard` / `trigger{tags,tools,shape}` / `evidence[{trace_id,task_id,turn,failure_type,quote,agent,detail}]` / `version` / `status(active|retired)` / `support` / `contradict` / `injected` / `ttl_days` / `created_at` / `updated_at` / `origin_split` / `meta`

**生产链**：`memory/extract.py` 从失败轮提炼 → `quality.py` 闸门（禁数字）→ `store.add()`。已实测到 `version` 递增、`support/contradict` 回填、`meta.last_outcome`。

**检索**：`memory/retrieve.py`，门槛两道——形态必须相同（单轮/多轮）、关键词二元组必须有重合；打分 `0.75*关键词 + 0.15*形态 + 0.10*置信`，默认阈值 0.12。

**检索命中统计**（`p5c_probe.py memory`，原始输出见 `reports/p5c-memory.json`）：

8 个库 × 5 个任务集 = **40 个组合，31 个有命中**，单组合最高平均命中 **2.33 条/任务**（`full-plan` 库 × `wf` 任务集）。代表性切片：

| 库 | shop_eval(20) | shop_multi(10) | shop_hard(13) | customer(6) | wf(3) |
|---|---|---|---|---|---|
| `20261117-001110`（2 条） | 8 任务 / 14 条 | 0 / 0 | 3 / 5 | 5 / 10 | 3 / 6 |
| `20261117-193648`（1 条） | 1 / 1 | 7 / 7 | 1 / 1 | 0 / 0 | 0 / 0 |
| `full-cold`（4 条） | 10 / 15 | 7 / 10 | 3 / 5 | 5 / 10 | 3 / 6 |
| `full-plan`（5 条） | 12 / 20 | 6 / 9 | 4 / 7 | 5 / 10 | 3 / 7 |
| `main-a2`（1 条） | 8 / 8 | 0 / 0 | 3 / 3 | 5 / 5 | 3 / 3 |

（格式：命中任务数 / 命中经验条数）

**这个结果不是"检索效果好"，反而暴露了门槛过松。** 三条证据：

1. **跨域误命中**：shop 经验库对 `customer` 任务集平均命中 **1.67 条/任务**，对 `wf` 任务集 **2.0–2.33 条/任务**。customer 的任务是退换货咨询、wf 的任务是订单查询，与 shop 的订单经验只是共享"订单""退款"这类泛词。
2. 打分公式里 `W_SHAPE=0.15` 是常数垫底（形态已升为门槛，能进打分的候选形态恒为 1），所以只要有一个关键词重合，分数就 ≥ 0.15 + 0.1×置信，轻松越过 0.12 阈值。
3. `shop_multi` 在部分库命中 7/10、另一些库命中 0——形态门槛确实起效（有多轮经验的库才命中），说明门槛机制本身工作，问题在**关键词门槛太松**，不在形态。

**Prompt 注入的轨迹证据**（`--inject-run`，轨迹 `runs/p5c/inject-20261120-221901.jsonl`）：

```
memory_call 事件 1 条 | 命中 3 条：exp-4225ebcdcd42, exp-3a30913c112a, exp-56d434f43f4a
（库：runs/full-plan/20261118-231325/experiences.jsonl）
llm_call 事件 1 条 | prompt 400 字
含【历史经验】表头：True
三条经验的文本进入 prompt：True / True / True
```

这是第一次拿到**真实任务上的**注入证据：命中 → 渲染 → 出现在下游 `llm_call` 的 `args.prompt` 里，三个环节都在同一条轨迹上可查。

一个必须说明的口径：`prompt 400 字` 是事件层的记录上限（`executor._TEXT_LIMIT`），不是 prompt 的真实长度。经验文本能被匹配到，说明它们位于被记录的前 400 字内；更靠后的内容无法从轨迹断言。

**必须说明的事实**：多次闭环实验差值 0。原因链是——任务落在模型常识范围内（customer 两次冷变体 12 次执行 0 失败）→ fit 无失败 → 提炼 0 条 → 注入组等于基线 → 覆盖 0/3。这是可解释的研究结论，不是 bug。

## 6. SFT / DPO 数据产物

| 文件 | 样本 | 含 tool_calls | 来源 | 生成方式 |
|---|---|---|---|---|
| `data/distill/sft.jsonl` | **22** | 21 | agent=`shop-cold`，来自 `runs/full-cold/20261118-000730` 的 fit 组（26 条轨迹） | `scripts/distill.py --run` |
| `data/distill-main-a2/sft.jsonl` | **25** | 24 | 来自 `runs/main-a2/20261119-194423` 的 fit 组 | 同上 + `--out` |
| `data/train_sft.jsonl` | **18** | 17 | 带 system prompt 的整理版，`meta.case_id/passed` | 手工/半自动整理 |
| `data/pref_pairs.jsonl` | **1** | — | DPO 偏好对（`prompt` / `chosen` / `rejected` / `failure_type`） | 手工构造 |
| `data/distill-wf/sft.jsonl` | **2** | 2 | `wf:branch-demo` / `wf:canvas-demo` 的 W01 | `p5c_probe.py wf-distill` |

合计 SFT 65 条（不含 wf 的 2 条），空样本 0 条。

每份文件各有 1 条不含 `tool_calls`——这些是纯咨询类问题（不需要调工具就能答），不是缺字段。

`distill` 样本字段：`id` / `task_id` / `split` / `agent` / `turns` / `tools` / `messages`。

`messages` 按 OpenAI tool_calls 形态：`user` → `assistant(tool_calls)` → `tool` → … → `assistant(answer)`。

三道闸门：工具报错的轨迹不进 SFT、判分未过的不进 SFT、按（输入+调用序列+答案）哈希去重。

**Workflow 轨迹进 distill：已实测。** 4 条轨迹（`wf:branch-demo` W01 + `wf:canvas-demo` W01/W02/W03）→ SFT 2 条 / 失败 2 条，去重丢弃 0。

逐条结构检查全部 `OK`（`msgs=4`：user → assistant(tool_calls) → tool → assistant(answer)），`tools=['get_order']` 正确投影自 `Event(kind=tool_call)`——Workflow 的 `Turn.calls` 是空的，这一步证明 `calls_of_turn` 的事件路径在 distill 里也生效。失败样本正确归到 W02(`missing_tool`)、W03(`wrong_arg`)。

## 7. Workflow 执行能力

4 类基础节点 + 2 类扩展节点：

| 节点 | 关键配置 | 事件 |
|---|---|---|
| `start` | — | node_start/node_end |
| `tool` | `tool` / `args`（支持模板） | + tool_call |
| `llm` | `prompt` / `model` | + llm_call |
| `end` | `answer`（留空取上一节点输出） | node_start/node_end |
| `condition` | `field` / `op`(11 种) / `value` / `branches{true,false}` | + branch |
| `memory` | `query` / `top_k` / `min_score` / `max_chars` | + memory_call |

执行语义（P5-b 起）：从 entry 出发的**激活集驱动**，condition 只激活命中的那一支，未选中分支连 `node_start` 都不产生；汇聚为 OR（任一入边激活即执行）。

模板：`{{input}}` / `{{nodes.<id>.output}}`，引用不存在的节点直接抛 `TemplateError`，不渲染成空串。

`wf:branch-demo` 实测（`runs/wf-branch-demo-20261120-210214.jsonl`）：

- `steps: 6`，事件含 start / get_order / recall / check_status / llm_ship / end_ship
- 全文搜 `llm_other` 无匹配——未选中分支零痕迹
- `calls: 1`，而事件里有 branch 与 memory_call——两类事件未污染工具计数
- 空经验库下 memory 节点输出空串，LLM 正常回答，未注入「历史经验」表头

限制：不做并行、不做 AND 汇聚、不支持循环；画布只能只读展示 condition/memory（改了连线再保存会导致 `next` 与 `branches` 不一致，加载时明确报错）。

## 8. 当前能力缺口

1. **检索门槛过松，已有跨域误命中的实测数字**：shop 库对 `customer` 任务集平均命中 1.67 条/任务、对 `wf` 任务集 2.0–2.33 条/任务。`W_SHAPE=0.15` 常数垫底 + 单个关键词重合即可越过 0.12 阈值。命中率高不等于检索准。
2. **没有「模型行为性失败 → 注入 → 改善」的正例**。现有差值全为 0，且原因已定位。
3. **Workflow 侧的失败类型与 Memory 适用范围不匹配**：`wf_eval` 的两条失败是图配置缺失（缺节点、参数写死），提炼出的经验对 LLM 行为无影响。
4. **参数提取未做**：`get_order` 的 `order_id` 写死 1002，导致 W03 必然 `wrong_arg`。这让 `wf_eval` 无法成为「会随改动变化」的评测集。
5. 画布不能编辑 condition/memory 参数；SFT 无训练环节；跨轮 `expect_order` 不可表达；`llm_call` / `node_end` 的 output 与 prompt 在事件层被截断到 400 字，长 prompt 的后半段无法从轨迹核对。

## 9. 下一阶段建议（P6）

**先做参数提取**。理由不是功能完整度，而是它能修复缺口 4：让 `order_id` 从 `{{input}}` 取，`wf_eval` 的 W03 才会从「按设计失败」变成「可能通过」，评测集才有区分度，后续任何改动的效果才有可观测的分母。

顺序建议：

1. 参数提取（从用户输入抽出订单号/金额等结构化值，填进 tool 节点参数）
2. `wf_eval` 扩到 8-10 条，覆盖单轮/多轮、有分支/无分支
3. 用扩展后的任务集重测 `wf:*`，拿到第一个有区分度的通过率
4. 在这个有区分度的任务集上再验证 Memory（此时失败才可能是模型行为性的）
5. 收紧急检索门槛（第 8 节缺口 1）

**为什么不先修检索门槛**：门槛改动的效果只能在有区分度的任务集上测量。现在 `wf_eval` 固定 1/3，改完门槛仍是 1/3，无法判断改对了还是改坏了。缺的不是一个更严的阈值，是一个能反映阈值好坏的分母。

**在此之前不建议做**：Memory 调参、SFT 训练、画布表单化——前两项缺有区分度的分母，第三项不解决闭环效果。
