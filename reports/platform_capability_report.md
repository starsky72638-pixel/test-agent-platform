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

磁盘 Workflow 通过 `wf:<id>` 前缀动态可寻址，当前 5 个：

```
wf:demo          start → tool(get_order) → llm → end
wf:canvas-demo   画布产出，单工具节点（P5-c 时参数写死 1002；P6 起改为 {{extract.order_id}}）
wf:branch-demo   start → tool → memory → condition → 两支 llm → 各自 end
wf:shop-demo     start → get_order → get_logistics → llm → end（P7 新增，双工具对照用）
wf:report-demo   start → get_order → get_logistics → llm(inject_experience) → end（P7 新增，经验注入实验载体）
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

> 上面那段是 **P5-c 的原始快照**（`runs/wf-canvas-demo-20261120-210241.jsonl`）。P6 之后同一命令的结果是 `2/3`、只剩 `missing_tool 1`，见第 10 节。

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

**必须说明的事实**：多次闭环实验差值 0。原因链是——任务落在模型常识范围内（customer 两次冷变体 12 次执行 0 失败）→ fit 无失败 → 提炼 0 条 → 注入组等于基线 → 覆盖 0/3。这是可解释的研究结论，不是 bug。注意这里的 `0/3` 是 **customer 域**的口径，不能推及 Workflow。

**P7 补充（2026-11-21，详见 `reports/p7_experiment.md`）**：Workflow 侧的注入通道已实测打通，但效果仍测不出来。

| 项 | 实测 |
|---|---|
| 注入覆盖 | `wf:report-demo` × 4 条 holdout：**4/4**，每题 **238 字 / 2 条经验** |
| 基线 vs 注入 | 4/4 vs 4/4，翻转 0 |
| 轨迹证据 | `holdout-injected.jsonl` 里 `【历史经验` 出现 4 处；`holdout-baseline.jsonl` **0 处**（基线零污染） |
| 成本 | 侧耗时 36,827 ms → 49,999 ms（**+35.8%**），收益 0 |
| 检索质量 | `exp-56d434f43f4a`（禁止调用 apply_refund）被注入到问金额/问状态的题上，同域误命中的逐字实证 |

结论：**基线 100% 形成天花板**，满分考卷测不出提升空间；本轮既不能证明 Memory 有效，也不能证明有害。

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

模板：`{{input}}` / `{{nodes.<id>.output}}` / `{{extract.<字段>}}`（P6 起），引用不存在的节点直接抛 `TemplateError`，不渲染成空串。

`wf:branch-demo` 实测（`runs/wf-branch-demo-20261120-210214.jsonl`）：

- `steps: 6`，事件含 start / get_order / recall / check_status / llm_ship / end_ship
- 全文搜 `llm_other` 无匹配——未选中分支零痕迹
- `calls: 1`，而事件里有 branch 与 memory_call——两类事件未污染工具计数
- 空经验库下 memory 节点输出空串，LLM 正常回答，未注入「历史经验」表头

限制：不做并行、不做 AND 汇聚、不支持循环；画布只能只读展示 condition/memory（改了连线再保存会导致 `next` 与 `branches` 不一致，加载时明确报错）。

## 8. 当前能力缺口

> 本节写于 P5-c，保留原文以便对照。到 P10 为止：缺口 4 已在 P6 修掉；
> 「Workflow 默认读不到经验」已在 P10-b 修掉（`wf:*` 默认注入 `data/memory/experiences.jsonl`，
> 实测 `hit_count ≥ 1`）；缺口 2 由 P7 推进为「注入已发生、但基线满分无提升空间」。
> 逐项现状见**第 11 节**。

1. **检索门槛过松，已有跨域误命中的实测数字**：shop 库对 `customer` 任务集平均命中 1.67 条/任务、对 `wf` 任务集 2.0–2.33 条/任务。`W_SHAPE=0.15` 常数垫底 + 单个关键词重合即可越过 0.12 阈值。命中率高不等于检索准。
2. **没有「模型行为性失败 → 注入 → 改善」的正例**。P7 把缺口的形态推进了一步：注入现在**确实发生了**（4/4 覆盖、238 字），但仍无改善——不是"没注入"，而是"基线已满分、无提升空间"（见第 5 节 P7 补充与 `reports/p7_experiment.md`）。
3. **Workflow 侧的失败类型与 Memory 适用范围不匹配**：`wf_eval` 的两条失败是图配置缺失（缺节点、参数写死），提炼出的经验对 LLM 行为无影响（P6 修掉了「参数写死」那一条，剩「缺节点」1 条）。
4. ~~**参数提取未做**：`get_order` 的 `order_id` 写死 1002，导致 W03 必然 `wrong_arg`。这让 `wf_eval` 无法成为「会随改动变化」的评测集。~~ **已在 P6 修复，见第 10 节**：`{{extract.order_id}}` 落地后 `wf_eval` 从 1/3 变 2/3。
5. 画布不能编辑 condition/memory 参数；SFT 无训练环节；跨轮 `expect_order` 不可表达；`llm_call` / `node_end` 的 output 与 prompt 在事件层被截断到 400 字，长 prompt 的后半段无法从轨迹核对。

## 9. 下一阶段建议（P6）

> **本节是 P5-c 当时写下的判断，保留原文以便对照。第 1 项已在 P6 实施并实测，结果见第 10 节。**

**先做参数提取**。理由不是功能完整度，而是它能修复缺口 4：让 `order_id` 从 `{{input}}` 取，`wf_eval` 的 W03 才会从「按设计失败」变成「可能通过」，评测集才有区分度，后续任何改动的效果才有可观测的分母。

顺序建议：

1. 参数提取（从用户输入抽出订单号/金额等结构化值，填进 tool 节点参数）
2. `wf_eval` 扩到 8-10 条，覆盖单轮/多轮、有分支/无分支
3. 用扩展后的任务集重测 `wf:*`，拿到第一个有区分度的通过率
4. 在这个有区分度的任务集上再验证 Memory（此时失败才可能是模型行为性的）
5. 收紧急检索门槛（第 8 节缺口 1）

**为什么不先修检索门槛**：门槛改动的效果只能在有区分度的任务集上测量。现在 `wf_eval` 固定 1/3，改完门槛仍是 1/3，无法判断改对了还是改坏了。缺的不是一个更严的阈值，是一个能反映阈值好坏的分母。

**在此之前不建议做**：Memory 调参、SFT 训练、画布表单化——前两项缺有区分度的分母，第三项不解决闭环效果。

## 10. P6：工具参数抽取（已完成，实测）

**目标**：让 tool 节点的参数随用户输入变化，使 `wf_eval` 从「固定 1/3」变成会随改动变化的评测集。

**做法**：复用现有 `{{...}}` 模板渲染链，不新增 JSON 结构——参数值仍是一个字符串，画布输入框能正常显示与编辑。若改成 `{"$extract": ...}` 对象，画布会渲染成 `[object Object]`。

| 字段 | 规则 | 例 |
|---|---|---|
| `order_id` | 先匹配「订单/单号/order + 4 位数字」，再退到裸 4 位数字 | 「帮我查下订单 1001 的金额」→ `1001` |
| `amount` | 数字 + 元/块/rmb | 「预算 899 元」→ `899` |
| `product_id` | `P` + 2–6 位数字 | 「商品 P1001」→ `P1001` |

三条边界，都是刻意的：

1. **抽不到抛 `ExtractError`**（`executor` 包成 `ExecutionError`，`adapter` 标 `STATUS_INFRA`）。返回空串会让工具带空参数打接口拿 404，最后被记成「模型答错」。
2. **只从本轮用户输入抽**，不从节点输出抽——节点输出是工具原始报文，要抽得先约定格式。
3. **字段名写错在加载期拦**：`schema.validate` 扫所有 config 里的模板引用，`{{extract.order}}` 直接报「未知抽取字段」。

### 实测结果

命令：`scripts\run_shop.py --agent wf:canvas-demo --golden data\wf_eval.jsonl --no-compare`
轨迹：`runs/wf-canvas-demo-20261120-223855.jsonl`

| 项 | 改动前（P5-a/P5-c） | P6 |
|---|---|---|
| 通过率 | 1/3 (33%) | **2/3 (67%)** |
| W01（问 1002 状态与金额） | PASS | PASS |
| W02（问 1002 快递） | FAIL `missing_tool@t1` | FAIL `missing_tool@t1`（不变） |
| W03（问 1001 金额） | FAIL `wrong_arg@t1[get_order]` | **PASS** |
| 未通过归因 | `missing_tool 1` + `wrong_arg 1` | `missing_tool 1` |
| 失败节点 | `get_order` | **无**（缺的是节点本身，不是某节点判错） |

**轨迹级证据**（不是靠通过率反推）：

| 检查 | 结果 | 含义 |
|---|---|---|
| 含 `"order_id": "1001"` 的行 | 1 行（且该行 `task_id` 就是 W03） | 只有问 1001 的题去查了 1001 |
| 含 `"order_id": "1002"` 的行 | 2 行（W01、W02） | 其余题仍按各自输入查 1002 |
| 含 `899` 的行 | 1 行，且与 W03 同一行 | W03 拿到 1001 的真实金额，没有复用 1002 的 429 |
| W02 行内 `get_logistics` | 0 次 | 失败确实是图缺节点，不是参数问题 |

单跑复核（`run_agent.py --agent wf:canvas-demo --question "帮我查下订单 1001 的金额"`）：事件里 `tool get_order {"order_id": "1001"}`，回答「已签收，899.0」；节点 4 个、事件 10 条、轮次仍为 1——参数抽取没有改变 Turn 语义。

**测试**：398 → **402 passed**（+4：规则优先级、模板参数渲染、抽不到落 infra 且不调工具、schema 拦未知字段）。

### 仍未解决的

- `wf:demo` 与 `wf:branch-demo` 的 `order_id` 仍是写死的 1002（它们是示例，不是评测对象）。
- 只剩 1 条失败，且是「图缺节点」类。要让 `wf_eval` 进一步有区分度，得先给画布加 `get_logistics` 节点并扩任务集（原计划第 2、3 步）。
- 经验检索门槛仍未收紧（第 8 节缺口 1）。

## 11. P7–P10：注入、多轮、可追溯、数据闸门（均已完成，实测）

第 1–10 节记录到 P6。P7 之后又有四项能力落地，此前只散落在 commit message 与对话里，这里补齐。

### 11.1 P7：Workflow 消费 Runtime 经验

**改动**：`llm` 节点支持 `config.inject_experience`；`WorkflowAgent` 把 `ctx.experience` 交给 `Executor`，在开了开关的节点把经验原样前置到 prompt。空经验不拼——基线组因此与"没有 runtime 经验"逐字一致。

**实验**（`runs/20261121-000424`，`wf:report-demo` × `wf_learn`，注入旧库 `runs/full-cold/20261118-000730`）：

| 项 | 基线 | 注入 |
|---|---|---|
| 通过 | 4/4 | 4/4 |
| 检索覆盖 | — | **4/4** |
| 每题注入 | 0 字 | **238 字 / 2 条经验** |
| 侧耗时 | 36.8 s | 50.0 s（+35.8%） |

**结论**：注入通道打通（`【历史经验】` 表头在注入组轨迹出现 4 次、基线组 0 次），但**差值 0**。原因不是没注入，是基线已满分、没有提升空间。同时暴露 1 例同域误命中：`exp-56d434f43f4a`（"不要调用 apply_refund"）被注入到问金额/问状态的题上。详见 `reports/p7_experiment.md`。

### 11.2 P8：跨轮上下文与参数槽位

**P8-a**：一条 Task 只建一个 `ctx`，逐轮传入。四个键各有归属——`input` 每轮覆盖、`nodes` 每轮清空、`history` 按轮按值归档、`experience` 整条任务共用。`trace.meta["context_turns"]` 记录实际归档轮数。

**P8-b**：新增 `slots`。每轮先抽本轮输入，抽到就写槽并记轮次；**抽不到才沿用**历史值，并发一条 `KIND_SLOT_REUSE` 事件（含 `field` / `value` / `source` / `from_turn` / `reused_from_previous_turn`）。冷启动无槽位仍抛 `ExtractError` 落 `infra_error`，不猜默认值。slot 存在 ctx 里，换 Task 不继承。

**实测**（`wf:report-demo` × `data/wf_multi.jsonl`）：3/3 PASS、`turns=6`、`calls=12`。

轨迹级证据（M03，`runs/wf-report-demo-20261121-003837.jsonl`）：

```
seq 13  slot_reuse  turn=1  {field: order_id, value: "1002", from_turn: 0, reused_from_previous_turn: true}
seq 17  tool_call   turn=1  get_order      {"order_id": "1002"}
seq 20  tool_call   turn=1  get_logistics  {"order_id": "1002"}
meta    context_turns: 2
```

第二轮输入是「那物流到哪了」，**一个数字都没有**，两个工具仍拿到 `1002`——只能来自槽位。`slot_reuse` 的 `node_id` 为空、`started_at` 为 0，因为它是轮级事件；判分器只认 `tool_call`，所以调用数没有虚增。

### 11.3 P10-a：实验与蒸馏产物的来源信息

新增 `pipeline/provenance.py`：

| 产物 | 位置 | 记录内容 |
|---|---|---|
| 实验 manifest | `runs/<stamp>/manifest.json`，建目录即写 | Agent、数据集路径 + SHA-256 + 字节数、split 实际条数与 task_id、7 项运行参数、Git commit / branch / dirty |
| distill provenance | 原 `manifest.json` 的 `provenance` 键 | 输入轨迹路径 + SHA-256 + 条数、输出条数、`agents` / `splits` 分布、代码文件 SHA-256 与 Git 状态 |

三条硬规则：取不到写 `null`；`dirty` 是**三态**（Git 失败时 `commit` 与 `dirty` 同时为 `null` 并保留 `error`，不伪装成 clean）；写盘失败抛 `ManifestError`，`run_loop` 打印警告后继续。

**实测**：`data/sft` 是同源核对，另用 `distill --trace runs/wf-report-demo-20261121-103250.jsonl --golden data/wf_multi.jsonl` 验证——`provenance.inputs[0].traces=3`、`outputs={'sft':3,'failed':0}`、`agents={'wf:report-demo':3}`，且独立复算 `sha_ok True`（哈希来自真实字节，不是抄的）。

### 11.4 P10-b：经验库归并与默认注入

**问题**：9 条唯一经验散在 8 个实验目录（76 行，11 份重复写入），而 `agents/builtin.py` 给 `wf:*` 只注入工具池、**不传 `memory_store`**，所以线上 Workflow 的 memory 节点恒 0 命中。

**做法**：`memory/store.py` 增加 `DEFAULT_STORE_PATH` 与 `merge_files`（按 `exp_id` 取 `(updated_at, version)` 最大的一份，**不累加** `support/contradict`，`.tmp` + `replace` 原子写，输入只读）；`_workflow_factory` 默认注入该库；`run_loop --store default` 显式选用（**默认仍是一次实验一个新库**，因为 `record_outcome` 会累加反证，共库会破坏可复现性）。

**实测**：`merge_memory.py` 8 库 76 行 → **9 条**（`data/memory/experiences.jsonl` 共 9 行 9 唯一 id）；`wf:branch-demo` 真跑轨迹里 `"hit_count": 0` 零次、`"hit_count": 1..9` 一次，全文含 `【历史经验】`——**经验第一次真正进了线上 Workflow 的 prompt**。

### 11.5 P10-c / P10-c2：SFT 质量闸门与冲突四分类

`pipeline/sft.py` 只做判定不做 IO，`scripts/build_sft.py` 负责读写。判定顺序是：

**结构校验 → split 分流 → 池内四分类**

| 类别 | 条件 | 处理 |
|---|---|---|
| `duplicate` | 同 `task_id` + `agent` + `split`，答案一致 | 留 canonical，其余记 `duplicate` |
| `rerun_variance` | 同 `task_id` + `agent` + `split`，答案不同 | 留第一条 canonical，其余记 rerun 并带 `canonical_sample_id`，**原样本不删** |
| `unresolved` | 来源含 `unknown`，无法判断同源 | 整组隔离，不猜 |
| `true_conflict` | 不同 Agent / 不同 `task_id` / 手工集与生成集答案不一致 | 整组隔离，等人工 |

canonical 只由 `(源文件路径, 行号, sample_id)` 决定——**不看答案长度、不做相似度、不掷随机数**。`unknown` 与 `holdout` 不参与训练侧的冲突裁决，否则一条来路不明的样本能拖走同问句的可训练样本。

**实测**（`data/sft/report.json`，67 条输入 / 4 个文件）：

| 项 | 数量 |
|---|---|
| 有效 | 67 |
| duplicate | 0 |
| rerun_variance | 21 组 / 42 条 |
| true_conflict | 1 组 / 2 条 |
| unresolved | 0 |
| canonical | 26 |
| **train** | **26** |
| **holdout** | **0** |
| unknown | 18 |
| 隔离合计 | 41 |

`train` 从 P10-c 的 5 条提到 26 条，不是放宽标准，而是把「同一次实验的两次运行」从不该占用的冲突队列里拿出来。**`holdout` 为 0 是当前最实质的缺口**：那 2 条是 W01 由 `wf:branch-demo` 与 `wf:canvas-demo` 分别产出、答案不同，属 `true_conflict`，需人工裁决。

### 11.6 本阶段累计

| 阶段 | 提交 | 内容 |
|---|---|---|
| P7-0 / P7-a | `60ed32b` | `inject_experience`、`report-demo`、`wf_learn` |
| P7 文档 | `3844e13` | `p7_experiment.md` |
| P8-a | `e6310c6` | 跨轮 ctx、按轮归档、任务间隔离 |
| P8-b | `15cb36e` | `slots` + `slot_reuse` 事件 |
| P10-a | `c09db7f` | `manifest.json` / distill `provenance` |
| P10-b | `b02e585` | 经验归并（76 行 → 9 条）+ `wf:*` 默认注入 |
| P10-c | `420e8a3` | SFT 质量闸门首版 |
| P10-c-fix | `80a074e` | 改为先按 split 分流、再判冲突 |
| P10-c2 | `85e1b30` | 同源重跑识别与冲突四分类 |

测试数从 P7 前的 404 涨到 **453**（2026-11-21 实测，与 README 声明一致）。中间各笔的增量为
逐次叠加的小步，不具解释价值，此处不逐一列出——需要精确数字时以对应 commit 的 message 与 `pytest -q` 为准。
