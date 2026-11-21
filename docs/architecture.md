# 架构说明

## 一句话

这是一个 **Agent Runtime**：统一接入不同结构的 Agent，把每次执行落成可回放的轨迹，
用判定器找出失败原因，把原因提炼成经验，再在下一次相似任务上把经验注入回去。
评测不是目的，是发现"Agent 哪里不行"的手段。

## 分层与依赖方向

```
api/ · web/        HTTP 门面（8020）· 画布 Builder（零依赖原生 JS + SVG）
   ↑
agents/            原生 Agent：shop（全量证据）· main_agent（仅单测）· customer
agentplatform/     Workflow 引擎：定义 / 校验 / 拓扑 / 执行 / Agent 适配 / 存储
   ↑ 只能单向依赖：core 不认识上面任何一层
core/              Agent 协议 · 执行循环 · 轨迹（Turn + Event 双层）
   ↑
evaluator/         轨迹 → 判定（只读轨迹，不调 Agent）
memory/            经验：提炼 · 过滤 · 存储 · 归并 · 检索 · 注入
pipeline/          串成闭环：learn / verify / sft / provenance
scripts/           命令行入口（run_shop / run_loop / run_agent / distill / build_sft / merge_memory）
```

依赖方向的硬约束：**`core/` 不 import `evaluator/`、`memory/`、`agents/`**。
经验是通过 `RunContext.experience` 这个字段传进 `run_task` 的，
不是 Runtime 主动去查经验库。这条保证了：删掉 `agents/` 与 `demos/` 之后，
`core/` 的测试仍然能独立跑通。

## 各层各自负责什么

| 层 | 文件 | 职责 | 不负责 |
|---|---|---|---|
| 接入 | `core/protocol.py`、`core/registry.py`、`agents/builtin.py` | 规定 `name` + `run(task, ctx) -> Trace`；按名字造 Agent，`wf:<id>` 前缀动态路由到磁盘定义 | 不知道任何业务 |
| 执行 | `core/runner.py`、`core/task.py`、`core/context.py` | 调 Agent、串任务、把崩溃隔离成 `status=crashed`；提供与业务无关的上下文机制（槽位反填/渲染） | 不实现 ReAct 循环，不知道哪个参数属于哪个槽 |
| 轨迹 | `core/trace.py`、`runs/**/*.jsonl` | `Turn`（对话语义）+ `Event`（执行语义）两层结构、落盘（`save_traces`）与回读（`load_traces`） | 不判分 |
| 判定 | `evaluator/verdict.py`、`matching.py` | 逐轮断言、事实溯源（`grounded_claims`）、失败分类、`fail_nodes` 定位 | 不改轨迹 |
| 经验 | `memory/{extract,quality,store,retrieve,inject}.py` | 失败 → 可执行动作 → 过闸门 → 存储 → 归并 → 检索 → 注入 | 不认识具体业务词 |
| 工作流 | `agentplatform/{definition,schema,topology,executor,adapter,store}.py` | 把 JSON 图编译成节点流：激活集驱动、`condition` 分支、`memory` 节点、`{{extract.*}}`、跨轮 ctx 与 slots | 不知道业务，不判分，`Turn.calls` 刻意留空 |
| 回流 | `pipeline/{learn,verify,sft,provenance}.py`、`scripts/{distill,build_sft,merge_memory}.py` | 学习/验证两阶段、轨迹 → SFT 与失败样本、质量闸门、来源信息 | 不做训练 |
| 门面 | `api/app.py`、`web/builder/` | HTTP（8020）与画布，只经 `agents/builtin` 造 Agent | 不参与执行与判分 |

## 一次 `run_loop` 的调用链（函数级，对应真实代码）

```
scripts/run_loop.py
 ├─ agents/builtin.make("shop-cold")          只认名字，不 import 具体 Agent
 ├─ learn()                                   学习阶段：只跑 fit，强制不注入
 │   └─ core/runner.run_tasks()               runner.py:138
 │       └─ run_task()                        runner.py:82，崩溃隔离 105-122
 │           └─ agent.run(task, ctx)          协议 core/protocol.py:65
 │               └─ Trace                     core/trace.py（Turn 对话层 + Event 执行层）
 │       → evaluator.evaluate(trace, task)    只读 Trace，不回写
 │       → memory/extract.py                  失败类型 → 可执行动作
 │       → memory/quality.py                  闸门：不合格的丢弃并写明原因
 │       → memory/store.add()                 追加写 jsonl，同 exp_id 合并
 └─ verify()                                  holdout 同一批跑两遍
     ├─ 基线侧：experience = ""                不检索，控制变量
     └─ 注入侧：retrieve → inject.render → provider 回调
         └─ run_task(experience=渲染文本, hits=[...])
     → 两侧各自 evaluate
     → record_injection()（分母）/ record_outcome()（分子）
```

两条旁路，都不进 pipeline：

- `scripts/probe.py` 直接调 `run_task`，唯一变量是 `experience`，用来做定点因果对照。
- `scripts/distill.py` 只读 `fit.jsonl`（`core/trace.load_traces`），不联网、不调模型。

四条边界都在代码里，不是口头约定：

1. `core/` 不 import `evaluator/`、`memory/`、`agents/`。经验以 `ExperienceProvider` 回调
   交给 `run_tasks`（`runner.py:34`），Runtime 从不主动查经验库。
2. `evaluator/` 只读 `Trace`、只返回 `Verdict`，不修改轨迹、不调 Agent。
3. `memory/` 依赖 `Trace` 与 `Verdict`，不依赖执行循环。
4. `pipeline/` 是唯一同时接触四者的地方，它属于编排，不属于 Runtime。

## Evaluator 怎么判（`evaluator/verdict.py`）

单轮判定分五段，任何一段不过，整轮不过：

| 段 | 检查什么 | 位置 |
|---|---|---|
| L1 工具选择 | `expect_tools` 是否都调过 | `verdict.py:208-215` |
| L2 参数 | `expect_args` 是否命中（任一调用满足即可） | `217-228` |
| L5 调用顺序 | `expect_order` 的子序列匹配 | `230-248` |
| L3 依据溯源 | `grounded_claims` 的值能否在指定工具的返回里找到 | `250-253`，`claims.py` |
| L4 终态事实 | `must_have` / `must_have_any` / `must_not_have`（支持 `re:`） | `255-264` |

失败归因有 15 类、两个层级：先是 `FAILURE_PRIORITY`（`verdict.py:29-45`）保证
"没真正执行"的类别（`infra_error` / `no_execution` / `empty_answer` / `invalid_tool` /
`forbidden_action` / `max_steps`）不被误记成模型的错；轮内则按固定顺序取第一个命中
（`279-296`）：`ungrounded` > `hallucination` > `missing_tool` > `wrong_order` >
`wrong_arg` > `non_answer` > `wrong_fact`。

这个顺序会让标签看起来不对：一轮里若同时用错参数（`wrong_arg`）并复述了禁用数字
（`hallucination`，即 `must_not_have` 命中），最终只会标 `hallucination`。
**查根因必须回到 `Trace` 里的工具实参，不能只看标签。**

## 为什么 ReAct 循环写在 Agent 里，不写进 Runtime

判定原则：**换一个 Agent 还要用的 → Runtime；体现某个 Agent 怎么想的 → Agent 本体。**

- "怎么拆任务、先查谁后查谁、工具失败后重试还是换参"属于后者，写在
  `agents/shop/core/{agent,planning,recovery}.py`。
- "循环骨架、步数预算、崩溃隔离、轨迹落盘、经验检索与注入"属于前者，写在 `core/`。

合并会让 Runtime 绑死在一个 Agent 的推理方式上，第二个 Agent 进来就得改核心。

## 两个 Agent 的结构差异（证据等级不同，不能并列说）

| | `agents/shop` | `agents/main_agent` |
|---|---|---|
| 上下文 | 全量 history | 窗口裁剪 `keep_turns=3` |
| 跨轮槽位 | 有：映射表在 `agents/shop/view/slots.py`，机制在 `core/context.py` | 有：`view/memory.py` 委托同一份 `core/context.py`，只是映射表不同 |
| 事实卡片 | 无 | 上限 8 条，跨轮保留已查实体 |
| 循环 | ReAct + 结构化计划 + 恢复分类 + 步数耗尽收尾 | 原有 ReAct，无计划 |
| 运行证据 | `runs/full-cold`、`runs/main-a2`、`runs/full-plan` 三批全量（各 60 次执行） | **无**。只有 `tests/test_main_adapter.py` |

**目录名有坑，写报告时别按名字猜**：`runs/main-a2/20261119-194423` 里的 `agent` 字段是
`shop-cold`，不是 `main_agent`。历史上按目录名归档过一次，结论错了；以 `report.json`
里的 `agent` 为准。

两者走同一个协议、同一份经验挂载措辞（`agents/experience.py`），跑同一套判定，
产出同构的 `Trace`。**这目前是协议级证据，不是"第二个 Agent 跑通了"的证据**：
`--agent main` 能构造出来，但没有任何一条基线数据，报告里不许写"两个 Agent 都验证过"。

## 证据分级（写报告和答辩按这个说，不许升级）

**A. 有运行证据**

- 工具调用：5 个接口，`max_steps` 由 `ctx.max_steps` 单一来源决定
- 多轮上下文（shop）：跨轮槽位来自工具实参反填（`core/context.py`），
  不解析用户原话；槽位有值才会渲染进提示词
- 结构化计划（shop）：多意图题先拆待办 → 渲染进提示词 → 收尾时若有待办未覆盖顶回一次
- 错误恢复（shop）：工具返回按 `retry / arg / missing / ok` 分类，只有 `retry` 自动重试；
  步数耗尽时用已有结果收尾，不交白卷
- 闭环链路本身：fit 失败 → 提炼 → 闸门 → 入库 → holdout 检索 → 注入 → 回填，
  每一环都在 `report.json` 里有对应字段
- **Workflow 引擎**：`wf:canvas-demo` 在 `wf_eval` 上 4/6（两条 `missing_tool` 均指向画布缺节点）；
  `wf:shop-demo` 5/6；`wf:branch-demo` 走 true 支时另一支**零事件**；
  `condition` / `memory` 事件不计入工具判分（`all_calls()` 只取 `tool_call`）
- **参数抽取**：`{{extract.order_id}}` 上线后 `wf_eval` 从 1/3 变 2/3，W03 的 `wrong_arg` 消失
- **跨轮上下文与槽位**：`wf_multi` 3/3，`turns=6`、`calls=12`；M03 第二轮输入「那物流到哪了」
  里没有任何数字，两个工具仍拿到 `1002`，轨迹有 `slot_reuse` 事件（`from_turn=0`）
- **HTTP API 与画布**：`api/app.py` 监听 8020，`/health` 返回 `agents:11`（6 内置 + 5 个 `wf:*`）；
  `web/builder/` 能拖拽连线、生成 JSON、保存并直接运行
- **数据回流**：`scripts/distill.py` 支持实验目录与任意轨迹文件两条输入；
  `scripts/build_sft.py` 把 67 条候选分成 train 26 / holdout 0 / unknown 18 / 隔离 41
- **来源信息**：`runs/<stamp>/manifest.json` 与 distill 的 `provenance` 记数据集哈希、Git commit 与 dirty
- **经验归并**：8 个库 76 行 → 9 条唯一经验（`data/memory/experiences.jsonl`），`wf:*` 默认读它

**B. 已实现，但只有单元测试**

- `main_agent` 接入（裁剪 / 事实卡片 / 槽位委托）：`tests/test_main_adapter.py`
- 依赖方向护栏（`core.registry.DEFAULT` 恒空、`core/` 不 import 业务）
- 经验质量闸门的多条规则：真实数据只产生过 9 条唯一经验，多数规则未被真实样本命中
- `Trace` 回放：`load_traces`（`core/trace.py`）可用，主表还没用它做过二次分析

**C. 未实现**

- 跨会话用户偏好（千人千面）：零
- 向量检索：零，检索是形态硬门槛 + 字符二元组
- RL / reward 模型 / 模型微调：零。全程调 API 模型，没有权重，**本机也没有训练环境**
- 并发执行、超时控制、重试退避策略：零
- 多 Agent 协作：零

## A1 计划 / A2 跨轮槽位：实现位置、证据、归因边界

| | A1 结构化计划 | A2 跨轮槽位 |
|---|---|---|
| 实现 | `agents/shop/core/planning.py`：`needed()` 门（多意图才开火）、`make_plan`、`uncovered`、`PLAN_RETRIES=1`；接入 `agents/shop/core/agent.py:380` | 机制 `core/context.py`（`update_slots` / `render_slots`，纯 Python、零业务词）；映射 `agents/shop/view/slots.py`；接入 `agents/shop/core/agent.py:249,369,381` |
| 落盘 | `Trace.meta.plan` / `plan_uncovered` / `plan_error` | `Trace.meta.slots` |
| 证据 | M08 重复 3 次：计划成功→PASS、计划失败→FAIL，6/6 吻合 | M07 零注入 6 次：槽位 6/6 有值（`category` + `budget_max`），5/6 通过；基线里 M07 通过时只调了 3 次 `search_products`，说明预算确实沿用了 |
| 归因边界 | 只有 1 道题、6 次；`needed()` 是关键词门，泛化性未验证 | A2 上线前后 M07 从 1/13 变成 5/6，但**同期叠加了 A1**，不是干净归因；只能声称"机制确实被触发并生效"，不能声称"提升 X%" |

两个机制共同的边界：它们都在 Agent 层（`agents/shop/core/`），**不是 Runtime 的能力**。
换一个 Agent 想要计划或槽位，得自己接——机制（`core/context.py`）可以复用，
策略（什么时候拆、留哪些槽）必须自己写。

## Memory 的效果：本轮没有证明

必须写清楚，否则整份文档会被一句话推翻。

主表（`runs/main-a2/20261119-194423`，`agent` 字段是 **`shop-cold`**，不是 `main_agent`；
任务集 = shop 20 + multi 10 + hard 13 = 43 题，fit 26 / holdout 17）：

- 基线 **16/17**，注入 **15/17**，差值 **-1**
- **有效注入覆盖 2/17**：只有 S12、S20 检索到了经验（各 213 字）
- 这 2 条在基线和注入**两轮都 PASS**，没有提升空间
- 三条翻转（M05 ↑、S04 ↓、S14 ↓）**全部落在注入 0 字的任务上** → 是抖动

所以这张表能得出的唯一结论是：**注入没有造成整体损害，也没能证明任何提升**。
不能写成"覆盖不足所以暂时看不出来"。定点对照（M07 / M10）有正证据，但样本是个位数，
且 A1、A2、经验注入三个因素叠在一起，不能单独归因给 Memory。

## 已知的结构性限制

1. **经验只有一种粒度**："某失败类型 → 该补什么动作"。没有任务摘要、没有用户画像。
2. **检索是形态硬门槛 + 字符重合**，同义改写会漏；跨形态（单轮↔多轮）经验不通。
3. **学习阶段强制不注入**，所以同一次实验里学不到第二轮；迭代学习未解决。
4. **判分是规则不是语义判官**，`must_have` 靠字面/正则；H04 那种"还没有签收"
   只能靠改正则救，改的是数据不是判定器。
5. **`temperature=0` 不等于确定**：同一题同一模型，基线在 FAIL/PASS 之间翻转出现过，
   所以任何单次翻转都不能当结论，必须重复对照。
6. **天花板换过位置，两个阶段要分开说**：早期主表的天花板是"经验太少"——fit 26 条只失败
   1 条、只提炼出 1 条经验，holdout 只有 2 条命中，那批数据不可能体现经验效果。
   P7 在 `wf_learn` 上做到了**覆盖 4/4、每题 238 字**，说明覆盖本身不是天花板；
   那一次的天花板变成了"基线已满分"（4/4 vs 4/4）。所以现在缺的不是更严或更宽的检索，
   是**既有区分度、又能被经验纠正的任务**。
7. **`support` 口径偏宽（已知缺陷，未修）**：注入后该任务通过就 `support += 1`
   （`memory/store.py:148-151`），不要求"基线本来失败"。本次 S12/S20 基线本来就过，
   support 仍从 1 涨到 3。这个数会进检索排序（`memory/retrieve.py:134`），
   等于让经验在没有证据的情况下更容易被检索到。正确口径应为"基线失败且注入后通过"。
