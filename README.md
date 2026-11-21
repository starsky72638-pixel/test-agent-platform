# test_agent — 可插拔 Agent 运行时

一个统一的 Agent Runtime：**按名字加载 Agent → 统一执行 → 记录完整轨迹 → 判分 → 从失败里提炼经验 → 下次相似任务再注入**。

它不是"给某个客服 Agent 做的评测脚本"。这里有三条独立的能力线：

| 能力线 | 目录 | 一句话 |
|---|---|---|
| Runtime | `core/` | 协议、注册、执行、崩溃隔离、轨迹落盘，**不 import 任何业务代码** |
| Agent | `agents/` | 具体怎么想、怎么调工具。换 Agent 只加适配层，不动 Runtime |
| Workflow | `agentplatform/` | 把 Agent 画成一张 JSON 图：分支节点、记忆节点、参数抽取、跨轮上下文与槽位。产物同样是 `Trace`，走同一套判分 |
| Learning | `memory/` `pipeline/` `evaluator/` | 判分 → 提炼 → 闸门 → 存储 → 检索 → 注入，外加轨迹蒸馏与 SFT 质量闸门 |
| 门面 | `api/` `web/builder/` | HTTP（8020）与拖拽画布，只经 `agents/builtin` 造 Agent |

购物客服 Agent 是第一个接入者，不是项目的全部。

---

## 1. 跑起来（Windows / PowerShell）

```powershell
# 1. 依赖
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt

# 2. 配置（四项，见 .env.example）
#    已有 .env 就跳过这两行，不要覆盖你填好的密钥
Copy-Item .env.example .env
#    编辑 .env，填入 OPENAI_API_KEY

# 3. 起假后端（另开一个窗口，常驻，端口 8000）
.venv\Scripts\python.exe mock_server.py
```

确认通道正常：

```powershell
.venv\Scripts\python.exe scripts\run_agent.py --list
```

```
可接入的 Agent：
  customer
  customer-cold
  main
  shop-cold（默认）
  shop-tuned
  wf:demo
  workflow
别名：cold -> shop-cold, cs -> customer, cs-cold -> customer-cold, main_agent -> main, shop -> shop-tuned, tuned -> shop-tuned, wf -> workflow
```

`--list` 不构造 Agent、不读密钥，这一步能过就说明"按名字加载"这一层是通的。

真跑一条（不需要数据集、不需要判分）：

```powershell
.venv\Scripts\python.exe scripts\run_agent.py --agent shop-cold --question "预算 500 以内有什么推荐"
```

输出四段：状态与耗时、逐条工具调用（含实参）、最终回答、`Trace` 落盘路径。

### 用 JSON 定义 Agent（workflow）

一个 Agent 也可以不写 Python，只写一份 JSON：节点 + 连线，平台按拓扑序执行。

```powershell
.venv\Scripts\python.exe scripts\run_agent.py --agent workflow --workflow demo --question "订单 1002 到哪了"
```

定义放在 `data/workflows/<id>.json`，支持的节点是 `start / llm / tool / condition / memory / end`：

```json
{"id": "lookup", "type": "tool",
 "config": {"tool": "get_order", "args": {"order_id": "{{extract.order_id}}"}},
 "next": ["explain"]}
```

- 引用前面的产出用 `{{input}}`、`{{nodes.<节点id>.output}}`；引用不存在的节点会**直接报错**，不会悄悄渲染成空串。
- 参数要随输入变化时用 `{{extract.<字段>}}`：从本轮用户输入里抽，可选 `order_id`（4 位订单号）、`amount`（数字+元）、`product_id`（形如 P1001）。抽不到**直接报错**（该条落 `infra_error` 并被剔除），不会带着空参数去打接口。抽取值是字符串，工具侧由 pydantic 宽松模式转成 `int`。
- 定义不合法（类型错、指向不存在的节点、有环、节点不可达、工具没注册）在**构造阶段**就报出全部问题，不会跑到一半才崩。
- 执行过程写在 `Trace.events` 里（`node_start` / `node_end` / `tool_call` / `llm_call` / `branch` / `memory_call` / `error`）；`Trace.turns` 仍然只表示"用户说了几轮"，单轮就是 1 个，节点不写进 Turn。

### 把 Agent 存到服务端

定义存在 `data/workflows/<id>.json`，一个 id 一个文件，可以直接 diff、进 git。
存下来之后**不需要注册、不需要重启**，用 `wf:<id>` 就能跑：

```powershell
.venv\Scripts\python.exe scripts\run_agent.py --agent wf:demo --question "订单 1002 到哪了"
```

HTTP 侧三个端点（保存时会做结构与图校验，不合法返回 400 并列出全部问题）：

```
GET  /workflows            列出全部定义（坏文件也会出现，带 error 字段）
GET  /workflows/{id}       取一份定义
POST /workflows            保存（覆盖时 version 自动 +1）
```

版本策略是"覆盖即递增，历史交给 git"，没有版本树——`git log -p data/workflows/demo.json` 就是全部改动记录。

单测（离线，不花钱）：

```powershell
.venv\Scripts\python.exe -m pytest tests -q
```

当前 **454 passed**。数字对不上说明改动破坏了既有契约，先查再往下走。

### 通过界面跑（可选）

HTTP 与 Web 界面走的是和 `run_agent.py` 同一条执行链，`core/` 零改动：

```powershell
# API + 界面（另开一个窗口；mock_server 保持运行）
.venv\Scripts\python.exe api\app.py
```

浏览器打开 `http://127.0.0.1:8020/ui/`，选 Agent、输入问题，右侧显示逐轮工具调用与原始返回。多行输入会被当作多轮任务一次提交。

- 四个端点：`GET /health`、`GET /agents`、`POST /runs`、`GET /runs/{run_id}`
- **同步执行**：请求等跑完才返回，单条约 10–50 秒
- 无鉴权、只监听 `127.0.0.1`，不做公网假设
- 端口 8020：8000 是 mock_server，8010 留给本机另一个项目

### 拖拽搭一个 Agent（画布）

浏览器打开 `http://127.0.0.1:8020/ui/builder/`：左边选节点、中间拖拽连线、右边填参数，底部点运行就能看到节点级事件。

- 工具栏只有四种节点：`start` / `tool` / `llm` / `end`。`condition` 与 `memory` 已能在后端跑，但画布还不提供参数表单，所以不摆出来——打开含这两类节点的定义能只读显示，工具栏不会给一个配不了的空壳
- 工具下拉来自 `GET /tools`，与运行期注入的是同一份工具池；该接口还返回每个工具的参数表（名字/类型/必填/说明），新建 Tool 节点默认选 `get_order`（只读）
- 参数按行填：一行一个参数，标出类型与「必填」，留空时保存会被拦下并指名缺哪个；值右侧「插入变量」可插 `{{input}}` 或上游节点输出，不必手打。复杂场景可展开「高级：直接编辑参数 JSON」，两边同步
- 节点坐标存在定义的 `meta.layout` 里，保存后重新打开位置不变；执行参数只认 `config`，坐标不会被当成工具参数
- 保存走 `POST /workflows`：结构或图不合法时，后端列出的全部问题会显示在画布右上角
- 运行时先保存再执行（跑的是磁盘上的定义，不是画布里的草稿）；轨迹面板显示 `轮次 N · 事件 M`，单轮 Workflow 恒为 `轮次 1`，节点数只影响事件数

画布是纯静态文件（`web/builder/`，零依赖、无 npm、无 CDN）——本机没有 Node 工具链，而答辩现场断网也不能白屏。它产出的就是 `data/workflows/<id>.json`，与手写 JSON 完全等价，两者可以互相覆盖。

画布产出的 Agent 走的是同一套判分器（`data/wf_eval.jsonl`，6 条验证任务）：

```powershell
.venv\Scripts\python.exe scripts\run_shop.py --agent wf:canvas-demo --golden data\wf_eval.jsonl --no-compare
.venv\Scripts\python.exe scripts\run_shop.py --agent wf:shop-demo --golden data\wf_eval.jsonl --no-compare
```

`canvas-demo` 只有 `get_order` 一个工具节点，`shop-demo`（`get_order → get_logistics`，参数同样走 `{{extract.order_id}}`）两个都有。同一个任务集跑两个 Agent，分数与失败项都不该一样——**这就是「平台能评估不同 Agent」的全部意思**，也是这两份定义存在的理由。

- W01/W03/W04 是查得到的题，两个 Agent 都该过
- W02 是物流题：`canvas-demo` 必然 `missing_tool`，`shop-demo` 才有机会过
- W05 查越界订单 9999，只做负向断言（不许冒出别的订单的状态与金额），测"查不到会不会编"
- W06 要求退款：两个 Agent 都没有 `apply_refund` 节点，都该 `missing_tool`——判分器报的是"这个 Agent 不会做"，不是"模型答错"

P6 之前 W03 恒为 `wrong_arg`（订单号写死 1002）。改成 `{{extract.order_id}}` 后它自然变绿，`canvas-demo` 从 `1/3` 变 `2/3`——同一个任务集对同一处改动产生了可观测的差值，这才是评测集有用的证据。

### 让 Workflow 吃 Runtime 的经验

原生 Agent 由自己的 prompt 决定怎么用经验；Workflow 默认不吃——经验只记在 `trace.meta["experience_chars"]` 里。要让某个 llm 节点吃，给它开 `"inject_experience": true`：

```json
{"id": "llm1", "type": "llm",
 "config": {"prompt": "…", "inject_experience": true}}
```

开了之后，`core.runner` 递进来的经验文本会**原样拼在该节点 prompt 最前面**（`memory.inject` 已经把表头、边界声明、尾注都拼好了，平台层不再包一层）。经验为空时**完全不拼**——基线组的 prompt 里不会出现一个空的「【历史经验】」表头。

为什么要有这个开关而不是自动注入：注入是有代价的行为。默认关，等于「这份定义的行为不因有没有经验而变」，基线组和注入组才可比。

配套的实验载体是 `data/workflows/report-demo.json`（双工具 + 极简 prompt，刻意不写行为准则）与 `data/wf_learn.jsonl`（3 条 fit / 4 条 holdout）：

```powershell
.venv\Scripts\python.exe scripts\run_loop.py --agent wf:report-demo --golden data\wf_learn.jsonl
```

### 跨轮上下文

一条 Task 的所有轮次共用一个执行上下文（`Executor.new_context()`），`WorkflowAgent` 每轮把它传下去；换一条 Task 就新造一个，所以不同任务之间**结构性隔离**，不靠"记得清空"。

上下文里五个键各归各的：`input` 每轮覆盖成本轮原话、`nodes` 每轮清空只装本轮产出、`history` 按轮归档已结束的产出、`experience` 整条任务共用、`slots` 存当前有效的抽取参数。**清空是刻意的**：上一轮的产出留在 `nodes` 里，第 2 轮的 `{{nodes.x.output}}` 会读到旧值，看起来跑通了其实是拿旧数据答新问题。

`trace.meta["context_turns"]` 记录上下文实际累积了几轮，这是"复用真的发生了"的机器可验证证据（2 轮任务跑成 1 就说明中途崩了）。

**参数槽位（slots）**：每轮开头把定义里出现过的 `{{extract.X}}` 解析一次——先抽本轮输入，抽到了就写进 `slots`（带上产生它的轮次），抽不到才沿用旧值。沿用会留一条 `slot_reuse` 事件：

```json
{"kind": "slot_reuse", "turn": 2, "name": "order_id",
 "args": {"field": "order_id", "value": "1001", "source": "previous_turn",
          "from_turn": 1, "reused_from_previous_turn": true}}
```

所以「订单 1002 的金额」→「订单 1001 的状态」→「那物流呢」这种追问能跑，且能证明第三轮用的是**最新**的 1001（`from_turn: 1`），不是最早的 1002。两条硬边界：**没有历史值仍然报错**（不猜默认单号），**换一条 Task 就换槽位**（不跨任务继承）。

槽位只在 `Task` 内有效，`WorkflowAgent.run()` 每次调用都新造 ctx；`slot_reuse` 不是工具调用，判分器不认它。真实多轮样例见 `data/wf_multi.jsonl`（M03 就是省略单号那条）：

```powershell
.venv\Scripts\python.exe scripts\run_shop.py --agent wf:report-demo --golden data\wf_multi.jsonl --no-compare
```

### 实验来源（manifest）

`run_loop.py` 每开一次实验，在建目录时就写一份 `manifest.json`：Agent 名、用到的任务文件及其 SHA-256、fit/holdout 实际条数与任务 id、全部运行参数、经验库路径、Python 版本，以及 Git 的 `commit / branch / dirty`。

三条规则写在 `pipeline/provenance.py` 顶部：取不到写 `null` 不猜；Git 失败记 `commit=null dirty=null` 并保留 `git.error`，绝不写成"干净"；`write_manifest` 失败抛 `ManifestError`，`run_loop` 会打印警告后继续（报告仍会落盘，但这次实验缺来源信息）。

先写 manifest 再开跑，所以中途崩掉也留得下配置；老实验目录不动，没有 manifest 就当作"历史产物"。

`distill.py` 在原有 `manifest.json` 里增加 `provenance` 块：每个输入轨迹文件的路径/哈希/条数、输出条数、按组的统计、涉及的 Agent 与 split、以及 distill 代码自身的哈希与 Git 状态。**SFT 的 `messages` 一个字段都不动**，元数据只挂在 manifest 上。

```powershell
.venv\Scripts\python.exe scripts\distill.py --run runs\full-cold\20261118-000730
```

### 经验库归并（默认库）

每次闭环都在 `runs/<时间戳>/` 下新建一个经验库（单次实验要能独立复现），代价是同一个失败模式在多处各存一份：8 个库 76 行，全局只有 9 条不同经验。

`scripts/merge_memory.py` 把散库合成一份：按 `exp_id` 取 `(updated_at, version)` 最新的那条，写到 `data/memory/experiences.jsonl`（`memory.store.DEFAULT_STORE_PATH`）。**不累加 `support/contradict`**——同一批实验被重复跑进多个目录，相加等于把"发生过一次"记成"发生过五次"；**不改任何原库**，先写 `.tmp` 再 `replace`，避免留下半个库。

```powershell
.venv\Scripts\python.exe scripts\merge_memory.py --dry-run
.venv\Scripts\python.exe scripts\merge_memory.py
```

`wf:*` 由 `agents/builtin.py` 默认注入这个库，所以 memory 节点终于能读到经验（此前 `memory_store` 恒为 `None`，`hit_count` 永远是 0）。库不存在就是空库，与"没有经验"等价；要显式关掉传 `memory_store=None`。

`run_loop --store default` 才用归并库，**默认仍是一次实验一个新库**：`record_outcome` 会累加反证并可能自动退休，共用一份库会让同一实验第二次跑出不同结果。

### SFT 质量闸门

`scripts/build_sft.py` 把已有的 SFT 文件过一遍闸门，产出三份互不重叠的数据 + 一份机器可读报告：

```
data/sft/train.jsonl       可训练（split=fit/train，且无结构问题、无答案冲突）
data/sft/holdout.jsonl     只评估（split=holdout/test）
data/sft/quarantine.jsonl  冲突 / unknown 来源 / 结构异常，每条带 reason
data/sft/report.json       计数、隔离原因、冲突清单、输入文件哈希
```

四类判定，按证据强弱从上往下：

- **duplicate**：同 `task_id`+`agent`+`split` 且答案一致——留 `sample_id` 排序第一条，其余记 `duplicate`；
- **rerun_variance**：同 `task_id`+`agent`+`split` 但答案不同——这是同一次实验的两次运行，留 **canonical**（按源文件路径 + 行号取第一条，**不看答案长度、不做相似度打分**），其余记 `rerun_variance` 并带 `canonical_sample_id`，原样本不删；
- **unresolved**：来源里出现 `unknown`，无法判断谁和谁同源——不猜，整组隔离；
- **true_conflict**：多来源之间答案不一致（不同 Agent、不同 `task_id`、手工集与生成集）——整组隔离等人工裁决，不自动合并。

隔离项都带 `reason` 与 `canonical_sample_id`（若适用）；`report.groups` 逐组列出 `sample_id / task_id / agent / split / 文件 / 行号`，`report.reruns` 给出 canonical 与 rerun 的对应。

判定顺序是**先按 split 分流、再在池内判同源与冲突**：unknown 与 holdout 不参与训练侧判定，否则一条来路不明的样本能把同问句的可训练样本一起拖进隔离。

每条样本带 `sample_id / task_id / agent / split / source_trace / tool_calls / messages`，`messages` 原样不动；另外附 `source: {file, line}` 指回原始文件的那一行。

```powershell
.venv\Scripts\python.exe scripts\build_sft.py
```

### 分支与记忆节点（目前只有 JSON 配置）

画布只画 `start/llm/tool/end` 四种节点，`condition` 与 `memory` 需要手写 JSON：

- `condition`：`config.field` 取待判值（可用 `{{...}}`）、`config.op` 比较方式、`config.value` 对照值，
  `config.branches` 写 `{"true": [...], "false": [...]}`。`next` 可以省略（由 branches 派生）；
  写了就必须与并集一致，不一致在加载期直接报错。第一版只有互斥两支：不并行、不做 AND 汇聚，
  未选中的分支连 `node_start` 都不会产生。
- `memory`：`config.query` 是检索词，`top_k` / `min_score` / `max_chars` 可选。
  经验库必须显式注入（`WorkflowAgent(..., memory_store=...)`），不注入就是空——
  刻意不设默认路径，避免一个 Workflow 悄悄读到另一个实验留下的经验。
  轨迹里留 `memory_call` 事件，带命中数量、命中 ID、耗时与注入字数。

两者都不计入工具调用：判分器只认 `tool_call` 事件，`branch` / `memory_call` 不会被算成调了工具。
完整示例见 `data/workflows/branch-demo.json`。

---

## 2. 入口命令，什么时候用哪个

| 入口 | 需要什么 | 用途 |
|---|---|---|
| `scripts\run_agent.py` | 只要 mock + 密钥 | **单条任意问题**试跑，不判分、不入库。新人第一个命令 |
| `scripts\run_shop.py` | mock + 密钥 | 在指定数据集上跑一批并**当场判分**，看通过率与失败类型 |
| `scripts\run_loop.py` | mock + 密钥 + 数据集 | 完整闭环：fit 学经验 → holdout 双跑对照（基线 / 注入）→ 回填效果 |
| `scripts\probe.py` | mock + 密钥 | **定点对照**：同一题重复 N 次，唯一变量是注入不注入 |
| `scripts\distill.py` | 已有 `runs/**/*.jsonl` | 把轨迹蒸馏成 SFT 样本与失败样本，**不调模型** |

常用命令：

```powershell
# 单条
.venv\Scripts\python.exe scripts\run_agent.py --agent shop-cold --question "预算 500 以内有什么推荐"

# 一批 + 判分（1 条，约 10 秒）
.venv\Scripts\python.exe scripts\run_shop.py --agent cold --split holdout --limit 1 --no-compare

# 定点对照（不带 --experience 时两组同条件，只量抖动）
.venv\Scripts\python.exe scripts\probe.py --task M07 --repeat 3

# 完整闭环（26 条 fit + 34 次 holdout，约 10 分钟）
.venv\Scripts\python.exe scripts\run_loop.py --agent shop-cold --out runs\main-a2

# 轨迹蒸馏（约 2 秒，零模型调用）
.venv\Scripts\python.exe scripts\distill.py --run runs\main-a2\20261119-194423 --out data\distill-main-a2
```

`run_loop.py --list-agents` 与 `run_agent.py --list` 输出同一份注册表。

---

## 3. 目录与职责

```
core/         Runtime：protocol（协议）/ runner（执行+崩溃隔离）/ trace（轨迹）/ task（数据与 split）/ registry / context
agents/       Agent：builtin（按名字构造）/ shop（ReAct+计划+恢复+槽位）/ main_agent（窗口裁剪+事实卡片，仅单测）
evaluator/    判分：五层断言 + 失败类型，只读 Trace，不回写
memory/       经验：extract（失败→经验）/ quality（闸门）/ store（jsonl）/ retrieve / inject
pipeline/     learn（fit 学经验）/ verify（holdout 双跑对照）
scripts/      CLI 入口：run_agent / run_shop / run_loop / probe / distill
data/         43 道题（fit 26 / holdout 17）+ 蒸馏产物
runs/         每次执行的轨迹与报告
docs/         架构、运行手册、演示脚本
reports/      实验报告
```

依赖方向是硬约束：`core/` 不认识 `agents/memory/evaluator`，经验以回调形式传进执行循环；判分器只读轨迹；`pipeline/` 是唯一能同时碰四者的地方。

---

## 4. 接入一个新 Agent（不改 Runtime）

1. 新建 `agents/<your_agent>/`，实现协议：`name` 属性 + `run(task, ctx) -> Trace`。
2. 在 `agents/builtin.py` 的工厂表里加一行名字映射（顺带可加别名）。
3. 数据文件每行加 `"split": "fit"` 或 `"holdout"`——**划分由数据自己声明**，不需要改 `core/task.py` 的常量表；不写则默认进 holdout。

现成参照：`agents/customer/` 是照这个流程加进来的第二个业务 Agent（订单 / 物流 / 退款三件套）。它的工具与循环都自包含，不 import `agents/shop`——`tests/test_customer_agent.py` 守着这条边界，也验证了"删掉 shop 照样跑"。

不改 `core/`、`evaluator/`、`memory/`、`pipeline/` 的任何一行，之后即可：

```powershell
.venv\Scripts\python.exe scripts\run_agent.py --agent <name> --question "..."
.venv\Scripts\python.exe scripts\run_shop.py --agent <name> --split holdout --limit 1 --no-compare
```

未实现的部分要说清：新 Agent 若还要判分，断言写在数据文件里，不在 Agent 里；若要用经验，需要在 `run()` 里读取 `ctx.experience` 并拼进提示词（参考 `agents/experience.py`）。

---

## 5. 一次执行留下了什么

```
任务 ──► Agent.run ──► Trace（轮次 / 工具调用 / 实参 / 返回 / meta）
                          ├─► evaluator 判分 → failure_type + fail_turn
                          ├─► memory.extract 提炼经验 → quality 闸门 → store
                          └─► scripts/distill 蒸馏 → SFT 样本 / 失败样本
```

轨迹落盘按入口分四处：

| 入口 | 落盘位置 |
|---|---|
| `scripts/run_loop.py` | `runs/<stamp>/{fit,holdout-baseline,holdout-injected}.jsonl` + `report.json` + `manifest.json` + `experiences.jsonl` |
| `scripts/run_shop.py` | `runs/<agent>-<stamp>.jsonl` |
| `scripts/run_agent.py` | `runs/adhoc/<stamp>/<stamp>.jsonl` |
| HTTP API（`POST /runs`） | `runs/api/` |

- `report.json` 记通过数、翻转明细、逐题注入字数、检索覆盖
- `manifest.json` 记 Agent、数据集路径与 SHA-256、实际 split 条数、全部运行参数、Git commit 与 dirty（见「实验来源」一节）
- `experiences.jsonl` 追加写，含 `support` / `injected` / `last_outcome`；归并后落在 `data/memory/experiences.jsonl`
- SFT 侧：`scripts/build_sft.py` 产出 `data/sft/{train,holdout,quarantine}.jsonl` + `report.json`
- 判分是纯函数：**不调模型、只读落盘轨迹也能复算出主表通过数**（命令见 `docs/runbook.md`）

一条必须守住的纪律：**注入 0 字的题不能算进提升归因**，它和基线条件相同，翻转只能是抖动。

---

## 6. 当前真实进度（按证据分级）

| 等级 | 内容 |
|---|---|
| 有运行证据 | Runtime 统一执行与崩溃隔离；Trace 记录与回读；五层判分与失败定位到节点；Workflow 引擎（分支节点 / 记忆节点 / `{{extract.<字段>}}` 参数抽取 / 跨轮上下文 / 参数槽位复用）；HTTP API（`api/app.py`，8020）与画布 Builder（`web/builder/`）；经验提炼 → 归并 → 检索 → 注入（含 Workflow 侧注入）；轨迹蒸馏；SFT 质量闸门 |
| 只有单测 | `main_agent` 接入（窗口裁剪、事实卡片），从未进过全量实跑 |
| 效果未证明 | **Memory 提升正确率**：P7 在 `wf_learn` 上做到注入覆盖 4/4、每题 238 字，但基线 4/4、注入 4/4、翻转 0——不是"没注入"，而是**基线已满分，没有提升空间**。早期主表同样是 null（基线 16/17、注入 15/17、有效覆盖 2/17） |
| 未实现 | 跨会话用户偏好（千人千面）、向量检索、RL / DPO / 模型微调（本机只调 API 模型，无训练环境）、并发与超时控制、多 Agent 协作 |

证据分流在 `reports/`：能力与产物清单见 `platform_capability_report.md`，注入实验见 `p7_experiment.md`，早期主表与逐条翻转见 `final_20261119.md`。

---

## 7. 已知局限

1. **`temperature=0` 不等于确定**：同一题同一模型在 PASS/FAIL 间翻转出现过，任何单次翻转都不能当结论。
2. **检索门槛过松，有实测误命中**：`W_SHAPE=0.15` 是常数垫底（形态已升级为门槛），只要有一个关键词重合，分数就 ≥0.15，轻松越过 `min_score=0.12`。实测 shop 库对 `customer` 任务集平均命中 1.67 条/任务、对 `wf` 任务集 2.0–2.33 条/任务；P7 里"不要调用 apply_refund"那条经验被注入到问金额的题上。**命中率高不等于检索准。**
3. **没有「行为性失败 → 注入 → 改善」的正例**：现有失败几乎全是 `missing_tool`（画布缺节点），提炼出的经验对 LLM 行为没有影响；换成图配齐的任务后基线又是满分。两个方向都测不出差值，这是当前最实质的空白。
4. **`support` 口径偏宽**：`pipeline/verify.py` 回填时未要求"基线失败、注入后才过"，已知缺陷未修。
5. **词表判语义是死路**：判分靠规则 + 正则，`re:` 只是把枚举写紧凑，不是语义理解。
6. **规模小**：7 个任务集共 65 条、单域、5 个工具、无并发、无真实流量。价值在方法，不在规模。
7. **SFT 数据规模很小，且从未训练**：`build_sft.py` 当前产出 train 26 条、holdout 4 条。holdout 来自 `wf:shop-demo` 的 W02–W05 真实轨迹（每条带 `source_trace`）；同一道题的 W01 由两个 Workflow 给出不同措辞，判 `true_conflict` 隔离，**没有为了凑数放开**。26 条只够验证管线跑通，不足以训练出可评估的模型；**全程没有训练过任何模型**，不宣称训练效果。
8. **画布只能只读展示 `condition` / `memory` 节点**：在画布上改它们的连线再保存，会让 `next` 与 `branches` 不一致，加载时明确报错（不会静默损坏，但会拦住加载）。跨轮 `expect_order` 不可表达。

---

## 8. 文档

| 文档 | 内容 |
|---|---|
| `docs/architecture.md` | 模块调用链、判分优先级、证据分级、结构性限制 |
| `docs/runbook.md` | 全流程命令、probe 判读、环境变量、接新 Agent 五步、异常对照 |
| `docs/demo.md` | 可现场演示的流程（含"不要演示什么"） |
| `reports/platform_capability_report.md` | 平台能力与数据产物总账：Runtime / Trace / Evaluator / Memory / SFT / Workflow，含 P6–P10 实测 |
| `reports/p7_experiment.md` | Workflow 经验注入实验：真实数字、轨迹证据、天花板结论 |
| `reports/final_20261119.md` | 早期主表：逐条翻转归因、已知缺陷、答辩能说/不能说 |
