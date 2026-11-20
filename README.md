# test_agent — 可插拔 Agent 运行时

一个统一的 Agent Runtime：**按名字加载 Agent → 统一执行 → 记录完整轨迹 → 判分 → 从失败里提炼经验 → 下次相似任务再注入**。

它不是"给某个客服 Agent 做的评测脚本"。这里有三条独立的能力线：

| 能力线 | 目录 | 一句话 |
|---|---|---|
| Runtime | `core/` | 协议、注册、执行、崩溃隔离、轨迹落盘，**不 import 任何业务代码** |
| Agent | `agents/` | 具体怎么想、怎么调工具。换 Agent 只加适配层，不动 Runtime |
| Learning | `memory/` `pipeline/` `evaluator/` | 判分 → 提炼 → 闸门 → 存储 → 检索 → 注入，外加轨迹蒸馏 |

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

定义放在 `data/workflows/<id>.json`，支持的节点是 `start / llm / tool / end`：

```json
{"id": "lookup", "type": "tool",
 "config": {"tool": "get_order", "args": {"order_id": 1002}},
 "next": ["explain"]}
```

- 引用前面的产出用 `{{input}}`、`{{nodes.<节点id>.output}}`；引用不存在的节点会**直接报错**，不会悄悄渲染成空串。
- 定义不合法（类型错、指向不存在的节点、有环、节点不可达、工具没注册）在**构造阶段**就报出全部问题，不会跑到一半才崩。
- 执行过程写在 `Trace.events` 里（`node_start` / `node_end` / `tool_call` / `llm_call` / `error`）；`Trace.turns` 仍然只表示"用户说了几轮"，单轮就是 1 个，节点不写进 Turn。

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

当前 **372 passed**。数字对不上说明改动破坏了既有契约，先查再往下走。

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

- 节点只有四种：`start` / `tool` / `llm` / `end`——这是后端 `NODE_TYPES` 的真实范围。`condition` 与 `memory` 节点还没实现，画布上也不摆假组件
- 工具下拉来自 `GET /tools`，与运行期注入的是同一份工具池；该接口还返回每个工具的参数表（名字/类型/必填/说明），新建 Tool 节点默认选 `get_order`（只读）
- 参数按行填：一行一个参数，标出类型与「必填」，留空时保存会被拦下并指名缺哪个；值右侧「插入变量」可插 `{{input}}` 或上游节点输出，不必手打。复杂场景可展开「高级：直接编辑参数 JSON」，两边同步
- 节点坐标存在定义的 `meta.layout` 里，保存后重新打开位置不变；执行参数只认 `config`，坐标不会被当成工具参数
- 保存走 `POST /workflows`：结构或图不合法时，后端列出的全部问题会显示在画布右上角
- 运行时先保存再执行（跑的是磁盘上的定义，不是画布里的草稿）；轨迹面板显示 `轮次 N · 事件 M`，单轮 Workflow 恒为 `轮次 1`，节点数只影响事件数

画布是纯静态文件（`web/builder/`，零依赖、无 npm、无 CDN）——本机没有 Node 工具链，而答辩现场断网也不能白屏。它产出的就是 `data/workflows/<id>.json`，与手写 JSON 完全等价，两者可以互相覆盖。

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

- 轨迹落盘：`runs/<批次>/<stamp>/{fit,holdout-baseline,holdout-injected}.jsonl`
- 汇总：`report.json`（通过数、翻转明细、逐题注入字数、检索覆盖）
- 经验库：`experiences.jsonl`（追加写，含 `support` / `injected` / `last_outcome`）
- 判分是纯函数：**不调模型、只读落盘轨迹也能复算出主表通过数**（命令见 `docs/runbook.md`）

一条必须守住的纪律：**注入 0 字的题不能算进提升归因**，它和基线条件相同，翻转只能是抖动。

---

## 6. 当前真实进度（按证据分级）

| 等级 | 内容 |
|---|---|
| 有运行证据 | Runtime 统一执行与崩溃隔离（60 次执行 0 crashed）；Trace 记录与回读；五层判分；经验检索与注入；双跑对照；轨迹蒸馏（25 SFT + 1 失败） |
| 只有单测 | `main_agent` 接入（窗口裁剪、事实卡片），从未进过全量实跑 |
| 效果未证明 | **Memory 提升正确率**：主表基线 16/17、注入 15/17、有效覆盖仅 2/17，且这 2 条基线本来就过 → 结论是 null，不是提升 |
| 未实现 | Web UI / HTTP API、跨会话用户偏好、向量检索、RL/DPO/微调、并发与超时控制、多 Agent 协作 |

完整数据与归因见 `reports/final_20261119.md`。

---

## 7. 已知局限

1. **`temperature=0` 不等于确定**：同一题同一模型在 PASS/FAIL 间翻转出现过，任何单次翻转都不能当结论。
2. **检索覆盖是天花板**：26 条 fit 只提炼出 1 条经验，只命中 2 条 holdout 任务。检索没坏，是失败样本太少。
3. **`support` 口径偏宽**：`pipeline/verify.py` 回填时未要求"基线失败、注入后才过"，已知缺陷未修。
4. **词表判语义是死路**：判分靠规则 + 正则，`re:` 只是把枚举写紧凑，不是语义理解。
5. **规模小**：43 题、单域、5 个工具、无并发、无真实流量。价值在方法，不在规模。

---

## 8. 文档

| 文档 | 内容 |
|---|---|
| `docs/architecture.md` | 六模块调用链、判分优先级、A1/A2 归因边界、证据分级 |
| `docs/runbook.md` | 全流程命令、probe 判读、环境变量、接新 Agent 五步、异常对照 |
| `docs/demo.md` | 三条可现场演示的最小流程（含"不要演示什么"） |
| `reports/final_20261119.md` | 最终实验报告：主表、逐条翻转归因、已知缺陷、答辩能说/不能说 |
