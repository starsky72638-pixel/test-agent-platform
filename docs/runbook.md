# 运行说明

所有命令在项目根目录、PowerShell 下执行，使用项目自带虚拟环境。
命令**只能由人执行**，助手不代跑——这是这个项目的硬规矩。

## 0. 准备

```
.venv\Scripts\python.exe -V          # 3.12
```
`.env` 里需要有模型 key 与 `MODEL_NAME`（见 `config/settings.py`）。
没有额度时所有任务会变成 `crashed`，此时先验模型通道：

```
.venv\Scripts\python.exe -c "from config import build_llm, load_settings; print(build_llm(load_settings(), streaming=False).invoke('说：ok').content)"
```
输出 `ok` 才算通道正常。报 `403 Access to model denied` = 账号没有该模型权限/额度，与代码无关。

再确认两组环境变量**空着**——它们是 PowerShell 会话级的，设过就一直留在这个窗口里：

```
Get-ChildItem Env:SHOP_*
Get-ChildItem Env:FAULT_*
```

两条都必须无输出。`SHOP_PLAN=0` 会让计划分支白做；`FAULT_*` 残留会在每次实验里
注入一次 500，`probe` 里表现为 `重试=1次`。

## 1. 起被测服务

窗口 A（**保持不关**）：

```
.venv\Scripts\python.exe mock_server.py
```
监听 `:8000`，提供 5 个接口。`/admin/reset` 重置退款状态与故障计数，
`probe` / `run_loop` 每次执行前都会调它，保证两组初始条件相同。

## 2. 离线测试（不联网、不花额度）

```
.venv\Scripts\python.exe -m pytest tests -q
```
当前应 **297 passed**（2026-11-19 实测）。数字对不上说明改动破坏了既有契约，先查再往下走。

## 3. 定点对照：一次只切一个变量

`scripts/probe.py` 的作用是回答"注入的这段经验，到底有没有改变 Agent 的行为"。
它固定题、固定提示词，只切注入文本，每组重复 N 次。

```
# 无注入 / 注入各 3 次
.venv\Scripts\python.exe scripts\probe.py --task M07 --repeat 3 --experience "……经验原文……"

# 两组都不注入，用来量抖动幅度
.venv\Scripts\python.exe scripts\probe.py --task S02 --repeat 2
```

看五样东西：

| 打印 | 含义 |
|---|---|
| `注入=N字` | 取自 `Trace.meta.experience_chars`，为 0 说明文本没进 Agent |
| `重试=N次` | 工具层自动重试次数，为 0 说明没有故障 |
| `计划=N项/漏N` | 本轮拆出的待办数与未完成数，`计划=0项` 说明门没开或没拆出来 |
| `计划失败=N轮(原因)` | 门开了但没拆出待办，与"不需要计划"是两件事 |
| `槽位=N项` | 最后一轮累积到的槽位数。为 0 说明工具实参没被反填（先查映射表写错没有） |

末三行是两组通过数、工具序列是否变化、判定。注意两种情形：

- **两组都注入 0 字**：这是抖动基准，末行会写"条件相同，通过数差异即抖动，
  不构成经验效果对照"。**这种运行不能当经验有效或有害的证据。**
- **判定行显示"通过数相同"但工具序列变了**：说明经验改变了过程、没改变结果，
  也算有信号，但不能当正确率提升。

`crashed` 会额外打印 `err:` 一行——那才是真正的原因。

## 4. 全量闭环

```
.venv\Scripts\python.exe scripts\run_loop.py --out runs\main-a2
```

两阶段：fit 只跑不注入（学经验），holdout 跑两遍（基线 / 注入）。
这条命令默认使用 shop 域三个任务集（20 + 10 + 13 = **43 题**，fit 26 / holdout 17），
单次全量约 **10.5 分钟**（2026-11-19 实测 631s）。其余任务集要靠 `--golden` 显式指定：

| 任务集 | 条数 | 用途 |
|---|---|---|
| `data/shop_eval.jsonl` / `shop_eval_multi.jsonl` / `shop_eval_hard.jsonl` | 20 / 10 / 13 | 原生 Agent 主表 |
| `data/customer_eval.jsonl` | 6 | customer 域 |
| `data/wf_eval.jsonl` | 6 | Workflow 评测（W01–W06） |
| `data/wf_learn.jsonl` | 7 | Workflow 闭环（fit 3 / holdout 4） |
| `data/wf_multi.jsonl` | 3 | 多轮 + 槽位复用（M01–M03） |

产物在 `runs/<out>/<stamp>/`：

```
fit.jsonl                 学习阶段轨迹
holdout-baseline.jsonl    基线组
holdout-injected.jsonl    注入组
experiences.jsonl         本次学的经验（含每次回填的版本行）
report.json               汇总（含逐题翻转、注入明细、覆盖率、可判分母）
manifest.json             来源：Agent、数据集 SHA-256、split 条数、全部参数、Git commit/dirty
```

**读结果的三步，顺序不能颠倒：**

1. 看 `检索覆盖：n/17`。为 0 说明经验根本没被检索到，后面两行不用看了。
2. 看 `逐题注入明细`。**逐条核对每个翻转题的注入字数**：
   0 字的翻转 = 抖动，无论涨跌都不算经验效果。2026-11-19 那次
   `基线 16/17 / 注入 15/17 / 覆盖 2/17`，三条翻转全是 0 字题 —
   正确结论是"未证明"，不是"经验有害"。
3. 只有"翻转了 **且** 注入了非 0 字"的题，才可以进归因表。

## 5. 数据回流

```
.venv\Scripts\python.exe scripts\distill.py --run runs\main-a2\20261119-194423 --out data\distill-main-a2
```

默认只读 `fit.jsonl`（学习阶段，没有注入经验，最干净）。
产物：`sft.jsonl`、`failed.jsonl`、`manifest.json`。

**统一用 `--out` 指定本次目录，不要覆盖默认的 `data/distill`**：
那份是 2026-11-18 那次实验的产物，覆盖掉就等于把历史证据换了。
当前交付的是 `data/distill-main-a2/`：**25 条 SFT + 1 条失败样本**（S13 的 `missing_tool`）。

三道闸门：工具报错的轨迹不进 SFT；判分没过的轨迹不进 SFT（进 `failed.jsonl` 并带
`failure_type` / `fail_turn`）；`(用户输入, 工具与参数序列, 答案)` 相同只留一条。

**任意轨迹文件也能直接蒸馏**（Workflow 轨迹在 `runs/*.jsonl`，不符合实验目录约定）：

```
.venv\Scripts\python.exe scripts\distill.py --trace runs\wf-report-demo-20261121-103250.jsonl --golden data\wf_multi.jsonl --out data\distill-wf-probe
```

`--golden` 必须一起给：`distill` 按 `task_id` 查断言表，用默认的 shop 任务集去判
Workflow 轨迹会**静默错配**（`wf_multi` 与 `shop_eval_multi` 都用 `M01/M02` 这种 id）。
`manifest.json` 的 `provenance` 会记下输入轨迹的 SHA-256、条数、输出条数与代码 Git 状态。

**生成训练集之前先过质量闸门**：

```
.venv\Scripts\python.exe scripts\build_sft.py
```

读 `data/distill{,-main-a2,-wf}/sft.jsonl` 与 `data/train_sft.jsonl`，产出
`data/sft/{train,holdout,quarantine}.jsonl` + `report.json`。判定顺序是
**结构校验 → split 分流 → 池内四分类**（`duplicate` / `rerun_variance` / `unresolved` / `true_conflict`），
canonical 只按（源文件路径, 行号, sample_id）取第一条，不看答案长度。
当前结果：67 条 → train 26 / holdout 0 / unknown 18 / 隔离 41。

**经验库归并**（8 个实验目录 76 行 → 9 条唯一）：

```
.venv\Scripts\python.exe scripts\merge_memory.py --dry-run
.venv\Scripts\python.exe scripts\merge_memory.py
```

先看后写；输入库只读，不删不改。写出的 `data/memory/experiences.jsonl` 就是 `wf:*` 的默认库。

## 6. 环境变量（对照实验只切这些，不回滚代码）

| 变量 | 默认 | 作用 |
|---|---|---|
| `SHOP_RETRY_LIMIT` | 1 | 工具失败自动重试上限。`0` = 改动前的行为 |
| `SHOP_PLAN` | 未设置=开 | 结构化计划与"待办未完成顶回去"。写 `0`/`false`/`no`/`off` 才关（`planning.py:120-123`） |
| `FAULT_PATH` | 空 | mock 服务故障注入的路径前缀，如 `/products/search` |
| `FAULT_STATUS` | 500 | 注入的状态码 |
| `FAULT_TIMES` | 1 | 每次 `reset` 后注入几次 |

注意：`FAULT_PATH` 是在**起 mock_server 时**读的。忘了取消，
之后每次实验都会带一次故障（`probe` 里显示 `重试=1次`），
结论仍可用但要在报告里注明。

## 7. 复现"经验让 Agent 变好"的最小流程

⚠️ 这条流程**不再以 0/3 为前提**：A2 跨轮槽位上线后，M07 在零注入下已经是
6 次 5 PASS（2026-11-19 实测）。这是能力提升，同时也意味着 M07 已经没有
"被经验救回来"的空间——原来的 0/3 → 3/3 复现不出来了。

按下面做，并说清每一步证明的是什么：

1. 起 mock_server（干净，不设 `FAULT_*`）。
2. 先测基线：`probe --task M07 --repeat 3`。**记录实际数字**，不要假定是 0/3。
3. 再测注入：`probe --task M07 --repeat 3 --experience "<经验原文>"`。
4. 经验原文从 `runs/*/experiences.jsonl` 复制 `action` 字段，不要自己重写——
   自己重写只能证明"注入通道有效"，证明不了"系统学到的经验有效"。
5. 判读：两组都没差距 → 写"该题已无提升空间"；有差距但组内自相矛盾 → 写"抖动"。
   两种都不能写成提升。

## 8. 接一个新 Agent

1. 在 `agents/<name>/` 下放 `core/`、`tools/`、`view/`、`llm.py`，参考 `agents/shop`。
2. 实现 `name` 与 `run(task, ctx) -> Trace`，`max_steps` 从 `ctx.max_steps` 取，不要自建第二个来源。
3. 在 `agents/builtin.py` 的工厂表里登记名字。
4. **不要**往 `core/registry.DEFAULT` 里塞实例——那里恒为空，有守卫盯着。
5. `pytest tests` 应全绿且核心测试照旧通过；`scripts/run_shop.py --list-agents` 应能看到新名字。

现状：`--agent main` 能构造、能跑单测，但**没有任何基线数据**。
不要拿它和 shop 并列写"两个 Agent 都验证过"——它缺的是实跑，不是代码。

## 9. 常见异常对照

| 现象 | 多半是 |
|---|---|
| 六次全 `crashed`、耗时几百毫秒 | 模型通道（额度 / 权限 / 限流），看 `err:` |
| `注入=0字` 且提示"文本没进 Agent" | 没传 `--experience`，或经验为空 |
| 判定 `missing_tool` 但回答看着对 | 这题要求"重新查"，答对但没调工具也算失败 |
| 同一题时过时不过 | `temperature=0` 不等于确定，必须重复 N 次再下结论 |
