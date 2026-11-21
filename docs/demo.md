# 演示脚本

五条演示，各自证明一件不同的事。演示零、四不需要模型额度；演示一至三走本地 mock 服务，不访问外网业务系统。

前提：两个窗口。窗口 A 起服务（**不要设 `FAULT_*`**），窗口 B 执行命令。

```
# 窗口 A
.venv\Scripts\python.exe mock_server.py

# 窗口 B 若要演示界面，再起 API
.venv\Scripts\python.exe api\app.py
```

---

## 演示零：把 Agent 画出来、存下来、跑起来（约 4 分钟，零模型调用）

**命题**：Agent 不是写死在代码里的——可视化成一张图，保存成 JSON，立刻能被 `wf:<id>` 寻址并执行。

1. 浏览器打开 `http://127.0.0.1:8020/ui/builder/`，左侧拖 `Start → Tool → LLM → End`，连线。
2. 点工具节点，参数面板选 `get_order`，`order_id` 填 `{{extract.order_id}}`（**这一步是重点**：参数随输入走，不是写死的）。
3. 点「保存」，记下工作流 id（如 `my-demo`）。服务端落在 `data/workflows/<id>.json`。

```
.venv\Scripts\python.exe scripts\run_agent.py --agent wf:my-demo --question "帮我查下订单 1001 的金额" --max-steps 6
```

期望看到：`节点执行 4 个 | 事件 10 条`，事件里 `tool get_order {"order_id": "1001"}`，回答含 `899`。
**要讲清的一点**：`轮次 1` 而事件 10 条——节点不写进 Turn，`Turn` 只表示"用户说了几轮"。

同一份定义也能直接进评测，用的是和原生 Agent 完全相同的判分器：

```
.venv\Scripts\python.exe scripts\run_shop.py --agent wf:canvas-demo --golden data\wf_eval.jsonl --no-compare
```

期望：`通过 2/3`，`FAIL` 那条归因 `missing_tool`（画布没画物流节点），**不带 `[节点]` 后缀**——缺的是节点本身，没有哪个已存在节点该负责。

---

## 演示一：跨轮不丢事实（约 3 分钟）

**命题**：三层对话里，第 3 轮用户不再提预算，Agent 仍然沿用第 2 轮的 900 去搜，
且这个"900"来自工具实参反填的槽位，不是从聊天记录里翻出来的。

```
.venv\Scripts\python.exe scripts\probe.py --task M07 --repeat 3
```

期望看到：每行都有 `槽位=2项`（`category` + `budget_max`）或 `3项`（多一个
`last_product_id`，取决于有没有调 `get_product_detail`）；6 次里约 5 次 PASS；
第 3 轮回答里出现 899 而不是 429。

**必须主动交代**：这条题**已经不能用来演示"经验效果"了**。
2026-11-19 之前它零注入 13 次 12 失败，A1/A2 上线后零注入 6 次 5 通过——
提升已经体现在基线上，所以再拿它比"注入 vs 不注入"会得到"两组都过"，
那是没有区分度，不是经验有效。

想演示经验效果，得先找到"基线仍失败、且能检索到经验"的题。
目前主表里没有这样的题（覆盖 2/17，且那 2 条本来就过），**这一点要如实说**。

---

## 演示二：结构化计划让多意图任务不再只答一半（约 8 分钟）

**命题**：M08 第 5 轮用户说"那两个订单加起来一共多少钱"，基线会凭前几轮的记忆直接算，
答对了金额但一次工具都没调；判定按"金额必须重新查"判失败。计划层会把它顶回去。

```
$env:SHOP_PLAN="0"; .venv\Scripts\python.exe scripts\probe.py --task M08 --repeat 2
$env:SHOP_PLAN="1"; .venv\Scripts\python.exe scripts\probe.py --task M08 --repeat 2
```

期望看到：

| | 结果 | 第 5 轮工具 | 打印 |
|---|---|---|---|
| `SHOP_PLAN=0` | `FAIL missing_tool@t5` | 无 | 无 `计划=` |
| `SHOP_PLAN=1` | 多数 `PASS`，少数仍 `FAIL` | `get_order ×2` | `计划=N项/漏N` |

2026-11-19 实测（重复 3 次）：**计划调用成功 5/6、任务通过 4/6**，
且"计划成功→PASS、计划失败→FAIL"6/6 吻合。

剩下那两次失败不是计划没拆出来，而是拆出来了（如 `计划=2项/漏2`）但模型回一句
"其实已经查过了"就收尾——ReAct 的固有边界（模型拒绝重查），不是计划模块的缺陷。
**这一点要在演示里主动说**，否则看起来像没修好。

代价也要讲：单次耗时从 20–38 秒涨到 58–144 秒，因为每轮多了一次计划调用。

---

## 演示三：轨迹变成数据（约 1 分钟）

**命题**：跑完的实验不是日志垃圾，能直接转成 SFT 样本与负样本。

```
.venv\Scripts\python.exe scripts\distill.py --run runs\main-a2\20261119-194423 --out data\distill-main-a2
```

期望看到：`[fit] 轨迹 26 条 → SFT 25 条 / 失败 1 条 | 去重丢弃 0 条 | 其它丢弃 无`，
失败类型分布 `missing_tool=1`（就是 S13：先 `apply_refund` 后没查 `get_order`）。

产物在 `data/distill-main-a2/`：`sft.jsonl`（25 条，OpenAI tool_calls 形态）、
`failed.jsonl`（1 条，带 `failure_type` 与 `fail_turn`）、`manifest.json`（计数与丢弃原因）。
**不要覆盖 `data/distill/`**，那份是 11-18 那次实验（22 + 4）的历史证据。

---

## 演示四：轨迹变成数据之前，先过一道闸门（约 3 分钟，零模型调用）

**命题**：不是"能产出 SFT"就算完——要能说清每条样本为什么进、为什么出。这一步不开源数据、不改判分，纯本地。

```
.venv\Scripts\python.exe scripts\build_sft.py
```

期望看到（当前真实输出）：

```
输入 72 条（5 个文件）
有效 72 | duplicate 0 | rerun_variance 21 组（42 条） | true_conflict 1 组（3 条） | unresolved 0 组
缺字段 18 | tool_calls 异常 0
canonical 30 | train 26 / holdout 4 / unknown 18 / 隔离 42
隔离原因：rerun_variance=21  true_conflict=3  unknown_split=18
可训练样本：26
```

四分类要能口头解释清楚：

| 类别 | 判据 | 处理 |
|---|---|---|
| `rerun_variance` | 同 `task_id`+`agent`+`split`，两次运行答案不同 | 保留第一条 canonical，其余留档并标 `canonical_sample_id`，**不删原始样本** |
| `true_conflict` | 不同 Agent 对同一题给出不同答案 | 整组隔离，等人工裁决 |
| `unknown_split` | 来源缺 `split`（手工集没有这两个字段） | 默认不进训练集 |
| `duplicate` | 同源同答案 | 只留一条，其余进隔离并记原因 |

```
.venv\Scripts\python.exe -c "import json;r=json.load(open('data/sft/report.json',encoding='utf-8'));print(r['counts']);print('groups',len(r['groups']),'reruns',len(r['reruns']))"
```

**必须主动交代**：`holdout` 现在有 **4 条**（W02–W05，全来自 `wf:shop-demo` 的真实轨迹，
每条带 `source_trace` 可回溯）。它们原先不存在——评估集此前是 0，因为输入里只有
`distill-wf` 的 2 条 W01，且那两个 Workflow 对同一题给了不同措辞，被判真冲突。
现在 W01 仍隔离（冲突组 3 条），**没有为了凑数放开它**。

还有一点要讲清楚：这 4 条是从 Workflow 轨迹投影来的，`assistant` 消息**按节点逐条拆开**
（一次一个 `tool_calls`），不是模型一次规划出的多调用。它是"该 Agent 实际执行的样子"，
不是"最优解"。训练时学到的是"每步一调"，不是"一次规划"。

**最后一句不能省**：全程没有训练过任何模型，也不宣称任何训练效果。

---

## 讲的时候要主动交代的事

1. **训练侧样本覆盖不足**：fit 侧原本没有 `wrong_arg` 样本，自动提炼产不出对应经验，
   为此补了一道同型题 M10。这不是"系统自动发现"，是补数据。答辩被问到就如实说。
2. **M05 不能算经验有效**：全量实验里它注入 0 字，条件与基线相同却翻转了，属于抖动。
   把它算进提升里是错的。
3. **主表对经验效果没有结论**：基线 16/17、注入 15/17、覆盖 2/17，三条翻转全是 0 字题。
   能说的只有"注入没造成整体损害"，不能说提升，也不能说有害。
4. **`support` 有虚增**：注入后任务通过就 +1，不要求基线本来失败。已知缺陷，冻结期间未修。
5. **第二个 Agent 只有单测**：`main_agent` 能构造、有 `test_main_adapter.py`，
   但从未进过 `run_loop`，没有基线数据，不能并列说"两个 Agent 都验证过"。
   **另外注意目录名有坑**：`runs/main-a2/` 里跑的是 `shop-cold`，不是 `main_agent`，
   以 `report.json` 的 `agent` 字段为准。
6. **Workflow 的注入已经做过，效果是 0**：P7 在 `wf_learn` 上做到覆盖 4/4、每题 238 字，
   基线 4/4 与注入 4/4 持平。原因不是没注入，是**基线已满分**。
   同一批还抓到 1 例误命中（退款经验被注入到问金额的题）。这条要说，不要藏。
7. **SFT 数据能训练但无法评估**：`train` 26 条、`holdout` 0 条。且**全程没有训练过模型**——
   只产出数据，任何"提升 X%"的说法都不成立。
8. **没做的**：跨会话用户偏好、向量检索、RL/DPO/模型微调（本机只调 API，无训练环境）、
   并发与超时、多 Agent 协作。零就是零，不说"做了一部分"。
