# 经验闭环实验记录（2026-11-17 ~ 11-18）

这份记录只回答一个问题：**Memory 里的经验，是不是真的改变了 Agent 在后续任务上的行为。**
所有数字都来自实际运行的输出，对应的 run 目录写在每节末尾，可逐条复核。

---

## 1. 实验结构

闭环被拆成三次运行，各自回答一个不同的问题：

| 实验 | 回答什么 | 对应目录 |
|---|---|---|
| probe M07 | 只改注入文本，行为会不会变（因果） | 终端输出，无 run 目录 |
| m10-search | 系统自己提炼的经验，能不能迁移到另一道题 | `runs/m10-search/20261117-233252` |
| full-cold | 放到全部 43 条任务上，整体是什么结果 | `runs/full-cold/20261118-000730` |

分工的理由：`run_loop` 一次只跑一遍，单题翻转分不清是经验还是模型抖动；`probe` 把变量压到只剩注入文本、每组重复 3 次，才能谈因果。前者证明"能迁移"，后者证明"是因果"。

---

## 2. 定点对照（probe M07）

同一道题、同一份提示词、同一模型（`temperature=0`），唯一变化是注入的 97 字文本：

```
调用工具时 max_price 必须带上具体值：本轮用户说了就按本轮的，本轮没提但前面说过的，
沿用最近一次的值，不能退回更早的旧值；换关键词继续搜时，之前的限制仍然有效；
不能留空，也不能凭空猜。
```

这段文本不是手写的，是 `pipeline.learn` 从 M10 的失败里自动提炼、经过四道闸门入库的 `exp-c8e70cd9d11b` 的 `action` 原文。

| 组 | 通过 | 工具序列 |
|---|---|---|
| 无注入 | **0/3** | 三次都判 `wrong_arg@t3` |
| 注入 97 字 | **3/3** | 三次都是 `search_products ×3` |

回答里的差异是直接的：

- 无注入 A1/A2：「耳机品类目前只有这一款」「全库只有 1 款匹配商品」——没有提任何价格条件。
- 注入 B1/B2/B3：「沿用 900 以内的预算上限」「沿用你最新的预算上限 900」「沿用你刚才的 900 预算上限」。

M07 第 3 轮用户只说「再帮我搜个耳机」，预算 900 是第 2 轮给的。无注入时模型把它丢了，注入后三次都带上了。

**累计口径**：不含本次，更早的 probe 轮次（11-17 22:5x）同样是 0/3 对 3/3。M07 无注入累计 14 次 FAIL、1 次 PASS（那 1 次在 `runs/20261117-230554`），自动经验注入累计 5 次全 PASS、无 FAIL。

---

## 3. 多轮集闭环（m10-search）

只跑多轮集：fit 6 条（M01/M02/M04/M06/M08/M10），holdout 4 条（M03/M05/M07/M09）。

学习阶段 M10 判 `wrong_arg@t3`，提炼出入库 1 条 `wrong_arg` 经验；验证阶段：

| 任务 | 基线 | 注入 | 注入字数 |
|---|---|---|---|
| M03 | PASS | PASS | 0 |
| M05 | FAIL hallucination@t1 | FAIL hallucination@t1 | 0 |
| **M07** | **FAIL wrong_arg@t3** | **PASS** | 268 |
| M09 | PASS | PASS | 333 |

基线 2/4 → 注入 3/4，唯一翻转是 M07，且它注入了 268 字。

---

## 4. 全量闭环（full-cold）

### 4.1 学习阶段

fit 26 条，通过 22 条，四种失败各一条，提炼入库 4 条：

| 经验 | 失败类型 | 来源题 |
|---|---|---|
| `exp-070cde8d3e25` | wrong_fact | S15 |
| `exp-011b486a50fb` | missing_tool | M08 |
| `exp-c8e70cd9d11b` | wrong_arg | M10 |
| `exp-56d434f43f4a` | forbidden_action | H11 |

### 4.2 验证阶段

| | 通过 | 失败分布 |
|---|---|---|
| 基线 | 14/17 | hallucination 1 / missing_tool 1 / wrong_arg 1 |
| 注入 | 16/17 | wrong_fact 1 |
| 差值 | **+2** | PASS→FAIL 0 条 |

17 条 holdout 里只有 4 条真的注入了经验（M07 268 字、M09 333 字、S12 204 字、S14 213 字），其余 13 条注入 0 字，条件与基线完全相同。

### 4.3 逐条归因（关键）

**+2 里只有 1 条能算经验的。**

| 任务 | 基线 | 注入 | 注入字数 | 归因 |
|---|---|---|---|---|
| M07 | FAIL wrong_arg@t3 | PASS | 268 | **可归因**，与 probe 3+3 独立吻合 |
| M05 | FAIL hallucination@t1 | PASS | **0** | **不可归因**，纯模型抖动 |
| S14 | FAIL missing_tool@t1 | FAIL wrong_fact@t1 | 213 | 经验生效，但未翻转 |
| S12 | PASS | PASS | 204 | 注入无害 |

**M05 必须排除。** 它注入 0 字，条件与基线一模一样，却从 FAIL 变 PASS：基线跑了 14.7 秒、两次 `search_products` 后编出商品（hallucination），注入组 2.8 秒、一次工具都没调、直接反问预算就通过了。同一条件下自己翻的面，不能记在经验头上。

**S14 是"经验生效但不够"的证据。** 基线判 `missing_tool@t1`（只调了两次 `apply_refund`，没查订单）；注入 `exp-011b486a50fb`（必须调用 get_order）后，工具序列变成 `get_order + apply_refund ×2` —— 经验要它补的动作它确实补了，但判定从 `missing_tool` 变成 `wrong_fact`，仍然没过。这说明注入通道和语义都没问题，是 S14 的失败原因不止一个。

---

## 5. 已知局限（答辩时必须主动说）

1. **fit 侧样本是我补的。** M10 是为覆盖 `wrong_arg` 失败模式新建的题。原有 fit 25 条没有这类失败，自动提炼根本产不出这条经验。这不是"系统自己发现了问题"，是"训练侧覆盖不足，补了一道同型题"。经验本身是系统自动提炼、自动检索、自动注入的。
2. **可归因提升只有 1 条。** 全量净 +2，扣掉抖动的 M05 后是 +1/17。样本太小，不给百分比结论。
3. **模型不确定。** 同一道题、`temperature=0`，M07 基线出现过 FAIL 14 次 / PASS 1 次。任何单次翻转都不足以定性，必须重复对照。
4. **覆盖率低是合理的，不是缺陷。** 17 条 holdout 里 14 条本来就通过，不需要经验；剩下 3 条失败的，2 条有对症经验（M07 救回、S14 部分生效），M05 是 hallucination，而 fit 侧没有任何 hallucination 样本，库中没有对症经验可注入。提高覆盖率不能解决 M05。
5. **只验证了一种失败模式的可迁移性。** `wrong_arg` 有 3+3 重复对照；其余三条经验各只在一次运行里被注入过 1–2 次。

---

## 6. 复现方式

```
# 窗口 A
.venv\Scripts\python.exe mock_server.py

# 窗口 B：定点对照（约 5 分钟）
.venv\Scripts\python.exe scripts\probe.py --task M07 --repeat 3 --experience "<experiences.jsonl 里的 action 原文>"

# 窗口 B：多轮集闭环（约 5 分钟）
.venv\Scripts\python.exe scripts\run_loop.py --agent cold --golden data\shop_eval_multi.jsonl --out runs\m10-search

# 窗口 B：全量闭环（约 15 分钟）
.venv\Scripts\python.exe scripts\run_loop.py --agent cold --out runs\full-cold
```

每次运行都会落一个自包含目录：`fit.jsonl`（学习轨迹）、`holdout-baseline.jsonl`、`holdout-injected.jsonl`、`experiences.jsonl`（本次用的库）、`report.json`（全部账目）。
