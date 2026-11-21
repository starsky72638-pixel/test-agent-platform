# reports 索引

读之前先看这一句：**这里的报告跨越 11-15 到 11-21 六个阶段，数字不代表同一时点。**
每份开头都标了日期或阶段；引用数字前，先确认它属于哪个阶段、分母是什么。

## 按用途查

| 想知道什么 | 看哪份 |
|---|---|
| 平台现在有什么能力、产出了什么数据 | `platform_capability_report.md` |
| 经验注入到 Workflow 到底有没有用 | `p7_experiment.md` |
| 早期主表的通过率与逐条翻转 | `final_20261119.md`、`final_20261120.md` |
| P5-c 盘点时的机器可读统计 | `p5c-experiences.json`、`p5c-sft.json`、`p5c-wf-distill.json`、`p5c-memory.json` |
| 单次评测的原始输出 | `shop_eval*.json` |

## 三份主线文档

**`platform_capability_report.md`** — 唯一的总账。第 1–10 节写到 P6，
第 11 节补齐 P7–P10（注入实验、跨轮上下文与槽位、来源信息、经验归并、SFT 四分类）。
能力按"有运行证据 / 只有单测 / 未实现"三档标注，**不要跨节搬运数字**：
第 4 节的 1/3 是 P5-c 快照，第 10 节才是 P6 之后的 2/3。

**`p7_experiment.md`** — 一次完整的注入实验。核心结论是
「注入通道已实测打通（4/4 覆盖、每题 238 字），但基线已满分，差值只能是 0」。
它是本项目最诚实的一份记录：证明的是机制，不是效果。

**`final_20261119.md` / `final_20261120.md`** — 早期主表与闭环结论。
主表基线 16/17、注入 15/17、有效覆盖 2/17，三条翻转全在注入 0 字的题上。
**这两份的数字已经被后续阶段部分覆盖**（参数抽取、跨轮槽位、经验归并都改变了输入条件），
引用时必须带日期。

## 机器可读产物

`p5c-*.json` 是 P5-c 盘点脚本（`scripts/p5c_probe.py`）直接吐出的统计，不是手写的：

| 文件 | 内容 |
|---|---|
| `p5c-experiences.json` | 8 个经验库：行数、唯一 exp_id、failure_type 分布、support/contradict |
| `p5c-sft.json` | 各 SFT 文件的样本数、来源 agent、split 分布、空样本检查 |
| `p5c-wf-distill.json` | Workflow 轨迹进 distill 的逐条结构检查结果 |
| `p5c-memory.json` | 8 库 × 5 任务集的检索命中矩阵 + 注入验证 |

这些是**快照**。要拿当前数字，重跑对应脚本，不要读旧 json。

## 已过期的部分（明确列出，避免误用）

| 内容 | 现状 |
|---|---|
| `p5c-*.json` 里的经验库统计 | 已归并为 `data/memory/experiences.jsonl` 的 9 条 |
| `p5c-wf-distill.json` 的 2 SFT / 2 失败 | W03 现在能过，重跑会变；该目录保留为历史证据，不再更新 |
| `final_*.md` 的 43 题 / 25 SFT | 任务集已扩到 7 个共 65 条；SFT 已重构为 train 26 / holdout 4 |
| `shop_eval*.json` | 早期单次输出，未随评测集变化更新 |

## 数字的来源优先级

同一件事有三处数字时，按这个顺序取：

1. **重新跑一遍**命令（唯一权威）
2. `data/sft/report.json`、`runs/*/report.json`、`runs/*/manifest.json`（机器生成、带哈希与时间）
3. 报告正文（人写，会滞后）

报告里任何数字都能追到第 2 类文件；追不到的就别引用。

**但这三类文件不在仓库里**：`runs/**/*.jsonl` 被 `.gitignore` 挡住，`runs/*/report.json`
与 `manifest.json` 也从未入库（它们和轨迹同批生成）。所以 clone 之后，本目录各报告引用的
轨迹路径**需要重跑对应命令才会出现**，不是文件损坏。哪些产物在仓库、哪些要自己生成，
见 `docs/runbook.md` 第 0 节的对照表。
