"""把评测报告变成「下一轮的输入」：闭环里"沉淀"这一段。

    跑轨迹 → 五层判分 → 【本脚本】→ 三份产物 → 补规则 / 补用例 → 重跑 → fixed 变长
                          ↑______________________________________________|
                              这一圈不需要权重，只需要提示词和用例

## 先说清楚这不是什么

这个项目调的是 DashScope 上的 qwen3.8-flash，走 OpenAI 兼容接口，**没有权重**，
所以不存在"微调"这一环。把通过的轨迹叫 SFT 数据、把失败/通过配对叫 DPO 数据，
会让人以为仓库里藏着一条训练管道，其实没有——本机既没有 torch/peft，也没有
qwen3.8-flash 的参数，拿 1.5B 的开源模型去训，训出来的东西接不回这个 Agent。

产物只有三种真实用途：

    passed 轨迹        → golden_candidates.jsonl   下一批评测用例的候选（真跑通过的）
    失败→通过 配对     → rule_hints.jsonl          某条规则补上之后，行为从什么变成什么
    只有失败、没正解   → failed_cases.jsonl        待定位的缺口，还没找到正确行为

## 输入的字段名为什么叫 train_messages

`replay.py` 落盘时把整条轨迹消息存在 `train_messages` 里，那是历史命名。它实际
是"这道题跑的时候完整发了哪些消息"，本脚本按只读快照用。**不改这个字段名**：
一改，`reports/` 下所有历史报告（含 `shop_eval_before.json` 那份 19/20 的基线）
就都对不上，逐题 diff 直接失效。

## 三条纪律（写死在代码里，不靠人记）

1. 配对的两边 prompt 必须逐字一致。旧报告的 system 是 5 条规则，新报告是 11 条，
   混着配对等于在说"在旧规则下这样答是错的"，线索会被污染。不一致就跳过并记原因。
2. 多轮用例不生成配对。配对要的是"同一个输入、两种回答"，多轮拼进去会把中间
   几轮的回答也算成优化目标，语义就错了。多轮只进 golden 候选。
3. 只认 passed=True 的轨迹当"正确答案"。失败轨迹哪怕看起来不错，也不能当正解。

用法：
    # 只出 golden 候选（三套一起，42 条）
    python tests/feedback.py --current reports/shop_eval.json reports/shop_eval_multi.json reports/shop_eval_hard.json

    # 走完整流程（需要一份失败跑 + 一份通过跑，且同 prompt 版本）
    python tests/feedback.py --baseline reports/shop_eval_before.json \
                             --current reports/shop_eval.json

    # 只想看逐题差异，不写文件
    python tests/feedback.py --baseline reports/old.json --current reports/new.json --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 允许直接运行：python tests/feedback.py
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PROJECT_ROOT

DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "generated"


# =========================
# 一、读报告
# =========================

def load_report(path: Path) -> dict:
    """读一份 replay 产出的 JSON 报告。"""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def index_cases(report: dict) -> dict[str, dict]:
    """把 results 列表变成 {case_id: case}，方便按题号对齐两份报告。"""
    return {c["id"]: c for c in report.get("results") or []}


def system_prompt_of(case: dict) -> str:
    """取这道题跑的时候用的 system 提示词。

    它必须来自轨迹快照的第一条，而不是从当前代码里读——报告是历史快照，
    要和当时的轨迹对上，不能拿今天的提示词去解释昨天的回答。
    """
    msgs = case.get("train_messages") or []
    if msgs and msgs[0].get("role") == "system":
        return msgs[0].get("content") or ""
    return ""


def is_multi_turn(case: dict) -> bool:
    """多轮用例：turns 或 rounds 超过一条。"""
    turns = case.get("turns") or []
    rounds = case.get("rounds") or []
    return len(turns) > 1 or len(rounds) > 1


# =========================
# 二、逐题对比
# =========================

def diff_reports(base: dict, cur: dict) -> dict:
    """对齐两份报告，按四种情况分类。

    fixed      旧失败 → 新通过   （这是"能力变好"的证据，也是规则线索的来源）
    regressed  旧通过 → 新失败   （回归，必须人工看）
    still_fail 两次都失败
    still_pass 两次都通过
    """
    base_cases = index_cases(base)
    cur_cases = index_cases(cur)

    out: dict[str, list[str]] = {
        "fixed": [],
        "regressed": [],
        "still_fail": [],
        "still_pass": [],
        "missing": [],
    }

    for cid in sorted(set(base_cases) | set(cur_cases)):
        b = base_cases.get(cid)
        c = cur_cases.get(cid)

        # 只在一份报告里出现，说明两次跑的题集不一样，不能比
        if b is None or c is None:
            out["missing"].append(cid)
            continue

        b_ok = bool(b.get("passed"))
        c_ok = bool(c.get("passed"))

        if not b_ok and c_ok:
            out["fixed"].append(cid)
        elif b_ok and not c_ok:
            out["regressed"].append(cid)
        elif not b_ok and not c_ok:
            out["still_fail"].append(cid)
        else:
            out["still_pass"].append(cid)

    return out


def render_diff_md(base_path: Path, cur_path: Path, diff: dict) -> str:
    """把对比结果写成一段 Markdown，贴进 README 或迭代记录。"""
    lines = [
        "# 两次评测的逐题对比",
        "",
        f"- 基线：`{base_path.name}`",
        f"- 本次：`{cur_path.name}`",
        "",
        "| 分类 | 题号 | 含义 |",
        "|---|---|---|",
    ]
    meaning = {
        "fixed": "旧失败 → 新通过（能力变好的证据）",
        "regressed": "旧通过 → 新失败（回归，必须查）",
        "still_fail": "两次都失败",
        "still_pass": "两次都通过",
        "missing": "只在一份报告里出现，不可比",
    }
    for key in ("fixed", "regressed", "still_fail", "still_pass", "missing"):
        ids = diff.get(key) or []
        if ids:
            lines.append(f"| `{key}` | {', '.join(ids)} | {meaning[key]} |")

    if diff.get("regressed"):
        lines.append("")
        lines.append("> 有回归：先查回归的题，不要急着拿 fixed 去证明进步。")

    return "\n".join(lines) + "\n"


# =========================
# 三、造产物
# =========================

def make_golden(case: dict, source: str) -> dict | None:
    """通过的轨迹 → 下一批 golden 的候选。

    只认 passed，失败轨迹一律不收。meta 里记 prompt_rules（当时的规则条数），
    以后拿它判断这批候选是不是又跟着提示词一起过期了。
    """
    if not case.get("passed"):
        return None

    msgs = case.get("train_messages") or []
    if not msgs:
        return None

    meta = {
        "case_id": case.get("id"),
        "input": case.get("input"),
        "note": case.get("note", ""),
        "passed": True,
        "source": source,
        "prompt_rules": count_rules(system_prompt_of(case)),
        "trace_id": case.get("trace_id", ""),
    }
    if is_multi_turn(case):
        meta["multi_turn"] = True
        meta["turns"] = case.get("turns") or []

    return {"messages": msgs, "meta": meta}


def split_prompt(messages: list[dict]) -> list[dict]:
    """取到"第一个 assistant 之前"的消息，作为配对的上下文。

    单轮用例就是 [system, user]，正好是"同一个输入、两种回答"需要的形状。
    """
    prompt: list[dict] = []
    for m in messages:
        if m.get("role") == "assistant":
            break
        prompt.append(m)
    return prompt


def make_hint(base_case: dict, cur_case: dict) -> tuple[dict | None, str]:
    """失败轨迹 + 通过轨迹 → 一条规则线索。

    返回 (样本, 跳过原因)。样本为 None 时，第二个值是为什么要跳。
    三道闸门都在这里：多轮、prompt 不一致、缺回答。

    产物里的 wrong_answer / correct_answer 是拿来讲"补了哪条规则、行为怎么变"的，
    它对应的是提示词里的一句话，不是一组梯度。
    """
    if is_multi_turn(base_case) or is_multi_turn(cur_case):
        return None, "多轮用例不生成配对"

    if bool(base_case.get("passed")):
        return None, "基线这一题本来就是通过，不构成对照"

    if not bool(cur_case.get("passed")):
        return None, "本次仍然失败，没有正确行为可用"

    base_prompt = system_prompt_of(base_case)
    cur_prompt = system_prompt_of(cur_case)
    if base_prompt != cur_prompt:
        return None, "两次的 system 提示词不同，配对会污染线索"

    wrong = (base_case.get("answer") or "").strip()
    correct = (cur_case.get("answer") or "").strip()
    if not wrong or not correct:
        return None, "缺 wrong_answer 或 correct_answer，无法配对"

    prompt = split_prompt(cur_case.get("train_messages") or [])
    if not prompt:
        return None, "取不到上下文"

    return (
        {
            "id": cur_case.get("id"),
            "input": cur_case.get("input"),
            "note": cur_case.get("note", ""),
            "prompt": prompt,
            "wrong_answer": wrong,
            "correct_answer": correct,
            "failure_type": base_case.get("failure_type", ""),
            "reason": explain_failure(base_case),
            "wrong_trace": base_case.get("trace_id", ""),
            "correct_trace": cur_case.get("trace_id", ""),
        },
        "",
    )


def make_failed(case: dict, why: str) -> dict:
    """只有失败轨迹的样本：把能拿到的都存下来，等定位出正确行为再补。"""
    return {
        "id": case.get("id"),
        "input": case.get("input"),
        "note": case.get("note", ""),
        "prompt": split_prompt(case.get("train_messages") or []),
        "wrong_answer": (case.get("answer") or "").strip(),
        "failure_type": case.get("failure_type", ""),
        "reason": explain_failure(case),
        "skip_reason": why,
        "trace_id": case.get("trace_id", ""),
    }


# =========================
# 四、小工具
# =========================

def count_rules(prompt: str) -> int:
    """数提示词里有几条规则，用来标记这批产物的版本。

    只数行首形如 "11. " 的行，避免把正文里的数字算进来。
    """
    n = 0
    for line in prompt.splitlines():
        head = line.strip().split(".")[0]
        if head.isdigit() and line.strip().startswith(head + "."):
            n += 1
    return n


def explain_failure(case: dict) -> str:
    """把失败原因压成一句话：类型 + 具体哪一层没过。"""
    parts: list[str] = []
    for key in ("tool_msg", "param_msg", "order_msg"):
        val = (case.get(key) or "").strip()
        # "命中" 是正常结果；"未校验" 是没配这条断言。都不是问题，别写进原因里
        if val and val not in ("命中", "未校验"):
            parts.append(val)

    missing = case.get("missing_any") or []
    if missing:
        parts.append("回答缺关键词 " + "/".join(str(x) for x in missing))

    ftype = case.get("failure_type") or ""
    if not parts:
        return ftype or "未记录原因"
    return f"{ftype}：" + "；".join(parts) if ftype else "；".join(parts)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """写 JSONL，中文不转义，方便直接读。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# =========================
# 五、主流程
# =========================

def main() -> int:
    ap = argparse.ArgumentParser(description="把评测报告转成 golden 候选与规则线索")
    ap.add_argument("--current", nargs="+", required=True, help="本次（通过）的报告，可给多份")
    ap.add_argument("--baseline", nargs="*", default=[], help="基线（失败）的报告，可给多份")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="输出目录")
    ap.add_argument("--dry-run", action="store_true", help="只打印，不写文件")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)

    # ---------- golden 候选：从所有 current 报告里收集通过的轨迹 ----------
    golden: list[dict] = []
    seen: set[str] = set()

    for raw in args.current:
        path = Path(raw)
        report = load_report(path)
        for case in report.get("results") or []:
            sample = make_golden(case, source=path.name)
            if not sample:
                continue
            cid = sample["meta"]["case_id"]
            # 同一题在多份报告里出现时只留第一条，避免候选里重复
            if cid in seen:
                continue
            seen.add(cid)
            golden.append(sample)

    # ---------- 规则线索：只在给了 baseline 时才做 ----------
    hints: list[dict] = []
    failed: list[dict] = []
    diffs: list[tuple[Path, Path, dict]] = []

    for raw_base in args.baseline:
        base_path = Path(raw_base)
        base = load_report(base_path)

        for raw_cur in args.current:
            cur_path = Path(raw_cur)
            cur = load_report(cur_path)

            diff = diff_reports(base, cur)
            diffs.append((base_path, cur_path, diff))

            base_cases = index_cases(base)
            cur_cases = index_cases(cur)

            for cid in diff["fixed"]:
                hint, why = make_hint(base_cases[cid], cur_cases[cid])
                if hint:
                    hints.append(hint)
                else:
                    failed.append(make_failed(base_cases[cid], why))

            for cid in diff["still_fail"]:
                failed.append(make_failed(base_cases[cid], "两次都失败，还没找到正确行为"))

    # ---------- 打印 ----------
    print(f"本次报告 {len(args.current)} 份，基线报告 {len(args.baseline)} 份")
    print(f"  golden 候选   {len(golden)} 条（只收 passed，可直接扩评测集）")
    print(f"  规则线索      {len(hints)} 条（失败→通过，同 prompt 版本）")
    print(f"  待定位缺口    {len(failed)} 条（进 failed_cases，等补正确行为）")

    if diffs:
        print("")
        for base_path, cur_path, diff in diffs:
            print(f"逐题对比 {base_path.name} → {cur_path.name}")
            for key, label in (
                ("fixed", "修复"),
                ("regressed", "回归"),
                ("still_fail", "仍失败"),
                ("still_pass", "仍通过"),
                ("missing", "不可比"),
            ):
                ids = diff.get(key) or []
                if ids:
                    print(f"  {label:<4} {len(ids):>3} 条  {', '.join(ids)}")
            print("")

    if not hints and args.baseline:
        print("提示：线索为 0 时，按顺序检查三件事——")
        print("  1. 基线报告里有没有失败题（全绿就没得配）")
        print("  2. 两次的 system 提示词是否逐字相同（版本变了就不许配）")
        print("  3. 失败题是不是多轮（多轮只进 golden 候选）")
        print("")

    # ---------- 写文件 ----------
    if args.dry_run:
        print("--dry-run：没有写文件")
        return 0

    write_jsonl(out_dir / "golden_candidates.jsonl", golden)
    write_jsonl(out_dir / "rule_hints.jsonl", hints)
    write_jsonl(out_dir / "failed_cases.jsonl", failed)

    for base_path, cur_path, diff in diffs:
        md = render_diff_md(base_path, cur_path, diff)
        target = out_dir / f"diff_{base_path.stem}_vs_{cur_path.stem}.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(md, encoding="utf-8")

    print(f"已写入 {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
