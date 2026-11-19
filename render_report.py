"""把 reports/*.json 渲染成一页可以截图进简历 / PPT 的 HTML 报告。

为什么用脚本生成而不是手写 HTML：
    42 条用例、每轮的工具调用与返回加起来几十 KB，手写必然和报告脱节。
    这里只做一件事——把已经落盘的判定结果原样搬到页面上，不重新计算、不补充结论。
    页面上的每个数字都来自某份 JSON 的 summary 或 rounds，改判分器后重跑汇报即可。

用法：
    python render_report.py                    # 读默认三份，写 reports/report.html
    python render_report.py --out reports/x.html

设计取舍：
    1. 数据内联进 HTML，不用 fetch —— file:// 下 fetch 会被浏览器拦，双击就该能看。
    2. 只内联必要字段，不搬 train_messages（system 提示词 11 条规则，42 份会撑到几 MB）。
    3. 纯原生 JS，无依赖、无 CDN，断网也能打开。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"

# 三套的顺序即页面顺序：基础 → 多轮 → 难集
SUITES = [
    {
        "key": "base",
        "file": "shop_eval.json",
        "name": "单轮基础集",
        "desc": "单轮问答，考工具选择与参数。20 条。",
    },
    {
        "key": "multi",
        "file": "shop_eval_multi.json",
        "name": "多轮记忆集",
        "desc": "多轮指代延续、改口径、跨轮组合。9 条。",
    },
    {
        "key": "hard",
        "file": "shop_eval_hard.json",
        "name": "边界难集",
        "desc": "404 / 422 / 409 / 待支付 / 超范围，考「不许编」。13 条。",
    },
]


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _calls_of(r: dict) -> list[dict]:
    """取出这一轮的每次调用，并让每次调用带上自己那次的返回。

    不能直接用 outputs[name]：同一轮里同一个工具被调两次时，outputs 按工具名存，
    两次会取到同一个值（M09 那类用例就会这样）。优先按 tool_results 的顺序对齐。
    """
    trs = r.get("tool_results") or []
    if trs:
        return [
            {
                "name": t.get("name"),
                "args": t.get("args") or {},
                "output": t.get("output") or "",
            }
            for t in trs
        ]
    outputs = r.get("outputs") or {}
    return [
        {
            "name": c.get("name"),
            "args": c.get("args") or {},
            "output": outputs.get(c.get("name"), ""),
        }
        for c in (r.get("calls") or [])
    ]


def slim_round(r: dict) -> dict:
    """保留判定链路需要的字段，丢掉消息体等大字段。"""
    return {
        "turn": r.get("turn"),
        "input": r.get("input") or "",
        "answer": r.get("answer") or "",
        "calls": _calls_of(r),
        "steps": r.get("steps"),
        # ---- 五层判定 ----
        "tool_ok": bool(r.get("tool_ok")),
        "tool_msg": r.get("tool_msg") or "",
        "param_ok": bool(r.get("param_ok")),
        "param_msg": r.get("param_msg") or "",
        "order_ok": bool(r.get("order_ok")),
        "order_msg": r.get("order_msg") or "",
        "ground_ok": bool(r.get("ground_ok")),
        "ungrounded": r.get("ungrounded") or [],
        "answer_ok": bool(r.get("answer_ok")),
        "missing_fact": r.get("missing_fact") or [],
        "missing_any": r.get("missing_any") or [],
        "forbidden_hit": r.get("forbidden_hit") or [],
        # ---- 附加信号 ----
        "non_answer": bool(r.get("non_answer")),
        "over_calls": r.get("over_calls") or [],
        "failure_type": r.get("failure_type") or "",
    }


def slim_case(c: dict) -> dict:
    return {
        "id": c.get("id") or "",
        "input": c.get("input") or "",
        "turns": c.get("turns") or [],
        "note": c.get("note") or "",
        "elapsed_ms": c.get("elapsed_ms") or 0,
        "passed": bool(c.get("passed")),
        "failure_type": c.get("failure_type") or "",
        "has_claims": bool(c.get("has_claims")),
        "rounds": [slim_round(r) for r in (c.get("rounds") or [])],
    }


def coverage(cases: list[dict]) -> dict:
    """算「真实分母」。

    报告汇总行里的 "顺序 N/N" 是把没配断言的用例也算成通过的，
    这里把真正检查过的条数单独数出来，页面上必须显示这个更小的分母。
    """
    order = 0
    ground = 0
    for c in cases:
        if any((r.get("order_msg") or "") != "未校验" for r in c["rounds"]):
            order += 1
        if c.get("has_claims"):
            ground += 1
    return {"order": order, "ground": ground, "total": len(cases)}


def build_payload() -> dict:
    suites: list[dict] = []

    for spec in SUITES:
        report = load(REPORTS / spec["file"])
        if not report:
            continue

        cases = [slim_case(c) for c in (report.get("results") or [])]
        summary = report.get("summary") or {}

        suites.append(
            {
                "key": spec["key"],
                "name": spec["name"],
                "desc": spec["desc"],
                "file": spec["file"],
                "model": report.get("model") or "",
                "summary": {
                    "total": summary.get("total", len(cases)),
                    "valid": summary.get("valid", len(cases)),
                    "passed": summary.get("passed", 0),
                    "avg_ms": summary.get("avg_ms", 0),
                    "avg_calls": summary.get("avg_calls", 0),
                    "infra_error": summary.get("infra_error", 0),
                },
                "coverage": coverage(cases),
                "cases": cases,
            }
        )

    total_cases = sum(len(s["cases"]) for s in suites)
    total_passed = sum(s["summary"]["passed"] for s in suites)
    total_valid = sum(s["summary"]["valid"] for s in suites)

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "totals": {
            "cases": total_cases,
            "passed": total_passed,
            "valid": total_valid,
            "suites": len(suites),
        },
        "suites": suites,
    }


TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agent 评测考场 · 报告</title>
<style>
  :root{
    --bg:#f6f7f9; --card:#ffffff; --ink:#1c2024; --mute:#646b76;
    --line:#e3e5e9; --blue:#2563eb; --blue-soft:#eef3ff;
    --ok:#15803d; --ok-soft:#eaf7ee;
    --bad:#c0271f; --bad-soft:#fdeceb;
    --idle:#8b929c; --idle-soft:#f0f1f3;
    --mono:ui-monospace,"Cascadia Code",Consolas,"SF Mono",Menlo,monospace;
  }
  *{box-sizing:border-box}
  body{
    margin:0; background:var(--bg); color:var(--ink);
    font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif;
    -webkit-font-smoothing:antialiased;
  }
  .wrap{max-width:1060px;margin:0 auto;padding:52px 24px 80px}

  /* ---------- hero ---------- */
  .eyebrow{font-size:12px;letter-spacing:.14em;color:var(--mute);text-transform:uppercase;margin-bottom:14px}
  h1{font-size:34px;line-height:1.25;margin:0 0 14px;font-weight:650;letter-spacing:-.01em}
  .lede{color:var(--mute);font-size:16px;margin:0 0 26px;max-width:760px}
  .lede b{color:var(--ink);font-weight:600}
  .hero{display:flex;gap:34px;flex-wrap:wrap;padding:20px 0 4px;border-top:1px solid var(--line)}
  .hero-item{display:flex;align-items:baseline;gap:8px}
  .hero-num{font-size:26px;font-weight:650;font-variant-numeric:tabular-nums}
  .hero-lab{font-size:13px;color:var(--mute)}

  /* ---------- 章节 ---------- */
  section{margin-top:46px}
  h2{font-size:19px;margin:0 0 6px;font-weight:620;letter-spacing:-.005em}
  .sub{color:var(--mute);font-size:13.5px;margin:0 0 18px}

  /* ---------- 成绩卡 ---------- */
  .suites{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}
  @media(max-width:780px){.suites{grid-template-columns:1fr}}
  .suite{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px 18px 16px}
  .suite-top{display:flex;justify-content:space-between;align-items:flex-start;gap:8px}
  .suite-name{font-weight:600;font-size:14.5px}
  .suite-file{font:11.5px var(--mono);color:var(--idle);margin-top:2px}
  .suite-score{font-size:30px;font-weight:660;font-variant-numeric:tabular-nums;margin:12px 0 2px}
  .suite-score span{font-size:15px;color:var(--idle);font-weight:500}
  .suite-meta{font-size:12.5px;color:var(--mute);font-variant-numeric:tabular-nums}
  .pill{display:inline-block;font-size:11.5px;padding:2px 8px;border-radius:999px;font-weight:600}
  .pill.ok{background:var(--ok-soft);color:var(--ok)}
  .pill.bad{background:var(--bad-soft);color:var(--bad)}

  /* ---------- 覆盖表 ---------- */
  table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden}
  th,td{text-align:left;padding:11px 14px;font-size:13.5px;border-bottom:1px solid var(--line);vertical-align:top}
  th{background:#fafbfc;font-weight:600;font-size:12.5px;color:var(--mute);letter-spacing:.02em}
  tr:last-child td{border-bottom:none}
  td.num{font-variant-numeric:tabular-nums;white-space:nowrap}
  .note-cell{color:var(--mute);font-size:12.5px}

  /* ---------- 时间线 ---------- */
  .tl{position:relative;padding-left:22px}
  .tl:before{content:"";position:absolute;left:5px;top:8px;bottom:8px;width:2px;background:var(--line)}
  .tl-item{position:relative;padding:0 0 18px 0}
  .tl-item:last-child{padding-bottom:0}
  .tl-item:before{content:"";position:absolute;left:-21px;top:7px;width:9px;height:9px;border-radius:50%;background:var(--card);border:2px solid var(--blue)}
  .tl-title{font-weight:600;font-size:14px}
  .tl-desc{color:var(--mute);font-size:13px;margin-top:2px}
  .tl-desc code{font:12px var(--mono);background:var(--idle-soft);padding:1px 5px;border-radius:4px}

  /* ---------- 逐题 ---------- */
  .filters{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px}
  .chip{border:1px solid var(--line);background:var(--card);color:var(--mute);font-size:12.5px;padding:5px 12px;border-radius:999px;cursor:pointer;font-family:inherit}
  .chip:hover{border-color:#c9cdd4}
  .chip.on{background:var(--blue-soft);border-color:#c7d7fe;color:var(--blue);font-weight:600}

  .case{background:var(--card);border:1px solid var(--line);border-radius:10px;margin-bottom:8px;overflow:hidden}
  .case-head{display:flex;align-items:center;gap:12px;padding:12px 15px;cursor:pointer;user-select:none}
  .case-head:hover{background:#fbfcfd}
  .cid{font:12.5px var(--mono);font-weight:600;color:var(--blue);min-width:42px}
  .cinput{flex:1;font-size:13.5px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .cmeta{font-size:12px;color:var(--idle);font-variant-numeric:tabular-nums;white-space:nowrap}
  .dot{width:8px;height:8px;border-radius:50%;flex:none}
  .dot.p{background:var(--ok)} .dot.f{background:var(--bad)}
  .caret{color:var(--idle);font-size:11px;transition:transform .15s}
  .case.open .caret{transform:rotate(90deg)}
  .case-body{display:none;border-top:1px solid var(--line);padding:16px 15px 18px;background:#fcfcfd}
  .case.open .case-body{display:block}
  .cnote{font-size:12.5px;color:var(--mute);background:var(--idle-soft);border-radius:7px;padding:8px 11px;margin-bottom:14px}
  .cnote b{color:var(--ink);font-weight:600}

  .round{margin-bottom:18px;padding-bottom:16px;border-bottom:1px dashed var(--line)}
  .round:last-child{border-bottom:none;margin-bottom:0;padding-bottom:0}
  .rtag{font-size:11.5px;color:var(--mute);font-weight:600;letter-spacing:.04em;margin-bottom:8px}

  .bubble{background:var(--blue-soft);border-radius:9px;padding:9px 12px;font-size:13.5px;margin-bottom:10px}
  .bubble.user{background:var(--blue-soft)}
  .bubble.answer{background:#fff;border:1px solid var(--line);white-space:pre-wrap;margin-bottom:12px}

  .call{border-left:3px solid #c7d7fe;background:#fff;border-radius:0 8px 8px 0;margin-bottom:9px}
  .call-h{display:flex;gap:8px;align-items:baseline;padding:7px 11px;font-size:12.5px}
  .call-h .tname{font:12px var(--mono);color:var(--blue);font-weight:600}
  .call-h .targs{font:11.5px var(--mono);color:var(--mute);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .call pre{margin:0;padding:8px 11px;border-top:1px solid var(--line);font:11.5px/1.55 var(--mono);color:#3d4450;white-space:pre-wrap;word-break:break-all;max-height:150px;overflow:auto}

  .layers{display:flex;gap:6px;flex-wrap:wrap;margin-top:10px}
  .ly{font-size:11.5px;padding:3px 9px;border-radius:6px;font-weight:600;white-space:nowrap}
  .ly.ok{background:var(--ok-soft);color:var(--ok)}
  .ly.bad{background:var(--bad-soft);color:var(--bad)}
  .ly.idle{background:var(--idle-soft);color:var(--idle)}
  .ly small{font-weight:400;opacity:.85}
  .ftype{margin-top:8px;font-size:12px;color:var(--bad);font-weight:600}
  .ftype code{font:11.5px var(--mono);background:var(--bad-soft);padding:1px 5px;border-radius:4px}

  footer{margin-top:56px;padding-top:18px;border-top:1px solid var(--line);color:var(--idle);font-size:12.5px;line-height:1.9}
  footer code{font:11.5px var(--mono)}
</style>
</head>
<body>
<div class="wrap">

  <div class="eyebrow">Agent Eval Harness</div>
  <h1>一个把「它答得对不对」拆成可判定信号的评测考场</h1>
  <p class="lede">
    LLM Agent 的失败很难复现：同一版提示词、<b>temperature=0</b>，同一条用例跑两次可以一次正常、一次拒答。
    所以「它变好了」这句话，没有评测系统就只是感觉。这里把每条轨迹拆成
    <b>五层断言</b>加<b>带优先级的失败归因</b>，判定结果原样落盘。
  </p>
  <div class="hero" id="hero"></div>

  <section>
    <h2>三套成绩</h2>
    <p class="sub">同一次运行、同一版提示词（11 条规则）、同一版判分器。分母是有效条数，链路故障已剔除。</p>
    <div class="suites" id="suites"></div>
  </section>

  <section>
    <h2>覆盖率：分母比分子重要</h2>
    <p class="sub">汇总行里的「顺序 N/N」把没配断言的用例也算成通过。下面是真正检查过的条数。</p>
    <table id="cov"></table>
  </section>

  <section>
    <h2>迭代记录</h2>
    <p class="sub">每一轮改动都是被一次真实失败推着走的，不是预先设计的架构。</p>
    <div class="tl">
      <div class="tl-item">
        <div class="tl-title">基线 19/20</div>
        <div class="tl-desc">S08 用户问「订单什么状态」，只查了订单没查物流，答了一半 → 归因 <code>missing_tool</code></div>
      </div>
      <div class="tl-item">
        <div class="tl-title">补规则 8：状态类问题订单 + 物流各查一次 → 20/20</div>
        <div class="tl-desc">修好的是「只答一半」，不是「答错」——这两类失败在归因里必须分开</div>
      </div>
      <div class="tl-item">
        <div class="tl-title">S13 查完订单直接答「已提交退款」，<code>apply_refund</code> 一次没调</div>
        <div class="tl-desc">补规则 11：动作类断言（已提交/已申请/已办理）必须有本轮工具成功返回作依据</div>
      </div>
      <div class="tl-item">
        <div class="tl-title">S14 被判失败，实际 Agent 完全正确</div>
        <div class="tl-desc">判分器依赖了别的用例留下的退款状态 —— <b>判分器自己的 bug 会伪装成 Agent 的失败</b>。改成两轮自足用例</div>
      </div>
      <div class="tl-item">
        <div class="tl-title">H13 答「没有这个订单号」被判答错 → 词表补了 4 次仍未穷尽</div>
        <div class="tl-desc">改用 <code>re:</code> 前缀正则。<code>re:</code> 只是把枚举写紧凑，不是语义理解 —— 这条局限写在 README 里</div>
      </div>
      <div class="tl-item">
        <div class="tl-title">M09 单题 9 次调用、139 秒仍 PASS</div>
        <div class="tl-desc">加 <code>max_calls</code>，但<b>只上报不判失败</b>：正确性优先于效率，把「多查两次」和「答错」混进同一结论会盖掉真问题</div>
      </div>
    </div>
  </section>

  <section>
    <h2>逐题明细</h2>
    <p class="sub">点任意一题展开：用户输入 → 工具调用与返回 → 回答 → 五层判定。所有内容来自 <code>reports/*.json</code>，未经改写。</p>
    <div class="filters" id="filters"></div>
    <div id="cases"></div>
  </section>

  <footer id="foot"></footer>
</div>

<script>
const DATA = __DATA__;

function esc(s){
  return String(s==null?"":s)
    .replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")
    .replace(/"/g,"&quot;");
}

/* ---------- hero ---------- */
(function(){
  const t = DATA.totals;
  const items = [
    [t.cases, "条用例"],
    [t.passed + "/" + t.valid, "通过"],
    [5, "层判分"],
    [11, "类归因"]
  ];
  document.getElementById("hero").innerHTML = items.map(function(x){
    return '<div class="hero-item"><span class="hero-num">'+x[0]+'</span><span class="hero-lab">'+x[1]+'</span></div>';
  }).join("");
})();

/* ---------- 成绩卡 ---------- */
(function(){
  const box = document.getElementById("suites");
  box.innerHTML = DATA.suites.map(function(s){
    const m = s.summary;
    const allOk = m.passed === m.valid;
    return '<div class="suite">'
      + '<div class="suite-top"><div><div class="suite-name">'+esc(s.name)+'</div>'
      + '<div class="suite-file">'+esc(s.file)+'</div></div>'
      + '<span class="pill '+(allOk?"ok":"bad")+'">'+(allOk?"全绿":"有失败")+'</span></div>'
      + '<div class="suite-score">'+m.passed+'<span> / '+m.valid+'</span></div>'
      + '<div class="suite-meta">avg '+m.avg_ms+' ms · 平均 '+m.avg_calls+' 次调用 · 链路故障 '+m.infra_error+'</div>'
      + '<div class="suite-meta" style="margin-top:6px;color:#8b929c">'+esc(s.desc)+'</div>'
      + '</div>';
  }).join("");
})();

/* ---------- 覆盖表 ---------- */
(function(){
  const rows = DATA.suites.map(function(s){
    const c = s.coverage;
    const orderNote = c.order === 0 ? "全部未配断言" : "只覆盖 "+c.order+" 条，其余 "+ (c.total-c.order) +" 条默认通过";
    const groundNote = "配了正向事实溯源的仅 "+c.ground+" 条";
    return '<tr>'
      + '<td><b>'+esc(s.name)+'</b><div class="note-cell">'+esc(s.file)+'</div></td>'
      + '<td class="num">'+s.summary.passed+' / '+s.summary.valid+'</td>'
      + '<td class="num">'+c.order+' / '+c.total+'</td>'
      + '<td class="note-cell">'+orderNote+'</td>'
      + '<td class="num">'+c.ground+' / '+c.total+'</td>'
      + '<td class="note-cell">'+groundNote+'</td>'
      + '</tr>';
  }).join("");
  document.getElementById("cov").innerHTML =
    '<thead><tr><th>集</th><th>通过</th><th>顺序断言</th><th>说明</th><th>依据溯源</th><th>说明</th></tr></thead>'
    + '<tbody>'+rows+'</tbody>';
})();

/* ---------- 五层 ---------- */
function layerTags(r){
  const out = [];
  out.push(tag("L1 工具", r.tool_ok, r.tool_msg, false));
  out.push(tag("L2 参数", r.param_ok, r.param_msg, false));

  const ung = (r.ungrounded||[]).join("、");
  out.push(tag("L3 依据", r.ground_ok, ung, false));

  const miss = [].concat(r.missing_fact||[], r.missing_any||[], r.forbidden_hit||[]).join("、");
  out.push(tag("L4 事实", r.answer_ok, miss, false));

  const unverified = (r.order_msg||"") === "未校验";
  out.push(tag("L5 顺序", r.order_ok, unverified ? "" : r.order_msg, unverified));
  return out.join("");
}

function tag(name, ok, note, unverified){
  let cls = ok ? "ok" : "bad";
  if (unverified) cls = "idle";
  let text = name + (ok ? " ✓" : " ✗");
  if (note && note !== "命中") {
    text += ' <small>'+esc(String(note).slice(0,40))+'</small>';
  }
  return '<span class="ly '+cls+'">'+text+'</span>';
}

/* ---------- 逐题 ---------- */
function roundHTML(r, idx){
  const calls = (r.calls||[]).map(function(c){
    const args = JSON.stringify(c.args||{});
    const out = c.output || "";
    return '<div class="call">'
      + '<div class="call-h"><span class="tname">'+esc(c.name)+'</span>'
      + '<span class="targs">'+esc(args)+'</span></div>'
      + (out ? '<pre>'+esc(out)+'</pre>' : '')
      + '</div>';
  }).join("");

  const noCall = (r.calls||[]).length === 0
    ? '<div class="call"><div class="call-h"><span class="tname" style="color:#8b929c">未调用任何工具</span></div></div>'
    : "";

  const ft = r.failure_type
    ? '<div class="ftype">归因 <code>'+esc(r.failure_type)+'</code></div>'
    : "";

  const over = (r.over_calls||[]).length
    ? '<div class="ftype" style="color:#8b929c">调用超限（只上报）：'
      + r.over_calls.map(function(x){return esc(x[0])+" ×"+x[1]+"（上限 "+x[2]+"）";}).join("、")
      + '</div>'
    : "";

  return '<div class="round">'
    + '<div class="rtag">第 '+r.turn+' 轮'+(r.steps!=null?' · '+r.steps+' 步':'')+'</div>'
    + '<div class="bubble user">'+esc(r.input)+'</div>'
    + calls + noCall
    + '<div class="bubble answer">'+esc(r.answer)+'</div>'
    + '<div class="layers">'+layerTags(r)+'</div>'
    + ft + over
    + '</div>';
}

function caseHTML(c, suiteKey){
  const body = (c.rounds||[]).map(function(r,i){return roundHTML(r,i);}).join("");
  const tools = [];
  (c.rounds||[]).forEach(function(r){
    (r.calls||[]).forEach(function(x){ tools.push(x.name); });
  });
  return '<div class="case" data-suite="'+suiteKey+'" data-passed="'+(c.passed?1:0)+'">'
    + '<div class="case-head">'
    +   '<span class="dot '+(c.passed?"p":"f")+'"></span>'
    +   '<span class="cid">'+esc(c.id)+'</span>'
    +   '<span class="cinput">'+esc(c.input)+'</span>'
    +   '<span class="cmeta">'+c.elapsed_ms+' ms · '+(tools.length? esc(tools.join(" → ")) : "无工具")+'</span>'
    +   '<span class="caret">▶</span>'
    + '</div>'
    + '<div class="case-body">'
    +   (c.note ? '<div class="cnote"><b>考点</b>　'+esc(c.note)+'</div>' : '')
    +   body
    + '</div>'
    + '</div>';
}

(function(){
  const host = document.getElementById("cases");
  const filters = document.getElementById("filters");

  let html = "";
  DATA.suites.forEach(function(s){
    html += '<div style="margin:22px 0 10px;font-size:13px;font-weight:600;color:#646b76;letter-spacing:.03em">'
         + esc(s.name) + '　<span style="font-weight:400;color:#8b929c">' + esc(s.file) + '</span></div>';
    html += s.cases.map(function(c){return caseHTML(c, s.key);}).join("");
  });
  host.innerHTML = html;

  const opts = [{k:"all", t:"全部"}]
    .concat(DATA.suites.map(function(s){return {k:s.key, t:s.name};}))
    .concat([{k:"fail", t:"只看失败"}]);
  filters.innerHTML = opts.map(function(o,i){
    return '<button class="chip'+(i===0?" on":"")+'" data-k="'+o.k+'">'+o.t+'</button>';
  }).join("");

  filters.addEventListener("click", function(e){
    const btn = e.target.closest(".chip");
    if(!btn) return;
    filters.querySelectorAll(".chip").forEach(function(b){b.classList.remove("on");});
    btn.classList.add("on");
    const k = btn.dataset.k;
    document.querySelectorAll(".case").forEach(function(el){
      let show = true;
      if (k === "fail") show = el.dataset.passed === "0";
      else if (k !== "all") show = el.dataset.suite === k;
      el.style.display = show ? "" : "none";
    });
  });

  host.addEventListener("click", function(e){
    const head = e.target.closest(".case-head");
    if(!head) return;
    head.parentElement.classList.toggle("open");
  });
})();

/* ---------- footer ---------- */
(function(){
  document.getElementById("foot").innerHTML =
    '生成时间 '+esc(DATA.generated_at)+'　·　模型 '+(DATA.suites[0]?esc(DATA.suites[0].model):"—")
    + '　·　数据来源 <code>reports/*.json</code><br>'
    + '页面只搬运已落盘的判定结果，不重新计算、不补充结论。改判分器后重跑 replay 并重新生成，数字才会变。';
})();
</script>
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="把评测报告渲染成单页 HTML")
    ap.add_argument("--out", default=str(REPORTS / "report.html"))
    args = ap.parse_args()

    payload = build_payload()
    if not payload["suites"]:
        raise SystemExit("reports/ 下没有可读的报告，先跑 tests/replay.py")

    # 内联进 <script> 的三处防护：
    #   </  拼进字符串会提前闭合脚本标签
    #   U+2028 / U+2029 在 JS 字符串字面量里是非法换行
    data_json = (
        json.dumps(payload, ensure_ascii=False)
        .replace("</", "<\\/")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    html = TEMPLATE.replace("__DATA__", data_json)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")

    total = payload["totals"]
    print(f"已写入 {out}")
    print(f"  {total['suites']} 套，{total['cases']} 条用例，通过 {total['passed']}/{total['valid']}")
    print(f"  体积 {out.stat().st_size / 1024:.0f} KB，双击即可打开（数据已内联，无需起服务）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
