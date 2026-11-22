/*
 * 新手导览（零依赖、无 CDN、可离线）
 * ------------------------------------------------------------
 * 机制：一层固定定位的遮罩，靠 box-shadow 的巨大扩散「挖洞」聚光，
 *       洞口是目标元素的位置；旁边浮一个白字气泡。
 * 为什么不用 driver.js：本项目前端是零依赖静态页，答辩现场可能断网，
 *       引 CDN 有白屏风险；自己实现约 200 行，行为可控。
 *
 * 跨页续播：点「下一步」时把目标步号写进 sessionStorage('mt.resume')，
 *       跳转后新页面读这个值，从对应步继续。三个页面共用同一份步骤表。
 *
 * 接法：在三个页面的 </body> 前各加一行
 *       <script src="/ui/tour/tour.js"></script>
 *       页面路径决定用哪组步骤，不需要别的标记。
 */
(function () {
  "use strict";

  if (window.__miniTour) return;
  window.__miniTour = true;

  var TOTAL = 10;
  var RESUME_KEY = "mt.resume";

  var PAGES = {
    console: {
      key: "console",
      url: "/ui/",
      steps: [
        {
          n: 1,
          sel: 'a[href="/ui/builder/"]',
          title: "第 1 步 · 搭一个 Agent",
          desc: "点这里进画布。拖 start → tool → end 三个节点，工具选 get_order，参数填 {{extract.order_id}}，保存后立即可跑。这一页的「下一步」会直接带你过去。",
          side: "bottom",
          next: "/ui/builder/"
        },
        {
          n: 2,
          sel: "#agent",
          up: "label.field",
          title: "第 2 步 · 选一个 Agent",
          desc: "带 wf: 前缀的是你在画布上存下来的图，不带的是内置 Agent。第一次体验建议选 wf:shop-demo（两节点图）或 shop-cold（原生 Agent）。",
          side: "right"
        },
        {
          n: 3,
          sel: "#q",
          up: ".composer",
          title: "第 3 步 · 写一句真实问题",
          desc: "例如「查询订单 1003 的状态和金额」。换行会被当作多轮任务一次提交，所以「1001 到哪了」换行「那帮我退了」就是两轮对话。",
          side: "top"
        },
        {
          n: 4,
          sel: "#send",
          title: "第 4 步 · 执行并看轨迹",
          desc: "点执行，右侧会逐条列出工具名、实参和原始返回，点任意一行可展开返回体。这一步只执行与展示，不判分、不入库。",
          side: "top"
        }
      ]
    },
    builder: {
      key: "builder",
      url: "/ui/builder/",
      steps: [
        {
          n: 5,
          sel: ".palette",
          title: "第 5 步 · 左边选节点",
          desc: "四种节点：start / llm / tool / end。点一下加到画布，也可以直接拖过去。condition 与 memory 后端能跑，但画布暂不提供参数表单，所以不摆出来。",
          side: "right"
        },
        {
          n: 6,
          sel: "#canvas",
          up: ".canvas-wrap",
          title: "第 6 步 · 连线成一张图",
          desc: "从一个节点右侧圆点拖到另一个节点即连线，点线段任意位置可删除。约束：一个 start、至少一个 end，不能有环。选中节点可在右侧改工具与参数。",
          side: "bottom"
        },
        {
          n: 7,
          sel: "#btn-save",
          up: ".actions",
          title: "第 7 步 · 保存（立刻生效）",
          desc: "保存写进 data/workflows/<id>.json，不需要注册、不需要重启，回运行页用 wf:<id> 就能跑。保存前会校验结构，不合法会一次性列出所有问题。",
          side: "bottom",
          next: "/ui/data/"
        }
      ]
    },
    data: {
      key: "data",
      url: "/ui/data/",
      steps: [
        {
          n: 8,
          sel: "#mem-count",
          up: "section",
          title: "第 8 步 · 经验记忆",
          desc: "每条经验 = 某类失败 → 该补什么动作，由失败轨迹提炼、过质量闸门后入库。它不是知识库，没有向量检索；输入框可以预览一句话会命中哪几条。",
          side: "bottom"
        },
        {
          n: 9,
          sel: "#ds-count",
          up: "section",
          title: "第 9 步 · 训练 / 评估数据",
          desc: "由评测轨迹蒸馏、过四道闸门后产出：精确去重、同源重跑去重、冲突隔离、split 分流。每张卡片的「下载 JSONL」直接走 /datasets/<name>。",
          side: "bottom"
        },
        {
          n: 10,
          sel: "#counts",
          up: "section",
          title: "第 10 步 · 质量报告与导出",
          desc: "被排除的每一条样本都能在这里找到原因（隔离 / 冲突 / 源不明），没有静默丢弃。这就是「数据可交付」的证据：数量对得上，来源查得到。",
          side: "bottom"
        }
      ]
    }
  };

  var ORDER = ["console", "builder", "data"];

  function pageKey() {
    var p = location.pathname;
    if (p.indexOf("/builder") >= 0) return "builder";
    if (p.indexOf("/data") >= 0) return "data";
    return "console";
  }

  var PAGE = PAGES[pageKey()];
  var steps = PAGE.steps;
  var idx = 0;
  var spots = null, pop = null, wrap = null, introEl = null, launch = null;

  function css() {
    if (document.getElementById("mt-style")) return;
    var s = document.createElement("style");
    s.id = "mt-style";
    s.textContent = [
      "#mt-wrap{position:fixed;inset:0;z-index:99990;display:none}",
      "#mt-spot{position:fixed;border-radius:8px;border:2px solid #7cc4ff;box-shadow:0 0 0 9999px rgba(0,0,0,.68);pointer-events:none;transition:left .16s,top .16s,width .16s,height .16s}",
      "#mt-pop{position:fixed;width:282px;background:rgba(19,21,26,.98);border:1px solid rgba(255,255,255,.22);border-radius:10px;padding:13px 14px;color:#fff;font:13px/1.65 -apple-system,'Segoe UI','Microsoft YaHei',system-ui,sans-serif;box-shadow:0 10px 34px rgba(0,0,0,.5)}",
      "#mt-pop .mt-k{font-size:10px;letter-spacing:.05em;color:rgba(255,255,255,.45)}",
      "#mt-pop .mt-t{font-size:13px;font-weight:600;color:#fff;margin:5px 0 6px}",
      "#mt-pop .mt-d{font-size:12px;line-height:1.75;color:rgba(255,255,255,.74);margin-bottom:12px}",
      "#mt-pop .mt-row{display:flex;align-items:center;justify-content:space-between;gap:8px}",
      ".mt-b{font:inherit;font-size:12px;padding:4px 11px;border-radius:6px;border:1px solid rgba(255,255,255,.26);background:transparent;color:#fff;cursor:pointer}",
      ".mt-b:hover{border-color:rgba(255,255,255,.6)}",
      ".mt-bp{background:#fff;color:#14161a;border-color:#fff;font-weight:600}",
      ".mt-ghost{border-color:transparent;color:rgba(255,255,255,.5)}",
      "#mt-launch{position:fixed;right:16px;bottom:16px;z-index:99980;font:inherit;font-size:12px;padding:7px 14px;border-radius:999px;background:#2563eb;color:#fff;border:1px solid #3b82f6;cursor:pointer;box-shadow:0 4px 16px rgba(37,99,235,.4)}",
      "#mt-launch:hover{background:#1d4ed8}",
      "#mt-intro{position:fixed;inset:0;z-index:99995;display:none;align-items:center;justify-content:center;background:rgba(0,0,0,.62);padding:20px}",
      "#mt-intro .box{width:100%;max-width:430px;background:#15181e;border:1px solid rgba(255,255,255,.2);border-radius:12px;padding:20px 22px;font:13px/1.7 -apple-system,'Segoe UI','Microsoft YaHei',system-ui,sans-serif;color:#e6e9ee;box-shadow:0 20px 60px rgba(0,0,0,.6)}",
      "#mt-intro h3{margin:0 0 8px;font-size:15px;color:#fff}",
      "#mt-intro p{margin:0 0 6px;color:rgba(230,233,238,.72)}",
      "#mt-intro ol{margin:8px 0 16px;padding-left:18px;color:rgba(230,233,238,.72);font-size:12px;line-height:1.9}",
      "#mt-intro .row{display:flex;flex-wrap:wrap;gap:8px;justify-content:flex-end}"
    ].join("\n");
    document.head.appendChild(s);
  }

  function build() {
    if (wrap) return;
    wrap = document.createElement("div");
    wrap.id = "mt-wrap";
    spots = document.createElement("div");
    spots.id = "mt-spot";
    pop = document.createElement("div");
    pop.id = "mt-pop";
    wrap.appendChild(spots);
    wrap.appendChild(pop);
    document.body.appendChild(wrap);
  }

  function launchBtn() {
    if (launch) return;
    launch = document.createElement("button");
    launch.id = "mt-launch";
    launch.type = "button";
    launch.textContent = "新手导览 · " + TOTAL + " 步";
    launch.addEventListener("click", function () { ask(); });
    document.body.appendChild(launch);
  }

  function ask() {
    if (!introEl) {
      introEl = document.createElement("div");
      introEl.id = "mt-intro";
      document.body.appendChild(introEl);
      // 监听只绑一次：ask() 每次访问都会被调用，重复绑定会让一次点击触发多次动作
      introEl.addEventListener("click", function (ev) {
        var b = ev.target.closest("[data-act]");
        if (!b) return;
        var act = b.getAttribute("data-act");
        close();
        if (act === "close") return;
        start(0);
      });
    }
    var first = steps[0].n;
    introEl.innerHTML = ""
      + '<div class="box">'
      +   '<h3>要不要走一遍完整流程？</h3>'
      +   '<p>10 步，约 2 分钟，从搭一个 Agent 一直到导出训练数据。每一步都可以跳过。</p>'
      +   '<ol>'
      +     '<li>搭 Agent（画布上连三个节点）</li>'
      +     '<li>选 Agent、写问题、执行看轨迹</li>'
      +     '<li>批量评估、看产物、导出</li>'
      +   '</ol>'
      +   '<p style="font-size:12px;color:rgba(230,233,238,.5);margin-bottom:14px">从这里开始是第 ' + first + ' 步。</p>'
      +   '<div class="row">'
      +     '<button class="mt-b mt-ghost" data-act="close">这次先不用</button>'
      +     '<button class="mt-b mt-bp" data-act="here">下一步</button>'
      +   '</div>'
      + '</div>';
    introEl.style.display = "flex";
  }

  function close() {
    if (introEl) introEl.style.display = "none";
  }

  function start(i) {
    build();
    wrap.style.display = "block";
    idx = Math.max(0, Math.min(i, steps.length - 1));
    render();
  }

  function end() {
    if (wrap) wrap.style.display = "none";
  }

  function resolve(st) {
    var node = document.querySelector(st.sel);
    if (!node) return null;
    if (st.up) {
      var up = node.closest(st.up);
      if (up) return up;
    }
    return node;
  }

  var seq = 0;

  function render() {
    var st = steps[idx];
    var node = resolve(st);

    // 目标还没渲染出来（例如数据页在等接口）：最多等 3 秒，仍没有就跳过这一步
    if (!node) {
      if (st._wait == null) st._wait = 0;
      st._wait += 1;
      if (st._wait > 12) {
        st._wait = 0;
        if (idx < steps.length - 1) { idx += 1; render(); } else { end(); }
        return;
      }
      setTimeout(render, 250);
      return;
    }

    try { node.scrollIntoView({ block: "center", inline: "nearest" }); } catch (e) {}

    // 序号守卫：快速连点「下一步」时，上一个 180ms 的定时器不该把旧节点画上去
    seq += 1;
    var my = seq;
    setTimeout(function () { if (my === seq) paint(st, node); }, 180);
  }

  function paint(st, node) {
    var r = node.getBoundingClientRect();
    var p = 6;
    spots.style.display = "block";
    spots.style.left = (r.left - p) + "px";
    spots.style.top = (r.top - p) + "px";
    spots.style.width = Math.max(r.width, 8) + p * 2 + "px";
    spots.style.height = Math.max(r.height, 8) + p * 2 + "px";

    var canPrevInPage = idx > 0;
    var prevPageKey = ORDER[ORDER.indexOf(PAGE.key) - 1];
    var canPrevPage = !canPrevInPage && st.n > 1 && prevPageKey;
    var last = idx === steps.length - 1;

    pop.style.display = "block";
    pop.innerHTML = ""
      + '<div class="mt-row" style="margin-bottom:2px">'
      +   '<span class="mt-k">第 ' + st.n + ' 步 / 共 ' + TOTAL + ' 步</span>'
      +   '<span>' + dots(st.n) + '</span>'
      + '</div>'
      + '<div class="mt-t">' + esc(st.title) + '</div>'
      + '<div class="mt-d">' + esc(st.desc) + '</div>'
      + '<div class="mt-row">'
      +   '<button class="mt-b mt-ghost" data-act="skip">跳过</button>'
      +   '<span style="display:flex;gap:6px">'
      +     (canPrevPage
              ? '<button class="mt-b" data-act="prevpage">上一步</button>'
              : (canPrevInPage ? '<button class="mt-b" data-act="prev">上一步</button>' : ""))
      +     '<button class="mt-b mt-bp" data-act="next">' + (last && !st.next ? "完成" : "下一步") + '</button>'
      +   '</span>'
      + '</div>';

    pop.onclick = function (ev) {
      var b = ev.target.closest("[data-act]");
      if (!b) return;
      var act = b.getAttribute("data-act");
      if (act === "skip") { end(); return; }
      if (act === "prev") { idx -= 1; render(); return; }
      if (act === "prevpage") {
        try { sessionStorage.setItem(RESUME_KEY, String(st.n - 1)); } catch (e) {}
        location.href = PAGES[prevPageKey].url;
        return;
      }
      if (act === "next") {
        if (st.next) {
          try { sessionStorage.setItem(RESUME_KEY, String(st.n + 1)); } catch (e) {}
          location.href = st.next;
          return;
        }
        if (idx < steps.length - 1) { idx += 1; render(); } else { end(); }
      }
    };

    place(r);
  }

  function place(r) {
    var w = 282, pad = 10;
    var vw = window.innerWidth, vh = window.innerHeight;
    var h = pop.offsetHeight || 190;
    var st = steps[idx];

    var left, top;
    if (st.side === "right") {
      left = r.right + 14;
      top = r.top;
      if (left + w > vw - pad) { left = Math.max(pad, r.left - w - 14); }
    } else if (st.side === "top") {
      left = Math.min(Math.max(pad, r.left), vw - w - pad);
      top = r.top - h - 14;
      if (top < pad) top = r.bottom + 14;
    } else {
      left = Math.min(Math.max(pad, r.left), vw - w - pad);
      top = r.bottom + 14;
      if (top + h > vh - pad) top = Math.max(pad, r.top - h - 14);
    }
    pop.style.left = Math.round(left) + "px";
    pop.style.top = Math.round(top) + "px";
  }

  function dots(n) {
    var out = "";
    for (var i = 1; i <= TOTAL; i++) {
      var on = i === n;
      out += '<span style="display:inline-block;width:5px;height:5px;border-radius:50%;margin-left:3px;background:'
        + (on ? "#7cc4ff" : "rgba(255,255,255,.22)") + '"></span>';
    }
    return out;
  }

  function esc(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  // Esc 关闭；左右方向键切换（只在导览打开时生效）
  document.addEventListener("keydown", function (ev) {
    if (!wrap || wrap.style.display === "none") return;
    if (ev.key === "Escape") { end(); }
    else if (ev.key === "ArrowRight") { if (idx < steps.length - 1) { idx += 1; render(); } }
    else if (ev.key === "ArrowLeft") { if (idx > 0) { idx -= 1; render(); } }
  });

  window.addEventListener("resize", function () {
    if (!wrap || wrap.style.display === "none") return;
    var node = resolve(steps[idx]);
    if (node) paint(steps[idx], node);
  });

  function boot() {
    css();
    launchBtn();

    var want = 0;
    try {
      want = Number(sessionStorage.getItem(RESUME_KEY) || 0);
      sessionStorage.removeItem(RESUME_KEY);
    } catch (e) {}

    if (want) {
      for (var i = 0; i < steps.length; i++) {
        if (steps[i].n === want) { start(i); return; }
      }
    }
    // 每次访问都问一次（不记忆、不打扰第二次点击；关掉即本次不再弹）
    ask();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
