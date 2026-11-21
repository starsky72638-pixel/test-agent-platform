'use strict';

/* 可视化 Builder：把画布变成 POST /workflows 能直接接收的 JSON。
 *
 * 分工写死在这里，避免以后有人往这层塞执行逻辑：
 *   - 这一层只产出定义：节点、连线、每个节点的 config；
 *   - 校验、拓扑、执行、Trace 全在后端（agentplatform/），一行都不在这里复制；
 *   - 节点坐标存在 meta.layout 里。NodeDef 只保留 id/type/config/next，
 *     位置若塞进 config 会被当成执行参数原样送给工具。
 */

(() => {
  // palette:false 的节点只能靠 JSON 写，不在左侧工具栏出现——
  // 参数面板还没做它们的表单，放出来等于给一个配不了的空壳。
  // 但必须登记在这里：渲染节点时按 TYPES[type] 取颜色与标题，缺条目会直接报错。
  const TYPES = {
    start: { label: 'Start', color: '#3b82f6', hint: '入口：把用户输入原样交给下游。一个 Workflow 只能有一个。' },
    tool: { label: 'Tool', color: '#10b981', hint: '调用平台已注册的工具；参数支持模板变量。' },
    llm: { label: 'LLM', color: '#a855f7', hint: '调用大模型；Prompt 支持模板变量。' },
    end: { label: 'End', color: '#f59e0b', hint: '终点：输出就是最终答案。' },
    condition: { label: 'Condition', color: '#ef4444', palette: false, hint: '按条件二选一；field/op/value/branches 需在 JSON 里配置，画布暂不支持编辑。' },
    memory: { label: 'Memory', color: '#14b8a6', palette: false, hint: '检索历史经验；query/top_k 需在 JSON 里配置，经验库由后端注入。' },
  };

  const NODE_W = 200;
  const NODE_H = 96;
  const ID_RE = /^[A-Za-z0-9_-]{1,64}$/;

  const $ = (sel) => document.querySelector(sel);
  const esc = (v) => String(v == null ? '' : v).replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  })[c]);

  const S = {
    id: '',
    version: 1,
    name: '',
    description: '',
    nodes: [],
    tools: [],
    params: {},
    defaultTool: '',
    sel: null,
    dirty: false,
  };

  /* ---------------- 基础设施 ---------------- */

  async function readBody(res) {
    let data = null;
    try { data = await res.json(); } catch (_) { data = null; }
    if (res.ok) return data;
    const d = data && data.detail;
    const msg = typeof d === 'string' ? d
      : Array.isArray(d) ? d.map((x) => x.msg || JSON.stringify(x)).join('；')
        : data ? JSON.stringify(data) : (res.statusText || '');
    throw new Error('HTTP ' + res.status + '：' + msg);
  }

  const api = {
    get: (p) => fetch(p).then(readBody),
    post: (p, body) => fetch(p, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }).then(readBody),
  };

  function setStatus(el, text, kind) {
    if (!el) return;
    el.textContent = text || '';
    el.classList.remove('ok', 'err');
    if (kind) el.classList.add(kind);
  }

  function markDirty() {
    S.dirty = true;
    setStatus($('#save-state'), '有未保存的改动');
    renderJson();
  }

  /* ---------------- 节点模型 ---------------- */

  const nodeById = (id) => S.nodes.find((n) => n.id === id) || null;
  const startNode = () => S.nodes.find((n) => n.type === 'start') || null;
  const endNode = () => S.nodes.find((n) => n.type === 'end') || null;

  function uid(type) {
    let i = 1;
    while (S.nodes.some((n) => n.id === type + i)) i += 1;
    return type + i;
  }

  function reaches(fromId, toId, seen) {
    if (fromId === toId) return true;
    if (seen.has(fromId)) return false;
    seen.add(fromId);
    const n = nodeById(fromId);
    return n ? n.next.some((t) => reaches(t, toId, seen)) : false;
  }

  function defaults(type) {
    // 默认只读工具：写操作有副作用、必填参数更多，
    // 拿它当默认值等于让新拖出来的节点第一次运行就报参数缺失。
    if (type === 'tool') return { tool: S.defaultTool || S.tools[0] || '', args: {} };
    if (type === 'llm') return { prompt: '用户问：{{input}}\n\n请用中文简洁回答，只使用上文里出现过的信息。' };
    if (type === 'end') return {};
    return {};
  }

  function paramsOf(toolName) {
    return S.params[toolName] || [];
  }

  function missingRequired(n) {
    const args = n.config.args || {};
    return paramsOf(n.config.tool)
      .filter((p) => p.required && !(p.name in args))
      .map((p) => p.name);
  }

  function updateArgsHint(n) {
    const el = $('#args-hint');
    if (!el) return;
    const specs = paramsOf(n.config.tool);
    if (!specs.length) {
      setStatus(el, '这个工具没有声明参数；确有额外参数时用下面的高级 JSON');
      return;
    }
    const req = specs.filter((p) => p.required).map((p) => p.name);
    if (!req.length) {
      setStatus(el, '这个工具没有必填参数');
      return;
    }
    const miss = missingRequired(n);
    if (miss.length) setStatus(el, '必填但尚未填：' + miss.join('、'), 'err');
    else setStatus(el, '必填已就绪：' + req.join('、'), 'ok');
  }

  // 只有能排在自己前面的节点，其输出才一定可用：
  // 执行器按拓扑序跑，引用下游节点会在运行时报"尚未执行"。
  function ancestorIds(n) {
    return S.nodes
      .filter((x) => x.id !== n.id && reaches(x.id, n.id, new Set()))
      .map((x) => x.id);
  }

  function setParamValue(n, el) {
    const name = el.dataset.param;
    n.config.args = n.config.args || {};
    if (el.value === '') {
      delete n.config.args[name];
    } else if (el.dataset.type === 'number' && /^-?\d+(\.\d+)?$/.test(el.value)) {
      // 只有纯数字才转成 number；模板串（{{...}}）必须原样保留，
      // 否则会被当成 NaN 或被 input[type=number] 判非法而静默丢掉。
      n.config.args[name] = Number(el.value);
    } else {
      n.config.args[name] = el.value;
    }
    markDirty();
    updateArgsHint(n);
    updateSummary(n);
  }

  function fillParamRows(n) {
    const box = $('#inspector');
    if (!box) return;
    box.querySelectorAll('.pval').forEach((el) => {
      const v = (n.config.args || {})[el.dataset.param];
      el.value = v === undefined || v === null ? '' : String(v);
    });
  }

  function summary(n) {
    if (n.type === 'tool') {
      const args = n.config.args || {};
      const body = Object.keys(args).length ? ' ' + JSON.stringify(args) : '';
      return (n.config.tool || '（未选工具）') + body;
    }
    if (n.type === 'llm') return (n.config.prompt || '（空 prompt）').replace(/\s+/g, ' ').slice(0, 72);
    if (n.type === 'end') {
      return n.config.answer ? '模板：' + String(n.config.answer).slice(0, 46) : '取上一个节点的输出';
    }
    if (n.type === 'condition') {
      const cfg = n.config || {};
      const left = String(cfg.field || '?').replace(/\s+/g, ' ').slice(0, 28);
      return left + ' ' + (cfg.op || '?') + ' ' + String(cfg.value == null ? '' : cfg.value).slice(0, 16) + ' → 二选一';
    }
    if (n.type === 'memory') {
      return '检索经验：' + String(n.config.query || '?').slice(0, 40);
    }
    return '用户输入 → 下游';
  }

  /* ---------------- 渲染：节点与连线 ---------------- */

  function renderNodes() {
    const canvas = $('#canvas');
    canvas.querySelectorAll('.node').forEach((el) => el.remove());

    for (const n of S.nodes) {
      const el = document.createElement('div');
      el.className = 'node' + (S.sel === n.id ? ' sel' : '');
      el.dataset.id = n.id;
      el.dataset.type = n.type;
      el.style.left = n.x + 'px';
      el.style.top = n.y + 'px';
      el.style.setProperty('--c', TYPES[n.type].color);
      el.innerHTML =
        '<div class="node-head">' +
        '<span class="badge">' + TYPES[n.type].label + '</span>' +
        '<span class="nid">' + esc(n.id) + '</span>' +
        '</div>' +
        '<div class="node-body">' + esc(summary(n)) + '</div>' +
        '<div class="port" title="按住拖到另一个节点"></div>';

      el.addEventListener('pointerdown', (e) => onNodeDown(e, n));
      const port = el.querySelector('.port');
      port.addEventListener('pointerdown', (e) => onPortDown(e, n));
      canvas.appendChild(el);
    }
  }

  function curve(x1, y1, x2, y2) {
    const dx = Math.max(50, Math.abs(x2 - x1) * 0.55);
    return 'M' + x1 + ',' + y1 + ' C' + (x1 + dx) + ',' + y1 + ' ' + (x2 - dx) + ',' + y2 + ' ' + x2 + ',' + y2;
  }

  function renderEdges() {
    const svg = $('#edges');
    svg.querySelectorAll('.edge').forEach((p) => p.remove());

    for (const n of S.nodes) {
      for (const targetId of n.next) {
        const b = nodeById(targetId);
        if (!b) continue;
        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        path.setAttribute('class', 'edge');
        path.setAttribute('d', curve(n.x + NODE_W, n.y + NODE_H / 2, b.x, b.y + NODE_H / 2));
        path.addEventListener('click', () => {
          n.next = n.next.filter((x) => x !== targetId);
          markDirty();
          renderEdges();
          renderInspector();
          setStatus($('#save-state'), '已删除连线 ' + n.id + ' → ' + targetId, 'ok');
        });
        svg.appendChild(path);
      }
    }
  }

  function setSel(id) {
    S.sel = id;
    document.querySelectorAll('.node').forEach((el) => {
      el.classList.toggle('sel', el.dataset.id === id);
    });
    renderInspector();
  }

  function updateSummary(n) {
    const el = document.querySelector('.node[data-id="' + n.id + '"] .node-body');
    if (el) el.textContent = summary(n);
  }

  /* ---------------- 交互：拖拽与连线 ---------------- */

  function onNodeDown(e, n) {
    if (e.button !== 0) return;
    e.stopPropagation();
    const onHead = !!(e.target.closest && e.target.closest('.node-head'));
    if (S.sel !== n.id) setSel(n.id);
    if (onHead) startDrag(e, n);
  }

  function startDrag(e, n) {
    e.preventDefault();
    const sx = e.clientX;
    const sy = e.clientY;
    const ox = n.x;
    const oy = n.y;
    const el = document.querySelector('.node[data-id="' + n.id + '"]');

    const move = (ev) => {
      n.x = Math.max(0, Math.round(ox + ev.clientX - sx));
      n.y = Math.max(0, Math.round(oy + ev.clientY - sy));
      if (el) {
        el.style.left = n.x + 'px';
        el.style.top = n.y + 'px';
      }
      renderEdges();
    };
    const up = () => {
      document.removeEventListener('pointermove', move);
      document.removeEventListener('pointerup', up);
      markDirty();
    };
    document.addEventListener('pointermove', move);
    document.addEventListener('pointerup', up);
  }

  function onPortDown(e, n) {
    e.stopPropagation();
    e.preventDefault();
    const rect = $('#canvas').getBoundingClientRect();
    const x1 = n.x + NODE_W;
    const y1 = n.y + NODE_H / 2;

    const temp = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    temp.setAttribute('class', 'edge temp');
    temp.setAttribute('d', curve(x1, y1, x1 + 40, y1));
    $('#edges').appendChild(temp);

    const move = (ev) => {
      temp.setAttribute('d', curve(x1, y1, ev.clientX - rect.left, ev.clientY - rect.top));
    };
    const up = (ev) => {
      document.removeEventListener('pointermove', move);
      document.removeEventListener('pointerup', up);
      temp.remove();
      const hit = document.elementFromPoint(ev.clientX, ev.clientY);
      const target = hit && hit.closest ? hit.closest('.node') : null;
      if (target && target.dataset.id) connect(n.id, target.dataset.id);
    };
    document.addEventListener('pointermove', move);
    document.addEventListener('pointerup', up);
  }

  function connect(fromId, toId) {
    if (fromId === toId) return;
    const a = nodeById(fromId);
    if (!a) return;
    if (a.next.includes(toId)) {
      setStatus($('#save-state'), '这条连线已经存在', '');
      return;
    }
    if (reaches(toId, fromId, new Set())) {
      setStatus($('#save-state'), '不能连成环：' + toId + ' 已经能走到 ' + fromId, 'err');
      return;
    }
    a.next.push(toId);
    markDirty();
    renderEdges();
    renderInspector();
    setStatus($('#save-state'), '已连接 ' + fromId + ' → ' + toId, 'ok');
  }

  /* ---------------- 增删节点 ---------------- */

  function addNode(type, x, y) {
    if (type === 'start' && startNode()) {
      setStatus($('#save-state'), '已经有 start 节点：一个 Workflow 只能有一个入口', 'err');
      return;
    }
    const node = {
      id: type === 'start' ? 'start' : uid(type),
      type,
      config: defaults(type),
      next: [],
      x: Math.round(x),
      y: Math.round(y),
    };
    S.nodes.push(node);
    renderNodes();
    renderEdges();
    setSel(node.id);
    markDirty();
  }

  function removeNode(id) {
    S.nodes = S.nodes.filter((n) => n.id !== id);
    for (const n of S.nodes) n.next = n.next.filter((t) => t !== id);
    if (S.sel === id) S.sel = null;
    renderNodes();
    renderEdges();
    renderInspector();
    markDirty();
  }

  function renameNode(oldId, newId) {
    const n = nodeById(oldId);
    if (!n) return;
    if (!newId || newId === oldId) { renderInspector(); return; }
    if (!ID_RE.test(newId)) {
      setStatus($('#save-state'), '节点 id 只允许字母、数字、下划线、短横线', 'err');
      renderInspector();
      return;
    }
    if (nodeById(newId)) {
      setStatus($('#save-state'), '节点 id 重复：' + newId, 'err');
      renderInspector();
      return;
    }
    n.id = newId;
    for (const other of S.nodes) other.next = other.next.map((t) => (t === oldId ? newId : t));
    if (S.sel === oldId) S.sel = newId;
    markDirty();
    renderNodes();
    renderEdges();
    renderInspector();
    setStatus($('#save-state'), '已重命名 ' + oldId + ' → ' + newId, 'ok');
  }

  /* ---------------- 检查器 ---------------- */

  function renderInspector() {
    const box = $('#inspector');
    const n = S.sel ? nodeById(S.sel) : null;
    if (!n) {
      box.innerHTML = '<p class="muted">选中一个节点</p>';
      return;
    }

    const t = TYPES[n.type];
    let html = '<div class="ins-head"><span class="badge" style="--c:' + t.color + '">' + t.label + '</span>' +
      '<button id="btn-del" class="danger">删除节点</button></div>';
    html += '<label>节点 id<input id="cfg-id" value="' + esc(n.id) + '" spellcheck="false" /></label>';
    html += '<p class="hint">' + esc(t.hint) + '</p>';

    if (n.type === 'tool') {
      const opts = S.tools.map((name) => '<option value="' + esc(name) + '"' +
        (name === n.config.tool ? ' selected' : '') + '>' + esc(name) + '</option>').join('');
      html += '<label>工具<select id="cfg-tool">' + opts + '</select></label>';

      // 变量下拉只列"能排在自己前面"的节点，避免引用下游导致运行时报错
      const varOpts = ['<option value="">插入变量…</option>']
        .concat(['<option value="{{input}}">用户输入 {{input}}</option>'])
        .concat(ancestorIds(n).map((id) =>
          '<option value="{{nodes.' + esc(id) + '.output}}">' + esc(id) + ' 的输出</option>'))
        .join('');

      const specs = paramsOf(n.config.tool);
      if (!specs.length) {
        html += '<p class="hint">这个工具没有声明参数；确有额外参数时用下面的高级 JSON。</p>';
      }
      for (const p of specs) {
        const raw = (n.config.args || {})[p.name];
        const val = raw === undefined || raw === null ? '' : String(raw);
        const isNum = p.type === 'number' || p.type === 'integer';
        html += '<div class="param">' +
          '<div class="param-head">' +
          '<span class="pname">' + esc(p.name) + '</span>' +
          '<span class="ptype">' + esc(p.type) + '</span>' +
          (p.required ? '<span class="pneed">必填</span>' : '<span class="popt">可选</span>') +
          '</div>' +
          '<div class="param-line">' +
          '<input class="pval" data-param="' + esc(p.name) + '" data-type="' + (isNum ? 'number' : 'text') +
          '" type="text"' + (isNum ? ' inputmode="numeric"' : '') +
          ' value="' + esc(val) + '" spellcheck="false" />' +
          '<select class="pvar" data-param="' + esc(p.name) + '">' + varOpts + '</select>' +
          '</div>' +
          (p.description ? '<p class="pdesc">' + esc(p.description) + '</p>' : '') +
          '</div>';
      }

      html += '<p class="hint" id="args-hint"></p>';
      html += '<details class="adv"><summary>高级：直接编辑参数 JSON</summary>' +
        '<textarea id="cfg-args" rows="6" spellcheck="false">' +
        esc(JSON.stringify(n.config.args || {}, null, 2)) + '</textarea></details>';
    } else if (n.type === 'llm') {
      html += '<label>Prompt<textarea id="cfg-prompt" rows="9" spellcheck="false">' +
        esc(n.config.prompt || '') + '</textarea></label>';
      html += '<label>模型（留空用 .env 里的默认）<input id="cfg-model" value="' +
        esc(n.config.model || '') + '" spellcheck="false" /></label>';
      html += '<p class="hint">可用变量：{{input}}、{{nodes.&lt;id&gt;.output}}</p>';
    } else if (n.type === 'end') {
      html += '<label>答案模板（留空则取上一个节点的输出）<textarea id="cfg-answer" rows="4" spellcheck="false">' +
        esc(n.config.answer || '') + '</textarea></label>';
    } else if (n.type === 'condition' || n.type === 'memory') {
      // 别落进下面的"入口节点"分支：那句提示会让人以为它是 start。
      const keys = n.type === 'condition'
        ? 'field / op / value / branches'
        : 'query / top_k / min_score / max_chars';
      html += '<p class="hint">这个节点的参数暂不支持在画布上编辑，请在下方 JSON 里改 config（' +
        keys + '）。</p>';
    } else {
      html += '<p class="hint">入口节点没有配置项：它把用户输入原样交给下一个节点。</p>';
    }

    html += '<p class="hint">下游：' + (n.next.length ? n.next.map(esc).join('、') : '（无）') + '</p>';
    box.innerHTML = html;

    $('#btn-del').addEventListener('click', () => removeNode(n.id));
    $('#cfg-id').addEventListener('change', (e) => renameNode(n.id, e.target.value.trim()));

    if (n.type === 'tool') {
      $('#cfg-tool').addEventListener('change', (e) => {
        const next = e.target.value;
        // 换工具要清掉不属于新工具的参数：
        // 残留键会被当成未知参数原样传给工具函数，报"unexpected keyword"，
        // 而在画布上完全看不出它从哪来的。
        const specs = paramsOf(next);
        let dropped = [];
        if (specs.length) {
          const allowed = new Set(specs.map((p) => p.name));
          const kept = {};
          for (const [k, v] of Object.entries(n.config.args || {})) {
            if (allowed.has(k)) kept[k] = v;
            else dropped.push(k);
          }
          n.config.args = kept;
        }
        n.config.tool = next;
        markDirty();
        updateSummary(n);
        // 换工具后参数表也换了，重渲染一次把行与提示对齐
        renderInspector();
        if (dropped.length) {
          setStatus($('#save-state'), '已切换工具，丢弃不匹配的参数：' + dropped.join('、'), '');
        }
      });

      box.querySelectorAll('.pval').forEach((el) => {
        el.addEventListener('input', () => setParamValue(n, el));
      });

      box.querySelectorAll('.pvar').forEach((sel) => {
        sel.addEventListener('change', () => {
          const template = sel.value;
          if (!template) return;
          const name = sel.dataset.param;
          n.config.args = n.config.args || {};
          n.config.args[name] = template;
          const input = box.querySelector('.pval[data-param="' + name + '"]');
          if (input) input.value = template;
          sel.value = '';
          markDirty();
          updateArgsHint(n);
          updateSummary(n);
        });
      });

      const ta = $('#cfg-args');
      ta.addEventListener('input', () => {
        try {
          const parsed = JSON.parse(ta.value || '{}');
          if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
            throw new Error('必须是 JSON 对象');
          }
          n.config.args = parsed;
          // 只回填行里的值，不整块重渲染：否则 JSON 文本框会失去焦点
          fillParamRows(n);
          updateArgsHint(n);
          markDirty();
          updateSummary(n);
        } catch (err) {
          setStatus($('#args-hint'), 'JSON 不合法，未生效：' + err.message, 'err');
        }
      });
      updateArgsHint(n);
    } else if (n.type === 'llm') {
      $('#cfg-prompt').addEventListener('input', (e) => {
        n.config.prompt = e.target.value;
        markDirty();
        updateSummary(n);
      });
      $('#cfg-model').addEventListener('input', (e) => {
        const v = e.target.value.trim();
        if (v) n.config.model = v; else delete n.config.model;
        markDirty();
      });
    } else if (n.type === 'end') {
      $('#cfg-answer').addEventListener('input', (e) => {
        const v = e.target.value;
        if (v.trim()) n.config.answer = v; else delete n.config.answer;
        markDirty();
        updateSummary(n);
      });
    }
  }

  /* ---------------- 定义与 JSON ---------------- */

  function definition() {
    const used = [...new Set(
      S.nodes.filter((n) => n.type === 'tool' && n.config.tool).map((n) => n.config.tool),
    )];
    const layout = {};
    for (const n of S.nodes) layout[n.id] = { x: Math.round(n.x), y: Math.round(n.y) };
    const start = startNode();

    return {
      workflow_id: S.id,
      version: S.version,
      name: S.name,
      description: S.description,
      entry: start ? start.id : '',
      tools: used,
      nodes: S.nodes.map((n) => ({
        id: n.id,
        type: n.type,
        config: n.config,
        next: n.next.slice(),
      })),
      meta: { layout },
    };
  }

  function renderJson() {
    $('#json').textContent = JSON.stringify(definition(), null, 2);
  }

  function localCheck() {
    const errs = [];
    if (!S.id) errs.push('缺少 workflow_id');
    else if (!ID_RE.test(S.id)) errs.push('workflow_id 只允许字母、数字、下划线、短横线');

    const starts = S.nodes.filter((n) => n.type === 'start').length;
    if (starts !== 1) errs.push('必须恰好一个 start 节点，当前 ' + starts + ' 个');
    if (!endNode()) errs.push('至少需要一个 end 节点');

    const ids = new Set();
    for (const n of S.nodes) {
      if (ids.has(n.id)) errs.push('节点 id 重复：' + n.id);
      ids.add(n.id);
    }
    for (const n of S.nodes) {
      for (const t of n.next) if (!ids.has(t)) errs.push(n.id + ' 指向不存在的 ' + t);
      if (n.type !== 'end' && n.next.length === 0) errs.push(n.id + ' 没有下游，流程会在这里断掉');
    }

    // 必填参数在保存前就拦：缺参数时后端能跑，但会在节点里抛 ValidationError，
    // 结果是一条 infra_error + 空回答，看起来像"模型不会答"，其实是参数没填。
    for (const n of S.nodes) {
      if (n.type !== 'tool') continue;
      if (!n.config.tool) {
        errs.push(n.id + ' 没有选工具');
        continue;
      }
      const miss = missingRequired(n);
      if (miss.length) {
        errs.push(n.id + ' 调用 ' + n.config.tool + ' 缺少必填参数：' + miss.join('、'));
      }
    }
    return errs;
  }

  /* ---------------- 保存 / 加载 / 新建 ---------------- */

  async function save() {
    const errs = localCheck();
    if (errs.length) {
      setStatus($('#save-state'), errs.join('；'), 'err');
      return false;
    }
    setStatus($('#save-state'), '保存中…');
    try {
      const res = await api.post('/workflows', definition());
      S.version = res.version || S.version;
      S.dirty = false;
      const idInput = $('#wf-id');
      if (idInput.value.trim() !== S.id) idInput.value = S.id;
      setStatus($('#save-state'), '已保存 ' + res.workflow_id + ' v' + res.version, 'ok');
      await refreshList();
      return true;
    } catch (err) {
      setStatus($('#save-state'), err.message, 'err');
      return false;
    }
  }

  async function refreshList() {
    const sel = $('#wf-list');
    const cur = sel.value;
    try {
      const data = await api.get('/workflows');
      const items = (data && data.agents) || [];
      sel.innerHTML = '<option value="">加载…</option>' + items.map((a) => {
        const tail = (a.name ? ' · ' + a.name : '') + (a.error ? ' · 坏文件' : '');
        return '<option value="' + esc(a.workflow_id) + '">' + esc(a.workflow_id) +
          ' · v' + a.version + tail + '</option>';
      }).join('');
      const keep = [...sel.options].some((o) => o.value === cur) ? cur : '';
      sel.value = keep;
    } catch (err) {
      setStatus($('#save-state'), '读取列表失败：' + err.message, 'err');
    }
  }

  function clearRunPanel() {
    $('#answer').textContent = '（尚未运行）';
    $('#timeline').innerHTML = '<p class="muted">（尚未运行）</p>';
    setStatus($('#run-state'), '');
  }

  function hydrate(d) {
    S.id = d.workflow_id || '';
    S.version = d.version || 1;
    S.name = d.name || '';
    S.description = d.description || '';
    S.sel = null;
    S.dirty = false;

    const layout = (d.meta && d.meta.layout) || {};
    S.nodes = (d.nodes || []).map((n, i) => {
      const pos = layout[n.id];
      const hasPos = pos && typeof pos.x === 'number' && typeof pos.y === 'number';
      return {
        id: n.id,
        type: n.type,
        config: n.config || {},
        next: (n.next || []).slice(),
        x: hasPos ? pos.x : 60 + (i % 3) * 250,
        y: hasPos ? pos.y : 60 + Math.floor(i / 3) * 140,
      };
    });

    syncHeader();
    renderNodes();
    renderEdges();
    renderInspector();
    renderJson();
    clearRunPanel();
  }

  async function openWorkflow(id) {
    if (!id) return;
    try {
      const d = await api.get('/workflows/' + encodeURIComponent(id));
      hydrate(d);
      setStatus($('#save-state'), '已加载 ' + id + ' v' + (d.version || 1), 'ok');
    } catch (err) {
      setStatus($('#save-state'), err.message, 'err');
    }
  }

  function newWorkflow() {
    S.id = '';
    S.version = 1;
    S.name = '';
    S.description = '';
    S.sel = null;
    S.dirty = false;
    S.nodes = [
      { id: 'start', type: 'start', config: {}, next: ['end'], x: 90, y: 120 },
      { id: 'end', type: 'end', config: {}, next: [], x: 470, y: 120 },
    ];
    syncHeader();
    renderNodes();
    renderEdges();
    renderInspector();
    renderJson();
    clearRunPanel();
    const list = $('#wf-list');
    if (list) list.value = '';
    setStatus($('#save-state'), '新建：先填 id，再拖节点');
  }

  function syncHeader() {
    $('#wf-id').value = S.id;
    $('#wf-name').value = S.name;
    $('#wf-desc').value = S.description;
  }

  /* ---------------- 运行与轨迹 ---------------- */

  function renderTimeline(trace) {
    const box = $('#timeline');
    const events = (trace && trace.events) || [];
    const turns = (trace && trace.turns) || [];

    if (!events.length) {
      box.innerHTML = '<p class="muted">这次执行没有节点级事件：' +
        esc((trace && trace.agent) || '') + ' 可能不是 Workflow Agent。</p>';
      return;
    }

    const head = '<div class="tl-head">轮次 ' + turns.length + ' · 事件 ' + events.length +
      ' · 状态 ' + esc(trace.status || '') + '</div>';

    const rows = events.map((e) => {
      let cls = 'muted';
      let label = e.kind;
      let detail = '';

      if (e.kind === 'node_start') {
        cls = 'start';
        label = '进入 ' + e.node_id;
        detail = e.node_type || '';
      } else if (e.kind === 'node_end') {
        cls = 'end';
        label = '离开 ' + e.node_id;
        detail = (e.elapsed_ms || 0) + 'ms · ' + String(e.output || '').slice(0, 50);
      } else if (e.kind === 'tool_call') {
        cls = e.ok === false ? 'error' : 'tool';
        label = '工具 ' + e.name;
        detail = (e.elapsed_ms || 0) + 'ms · ' + JSON.stringify(e.args || {});
      } else if (e.kind === 'llm_call') {
        cls = 'llm';
        label = '模型 ' + e.name;
        detail = (e.elapsed_ms || 0) + 'ms';
      } else if (e.kind === 'error') {
        cls = 'error';
        label = '错误 ' + e.node_id;
        detail = e.error || '';
      }

      const expandable = e.output && e.output.length > 50
        && (e.kind === 'tool_call' || e.kind === 'llm_call' || e.kind === 'node_end');
      const out = expandable
        ? '<details><summary>输出</summary><pre>' + esc(e.output) + '</pre></details>'
        : '';

      return '<div class="tl-row ' + cls + '">' +
        '<span class="seq">' + e.seq + '</span>' +
        '<span class="k">' + esc(label) + '</span>' +
        '<span class="d" title="' + esc(detail) + '">' + esc(detail) + '</span>' +
        out + '</div>';
    });

    box.innerHTML = head + rows.join('');
  }

  async function run() {
    const question = $('#question').value.trim();
    if (!question) {
      setStatus($('#run-state'), '先填一个问题', 'err');
      return;
    }
    if (!S.id) {
      setStatus($('#run-state'), '先填 workflow id 并保存', 'err');
      return;
    }
    if (S.dirty) {
      const saved = await save();
      if (!saved) {
        setStatus($('#run-state'), '保存失败，没有运行——跑的是磁盘上的定义，不是画布', 'err');
        return;
      }
    }

    setStatus($('#run-state'), '运行中…');
    $('#btn-run').disabled = true;
    try {
      const res = await api.post('/runs', { agent: 'wf:' + S.id, question, max_steps: 12 });
      // 节点崩了会返回 status=infra_error 且 answer 为空：
      // 直接显示空回答会让人以为"模型不会答"，必须把 error 摆出来。
      $('#answer').textContent = res.error
        ? '运行中断（' + res.status + '）：' + res.error
        : (res.answer || '（空回答）');
      const full = await api.get('/runs/' + encodeURIComponent(res.run_id));
      renderTimeline(full);
      setStatus($('#run-state'), '完成 · ' + res.status + ' · ' + res.elapsed_ms + 'ms · ' + res.run_id,
        res.error ? 'err' : 'ok');
    } catch (err) {
      $('#answer').textContent = '运行失败：' + err.message;
      $('#timeline').innerHTML = '<p class="muted">（没有轨迹）</p>';
      setStatus($('#run-state'), err.message, 'err');
    } finally {
      $('#btn-run').disabled = false;
    }
  }

  /* ---------------- 左侧节点栏 ---------------- */

  function buildPalette() {
    const box = $('#palette');
    box.innerHTML = '';
    for (const [type, meta] of Object.entries(TYPES)) {
      if (meta.palette === false) continue;
      const btn = document.createElement('div');
      btn.className = 'pal-btn';
      btn.draggable = true;
      btn.style.setProperty('--c', meta.color);
      btn.innerHTML = '<b>' + meta.label + '</b>';
      btn.title = meta.hint;
      btn.addEventListener('click', (e) => {
        const rect = $('#canvas').getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        if (x > 0 && y > 0 && x < 2600 && y < 1800) addNode(type, x - 40, y - 30);
        else addNode(type, 90 + S.nodes.length * 26, 90 + S.nodes.length * 26);
      });
      btn.addEventListener('dragstart', (e) => {
        e.dataTransfer.setData('text/x-node-type', type);
        e.dataTransfer.effectAllowed = 'copy';
      });
      box.appendChild(btn);
    }
  }

  /* ---------------- 绑定与启动 ---------------- */

  function bind() {
    $('#wf-id').addEventListener('input', (e) => { S.id = e.target.value.trim(); markDirty(); });
    $('#wf-name').addEventListener('input', (e) => { S.name = e.target.value; markDirty(); });
    $('#wf-desc').addEventListener('input', (e) => { S.description = e.target.value; markDirty(); });

    $('#btn-save').addEventListener('click', save);
    $('#btn-new').addEventListener('click', newWorkflow);

    // 导出 = 下载当前定义的 JSON，文件名用 workflow_id。
    // 先跑与保存同一套本地校验：结构不合法时导出，拿到的是一个跑不起来的文件，比不给更糟。
    $('#btn-export').addEventListener('click', () => {
      const errs = localCheck();
      if (errs.length) { setStatus($('#save-state'), errs.join('；'), 'err'); return; }
      const name = (S.id || 'workflow') + '.json';
      const blob = new Blob([JSON.stringify(definition(), null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = name;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 0);
      setStatus($('#save-state'), '已导出 ' + name, 'ok');
    });
    $('#wf-list').addEventListener('change', (e) => openWorkflow(e.target.value));
    $('#btn-run').addEventListener('click', run);

    $('#btn-copy').addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText($('#json').textContent || '{}');
        setStatus($('#copy-state'), '已复制', 'ok');
      } catch (err) {
        setStatus($('#copy-state'), '复制失败：' + err.message, 'err');
      }
    });

    const canvas = $('#canvas');
    canvas.addEventListener('dragover', (e) => { e.preventDefault(); e.dataTransfer.dropEffect = 'copy'; });
    canvas.addEventListener('drop', (e) => {
      e.preventDefault();
      const type = e.dataTransfer.getData('text/x-node-type');
      if (!type || !TYPES[type]) return;
      const rect = canvas.getBoundingClientRect();
      addNode(type, e.clientX - rect.left - 100, e.clientY - rect.top - 48);
    });
    canvas.addEventListener('pointerdown', () => setSel(null));

    document.addEventListener('keydown', (e) => {
      const tag = (e.target && e.target.tagName || '').toLowerCase();
      if (tag === 'input' || tag === 'textarea' || tag === 'select') return;
      if ((e.key === 'Delete' || e.key === 'Backspace') && S.sel) {
        e.preventDefault();
        removeNode(S.sel);
      }
    });
  }

  async function boot() {
    bind();
    try {
      const data = await api.get('/tools');
      S.tools = Array.isArray(data && data.tools) ? data.tools : [];
      S.params = (data && data.params) || {};
      S.defaultTool = (data && data.default) || '';
    } catch (err) {
      S.tools = [];
      S.params = {};
      S.defaultTool = '';
      setStatus($('#save-state'), '读取工具列表失败：' + err.message, 'err');
    }
    buildPalette();
    newWorkflow();
    await refreshList();
  }

  boot();
})();
