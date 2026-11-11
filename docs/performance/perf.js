// Charts for the Performance page. Reads window.PRISM_PERF (data.js) and draws into each
// .pq-figure .pq-plot slot; titles, notes and exact-number tables stay static in index.md.
(function () {
'use strict';
const D = window.PRISM_PERF;
const FIGS = Array.prototype.slice.call(document.querySelectorAll('.pq-figure[data-chart]'));
if (!D || !FIGS.length) return;
const ROOT = document.documentElement;
ROOT.classList.add('pq-perf');

// SVG text sizes in px. perf.css reads them as --pq-svg-*, so label measurement and rendering agree.
const FS = {text: 12.5, col: 12, value: 11.5, label: 11, tick: 10.5};
Object.keys(FS).forEach(k => ROOT.style.setProperty('--pq-svg-' + k, FS[k] + 'px'));
const NARROW = 560, boxW = b => b.clientWidth || 640;

const NS = 'http://www.w3.org/2000/svg', O = D.observations, draws = [];
const NAME = {prismq: 'PRISM-Q', 'aer-statevector': 'Aer', 'aer-statevector-f32': 'Aer fp32', 'aer-automatic': 'Aer automatic',
  'aer-mps': 'Aer MPS', qsim: 'qsim fp32', quest: 'QuEST', spinoza: 'Spinoza', qip: 'qip', stim: 'Stim',
  'quimb-tn': 'quimb TN', 'quimb-mps:256': 'quimb MPS cap 256', 'aer-mps:256': 'Aer MPS cap 256',
  'prismq:mps-256': 'PRISM-Q MPS cap 256', 'prismq:tensor-network': 'PRISM-Q TN', 'prismq:no-qft-block': 'PRISM-Q, QFT recognition off'};
const VER = {prismq: 'PRISM-Q', 'aer-statevector': 'Aer statevector', 'aer-statevector-f32': 'Aer statevector (fp32)',
  'aer-automatic': 'Aer automatic', 'aer-mps': 'Aer MPS', qsim: 'qsim (fp32)', quest: 'QuEST', spinoza: 'Spinoza', qip: 'qip', stim: 'Stim',
  'quimb-tn': 'quimb TN', 'quimb-mps': 'quimb MPS'};
const STAT = {oom: 'out of memory', crash: 'crash', timeout: 'timeout', invalid: 'invalid', error: 'refused',
  'not-applicable': 'cannot run', skipped: 'not run'};
const STUB = Object.assign({}, STAT, {oom: 'oom'});
const key = s => s.split(':')[0];
const nm = o => NAME[o.simulator] || o.name;
const ver = o => o.version || D.versions[VER[key(o.simulator)]] || '';
const statusWord = (o, short) => (short ? STUB : STAT)[o.status] || o.status;
const ROLE = {pq: 0, ref: 1, peer: 2};
const roleOf = (s, ref) => s === 'prismq' ? 'pq' : s === ref ? 'ref' : 'peer';

// Formats mirror comparison/harness/pages.py so chart labels agree with the static tables.
const fms = v => v == null ? 'n/a' : v < 1 ? Math.round(v * 1000) + ' us' : v >= 1000 ? (v / 1000).toFixed(2) + ' s'
  : v < 10 ? v.toFixed(2) + ' ms' : v < 100 ? v.toFixed(1) + ' ms' : Math.round(v) + ' ms';
const fx = v => (v >= 100 ? Math.round(v) : v >= 10 ? v.toFixed(1) : v.toFixed(2)) + 'x';
const fmb = v => v == null ? 'n/a' : v >= 1024 ? (v / 1024).toFixed(2) + ' GB' : Math.round(v) + ' MB';
const fdist = t => t == null ? 'n/a' : t >= 0.01 ? String(+t.toPrecision(2)) : t.toExponential(1);
const span = rs => { const a = fx(Math.min.apply(null, rs)), b = fx(Math.max.apply(null, rs)); return a === b ? a : a + ' to ' + b; };
const ftick = v => v < 1 ? +(v * 1000).toPrecision(3) + ' us' : v < 1000 ? +v.toPrecision(3) + ' ms' : +(v / 1000).toPrecision(3) + ' s';
const fxt = v => +v.toPrecision(3) + 'x';
const ferr = t => t >= 0.01 ? String(t) : '1e' + Math.round(Math.log10(t));
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
const median = a => { const s = a.slice().sort((x, y) => x - y), m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };
const uniq = a => a.filter((x, i) => a.indexOf(x) === i);
const plural = (n, w) => n + ' ' + w + (n > 1 ? 's' : '');

// Each simulator's record at its own top thread count; `at` keeps every thread count, and
// `get(g, w, 'sim@T')` reads one. A ratio pairs a peer with the PRISM-Q record at its thread
// count where one exists.
const top = {}, idx = {}, at = {};
O.forEach(o => { const g = o.group + '|' + o.workload + '|' + o.simulator; if (!(top[g] >= o.threads)) top[g] = o.threads; });
O.forEach(o => {
  const g = o.group + '|' + o.workload + '|' + o.simulator;
  at[g + '@' + o.threads] = o;
  if (o.threads === top[g]) idx[g] = o;
});
const get = (g, w, s) => s.indexOf('@') > 0 ? at[g + '|' + w + '|' + s] : idx[g + '|' + w + '|' + s];
const BASE = {mpsdeep: 'prismq:mps-256'};
const baseOf = g => BASE[g] || 'prismq';
const isTop = o => o.threads === top[o.group + '|' + o.workload + '|' + o.simulator];
const ok = o => !!o && o.status === 'ok' && o.median_ms != null;
// `validated` and `valid` come from comparison/harness/aggregate.py: a ratio needs both sides
// validated (passed a check or the row's reference); `valid` also admits untested outputs.
const valid = o => ok(o) && o.validated === true;
const usable = o => ok(o) && o.valid === true;
const mate = (o, pq) => pq && (at[o.group + '|' + o.workload + '|' + pq.simulator + '@' + o.threads] || pq);
// `o.ratio` is the harness's ratio against the group's base PRISM-Q record at the same thread
// count, so the charts print the digits the static tables print; dividing rounded medians can differ.
const pairRatio = (o, q) => o.ratio != null && q.threads === o.threads && q.simulator === baseOf(o.group) ? o.ratio
  : o.median_ms / q.median_ms;
const ratio = (p, q) => { q = p && mate(p, q); return valid(p) && valid(q) ? pairRatio(p, q) : null; };
const peerOf = s => s.indexOf('prismq') !== 0;

// `split` names the field and values of a section's View control; `base` is the ratio column's
// denominator when it is not plain PRISM-Q.
const SEC = [
  {id: 'sv64', name: 'Statevector, fp64', chip: 'Statevector fp64', g: 'headline', fam: ['rcs', 'qft', 'qv'],
    sims: ['prismq', 'aer-statevector', 'quest', 'spinoza', 'qip'], ref: 'aer-statevector', split: ['qubits', [16, 20, 24, 26]]},
  {id: 'sv32', name: 'Statevector against fp32 peers', chip: 'fp32 peers', g: 'headline', fam: ['rcs', 'qft', 'qv'],
    sims: ['prismq', 'aer-statevector-f32', 'qsim'], ref: 'qsim', split: ['qubits', [16, 20, 24, 26]]},
  {id: 'stabilizer', name: 'Stabilizer', chip: 'Stabilizer', g: 'headline', fam: ['clifford'], sims: ['prismq', 'prismq@1', 'stim'],
    ref: 'stim'},
  {id: 'highshot', name: 'High-shot stabilizer', chip: 'High-shot', g: 'highshot', fam: ['clifford'],
    sims: ['prismq', 'prismq@1', 'stim'], ref: 'stim', base: 'PRISM-Q 1t', byThreads: true},
  {id: 'mps', name: 'Matrix product state', chip: 'MPS', g: 'headline', fam: ['lowent'],
    sims: ['prismq', 'aer-mps', 'aer-automatic'], ref: 'aer-mps', split: ['depth', [4, 8]]},
  {id: 'mpsdeep', name: 'Deep MPS', chip: 'Deep MPS', g: 'mpsdeep', fam: ['lowent'],
    sims: ['prismq:mps-256', 'aer-mps:256', 'quimb-mps:256'], ref: 'aer-mps:256', split: ['depth', [12, 16]], base: 'PRISM-Q MPS'},
  {id: 'tn', name: 'Tensor network expectation values', chip: 'TN expval', g: 'tn', fam: ['rcs'],
    sims: ['prismq', 'prismq:tensor-network', 'prismq:mps-256', 'quimb-tn', 'aer-mps'], ref: 'quimb-tn'},
  {id: 'sampling', name: 'Shot sampling', chip: 'Sampling', g: 'sampling', fam: ['sampling'],
    sims: ['prismq', 'aer-statevector', 'aer-statevector-f32', 'qsim'], ref: 'aer-statevector', split: ['qubits', [16, 20, 24]]},
  {id: 'disjoint', name: 'Disjoint components', chip: 'Disjoint', g: 'headline', fam: ['disjoint'],
    sims: ['prismq', 'aer-mps', 'aer-automatic'], ref: 'aer-mps'}
];
const SECM = {};
SEC.forEach(s => { SECM[s.id] = s; });
const FAM = ['rcs', 'qft', 'qv', 'clifford', 'lowent', 'sampling', 'disjoint'];
const order = (a, b) => FAM.indexOf(a.family) - FAM.indexOf(b.family) || (a.depth || 0) - (b.depth || 0) || a.qubits - b.qubits ||
  (a.measured === 'half') - (b.measured === 'half') || String(a.layout || '').localeCompare(String(b.layout || '')) || (a.shots || 0) - (b.shots || 0);
function secRows(sec) {
  const pqs = O.filter(o => o.group === sec.g && sec.fam.indexOf(o.family) >= 0 && o.simulator === sec.sims[0] && isTop(o)).sort(order);
  const one = sec.fam.length === 1;
  return pqs.map(pq => ({w: pq.workload, label: pq.label, short: one ? pq.label.replace(pq.family + ' ', '') : pq.label, pq: pq,
    peers: sec.sims.slice(1).map(s => get(sec.g, pq.workload, s)).filter(o => o && o !== pq), ref: sec.ref}));
}

// ---- DOM helpers
function S(tag, attrs, par) {
  const e = document.createElementNS(NS, tag);
  for (const a in attrs) e.setAttribute(a, attrs[a]);
  if (par) par.appendChild(e);
  return e;
}
function T(par, x, y, s, cls, anchor) {
  const t = S('text', {x: x, y: y}, par);
  if (cls) t.setAttribute('class', cls);
  if (anchor) t.setAttribute('text-anchor', anchor);
  t.textContent = s;
  return t;
}
function H(tag, cls, par, txt) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (txt != null) e.textContent = txt;
  if (par) par.appendChild(e);
  return e;
}
const cv = document.createElement('canvas').getContext('2d');
let FF = 'sans-serif';
function tw(s, px, bold) { cv.font = (bold ? '600 ' : '') + px + 'px ' + FF; return cv.measureText(s).width; }
const hit = (b, L) => L.some(c => b.x0 < c.x1 && b.x1 > c.x0 && b.y0 < c.y1 && b.y1 > c.y0);
function svgEl(W, Hh, label) {
  return S('svg', {class: 'pq-svg', role: 'group', 'aria-label': label, width: W, height: Hh, viewBox: '0 0 ' + W + ' ' + Hh});
}
function size(svg, W, Hh) {
  svg.setAttribute('width', W);
  svg.setAttribute('height', Hh);
  svg.setAttribute('viewBox', '0 0 ' + W + ' ' + Hh);
}
// One tab stop per chart or matrix; arrow keys move between its marks.
function rove(root) {
  root.querySelectorAll('svg.pq-svg, table.pq-matrix').forEach(c => {
    c.querySelectorAll('.pq-hot').forEach((h, i) => h.setAttribute('tabindex', i ? '-1' : '0'));
  });
}
// Register a panel's draw for resizes and run it; the returned function redraws after a control changes.
function mount(plot, draw) {
  const run = () => { draw(); rove(plot); highlight(pin); };
  draws.push(run);
  run();
  return run;
}
function legend(el, items) {
  el.replaceChildren();
  el.hidden = !items.length;
  items.forEach(it => {
    const b = H(it.k ? 'button' : 'span', 'pq-key ' + (it.cls || '') + (it.k ? '' : ' static'), el);
    H('i', '', b);
    b.appendChild(document.createTextNode(it.name));
    if (it.k) { b.type = 'button'; b.setAttribute('data-k', it.k); }
  });
}
// One key per series in role order, then the hollow-mark key when a drawn point failed its check.
function legendItems(list, failed) {
  const seen = {}, items = list.filter(it => !seen[it.k] && (seen[it.k] = 1));
  items.sort((a, b) => ROLE[a.cls] - ROLE[b.cls]);
  if (failed && items.length) items.push({cls: 'open', name: 'output failed its check'});
  return items;
}
function hot(el, tipHtml, label) {
  el.__tip = tipHtml;
  el.classList.add('pq-hot');
  el.setAttribute('tabindex', '-1');
  if (label) el.setAttribute('aria-label', label);
  return el;
}
function mark(el, o, k) {
  el.__o = o;
  el.classList.add('pq-hot');
  el.setAttribute('data-k', k || key(o.simulator));
  el.setAttribute('tabindex', '-1');
  el.setAttribute('aria-label', nm(o) + ', ' + o.label + ': ' + (o.median_ms != null ? fms(o.median_ms) : statusWord(o)));
  return el;
}
function nameText(par, x, y, s, k, cls, anchor) {
  const t = T(par, x, y, s, 'pq-name ' + (cls || ''), anchor);
  t.setAttribute('data-k', k);
  return t;
}
function rowBg(g, W, y, h) {
  const r = S('rect', {class: 'pq-rowbg', x: -6, width: W + 12, rx: 6}, g);
  if (y != null) { r.setAttribute('y', y); r.setAttribute('height', h); }
  return r;
}
// A dashed pill holding a status word where a bar or value would be.
function stub(g, x, cy, word) {
  const w = tw(word, FS.tick) + 14, r = S('rect', {class: 'pq-stub', x: x, y: cy - 7, width: w, height: 14, rx: 7}, g);
  T(g, x + w / 2, cy + 3.6, word, 'pq-stubt', 'middle');
  return r;
}
function title(fig) { const t = fig.querySelector('.pq-title'); return t ? t.textContent : fig.getAttribute('data-chart'); }

function ticks(lo, hi, log, maxN) {
  const t = [];
  if (log) {
    const dec = Math.log10(hi / lo), muls = dec < 2.3 ? [1, 2, 5] : [1], st = Math.max(1, Math.ceil(dec / (maxN || 9)));
    for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) {
      if (((e % st) + st) % st) continue;
      muls.forEach(m => { const v = +(m * Math.pow(10, e)).toPrecision(3); if (v >= lo * 0.999 && v <= hi * 1.001) t.push(v); });
    }
  } else {
    const raw = (hi - lo) / Math.max(2, maxN || 5), p = Math.pow(10, Math.floor(Math.log10(raw)));
    const st = [1, 2, 2.5, 5, 10].map(m => m * p).find(s => s >= raw);
    for (let v = Math.ceil(lo / st - 1e-9) * st; v <= hi + 1e-9; v += st) t.push(+v.toPrecision(6));
  }
  return t;
}
// Log domain [d0, d1] that keeps [lo, hi] inside with wl/wr pixels free at the ends.
function fitLog(lo, hi, P, wl, wr) {
  let a = Math.log(lo), b = Math.log(hi);
  if (b - a < Math.log(3)) { const m = (a + b) / 2; a = m - Math.log(3) / 2; b = m + Math.log(3) / 2; }
  const k = Math.min(0.7, (wl + wr) / P), s = (b - a) / (1 - k);
  return [a - wl * s / P, b + wr * s / P];
}
const logScale = (dom, a, P) => x => a + (Math.log(x) - dom[0]) / (dom[1] - dom[0]) * P;
// Vertical grid lines over `o.spans` (class from `o.line`, none when it returns null) and tick
// labels at `o.y`, skipping any label that would touch its neighbour or leave [0, o.W].
function xAxis(g, list, X, fmt, o) {
  let last = -1e9;
  list.forEach(t => {
    const x = X(t), s = fmt(t), w = tw(s, FS.tick), cls = o.line ? o.line(t) : 'pq-grid';
    if (cls) (o.spans || []).forEach(sp => S('line', {class: cls, x1: x, x2: x, y1: sp[0], y2: sp[1]}, g));
    if (x - w / 2 < Math.max(0, last + 8) || x + w / 2 > o.W) return;
    T(g, x, o.y, s, 'pq-tick' + (o.one && t === 1 ? ' one' : ''), 'middle');
    last = x + w / 2;
  });
}

function seg(par, labels, def, on, aria) {
  const d = H('div', 'pq-seg', par);
  d.setAttribute('role', 'group');
  d.setAttribute('aria-label', aria);
  labels.forEach((l, i) => {
    const b = H('button', '', d, l);
    b.type = 'button';
    b.setAttribute('aria-pressed', i === def);
    b.onclick = () => { Array.prototype.forEach.call(d.children, (x, j) => x.setAttribute('aria-pressed', j === i)); on(i); };
  });
  return d;
}

// ---- tooltip, linked highlight, pinning
const tip = H('div', 'pq-tip', document.body);
tip.setAttribute('role', 'tooltip');
tip.id = 'pq-tip';
let pin = null, hov = null, tipFor = null, touch = false, hideT = 0;
function highlight(k) {
  document.querySelectorAll('.pq-plot [data-k]').forEach(e => {
    if (e.classList.contains('pq-key')) { e.setAttribute('aria-pressed', !!pin && e.getAttribute('data-k') === pin); return; }
    const m = e.getAttribute('data-k') === k || key(e.getAttribute('data-k')) === k && k === 'prismq';
    e.classList.toggle('pq-dim', !!k && !m);
    e.classList.toggle('pq-hl', !!k && m);
  });
  ROOT.classList.toggle('pq-pinned', !!pin);
}
const tgrid = rows => '<dl class="pq-tg">' + rows.map(r => '<dt>' + esc(r[0]) + '</dt><dd>' + esc(r[1]) + '</dd>').join('') + '</dl>';
function tipHTML(o) {
  const meta = [esc(o.label), plural(o.threads, 'thread')];
  if (o.precision) meta.push(esc(o.precision));
  if (o.resolved_backend) meta.push(esc(o.resolved_backend));
  else if (o.method) meta.push(esc(o.method));
  const G = [];
  const pq = peerOf(o.simulator) && mate(o, get(o.group, o.workload, baseOf(o.group))), r = pq && ratio(o, pq);
  if (r) G.push(['x ' + (NAME[pq.simulator] || pq.name) + ' (' + fms(pq.median_ms) +
    (pq.threads !== o.threads ? ', ' + plural(pq.threads, 'thread') : '') + ')', fx(r)]);
  if (o.median_ms != null) G.push(['median', fms(o.median_ms)]);
  if (o.p25_ms != null) G.push(['p25 to p75' + (o.n ? ', n = ' + o.n : ''), fms(o.p25_ms) + ' to ' + fms(o.p75_ms)]);
  else if (o.n) G.push(['n', o.n]);
  if (o.first_ms != null) G.push(['cold call', fms(o.first_ms)]);
  if (o.min_ms != null) G.push(['fastest warm', fms(o.min_ms)]);
  if (o.internal_ms != null) G.push(['time Aer reports', fms(o.internal_ms)]);
  if (o.peak_rss_mb) G.push(['peak memory', fmb(o.peak_rss_mb)]);
  if (o.max_abs_error != null) G.push(['largest error against exact', fdist(o.max_abs_error)]);
  if (o.tvd != null) G.push(['distance from reference', fdist(o.tvd)]);
  if (o.min_detectable != null && o.role !== 'weakly-tested') G.push(['check detects shifts above', fdist(o.min_detectable)]);
  let s = '';
  if (o.status !== 'ok') s = esc(statusWord(o)) + (o.reason ? ': ' + esc(o.reason) : '');
  else if (o.role === 'too-few-shots') s = 'under 100 shots: checked on deterministic bits only, no ratio';
  else if (o.role === 'reference') s = 'reference output' + (o.cross_checked ? ', cross-checked' : ', no cross-check');
  else if (o.role === 'weakly-tested') s = 'weakly tested: detects shifts above ' + fdist(o.min_detectable) + ' in a marginal';
  else if (!o.validated) s = 'timed, output not checked';
  return '<div class="pq-th">' + esc(nm(o)) + ' <span>' + esc(ver(o)) + '</span></div><div class="pq-tm">' + meta.join(' · ') + '</div>' +
    (G.length ? tgrid(G) : '') + (s ? '<div class="pq-ts">' + s + '</div>' : '');
}
function show(m) {
  tipFor = m;
  tip.innerHTML = m.__o ? tipHTML(m.__o) : typeof m.__tip === 'function' ? m.__tip() : m.__tip;
  tip.classList.add('on');
  m.setAttribute('aria-describedby', 'pq-tip');
  place(m);
  const row = m.closest('.pq-row');
  if (row) row.classList.add('on');
}
function place(m) {
  const b = m.getBoundingClientRect(), tb = tip.getBoundingClientRect(), ax = m.__ax != null ? b.left + m.__ax : b.left + b.width / 2;
  const x = ax - tb.width / 2;
  let y = b.top - tb.height - 10;
  const fig = m.closest('.pq-figure'), tt = fig && fig.querySelector('.pq-title');
  if (tt) { const r = tt.getBoundingClientRect(); if (y < r.bottom && y + tb.height > r.top) y = b.bottom + 10; }
  if (y < 4) y = b.bottom + 10;
  if (y + tb.height > innerHeight - 4) y = Math.max(4, b.top - tb.height - 10);
  tip.style.left = Math.max(6, Math.min(x, innerWidth - tb.width - 6)) + 'px';
  tip.style.top = y + 'px';
}
function hide() {
  clearTimeout(hideT);
  tip.classList.remove('on');
  if (tipFor) tipFor.removeAttribute('aria-describedby');
  tipFor = null;
  document.querySelectorAll('.pq-row.on').forEach(e => e.classList.remove('on'));
}
// A short grace period lets the pointer cross from a mark onto the tooltip.
function hideSoon() { clearTimeout(hideT); hideT = setTimeout(hide, 160); }
function over(e) {
  if (!e.target.closest) return;
  if (tip.contains(e.target)) { clearTimeout(hideT); return; }
  const t = e.target.closest('.pq-plot .pq-hot');
  if (t) { clearTimeout(hideT); if (t !== tipFor) { hide(); show(t); } }
  const n = e.target.closest('.pq-plot [data-k]');
  const k = n && (n.classList.contains('pq-name') || n.classList.contains('pq-key') || e.type === 'focusin') ? n.getAttribute('data-k') : null;
  if (k !== hov) { hov = k; highlight(hov || pin); }
}
// A touch tap leaves the tooltip up until the next tap elsewhere, so synthetic mouseouts are ignored.
function out(e) {
  if (!e.target.closest) return;
  const t = e.target.closest('.pq-plot .pq-hot'), to = e.relatedTarget;
  if (t && !(to && (t.contains(to) || tip.contains(to)))) {
    if (e.type === 'focusout') hide(); else if (!touch) hideSoon();
  }
  if (hov && (!to || e.type === 'focusout')) { hov = null; highlight(pin); }
}
function toggle(t) {
  pin = pin === t.getAttribute('data-k') ? null : t.getAttribute('data-k');
  hov = null;
  highlight(pin);
}
document.addEventListener('mouseover', over);
document.addEventListener('mouseout', out);
document.addEventListener('focusin', over);
document.addEventListener('focusout', out);
tip.addEventListener('mouseleave', e => { if (!touch && !(e.relatedTarget && tipFor && tipFor.contains(e.relatedTarget))) hideSoon(); });
document.addEventListener('click', e => {
  const t = e.target.closest && e.target.closest('.pq-plot [data-k]');
  if (t) toggle(t);
});
document.addEventListener('pointerdown', e => {
  touch = e.pointerType === 'touch';
  if (!(e.target.closest && e.target.closest('.pq-hot, .pq-tip'))) hide();
});
const STEP = {ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1, Home: -1e9, End: 1e9};
const clampAt = (list, i) => list[Math.max(0, Math.min(list.length - 1, i))];
// In a matrix, Up and Down keep the column and the other keys stay in the row.
function step(h, k) {
  const c = h.closest('svg, table');
  if (c.tagName !== 'TABLE') {
    const hs = Array.prototype.slice.call(c.querySelectorAll('.pq-hot'));
    return clampAt(hs, hs.indexOf(h) + STEP[k]);
  }
  if (k === 'ArrowUp' || k === 'ArrowDown') {
    for (let tr = h.parentNode; (tr = k === 'ArrowUp' ? tr.previousElementSibling : tr.nextElementSibling);) {
      const td = tr.cells[h.cellIndex];
      if (td && td.classList.contains('pq-hot')) return td;
    }
    return null;
  }
  const hs = Array.prototype.filter.call(h.parentNode.cells, td => td.classList.contains('pq-hot'));
  return clampAt(hs, hs.indexOf(h) + STEP[k]);
}
// Capture phase, so a key a chart consumes never reaches the book's chapter navigation.
document.addEventListener('keydown', e => {
  const h = STEP[e.key] && !e.altKey && !e.ctrlKey && !e.metaKey && e.target.closest && e.target.closest('.pq-plot .pq-hot');
  if (!h) return;
  e.preventDefault();
  e.stopImmediatePropagation();
  const n = step(h, e.key);
  if (n && n !== h) { h.setAttribute('tabindex', '-1'); n.setAttribute('tabindex', '0'); n.focus(); }
}, true);
document.addEventListener('keydown', e => {
  if (e.key === 'Escape' && (pin || tipFor)) { pin = null; hov = null; hide(); highlight(null); }
  const t = e.target.closest && e.target.closest('.pq-plot [data-k]');
  if ((e.key === 'Enter' || e.key === ' ') && t && t.tagName !== 'BUTTON') { e.preventDefault(); toggle(t); }
});
addEventListener('scroll', () => { if (tipFor) { if (tipFor.isConnected) place(tipFor); else hide(); } }, {passive: true});

// ---- dot panel: one row per workload, a log value axis, direct labels placed without overlap
function dotLayout(W, spec, v) {
  const narrow = W < NARROW, cols = (v.cols || []).filter(c => !(narrow && c.wide));
  const colW = cols.reduce((a, c) => a + c.w, 0);
  let maxL = 0;
  spec.views.forEach(vw => vw.rows.forEach(r => { maxL = Math.max(maxL, tw(r.short || r.label, FS.text)); }));
  const labW = narrow ? 0 : Math.round(Math.max(96, Math.min(maxL + 10, W * 0.3)));
  return {W: W, narrow: narrow, cols: cols, labW: labW, lineW: narrow ? W : labW,
    ax0: narrow ? 10 : labW + 16, ax1: W - colW - (cols.length ? 20 : 10)};
}
// Text items under a row label: an optional subline, then each record that has no dot with its status.
function failItems(R, nmf) {
  const items = R.r.sub ? [{s: R.r.sub}] : [];
  let last;
  R.fails.slice().sort((a, b) => a.status.localeCompare(b.status)).forEach(o => {
    const w = statusWord(o);
    if (R.fails.length > 1) { if (o.status !== last) items.push({s: w + ':', f: 1}); items.push({o: o, s: nmf(o)}); }
    else items.push({o: o, s: nmf(o) + ' ' + w});
    last = o.status;
  });
  return items;
}
// Draw one workload row from top `y`; returns its bottom. Dots and labels go into obst and reqs.
function dotRow(gRows, R, v, L, X, y, obst, reqs) {
  const g = S('g', {class: 'pq-row'}, gRows), bg = rowBg(g, L.W), r = R.r, refSim = r.ref || v.ref, nmf = v.name || nm;
  const rl = r.short || r.label, lw = tw(rl, FS.text), ly = L.narrow ? y + 15 : y + 24;
  const lab = T(g, 0, ly, rl, 'pq-rl'), lob = {x0: 0, x1: lw + 2, y0: ly - 12, y1: ly + 4};
  obst.push(lob);
  let lx = L.narrow ? lw + 12 : 0, fy = L.narrow ? ly : ly + 15;
  const items = failItems(R, nmf);
  items.forEach((it, i) => {
    const w = tw(it.s, FS.label) + (it.f ? 4 : 10), need = it.f && items[i + 1] ? w + tw(items[i + 1].s, FS.label) : w;
    if (lx + need > L.lineW && lx > 0) { lx = 0; fy += 14; }
    const t = T(g, lx, fy, it.s, it.o ? 'pq-fail' : it.f ? 'pq-fail pq-fh' : 'pq-sub');
    if (it.o) mark(t, it.o, v.kOf ? v.kOf(it.o) : null);
    obst.push({x0: lx, x1: lx + w, y0: fy - 11, y1: fy + 4});
    lx += w;
  });
  const cy = L.narrow ? fy + 27 : y + 20, val = v.val || (o => o.median_ms);
  const xs = R.pts.map(o => X(val(o, r)));
  if (xs.length > 1) S('line', {class: 'pq-span', x1: Math.min.apply(null, xs), x2: Math.max.apply(null, xs), y1: cy, y2: cy}, g);
  R.pts.slice().sort((a, b) => (a === r.pq) - (b === r.pq) || (a.simulator === refSim) - (b.simulator === refSim)).forEach(o => {
    const role = o === r.pq ? 'pq' : o.simulator === refSim ? 'ref' : 'peer', cx = X(val(o, r)), rad = role === 'pq' ? 5.5 : 4.5;
    const k = v.kOf ? v.kOf(o) : role === 'pq' ? 'prismq' : key(o.simulator);
    mark(S('circle', {class: 'pq-dot ' + role + (o.status !== 'ok' ? ' hollow' : ''), cx: cx, cy: cy, r: rad}, g), o, k);
    obst.push({x0: cx - rad - 1, x1: cx + rad + 1, y0: cy - rad - 1, y1: cy + rad + 1});
    reqs.push({cx: cx, cy: cy, role: role, k: k, s: nmf(o), name: v.kOf ? NAME[k] || nmf(o) : nmf(o)});
  });
  L.cols.forEach(c => {
    const res = c.f(r);
    T(g, c.x, cy + 4, res.t, 'pq-col' + (res.miss ? ' miss' : ''), 'end');
  });
  const bot = Math.max(cy + (L.narrow ? 24 : 20), fy + 10);
  bg.setAttribute('y', y); bg.setAttribute('height', bot - y);
  if (!L.narrow) { lab.setAttribute('y', cy + 4.5); lob.y0 = cy - 8; lob.y1 = cy + 8; }
  return bot;
}
const LANES = [[-9, -19, -6], [18, 7, 21], [-22, -32, -19], [31, 20, 34]];
// Direct labels in the first free lane above or below each dot, one per series unless labelAll.
function placeLabels(gLab, reqs, obst, L, y0, labelAll) {
  reqs.sort((a, b) => (labelAll ? a.cy - b.cy : 0) || ROLE[a.role] - ROLE[b.role]);
  const done = {};
  reqs.forEach(q => {
    const id = labelAll ? q.cy + '|' + q.cx : q.role + '|' + q.k + '|' + q.s;
    if (done[id]) return;
    const w = tw(q.s, FS.label, q.role !== 'peer') + 2;
    for (let li = 0; li < LANES.length; li++) {
      const ln = LANES[li];
      for (const x0r of [q.cx - w / 2, q.cx - 4, q.cx - w + 4]) {
        const x0 = Math.max(L.ax0 - 8, Math.min(x0r, L.ax1 + 10 - w));
        const b = {x0: x0 - 2, x1: x0 + w + 2, y0: q.cy + ln[1], y1: q.cy + ln[2]};
        if (b.y0 < y0 - 4 || hit(b, obst)) continue;
        obst.push(b);
        done[id] = 1;
        if (li > 1) S('line', {class: 'pq-lead', x1: q.cx, y1: q.cy + (ln[0] < 0 ? -6 : 6),
          x2: Math.max(x0, Math.min(q.cx, x0 + w)), y2: q.cy + (ln[0] < 0 ? ln[2] : ln[1])}, gLab);
        nameText(gLab, x0, q.cy + ln[0], q.s, q.k, 'pq-dl ' + q.role);
        return;
      }
    }
  });
}
function dotPanel(fig, spec) {
  const plot = fig.querySelector('.pq-plot');
  let view = spec.def || 0;
  if (spec.views.length > 1) seg(plot, spec.views.map(v => v.label), view, i => { view = i; run(); }, spec.ctl || 'View');
  const lg = H('div', 'pq-legend', plot), box = H('div', 'pq-box', plot);
  const run = mount(plot, () => {
    const v = Object.assign({}, spec, spec.views[view]), L = dotLayout(boxW(box), spec, v), W = L.W;
    const val = v.val || (o => o.median_ms);
    const plotOk = o => o && o.median_ms != null && (o.status === 'ok' || v.plotInvalid);
    const rows = v.rows.map(r => {
      const mem = [r.pq].concat(r.peers || []).filter(Boolean);
      return {r: r, pts: mem.filter(plotOk), fails: mem.filter(o => !plotOk(o))};
    });
    const vals = [];
    rows.forEach(R => R.pts.forEach(o => vals.push(val(o, R.r))));
    const lo = v.domain ? v.domain[0] : Math.min.apply(null, vals) / 1.5, hi = v.domain ? v.domain[1] : Math.max.apply(null, vals) * 1.5;
    const X = v.scale === 'lin' ? x => L.ax0 + (x - lo) / (hi - lo) * (L.ax1 - L.ax0) : logScale([Math.log(lo), Math.log(hi)], L.ax0, L.ax1 - L.ax0);
    const svg = svgEl(W, 100, title(fig));
    const gGrid = S('g', {}, svg), gRows = S('g', {}, svg), gLab = S('g', {}, svg);
    const hl = Math.max(0, ...L.cols.map(c => [].concat(c.h).length));
    let edge = W - 2;
    L.cols.slice().reverse().forEach(c => {
      [].concat(c.h).forEach((h, i) => T(gGrid, edge, 11 + i * 12, h, 'pq-colh', 'end'));
      c.x = edge; edge -= c.w;
    });
    const obst = [{x0: L.ax1 + 12, x1: W + 50, y0: 0, y1: 99999}], reqs = [], y0 = 10 + 12 * hl;
    let y = y0;
    rows.forEach(R => { y = dotRow(gRows, R, v, L, X, y, obst, reqs); });
    placeLabels(gLab, reqs, obst, L, y0, v.labelAll);
    const yb = y + 4;
    if (v.band) S('rect', {class: 'pq-band', x: X(v.band[0]), width: X(v.band[1]) - X(v.band[0]), y: y0, height: yb - y0}, gGrid);
    xAxis(gGrid, ticks(lo, hi, v.scale !== 'lin', (L.ax1 - L.ax0) / 70), X, v.tick || ftick,
      {W: W, y: yb + 15, spans: [[y0, yb]], line: t => t === v.refLine ? 'pq-ref1' : 'pq-grid'});
    size(svg, W, yb + 22);
    box.replaceChildren(svg);
    const items = spec.legend === false ? [] : reqs.map(q => ({k: q.k, cls: q.role,
      name: q.role === 'ref' && L.cols.length ? q.name + ' (ratio column)' : q.name}));
    legend(lg, legendItems(items, rows.some(R => R.pts.some(o => o.status !== 'ok'))));
  });
}

// ---- line panel: small multiples; x categorical, linear or log
function linePanel(fig, spec) {
  const plot = fig.querySelector('.pq-plot');
  let view = 0;
  if (spec.views) seg(plot, spec.views.map(v => v.label), 0, i => { view = i; run(); }, spec.ctl || 'Measure');
  const lg = H('div', 'pq-legend', plot), grid = H('div', 'pq-mult', plot);
  grid.setAttribute('data-n', spec.mult.length);
  const cells = spec.mult.map(m => { const d = H('div', 'pq-cell', grid); H('p', 'pq-mt', d, m.title); return H('div', '', d); });
  const run = mount(plot, () => {
    const v = Object.assign({}, spec, spec.views ? spec.views[view] : {});
    let failed = false;
    spec.mult.forEach((m0, mi) => {
      const m = Object.assign({}, v, m0), box = cells[mi];
      if (lineCell(box, m, m0.title)) failed = true;
    });
    const items = [];
    if (spec.legend !== false) spec.mult.forEach(m => m.series.forEach(s => { if (s.pts.length > 1) items.push({k: s.k, cls: s.cls || 'peer', name: s.name}); }));
    legend(lg, legendItems(items, failed));
  });
}
// One small multiple; returns whether any drawn point failed its check.
function lineCell(box, m, label) {
  const W = box.clientWidth || 260, Hh = m.h || 196;
  const ser = m.series.map(s => ({s: s, pts: s.pts.map(p => ({x: p.x, o: p.o, y: m.y(p.o, s)})).filter(p => p.y != null)}))
    .filter(s => s.pts.length > 1);
  if (!ser.length) { box.replaceChildren(); return false; }
  const ys = [];
  ser.forEach(s => s.pts.forEach(p => ys.push(p.y)));
  let lo = Math.min.apply(null, ys), hi = Math.max.apply(null, ys);
  if (m.log) { lo /= 1.5; hi *= 1.5; } else { lo = 0; hi = Math.max(hi * 1.1, m.minHi || 2); }
  const tk = ticks(lo, hi, m.log, m.log ? 5 : Math.min(6, Math.ceil(hi)));
  const lw = Math.max(0, ...ser.map(s => tw(s.s.name, FS.label, s.s.cls !== 'peer')));
  const l = Math.max(...tk.map(t => tw(m.tick(t), FS.tick))) + 10, r = lw + 16, t = 10, b = 24, iw = W - l - r, ih = Hh - t - b;
  const xs = m.xs, xl = xs[0], xh = xs[xs.length - 1];
  const xi = m.xscale === 'cat' ? x => xs.indexOf(x) / (xs.length - 1) : m.xscale === 'log' ? x => Math.log(x / xl) / Math.log(xh / xl)
    : x => (x - xl) / (xh - xl);
  const X = x => l + xi(x) * iw;
  const Y = yv => t + ih - (m.log ? Math.log(yv / lo) / Math.log(hi / lo) : (yv - lo) / (hi - lo)) * ih;
  const svg = svgEl(W, Hh, label);
  tk.forEach(k => {
    S('line', {class: k === m.refLine ? 'pq-ref1' : 'pq-grid', x1: l, x2: l + iw, y1: Y(k), y2: Y(k)}, svg);
    T(svg, l - 7, Y(k) + 3.5, m.tick(k), 'pq-tick', 'end');
  });
  S('line', {class: 'pq-axis', x1: l, x2: l + iw, y1: t + ih, y2: t + ih}, svg);
  const xt = m.xticks || xs;
  xAxis(svg, xt, X, m.xfmt, {W: W, y: Hh - 7});
  if (m.xtitle) T(svg, X(xt[xt.length - 1]) + tw(m.xfmt(xt[xt.length - 1]), FS.tick) / 2 + 6, Hh - 7, m.xtitle, 'pq-tick');
  const ends = [];
  let failed = false;
  ser.sort((a, c) => (a.s.cls === 'pq') - (c.s.cls === 'pq')).forEach(s => {
    const cls = s.s.cls || 'peer';
    S('path', {class: 'pq-line ' + cls + (s.s.dash ? ' dash' : ''), 'data-k': s.s.k,
      d: s.pts.map((p, i) => (i ? 'L' : 'M') + X(p.x).toFixed(1) + ',' + Y(p.y).toFixed(1)).join('')}, svg);
    s.pts.forEach(p => {
      if (p.o.status !== 'ok') failed = true;
      mark(S('circle', {class: 'pq-pt ' + cls + (p.o.status !== 'ok' ? ' hollow' : ''), cx: X(p.x), cy: Y(p.y), r: cls === 'pq' ? 3.6 : 3}, svg),
        p.o, s.s.k);
    });
    const lp = s.pts[s.pts.length - 1];
    ends.push({y: Y(lp.y), y0: Y(lp.y), x: X(lp.x), s: s.s, cls: cls});
  });
  ends.sort((a, c) => a.y - c.y);
  for (let i = 1; i < ends.length; i++) ends[i].y = Math.max(ends[i].y, ends[i - 1].y + 13);
  const ov = ends.length ? ends[ends.length - 1].y - (t + ih + 4) : 0;
  if (ov > 0) {
    ends[ends.length - 1].y -= ov;
    for (let i = ends.length - 2; i >= 0; i--) ends[i].y = Math.min(ends[i].y, ends[i + 1].y - 13);
  }
  ends.forEach(e => {
    if (Math.abs(e.y - e.y0) > 2) S('path', {class: 'pq-lead', d: 'M' + (e.x + 5) + ',' + e.y0 + 'L' + (e.x + 11) + ',' + e.y}, svg);
    nameText(svg, e.x + 13, e.y + 3.5, e.s.name, e.s.k, 'pq-dl ' + e.cls);
  });
  box.replaceChildren(svg);
  return failed;
}

// ---- summary: one range bar per peer and method against the PRISM-Q 1x rule
function summary(fig) {
  const plot = fig.querySelector('.pq-plot');
  legend(H('div', 'pq-legend', plot), [{cls: 'bar', name: 'bar: lowest to highest ratio'}, {cls: 'tick', name: 'tick: median'}]);
  const box = H('div', 'pq-box', plot);
  const groups = SEC.map(sec => {
    const rows = secRows(sec);
    return {sec: sec, peers: sec.sims.slice(1).filter(peerOf).map(s => {
      const pts = rows.map(r => ({r: r, x: ratio(get(sec.g, r.w, s), r.pq)})).filter(p => p.x != null);
      const xs = pts.map(p => p.x);
      return {s: s, pts: pts, ref: s === sec.ref, lo: Math.min.apply(null, xs), hi: Math.max.apply(null, xs), med: xs.length && median(xs)};
    }).filter(p => p.pts.length)};
  }).filter(g => g.peers.length);
  const all = [];
  groups.forEach(g => g.peers.forEach(p => all.push(p)));
  mount(plot, () => {
    const W = boxW(box), narrow = W < NARROW;
    const labW = narrow ? Math.min(W * 0.4, Math.ceil(Math.max(...all.map(p => tw(NAME[p.s], FS.text, p.ref)))) + 10) : 118;
    const ax0 = labW + 8, P = W - 4 - ax0;
    const wr = Math.max.apply(null, all.map(p => tw(span([p.lo, p.hi]), FS.value, true))) + 12;
    const dom = fitLog(Math.min(1, ...all.map(p => p.lo)) / 1.12, Math.max(1, ...all.map(p => p.hi)), P, 6, wr), X = logScale(dom, ax0, P);
    const svg = svgEl(W, 100, title(fig));
    const gGrid = S('g', {}, svg), gRows = S('g', {}, svg);
    const y0 = 20, RH = narrow ? 24 : 26, SH = 26, spans = [];
    let y = y0;
    groups.forEach((g, gi) => {
      if (gi) S('line', {class: 'pq-sep', x1: 0, x2: W, y1: y + 6, y2: y + 6}, gRows);
      T(gRows, 0, y + SH - 7, g.sec.name, 'pq-subhead');
      y += SH;
      spans.push([y, y + g.peers.length * RH]);
      g.peers.forEach(p => {
        const cy = y + RH / 2, row = S('g', {class: 'pq-row pq-srow'}, gRows), cls = p.ref ? 'ref' : 'peer';
        rowBg(row, W, y, RH);
        nameText(row, narrow ? 4 : 10, cy + 4, NAME[p.s], p.s, cls);
        const x0 = X(p.lo), x1 = X(p.hi);
        S('line', {class: 'pq-beam', x1: X(1), x2: x0, y1: cy, y2: cy}, row);
        S('rect', {class: 'pq-rbar ' + cls, 'data-k': p.s, x: x0 - 1, y: cy - 4, width: Math.max(2, x1 - x0) + 2, height: 8, rx: 4}, row);
        S('line', {class: 'pq-medbg', x1: X(p.med), x2: X(p.med), y1: cy - 7.5, y2: cy + 7.5}, row);
        S('line', {class: 'pq-med', x1: X(p.med), x2: X(p.med), y1: cy - 7.5, y2: cy + 7.5}, row);
        p.pts.forEach(q => S('circle', {class: 'pq-spt', cx: X(q.x), cy: cy, r: 3.4}, row));
        T(row, x1 + 9, cy + 4, span([p.lo, p.hi]), 'pq-rlab');
        const rs = p.pts.slice().sort((a, b) => a.x - b.x);
        hot(row, () => '<div class="pq-th">' + esc(NAME[p.s]) + ' <span>' + esc(g.sec.name) + '</span></div><div class="pq-tm">peer time over PRISM-Q time, median ' +
          fx(p.med) + '</div>' + tgrid(rs.map(q => [q.r.label, fx(q.x)])),
          NAME[p.s] + ', ' + g.sec.name + ': ' + span([p.lo, p.hi]) + ' over ' + rs.length + ' rows, median ' + fx(p.med));
        row.__ax = x0 + (x1 - x0) / 2;
        y += RH;
      });
    });
    const yb = y + 6;
    xAxis(gGrid, ticks(Math.exp(dom[0]), Math.exp(dom[1]), true, Math.max(narrow ? 8 : 3, P / 60)), X, fxt,
      {W: W, y: yb + 15, spans: spans, one: true, line: t => t === 1 ? null : 'pq-grid'});
    spans.forEach(sp => S('line', {class: 'pq-rule', x1: X(1), x2: X(1), y1: sp[0] + 3, y2: sp[1] - 3}, gGrid));
    T(gGrid, X(1), 12, 'PRISM-Q = 1x', 'pq-rulelab', X(1) - 40 < 0 ? 'start' : 'middle');
    size(svg, W, yb + 22);
    box.replaceChildren(svg);
  });
}

// ---- explore: filter panel over every comparison row; bars diverge from the PRISM-Q rule
const EXPLORE_LIMIT = 6;
// Clearing the last selected chip of a row selects the whole row again.
const tog = (arr, x, all) => arr.indexOf(x) >= 0 ? (arr.length > 1 ? arr.filter(y => y !== x) : all.slice()) : arr.concat([x]);
function exploreState() {
  const st = {m: (SEC.find(s => secRows(s).length) || SEC[0]).id, f: null, q: null, s: null, v: 'time', w: 'bars', all: false, t: ''};
  if (location.hash.indexOf('#explore') !== 0) return st;
  const p = new URLSearchParams(location.hash.split('?')[1] || '');
  if (SECM[p.get('m')]) st.m = p.get('m');
  if (p.get('v') === 'mem') st.v = 'mem';
  if (p.get('w') === 'bars' || p.get('w') === 'matrix') st.w = p.get('w');
  ['f', 's'].forEach(k => { if (p.get(k)) st[k] = p.get(k).split(','); });
  if (p.get('q')) st.q = p.get('q').split(',').map(Number);
  return st;
}
// The rows of the chosen method and the options each chip row offers; selections outside them are dropped.
function exploreScope(st) {
  const sec = SECM[st.m], rows = secRows(sec);
  const sc = {sec: sec, rows: rows, f: uniq(rows.map(r => r.pq.family)), q: uniq(rows.map(r => r.pq.qubits)).sort((a, b) => a - b),
    s: sec.sims.slice(1).filter(s => peerOf(s) && rows.some(r => get(sec.g, r.w, s)))};
  const keep = (sel, opts) => { const k = (sel || []).filter(x => opts.indexOf(x) >= 0); return k.length ? k : opts.slice(); };
  ['f', 'q', 's'].forEach(k => { st[k] = keep(st[k], sc[k]); });
  return sc;
}
// Everything one draw of the matrix or bars needs, read from the state once.
function exploreModel(st, sc) {
  const mem = st.v === 'mem', amount = o => mem ? o.peak_rss_mb : o.median_ms;
  const cell = (o, pq) => {
    if (!o) return null;
    pq = mate(o, pq);
    if (o.status !== 'ok' || amount(o) == null || pq.status !== 'ok' || amount(pq) == null) return {o: o, fail: 1};
    return {o: o, r: mem ? amount(o) / amount(pq) : pairRatio(o, pq), soft: !mem && !(valid(o) && valid(pq))};
  };
  const words = st.t ? st.t.split(/\s+/) : [];
  const text = r => (r.label + ' ' + r.pq.family + ' ' + r.pq.qubits + 'q').toLowerCase();
  const shown = sc.rows.filter(r => st.f.indexOf(r.pq.family) >= 0 && st.q.indexOf(r.pq.qubits) >= 0 && words.every(w => text(r).indexOf(w) >= 0));
  return {sec: sc.sec, sims: st.s, amount: amount, fa: mem ? fmb : fms, shown: shown,
    label: 'Peer ' + (mem ? 'peak memory' : 'time') + ' as a multiple of PRISM-Q, ' + sc.sec.name,
    cells: r => st.s.map(s => ({s: s, c: cell(get(sc.sec.g, r.w, s), r.pq)}))};
}
function matrixView(box, M) {
  if (!M.shown.length) { box.replaceChildren(H('p', 'pq-empty', null, 'No rows match these filters.')); return; }
  const wrap = H('div', 'pq-mwrap'), tb = H('table', 'pq-matrix', wrap), hr = H('tr', '', H('thead', '', tb)), body = H('tbody', '', tb);
  tb.setAttribute('aria-label', M.label);
  ['Workload', 'PRISM-Q'].concat(M.sims.map(s => NAME[s])).forEach((h, i) => H('th', i === 1 ? 'pq' : '', hr, h).setAttribute('scope', 'col'));
  const rows = M.shown.map(r => ({r: r, cs: M.cells(r)}));
  let lo = 1, hi = 1;
  rows.forEach(g => g.cs.forEach(x => { if (x.c && !x.c.fail) { lo = Math.min(lo, x.c.r); hi = Math.max(hi, x.c.r); } }));
  const Q = logScale(fitLog(lo / 1.05, hi * 1.05, 100, 2, 2), 0, 100);
  rows.forEach(g => {
    const tr = H('tr', '', body), pq = g.r.pq, th = H('th', '', tr, g.r.label), pv = M.amount(pq);
    th.setAttribute('scope', 'row');
    if (pq.gates) H('small', '', th, pq.gates.toLocaleString('en-US') + ' gates');
    H('b', '', mark(H('td', 'pq-mcell pq', tr), pq, 'prismq'), pq.status === 'ok' && pv != null ? M.fa(pv) : statusWord(pq));
    g.cs.forEach(x => {
      const td = H('td', 'pq-mcell' + (x.s === M.sec.ref ? ' ref' : ''), tr), c = x.c;
      if (!c) { td.classList.add('none'); td.textContent = STAT.skipped; return; }
      mark(td, c.o, x.s);
      if (c.fail) {
        if (c.o.status === 'ok' && M.amount(c.o) != null) H('span', 'pq-mval', td, M.fa(M.amount(c.o)));
        else H('span', 'pq-mstub', td, statusWord(c.o));
        return;
      }
      if (c.soft) td.classList.add('soft');
      H('b', '', td, fx(c.r));
      H('span', '', td, M.fa(M.amount(c.o)));
      const bar = H('div', 'pq-mbar', td), f = H('i', '', bar);
      f.style.left = Math.min(Q(1), Q(c.r)) + '%';
      f.style.width = Math.max(1, Math.abs(Q(c.r) - Q(1))) + '%';
      H('u', '', bar).style.left = Q(1) + '%';
    });
  });
  box.replaceChildren(wrap);
}
function barsView(box, M, list) {
  const W = boxW(box), narrow = W < NARROW;
  const groups = list.map(r => ({r: r, cells: M.cells(r).filter(x => x.c)}));
  const labW = narrow ? 70 : 104, ax0 = labW + 6, P = W - 4 - ax0;
  const lab = c => fx(c.r) + ' · ' + M.fa(M.amount(c.o));
  let lo = 1, hi = 1, wl = 8, wr = 8;
  groups.forEach(g => g.cells.forEach(x => {
    const c = x.c;
    if (c.fail) { wr = Math.max(wr, tw(statusWord(c.o, true), FS.tick) + 22); return; }
    lo = Math.min(lo, c.r); hi = Math.max(hi, c.r);
    const w = tw(lab(c), FS.value, true) + 12;
    if (c.r < 1) wl = Math.max(wl, w); else wr = Math.max(wr, w);
  }));
  groups.forEach(g => { wr = Math.max(wr, tw(M.fa(M.amount(g.r.pq)), FS.value, true) + 14); });
  const dom = fitLog(lo / 1.05, hi * 1.05, P, wl, wr), X = logScale(dom, ax0, P);
  const svg = svgEl(W, 100, M.label);
  const gGrid = S('g', {}, svg), gRows = S('g', {}, svg);
  const y0 = 22, RH = 19, HH = 24, spans = [];
  let y = y0;
  if (!groups.length) T(gRows, 0, y + 20, 'No rows match these filters.', 'pq-sub');
  groups.forEach((g, gi) => {
    if (gi) S('line', {class: 'pq-sep', x1: 0, x2: W, y1: y + 2, y2: y + 2}, gRows);
    const pq = g.r.pq, ht = T(gRows, 0, y + HH - 6, g.r.label, 'pq-ghead');
    if (pq.gates && !narrow) S('tspan', {class: 'pq-gsub'}, ht).textContent = '  ' + pq.gates.toLocaleString('en-US') + ' gates';
    y += HH;
    const rx = X(1), top0 = y, pr = S('g', {class: 'pq-row'}, gRows), pv = M.amount(pq), pqok = pq.status === 'ok' && pv != null;
    rowBg(pr, W, y, RH);
    nameText(pr, 0, y + RH / 2 + 4, 'PRISM-Q', 'prismq', 'pq', null);
    const pd = mark(S('circle', {class: 'pq-dot pq', cx: rx, cy: y + RH / 2, r: 4}, pr), pq, 'prismq');
    T(pr, rx + 9, y + RH / 2 + 4, pqok ? M.fa(pv) : statusWord(pq), 'pq-blab pq');
    pd.setAttribute('aria-label', 'PRISM-Q, ' + g.r.label + ': ' + (pqok ? M.fa(pv) : statusWord(pq)));
    y += RH;
    g.cells.forEach(x => {
      const c = x.c, cy = y + RH / 2, row = S('g', {class: 'pq-row'}, gRows), role = roleOf(x.s, M.sec.ref);
      rowBg(row, W, y, RH);
      nameText(row, 0, cy + 4, NAME[x.s], x.s, role);
      if (c.fail) {
        const st = mark(stub(row, rx + 3, cy, statusWord(c.o, true)), c.o, x.s);
        st.setAttribute('aria-label', NAME[x.s] + ', ' + g.r.label + ': ' + statusWord(c.o) + (c.o.reason ? ', ' + c.o.reason : ''));
      } else {
        const xe = X(c.r), left = c.r < 1;
        const b = mark(S('rect', {class: 'pq-bar ' + role + (c.soft ? ' soft' : ''),
          x: Math.min(rx, xe), y: cy - 4.5, width: Math.max(1.5, Math.abs(xe - rx)), height: 9, rx: 2}, row), c.o, x.s);
        b.__ax = left ? 0 : Math.abs(xe - rx);
        const t = T(row, left ? xe - 7 : xe + 7, cy + 4, '', 'pq-blab', left ? 'end' : 'start');
        S('tspan', {class: 'pq-bx'}, t).textContent = fx(c.r);
        S('tspan', {}, t).textContent = ' · ' + M.fa(M.amount(c.o));
      }
      y += RH;
    });
    S('line', {class: 'pq-rule', x1: rx, x2: rx, y1: top0 + 2, y2: y - 2}, gRows);
    spans.push([top0, y]);
    y += 6;
  });
  if (groups.length) xAxis(gGrid, ticks(Math.exp(dom[0]), Math.exp(dom[1]), true, P / 60), X, fxt,
    {W: W, y: 12, spans: spans, one: true, line: t => t === 1 ? null : 'pq-grid'});
  size(svg, W, y + 4);
  box.replaceChildren(svg);
}
function explore(fig) {
  const plot = fig.querySelector('.pq-plot'), st = exploreState(), fromHash = location.hash.indexOf('#explore') === 0;
  const cmd = H('div', 'pq-cmd', plot), find = H('input', '', cmd), count = H('span', 'pq-count', cmd), cp = H('button', 'pq-copy', cmd, 'Copy link');
  find.type = 'search';
  find.placeholder = 'Filter rows: qft, 24q, depth';
  find.setAttribute('aria-label', 'Filter rows by workload text');
  find.setAttribute('autocomplete', 'off');
  find.setAttribute('spellcheck', 'false');
  find.oninput = () => { st.t = find.value.trim().toLowerCase(); st.all = false; run(); };
  find.onkeydown = e => { e.stopPropagation(); if (e.key === 'Escape' && find.value) { find.value = ''; find.oninput(); } };
  count.setAttribute('role', 'status');
  count.setAttribute('aria-live', 'polite');
  cp.type = 'button';
  const disc = H('details', 'pq-fdisc', plot), sum = H('summary', '', disc, 'Filters '), sumN = H('span', 'pq-fsum', sum);
  const ctl = H('div', 'pq-filters', disc), box = H('div', 'pq-box', plot), more = H('button', 'pq-more', plot);
  disc.open = boxW(plot) >= NARROW;
  more.type = 'button';
  more.onclick = () => { st.all = !st.all; run(); };
  let sc = exploreScope(st);
  const enc = () => {
    const p = ['m=' + st.m];
    ['f', 'q', 's'].forEach(k => { if (st[k].length < sc[k].length) p.push(k + '=' + st[k].join(',')); });
    if (st.v === 'mem') p.push('v=mem');
    if (st.w !== 'bars') p.push('w=' + st.w);
    return '#explore?' + p.join('&');
  };
  const sync = () => { try { history.replaceState(null, '', enc()); } catch (e) { /* file:// in some browsers */ } };
  const changed = () => { st.all = false; run(); sync(); };
  const frow = label => { const r = H('div', 'pq-frow', ctl); H('span', 'pq-flab', r, label); return r; };
  seg(frow('Method'), SEC.map(s => s.chip), SEC.findIndex(s => s.id === st.m), i => {
    st.m = SEC[i].id; st.f = st.q = st.s = null;
    sc = exploreScope(st);
    chips.forEach(fillChips);
    changed();
  }, 'Method');
  const chips = [['Family', 'f', String, false], ['Qubits', 'q', x => x.toLocaleString('en-US') + 'q', false], ['Simulators', 's', x => NAME[x], true]]
    .map(c => {
      const row = frow(c[0]), g = H('div', 'pq-chips', row);
      g.setAttribute('role', 'group');
      g.setAttribute('aria-label', c[0]);
      return {k: c[1], fmt: c[2], always: c[3], row: row, g: g};
    });
  const pressChips = c => Array.prototype.forEach.call(c.g.children, b => b.setAttribute('aria-pressed', st[c.k].indexOf(b.__v) >= 0));
  function fillChips(c) {
    c.g.replaceChildren();
    c.row.hidden = !c.always && sc[c.k].length < 2;
    sc[c.k].forEach(it => {
      const b = H('button', 'pq-chip', c.g, c.fmt(it));
      b.type = 'button';
      b.__v = it;
      if (c.k === 'q' && sc.q.length > 1) b.setAttribute('data-ord', Math.round(sc.q.indexOf(it) * 5 / (sc.q.length - 1)));
      b.onclick = () => { st[c.k] = tog(st[c.k], it, sc[c.k]); pressChips(c); changed(); };
    });
    pressChips(c);
  }
  chips.forEach(fillChips);
  const VIEWS = ['bars', 'matrix'];
  seg(frow('View'), ['Bars', 'Matrix'], VIEWS.indexOf(st.w), i => { st.w = VIEWS[i]; changed(); }, 'View');
  const mr = frow('Metric');
  seg(mr, ['Time', 'Peak memory'], st.v === 'mem' ? 1 : 0, i => { st.v = i ? 'mem' : 'time'; changed(); }, 'Metric');
  cp.onclick = () => {
    sync();
    const url = location.href.split('#')[0] + enc();
    const done = () => { cp.textContent = 'Link copied'; setTimeout(() => { cp.textContent = 'Copy link'; }, 1600); };
    const legacy = () => {
      const t = H('textarea', '', document.body, url);
      t.style.position = 'fixed'; t.style.opacity = '0';
      t.select();
      try { document.execCommand('copy'); done(); } catch (e) { cp.textContent = 'Copy failed'; }
      t.remove();
    };
    if (navigator.clipboard && window.isSecureContext) navigator.clipboard.writeText(url).then(done, legacy); else legacy();
  };
  const run = mount(plot, () => {
    const M = exploreModel(st, sc), wide = boxW(plot) >= NARROW;
    const narrowed = ['f', 'q', 's'].filter(k => st[k].length < sc[k].length).length;
    count.textContent = M.shown.length + ' of ' + sc.rows.length + ' rows · ' + plural(st.s.length, 'simulator');
    sumN.textContent = sc.sec.chip + (narrowed ? ' · ' + narrowed + ' active' : '');
    disc.classList.toggle('pq-wide', wide);
    if (wide) disc.open = true;
    if (st.w === 'matrix') { more.hidden = true; matrixView(box, M); return; }
    more.hidden = M.shown.length <= EXPLORE_LIMIT;
    more.textContent = st.all ? 'Show fewer rows' : 'Show all ' + M.shown.length + ' rows';
    more.setAttribute('aria-expanded', st.all);
    barsView(box, M, st.all ? M.shown : M.shown.slice(0, EXPLORE_LIMIT));
  });
  if (fromHash) setTimeout(() => fig.scrollIntoView({block: 'start'}), 60);
}

// ---- memory: bars from zero per simulator, with the bare fp64 state as a reference line
function memPanel(fig) {
  const plot = fig.querySelector('.pq-plot'), sims = ['prismq', 'aer-statevector', 'quest', 'aer-statevector-f32', 'qsim'];
  const rows = secRows(SECM.sv64).filter(r => r.pq.family === 'rcs' && r.pq.depth === 10);
  let view = rows.length - 1;
  seg(plot, rows.map(r => r.pq.qubits + 'q'), view, i => { view = i; run(); }, 'Qubits');
  const lg = H('div', 'pq-legend', plot), box = H('div', 'pq-box', plot);
  const run = mount(plot, () => {
    const r = rows[view], n = r.pq.qubits, state = 16 * Math.pow(2, n) / 1048576;
    const W = boxW(box), labW = W < NARROW ? 70 : 104, ax0 = labW + 6;
    const items = sims.map(s => ({s: s, o: get('headline', r.w, s), role: roleOf(s, 'aer-statevector')}));
    const mb = it => it.o && it.o.status === 'ok' ? it.o.peak_rss_mb : null;
    const hi = Math.max(state, ...items.map(mb).filter(v => v != null)) * 1.04;
    const wr = Math.max(...items.map(it => mb(it) != null ? tw(fmb(mb(it)), FS.value, true) : 0)) + 12, P = W - 4 - ax0 - wr;
    const X = v => ax0 + v / hi * P;
    const svg = svgEl(W, 100, title(fig)), gGrid = S('g', {}, svg), gRows = S('g', {}, svg);
    const y0 = 30, RH = 24;
    let y = y0;
    items.forEach(it => {
      const cy = y + RH / 2, row = S('g', {class: 'pq-row'}, gRows), v = mb(it);
      rowBg(row, W, y, RH);
      nameText(row, 0, cy + 4, NAME[it.s], it.s, it.role);
      if (v != null) {
        mark(S('rect', {class: 'pq-bar ' + it.role, x: ax0, y: cy - 5, width: Math.max(1.5, X(v) - ax0), height: 10, rx: 2}, row), it.o, it.s);
        T(row, X(v) + 7, cy + 4, fmb(v), 'pq-blab pq-bx' + (it.role === 'pq' ? ' pq' : ''));
      } else {
        const st = stub(row, ax0 + 2, cy, it.o ? statusWord(it.o, true) : STAT.skipped);
        if (it.o) mark(st, it.o, it.s);
        else hot(st, '<div class="pq-th">' + esc(NAME[it.s]) + '</div><div class="pq-ts">no run at ' + n + ' qubits</div>', NAME[it.s] + ': no run at ' + n + ' qubits');
      }
      y += RH;
    });
    const yb = y + 2, u = hi >= 1024 ? 1024 : 1, un = u > 1 ? ' GB' : ' MB';
    xAxis(gGrid, ticks(0, hi / u, false, P / 90), t => X(t * u), t => +t.toPrecision(3) + un,
      {W: W, y: yb + 15, spans: [[y0, yb]], line: t => t ? 'pq-grid' : 'pq-axis'});
    const sx = X(state), sl = 'fp64 state ' + fmb(state), sw = tw(sl, FS.label);
    S('line', {class: 'pq-stateline', x1: sx, x2: sx, y1: y0 - 6, y2: yb}, gGrid);
    T(gGrid, Math.min(Math.max(sx, ax0 + sw / 2), W - sw / 2), y0 - 12, sl, 'pq-statelab', 'middle');
    size(svg, W, yb + 22);
    box.replaceChildren(svg);
    legend(lg, items.filter(mb).map(it => ({k: it.s, cls: it.role, name: NAME[it.s]})));
  });
}

// ---- panels
const views = (rows, keyOf, labels) => [{label: 'All', rows: rows}].concat(labels.map(l => ({label: l[1], rows: rows.filter(r => keyOf(r.pq) === l[0])})));
const ratioCol = (h, ref) => ({h: h, w: 62, f: r => {
  const o = r.peers.find(p => p.simulator === ref), x = ratio(o, r.pq);
  return x != null ? {t: fx(x)} : {t: o ? (o.status !== 'ok' ? statusWord(o, true) : 'unchecked') : STAT.skipped, miss: 1};
}});
// Series per prefix over the bond caps, for the sweep and bond-cap charts.
const capSeries = (defs, caps, find) => defs.map(s => ({k: s[0], name: s[1], cls: s[2],
  pts: caps.map(c => ({x: c, o: find(s[3] + c)})).filter(p => p.o)}));
const PANELS = {
  summary: summary,
  explore: explore,
  sweep: fig => {
    const sw = O.filter(o => o.group === 'bondsweep'), caps = uniq(sw.filter(o => o.bond_cap).map(o => o.bond_cap)).sort((a, b) => a - b);
    const ns = uniq(sw.map(o => o.qubits)).sort((a, b) => a - b);
    const defs = [['aer-mps', 'Aer MPS', 'ref', 'aer-mps:'], ['quimb-mps', 'quimb MPS', 'peer', 'quimb-mps:'], ['prismq', 'PRISM-Q MPS', 'pq', 'prismq:mps-']];
    const ser = n => capSeries(defs, caps, s => sw.find(o => o.qubits === n && o.simulator === s));
    const err = o => o.max_abs_error == null ? null : Math.max(o.max_abs_error, 1e-16);
    linePanel(fig, {xs: caps, xscale: 'cat', xfmt: c => 'cap ' + c, log: true, mult: [].concat.apply([], ns.map(n => [
      {title: 'Largest error, ' + n + ' qubits', series: ser(n), y: err, tick: ferr},
      {title: 'Median time, ' + n + ' qubits', series: ser(n), y: o => o.median_ms, tick: ftick}]))});
  },
  bondcap: fig => {
    const acc = O.filter(o => o.group === 'accuracy'), caps = uniq(acc.filter(o => o.bond_cap).map(o => o.bond_cap)).sort((a, b) => a - b);
    const ser = capSeries([['aer-mps', 'Aer MPS', 'ref', 'aer-mps:'], ['prismq', 'PRISM-Q MPS', 'pq', 'prismq:mps-']], caps,
      s => acc.find(o => o.simulator === s));
    linePanel(fig, {xs: caps, xscale: 'cat', xfmt: c => 'cap ' + c, log: true, mult: [
      {title: 'Median time', series: ser, y: o => o.median_ms, tick: ftick},
      {title: 'Distance from the exact output', series: ser, y: o => o.tvd, tick: ferr}]});
  },
  threads: fig => {
    const th = O.filter(o => o.group === 'threads'), ws = uniq(th.map(o => o.workload)).sort((a, b) => order(th.find(o => o.workload === a), th.find(o => o.workload === b)));
    const find = (w, s, t) => th.find(o => o.workload === w && o.simulator === s && o.threads === t);
    const mult = ws.map(w => ({title: th.find(o => o.workload === w).label, series: ['quest', 'qsim', 'aer-statevector', 'prismq'].map(s => ({k: s, name: NAME[s],
      cls: s === 'prismq' ? 'pq' : 'peer', pts: [1, 2, 4, 8].map(t => ({x: t, o: find(w, s, t)})).filter(p => ok(p.o))}))}));
    linePanel(fig, {mult: mult, xs: [1, 2, 4, 8], xscale: 'cat', xfmt: String, xtitle: 'threads', h: 200, views: [
      {label: 'Median time', y: o => o.median_ms, log: true, tick: ftick},
      {label: 'Speedup over 1 thread', y: (o, s) => { const p = s.pts[0].o; return p.threads === 1 ? p.median_ms / o.median_ms : null; },
        log: false, tick: x => x + 'x', refLine: 1, minHi: 4}]});
  },
  dispatch: fig => {
    const dp = O.filter(o => o.group === 'dispatch'), ws = uniq(dp.map(o => o.workload));
    const rows = ws.map(w => {
      const pq = get('dispatch', w, 'prismq'), fs = dp.filter(o => o.workload === w && o.simulator.indexOf('prismq:') === 0);
      const best = fs.filter(usable).sort((a, b) => a.median_ms - b.median_ms)[0];
      return {label: pq.label, pq: pq, peers: fs, ref: best.simulator, ratio: pq.median_ms / best.median_ms,
        sub: 'auto ' + pq.resolved_backend + ', fastest ' + best.simulator.slice(7)};
    }).sort((a, b) => a.ratio - b.ratio);
    const strip = rows.map(r => ({label: r.label, pq: r.pq, peers: [], ratio: r.ratio, sub: r.sub, ref: r.ref}));
    dotPanel(fig, {ctl: 'View', legend: false, name: o => o.simulator === 'prismq' ? 'auto' : o.simulator.slice(7),
      kOf: o => o.simulator, cols: [{h: ['auto /', 'fastest'], w: 60, f: r => ({t: fx(r.ratio)})}],
      views: [{label: 'Against the fastest', rows: strip, scale: 'lin', domain: [0.9, 1.1], band: [0.95, 1.05], refLine: 1,
        val: (o, r) => r.ratio, tick: x => x.toFixed(2) + 'x'}, {label: 'All backends', rows: rows, labelAll: true}]});
  },
  memory: memPanel,
  suite: fig => {
    const R = D.suite.rows, so = r => ({name: 'PRISM-Q', simulator: 'prismq', version: D.suite.version, label: r.family + ' ' + r.qubits + 'q',
      median_ms: r.ms, threads: D.suite.rayon_threads, method: 'automatic', status: 'ok', valid: true, validated: true});
    const fam = f => R.filter(r => r.family === f).sort((a, b) => a.qubits - b.qubits).map(r => ({x: r.qubits, o: so(r)}));
    const gq = fam('ghz'), small = gq.filter(p => p.x <= 28), wide = gq.filter(p => p.x > 28);
    const dense = ['qft', 'hea', 'qv', 'ghz'].map((f, i) => ({k: 'prismq', name: f.toUpperCase(), cls: 'pq', dash: i % 2 === 1,
      pts: f === 'ghz' ? small : fam(f)}));
    linePanel(fig, {log: true, legend: false, y: o => o.median_ms, tick: ftick, mult: [
      {title: 'QFT, HEA, QV, GHZ · 16 to 28 qubits', series: dense, xs: [16, 28], xticks: [16, 20, 24, 28], xscale: 'lin', xfmt: x => x + 'q'},
      {title: 'GHZ · ' + wide[0].x + ' to ' + wide[wide.length - 1].x.toLocaleString('en-US') + ' qubits',
        series: [{k: 'prismq', name: 'GHZ', cls: 'pq', pts: wide}], xs: [wide[0].x, wide[wide.length - 1].x],
        xticks: wide.map(p => p.x), xscale: 'log', xfmt: x => x.toLocaleString('en-US') + 'q'}]});
  }
};
SEC.forEach(sec => {
  PANELS[sec.id] = fig => {
    const rows = secRows(sec), sp = sec.split, qb = sp && sp[0] === 'qubits';
    const spec = {ref: sec.ref, cols: [ratioCol([NAME[key(sec.ref)] + ' /', sec.base || 'PRISM-Q'], sec.ref)],
      views: sp ? views(rows, o => o[sp[0]], sp[1].map(x => [x, qb ? x + 'q' : 'depth ' + x])) : [{rows: rows}], ctl: sp && (qb ? 'Qubits' : 'Depth')};
    if (sec.byThreads) Object.assign(spec, {name: o => o.simulator === 'prismq' ? 'PRISM-Q ' + o.threads + 't' : nm(o), kOf: o => key(o.simulator)});
    dotPanel(fig, spec);
  };
});

// Each section tile's range on one log scale shared by every tile, with the 1x tick.
function tiles() {
  const bs = Array.prototype.slice.call(document.querySelectorAll('.pq-kpi-b'));
  if (!bs.length) return;
  const v = bs.map(b => [+b.getAttribute('data-lo'), +b.getAttribute('data-hi')]);
  const Q = logScale([Math.log(Math.min(1, ...v.map(x => x[0])) / 1.25), Math.log(Math.max(1, ...v.map(x => x[1])) * 1.25)], 0, 100);
  const list = document.querySelector('ul.pq-kpis');
  if (list) {
    const ax = H('p', 'pq-kpis-ax');
    H('span', '', ax, 'Section');
    const sc = H('span', 'pq-kpis-sc', ax);
    sc.setAttribute('aria-hidden', 'true');
    [1, 10, 100].forEach(t => { H('i', t === 1 ? 'one' : '', sc, t === 1 ? 'PRISM-Q = 1x' : t + 'x').style.left = Q(t) + '%'; });
    H('span', 'pq-kpis-un', ax, 'peer time over PRISM-Q time');
    list.before(ax);
    bs.forEach(b => { b.style.setProperty('--pq-x10', Q(10) + '%'); b.style.setProperty('--pq-x100', Q(100) + '%'); });
  }
  bs.forEach((b, i) => {
    const r = H('i', '', b);
    r.style.left = Q(v[i][0]) + '%';
    r.style.width = Math.max(1.5, Q(v[i][1]) - Q(v[i][0])) + '%';
    H('b', '', b).style.left = Q(1) + '%';
  });
}

function init() {
  FF = getComputedStyle(document.querySelector('.pq-figure') || document.body).fontFamily || 'sans-serif';
  tiles();
  FIGS.forEach(fig => {
    const f = PANELS[fig.getAttribute('data-chart')];
    if (!f || !fig.querySelector('.pq-plot')) return;
    try { f(fig); } catch (e) { if (window.console) console.error('perf chart ' + fig.getAttribute('data-chart'), e); }
  });
  let lastW = 0, rt;
  const redraw = () => { hide(); draws.forEach(d => d()); };
  const host = document.querySelector('.content main') || document.body;
  const onSize = w => { if (Math.abs(w - lastW) < 1) return; lastW = w; clearTimeout(rt); rt = setTimeout(redraw, 90); };
  if (window.ResizeObserver) new ResizeObserver(es => onSize(es[0].contentRect.width)).observe(host);
  else addEventListener('resize', () => onSize(innerWidth));
  lastW = host.clientWidth;
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => {
    FF = getComputedStyle(document.querySelector('.pq-figure') || document.body).fontFamily || FF;
    redraw();
  });
}
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
