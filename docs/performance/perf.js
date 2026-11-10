// Charts for the Performance page. Reads window.PRISM_PERF (data.js) and draws into each
// .pq-figure .pq-plot slot; titles, notes and exact-number tables stay static in index.md.
(function () {
'use strict';
const D = window.PRISM_PERF;
const FIGS = Array.prototype.slice.call(document.querySelectorAll('.pq-figure[data-chart]'));
if (!D || !FIGS.length) return;
document.documentElement.classList.add('pq-perf');

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

// Formats mirror comparison/harness/pages.py so chart labels agree with the static tables.
const fms = v => v == null ? 'n/a' : v < 1 ? Math.round(v * 1000) + ' us' : v >= 1000 ? (v / 1000).toFixed(2) + ' s'
  : v < 10 ? v.toFixed(2) + ' ms' : v < 100 ? v.toFixed(1) + ' ms' : Math.round(v) + ' ms';
const fx = v => (v >= 100 ? Math.round(v) : v >= 10 ? v.toFixed(1) : v.toFixed(2)) + 'x';
const fmb = v => v == null ? 'n/a' : v >= 1024 ? (v / 1024).toFixed(2) + ' GB' : Math.round(v) + ' MB';
const fdist = t => t == null ? 'n/a' : t >= 0.01 ? String(+t.toPrecision(2)) : t.toExponential(1);
const span = rs => { const a = fx(Math.min.apply(null, rs)), b = fx(Math.max.apply(null, rs)); return a === b ? a : a + ' to ' + b; };
const ftick = v => v < 1 ? +(v * 1000).toPrecision(3) + ' us' : v < 1000 ? +v.toPrecision(3) + ' ms' : +(v / 1000).toPrecision(3) + ' s';
const fxt = v => +v.toPrecision(3) + 'x';
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
const median = a => { const s = a.slice().sort((x, y) => x - y), m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };
const uniq = a => a.filter((x, i) => a.indexOf(x) === i);

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
const ratio = (p, q) => { q = p && mate(p, q); return valid(p) && valid(q) ? p.median_ms / q.median_ms : null; };
const peerOf = s => s.indexOf('prismq') !== 0;

const SEC = [
  {id: 'sv64', name: 'Statevector, fp64', chip: 'Statevector fp64', g: 'headline', fam: ['rcs', 'qft', 'qv'],
    sims: ['prismq', 'aer-statevector', 'quest', 'spinoza', 'qip'], ref: 'aer-statevector'},
  {id: 'sv32', name: 'Statevector against fp32 peers', chip: 'fp32 peers', g: 'headline', fam: ['rcs', 'qft', 'qv'],
    sims: ['prismq', 'aer-statevector-f32', 'qsim'], ref: 'qsim'},
  {id: 'stabilizer', name: 'Stabilizer', chip: 'Stabilizer', g: 'headline', fam: ['clifford'], sims: ['prismq', 'prismq@1', 'stim'],
    ref: 'stim'},
  {id: 'highshot', name: 'High-shot stabilizer', chip: 'High-shot', g: 'highshot', fam: ['clifford'],
    sims: ['prismq', 'prismq@1', 'stim'], ref: 'stim'},
  {id: 'mps', name: 'Matrix product state', chip: 'MPS', g: 'headline', fam: ['lowent'],
    sims: ['prismq', 'aer-mps', 'aer-automatic'], ref: 'aer-mps'},
  {id: 'mpsdeep', name: 'Deep MPS', chip: 'Deep MPS', g: 'mpsdeep', fam: ['lowent'],
    sims: ['prismq:mps-256', 'aer-mps:256', 'quimb-mps:256'], ref: 'aer-mps:256'},
  {id: 'tn', name: 'Tensor network expectation values', chip: 'TN expval', g: 'tn', fam: ['rcs'],
    sims: ['prismq', 'prismq:tensor-network', 'prismq:mps-256', 'quimb-tn', 'aer-mps'], ref: 'quimb-tn'},
  {id: 'sampling', name: 'Shot sampling', chip: 'Sampling', g: 'sampling', fam: ['sampling'],
    sims: ['prismq', 'aer-statevector', 'aer-statevector-f32', 'qsim'], ref: 'aer-statevector'},
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
function S(tag, at, par) {
  const e = document.createElementNS(NS, tag);
  for (const a in at) e.setAttribute(a, at[a]);
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
  return S('svg', {class: 'pq-svg', role: 'img', 'aria-label': label, width: W, height: Hh, viewBox: '0 0 ' + W + ' ' + Hh});
}
function size(svg, W, Hh) {
  svg.setAttribute('width', W);
  svg.setAttribute('height', Hh);
  svg.setAttribute('viewBox', '0 0 ' + W + ' ' + Hh);
}
function hot(el, tip, label) {
  el.__tip = tip;
  el.classList.add('pq-hot');
  el.setAttribute('tabindex', '0');
  if (label) el.setAttribute('aria-label', label);
  return el;
}
function mark(el, o, k) {
  el.__o = o;
  el.classList.add('pq-hot');
  el.setAttribute('data-k', k || key(o.simulator));
  el.setAttribute('tabindex', '0');
  el.setAttribute('aria-label', nm(o) + ', ' + o.label + ': ' + (o.median_ms != null ? fms(o.median_ms) : STAT[o.status] || o.status));
  return el;
}
function nameText(par, x, y, s, k, cls, anchor) {
  const t = T(par, x, y, s, 'pq-name ' + (cls || ''), anchor);
  t.setAttribute('data-k', k);
  return t;
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
let pin = null, hov = null, cur = null;
function highlight(k) {
  document.querySelectorAll('.pq-plot [data-k]').forEach(e => {
    const m = e.getAttribute('data-k') === k || key(e.getAttribute('data-k')) === k && k === 'prismq';
    e.classList.toggle('pq-dim', !!k && !m);
    e.classList.toggle('pq-hl', !!k && m);
  });
  document.documentElement.classList.toggle('pq-pinned', !!pin);
}
function tipHTML(o) {
  const L = ['<b>' + esc(nm(o)) + '</b> <span>' + esc(ver(o)) + '</span>'];
  const meta = [esc(o.label), o.threads + ' thread' + (o.threads > 1 ? 's' : '')];
  if (o.precision) meta.push(esc(o.precision));
  if (o.resolved_backend) meta.push(esc(o.resolved_backend));
  else if (o.method) meta.push(esc(o.method));
  L.push(meta.join(' · '));
  if (o.median_ms != null) L.push('median <b>' + fms(o.median_ms) + '</b>' +
    (o.p25_ms != null ? ' · p25 to p75 ' + fms(o.p25_ms) + ' to ' + fms(o.p75_ms) : '') + (o.n ? ' · n ' + o.n : ''));
  if (o.first_ms != null) L.push('cold call ' + fms(o.first_ms) + (o.min_ms != null ? ' · fastest warm ' + fms(o.min_ms) : ''));
  if (o.internal_ms != null) L.push('time Aer reports ' + fms(o.internal_ms));
  if (o.peak_rss_mb) L.push('peak memory ' + fmb(o.peak_rss_mb));
  const pq = peerOf(o.simulator) && mate(o, get(o.group, o.workload, baseOf(o.group))), r = pq && ratio(o, pq);
  if (r) L.push('<b>' + fx(r) + '</b> the ' + (NAME[pq.simulator] || pq.name) + ' median at ' + pq.threads + ' thread' +
    (pq.threads > 1 ? 's' : '') + ' (' + fms(pq.median_ms) + ')');
  if (o.max_abs_error != null) L.push('largest error against exact ' + fdist(o.max_abs_error));
  if (o.tvd != null) L.push('distance from reference ' + fdist(o.tvd));
  if (o.status !== 'ok') L.push('<i>' + esc(STAT[o.status] || o.status) + (o.reason ? ': ' + esc(o.reason) : '') + '</i>');
  else if (o.role === 'too-few-shots') L.push('<i>under 100 shots: checked on deterministic bits only, no ratio</i>');
  else if (o.role === 'reference') L.push('<i>reference output' + (o.cross_checked ? ', cross-checked' : ', no cross-check') + '</i>');
  else if (o.role === 'weakly-tested') L.push('<i>weakly tested: detects shifts above ' + fdist(o.min_detectable) + ' in a marginal</i>');
  else if (!o.validated) L.push('<i>timed, output not checked</i>');
  if (o.min_detectable != null && o.role !== 'weakly-tested') L.push('check detects shifts above ' + fdist(o.min_detectable));
  return L.join('<br>');
}
function show(m) {
  cur = m;
  tip.innerHTML = m.__o ? tipHTML(m.__o) : typeof m.__tip === 'function' ? m.__tip() : m.__tip;
  tip.classList.add('on');
  m.setAttribute('aria-describedby', 'pq-tip');
  place(m);
  const row = m.closest('.pq-row');
  if (row) row.classList.add('on');
}
function place(m) {
  const b = m.getBoundingClientRect(), tb = tip.getBoundingClientRect(), ax = m.__ax != null ? b.left + m.__ax : b.left + b.width / 2;
  let x = ax - tb.width / 2, y = b.top - tb.height - 10;
  if (y < 4) y = b.bottom + 10;
  tip.style.left = Math.max(6, Math.min(x, innerWidth - tb.width - 6)) + 'px';
  tip.style.top = y + 'px';
}
function hide() {
  tip.classList.remove('on');
  if (cur) cur.removeAttribute('aria-describedby');
  cur = null;
  document.querySelectorAll('.pq-row.on').forEach(e => e.classList.remove('on'));
}
function over(e) {
  if (!e.target.closest) return;
  const t = e.target.closest('.pq-plot .pq-hot');
  if (t && t !== cur) { hide(); show(t); }
  const n = e.target.closest('.pq-plot [data-k]');
  const k = n && (n.classList.contains('pq-name') || e.type === 'focusin') ? n.getAttribute('data-k') : null;
  if (k !== hov) { hov = k; highlight(hov || pin); }
}
function out(e) {
  if (!e.target.closest) return;
  const t = e.target.closest('.pq-plot .pq-hot'), to = e.relatedTarget;
  if (t && !(to && t.contains(to))) hide();
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
document.addEventListener('click', e => {
  const t = e.target.closest && e.target.closest('.pq-plot [data-k]');
  if (t) toggle(t);
});
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') { pin = null; hov = null; hide(); highlight(null); }
  const t = e.target.closest && e.target.closest('.pq-plot [data-k]');
  if ((e.key === 'Enter' || e.key === ' ') && t && t.tagName !== 'BUTTON') { e.preventDefault(); toggle(t); }
});
addEventListener('scroll', () => { if (cur) { if (cur.isConnected) place(cur); else hide(); } }, {passive: true});

// ---- dot panel: one row per workload, a log value axis, direct labels placed without overlap
function dotPanel(fig, spec) {
  const plot = fig.querySelector('.pq-plot');
  let cur = spec.def || 0;
  if (spec.views.length > 1) seg(plot, spec.views.map(v => v.label), cur, i => { cur = i; draw(); }, spec.ctl || 'View');
  const box = H('div', 'pq-box', plot);
  function draw() {
    const v = Object.assign({}, spec, spec.views[cur]);
    const W = box.clientWidth || 640, narrow = W < 560, SM = 11;
    const cols = (v.cols || []).filter(c => !(narrow && c.wide));
    const colW = cols.reduce((a, c) => a + c.w, 0);
    let maxL = 0;
    spec.views.forEach(vw => vw.rows.forEach(r => { maxL = Math.max(maxL, tw(r.short || r.label, 12.5)); }));
    const labW = narrow ? 0 : Math.round(Math.max(96, Math.min(maxL + 10, W * 0.3)));
    const ax0 = narrow ? 10 : labW + 16, ax1 = W - colW - (cols.length ? 20 : 10);
    const val = v.val || (o => o.median_ms), nmf = v.name || nm;
    const plotOk = o => o && o.median_ms != null && (o.status === 'ok' || v.plotInvalid);
    const rows = v.rows.map(r => {
      const mem = [r.pq].concat(r.peers || []).filter(Boolean);
      return {r: r, pts: mem.filter(plotOk), fails: mem.filter(o => !plotOk(o))};
    });
    const vals = [];
    rows.forEach(R => R.pts.forEach(o => vals.push(val(o, R.r))));
    let lo, hi;
    if (v.domain) { lo = v.domain[0]; hi = v.domain[1]; } else { lo = Math.min.apply(null, vals) / 1.5; hi = Math.max.apply(null, vals) * 1.5; }
    const X = v.scale === 'lin' ? x => ax0 + (x - lo) / (hi - lo) * (ax1 - ax0)
      : x => ax0 + Math.log(x / lo) / Math.log(hi / lo) * (ax1 - ax0);
    const svg = svgEl(W, 100, title(fig));
    const gGrid = S('g', {}, svg), gRows = S('g', {}, svg), gLab = S('g', {}, svg);
    const hl = Math.max(0, ...cols.map(c => [].concat(c.h).length));
    let edge = W - 2;
    cols.slice().reverse().forEach(c => {
      [].concat(c.h).forEach((h, i) => T(gGrid, edge, 11 + i * 12, h, 'pq-colh', 'end'));
      c.x = edge; edge -= c.w;
    });
    const obst = [{x0: ax1 + 12, x1: W + 50, y0: 0, y1: 99999}];
    const reqs = [];
    const y0 = 10 + 12 * hl;
    let y = y0;
    rows.forEach(R => {
      const g = S('g', {class: 'pq-row'}, gRows), rt = y, bg = S('rect', {class: 'pq-rowbg', x: -6, width: W + 12, rx: 8}, g);
      const r = R.r, refSim = r.ref || v.ref;
      const items = r.sub ? [{s: r.sub}] : [];
      let last;
      R.fails.slice().sort((a, b) => a.status.localeCompare(b.status)).forEach(o => {
        const w = STAT[o.status] || o.status;
        if (R.fails.length > 1) { if (o.status !== last) items.push({s: w + ':', f: 1}); items.push({o: o, s: nmf(o)}); }
        else items.push({o: o, s: nmf(o) + ' ' + w});
        last = o.status;
      });
      let lx, ly;
      const rl = r.short || r.label, lw = tw(rl, 12.5);
      if (narrow) { ly = y + 15; lx = lw + 12; } else { ly = y + 24; lx = 0; }
      const lineW = narrow ? W : labW;
      const lab = T(g, 0, ly, rl, 'pq-rl');
      const lob = {x0: 0, x1: lw + 2, y0: ly - 12, y1: ly + 4};
      obst.push(lob);
      let fy = narrow ? ly : ly + 15;
      items.forEach((it, i) => {
        const w = tw(it.s, SM) + (it.f ? 4 : 10), need = it.f && items[i + 1] ? w + tw(items[i + 1].s, SM) : w;
        if (lx + need > lineW && lx > 0) { lx = 0; fy += 14; }
        const t = T(g, lx, fy, it.s, it.o ? 'pq-fail' : it.f ? 'pq-fail pq-fh' : 'pq-sub');
        if (it.o) mark(t, it.o, v.kOf ? v.kOf(it.o) : null);
        obst.push({x0: lx, x1: lx + w, y0: fy - 11, y1: fy + 4});
        lx += w;
      });
      const cy = narrow ? fy + 27 : y + 20;
      const xs = R.pts.map(o => X(val(o, r)));
      if (xs.length > 1) S('line', {class: 'pq-span', x1: Math.min.apply(null, xs), x2: Math.max.apply(null, xs), y1: cy, y2: cy}, g);
      R.pts.slice().sort((a, b) => (a === r.pq) - (b === r.pq) || (a.simulator === refSim) - (b.simulator === refSim)).forEach(o => {
        const role = o === r.pq ? 'pq' : o.simulator === refSim ? 'ref' : 'peer', cx = X(val(o, r)), rad = role === 'pq' ? 5.5 : 4.5;
        const k = v.kOf ? v.kOf(o) : role === 'pq' ? 'prismq' : key(o.simulator);
        mark(S('circle', {class: 'pq-dot ' + role + (o.status !== 'ok' ? ' hollow' : ''), cx: cx, cy: cy, r: rad}, g), o, k);
        obst.push({x0: cx - rad - 1, x1: cx + rad + 1, y0: cy - rad - 1, y1: cy + rad + 1});
        reqs.push({cx: cx, cy: cy, role: role, k: k, s: nmf(o)});
      });
      cols.forEach(c => {
        const res = c.f(r);
        T(g, c.x, cy + 4, res.t, 'pq-col' + (res.miss ? ' miss' : ''), 'end');
      });
      const bot = Math.max(cy + (narrow ? 24 : 20), fy + 10);
      bg.setAttribute('y', rt); bg.setAttribute('height', bot - rt);
      if (!narrow) { lab.setAttribute('y', cy + 4.5); lob.y0 = cy - 8; lob.y1 = cy + 8; }
      y = bot;
    });
    const rank = ['pq', 'ref', 'peer'];
    reqs.sort((a, b) => (v.labelAll ? a.cy - b.cy : 0) || rank.indexOf(a.role) - rank.indexOf(b.role));
    const done = {};
    const lanes = [[-9, -19, -6], [18, 7, 21], [-22, -32, -19], [31, 20, 34]];
    reqs.forEach(q => {
      const id = v.labelAll ? q.cy + '|' + q.cx : q.role + '|' + q.k + '|' + q.s;
      if (done[id]) return;
      const bold = q.role === 'pq' || q.role === 'ref', w = tw(q.s, SM, bold) + 2;
      for (let li = 0; li < lanes.length; li++) {
        const L = lanes[li];
        for (const x0r of [q.cx - w / 2, q.cx - 4, q.cx - w + 4]) {
          const x0 = Math.max(ax0 - 8, Math.min(x0r, ax1 + 10 - w));
          const b = {x0: x0 - 2, x1: x0 + w + 2, y0: q.cy + L[1], y1: q.cy + L[2]};
          if (b.y0 < y0 - 4 || hit(b, obst)) continue;
          obst.push(b);
          done[id] = 1;
          if (li > 1) S('line', {class: 'pq-lead', x1: q.cx, y1: q.cy + (L[0] < 0 ? -6 : 6),
            x2: Math.max(x0, Math.min(q.cx, x0 + w)), y2: q.cy + (L[0] < 0 ? L[2] : L[1])}, gLab);
          nameText(gLab, x0, q.cy + L[0], q.s, q.k, 'pq-dl ' + q.role);
          return;
        }
      }
    });
    const yb = y + 4;
    if (v.band) S('rect', {class: 'pq-band', x: X(v.band[0]), width: X(v.band[1]) - X(v.band[0]), y: y0, height: yb - y0}, gGrid);
    let last = -1e9;
    ticks(lo, hi, v.scale !== 'lin', (ax1 - ax0) / 70).forEach(t => {
      const x = X(t), s = (v.tick || ftick)(t), w = tw(s, 10.5);
      S('line', {class: t === v.refLine ? 'pq-ref1' : 'pq-grid', x1: x, x2: x, y1: y0, y2: yb}, gGrid);
      if (x - w / 2 < last + 8 || x + w / 2 > W || x - w / 2 < 0) return;
      T(gGrid, x, yb + 15, s, 'pq-tick', 'middle');
      last = x + w / 2;
    });
    size(svg, W, yb + 22);
    box.replaceChildren(svg);
    highlight(pin);
  }
  draws.push(draw);
  draw();
}

// ---- line panel: small multiples; x categorical, linear or log
function linePanel(fig, spec) {
  const plot = fig.querySelector('.pq-plot');
  let cur = 0;
  if (spec.views) seg(plot, spec.views.map(v => v.label), 0, i => { cur = i; draw(); }, spec.ctl || 'Measure');
  const grid = H('div', 'pq-mult', plot);
  const cells = spec.mult.map(m => { const d = H('div', 'pq-cell', grid); H('p', 'pq-mt', d, m.title); return H('div', '', d); });
  function draw() {
    const v = Object.assign({}, spec, spec.views ? spec.views[cur] : {});
    spec.mult.forEach((m0, mi) => {
      const m = Object.assign({}, v, m0), box = cells[mi], W = box.clientWidth || 260, Hh = m.h || 196;
      const ser = m.series.map(s => ({s: s, pts: s.pts.map(p => ({x: p.x, o: p.o, y: m.y(p.o, s)})).filter(p => p.y != null)}))
        .filter(s => s.pts.length > 1);
      if (!ser.length) { box.replaceChildren(); return; }
      const ys = [];
      ser.forEach(s => s.pts.forEach(p => ys.push(p.y)));
      let lo = Math.min.apply(null, ys), hi = Math.max.apply(null, ys);
      if (m.log) { lo /= 1.5; hi *= 1.5; } else { lo = 0; hi = Math.max(hi * 1.1, m.minHi || 2); }
      const tk = ticks(lo, hi, m.log, m.log ? 5 : Math.min(6, Math.ceil(hi)));
      const lw = Math.max(0, ...ser.map(s => tw(s.s.name, 11, s.s.cls !== 'peer')));
      const l = Math.max(...tk.map(t => tw(m.tick(t), 10.5))) + 10, r = lw + 16, t = 10, b = 24, iw = W - l - r, ih = Hh - t - b;
      const xs = m.xs, xl = xs[0], xh = xs[xs.length - 1];
      const xi = m.xscale === 'cat' ? x => xs.indexOf(x) / (xs.length - 1) : m.xscale === 'log' ? x => Math.log(x / xl) / Math.log(xh / xl)
        : x => (x - xl) / (xh - xl);
      const X = x => l + xi(x) * iw;
      const Y = yv => t + ih - (m.log ? Math.log(yv / lo) / Math.log(hi / lo) : (yv - lo) / (hi - lo)) * ih;
      const svg = svgEl(W, Hh, m0.title);
      tk.forEach(k => {
        S('line', {class: k === m.refLine ? 'pq-ref1' : 'pq-grid', x1: l, x2: l + iw, y1: Y(k), y2: Y(k)}, svg);
        T(svg, l - 7, Y(k) + 3.5, m.tick(k), 'pq-tick', 'end');
      });
      S('line', {class: 'pq-axis', x1: l, x2: l + iw, y1: t + ih, y2: t + ih}, svg);
      (m.xticks || xs).forEach(x => T(svg, X(x), Hh - 7, m.xfmt(x), 'pq-tick', 'middle'));
      const ends = [];
      ser.sort((a, c) => (a.s.cls === 'pq') - (c.s.cls === 'pq')).forEach(s => {
        const cls = s.s.cls || 'peer';
        S('path', {class: 'pq-line ' + cls + (s.s.dash ? ' dash' : ''), 'data-k': s.s.k,
          d: s.pts.map((p, i) => (i ? 'L' : 'M') + X(p.x).toFixed(1) + ',' + Y(p.y).toFixed(1)).join('')}, svg);
        s.pts.forEach(p => mark(S('circle', {class: 'pq-pt ' + cls + (p.o.status !== 'ok' ? ' hollow' : ''),
          cx: X(p.x), cy: Y(p.y), r: cls === 'pq' ? 3.6 : 3}, svg), p.o, s.s.k));
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
    });
    highlight(pin);
  }
  draws.push(draw);
  draw();
}

// ---- summary: one range bar per peer and method against the PRISM-Q 1x rule
function summary(fig) {
  const box = H('div', 'pq-box', fig.querySelector('.pq-plot'));
  const groups = SEC.map(sec => {
    const rows = secRows(sec);
    return {sec: sec, peers: sec.sims.slice(1).filter(peerOf).map(s => {
      const pts = rows.map(r => ({r: r, o: get(sec.g, r.w, s)})).map(p => ({r: p.r, o: p.o, x: ratio(p.o, p.r.pq)})).filter(p => p.x != null);
      const xs = pts.map(p => p.x);
      return {s: s, pts: pts, ref: s === sec.ref, lo: Math.min.apply(null, xs), hi: Math.max.apply(null, xs), med: xs.length && median(xs)};
    }).filter(p => p.pts.length)};
  }).filter(g => g.peers.length);
  const all = [];
  groups.forEach(g => g.peers.forEach(p => all.push(p)));
  function draw() {
    const W = box.clientWidth || 640, narrow = W < 560;
    const labW = narrow ? 76 : 118, ax0 = labW + 8, ax1 = W - 4, P = ax1 - ax0;
    const wr = Math.max.apply(null, all.map(p => tw(span([p.lo, p.hi]), 11.5, true))) + 12;
    const dom = fitLog(Math.min(1, ...all.map(p => p.lo)) / 1.12, Math.max(1, ...all.map(p => p.hi)), P, 6, wr);
    const X = x => ax0 + (Math.log(x) - dom[0]) / (dom[1] - dom[0]) * P;
    const svg = svgEl(W, 100, title(fig));
    const gGrid = S('g', {}, svg), gRows = S('g', {}, svg);
    const y0 = 20, RH = narrow ? 24 : 26, SH = 26;
    let y = y0;
    const spans = [];
    groups.forEach((g, gi) => {
      if (gi) S('line', {class: 'pq-sep', x1: 0, x2: W, y1: y + 6, y2: y + 6}, gRows);
      T(gRows, 0, y + SH - 7, g.sec.name, 'pq-subhead');
      y += SH;
      spans.push([y, y + g.peers.length * RH]);
      g.peers.forEach(p => {
        const cy = y + RH / 2, row = S('g', {class: 'pq-row pq-srow'}, gRows);
        S('rect', {class: 'pq-rowbg', x: -6, y: y, width: W + 12, height: RH, rx: 8}, row);
        nameText(row, narrow ? 4 : 10, cy + 4, NAME[p.s], p.s, p.ref ? 'ref' : 'peer');
        const x0 = X(p.lo), x1 = X(p.hi);
        const bar = S('rect', {class: 'pq-rbar ' + (p.ref ? 'ref' : 'peer'), x: x0 - 1, y: cy - 4, width: Math.max(2, x1 - x0) + 2, height: 8, rx: 4}, row);
        bar.setAttribute('data-k', p.s);
        S('line', {class: 'pq-medbg', x1: X(p.med), x2: X(p.med), y1: cy - 7.5, y2: cy + 7.5}, row);
        S('line', {class: 'pq-med', x1: X(p.med), x2: X(p.med), y1: cy - 7.5, y2: cy + 7.5}, row);
        p.pts.forEach(q => S('circle', {class: 'pq-spt', cx: X(q.x), cy: cy, r: 3.4}, row));
        const lab = T(row, x1 + 9, cy + 4, '', 'pq-rlab');
        lab.textContent = span([p.lo, p.hi]);
        const rs = p.pts.slice().sort((a, b) => a.x - b.x);
        hot(row, () => '<b>' + esc(NAME[p.s]) + '</b> <span>' + esc(g.sec.name) + '</span><br><span>peer time over PRISM-Q time, median ' +
          fx(p.med) + '</span><table class="pq-tt">' + rs.map(q => '<tr><td>' + esc(q.r.label) + '</td><td>' + fx(q.x) + '</td></tr>').join('') + '</table>',
          NAME[p.s] + ', ' + g.sec.name + ': ' + span([p.lo, p.hi]) + ' over ' + rs.length + ' rows, median ' + fx(p.med));
        row.__ax = x0 + (x1 - x0) / 2;
        y += RH;
      });
    });
    const yb = y + 6;
    let last = -1e9;
    ticks(Math.exp(dom[0]), Math.exp(dom[1]), true, P / 60).forEach(t => {
      const x = X(t), s = fxt(t), w = tw(s, 10.5);
      if (t !== 1) spans.forEach(sp => S('line', {class: 'pq-grid', x1: x, x2: x, y1: sp[0], y2: sp[1]}, gGrid));
      if (x - w / 2 < last + 8 || x + w / 2 > W) return;
      T(gGrid, x, yb + 15, s, 'pq-tick' + (t === 1 ? ' one' : ''), 'middle');
      last = x + w / 2;
    });
    spans.forEach(sp => S('line', {class: 'pq-rule', x1: X(1), x2: X(1), y1: sp[0] + 3, y2: sp[1] - 3}, gGrid));
    T(gGrid, X(1), 12, 'PRISM-Q = 1x', 'pq-rulelab', X(1) - 40 < 0 ? 'start' : 'middle');
    size(svg, W, yb + 22);
    box.replaceChildren(svg);
    highlight(pin);
  }
  draws.push(draw);
  draw();
}

// ---- explore: filter matrix over every comparison row, bars diverging from the PRISM-Q rule
function explore(fig) {
  const plot = fig.querySelector('.pq-plot');
  const st = {m: (SEC.find(s => secRows(s).length) || SEC[0]).id, f: null, q: null, s: null, v: 'time', all: false};
  const fromHash = location.hash.indexOf('#explore') === 0;
  if (fromHash) {
    const p = new URLSearchParams(location.hash.slice(location.hash.indexOf('?') + 1 || location.hash.length));
    if (SECM[p.get('m')]) st.m = p.get('m');
    if (p.get('v') === 'mem') st.v = 'mem';
    ['f', 's'].forEach(k => { if (p.get(k)) st[k] = p.get(k).split(','); });
    if (p.get('q')) st.q = p.get('q').split(',').map(Number);
  }
  const ctl = H('div', 'pq-filters', plot), box = H('div', 'pq-box', plot), more = H('button', 'pq-more', plot);
  more.type = 'button';
  more.onclick = () => { st.all = !st.all; draw(); };
  let rows, fams, qs, sims;
  function scope() {
    const sec = SECM[st.m];
    rows = secRows(sec);
    fams = uniq(rows.map(r => r.pq.family));
    qs = uniq(rows.map(r => r.pq.qubits)).sort((a, b) => a - b);
    sims = sec.sims.slice(1).filter(s => peerOf(s) && rows.some(r => get(sec.g, r.w, s)));
    const keep = (sel, opts) => { const k = (sel || []).filter(x => opts.indexOf(x) >= 0); return k.length ? k : opts.slice(); };
    st.f = keep(st.f, fams); st.q = keep(st.q, qs); st.s = keep(st.s, sims);
  }
  function enc() {
    const p = ['m=' + st.m];
    if (st.f.length < fams.length) p.push('f=' + st.f.join(','));
    if (st.q.length < qs.length) p.push('q=' + st.q.join(','));
    if (st.s.length < sims.length) p.push('s=' + st.s.join(','));
    if (st.v === 'mem') p.push('v=mem');
    return '#explore?' + p.join('&');
  }
  function sync() { try { history.replaceState(null, '', enc()); } catch (e) { /* file:// in some browsers */ } }
  function chipRow(label, items, isOn, click, fmt) {
    const row = H('div', 'pq-frow', ctl);
    H('span', 'pq-flab', row, label);
    const g = H('div', 'pq-chips', row);
    g.setAttribute('role', 'group');
    g.setAttribute('aria-label', label);
    items.forEach(it => {
      const b = H('button', 'pq-chip' + (it === 'prismq' ? ' pq' : ''), g, fmt ? fmt(it) : String(it));
      b.type = 'button';
      b.setAttribute('aria-pressed', isOn(it));
      b.onclick = () => { click(it); st.all = false; render(); sync(); };
    });
    return row;
  }
  const tog = (arr, x) => arr.indexOf(x) >= 0 ? (arr.length > 1 ? arr.filter(y => y !== x) : arr) : arr.concat([x]);
  function render() {
    scope();
    ctl.replaceChildren();
    chipRow('Method', SEC.map(s => s.id), x => x === st.m, x => { st.m = x; st.f = st.q = st.s = null; }, x => SECM[x].chip);
    if (fams.length > 1) chipRow('Family', fams, x => st.f.indexOf(x) >= 0, x => { st.f = tog(st.f, x); });
    if (qs.length > 1) chipRow('Qubits', qs, x => st.q.indexOf(x) >= 0, x => { st.q = tog(st.q, x); }, x => x.toLocaleString('en-US') + 'q');
    chipRow('Simulators', sims, x => st.s.indexOf(x) >= 0, x => { st.s = tog(st.s, x); }, x => NAME[x]);
    const mr = chipRow('Metric', ['time', 'mem'], x => x === st.v, x => { st.v = x; }, x => x === 'time' ? 'Time' : 'Peak memory');
    const cp = H('button', 'pq-copy', mr, 'Copy link');
    cp.type = 'button';
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
    draw();
  }
  function draw() {
    const sec = SECM[st.m], mem = st.v === 'mem';
    const shown = rows.filter(r => st.f.indexOf(r.pq.family) >= 0 && st.q.indexOf(r.pq.qubits) >= 0);
    const LIMIT = 6, list = st.all ? shown : shown.slice(0, LIMIT);
    more.hidden = shown.length <= LIMIT;
    more.textContent = st.all ? 'Show fewer rows' : 'Show all ' + shown.length + ' rows';
    more.setAttribute('aria-expanded', st.all);
    const W = box.clientWidth || 640, narrow = W < 560;
    const amount = o => mem ? o.peak_rss_mb : o.median_ms, fa = mem ? fmb : fms;
    const cell = (o, pq) => {
      if (!o) return null;
      pq = mate(o, pq);
      if (o.status !== 'ok' || amount(o) == null) return {o: o, fail: 1};
      if (pq.status !== 'ok' || amount(pq) == null) return {o: o, fail: 1};
      return {o: o, r: amount(o) / amount(pq), soft: !mem && !(valid(o) && valid(pq))};
    };
    const groups = list.map(r => ({r: r, cells: st.s.map(s => ({s: s, c: cell(get(sec.g, r.w, s), r.pq)})).filter(x => x.c)}));
    const labW = narrow ? 70 : 104, ax0 = labW + 6, ax1 = W - 4, P = ax1 - ax0, FS = narrow ? 10.5 : 11.5;
    const lab = c => fx(c.r) + ' · ' + fa(amount(c.o));
    let lo = 1, hi = 1, wl = 8, wr = 8;
    groups.forEach(g => g.cells.forEach(x => {
      const c = x.c;
      if (c.fail) { wr = Math.max(wr, tw(STUB[c.o.status] || c.o.status, 10.5) + 22); return; }
      lo = Math.min(lo, c.r); hi = Math.max(hi, c.r);
      const w = tw(lab(c), FS, true) + 12;
      if (c.r < 1) wl = Math.max(wl, w); else wr = Math.max(wr, w);
    }));
    groups.forEach(g => { wr = Math.max(wr, tw(fa(amount(g.r.pq)), FS, true) + 14); });
    const dom = fitLog(lo / 1.05, hi * 1.05, P, wl, wr);
    const X = x => ax0 + (Math.log(x) - dom[0]) / (dom[1] - dom[0]) * P;
    const svg = svgEl(W, 100, 'Peer ' + (mem ? 'peak memory' : 'time') + ' as a multiple of PRISM-Q, ' + sec.name);
    const gGrid = S('g', {}, svg), gRows = S('g', {}, svg);
    const y0 = 22, RH = 19, HH = 24;
    let y = y0;
    const spans = [];
    if (!groups.length) T(gRows, 0, y + 20, 'No rows match these filters.', 'pq-sub');
    groups.forEach((g, gi) => {
      if (gi) S('line', {class: 'pq-sep', x1: 0, x2: W, y1: y + 2, y2: y + 2}, gRows);
      const pq = g.r.pq;
      const ht = T(gRows, 0, y + HH - 6, g.r.label, 'pq-ghead');
      if (pq.gates && !narrow) { const ts = S('tspan', {class: 'pq-gsub'}, ht); ts.textContent = '  ' + pq.gates.toLocaleString('en-US') + ' gates'; }
      y += HH;
      const rx = X(1), top0 = y;
      const pr = S('g', {class: 'pq-row'}, gRows);
      S('rect', {class: 'pq-rowbg', x: -6, y: y, width: W + 12, height: RH, rx: 6}, pr);
      nameText(pr, 0, y + RH / 2 + 4, 'PRISM-Q', 'prismq', 'pq', null);
      const pv = amount(pq), pqok = pq.status === 'ok' && pv != null;
      const pd = mark(S('circle', {class: 'pq-dot pq', cx: rx, cy: y + RH / 2, r: 4}, pr), pq, 'prismq');
      T(pr, rx + 9, y + RH / 2 + 4, pqok ? fa(pv) : STAT[pq.status], 'pq-blab pq');
      pd.setAttribute('aria-label', 'PRISM-Q, ' + g.r.label + ': ' + (pqok ? fa(pv) : STAT[pq.status]));
      y += RH;
      g.cells.forEach(x => {
        const c = x.c, cy = y + RH / 2, row = S('g', {class: 'pq-row'}, gRows), ref = x.s === sec.ref;
        S('rect', {class: 'pq-rowbg', x: -6, y: y, width: W + 12, height: RH, rx: 6}, row);
        nameText(row, 0, cy + 4, NAME[x.s], x.s, ref ? 'ref' : 'peer');
        if (c.fail) {
          const word = STUB[c.o.status] || c.o.status, w = tw(word, 10.5) + 14;
          const stub = mark(S('rect', {class: 'pq-stub', x: rx + 3, y: cy - 7, width: w, height: 14, rx: 7}, row), c.o, x.s);
          stub.setAttribute('aria-label', NAME[x.s] + ', ' + g.r.label + ': ' + STAT[c.o.status] + (c.o.reason ? ', ' + c.o.reason : ''));
          T(row, rx + 3 + w / 2, cy + 3.6, word, 'pq-stubt', 'middle');
        } else {
          const xe = X(c.r), left = c.r < 1;
          const b = mark(S('rect', {class: 'pq-bar ' + (ref ? 'ref' : 'peer') + (c.soft ? ' soft' : ''),
            x: Math.min(rx, xe), y: cy - 4.5, width: Math.max(1.5, Math.abs(xe - rx)), height: 9, rx: 2}, row), c.o, x.s);
          b.__ax = left ? 0 : Math.abs(xe - rx);
          const t = T(row, left ? xe - 7 : xe + 7, cy + 4, '', 'pq-blab', left ? 'end' : 'start');
          S('tspan', {class: 'pq-bx'}, t).textContent = fx(c.r);
          S('tspan', {}, t).textContent = ' · ' + fa(amount(c.o));
        }
        y += RH;
      });
      S('line', {class: 'pq-rule', x1: rx, x2: rx, y1: top0 + 2, y2: y - 2}, gRows);
      spans.push([top0, y]);
      y += 6;
    });
    const yb = y;
    let last = -1e9;
    if (groups.length) ticks(Math.exp(dom[0]), Math.exp(dom[1]), true, P / 60).forEach(t => {
      const x = X(t), s = fxt(t), w = tw(s, 10.5);
      if (t !== 1) spans.forEach(sp => S('line', {class: 'pq-grid', x1: x, x2: x, y1: sp[0], y2: sp[1]}, gGrid));
      if (x - w / 2 < last + 8 || x + w / 2 > W) return;
      T(gGrid, x, 12, s, 'pq-tick' + (t === 1 ? ' one' : ''), 'middle');
      last = x + w / 2;
    });
    size(svg, W, yb + 4);
    box.replaceChildren(svg);
    highlight(pin);
  }
  draws.push(draw);
  render();
  if (fromHash) setTimeout(() => fig.scrollIntoView({block: 'start'}), 60);
}

// ---- memory: bars from zero per simulator, with the bare fp64 state as a reference line
function memPanel(fig) {
  const plot = fig.querySelector('.pq-plot'), sims = ['prismq', 'aer-statevector', 'quest', 'aer-statevector-f32', 'qsim'];
  const rows = secRows(SECM.sv64).filter(r => r.pq.family === 'rcs' && r.pq.depth === 10);
  let cur = rows.length - 1;
  seg(plot, rows.map(r => r.pq.qubits + 'q'), cur, i => { cur = i; draw(); }, 'Qubits');
  const box = H('div', 'pq-box', plot);
  function draw() {
    const r = rows[cur], n = r.pq.qubits, state = 16 * Math.pow(2, n) / 1048576;
    const W = box.clientWidth || 640, narrow = W < 560, labW = narrow ? 70 : 104, ax0 = labW + 6;
    const items = sims.map(s => ({s: s, o: get('headline', r.w, s)})), mb = it => it.o && it.o.status === 'ok' ? it.o.peak_rss_mb : null;
    const hi = Math.max(state, ...items.map(mb).filter(v => v != null)) * 1.04;
    const wr = Math.max(...items.map(it => mb(it) != null ? tw(fmb(mb(it)), 11.5, true) : 0)) + 12, P = W - 4 - ax0 - wr;
    const X = v => ax0 + v / hi * P;
    const svg = svgEl(W, 100, title(fig)), gGrid = S('g', {}, svg), gRows = S('g', {}, svg);
    const y0 = 30, RH = 24;
    let y = y0;
    items.forEach(it => {
      const cy = y + RH / 2, row = S('g', {class: 'pq-row'}, gRows), role = it.s === 'prismq' ? 'pq' : it.s === 'aer-statevector' ? 'ref' : 'peer';
      S('rect', {class: 'pq-rowbg', x: -6, y: y, width: W + 12, height: RH, rx: 6}, row);
      nameText(row, 0, cy + 4, NAME[it.s], it.s, role);
      const v = mb(it);
      if (v != null) {
        mark(S('rect', {class: 'pq-bar ' + role, x: ax0, y: cy - 5, width: Math.max(1.5, X(v) - ax0), height: 10, rx: 2}, row), it.o, it.s);
        T(row, X(v) + 7, cy + 4, fmb(v), 'pq-blab pq-bx' + (role === 'pq' ? ' pq' : ''));
      } else {
        const word = it.o ? STUB[it.o.status] || it.o.status : 'not run', w = tw(word, 10.5) + 14;
        const stub = S('rect', {class: 'pq-stub', x: ax0 + 2, y: cy - 7, width: w, height: 14, rx: 7}, row);
        if (it.o) mark(stub, it.o, it.s);
        else hot(stub, '<b>' + esc(NAME[it.s]) + '</b><br><i>no run at ' + n + ' qubits</i>', NAME[it.s] + ': no run at ' + n + ' qubits');
        T(row, ax0 + 2 + w / 2, cy + 3.6, word, 'pq-stubt', 'middle');
      }
      y += RH;
    });
    const yb = y + 2, u = hi >= 1024 ? 1024 : 1, un = u > 1 ? ' GB' : ' MB';
    let last = -1e9;
    ticks(0, hi / u, false, P / 90).forEach(t => {
      const x = X(t * u), s = +t.toPrecision(3) + un, w = tw(s, 10.5);
      S('line', {class: t ? 'pq-grid' : 'pq-axis', x1: x, x2: x, y1: y0, y2: yb}, gGrid);
      if (x - w / 2 < last + 8 || x + w / 2 > W) return;
      T(gGrid, x, yb + 15, s, 'pq-tick', 'middle');
      last = x + w / 2;
    });
    const sx = X(state), sl = 'fp64 state ' + fmb(state), sw = tw(sl, 11);
    S('line', {class: 'pq-stateline', x1: sx, x2: sx, y1: y0 - 6, y2: yb}, gGrid);
    T(gGrid, Math.min(Math.max(sx, ax0 + sw / 2), W - sw / 2), y0 - 12, sl, 'pq-statelab', 'middle');
    size(svg, W, yb + 22);
    box.replaceChildren(svg);
    highlight(pin);
  }
  draws.push(draw);
  draw();
}

// ---- panels
const views = (rows, keyOf, labels) => [{label: 'All', rows: rows}].concat(labels.map(l => ({label: l[1], rows: rows.filter(r => keyOf(r.pq) === l[0])})));
const ratioCol = (h, ref) => ({h: h, w: 62, f: r => {
  const o = r.peers.find(p => p.simulator === ref), x = ratio(o, r.pq);
  return x != null ? {t: fx(x)} : {t: o ? (o.status !== 'ok' ? STUB[o.status] || o.status : 'unchecked') : 'not run', miss: 1};
}});
const PANELS = {
  summary: summary,
  explore: explore,
  sv64: fig => { const rows = secRows(SECM.sv64); dotPanel(fig, {ctl: 'Qubits', ref: 'aer-statevector',
    views: views(rows, o => o.qubits, [16, 20, 24, 26].map(q => [q, q + 'q'])), cols: [ratioCol(['Aer /', 'PRISM-Q'], 'aer-statevector')]}); },
  sv32: fig => { const rows = secRows(SECM.sv32); dotPanel(fig, {ctl: 'Qubits', ref: 'qsim',
    views: views(rows, o => o.qubits, [16, 20, 24, 26].map(q => [q, q + 'q'])), cols: [ratioCol(['qsim /', 'PRISM-Q'], 'qsim')]}); },
  stabilizer: fig => dotPanel(fig, {ref: 'stim', views: [{rows: secRows(SECM.stabilizer)}], cols: [ratioCol(['Stim /', 'PRISM-Q'], 'stim')]}),
  mps: fig => { const rows = secRows(SECM.mps); dotPanel(fig, {ctl: 'Depth', ref: 'aer-mps',
    views: views(rows, o => o.depth, [[4, 'depth 4'], [8, 'depth 8']]), cols: [ratioCol(['Aer MPS /', 'PRISM-Q'], 'aer-mps')]}); },
  sampling: fig => { const rows = secRows(SECM.sampling); dotPanel(fig, {ctl: 'Qubits', ref: 'aer-statevector',
    views: views(rows, o => o.qubits, [16, 20, 24].map(q => [q, q + 'q'])), cols: [ratioCol(['Aer /', 'PRISM-Q'], 'aer-statevector')]}); },
  disjoint: fig => dotPanel(fig, {ref: 'aer-mps', views: [{rows: secRows(SECM.disjoint)}], cols: [ratioCol(['Aer MPS /', 'PRISM-Q'], 'aer-mps')]}),
  highshot: fig => dotPanel(fig, {ref: 'stim', views: [{rows: secRows(SECM.highshot)}],
    name: o => o.simulator === 'prismq' ? 'PRISM-Q ' + o.threads + 't' : nm(o), kOf: o => key(o.simulator),
    cols: [ratioCol(['Stim /', 'PRISM-Q 1t'], 'stim')]}),
  mpsdeep: fig => { const rows = secRows(SECM.mpsdeep); dotPanel(fig, {ctl: 'Depth', ref: 'aer-mps:256',
    views: views(rows, o => o.depth, [[12, 'depth 12'], [16, 'depth 16']]), cols: [ratioCol(['Aer MPS /', 'PRISM-Q MPS'], 'aer-mps:256')]}); },
  tn: fig => dotPanel(fig, {ref: 'quimb-tn', views: [{rows: secRows(SECM.tn)}], cols: [ratioCol(['quimb TN /', 'PRISM-Q'], 'quimb-tn')]}),
  sweep: fig => {
    const sw = O.filter(o => o.group === 'bondsweep'), caps = uniq(sw.filter(o => o.bond_cap).map(o => o.bond_cap)).sort((a, b) => a - b);
    const ns = uniq(sw.map(o => o.qubits)).sort((a, b) => a - b);
    const ser = n => [{k: 'aer-mps', name: 'Aer MPS', cls: 'ref', pfx: 'aer-mps:'}, {k: 'quimb-mps', name: 'quimb MPS', cls: 'peer', pfx: 'quimb-mps:'},
      {k: 'prismq', name: 'PRISM-Q MPS', cls: 'pq', pfx: 'prismq:mps-'}]
      .map(s => ({k: s.k, name: s.name, cls: s.cls, pts: caps.map(c => ({x: c, o: sw.find(o => o.qubits === n && o.simulator === s.pfx + c)})).filter(p => p.o)}));
    const floor = 1e-16, err = o => o.max_abs_error == null ? null : Math.max(o.max_abs_error, floor);
    linePanel(fig, {xs: caps, xscale: 'cat', xfmt: c => 'cap ' + c, log: true, mult: [].concat.apply([], ns.map(n => [
      {title: 'Largest error, ' + n + ' qubits', series: ser(n), y: err, tick: t => t >= 0.01 ? String(t) : '1e' + Math.round(Math.log10(t))},
      {title: 'Median time, ' + n + ' qubits', series: ser(n), y: o => o.median_ms, tick: ftick}]))});
  },
  bondcap: fig => {
    const acc = O.filter(o => o.group === 'accuracy'), caps = uniq(acc.filter(o => o.bond_cap).map(o => o.bond_cap)).sort((a, b) => a - b);
    const at = s => acc.find(o => o.simulator === s);
    const ser = [{k: 'aer-mps', name: 'Aer MPS', cls: 'ref', pfx: 'aer-mps:'}, {k: 'prismq', name: 'PRISM-Q MPS', cls: 'pq', pfx: 'prismq:mps-'}]
      .map(s => ({k: s.k, name: s.name, cls: s.cls, pts: caps.map(c => ({x: c, o: at(s.pfx + c)})).filter(p => p.o)}));
    linePanel(fig, {xs: caps, xscale: 'cat', xfmt: c => 'cap ' + c, log: true, mult: [
      {title: 'Median time', series: ser, y: o => o.median_ms, tick: ftick},
      {title: 'Distance from the exact output', series: ser, y: o => o.tvd, tick: t => t >= 0.01 ? String(t) : '1e' + Math.round(Math.log10(t))}]});
  },
  threads: fig => {
    const th = O.filter(o => o.group === 'threads'), ws = uniq(th.map(o => o.workload)).sort((a, b) => order(th.find(o => o.workload === a), th.find(o => o.workload === b)));
    const at = (w, s, t) => th.find(o => o.workload === w && o.simulator === s && o.threads === t);
    const mult = ws.map(w => ({title: th.find(o => o.workload === w).label, series: ['quest', 'qsim', 'aer-statevector', 'prismq'].map(s => ({k: s, name: NAME[s],
      cls: s === 'prismq' ? 'pq' : 'peer', pts: [1, 2, 4, 8].map(t => ({x: t, o: at(w, s, t)})).filter(p => ok(p.o))}))}));
    linePanel(fig, {mult: mult, xs: [1, 2, 4, 8], xscale: 'cat', xfmt: x => x + (x > 1 ? ' threads' : ' thread'), h: 200, views: [
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
    dotPanel(fig, {ctl: 'View', name: o => o.simulator === 'prismq' ? 'auto' : o.simulator.slice(7),
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
    linePanel(fig, {log: true, y: o => o.median_ms, tick: ftick, mult: [
      {title: 'QFT, HEA, QV, GHZ · 16 to 28 qubits', series: dense, xs: [16, 28], xticks: [16, 20, 24, 28], xscale: 'lin', xfmt: x => x + 'q'},
      {title: 'GHZ · ' + wide[0].x + ' to ' + wide[wide.length - 1].x.toLocaleString('en-US') + ' qubits',
        series: [{k: 'prismq', name: 'GHZ', cls: 'pq', pts: wide}], xs: [wide[0].x, wide[wide.length - 1].x],
        xticks: wide.map(p => p.x), xscale: 'log', xfmt: x => x.toLocaleString('en-US') + 'q'}]});
  }
};

function init() {
  FF = getComputedStyle(document.querySelector('.content') || document.body).fontFamily || 'sans-serif';
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
    FF = getComputedStyle(document.querySelector('.content') || document.body).fontFamily || FF;
    redraw();
  });
}
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
