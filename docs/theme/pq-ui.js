// Reading aids layered on the mdBook page: an "on this page" rail, previous and next cards
// and a Ctrl+K search shortcut. Every page reads fine without it.
(function () {
'use strict';
const root = document.documentElement, main = document.querySelector('.content main');
if (!main) return;
root.classList.add('pq-ui');
if (/Mac|iPhone|iPad/.test(navigator.platform || '')) root.classList.add('pq-mac');
const el = (tag, cls, par, txt) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (txt != null) e.textContent = txt;
    if (par) par.appendChild(e);
    return e;
};

// On this page
const heads = Array.prototype.slice.call(main.querySelectorAll('h2[id], h3[id]')).filter(h => !h.closest('.pq-figure'));
if (heads.length >= 3) {
    const nav = el('nav', 'pq-toc', document.getElementById('page-wrapper'));
    nav.setAttribute('aria-label', 'On this page');
    el('p', '', nav, 'On this page');
    const ol = el('ol', '', nav), links = new Map();
    const lean = heads.length > 18;
    heads.forEach(h => {
        if (lean && h.tagName === 'H3') return;
        const a = el('a', h.tagName === 'H3' ? 'sub' : '', el('li', '', ol), h.textContent);
        a.href = '#' + h.id;
        links.set(h, a);
    });
    root.classList.add('pq-has-toc');
    let cur = null;
    const spy = () => {
        let hit = null;
        for (const h of links.keys()) { if (h.getBoundingClientRect().top < 120) hit = h; else break; }
        if (hit === cur) return;
        if (cur) links.get(cur).removeAttribute('aria-current');
        cur = hit;
        if (cur) {
            const a = links.get(cur);
            a.setAttribute('aria-current', 'true');
            if (nav.scrollHeight > nav.clientHeight) nav.scrollTop = a.offsetTop - nav.clientHeight / 2;
        }
    };
    let raf = 0;
    addEventListener('scroll', () => { if (!raf) raf = requestAnimationFrame(() => { raf = 0; spy(); }); }, {passive: true});
    spy();
}

// Previous and next cards, titled from the sidebar once toc.js has filled it
const pager = () => {
    const prev = document.querySelector('.nav-chapters.previous, .mobile-nav-chapters.previous');
    const next = document.querySelector('.nav-chapters.next, .mobile-nav-chapters.next');
    if (!prev && !next) return;
    const name = a => {
        if (!a) return '';
        const hit = Array.prototype.find.call(document.querySelectorAll('.sidebar .chapter a[href]'), s => s.href === a.href);
        if (!hit) return '';
        const c = hit.cloneNode(true), n = c.querySelector('strong');
        if (n) n.remove();
        return c.textContent.trim();
    };
    const box = el('nav', 'pq-pager');
    box.setAttribute('aria-label', 'Previous and next page');
    [[prev, 'previous', 'Previous'], [next, 'next', 'Next']].forEach(x => {
        if (!x[0]) return;
        const a = el('a', 'pq-card ' + x[1], box);
        a.href = x[0].getAttribute('href');
        a.rel = x[1] === 'next' ? 'next' : 'prev';
        el('small', '', a, x[2]);
        el('span', '', a, name(x[0]) || (x[1] === 'next' ? 'Next page' : 'Previous page'));
    });
    main.after(box);
};
if (document.querySelector('.sidebar .chapter a')) pager(); else addEventListener('DOMContentLoaded', pager);

// Ctrl+K or Cmd+K opens search; a click on the scrim closes it
const toggle = document.getElementById('search-toggle'), wrap = document.getElementById('search-wrapper');
if (toggle && wrap) {
    addEventListener('keydown', e => {
        if ((e.ctrlKey || e.metaKey) && !e.shiftKey && !e.altKey && e.key.toLowerCase() === 'k') { e.preventDefault(); toggle.click(); }
    });
    wrap.addEventListener('mousedown', e => { if (e.target === wrap) toggle.click(); });
}
})();
