/* LoopSeek project page: the overview animation (Fig. 2 of the paper, step by step) and the replays of the
   App. E searches (window tree, timeline, kept frames). Every replay state is a pure function of the step index,
   so play, pause, previous/next and the progress bar all go through render(s). */
(function () {
  'use strict';
  var RM = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;
  var NS = 'http://www.w3.org/2000/svg';

  function R(path) { var m = window.LOOPSEEK_INLINE; return (m && m[path]) || path; }   // single-file preview only
  function S(tag, attrs, parent) {
    if (tag === 'image' && attrs.href) attrs.href = R(attrs.href);
    var e = document.createElementNS(NS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function H(tag, cls, html, parent) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html != null) e.innerHTML = html;
    if (parent) parent.appendChild(e);
    return e;
  }
  function txt(parent, x, y, s, attrs) {
    var a = { x: x, y: y };
    for (var k in attrs) a[k] = attrs[k];
    var t = S('text', a, parent);
    t.innerHTML = s;
    return t;
  }
  var DASH = '–';
  function fmt(w) { return w[0].toFixed(0) + DASH + w[1].toFixed(0); }

  /* attribute tween for SVG geometry (CSS transitions on x/width are not supported everywhere) */
  function tween(el, to, ms) {
    if (el._tw) cancelAnimationFrame(el._tw);
    var from = {}, k;
    for (k in to) from[k] = parseFloat(el.getAttribute(k)) || 0;
    if (RM || !ms) { for (k in to) el.setAttribute(k, to[k]); return; }
    var t0 = performance.now();
    function f(now) {
      var p = Math.min(1, (now - t0) / ms), e = p < 0.5 ? 2 * p * p : 1 - Math.pow(-2 * p + 2, 2) / 2;
      for (var k in to) el.setAttribute(k, from[k] + (to[k] - from[k]) * e);
      if (p < 1) el._tw = requestAnimationFrame(f);
    }
    el._tw = requestAnimationFrame(f);
  }

  /* elements carry data-on (first state shown) and optionally data-off (first state hidden again) */
  function showByState(root, s) {
    var els = root.querySelectorAll('[data-on]');
    for (var i = 0; i < els.length; i++) {
      var e = els[i], on = +e.getAttribute('data-on'), off = e.getAttribute('data-off');
      var vis = s >= on && (off == null || s < +off);
      e.classList.toggle('off', !vis);
    }
  }
  function prepDraw(root) {          // stroke-draw effect for paths and lines marked .draw
    var els = root.querySelectorAll('.draw');
    for (var i = 0; i < els.length; i++) {
      var L = els[i].getTotalLength ? els[i].getTotalLength() : 0;
      els[i].style.strokeDasharray = L + ' ' + L;
      els[i].style.setProperty('--len', L);
    }
  }

  /* ---------------- player: controls, timing, autoplay when visible ---------------- */
  function Player(host, label) {
    var self = this;
    this.n = 1; this.s = 0; this.timer = null; this.playing = false; this.started = false;
    var bar = H('div', 'pl-ctl', null, host);
    function btn(sym, name) {
      var b = H('button', 'pl-btn', sym, bar);
      b.type = 'button'; b.setAttribute('aria-label', name); b.title = name;
      return b;
    }
    this.bReplay = btn('&#8634;', 'Replay from the start');
    this.bPrev = btn('&#8249;', 'Previous step');
    this.bPlay = btn('&#9654;', 'Play');
    this.bNext = btn('&#8250;', 'Next step');
    this.prog = H('div', 'pl-prog', null, bar);
    this.prog.setAttribute('aria-label', label + ' progress');
    this.count = H('span', 'pl-count', '', bar);
    this.bReplay.onclick = function () { self.go(0, false); self.play(); };
    this.bPrev.onclick = function () { self.pause(); self.go(self.s - 1, true); };
    this.bNext.onclick = function () { self.pause(); self.go(self.s + 1, true); };
    this.bPlay.onclick = function () { self.playing ? self.pause() : self.play(); };
    if ('IntersectionObserver' in window) {
      new IntersectionObserver(function (es) {
        es.forEach(function (e) {
          if (e.isIntersecting && !self.started && !RM) { self.started = true; self.play(); }
        });
      }, { threshold: 0.35 }).observe(host);
    }
  }
  Player.prototype.configure = function (n, render, dur, names) {
    this.pause();
    this.n = n; this.render = render; this.dur = dur;
    this.prog.innerHTML = '';
    var self = this;
    this.segs = [];
    for (var i = 0; i < n; i++) {
      var b = H('button', 'pl-seg', '', this.prog);
      b.type = 'button';
      b.setAttribute('aria-label', names ? names[i] : 'Step ' + i);
      b.title = names ? names[i] : '';
      (function (k) { b.onclick = function () { self.pause(); self.go(k, true); }; })(i);
      this.segs.push(b);
    }
    this.go(RM ? n - 1 : 0, false);
  };
  Player.prototype.go = function (k, anim) {
    k = Math.max(0, Math.min(this.n - 1, k));
    this.s = k;
    this.render(k, anim && !RM);
    for (var i = 0; i < this.segs.length; i++) {
      this.segs[i].classList.toggle('done', i < k);
      this.segs[i].classList.toggle('cur', i === k);
    }
    this.count.textContent = k + ' / ' + (this.n - 1);
    this.bPrev.disabled = k === 0;
    this.bNext.disabled = k === this.n - 1;
  };
  Player.prototype.play = function () {
    var self = this;
    if (this.s >= this.n - 1) this.go(0, false);
    this.playing = true;
    this.bPlay.innerHTML = '&#10074;&#10074;'; this.bPlay.setAttribute('aria-label', 'Pause'); this.bPlay.title = 'Pause';
    clearTimeout(this.timer);
    (function tick() {
      self.timer = setTimeout(function () {
        if (!self.playing) return;
        self.go(self.s + 1, true);
        if (self.s >= self.n - 1) self.pause(); else tick();
      }, self.dur(self.s));
    })();
  };
  Player.prototype.pause = function () {
    this.playing = false;
    clearTimeout(this.timer);
    this.bPlay.innerHTML = '&#9654;'; this.bPlay.setAttribute('aria-label', 'Play'); this.bPlay.title = 'Play';
  };

  /* ---------------- overview animation (Fig. 2) ---------------- */
  var C = { ctrl: '#1f6fb2', loc: '#1a9a80', sel: '#c9971c', ver: '#e0742f', sys2: '#b04f8c', green: '#16a085',
            yellow: '#f2c200', ink: '#1f2328', muted: '#5b636d' };

  function sub(base, s) { return base + '<tspan baseline-shift="sub" font-size="72%">' + s + '</tspan>'; }

  function film(g, x, y, w, h, src, on, off) {
    var grp = S('g', { 'data-on': on }, g);
    if (off != null) grp.setAttribute('data-off', off);
    S('rect', { x: x - 3, y: y - 4, width: w + 6, height: h + 8, rx: 3, fill: '#111' }, grp);
    S('line', { x1: x - 1, y1: y - 2, x2: x + w + 1, y2: y - 2, stroke: '#fff', 'stroke-width': 1.6, 'stroke-dasharray': '3 3' }, grp);
    S('line', { x1: x - 1, y1: y + h + 2, x2: x + w + 1, y2: y + h + 2, stroke: '#fff', 'stroke-width': 1.6, 'stroke-dasharray': '3 3' }, grp);
    S('image', { href: src, x: x, y: y, width: w, height: h, preserveAspectRatio: 'none' }, grp);
    return grp;
  }

  function buildOverview(host) {
    var base = 'static/images/anim/ov/';
    var wrap = H('div', 'anim-scroll', null, host);
    var svg = S('svg', { viewBox: '0 0 1000 560', class: 'ov-svg', role: 'img',
                         'aria-label': 'Animated overview of LoopSeek, step by step as in Fig. 2 of the paper' }, wrap);
    var defs = S('defs', {}, svg);
    [['ah-g', C.green], ['ah-o', C.ver], ['ah-b', C.ctrl], ['ah-k', '#666'], ['ah-p', C.sys2]].forEach(function (m) {
      var mk = S('marker', { id: m[0], viewBox: '0 0 10 10', refX: 8, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' }, defs);
      S('path', { d: 'M0 0L10 5L0 10z', fill: m[1] }, mk);
    });
    var cap = H('p', 'anim-cap', '', host);
    cap.setAttribute('aria-live', 'polite');

    // panels
    S('rect', { x: 4, y: 4, width: 262, height: 412, rx: 12, class: 'panel-bg' }, svg);
    S('rect', { x: 274, y: 4, width: 722, height: 412, rx: 12, class: 'panel-bg' }, svg);
    S('rect', { x: 4, y: 424, width: 992, height: 132, rx: 12, class: 'panel-bg' }, svg);
    txt(svg, 16, 30, 'System I: overview', { class: 'ov-title' });
    txt(svg, 286, 30, 'System I: search on a window tree', { class: 'ov-title' });
    txt(svg, 984, 452, 'System II: answer once', { class: 'ov-title', 'text-anchor': 'end' });

    // ---- left: overview frames, Verifier, F_a
    var L = S('g', {}, svg);
    txt(L, 135, 54, sub('<tspan font-style="italic">F</tspan>', 'o') + ': ' + sub('<tspan font-style="italic">K</tspan>', 'o') + ' uniform frames', { class: 'ov-lab', 'text-anchor': 'middle', 'data-on': 1 });
    S('rect', { x: 12, y: 62, width: 246, height: 196, rx: 6, fill: '#1c1c1c', 'data-on': 1 }, L);
    var fw = 56, fh = 43, gx = 4, x0 = 17, y0 = 68, pos = [];
    for (var i = 0; i < 16; i++) {
      var x = x0 + (i % 4) * (fw + gx), y = y0 + Math.floor(i / 4) * (fh + 4);
      pos.push([x, y]);
      var im = S('image', { href: base + 'o' + (i < 10 ? '0' : '') + i + '.webp', x: x, y: y, width: fw, height: fh,
                            preserveAspectRatio: 'none', 'data-on': 1, class: 'pop' }, L);
      im.style.transitionDelay = (i * 45) + 'ms';
    }
    var FA = [9, 10, 11, 14];
    FA.forEach(function (k) {
      S('rect', { x: pos[k][0] - 1.5, y: pos[k][1] - 1.5, width: fw + 3, height: fh + 3, fill: 'none', stroke: C.ver,
                  'stroke-width': 3, rx: 2, 'data-on': 2 }, L);
    });
    var vbox = S('g', { class: 'agent', 'data-lit': '2' }, L);
    S('rect', { x: 22, y: 282, width: 152, height: 48, rx: 9, fill: '#fdf1e8', stroke: C.ver, 'stroke-width': 2 }, vbox);
    txt(vbox, 98, 302, 'Verifier', { class: 'ov-agent', fill: C.ver, 'text-anchor': 'middle' });
    txt(vbox, 98, 321, 'Shows evidence?', { class: 'ov-lab', 'font-style': 'italic', 'text-anchor': 'middle' });
    S('path', { d: 'M222 262 V 360', stroke: C.ver, 'stroke-width': 2, fill: 'none', 'marker-end': 'url(#ah-o)', class: 'draw', 'data-on': 3 }, L);
    txt(L, 230, 316, 'top ' + sub('<tspan font-style="italic">K</tspan>', 'a'), { class: 'ov-lab', fill: C.ver, 'data-on': 3 });
    txt(L, 16, 396, sub('<tspan font-style="italic">F</tspan>', 'a'), { class: 'ov-sym', 'data-on': 3 });
    var fa_w = 50, fa_h = 39;
    FA.forEach(function (k, j) {
      var x = 44 + j * (fa_w + 4), y = 370;
      var g = S('g', { 'data-on': 3, class: 'fly' }, L);
      g.style.setProperty('--dx', (pos[k][0] - x) + 'px');
      g.style.setProperty('--dy', (pos[k][1] - y) + 'px');
      g.style.transitionDelay = (j * 90) + 'ms';
      S('image', { href: base + 'o' + (k < 10 ? '0' : '') + k + '.webp', x: x, y: y, width: fa_w, height: fa_h, preserveAspectRatio: 'none' }, g);
      S('rect', { x: x - 1, y: y - 1, width: fa_w + 2, height: fa_h + 2, fill: 'none', stroke: C.ver, 'stroke-width': 2.5 }, g);
    });

    // ---- middle: window tree
    var M = S('g', {}, svg);
    txt(M, 300, 52, 'frames', { class: 'ov-lab', 'font-style': 'italic', fill: C.muted });
    S('path', { d: 'M350 48 H 628', stroke: '#888', 'stroke-width': 1.2, 'marker-end': 'url(#ah-k)', fill: 'none' }, M);
    var SX = 294, SW = 334, SY = 64, SH = 30;
    film(M, SX, SY, SW, SH, base + 'strip.webp', 0);
    for (var t = 0; t < 16; t++) {
      var tx = SX + (t + 0.5) * SW / 16;
      S('line', { x1: tx, y1: SY - 7, x2: tx, y2: SY + SH + 7, stroke: C.ver, 'stroke-width': 1.6, 'data-on': 1, 'data-off': 4 }, M);
    }
    S('rect', { x: SX + 0.51 * SW - 3, y: SY - 2, width: 7, height: SH + 4, fill: 'none', stroke: C.yellow, 'stroke-width': 2.4 }, M);

    var CW = 76, CH = 33, CX = [294, 380, 466, 552];
    var R1 = 150, R2 = 246;
    txt(M, 286, R1 - 13, sub('<tspan font-style="italic">w</tspan>', 'I' + DASH + 'IV'), { class: 'ov-lab', 'data-on': 4 });
    for (var c = 0; c < 4; c++) {
      S('line', { x1: SX + (c + 0.5) * SW / 4, y1: SY + SH + 4, x2: CX[c] + CW / 2, y2: R1 - 4, stroke: '#bbb', 'stroke-width': 1, 'data-on': 4 }, M);
      film(M, CX[c], R1, CW, CH, base + 'w1_' + c + '.webp', 4);
    }
    S('rect', { x: CX[2] + 5, y: R1 - 2, width: 7, height: CH + 4, fill: 'none', stroke: C.yellow, 'stroke-width': 2.2, 'data-on': 4 }, M);
    S('rect', { x: CX[1] - 6, y: R1 - 7, width: CW + 12, height: CH + 14, rx: 5, fill: 'none', stroke: '#9a9a9a', 'stroke-width': 2, 'data-on': 5 }, M);
    // step arrows
    S('path', { d: 'M' + (CX[1] + CW / 2) + ' ' + (SY + SH + 6) + ' V ' + (R1 - 9), stroke: C.green, 'stroke-width': 2.4, fill: 'none', 'marker-end': 'url(#ah-g)', class: 'draw', 'data-on': 4 }, M);
    badge(M, CX[1] + CW / 2 - 14, 122, 1, 4);
    S('path', { d: 'M' + (CX[1] + CW / 2) + ' ' + (R1 - 8) + ' Q ' + (CX[2] - 5) + ' ' + (R1 - 44) + ' ' + (CX[2] + CW / 2 - 4) + ' ' + (R1 - 9),
                stroke: C.green, 'stroke-width': 2.4, fill: 'none', 'marker-end': 'url(#ah-g)', class: 'draw', 'data-on': 5 }, M);
    badge(M, CX[2] + CW / 2 + 12, 124, 2, 5);
    // level 2 under w_III
    S('path', { d: 'M' + CX[2] + ' ' + (R1 + CH + 4) + ' L ' + (CX[2] + CW) + ' ' + (R1 + CH + 4) + ' L ' + (CX[3] + CW) + ' ' + (R2 - 8) + ' L ' + CX[0] + ' ' + (R2 - 8) + ' Z',
                fill: '#e4e6ea', opacity: 0.8, 'data-on': 6 }, M);
    txt(M, 286, R2 - 12, sub('<tspan font-style="italic">w</tspan>', 'III, I' + DASH + 'IV'), { class: 'ov-lab', 'data-on': 6 });
    for (c = 0; c < 4; c++) film(M, CX[c], R2, CW, CH, base + 'w2_' + c + '.webp', 6);
    S('rect', { x: CX[1] + 5, y: R2 - 2, width: 7, height: CH + 4, fill: 'none', stroke: C.yellow, 'stroke-width': 2.2, 'data-on': 6 }, M);
    S('path', { d: 'M' + (CX[2] + 18) + ' ' + (R1 + CH + 6) + ' L ' + (CX[1] + CW / 2 + 6) + ' ' + (R2 - 9),
                stroke: C.green, 'stroke-width': 2.4, fill: 'none', 'marker-end': 'url(#ah-g)', class: 'draw', 'data-on': 6 }, M);
    badge(M, CX[1] + CW / 2 + 22, R2 - 34, 3, 6);
    S('path', { d: 'M' + (CX[1] + CW / 2 + 28) + ' ' + (R2 - 8) + ' L ' + (CX[2] + 44) + ' ' + (R1 + CH + 7),
                stroke: C.ctrl, 'stroke-width': 2, 'stroke-dasharray': '5 4', fill: 'none', 'marker-end': 'url(#ah-b)', 'data-on': 6 }, M);
    txt(M, CX[2] + 46, R2 - 22, 'Up', { class: 'ov-lab', fill: C.ctrl, 'font-weight': 600, 'data-on': 6 });
    // the active window: one outline that moves
    var act = S('rect', { x: 0, y: 0, width: CW + 12, height: CH + 14, rx: 5, fill: 'none', stroke: C.green, 'stroke-width': 3, class: 'mover', 'data-on': 4 }, M);
    // Inspect: frames picked in w_III,II and kept as F_t
    var ftx = 336;
    for (var q = 0; q < 4; q++) {
      var px = CX[1] + 6 + q * 3.2;
      S('line', { x1: px, y1: R2 - 1, x2: px, y2: R2 + CH + 1, stroke: C.ver, 'stroke-width': 1.6, 'data-on': 7 }, M);
      S('path', { d: 'M' + px + ' ' + (R2 + CH + 6) + ' V ' + (R2 + CH + 16) + ' H ' + (ftx + q * 60 + 27) + ' V ' + 322,
                  stroke: C.ver, 'stroke-width': 1.2, fill: 'none', class: 'draw', 'data-on': 7 }, M);
      var g = S('g', { 'data-on': 7, class: 'pop' }, M);
      g.style.transitionDelay = (300 + q * 110) + 'ms';
      S('image', { href: base + 'ft' + q + '.webp', x: ftx + q * 60, y: 326, width: 54, height: 41, preserveAspectRatio: 'none' }, g);
      S('rect', { x: ftx + q * 60 - 1, y: 325, width: 56, height: 43, fill: 'none', stroke: C.ver, 'stroke-width': 2.5 }, g);
    }
    badge(M, CX[1] - 14, R2 + CH + 16, 4, 7);
    txt(M, ftx + 246, 352, sub('<tspan font-style="italic">F</tspan>', 't'), { class: 'ov-sym', 'data-on': 7 });

    // ---- agents
    var A = S('g', {}, svg);
    function agent(y, h, name, col, fill, lit) {
      var g = S('g', { class: 'agent', 'data-lit': lit }, A);
      S('rect', { x: 652, y: y, width: 110, height: h, rx: 7, fill: fill, stroke: col, 'stroke-width': 2 }, g);
      txt(g, 707, y + h / 2 + 5, name, { class: 'ov-agent', fill: col, 'text-anchor': 'middle' });
      return g;
    }
    agent(58, 32, 'Controller', C.ctrl, '#fff', '4,5,6,7');
    agent(112, 27, 'Locator', C.loc, '#e8f6f1', '4,5,6');
    agent(139, 27, 'Selector', C.sel, '#fbf4e2', '7');
    agent(190, 46, 'Verifier', C.ver, '#fdf1e8', '4,5,6,7');
    S('path', { d: 'M707 91 V 108', stroke: '#555', 'stroke-width': 1.5, 'marker-end': 'url(#ah-k)' }, A);
    S('path', { d: 'M707 167 V 186', stroke: '#555', 'stroke-width': 1.5, 'marker-end': 'url(#ah-k)' }, A);
    S('path', { d: 'M652 213 H 642 V 74 H 648', stroke: C.ver, 'stroke-width': 1.5, fill: 'none', 'marker-end': 'url(#ah-o)' }, A);

    // ---- decision table (one column per step)
    var T = S('g', {}, svg);
    var cols = [['Refine', sub('w', 'II'), DASH, ['Evidence', 'AFTER', 'window']],
                ['Switch', sub('w', 'III'), DASH, ['Evidence', 'IN', 'window']],
                ['Refine', sub('w', 'III,II'), DASH, ['Evidence', 'IN', 'window']],
                ['Inspect', DASH, 'Select 4', ['Keep all']]];
    cols.forEach(function (cd, k) {
      var x = 774 + k * 55, cx = x + 25, g = S('g', { 'data-on': 4 + k, class: 'tcol', 'data-lit': '' + (4 + k) }, T);
      S('circle', { cx: cx, cy: 52, r: 9, fill: C.ctrl }, g);
      txt(g, cx, 56, '' + (k + 1), { class: 'ov-num', 'text-anchor': 'middle' });
      S('rect', { x: x, y: 66, width: 50, height: 24, rx: 4, fill: '#dcebf8' }, g);
      txt(g, cx, 82, cd[0], { class: 'ov-mono', fill: C.ctrl, 'text-anchor': 'middle', 'font-weight': 700 });
      S('rect', { x: x, y: 112, width: 50, height: 27, fill: '#dff3ec' }, g);
      txt(g, cx, 130, cd[1], { class: 'ov-mono', fill: '#0f6e5a', 'text-anchor': 'middle' });
      S('rect', { x: x, y: 139, width: 50, height: 27, fill: '#fbf1d6' }, g);
      txt(g, cx, 157, cd[2], { class: 'ov-mono', fill: '#8a6400', 'text-anchor': 'middle' });
      S('rect', { x: x, y: 190, width: 50, height: 62, rx: 4, fill: '#fde6d6' }, g);
      var lines = cd[3], y0 = 221 - (lines.length - 1) * 6.5;
      lines.forEach(function (s, j) { txt(g, cx, y0 + j * 13, s, { class: 'ov-mono', fill: '#a84a12', 'text-anchor': 'middle' }); });
      S('rect', { x: x - 3, y: 40, width: 56, height: 216, rx: 6, fill: 'none', class: 'tcol-ring' }, g);
    });
    txt(T, 774, 276, 'repeat until |' + sub('<tspan font-style="italic">F</tspan>', 't') + '| = ' + sub('<tspan font-style="italic">K</tspan>', 'd'), { class: 'ov-lab', fill: C.muted, 'data-on': 7 });
    txt(T, 774, 300, 'yellow: where the evidence lies', { class: 'ov-lab ov-small', fill: C.muted });

    // ---- System II
    var B = S('g', {}, svg);
    S('path', { d: 'M138 414 V 440 H 330 M 459 372 V 440 H 330 M 330 440 V 466', stroke: '#555', 'stroke-width': 1.8, fill: 'none', class: 'draw', 'data-on': 8 }, B);
    S('path', { d: 'M330 452 V 466', stroke: '#555', 'stroke-width': 1.8, 'marker-end': 'url(#ah-k)', 'data-on': 8 }, B);
    var s2 = S('g', { class: 'agent', 'data-lit': '8' }, B);
    S('rect', { x: 60, y: 470, width: 540, height: 44, rx: 9, fill: '#fbeef5', stroke: C.sys2, 'stroke-width': 2 }, s2);
    txt(s2, 318, 498, 'System II ❄', { class: 'ov-agent', fill: C.sys2, 'text-anchor': 'end' });
    txt(s2, 334, 498, 'frozen MLLM', { class: 'ov-lab', 'font-style': 'italic', 'font-size': 15 });
    S('path', { d: 'M602 492 H 690', stroke: '#555', 'stroke-width': 1.8, 'marker-end': 'url(#ah-k)', class: 'draw', 'data-on': 8 }, B);
    var ans = S('g', { 'data-on': 8, class: 'pop late' }, B);
    S('rect', { x: 696, y: 472, width: 128, height: 40, rx: 9, fill: '#fff', stroke: '#555', 'stroke-width': 1.6 }, ans);
    txt(ans, 760, 498, 'answer: (A)', { class: 'ov-ans', 'text-anchor': 'middle' });

    function badge(g, cx, cy, n, on) {
      var b = S('g', { 'data-on': on }, g);
      S('circle', { cx: cx, cy: cy, r: 9, fill: C.ctrl }, b);
      txt(b, cx, cy + 4, '' + n, { class: 'ov-num', 'text-anchor': 'middle' });
    }

    var I = '<i>', i_ = '</i>';
    var CAPS = [
      'The search starts from the whole video. (In this illustration, yellow marks where the evidence lies.)',
      'System I first looks at ' + I + 'K' + i_ + '<sub>o</sub> uniform overview frames ' + I + 'F' + i_ + '<sub>o</sub>.',
      'The <span class="sc">Verifier</span> judges whether each overview frame shows evidence.',
      'The top ' + I + 'K' + i_ + '<sub>a</sub> overview frames are selected as ' + I + 'F' + i_ + '<sub>a</sub>.',
      'Step 1, <code>Refine</code>: the <span class="sc">Controller</span> picks the action, the <span class="sc">Locator</span> picks ' + I + 'w' + i_ + '<sub>II</sub>, and the <span class="sc">Verifier</span> judges that the evidence lies after this window.',
      'Step 2, <code>Switch</code>: the <span class="sc">Locator</span> moves to the sibling ' + I + 'w' + i_ + '<sub>III</sub>; the <span class="sc">Verifier</span> judges that the evidence is in this window.',
      'Step 3, <code>Refine</code>: the <span class="sc">Locator</span> picks ' + I + 'w' + i_ + '<sub>III,II</sub> among the children of ' + I + 'w' + i_ + '<sub>III</sub>. <code>Up</code> (dashed) would return to the parent.',
      'Step 4, <code>Inspect</code>: the <span class="sc">Selector</span> picks 4 frames in the active window and the <span class="sc">Verifier</span> keeps all of them in ' + I + 'F' + i_ + '<sub>t</sub>. The loop repeats until |' + I + 'F' + i_ + '<sub>t</sub>| = ' + I + 'K' + i_ + '<sub>d</sub>.',
      'After the search, System II, a frozen MLLM, answers once from ' + I + 'F' + i_ + '<sub>a</sub> and ' + I + 'F' + i_ + '<sub>t</sub>.'
    ];
    var NAMES = ['Start', 'Overview frames', 'Verifier', 'Selected overview frames', 'Step 1: Refine', 'Step 2: Switch',
                 'Step 3: Refine', 'Step 4: Inspect', 'System II answers'];
    var ACT = { 4: CX[1], 5: CX[2], 6: CX[1] };
    prepDraw(svg);
    var litEls = svg.querySelectorAll('[data-lit]');

    function render(s, anim) {
      svg.classList.toggle('still', !anim);
      showByState(svg, s);
      for (var i = 0; i < litEls.length; i++) {
        var on = litEls[i].getAttribute('data-lit').split(',').map(Number);
        litEls[i].classList.toggle('lit', on.indexOf(s) >= 0);
      }
      var row = s >= 6 ? R2 : R1, cx = s >= 6 ? CX[1] : (ACT[s] != null ? ACT[s] : CX[1]);
      tween(act, { x: cx - 6, y: row - 7 }, anim ? 520 : 0);
      cap.innerHTML = CAPS[s];
    }
    var player = new Player(host, 'Overview animation');
    player.configure(CAPS.length, render, function (s) { return [1300, 1700, 1500, 1700, 2300, 2300, 2500, 2900][s] || 2000; }, NAMES);
  }

  /* ---------------- search replays (App. E) ---------------- */
  var ACTNAME = { REFINE_LOCAL: 'Refine', SWITCH_LOCAL: 'Switch', UP: 'Up', INSPECT_FRAMES: 'Inspect' };

  function quarters(w) {
    var L = (w[1] - w[0]) / 4, r = [];
    for (var k = 0; k < 4; k++) r.push([w[0] + k * L, w[0] + (k + 1) * L]);
    return r;
  }
  function same(u, v) {
    var tol = Math.max(0.05, 1e-4 * (v[1] - v[0]));
    return Math.abs(u[0] - v[0]) < tol && Math.abs(u[1] - v[1]) < tol;
  }
  function meets(w, E) { return E.some(function (e) { return Math.min(w[1], e[1]) > Math.max(w[0], e[0]); }); }

  /* replay the controller actions on the window tree (same rules as the paper's figure script) */
  function buildSearch(sp) {
    var root = { w: [0, sp.T], lv: 0, parent: null, kids: null };
    function kids(n) {
      if (!n.kids) n.kids = quarters(n.w).map(function (w) { return { w: w, lv: n.lv + 1, parent: n, kids: null }; });
      return n.kids;
    }
    function child(p, w) {
      var k = kids(p).filter(function (c) { return same(w, c.w); });
      if (!k.length) throw new Error('window ' + fmt(w) + ' is not a child of ' + fmt(p.w));
      if (!k[0].kids) k[0].w = w.slice();   // the recorded window (decoded length), as the paper's figure script uses
      return k[0];
    }
    var win = sp.searched.slice(), cur = root, raw = [];
    sp.actions.forEach(function (a, i) {
      var src = cur, dst;
      if (a === 'REFINE_LOCAL') dst = child(cur, win.shift());
      else if (a === 'SWITCH_LOCAL') dst = child(cur.parent, win.shift());
      else if (a === 'UP') dst = cur.parent;
      else dst = cur;
      raw.push({ k: i + 1, a: a, src: src, dst: dst });
      cur = dst;
    });
    // display steps: consecutive inspections are one step, as in the paper's step list
    var steps = [];
    raw.forEach(function (r) {
      var last = steps[steps.length - 1];
      if (last && last.a === 'INSPECT_FRAMES' && r.a === 'INSPECT_FRAMES') last.to = r.k;
      else steps.push({ from: r.k, to: r.k, a: r.a, src: r.src, dst: r.dst });
    });
    steps.forEach(function (st, j) { st.state = j + 1; });
    // groups (children of a parent) appear when the search first enters one of them
    var groups = [], seen = new Map(), visit = new Map([[root, 0]]);
    steps.forEach(function (st) {
      if (st.a === 'REFINE_LOCAL' || st.a === 'SWITCH_LOCAL') {
        var p = st.dst.parent;
        if (!seen.has(p)) { seen.set(p, st.state); groups.push({ p: p, state: st.state }); }
      }
      if (!visit.has(st.dst)) visit.set(st.dst, st.state);
    });
    var depth = groups.reduce(function (m, g) { return Math.max(m, g.p.lv + 1); }, 0);
    var E = sp.E, evLevel = 0;
    for (var lv = depth; lv > 0 && !evLevel; lv--) {
      groups.forEach(function (g) { if (g.p.lv + 1 === lv && kids(g.p).some(function (c) { return meets(c.w, E); })) evLevel = lv; });
    }
    return { root: root, steps: steps, groups: groups, depth: depth, visit: visit, evLevel: evLevel, kids: kids };
  }

  function stepText(st) {
    var n = st.from === st.to ? 'Step ' + st.from : 'Steps ' + st.from + DASH + st.to;
    var w = fmt(st.dst.w) + '&thinsp;s';
    if (st.a === 'REFINE_LOCAL') return n + ', <code>Refine</code>: the <span class="sc">Locator</span> picks ' + w + ', a child of ' + fmt(st.src.w) + '&thinsp;s.';
    if (st.a === 'SWITCH_LOCAL') return n + ', <code>Switch</code>: the <span class="sc">Locator</span> moves from ' + fmt(st.src.w) + '&thinsp;s to its sibling ' + w + '.';
    if (st.a === 'UP') return n + ', <code>Up</code>: back to the parent window ' + w + '.';
    return n + ', <code>Inspect</code>: the <span class="sc">Selector</span> picks frames in ' + w + ' and the <span class="sc">Verifier</span> keeps those that show the evidence; the search ends with 8 kept frames.';
  }

  function buildExamples(host) {
    var DATA = window.LOOPSEEK_EXAMPLES;
    if (!DATA) return;
    var texts = host.querySelectorAll('.ex-text');
    var keys = ['direct', 'switch', 'up'], LABEL = { direct: 'Direct narrowing', 'switch': 'Recovery with Switch', up: 'Recovery with Up' };
    var tabs = H('div', 'seg', null, null);
    tabs.setAttribute('role', 'group'); tabs.setAttribute('aria-label', 'Search example');
    host.insertBefore(tabs, host.firstChild);
    var stage = H('div', 'ex-stage', null, null);
    host.insertBefore(stage, tabs.nextSibling);
    var grid = H('div', 'ex-grid', null, stage);
    var left = H('div', 'ex-left', null, grid), right = H('div', 'ex-right', null, grid);
    var qcard = H('div', 'ex-q', null, left);
    var slist = H('ol', 'ex-steps', null, left);
    var answers = H('div', 'ex-ans', null, left);
    var cap = H('p', 'anim-cap ex-cap', '', right);
    cap.setAttribute('aria-live', 'polite');
    var treeWrap = H('div', 'anim-scroll', null, right);
    var tl = S('svg', { viewBox: '0 0 640 92', class: 'ex-tl', role: 'img', 'aria-label': 'Timeline of the video with the active window' }, treeWrap);
    var tree = S('svg', { class: 'ex-tree', role: 'img', 'aria-label': 'Window tree of the search' }, treeWrap);
    var keptHead = H('p', 'ex-kept-head', '', stage);
    var kept = H('div', 'ex-kept', null, stage);
    var player = new Player(stage, 'Search replay');
    var btns = {};
    keys.forEach(function (k) {
      var b = H('button', 'seg-btn', LABEL[k].replace(/(Switch|Up)$/, '<code>$1</code>'), tabs);
      b.type = 'button';
      b.onclick = function () { pick(k, true); };
      btns[k] = b;
    });

    function pick(key, play) {
      keys.forEach(function (k) { btns[k].classList.toggle('on', k === key); btns[k].setAttribute('aria-pressed', k === key); });
      for (var i = 0; i < texts.length; i++) texts[i].hidden = texts[i].getAttribute('data-key') !== key;
      var sp = DATA[key], X = buildSearch(sp), T = sp.T, E = sp.E, W = 640;
      var N = X.steps.length, last = N + 1;     // states: 0 start, 1..N steps, N+1 answer

      // question and options
      qcard.innerHTML = '<div class="ex-lab">Question</div><p>' + sp.question.replace(/&/g, '&amp;').replace(/</g, '&lt;') + '</p>';
      var ol = H('ul', 'ex-opts', null, qcard);
      Object.keys(sp.options).forEach(function (o) {
        var li = H('li', o === sp.gt ? 'gt' : '', '(' + o + ') ' + sp.options[o].replace(/&/g, '&amp;').replace(/</g, '&lt;'), ol);
        li.setAttribute('data-letter', o);
      });
      // step list
      slist.innerHTML = '';
      var lis = X.steps.map(function (st) {
        var n = st.from === st.to ? st.from : st.from + DASH + st.to;
        var where = st.a === 'INSPECT_FRAMES' ? '8 frames kept' : fmt(st.dst.w) + '&thinsp;s';
        return H('li', 'a-' + ACTNAME[st.a].toLowerCase(), '<b>' + n + '</b><code>' + ACTNAME[st.a] + '</code><span>' + where + '</span>', slist);
      });
      answers.innerHTML = '<div class="ex-lab">System II answers</div>' +
        '<div class="row"><span>LoopSeek, all 32 frames</span><b>' + sp.ours + '</b><i class="' + (sp.ours_ok ? 'ok' : 'no') + '">' + (sp.ours_ok ? '&#10003;' : '&#10007;') + '</i></div>' +
        '<div class="row"><span>24 overview frames only</span><b>' + sp.t24 + '</b><i class="' + (sp.t24_ok ? 'ok' : 'no') + '">' + (sp.t24_ok ? '&#10003;' : '&#10007;') + '</i></div>';

      // timeline: whole video, active window, zoomed active window
      tl.innerHTML = '';
      var X1 = 6, X2 = 634, TW = X2 - X1, sx = function (t) { return X1 + TW * t / T; };
      txt(tl, X1, 11, 'Whole video', { class: 'tl-lab' });
      txt(tl, X2, 11, 'annotated evidence (not seen by the search)', { class: 'tl-lab tl-ev', 'text-anchor': 'end' });
      S('rect', { x: X1, y: 17, width: TW, height: 10, rx: 5, class: 'tl-bar' }, tl);
      E.forEach(function (e) { S('rect', { x: sx(e[0]) - 1, y: 15, width: Math.max(3, sx(e[1]) - sx(e[0])), height: 14, class: 'tl-evr' }, tl); });
      txt(tl, X1, 40, '0&thinsp;s', { class: 'tl-ax' });
      txt(tl, X2, 40, T.toFixed(0) + '&thinsp;s', { class: 'tl-ax', 'text-anchor': 'end' });
      var wr = S('rect', { x: X1, y: 13, width: TW, height: 18, rx: 3, class: 'tl-win' }, tl);
      var c1 = S('line', { x1: X1, y1: 31, x2: X1, y2: 66, class: 'tl-zl' }, tl);
      var c2 = S('line', { x1: X2, y1: 31, x2: X2, y2: 66, class: 'tl-zl' }, tl);
      S('rect', { x: X1, y: 66, width: TW, height: 12, rx: 6, class: 'tl-bar tl-zoom' }, tl);
      var zev = S('rect', { x: X1, y: 66, width: 0, height: 12, class: 'tl-evr' }, tl);
      var zl = txt(tl, X1, 90, '', { class: 'tl-ax' });
      var zr = txt(tl, X2, 90, '', { class: 'tl-ax', 'text-anchor': 'end' });
      var zc = txt(tl, (X1 + X2) / 2, 90, '', { class: 'tl-lab', 'text-anchor': 'middle' });

      // tree layout
      tree.innerHTML = '';
      var defs = S('defs', {}, tree);
      var mk = S('marker', { id: 'ah-ex-' + key, viewBox: '0 0 10 10', refX: 8, refY: 5, markerWidth: 6, markerHeight: 6, orient: 'auto-start-reverse' }, defs);
      S('path', { d: 'M0 0L10 5L0 10z', fill: '#1f5fd1' }, mk);
      var PITCH = 66, BH = 28, TOP = 6, rows = {}, box = new Map();
      var Hh = TOP + X.depth * PITCH + BH + 26;
      tree.setAttribute('viewBox', '0 0 ' + W + ' ' + Hh);
      box.set(X.root, { x: W / 2 - 75, y: TOP, w: 150 });
      for (var lv = 1; lv <= X.depth; lv++) {
        var gs = X.groups.filter(function (g) { return g.p.lv + 1 === lv; }).sort(function (a, b) { return a.p.w[0] - b.p.w[0]; });
        var GG = 18, gw = (W - (gs.length - 1) * GG) / gs.length, bw = (gw - 3 * 6) / 4;
        gs.forEach(function (g, gi) {
          X.kids(g.p).forEach(function (c, ci) { box.set(c, { x: gi * (gw + GG) + ci * (bw + 6), y: TOP + lv * PITCH, w: bw, g: g }); });
        });
      }
      var edgeG = S('g', {}, tree), boxG = S('g', {}, tree), arrG = S('g', {}, tree);
      var nodes = [];
      box.forEach(function (b, n) {
        var on = n === X.root ? 0 : b.g.state;
        if (n !== X.root) {
          var pb = box.get(n.parent);
          S('line', { x1: pb.x + pb.w / 2, y1: pb.y + BH, x2: b.x + b.w / 2, y2: b.y, class: 'tr-edge', 'data-on': on }, edgeG);
        }
        var g = S('g', { class: 'tr-box', 'data-on': on }, boxG);
        var r = S('rect', { x: b.x, y: b.y, width: b.w, height: BH, rx: 4 }, g);
        var label = fmt(n.w), fs = b.w >= 96 ? 12 : b.w >= 66 ? 10.5 : 0;
        var t = txt(g, b.x + b.w / 2, b.y + BH / 2 + (fs ? fs * 0.36 : 4), fs ? label : '&hellip;', { 'text-anchor': 'middle', 'font-size': fs || 12 });
        S('title', {}, g).textContent = label + ' s';
        var ev = X.evLevel && n.lv === X.evLevel && meets(n.w, E);
        if (ev) {
          var f = txt(g, b.x + b.w / 2, b.y + BH + 16, '&#9654; Evidence', { 'text-anchor': 'middle', class: 'tr-flag', 'data-on': last });
        }
        nodes.push({ n: n, g: g, ev: ev, vis: X.visit.has(n) ? X.visit.get(n) : Infinity });
      });
      // path arrows
      X.steps.forEach(function (st) {
        if (st.a === 'INSPECT_FRAMES') return;
        var a = box.get(st.src), b = box.get(st.dst), d;
        if (st.a === 'REFINE_LOCAL') {
          // exactly on the gray edge (parent bottom-centre to child top-centre); stop 2 px early for the arrow head
          var ex1 = a.x + a.w / 2, ey1 = a.y + BH, ex2 = b.x + b.w / 2, ey2 = b.y, el = Math.hypot(ex2 - ex1, ey2 - ey1);
          d = 'M' + ex1 + ' ' + ey1 + ' L ' + (ex2 - 2 * (ex2 - ex1) / el) + ' ' + (ey2 - 2 * (ey2 - ey1) / el);
        } else if (st.a === 'SWITCH_LOCAL') {
          // as in the paper: a switch to the right runs under the row, a switch to the left over it
          var right = b.x > a.x, x1 = a.x + a.w / 2 + (right ? 14 : -14), x2 = b.x + b.w / 2 + (right ? -14 : 14);
          var y = right ? a.y + BH + 2 : a.y - 2, bend = right ? 44 : -40;
          d = 'M' + x1 + ' ' + y + ' Q ' + (x1 + x2) / 2 + ' ' + (y + bend) + ' ' + x2 + ' ' + (y + (right ? 1 : -1));
        } else {
          var ux1 = a.x + a.w / 2 - 10, uy1 = a.y - 2, ux2 = b.x + b.w / 2 - 12, uy2 = b.y + BH + 3;
          d = 'M' + ux1 + ' ' + uy1 + ' Q ' + (Math.min(ux1, ux2) - 26) + ' ' + (uy1 + uy2) / 2 + ' ' + ux2 + ' ' + uy2;
        }
        var p = S('path', { d: d, class: st.a === 'UP' ? 'tr-arrow tr-up' : 'tr-arrow draw', 'marker-end': 'url(#ah-ex-' + key + ')', 'data-on': st.state }, arrG);
        st.path = p;
      });
      prepDraw(tree);
      var placed = [];
      X.steps.forEach(function (st) {        // number badges on each arrow, moved along it until clear of the others
        if (!st.path) return;
        var Lp = st.path.getTotalLength(), m = null;
        [0.5, 0.38, 0.62, 0.28, 0.72, 0.2, 0.8].some(function (f) {
          var q = st.path.getPointAtLength(Lp * f);
          if (!m) m = q;
          if (placed.every(function (o) { return Math.hypot(o.x - q.x, o.y - q.y) >= 22; })) { m = q; return true; }
          return false;
        });
        placed.push(m);
        var bg = S('g', { class: 'tr-badge', 'data-on': st.state }, arrG);
        S('circle', { cx: m.x, cy: m.y, r: 8.5 }, bg);
        txt(bg, m.x, m.y + 3.8, '' + st.from, { 'text-anchor': 'middle' });
      });
      var insp = X.steps[N - 1];
      if (insp.a === 'INSPECT_FRAMES') {
        var ib = box.get(insp.dst);
        var tag = S('g', { class: 'tr-badge tr-insp', 'data-on': insp.state }, arrG);
        var tx = ib.x + ib.w / 2, ty = ib.y + BH + 16, lab = insp.from + DASH + insp.to + ' Inspect';
        S('rect', { x: tx - 46, y: ty - 10, width: 92, height: 18, rx: 9 }, tag);
        txt(tag, tx, ty + 3.5, lab, { 'text-anchor': 'middle' });
      }

      // kept frames
      keptHead.innerHTML = 'The 8 frames kept by the search <span>(red border: inside the annotated evidence)</span>';
      keptHead.setAttribute('data-on', N);
      kept.innerHTML = '';
      kept.style.setProperty('--ar', 1 / sp.aspect);
      sp.kept.forEach(function (t, i) {
        var inE = E.some(function (e) { return t >= e[0] && t <= e[1]; });
        var f = H('figure', 'kf' + (inE ? ' in' : ''), null, kept);
        f.setAttribute('data-on', N);
        f.style.transitionDelay = (i * 70) + 'ms';
        var im = H('img', null, null, f);
        im.src = R(sp.frames[i]); im.alt = 'Kept frame at ' + t.toFixed(1) + ' s'; im.loading = 'lazy';
        H('figcaption', null, t.toFixed(1) + '&thinsp;s', f);
      });

      var NAMES = ['Start'].concat(X.steps.map(function (st) { return (st.from === st.to ? 'Step ' + st.from : 'Steps ' + st.from + DASH + st.to) + ': ' + ACTNAME[st.a]; }), ['System II answers']);
      function cur(s) { return s === 0 ? X.root : X.steps[Math.min(s, N) - 1].dst; }

      function render(s, anim) {
        stage.classList.toggle('still', !anim);
        showByState(stage, s);
        var c = cur(s);
        nodes.forEach(function (o) {
          o.g.classList.toggle('vis', o.vis <= s);
          o.g.classList.toggle('cur', o.n === c && s < last);
          o.g.classList.toggle('ev', o.ev && s >= last);
        });
        lis.forEach(function (li, i) {
          li.classList.toggle('off', i + 1 > s);
          li.classList.toggle('cur', i + 1 === s);
        });
        qcard.classList.toggle('reveal', s >= last);
        answers.classList.toggle('off', s < last);
        var w = c.w, ms = anim ? 650 : 0;
        tween(wr, { x: sx(w[0]) - 1.5, width: Math.max(4, sx(w[1]) - sx(w[0]) + 3) }, ms);
        tween(c1, { x1: sx(w[0]) }, ms); tween(c2, { x1: sx(w[1]) }, ms);
        var ov = E.map(function (e) { return [Math.max(e[0], w[0]), Math.min(e[1], w[1])]; }).filter(function (o) { return o[1] > o[0]; });
        var zx = function (t) { return X1 + TW * (t - w[0]) / (w[1] - w[0]); };
        if (ov.length) tween(zev, { x: zx(ov[0][0]), width: Math.max(3, zx(ov[0][1]) - zx(ov[0][0])) }, ms);
        else tween(zev, { x: X1 + TW / 2, width: 0 }, ms);
        zl.innerHTML = w[0].toFixed(0) + '&thinsp;s'; zr.innerHTML = w[1].toFixed(0) + '&thinsp;s';
        var share = ov.reduce(function (m, o) { return m + o[1] - o[0]; }, 0) / (w[1] - w[0]);
        zc.innerHTML = 'Active window, ' + (w[1] - w[0]).toFixed(0) + '&thinsp;s' +
          (ov.length ? ' &middot; evidence covers ' + (share * 100 >= 10 ? (share * 100).toFixed(0) : (share * 100).toFixed(1)) + '%' : ' &middot; no evidence');
        if (s === 0) cap.innerHTML = 'Start: the whole video, ' + fmt(sp.root || [0, T]) + '&thinsp;s. The search does not see the annotated evidence (red).';
        else if (s <= N) cap.innerHTML = stepText(X.steps[s - 1]);
        else cap.innerHTML = 'System II answers once: <b>(' + sp.ours + ')</b> from all 32 frames' + (sp.ours_ok ? ', correct' : '') +
          '; <b>(' + sp.t24 + ')</b> from the 24 overview frames alone' + (sp.t24_ok ? ', correct' : ', wrong') + '. The correct option is (' + sp.gt + ').';
      }
      player.configure(last + 1, render, function (s) {
        if (s === 0) return 1100;
        if (s === N - 1) return 1700;
        return s === N ? 2600 : 1550;
      }, NAMES);
      if (play && !RM) player.play();
    }
    pick('direct', false);
  }

  function init() {
    var ov = document.getElementById('ov-anim');
    if (ov) buildOverview(ov);
    var ex = document.getElementById('ex-player');
    if (ex) buildExamples(ex);
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
