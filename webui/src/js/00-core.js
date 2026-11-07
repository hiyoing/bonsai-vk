// ===== 00-core.js =====
// Shared state and the two things every deterministic test needs to control: the clock and the
// random source. Nothing in this bundle calls Date.now() or Math.random() directly - they go through
// now() and rand(), both of which __APP__ can replace. That is what lets the self-test replay the
// exact same typewriter timing and the exact same mock stream twice and compare the results.

'use strict';

const LIMITS = {
  colPx: 768,          // the delivery contract's conversation column
  headerPx: 56,
  maxInputRows: 8,
  pumpMs: 24,          // typewriter tick; setTimeout, not rAF, so it also runs where rAF is not driven
  minCharsPerSec: 45,
  maxCharsPerSec: 900,
  drainSeconds: 0.18,  // a large backlog is drained in this long, so the UI never lags the model
  statusPollMs: 2000,
  maxTurns: 200,
  toastMs: 2600
};

const S = {
  turns: [],
  busy: false,
  online: false,
  status: null,
  statusError: '',
  seq: 0,
  seed: 20261108,
  clock: null,        // () => ms, injected by the self-test
  transport: null,    // injected by the self-test; null means the real fetch transport
  errors: [],
  abort: null,
  lastStatusAt: 0,
  pollTimer: null,
  requestsSent: 0,
  justFinished: null   // turn id awaiting the engine's measured tok/s from the next /v1/status
};

function now() {
  return S.clock ? S.clock() : performance.now();
}

// mulberry32: 32 bits of state, no dependency, and the same seed gives the same sequence on every
// engine. Used only for the mock stream's jitter in tests, never for anything the user sees.
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

let rng = mulberry32(S.seed);

function setSeed(n) {
  S.seed = n >>> 0;
  rng = mulberry32(S.seed);
}

function rand() {
  return rng();
}

function clamp(v, lo, hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

function el(id) {
  return document.getElementById(id);
}

function mk(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined && text !== null) n.textContent = text;
  return n;
}

function on(node, ev, fn, opts) {
  node.addEventListener(ev, fn, opts);
  return fn;
}

function clearNode(n) {
  while (n.firstChild) n.removeChild(n.firstChild);
}

// The only place a number becomes a display string. engine_kib comes from the server measuring its
// own executable, so the badge never carries a hardcoded size that a rebuild would silently falsify.
function fmtKib(kib) {
  return String(Math.round(Number(kib) || 0)) + ' KiB';
}

function fmtRate(v) {
  const n = Number(v);
  if (!isFinite(n) || n <= 0) return '';
  return n.toFixed(2) + ' tok/s';
}

function shortDevice(name) {
  const s = String(name || '').trim();
  if (!s) return 'GPU';
  return s.replace(/^AMD\s+/i, '').replace(/^Radeon\s+/i, '').replace(/\s*\(TM\)/gi, '').trim() || s;
}

let toastTimer = null;

function toast(msg) {
  const t = el('toast');
  if (!t) return;
  t.textContent = String(msg);
  t.hidden = false;
  if (toastTimer !== null) clearTimeout(toastTimer);
  toastTimer = setTimeout(function () {
    t.hidden = true;
    toastTimer = null;
  }, LIMITS.toastMs);
}

function hideToast() {
  const t = el('toast');
  if (t) t.hidden = true;
  if (toastTimer !== null) { clearTimeout(toastTimer); toastTimer = null; }
}

// A runtime error is data, not a console message the test runner might miss. Both window.onerror and
// unhandledrejection land here so runAllTests() can report a count instead of guessing.
function recordError(kind, detail) {
  S.errors.push({ kind: kind, detail: String(detail).slice(0, 400), at: now() });
  if (S.errors.length > 60) S.errors.shift();
}

function installErrorCapture() {
  const prev = window.onerror;
  window.onerror = function (msg, src, line, col, err) {
    recordError('window.onerror', (err && err.stack) || (msg + ' @' + line + ':' + col));
    if (typeof prev === 'function') return prev.apply(this, arguments);
    return false;
  };
  on(window, 'unhandledrejection', function (ev) {
    const r = ev.reason;
    recordError('unhandledrejection', (r && r.stack) || r);
  });
}

function nextId(prefix) {
  S.seq += 1;
  return prefix + '-' + S.seq;
}

function findTurn(id) {
  for (let i = 0; i < S.turns.length; i++) if (S.turns[i].id === id) return S.turns[i];
  return null;
}

function lastAssistant() {
  for (let i = S.turns.length - 1; i >= 0; i--) if (S.turns[i].role === 'assistant') return S.turns[i];
  return null;
}