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

// ===== i18n =====
// A deliberately tiny dictionary, not a library. Every user-facing string the chrome owns lives here
// in two languages; model output is never translated (it is the model's, not ours). t(key) reads the
// active language, applyLang() repaints the static [data-i18n] nodes plus the dynamic badge/transcript,
// and setLang() persists the choice. Default is English; the header toggle swaps instantly, no reload.
const I18N = {
  en: {
    langTitle: 'Switch language',
    newChat: 'New chat',
    newChatTitle: 'New chat (Ctrl+Shift+O)',
    badgeTitle: 'Engine status',
    emptyTitle: 'What can I help you with?',
    emptySub: 'Local inference · weights never leave this machine · no cloud dependency',
    chip1: 'Explain ternary quantization',
    chip2: 'Write some Vulkan code',
    chip3: 'Translate a sentence',
    chip1prompt: 'Explain in three sentences what ternary quantization is, and why it saves VRAM.',
    chip2prompt: 'Write a Python function that uses the Vulkan API to enumerate every physical device on this machine that supports compute, printing only the device name.',
    chip3prompt: 'Translate this sentence into English and give a more colloquial version: This graphics card costs only a hundred yuan, yet it can run a nine-billion-parameter large model.',
    inputPlaceholder: 'Message Bonsai…',
    inputAria: 'Message input',
    stopTitle: 'Stop generating',
    sendTitle: 'Send (Enter)',
    sendAria: 'Send',
    hintSend: 'to send',
    hintNewline: 'for a newline',
    ctxDefault: 'Content is generated by a local model; please verify it yourself.',
    thinking: 'Thinking',
    charsUnit: 'chars',
    stopped: 'Stopped',
    reasoning: 'Thought process',
    badgeOffline: '【Not connected | localhost:8080】',
    badgeOfflineTitle: 'Engine not ready: double-click bonsai_vk.exe to start it; this page opens automatically.',
    generating: 'generating',
    statusTimeout: 'Status request timed out',
    ctxPrefix: 'Context',
    ctxSuffix: 'generated locally, please verify',
    err429: 'The engine is generating another reply (this build runs one conversation at a time). ',
    err400: 'Request rejected: ',
    errOffline: 'Cannot reach the engine: ',
    errOfflineSuffix: ' (make sure bonsai_vk.exe is still running)',
    copied: 'Copied',
    copyFailed: 'Copy failed',
    copy: 'Copy',
    fileOffline: 'This page was opened straight from disk. Double-click bonsai_vk.exe so the engine serves it at localhost:8080.',
    advancedTitle: 'Advanced Settings',
    advTemp: 'Temperature',
    advPres: 'Presence Penalty',
    advRep: 'Repetition Penalty',
    advNote: 'Engine defaults 0.70 / 0.50 / 1.15 — sent with every request.'
  },
  zh: {
    langTitle: '切换语言',
    newChat: '新对话',
    newChatTitle: '开始新对话 (Ctrl+Shift+O)',
    badgeTitle: '引擎状态',
    emptyTitle: '有什么可以帮你的？',
    emptySub: '本地推理 · 权重不出这台机器 · 无云端依赖',
    chip1: '解释三值量化',
    chip2: '写一段 Vulkan 代码',
    chip3: '翻译一句话',
    chip1prompt: '用三句话解释什么是三值量化（ternary quantization），并说明它为什么省显存。',
    chip2prompt: '写一个 Python 函数，用 Vulkan 的 API 枚举本机所有支持 compute 的物理设备，只打印设备名。',
    chip3prompt: '把下面这句话翻译成英文，并给出一个更口语化的版本：这张显卡只要一百块钱，但它能跑九十亿参数的大模型。',
    inputPlaceholder: '给 Bonsai 发消息…',
    inputAria: '消息输入框',
    stopTitle: '停止生成',
    sendTitle: '发送 (Enter)',
    sendAria: '发送',
    hintSend: '发送',
    hintNewline: '换行',
    ctxDefault: '内容由本地模型生成，请自行核对',
    thinking: '正在思考',
    charsUnit: '字',
    stopped: '已停止',
    reasoning: '思考过程',
    badgeOffline: '【未连接 | localhost:8080】',
    badgeOfflineTitle: '引擎尚未就绪：双击 bonsai_vk.exe 启动后会自动打开本页',
    generating: '生成中',
    statusTimeout: '状态请求超时',
    ctxPrefix: '上下文',
    ctxSuffix: '内容本地生成，请自行核对',
    err429: '引擎正在生成另一条回复（本构建一次只跑一个对话）。',
    err400: '请求被拒绝：',
    errOffline: '连不上引擎：',
    errOfflineSuffix: '（确认 bonsai_vk.exe 还在运行）',
    copied: '已复制',
    copyFailed: '复制失败',
    copy: '复制',
    fileOffline: '本页是从磁盘直接打开的。请双击 bonsai_vk.exe，由引擎在 localhost:8080 上提供本页。',
    advancedTitle: '高级设置',
    advTemp: '温度 Temperature',
    advPres: '存在惩罚 Presence Penalty',
    advRep: '重复惩罚 Repetition Penalty',
    advNote: '引擎默认值 0.70 / 0.50 / 1.15 —— 随每次请求发送。'
  }
};

let LANG = 'en';

function i18n(key) {
  const d = I18N[LANG] || I18N.en;
  if (Object.prototype.hasOwnProperty.call(d, key)) return d[key];
  if (Object.prototype.hasOwnProperty.call(I18N.en, key)) return I18N.en[key];
  return key;
}

function getLang() { return LANG; }

// Repaint every node the chrome owns. Static text/placeholder/title/aria/prompt come from data-i18n*
// attributes; the dynamic badge, context note and transcript are re-rendered through their own real
// entry points so a language switch never diverges from what a fresh render would produce.
function applyLang() {
  document.documentElement.lang = (LANG === 'zh') ? 'zh-CN' : 'en';
  const nodes = document.querySelectorAll('[data-i18n]');
  for (let i = 0; i < nodes.length; i++) nodes[i].textContent = i18n(nodes[i].getAttribute('data-i18n'));
  const ph = document.querySelectorAll('[data-i18n-ph]');
  for (let i = 0; i < ph.length; i++) ph[i].setAttribute('placeholder', i18n(ph[i].getAttribute('data-i18n-ph')));
  const ti = document.querySelectorAll('[data-i18n-title]');
  for (let i = 0; i < ti.length; i++) ti[i].setAttribute('title', i18n(ti[i].getAttribute('data-i18n-title')));
  const ar = document.querySelectorAll('[data-i18n-aria]');
  for (let i = 0; i < ar.length; i++) ar[i].setAttribute('aria-label', i18n(ar[i].getAttribute('data-i18n-aria')));
  const pr = document.querySelectorAll('[data-i18n-prompt]');
  for (let i = 0; i < pr.length; i++) pr[i].setAttribute('data-prompt', i18n(pr[i].getAttribute('data-i18n-prompt')));
  const opts = document.querySelectorAll('#btn-lang .lang-opt');
  for (let i = 0; i < opts.length; i++) {
    const on = opts[i].getAttribute('data-lang') === LANG;
    opts[i].classList.toggle('active', on);
  }
  const lb = el('btn-lang');
  if (lb) { lb.title = i18n('langTitle'); lb.setAttribute('aria-label', i18n('langTitle')); }
  if (typeof setBadge === 'function') setBadge();
  if (typeof updateCtxNote === 'function') updateCtxNote();
  if (typeof renderAll === 'function') renderAll();
}

function setLang(lang) {
  LANG = (lang === 'zh') ? 'zh' : 'en';
  try { localStorage.setItem('bonsai-lang', LANG); } catch (err) { /* file:// or private mode: no persistence */ }
  applyLang();
}

function toggleLang() { setLang(LANG === 'zh' ? 'en' : 'zh'); }

function initLang() {
  let saved = 'en';
  try { const v = localStorage.getItem('bonsai-lang'); if (v === 'zh' || v === 'en') saved = v; } catch (err) { /* default English */ }
  LANG = saved;
  applyLang();
}