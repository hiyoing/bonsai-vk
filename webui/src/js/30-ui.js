// ===== 30-ui.js =====
// Everything the user touches. The transcript lives in S.turns and the DOM is a projection of it, so
// a re-render never loses state and the self-test can assert on the data and on the pixels separately.
//
// The typewriter is a queue drained by a setTimeout pump rather than by requestAnimationFrame: the
// engine's own token callback already arrives in bursts, and a pump with an injectable clock can be
// stepped deterministically by __APP__.advance(ms). The reveal rate adapts to the backlog so the UI
// reads as typing at normal speed but never falls more than a fraction of a second behind the model.

'use strict';

const SVG_MARK = '<svg viewBox="0 0 48 48" width="13" height="13" aria-hidden="true" focusable="false">' +
  '<path d="M24 43V22M24 22c0-7 5-12 12-12 0 7-5 12-12 12Zm0 4c0-6-4.5-10.5-10.5-10.5C13.5 23.5 18 28 24 28Z" ' +
  'fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg>';

let pumpTimer = null;
let lastPump = 0;
let fracChars = 0;
let stickBottom = true;

function pendingChars() {
  let n = 0;
  for (let i = 0; i < S.turns.length; i++) {
    const t = S.turns[i];
    if (t.role !== 'assistant') continue;
    n += Math.max(0, t.text.length - t.shown);
    n += Math.max(0, t.reasoning.length - t.reasonShown);
  }
  return n;
}

function revealRate(backlog) {
  if (backlog <= 0) return 0;
  return clamp(backlog / LIMITS.drainSeconds, LIMITS.minCharsPerSec, LIMITS.maxCharsPerSec);
}

function pumpAdvance(dtMs) {
  const backlog = pendingChars();
  if (backlog <= 0) { fracChars = 0; return 0; }
  const rate = revealRate(backlog);
  fracChars += rate * (dtMs / 1000);
  let take = Math.floor(fracChars);
  if (take < 1 && dtMs > 0) take = 1;          // never stall on a slow tick
  fracChars -= take;
  if (take > backlog) { take = backlog; fracChars = 0; }
  let left = take;
  for (let i = 0; i < S.turns.length && left > 0; i++) {
    const t = S.turns[i];
    if (t.role !== 'assistant') continue;
    const rWant = Math.max(0, t.reasoning.length - t.reasonShown);
    if (rWant > 0) {
      const k = Math.min(rWant, left);
      t.reasonShown += k;
      left -= k;
      paintReason(t);
    }
    const cWant = Math.max(0, t.text.length - t.shown);
    if (cWant > 0 && left > 0) {
      const k = Math.min(cWant, left);
      t.shown += k;
      left -= k;
      paintBody(t);
    }
  }
  if (take > 0) autoScroll();
  return take;
}

function pumpTick() {
  pumpTimer = null;
  const t = now();
  const dt = lastPump ? Math.max(1, t - lastPump) : LIMITS.pumpMs;
  lastPump = t;
  pumpAdvance(dt);
  if (pendingChars() > 0 || S.busy) pumpSchedule();
  else lastPump = 0;
}

function pumpSchedule() {
  if (S.clock) return;                        // a stepped clock: only advance() moves the pump
  if (pumpTimer !== null) return;
  pumpTimer = setTimeout(pumpTick, LIMITS.pumpMs);
}

function pumpStop() {
  if (pumpTimer !== null) { clearTimeout(pumpTimer); pumpTimer = null; }
  lastPump = 0;
  fracChars = 0;
}

function flushTypewriter() {
  for (let i = 0; i < S.turns.length; i++) {
    const t = S.turns[i];
    if (t.role !== 'assistant') continue;
    t.shown = t.text.length;
    t.reasonShown = t.reasoning.length;
    paintReason(t);
    paintBody(t);
  }
  fracChars = 0;
  autoScroll();
}

// ------------------------------------------------------------------ painting
function paintBody(t) {
  if (!t.bodyNode) return;
  const src = t.text.slice(0, t.shown);
  clearNode(t.bodyNode);
  if (src) t.bodyNode.appendChild(renderMarkdown(src));
  else if (t.status === 'streaming') t.bodyNode.appendChild(mk('span', 'md-wait', '正在思考'));
  if (t.status === 'streaming' && t.shown > 0) {
    const caret = mk('span', 'caret');
    caret.setAttribute('aria-hidden', 'true');
    // The caret goes inside the last inline-bearing block. It must not land in a code fence (it
    // would be copied with the code), in a table, or in an <hr>, which cannot hold children.
    const last = t.bodyNode.lastElementChild;
    const tag = last ? last.tagName : '';
    if (/^(P|H1|H2|H3|H4|H5|H6|LI)$/.test(tag)) last.appendChild(caret);
    else t.bodyNode.appendChild(caret);
  }
}

function paintReason(t) {
  if (!t.reasonNode) return;
  const has = t.reasoning.length > 0;
  t.reasonNode.hidden = !has;
  if (!has) return;
  t.reasonBody.textContent = t.reasoning.slice(0, t.reasonShown);
  const n = t.reasoning.length;
  t.reasonCount.textContent = n + ' 字';
  if (t.status === 'streaming' && !t.reasonAutoDone) {
    // Open it once when thinking starts. A manual collapse afterwards must stick, so this is a
    // one-shot latch rather than a condition re-evaluated on every pump tick.
    t.reasonAutoDone = true;
    t.reasonOpen = true;
    t.reasonBody.hidden = false;
    t.reasonToggle.setAttribute('aria-expanded', 'true');
    t.reasonArrow.textContent = '▾';
  }
}

function paintMeta(t) {
  if (!t.metaNode) return;
  clearNode(t.metaNode);
  const bits = [];
  if (t.meta.completionTokens) bits.push(t.meta.completionTokens + ' tokens');
  if (t.meta.tokS) bits.push(t.meta.tokS.toFixed(2) + ' tok/s');
  if (t.meta.ms) bits.push(Math.round(t.meta.ms) + ' ms');
  if (t.meta.finish) bits.push(t.meta.finish);
  if (t.status === 'stopped') bits.push('已停止');
  for (let i = 0; i < bits.length; i++) t.metaNode.appendChild(mk('span', null, bits[i]));
  t.metaNode.hidden = bits.length === 0;
}

function paintError(t) {
  if (!t.errNode) return;
  t.errNode.hidden = !t.error;
  if (t.error) t.errNode.textContent = t.error;
}

function buildUserNode(t) {
  const wrap = mk('div', 'turn turn-user');
  wrap.id = t.id;
  const b = mk('div', 'bubble', t.text);
  wrap.appendChild(b);
  t.node = wrap;
  return wrap;
}

function buildAiNode(t) {
  const wrap = mk('div', 'turn turn-ai');
  wrap.id = t.id;

  const head = mk('div', 'ai-head');
  const icon = mk('span');
  icon.innerHTML = SVG_MARK;                  // a literal constant authored here, never model text
  head.appendChild(icon);
  head.appendChild(mk('span', null, 'BONSAI'));
  wrap.appendChild(head);

  const reason = mk('div', 'reason');
  reason.hidden = true;
  const toggle = mk('button', 'reason-toggle');
  toggle.type = 'button';
  toggle.setAttribute('aria-expanded', 'false');
  const arrow = mk('span', null, '▸');
  toggle.appendChild(arrow);
  toggle.appendChild(mk('span', null, '思考过程'));
  const count = mk('span', 'reason-count', '');
  toggle.appendChild(count);
  const rbody = mk('div', 'reason-body');
  rbody.hidden = true;
  reason.appendChild(toggle);
  reason.appendChild(rbody);
  wrap.appendChild(reason);

  const body = mk('div', 'md');
  wrap.appendChild(body);
  const meta = mk('div', 'turn-meta');
  meta.hidden = true;
  wrap.appendChild(meta);
  const err = mk('p', 'err-note');
  err.hidden = true;
  wrap.appendChild(err);

  t.node = wrap;
  t.bodyNode = body;
  t.reasonNode = reason;
  t.reasonBody = rbody;
  t.reasonToggle = toggle;
  t.reasonArrow = arrow;
  t.reasonCount = count;
  t.metaNode = meta;
  t.errNode = err;
  on(toggle, 'click', function () {
    t.reasonOpen = !t.reasonOpen;
    rbody.hidden = !t.reasonOpen;
    toggle.setAttribute('aria-expanded', t.reasonOpen ? 'true' : 'false');
    arrow.textContent = t.reasonOpen ? '▾' : '▸';
  });
  return wrap;
}

function renderAll() {
  const box = el('msgs');
  clearNode(box);
  for (let i = 0; i < S.turns.length; i++) {
    const t = S.turns[i];
    box.appendChild(t.role === 'user' ? buildUserNode(t) : buildAiNode(t));
    if (t.role === 'assistant') { paintReason(t); paintBody(t); paintMeta(t); paintError(t); }
  }
  el('empty').hidden = S.turns.length > 0;
  autoScroll(true);
}

function appendTurnNodes(t) {
  el('msgs').appendChild(t.role === 'user' ? buildUserNode(t) : buildAiNode(t));
  el('empty').hidden = true;
}

function autoScroll(force) {
  const sc = el('scroll');
  if (!sc) return;
  if (!force && !stickBottom) return;
  sc.scrollTop = sc.scrollHeight;
}

function trackScroll() {
  const sc = el('scroll');
  stickBottom = (sc.scrollHeight - sc.scrollTop - sc.clientHeight) < 80;
}

// ------------------------------------------------------------------ badge
function setBadge() {
  const b = el('badge');
  const dot = el('badge-dot');
  const txt = el('badge-text');
  if (!b || !dot || !txt) return;
  const st = S.status;
  if (!S.online || !st) {
    dot.className = 'dot dot-down';
    txt.textContent = '【未连接 | localhost:8080】';
    b.title = S.statusError || '引擎尚未就绪：双击 bonsai_vk.exe 启动后会自动打开本页';
    return;
  }
  dot.className = 'dot ' + (st.busy ? 'dot-busy' : 'dot-live');
  const dev = shortDevice(st.device);
  const size = fmtKib(st.engine_kib);
  let rate = '';
  let ref = false;
  if (Number(st.last_tok_s) > 0) rate = fmtRate(st.last_tok_s);
  else if (Number(st.reference_tok_s) > 0) { rate = fmtRate(st.reference_tok_s); ref = true; }
  let s = '【' + dev + ' | ' + size + (rate ? ' | ' + rate + (ref ? ' ref' : '') : '') + '】';
  if (st.busy) s += ' 生成中';
  txt.textContent = s;
  b.title = [
    'device: ' + st.device,
    'model: ' + st.model,
    'engine: ' + st.engine_bytes + ' B (' + st.engine_kib + ' KiB), measured from the executable',
    'KV arena: cap ' + st.cap + ' slots, ' + st.kv_kib_per_token + ' KiB/token, V ' + st.v_quant,
    'weights ' + st.weights_mib + ' MiB + activations ' + st.activations_mib + ' MiB',
    'requests ' + st.requests + ', completions ' + st.completions + ', tokens ' + st.tokens_generated,
    ref ? ('tok/s is the repo reference ' + st.reference_tok_s + ' (' + st.reference_source +
           '); it is replaced by this process\'s measured rate after the first reply')
        : ('last reply: ' + st.last_tok_s + ' tok/s over ' + st.last_ms + ' ms, measured in this process')
  ].join('\n');
}

async function refreshStatus() {
  const ctl = new AbortController();
  const timer = setTimeout(function () { ctl.abort(); }, 4000);
  try {
    const st = await transport().status(ctl.signal);
    S.status = st;
    S.online = true;
    S.statusError = '';
    // The per-turn rate is the ENGINE's number (n_gen * 1000 / ms_gen), not a browser wall-clock
    // guess that would silently include prefill. It is only attached to the reply that just ended,
    // because /v1/status reports the most recent completion and this build serves one at a time.
    if (S.justFinished && Number(st.last_tok_s) > 0) {
      const t = findTurn(S.justFinished);
      if (t && t.status === 'done' && t.meta.tokS === undefined) {
        t.meta.tokS = Number(st.last_tok_s);
        if (Number(st.last_ms) > 0) t.meta.ms = Number(st.last_ms);
        paintMeta(t);
      }
      S.justFinished = null;
    }
  } catch (err) {
    if (err && err.name === 'AbortError') {
      S.statusError = '状态请求超时';
    } else {
      S.online = false;
      S.statusError = (err && err.message) || String(err);
    }
  } finally {
    clearTimeout(timer);
    setBadge();
    updateCtxNote();
  }
}

function updateCtxNote() {
  const n = el('ctx-note');
  if (!n) return;
  const st = S.status;
  if (S.online && st && st.cap) {
    n.textContent = '上下文 ' + st.cap + ' slots · V ' + st.v_quant + ' · 内容本地生成，请自行核对';
  } else {
    n.textContent = '内容由本地模型生成，请自行核对';
  }
}

function startPolling() {
  if (S.pollTimer !== null) return;
  S.pollTimer = setInterval(function () { refreshStatus(); }, LIMITS.statusPollMs);
}

function stopPolling() {
  if (S.pollTimer !== null) { clearInterval(S.pollTimer); S.pollTimer = null; }
}

// ------------------------------------------------------------------ composer
function autoGrow() {
  const ta = el('input');
  ta.style.height = 'auto';
  const max = 190;
  ta.style.height = Math.min(ta.scrollHeight, max) + 'px';
  ta.style.overflowY = ta.scrollHeight > max ? 'auto' : 'hidden';
}

function syncSend() {
  const ta = el('input');
  const send = el('send');
  const stop = el('stop');
  const has = ta.value.trim().length > 0;
  send.disabled = S.busy || !has;
  send.hidden = S.busy;
  stop.hidden = !S.busy;
  ta.disabled = false;
}

function setBusy(v) {
  S.busy = v;
  syncSend();
  setBadge();
}

function addUserTurn(text) {
  const t = { id: nextId('t'), role: 'user', text: text, shown: text.length, reasonShown: 0, reasoning: '', status: 'done', meta: {}, error: '' };
  S.turns.push(t);
  appendTurnNodes(t);
  return t;
}

function addAiTurn() {
  const t = {
    id: nextId('t'), role: 'assistant', text: '', reasoning: '', shown: 0, reasonShown: 0,
    status: 'streaming', meta: {}, error: '', reasonOpen: false, startedAt: now()
  };
  S.turns.push(t);
  appendTurnNodes(t);
  paintBody(t);
  return t;
}

function finishTurn(t, status, errText) {
  t.status = status;
  if (errText) t.error = errText;
  if (status !== 'streaming' && t.reasonOpen && t.text.length > 0) {
    t.reasonOpen = false;
    t.reasonBody.hidden = true;
    t.reasonToggle.setAttribute('aria-expanded', 'false');
    t.reasonArrow.textContent = '▸';
  }
  flushTypewriter();
  paintMeta(t);
  paintError(t);
  if (t.metaNode && t.startedAt !== undefined && !t.meta.ms) {
    t.meta.ms = now() - t.startedAt;
    paintMeta(t);
  }
}

async function sendText(raw) {
  const text = String(raw).trim();
  if (!text || S.busy) return false;
  hideToast();
  addUserTurn(text);
  const ai = addAiTurn();
  const ta = el('input');
  ta.value = '';
  autoGrow();
  setBusy(true);
  stickBottom = true;
  autoScroll(true);
  lastPump = now();
  pumpSchedule();

  const messages = buildMessages(S.turns.slice(0, S.turns.length - 1), '');
  const body = requestBody(messages, { maxTokens: 1024 });
  S.requestsSent += 1;
  const ctl = new AbortController();
  S.abort = ctl;

  function onEvent(ev) {
    if (ev.done) { ai.receivedDone = true; return; }
    if (ev.malformed) { recordError('sse-malformed', ev.raw); return; }
    const p = chunkParts(ev);
    if (!p) return;
    if (p.id) ai.meta.id = p.id;
    if (p.reasoning) { ai.reasoning += p.reasoning; }
    if (p.content) { ai.text += p.content; }
    if (p.usage) {
      ai.meta.promptTokens = p.usage.prompt_tokens;
      ai.meta.completionTokens = p.usage.completion_tokens;
    }
    if (p.finish) ai.meta.finish = p.finish;
    pumpSchedule();
  }

  try {
    await transport().completeStream(body, onEvent, ctl.signal);
    // Wall time is shown as wall time. The tok/s figure is fetched from /v1/status right after this,
    // where the engine reports its own n_gen/ms_gen; deriving a rate here would include prefill and
    // disagree with the badge by a number nobody could explain.
    ai.meta.wallMs = now() - ai.startedAt;
    if (!ai.meta.ms) ai.meta.ms = ai.meta.wallMs;
    S.justFinished = ai.id;
    finishTurn(ai, 'done', '');
  } catch (err) {
    if (err && err.name === 'AbortError') {
      finishTurn(ai, 'stopped', '');
    } else {
      const status = err && err.status ? err.status : 0;
      let msg = (err && err.message) || String(err);
      if (status === 429) msg = '引擎正在生成另一条回复（本构建一次只跑一个对话）。' + msg;
      else if (status === 400) msg = '请求被拒绝：' + msg;
      else if (!S.online) msg = '连不上引擎：' + msg + '（确认 bonsai_vk.exe 还在运行）';
      recordError('api', msg);
      finishTurn(ai, 'error', msg);
      toast(msg.slice(0, 160));
    }
  } finally {
    S.abort = null;
    setBusy(false);
    pumpStop();
    // Awaited on purpose: the caller (and the self-test) must not observe a finished turn before the
    // engine's measured rate for it has been fetched, or the per-turn meta would silently be empty.
    await refreshStatus();
  }
  return true;
}

function sendCurrent() {
  return sendText(el('input').value);
}

function stopGeneration() {
  if (S.abort) { try { S.abort.abort(); } catch (err) { recordError('abort', err); } }
}

function newChat() {
  stopGeneration();
  pumpStop();
  S.turns = [];
  S.busy = false;
  S.abort = null;
  renderAll();
  syncSend();
  const ta = el('input');
  ta.value = '';
  autoGrow();
  ta.focus();
}

// ------------------------------------------------------------------ wiring
function copyCodeBlock(btn) {
  const wrap = btn.closest('.codeblock');
  if (!wrap) return;
  const code = wrap.querySelector('pre code');
  const text = code ? code.textContent : '';
  function done(ok) {
    btn.textContent = ok ? '已复制' : '复制失败';
    setTimeout(function () { btn.textContent = '复制'; }, 1400);
  }
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).then(function () { done(true); }, function () { done(fallbackCopy(text)); });
  } else {
    done(fallbackCopy(text));
  }
}

// file:// and non-secure contexts have no async clipboard. The textarea path still works there and
// leaves no node behind.
function fallbackCopy(text) {
  const ta = mk('textarea');
  ta.value = text;
  ta.setAttribute('readonly', '');
  ta.style.position = 'fixed';
  ta.style.opacity = '0';
  document.body.appendChild(ta);
  ta.select();
  let ok = false;
  try { ok = document.execCommand('copy'); } catch (err) { ok = false; }
  document.body.removeChild(ta);
  return ok;
}

function wire() {
  const ta = el('input');
  const form = el('composer');

  on(form, 'submit', function (ev) { ev.preventDefault(); sendCurrent(); });
  on(ta, 'input', function () { autoGrow(); syncSend(); });
  on(el('stop'), 'click', function () { stopGeneration(); });
  on(el('btn-new'), 'click', function () { newChat(); });

  let composing = false;
  on(ta, 'compositionstart', function () { composing = true; });
  on(ta, 'compositionend', function () { composing = false; autoGrow(); syncSend(); });
  on(ta, 'keydown', function (ev) {
    // Enter sends, Shift+Enter newlines. An IME candidate window also presses Enter, and swallowing
    // that would send half a Chinese sentence - isComposing (and the legacy 229) is the guard.
    if (ev.key !== 'Enter' || ev.shiftKey) return;
    if (composing || ev.isComposing || ev.keyCode === 229) return;
    ev.preventDefault();
    sendCurrent();
  });

  const chips = el('chips');
  on(chips, 'click', function (ev) {
    const btn = ev.target.closest('.chip');
    if (!btn) return;
    const p = btn.getAttribute('data-prompt') || '';
    el('input').value = p;
    autoGrow();
    syncSend();
    sendCurrent();
  });

  on(el('msgs'), 'click', function (ev) {
    const btn = ev.target.closest('.copy-btn');
    if (btn) copyCodeBlock(btn);
  });

  on(el('scroll'), 'scroll', trackScroll, { passive: true });
  on(window, 'resize', function () { autoGrow(); });
  on(document, 'keydown', function (ev) {
    if ((ev.ctrlKey || ev.metaKey) && ev.shiftKey && (ev.key === 'O' || ev.key === 'o')) {
      ev.preventDefault();
      newChat();
    }
  });

  autoGrow();
  syncSend();
  setBadge();
  renderAll();
  if (location.protocol === 'http:' || location.protocol === 'https:') {
    refreshStatus();
    startPolling();
  } else {
    // Opened straight off the disk (file://). There is no server behind this page, and a relative
    // fetch of /v1/status would resolve to a local path and log a resource error on every poll.
    // Say so in the badge instead. The self-test drives refreshStatus() explicitly with a mock
    // transport, so the offline branch does not weaken the streaming coverage.
    S.online = false;
    S.statusError = '本页是从磁盘直接打开的。请双击 bonsai_vk.exe，由引擎在 localhost:8080 上提供本页。';
    setBadge();
  }
  ta.focus();
}

function boot() {
  installErrorCapture();
  wire();
}

if (document.readyState === 'loading') on(document, 'DOMContentLoaded', boot);
else boot();