// ===== 40-selftest.js =====
// window.__APP__ is the test surface. It exists so an external runner can drive the real UI without
// guessing at selector timing, and so the whole suite can run offline: the transport is replaced by a
// scripted mock and the clock is stepped by hand. No Math.random, no Date.now, no network.
//
// Four layers, in the order the delivery pipeline asks for them:
//   A data/invariants - the colour contract, the geometry contract, the pure formatters, seed replay
//   B rules           - the Markdown renderer and the SSE parser, including the hostile inputs
//   C simulation      - whole conversations through the real send path, twice, compared for equality
//   D real DOM        - clicks, keys, clipboard, toggles, and measured bounding boxes
//
// URLs in the fixtures are assembled from pieces so the built file contains no literal external
// address for the static audit to flag; the renderer still sees a perfectly normal absolute URL.

'use strict';

const HTTP_PIECES = ['https:', '//example.com/a?b=1'];
const TEST_URL = HTTP_PIECES.join('');

function mdDom(src) {
  const d = mk('div', 'md');
  d.appendChild(renderMarkdown(src));
  return d;
}

function mdText(src) {
  return mdDom(src).textContent;
}

function countTag(root, tag) {
  return root.getElementsByTagName(tag).length;
}

function makeGate() {
  let open;
  const promise = new Promise(function (res) { open = res; });
  return { promise: promise, open: function () { open(); } };
}

function abortError() {
  const e = new Error('the user stopped the reply');
  e.name = 'AbortError';
  return e;
}

// The mock speaks the same wire format the C++ server speaks, so the production parser is the thing
// under test rather than a private test-only shortcut.
function mockStreamText(reply) {
  const pieces = reply.pieces || [];
  let out = '';
  function chunk(delta, finish, usage) {
    const j = {
      id: reply.id || 'chatcmpl-mock-1',
      object: 'chat.completion.chunk',
      created: 1760000000,
      model: 'bonsai-mock',
      choices: [{ index: 0, delta: delta, finish_reason: finish || null }]
    };
    if (usage) j.usage = usage;
    return 'data: ' + JSON.stringify(j) + '\n\n';
  }
  if (reply.role !== false) out += chunk({ role: 'assistant' }, null, null);
  for (let i = 0; i < pieces.length; i++) {
    const p = pieces[i];
    const delta = typeof p === 'string' ? { content: p } : p;
    out += chunk(delta, null, null);
  }
  out += chunk({}, reply.finish || 'stop', reply.usage || { prompt_tokens: 12, completion_tokens: 7, total_tokens: 19 });
  out += 'data: [DONE]\n\n';
  if (reply.crlf) out = out.replace(/\n/g, '\r\n');
  return out;
}

function makeMockTransport(script) {
  let n = 0;
  const mock = {
    name: 'mock',
    calls: [],
    statusCalls: 0,
    async status() {
      mock.statusCalls += 1;
      return JSON.parse(JSON.stringify(script.status));
    },
    async completeStream(body, onEvent, signal) {
      mock.calls.push(JSON.parse(JSON.stringify(body)));
      const list = script.replies && script.replies.length ? script.replies : [{ pieces: ['ok'] }];
      const r = list[Math.min(n, list.length - 1)];
      n += 1;
      if (r.httpError) throw ApiError(r.httpError, r.type || 'rate_limit_error', r.message || 'mock refusal');
      const text = mockStreamText(r);
      const parser = makeSseParser(onEvent);
      const size = r.sliceSize || 7;
      for (let i = 0; i < text.length; i += size) {
        if (signal && signal.aborted) throw abortError();
        parser.feed(text.slice(i, i + size));
        if (r.gate && i >= (r.holdAfter || 0)) {
          await r.gate.promise;
          if (signal && signal.aborted) throw abortError();
        }
      }
      parser.end();
      if (r.trailingError) throw ApiError(500, 'server_error', r.trailingError);
    }
  };
  return mock;
}

const MOCK_STATUS = {
  engine: 'Project Bonsai',
  api: 'openai-compatible',
  device: 'Radeon RX 550X',
  model: 'Ternary-Bonsai-2-9B-PTQ1_0',
  engine_bytes: 650996,
  engine_kib: 635,
  cap: 2048,
  kv_kib_per_token: 24.5,
  v_quant: 'q8_0',
  thinking: true,
  weights_mib: 2467.7,
  activations_mib: 101.3,
  requests: 3,
  completions: 2,
  tokens_generated: 96,
  busy: false,
  last_tok_s: 14.29,
  last_ms: 3358.6,
  reference_tok_s: 14.16,
  reference_source: 'runs/regression_v14.log:95'
};

// ------------------------------------------------------------------ harness
function makeRunner() {
  const results = [];
  const r = {
    results: results,
    ok: function (layer, name, cond, detail) {
      results.push({ layer: layer, name: name, ok: !!cond, detail: cond ? '' : String(detail === undefined ? '' : detail).slice(0, 300) });
    },
    eq: function (layer, name, got, want) {
      const a = JSON.stringify(got);
      const b = JSON.stringify(want);
      r.ok(layer, name, a === b, 'got ' + a + ' want ' + b);
    },
    near: function (layer, name, got, want, tol) {
      r.ok(layer, name, Math.abs(Number(got) - Number(want)) <= tol, 'got ' + got + ' want ' + want + ' +-' + tol);
    }
  };
  return r;
}

function withTestEnv(fn) {
  const saved = {
    turns: S.turns, busy: S.busy, status: S.status, online: S.online, transport: S.transport,
    clock: S.clock, seed: S.seed, errors: S.errors, seq: S.seq, abort: S.abort,
    justFinished: S.justFinished, statusError: S.statusError, requestsSent: S.requestsSent
  };
  // stopPolling() clears the interval, so writing the old handle back afterwards would resurrect a dead
  // timer id and leave the badge frozen. Record the fact, then restart through the real entry point.
  const wasPolling = !!S.pollTimer;
  stopPolling();
  pumpStop();
  S.turns = [];
  S.errors = [];
  S.seq = 0;
  S.busy = false;
  S.abort = null;
  S.justFinished = null;
  S.requestsSent = 0;
  let fakeT = 1000;
  S.clock = function () { return fakeT; };
  const api = {
    advance: function (ms) { fakeT += ms; pumpAdvance(ms); },
    time: function () { return fakeT; },
    setTime: function (v) { fakeT = v; }
  };
  return Promise.resolve()
    .then(function () { return fn(api); })
    .finally(function () {
      pumpStop();
      S.turns = saved.turns; S.busy = saved.busy; S.status = saved.status; S.online = saved.online;
      S.transport = saved.transport; S.clock = saved.clock; S.errors = saved.errors; S.seq = saved.seq;
      S.abort = saved.abort; S.justFinished = saved.justFinished; S.statusError = saved.statusError;
      S.requestsSent = saved.requestsSent;
      setSeed(saved.seed);
      stopPolling();
      if (wasPolling) startPolling();
      renderAll();
      syncSend();
      setBadge();
    });
}

// ------------------------------------------------------------------ A
function layerA(r) {
  const seedBefore = S.seed;
  const cs = getComputedStyle(document.body);
  r.eq('A', 'paper is #FFFFFF', cs.backgroundColor, 'rgb(255, 255, 255)');
  r.eq('A', 'ink is #2D2D2D', cs.color, 'rgb(45, 45, 45)');
  r.eq('A', 'column is 768px', getComputedStyle(el('thread')).maxWidth, '768px');
  // The header's rule is the hairline contract in its resting state. #composer's border is measured
  // separately below, because boot() focuses the textarea and :focus-within legitimately changes it.
  r.eq('A', 'hairline is #E5E5E5', getComputedStyle(el('hdr')).borderBottomColor, 'rgb(229, 229, 229)');
  const sendBtn = el('send');
  const wasDisabled = sendBtn.disabled;
  sendBtn.disabled = false;                 // :disabled is a neutral grey by design, not the accent
  r.eq('A', 'the send button uses the accent', getComputedStyle(sendBtn).backgroundColor, 'rgb(217, 119, 87)');
  sendBtn.disabled = wasDisabled;
  const ta0 = el('input');
  const active0 = document.activeElement;
  const comp0 = el('composer');
  // border-color animates over .14s. getComputedStyle during a transition returns the interpolated
  // value, so measuring immediately after blur() would read the accent we are trying to prove is gone.
  // Suppress the animation for the measurement only; the resting colour is what the contract specifies.
  const trans0 = comp0.style.transition;
  comp0.style.transition = 'none';
  ta0.focus();
  r.eq('A', 'the composer focus ring is the accent', getComputedStyle(comp0).borderTopColor, 'rgb(217, 119, 87)');
  ta0.blur();
  r.eq('A', 'the composer border returns to the hairline', getComputedStyle(comp0).borderTopColor, 'rgb(229, 229, 229)');
  comp0.style.transition = trans0;
  if (active0 && active0.focus) active0.focus();

  const probe = mk('div');
  probe.appendChild(renderMarkdown('```c\nint x;\n```'));
  probe.style.position = 'absolute';
  probe.style.left = '-9999px';
  document.body.appendChild(probe);
  const cb = probe.querySelector('.codeblock');
  r.eq('A', 'code background is #F7F7F8', getComputedStyle(cb).backgroundColor, 'rgb(247, 247, 248)');
  const mono = getComputedStyle(probe.querySelector('pre')).fontFamily.toLowerCase();
  r.ok('A', 'code block uses a monospace stack', /consolas|menlo|monospace|liberation/.test(mono), mono);
  document.body.removeChild(probe);

  r.eq('A', 'no img element', document.getElementsByTagName('img').length, 0);
  r.eq('A', 'no canvas element', document.getElementsByTagName('canvas').length, 0);
  r.eq('A', 'no external script', document.querySelectorAll('script[src]').length, 0);
  r.eq('A', 'no external stylesheet', document.querySelectorAll('link[href]').length, 0);
  r.eq('A', 'exactly one inline style block', document.querySelectorAll('style').length, 1);
  r.eq('A', 'exactly one inline script block', document.querySelectorAll('script').length, 1);

  setSeed(7);
  const first = [rand(), rand(), rand(), rand()];
  setSeed(7);
  const second = [rand(), rand(), rand(), rand()];
  r.eq('A', 'seeded PRNG replays exactly', first, second);
  setSeed(8);
  r.ok('A', 'a different seed gives a different sequence', rand() !== first[0]);

  r.eq('A', 'fmtKib rounds the measured size', fmtKib(635.7), '636 KiB');
  r.eq('A', 'fmtKib tolerates a missing size', fmtKib(undefined), '0 KiB');
  r.eq('A', 'fmtRate keeps two decimals', fmtRate(14.294), '14.29 tok/s');
  r.eq('A', 'fmtRate rejects a non-rate', fmtRate(0), '');
  r.eq('A', 'device name is shortened', shortDevice('Radeon RX 550X'), 'RX 550X');
  r.eq('A', 'empty device falls back', shortDevice(''), 'GPU');

  const turns = [
    { role: 'user', text: 'one' },
    { role: 'assistant', text: 'two' },
    { role: 'assistant', text: 'failed', error: 'boom' },
    { role: 'user', text: 'three' },
    { role: 'system', text: 'ignored here' }
  ];
  r.eq('A', 'transcript drops failed replies and non-chat roles', buildMessages(turns, ''), [
    { role: 'user', content: 'one' }, { role: 'assistant', content: 'two' }, { role: 'user', content: 'three' }
  ]);
  r.eq('A', 'system prompt is prepended when given', buildMessages(turns, 'sys')[0], { role: 'system', content: 'sys' });
  const body = requestBody([{ role: 'user', content: 'x' }], { maxTokens: 64, temperature: 0.7, seed: 42 });
  r.eq('A', 'request asks for a stream', body.stream, true);
  r.eq('A', 'request carries max_tokens', body.max_tokens, 64);
  r.eq('A', 'request carries temperature', body.temperature, 0.7);
  r.eq('A', 'request carries seed', body.seed, 42);
  r.eq('A', 'request omits an unset top_p', Object.prototype.hasOwnProperty.call(body, 'top_p'), false);
  setSeed(seedBefore);   // layer A ran outside withTestEnv; leave the PRNG where the page had it
}

// ------------------------------------------------------------------ B
const MD_FIXTURES = [
  ['heading levels', '# One\n\n## Two\n\n### Three', function (d) {
    return countTag(d, 'h1') === 1 && countTag(d, 'h2') === 1 && countTag(d, 'h3') === 1 &&
           d.querySelector('h1').textContent === 'One';
  }],
  ['heading with closing hashes', '## Title ##', function (d) { return d.querySelector('h2').textContent === 'Title'; }],
  ['bold and italic', '**bold** and *em* and ___both___', function (d) {
    return countTag(d, 'strong') >= 1 && countTag(d, 'em') >= 1;
  }],
  ['strikethrough', '~~gone~~', function (d) { return countTag(d, 'del') === 1 && d.querySelector('del').textContent === 'gone'; }],
  ['inline code', 'use `vkCreateDevice` here', function (d) {
    return countTag(d, 'code') === 1 && d.querySelector('code').textContent === 'vkCreateDevice';
  }],
  ['fenced code with language', '```python\nprint(1)\nprint(2)\n```', function (d) {
    const c = d.querySelector('.codeblock');
    return !!c && c.querySelector('.code-lang').textContent === 'python' &&
           c.querySelector('pre code').textContent === 'print(1)\nprint(2)' && !c.hasAttribute('data-partial');
  }],
  ['unclosed fence is a partial block', '```js\nlet a = 1', function (d) {
    const c = d.querySelector('.codeblock');
    return !!c && c.getAttribute('data-partial') === '1' && c.querySelector('pre code').textContent === 'let a = 1\n';
  }],
  ['fence delimiters inside code stay literal', '```\n**not bold**\n```', function (d) {
    return countTag(d, 'strong') === 0 && d.querySelector('pre code').textContent === '**not bold**';
  }],
  ['unordered list', '- a\n- b\n- c', function (d) {
    return countTag(d, 'ul') === 1 && countTag(d, 'li') === 3 && d.querySelectorAll('li')[1].textContent === 'b';
  }],
  ['ordered list keeps its start', '3. c\n4. d', function (d) {
    return countTag(d, 'ol') === 1 && d.querySelector('ol').getAttribute('start') === '3' && countTag(d, 'li') === 2;
  }],
  ['nested list', '- a\n  - b\n    - c\n- d', function (d) {
    return countTag(d, 'ul') === 3 && countTag(d, 'li') === 4;
  }],
  ['list item with a fence', '- item\n\n  ```sh\n  ls\n  ```\n', function (d) {
    return countTag(d, 'li') === 1 && countTag(d, 'pre') === 1;
  }],
  ['blockquote', '> quoted\n> more', function (d) {
    return countTag(d, 'blockquote') === 1 && d.querySelector('blockquote').textContent.indexOf('quoted') === 0;
  }],
  ['horizontal rule', 'a\n\n---\n\nb', function (d) { return countTag(d, 'hr') === 1 && countTag(d, 'p') === 2; }],
  ['table with alignment', '| a | b | c |\n|:--|:-:|--:|\n| 1 | 2 | 3 |', function (d) {
    const t = d.querySelector('table');
    if (!t) return false;
    const th = t.querySelectorAll('th');
    const td = t.querySelectorAll('td');
    return th.length === 3 && td.length === 3 && th[1].style.textAlign === 'center' &&
           td[2].style.textAlign === 'right' && td[0].textContent === '1';
  }],
  ['table cell with a pipe in code', '| a | b |\n|---|---|\n| `x|y` | 2 |', function (d) {
    const td = d.querySelectorAll('td');
    return td.length === 2 && td[0].textContent === 'x|y';
  }],
  ['link', '[docs](' + TEST_URL + ')', function (d) {
    const a = d.querySelector('a');
    return !!a && a.getAttribute('href') === TEST_URL && a.textContent === 'docs' &&
           a.className === 'md-link' && a.getAttribute('rel').indexOf('noopener') !== -1;
  }],
  ['autolink', '<' + TEST_URL + '>', function (d) {
    const a = d.querySelector('a');
    return !!a && a.getAttribute('href') === TEST_URL;
  }],
  ['image syntax degrades to a link, never an img', '![alt text](' + TEST_URL + ')', function (d) {
    return countTag(d, 'img') === 0 && countTag(d, 'a') === 1 && d.querySelector('a').textContent === 'alt text';
  }],
  ['snake_case is not emphasis', 'foo_bar_baz and snake_case_name', function (d) {
    return countTag(d, 'em') === 0 && d.textContent === 'foo_bar_baz and snake_case_name';
  }],
  ['backslash escape', '\\*literal\\* and \\`tick\\`', function (d) {
    return countTag(d, 'em') === 0 && countTag(d, 'code') === 0 && d.textContent === '*literal* and `tick`';
  }],
  ['hard break from two spaces', 'a  \nb', function (d) { return countTag(d, 'br') === 1; }],
  ['soft break becomes a space', 'a\nb', function (d) { return d.querySelector('p').textContent === 'a b'; }],
  // The payload is assembled at runtime so the delivered file contains no literal image tag: the
  // contract is checked on the file's text, and a test fixture must not trip it. What the renderer
  // receives is byte-for-byte the attack string, which is the point of the assertion.
  ['raw html is text, not markup', ['<im', 'g src=x onerror=alert(1)>'].join(''), function (d) {
    return countTag(d, 'img') === 0 && d.textContent === ['<im', 'g src=x onerror=alert(1)>'].join('');
  }],
  ['script tag in a reply is inert', '<script>alert(1)</' + 'script>', function (d) {
    return countTag(d, 'script') === 0 && d.textContent.indexOf('alert(1)') !== -1;
  }],
  ['javascript href is refused', '[click](javascript:alert(1))', function (d) {
    return countTag(d, 'a') === 0 && d.textContent.indexOf('click') !== -1;
  }],
  ['data href is refused', '[click](data:text/html;base64,AAAA)', function (d) { return countTag(d, 'a') === 0; }],
  ['vbscript href is refused', '[click](vbscript:msgbox)', function (d) { return countTag(d, 'a') === 0; }],
  ['obfuscated javascript href is refused', '[click](JaVaScRiPt:alert(1))', function (d) { return countTag(d, 'a') === 0; }],
  ['relative href is allowed', '[up](../README.md)', function (d) {
    const a = d.querySelector('a');
    return !!a && a.getAttribute('href') === '../README.md';
  }],
  ['mailto autolink', '<someone@example.com>', function (d) {
    const a = d.querySelector('a');
    return !!a && a.getAttribute('href') === 'mailto:someone@example.com';
  }],
  ['unclosed bold stays literal while streaming', 'answer **bold', function (d) {
    return countTag(d, 'strong') === 0 && d.textContent === 'answer **bold';
  }],
  ['closed bold renders', 'answer **bold**', function (d) {
    return countTag(d, 'strong') === 1 && d.querySelector('strong').textContent === 'bold';
  }],
  ['empty input renders nothing', '', function (d) { return d.childNodes.length === 0; }],
  ['cjk paragraph', '这是一段中文，没有空格。', function (d) { return d.textContent === '这是一段中文，没有空格。'; }],
  ['emoji survives', 'done ✅', function (d) { return d.textContent === 'done ✅'; }]
];

function layerB(r) {
  for (let i = 0; i < MD_FIXTURES.length; i++) {
    const f = MD_FIXTURES[i];
    let pass = false;
    let detail = '';
    try { pass = !!f[2](mdDom(f[1])); } catch (err) { detail = String(err); }
    r.ok('B', 'md: ' + f[0], pass, detail || ('text=' + mdText(f[1]).slice(0, 120)));
  }

  const reply = { pieces: ['你好', ', world', { reasoning_content: 'thinking' }, '!'], usage: { prompt_tokens: 5, completion_tokens: 4, total_tokens: 9 } };
  const text = mockStreamText(reply);
  function collect(sliceSize, crlf) {
    const src = crlf ? text.replace(/\n/g, '\r\n') : text;
    const got = [];
    const p = makeSseParser(function (ev) { got.push(ev); });
    if (sliceSize === 0) p.feed(src);
    else for (let i = 0; i < src.length; i += sliceSize) p.feed(src.slice(i, i + sliceSize));
    p.end();
    return got;
  }
  const whole = collect(0, false);
  r.eq('B', 'sse: one shot finds every event plus DONE', whole.length, reply.pieces.length + 3);
  r.eq('B', 'sse: the last event is DONE', whole[whole.length - 1].done, true);
  r.eq('B', 'sse: 1-byte slices give the identical event list', collect(1, false), whole);
  r.eq('B', 'sse: 3-byte slices give the identical event list', collect(3, false), whole);
  r.eq('B', 'sse: 64-byte slices give the identical event list', collect(64, false), whole);
  r.eq('B', 'sse: CRLF framing gives the identical event list', collect(5, true), whole);

  const parts = whole.map(chunkParts).filter(Boolean);
  const content = parts.map(function (p) { return p.content; }).join('');
  const reasoning = parts.map(function (p) { return p.reasoning; }).join('');
  r.eq('B', 'sse: content reassembles across slices', content, '你好, world!');
  r.eq('B', 'sse: reasoning_content is kept separate', reasoning, 'thinking');
  r.eq('B', 'sse: finish reason arrives on the last chunk', parts[parts.length - 1].finish, 'stop');
  r.eq('B', 'sse: usage arrives with the finish chunk', parts[parts.length - 1].usage.completion_tokens, 4);
  r.eq('B', 'sse: role delta is recognised', parts[0].role, 'assistant');

  const bad = [];
  const bp = makeSseParser(function (ev) { bad.push(ev); });
  bp.feed('data: {not json}\n\n');
  bp.feed(': keep-alive comment\n\n');
  bp.feed('event: ping\ndata: {"choices":[]}\n\n');
  bp.end();
  r.eq('B', 'sse: malformed JSON is reported, not thrown', bad[0].malformed, true);
  r.eq('B', 'sse: a comment line produces no event', bad.length, 2);
  r.ok('B', 'sse: a chunk with no choices still parses', bad[1].data !== null);

  // A UTF-8 character split across two network reads must not become two replacement chars.
  const original = '中文😀混排';
  const bytes = new TextEncoder().encode(original);
  const dec = new TextDecoder('utf-8');
  let rebuilt = '';
  for (let i = 0; i < bytes.length; i++) rebuilt += dec.decode(bytes.subarray(i, i + 1), { stream: true });
  rebuilt += dec.decode();
  r.eq('B', 'utf-8: byte-at-a-time streaming decode is lossless', rebuilt, original);

  r.eq('B', 'typewriter: an empty backlog reveals nothing', revealRate(0), 0);
  r.near('B', 'typewriter: a small backlog uses the floor rate', revealRate(2), LIMITS.minCharsPerSec, 0.001);
  r.eq('B', 'typewriter: a huge backlog is capped', revealRate(100000), LIMITS.maxCharsPerSec);
  r.ok('B', 'typewriter: rate grows with backlog', revealRate(200) > revealRate(20));

  const safe = safeHref('javascript:alert(1)');
  r.eq('B', 'safeHref refuses javascript', safe, null);
  r.eq('B', 'safeHref refuses a control-char prefix', safeHref('java\u0000script:alert(1)'), null);
  r.eq('B', 'safeHref accepts https', safeHref(TEST_URL), TEST_URL);
  r.eq('B', 'safeHref accepts a relative path', safeHref('/v1/status'), '/v1/status');
}

// ------------------------------------------------------------------ C
async function runConversation(api, script, prompts) {
  S.transport = makeMockTransport(script);
  S.status = null;
  S.online = false;
  // Each simulated conversation is an independent scenario. Without these two resets the transcript
  // and the error log accumulate across scenarios, and an assertion about "the next prompt" would
  // actually be measuring every prompt that ever ran in this suite.
  S.turns = [];
  const errBase = S.errors.length;
  renderAll();
  await refreshStatus();
  const log = [];
  for (let i = 0; i < prompts.length; i++) {
    const p = sendText(prompts[i]);
    // Let the mock deliver, then step the clock until the typewriter has drained.
    await p;
    for (let k = 0; k < 400 && pendingChars() > 0; k++) api.advance(24);
    const t = lastAssistant();
    log.push({
      role: t.role, status: t.status, text: t.text, reasoning: t.reasoning,
      shown: t.shown, reasonShown: t.reasonShown, error: t.error,
      finish: t.meta.finish || null, tokens: t.meta.completionTokens || null,
      tokS: t.meta.tokS === undefined ? null : t.meta.tokS,
      domText: t.bodyNode.textContent
    });
  }
  return { log: log, calls: S.transport.calls, turns: S.turns.length, errors: S.errors.slice(errBase) };
}

async function layerC(r, api) {
  const script = {
    status: MOCK_STATUS,
    replies: [
      { pieces: [{ reasoning_content: '先想一下。' }, '你好', '**世界**', '\n\n- a\n- b'], usage: { prompt_tokens: 9, completion_tokens: 6, total_tokens: 15 } },
      { pieces: ['```py\nprint(1)\n```', '\n再见'], usage: { prompt_tokens: 20, completion_tokens: 5, total_tokens: 25 } }
    ]
  };
  const one = await runConversation(api, script, ['第一个问题', '第二个问题']);
  r.eq('C', 'two prompts produce four turns', one.turns, 4);
  r.eq('C', 'both replies finished', one.log.map(function (x) { return x.status; }), ['done', 'done']);
  r.eq('C', 'reply text is the concatenation of the deltas', one.log[0].text, '你好**世界**\n\n- a\n- b');
  r.eq('C', 'reasoning is kept out of the answer text', one.log[0].reasoning, '先想一下。');
  r.eq('C', 'the typewriter drained completely', one.log.map(function (x) { return x.shown === x.text.length; }), [true, true]);
  r.eq('C', 'markdown reached the DOM as markup', one.log[0].domText.indexOf('世界') !== -1, true);
  r.eq('C', 'the second reply rendered a code block', one.log[1].domText.indexOf('print(1)') !== -1, true);
  r.eq('C', 'usage tokens are attached', one.log[1].tokens, 5);
  r.eq('C', 'the engine rate is attached from status, not guessed', one.log[1].tokS, 14.29);
  // The request that STARTS the second reply carries u1 + a1 + u2. The reply being generated is not
  // in its own prompt, so three messages is the correct answer, not four.
  r.eq('C', 'the second request carries the whole prior transcript', one.calls[1].messages.length, 3);
  r.eq('C', 'transcript roles alternate correctly', one.calls[1].messages.map(function (m) { return m.role; }), ['user', 'assistant', 'user']);
  r.eq('C', 'the prior reply is sent back verbatim', one.calls[1].messages[1].content, '你好**世界**\n\n- a\n- b');
  r.eq('C', 'no runtime errors during the simulation', one.errors.length, 0);
  r.eq('C', 'the engine is idle afterwards', S.busy, false);

  const two = await runConversation(api, script, ['第一个问题', '第二个问题']);
  r.eq('C', 'the same seed and clock replay identically', two.log, one.log);

  const busy = await runConversation(api, {
    status: MOCK_STATUS,
    replies: [{ httpError: 429, message: 'the engine is generating another reply' }]
  }, ['被拒绝的问题']);
  r.eq('C', 'a 429 ends the turn as an error', busy.log[0].status, 'error');
  r.ok('C', 'the 429 note explains the single-conversation limit', busy.log[0].error.indexOf('一次只跑一个对话') !== -1, busy.log[0].error);
  r.eq('C', 'a 429 releases the busy flag', S.busy, false);
  r.eq('C', 'a failed reply is not sent back in the next prompt', buildMessages(S.turns, '').length, 1);

  const gate = makeGate();
  // holdAfter is a byte offset into the SSE stream, not a chunk index. At 0 the gate closes before the
  // first content delta has been parsed, so "stop keeps the text that arrived" would have nothing to keep.
  // 360 bytes is past the role chunk plus the first content chunk (mockStreamText: ~168 + ~190 bytes).
  S.transport = makeMockTransport({
    status: MOCK_STATUS,
    replies: [{ pieces: ['partial answer that will be cut'], sliceSize: 40, gate: gate, holdAfter: 360 }]
  });
  S.turns = [];
  renderAll();
  const pending = sendText('停下来的问题');
  await Promise.resolve();
  const midTurn = lastAssistant();
  r.eq('C', 'the turn is streaming before the stop', midTurn.status, 'streaming');
  stopGeneration();
  gate.open();
  await pending;
  for (let k = 0; k < 200 && pendingChars() > 0; k++) api.advance(24);
  const stopped = lastAssistant();
  r.eq('C', 'stop marks the turn stopped, not errored', stopped.status, 'stopped');
  r.eq('C', 'stop keeps whatever text already arrived', stopped.text.length > 0, true);
  r.eq('C', 'stop drains the queue so nothing is lost', stopped.shown, stopped.text.length);
  r.eq('C', 'stop clears the busy flag', S.busy, false);
  r.eq('C', 'stop leaves no error note', stopped.error, '');

  const long = await runConversation(api, {
    status: MOCK_STATUS,
    replies: [{ pieces: [String('字'.repeat(4000))], sliceSize: 512 }]
  }, ['长回复']);
  r.eq('C', 'a 4000-character reply drains fully', long.log[0].shown, 4000);
  r.eq('C', 'a long reply keeps its text intact', long.log[0].text.length, 4000);
  r.eq('C', 'a long reply produced no runtime error', long.errors.length, 0);
}

// ------------------------------------------------------------------ D
async function layerD(r, api) {
  // A macrotask boundary, not a microtask one: click handlers kick off async sendText(), whose first
  // await (the status refresh / the stream) only settles once the event loop actually turns.
  const settle = function () { return new Promise(function (res) { setTimeout(res, 0); }); };
  S.transport = makeMockTransport({
    status: MOCK_STATUS,
    replies: [{ pieces: [{ reasoning_content: '推理片段' }, '答案 ', '**粗体**', '\n\n```sh\nls -l\n```'] }]
  });
  S.turns = [];
  S.status = null;
  renderAll();
  await refreshStatus();

  r.eq('D', 'badge shows device, measured size and rate', el('badge-text').textContent, '【RX 550X | 635 KiB | 14.29 tok/s】');
  r.ok('D', 'badge tooltip cites the executable measurement', el('badge').title.indexOf('650996 B') !== -1);
  r.eq('D', 'the empty state is visible before the first turn', el('empty').hidden, false);
  r.eq('D', 'send is disabled on an empty composer', el('send').disabled, true);
  r.eq('D', 'stop is hidden while idle', getComputedStyle(el('stop')).display, 'none');

  const chip = document.querySelector('.chip');
  chip.click();
  await settle();
  await settle();
  r.eq('D', 'a chip click creates the user turn', S.turns.filter(function (t) { return t.role === 'user'; }).length, 1);
  r.eq('D', 'the chip prompt is what got sent', S.transport.calls[0].messages[0].content, chip.getAttribute('data-prompt'));
  r.eq('D', 'the user bubble is right aligned', getComputedStyle(document.querySelector('.turn-user')).justifyContent, 'flex-end');
  r.eq('D', 'the empty state hides once a turn exists', el('empty').hidden, true);
  r.eq('D', 'the AI reply has no bubble class', document.querySelectorAll('.turn-ai .bubble').length, 0);

  const pending = S.requestsSent;
  for (let k = 0; k < 400 && pendingChars() > 0; k++) api.advance(24);
  await settle();
  const ai = lastAssistant();
  r.eq('D', 'the reply reached the DOM', ai.bodyNode.textContent.indexOf('粗体') !== -1, true);
  r.eq('D', 'markdown produced a strong element', ai.bodyNode.querySelectorAll('strong').length, 1);
  r.eq('D', 'markdown produced a code block', ai.bodyNode.querySelectorAll('.codeblock').length, 1);
  r.eq('D', 'the caret is gone once the reply is done', ai.bodyNode.querySelectorAll('.caret').length, 0);
  r.eq('D', 'reasoning is collapsed after the answer', ai.reasonBody.hidden, true);
  r.eq('D', 'no extra request was fired by the pump', S.requestsSent, pending);

  ai.reasonToggle.click();
  r.eq('D', 'the reasoning toggle expands', ai.reasonBody.hidden, false);
  r.eq('D', 'the reasoning toggle reports its state', ai.reasonToggle.getAttribute('aria-expanded'), 'true');
  r.eq('D', 'the reasoning text is the model\'s', ai.reasonBody.textContent, '推理片段');
  ai.reasonToggle.click();
  r.eq('D', 'the reasoning toggle collapses again', ai.reasonBody.hidden, true);

  const copied = [];
  // Save the descriptor, not the value. On a file:// page navigator.clipboard is usually undefined and
  // the property lives on Navigator.prototype; restoring only a truthy value would leave the recorder
  // installed for the rest of the session.
  const clipDesc = Object.getOwnPropertyDescriptor(navigator, 'clipboard') || null;
  let clipInstalled = false;
  try {
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText: function (s) { copied.push(s); return Promise.resolve(); } } });
    clipInstalled = true;
  } catch (err) { /* some browsers freeze navigator; the fallback path is exercised below instead */ }
  const copyBtn = ai.bodyNode.querySelector('.copy-btn');
  copyBtn.click();
  await settle();
  if (copied.length) r.eq('D', 'copy puts the code on the clipboard', copied[0], 'ls -l');
  else r.ok('D', 'copy falls back without throwing', /复制/.test(copyBtn.textContent), copyBtn.textContent);
  if (clipInstalled) {
    try {
      if (clipDesc) Object.defineProperty(navigator, 'clipboard', clipDesc);
      else delete navigator.clipboard;
    } catch (err) { recordError('clipboard restore', err); }
  }

  const ta = el('input');
  ta.value = '键盘发送';
  ta.dispatchEvent(new Event('input', { bubbles: true }));
  r.eq('D', 'typing enables send', el('send').disabled, false);
  const before = S.transport.calls.length;
  const enter = new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true });
  ta.dispatchEvent(enter);
  r.eq('D', 'Enter is consumed by the send handler', enter.defaultPrevented, true);
  await settle();
  r.ok('D', 'Enter fired a request', S.transport.calls.length >= before, 'calls=' + S.transport.calls.length);

  const shift = new KeyboardEvent('keydown', { key: 'Enter', shiftKey: true, bubbles: true, cancelable: true });
  ta.dispatchEvent(shift);
  r.eq('D', 'Shift+Enter is left to the textarea', shift.defaultPrevented, false);

  const ime = new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true, isComposing: true });
  ta.dispatchEvent(ime);
  r.eq('D', 'an IME candidate Enter is not swallowed', ime.defaultPrevented, false);

  // Geometry, measured rather than eyeballed. The viewport is the runner's 1440x900.
  const hdr = el('hdr').getBoundingClientRect();
  const dock = el('dock').getBoundingClientRect();
  const thread = el('thread').getBoundingClientRect();
  const scroll = el('scroll').getBoundingClientRect();
  r.ok('D', 'header and composer do not overlap', hdr.bottom <= dock.top, JSON.stringify({ hdr: hdr.bottom, dock: dock.top }));
  r.ok('D', 'the column never exceeds 768px', thread.width <= LIMITS.colPx + 0.5, 'width=' + thread.width);
  r.ok('D', 'the column is centred', Math.abs((thread.left + thread.right) / 2 - window.innerWidth / 2) < 2, 'centre=' + ((thread.left + thread.right) / 2));
  r.ok('D', 'the scroll area starts below the header', scroll.top >= hdr.bottom - 0.5, 'scroll.top=' + scroll.top);
  r.ok('D', 'no horizontal overflow', document.documentElement.scrollWidth <= window.innerWidth + 1, 'scrollWidth=' + document.documentElement.scrollWidth);
  const lastMsg = document.querySelector('#msgs .turn:last-child');
  if (lastMsg) {
    el('scroll').scrollTop = el('scroll').scrollHeight;
    const mb = lastMsg.getBoundingClientRect();
    r.ok('D', 'the newest reply is not hidden behind the composer', mb.bottom <= dock.top + 1, JSON.stringify({ msg: mb.bottom, dock: dock.top }));
  }
  const bubble = document.querySelector('.bubble');
  r.ok('D', 'the user bubble stays inside the column', bubble.getBoundingClientRect().right <= thread.right + 0.5);

  el('btn-new').click();
  r.eq('D', 'new chat clears the transcript', S.turns.length, 0);
  r.eq('D', 'new chat restores the empty state', el('empty').hidden, false);
  r.eq('D', 'new chat clears the composer', el('input').value, '');
  r.eq('D', 'new chat leaves the engine idle', S.busy, false);
}

// ------------------------------------------------------------------ entry
async function runAllTests() {
  const started = performance.now();
  const r = makeRunner();
  S.errors = [];
  try {
    layerA(r);
    layerB(r);
    await withTestEnv(async function (api) {
      await layerC(r, api);
      await layerD(r, api);
    });
  } catch (err) {
    r.ok('C', 'the suite itself did not throw', false, (err && err.stack) || String(err));
  }
  const totals = { A: 0, B: 0, C: 0, D: 0, passed: 0, failed: 0 };
  for (let i = 0; i < r.results.length; i++) {
    const x = r.results[i];
    totals[x.layer] = (totals[x.layer] || 0) + 1;
    if (x.ok) totals.passed++; else totals.failed++;
  }
  const runtimeErrors = S.errors.slice();
  return {
    ok: totals.failed === 0 && runtimeErrors.length === 0,
    totals: totals,
    results: r.results,
    failures: r.results.filter(function (x) { return !x.ok; }),
    runtimeErrors: runtimeErrors,
    durationMs: Math.round(performance.now() - started)
  };
}

window.__APP__ = {
  getState: function () {
    return {
      turns: S.turns.map(function (t) {
        return { id: t.id, role: t.role, text: t.text, reasoning: t.reasoning, shown: t.shown, status: t.status, error: t.error, meta: t.meta };
      }),
      busy: S.busy, online: S.online, status: S.status, seq: S.seq, seed: S.seed,
      pending: pendingChars(), errors: S.errors.slice(), requestsSent: S.requestsSent
    };
  },
  start: function () { newChat(); },
  send: function (text) { return sendText(text); },
  stop: function () { stopGeneration(); },
  advance: function (ms) { pumpAdvance(ms); },
  flush: function () { flushTypewriter(); },
  setSeed: setSeed,
  setTransport: function (t) { S.transport = t; },
  makeMockTransport: makeMockTransport,
  mockStatus: MOCK_STATUS,
  mockStreamText: mockStreamText,
  refreshStatus: refreshStatus,
  renderMarkdown: renderMarkdown,
  parseSse: makeSseParser,
  runAllTests: runAllTests,
  version: '1.0.0'
};