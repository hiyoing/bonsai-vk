#!/usr/bin/env node
// Real-input UI drive for the Bonsai WebUI. Zero npm dependencies (Node built-ins + CDP over WebSocket).
//
//   $ node webui/scripts/cdp-ui-drive.mjs [--url <file-or-http>] [--outdir webui/shots] [--viewport 1440,900]
//
// Unlike cdp-run-bot.mjs (which calls the page's own deterministic suite), this driver presses the page
// the way a person does: Input.dispatchMouseEvent for clicks and Input.dispatchKeyEvent / insertText for
// typing. The transport is a mock that speaks the server's real SSE framing, so the drive needs no GPU
// and no running engine; the production parser, pump and DOM are still the things under test.
//
// Every state is captured at full resolution and every geometry claim is a measured boundingBox.
import { spawn } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const argv = process.argv.slice(2);
function take(name, fallback = '') {
  const i = argv.indexOf(name);
  if (i === -1) return fallback;
  const v = argv[i + 1];
  return v === undefined || v.startsWith('--') ? fallback : v;
}
const urlInput = take('--url', '');
const outdir = resolve(take('--outdir', 'webui/shots'));
const viewport = take('--viewport', '1440,900').replace(/x/i, ',');
const [VW, VH] = viewport.split(',').map(Number);

let targetUrl = urlInput;
if (!targetUrl) targetUrl = pathToFileURL(resolve('runtime_vk/index.html')).href;
else if (!/^(https?:|file:|data:|about:)/i.test(targetUrl)) targetUrl = pathToFileURL(resolve(targetUrl)).href;

const browserCandidates = [
  take('--browser', ''), process.env.BROWSER_PATH || '',
  'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
  'C:/Program Files/Microsoft/Edge/Application/msedge.exe',
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
  '/usr/bin/google-chrome', '/usr/bin/chromium',
].filter(Boolean);
const browserPath = [...new Set(browserCandidates)].find(c => existsSync(c));
if (!browserPath) { console.error('No Edge/Chrome found; set --browser or BROWSER_PATH.'); process.exit(2); }

const sleep = ms => new Promise(r => setTimeout(r, ms));
const profile = mkdtempSync(join(tmpdir(), 'cdp-ui-'));
const port = 9400 + Math.floor(Math.random() * 400);
const results = [];
const ok = (name, cond, detail = '') => { results.push({ name, ok: !!cond, detail }); console.log(`${cond ? 'PASS' : 'FAIL'} | ${name}${detail ? ' | ' + detail : ''}`); };

let browser = null, socket = null, spawnError = null;
async function waitJson(url, waitMs) {
  const end = Date.now() + waitMs;
  while (Date.now() < end) {
    if (spawnError) throw spawnError;
    try { const r = await fetch(url); if (r.ok) return await r.json(); } catch { /* starting */ }
    await sleep(150);
  }
  throw new Error('CDP endpoint not ready: ' + url);
}

try {
  browser = spawn(browserPath, [
    '--headless=new', '--disable-gpu', '--hide-scrollbars', '--allow-file-access-from-files',
    '--no-first-run', '--no-default-browser-check', `--remote-debugging-port=${port}`,
    `--user-data-dir=${profile}`, `--window-size=${VW},${VH}`, 'about:blank',
  ], { stdio: 'ignore', windowsHide: true });
  browser.on('error', e => { spawnError = e; });
  await waitJson(`http://127.0.0.1:${port}/json/version`, 15000);
  const targets = await waitJson(`http://127.0.0.1:${port}/json/list`, 10000);
  const page = targets.find(t => t.type === 'page' && t.webSocketDebuggerUrl);
  if (!page) throw new Error('no page target');
  socket = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((res, rej) => { socket.onopen = res; socket.onerror = () => rej(new Error('ws failed')); });

  let nextId = 1; const pending = new Map();
  const exceptions = [], consoleErrors = [], logErrors = [];
  function send(method, params = {}, ms = 15000) {
    return new Promise((res, rej) => {
      const id = nextId++;
      const timer = setTimeout(() => { pending.delete(id); rej(new Error('CDP timeout: ' + method)); }, ms);
      pending.set(id, { res, rej, timer });
      socket.send(JSON.stringify({ id, method, params }));
    });
  }
  socket.onmessage = ev => {
    const m = JSON.parse(typeof ev.data === 'string' ? ev.data : Buffer.from(ev.data).toString('utf8'));
    if (m.id && pending.has(m.id)) { const e = pending.get(m.id); pending.delete(m.id); clearTimeout(e.timer); m.error ? e.rej(new Error(m.error.message)) : e.res(m.result); return; }
    if (m.method === 'Runtime.exceptionThrown') exceptions.push(m.params.exceptionDetails?.exception?.description || m.params.exceptionDetails?.text || 'exception');
    else if (m.method === 'Runtime.consoleAPICalled' && m.params.type === 'error') consoleErrors.push(m.params.args.map(a => a.value ?? a.description ?? '').join(' '));
    else if (m.method === 'Log.entryAdded' && m.params.entry.level === 'error') logErrors.push(m.params.entry.text || 'log error');
  };
  await send('Page.enable'); await send('Runtime.enable'); await send('Log.enable');
  await send('Emulation.setDeviceMetricsOverride', { width: VW, height: VH, deviceScaleFactor: 1, mobile: false });
  const nav = await send('Page.navigate', { url: targetUrl }, 30000);
  if (nav.errorText) throw new Error('navigation: ' + nav.errorText);

  async function evalJs(expression, ms = 10000) {
    const r = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true }, ms);
    if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description || r.exceptionDetails.text || 'eval failed');
    return r.result?.value;
  }
  async function waitFor(expression, waitMs = 15000) {
    const end = Date.now() + waitMs;
    while (Date.now() < end) { if (await evalJs(expression)) return; await sleep(80); }
    throw new Error('timed out: ' + expression);
  }
  async function box(sel) {
    return evalJs(`(() => { const e = document.querySelector(${JSON.stringify(sel)}); if (!e) return null; const r = e.getBoundingClientRect(); return { x: r.x, y: r.y, w: r.width, h: r.height, bottom: r.bottom, right: r.right }; })()`);
  }
  async function clickSel(sel) {
    const b = await box(sel);
    if (!b) throw new Error('no element for ' + sel);
    const x = b.x + b.w / 2, y = b.y + b.h / 2;
    await send('Input.dispatchMouseEvent', { type: 'mousePressed', x, y, button: 'left', clickCount: 1 });
    await send('Input.dispatchMouseEvent', { type: 'mouseReleased', x, y, button: 'left', clickCount: 1 });
    await sleep(60);
  }
  async function shot(name) {
    const c = await send('Page.captureScreenshot', { format: 'png', fromSurface: true }, 20000);
    mkdirSync(outdir, { recursive: true });
    const p = join(outdir, name);
    writeFileSync(p, Buffer.from(c.data, 'base64'));
    console.log('screenshot: ' + p);
  }
  async function key(type, keyName, code, vk, modifiers = 0, text = '') {
    await send('Input.dispatchKeyEvent', { type, key: keyName, code, windowsVirtualKeyCode: vk, nativeVirtualKeyCode: vk, modifiers, text, unmodifiedText: text });
  }

  await waitFor("window.__APP__ && document.readyState === 'complete'");
  await sleep(200);

  // Install a mock that speaks the real wire format, with a rich markdown reply so the screenshots
  // show headings, a table, a fenced block and a collapsible reasoning panel.
  await evalJs(`(() => {
    const A = window.__APP__;
    A.setSeed(7);
    // Hold the stream open partway through so the streaming state and caret are genuinely observable.
    // Without a pause the mock delivers in microtasks and finishTurn() flushes the typewriter at once.
    let openGate; const gate = { promise: new Promise(r => { openGate = r; }) };
    window.__gateOpen = openGate;
    A.setTransport(A.makeMockTransport({
      status: A.mockStatus,
      replies: [{
        pieces: [
          { reasoning_content: '先确认显存预算：权重 2467.7 MiB，活动区随 cap 增长；KV 只按 8 层计费。' },
          '## 三值量化一句话\\n\\n',
          '把每个权重压到 **{-1, 0, +1}** 三个取值，再乘一个 per-block 缩放因子。',
          '相对 FP16，权重字节数降到约 **1/10**，因此 4 GB 显存也装得下 9B 参数。\\n\\n',
          '| 格式 | 每权重字节 | 9B 权重体积 |\\n|---|---:|---:|\\n| FP16 | 2.0 | ~18 GB |\\n| PTQ1_0 | ~0.28 | ~2.4 GB |\\n\\n',
          '一个最小反量化示意：\\n\\n\`\`\`c\\nfloat w = scale * (float)trit;  // trit in {-1,0,1}\\n\`\`\`\\n\\n',
          '- 权重常驻显存，避免每 token 重新上传\\n- 只有 V 走 q8_0，K 保持 FP16\\n- 解包结果不回写全局再读\\n\\n',
          '更多背景见 [README](../README.md)。'
        ],
        sliceSize: 24,
        gate: gate,
        holdAfter: 900,
        usage: { prompt_tokens: 31, completion_tokens: 168, total_tokens: 199 }
      }]
    }));
    return A.refreshStatus();
  })()`);
  await sleep(150);
  ok('badge is live after mock status', await evalJs(`document.getElementById('badge-text').textContent.indexOf('RX 550X') !== -1`));
  await shot('ui-01-empty-1440.png');

  // Real mouse click on the first suggestion chip -> starts a streamed reply.
  const chipBox = await box('.chip');
  await clickSel('.chip');
  await sleep(450);   // catch the typewriter mid-reveal: caret + partial markdown
  const midState = await evalJs(`(() => { const s = window.__APP__.getState(); const t = s.turns[s.turns.length - 1]; return { pending: s.pending, status: t && t.status, shown: t && t.shown, len: t && t.text.length, caret: document.querySelectorAll('#msgs .caret').length }; })()`);
  ok('a real chip click started a streaming turn', midState.status === 'streaming', JSON.stringify(midState));
  ok('the streaming caret is visible mid-reveal', midState.caret === 1, JSON.stringify(midState));
  await shot('ui-02-streaming-1440.png');
  await evalJs(`window.__gateOpen()`);   // let the held stream finish

  await waitFor(`(() => { const s = window.__APP__.getState(); const t = s.turns[s.turns.length - 1]; return s.pending === 0 && t && t.status === 'done'; })()`, 20000);
  await sleep(200);
  ok('the reply finished and drained', await evalJs(`window.__APP__.getState().turns.slice(-1)[0].status === 'done'`));
  ok('markdown produced a table in the real DOM', await evalJs(`document.querySelectorAll('#msgs table').length === 1`));
  ok('markdown produced a fenced code block', await evalJs(`document.querySelectorAll('#msgs .codeblock').length === 1`));
  ok('the caret is gone when done', await evalJs(`document.querySelectorAll('#msgs .caret').length === 0`));
  await shot('ui-03-done-1440.png');

  // Geometry, measured: header vs dock, column width, centring, overflow, newest reply above composer.
  const hdr = await box('#hdr'), dock = await box('#dock'), thread = await box('#thread');
  ok('header and composer do not overlap', hdr.bottom <= dock.y, `hdr.bottom=${hdr.bottom} dock.y=${dock.y}`);
  ok('column <= 768px', thread.w <= 768.5, 'w=' + thread.w);
  ok('column centred', Math.abs((thread.x + thread.right) / 2 - VW / 2) < 2);
  ok('no horizontal overflow', await evalJs(`document.documentElement.scrollWidth <= window.innerWidth + 1`));
  const dTitle = await box('#hdr-title'), dBadge = await box('#badge');
  ok('desktop: title does not overprint the badge', dTitle.right <= dBadge.x + 0.5, `title.right=${dTitle.right} badge.x=${dBadge.x}`);
  await evalJs(`(() => { const sc = document.getElementById('scroll'); sc.scrollTop = sc.scrollHeight; return true; })()`);
  await sleep(80);
  const lastMsg = await box('#msgs .turn:last-child');
  ok('newest reply not hidden behind composer', lastMsg.bottom <= dock.y + 1, `msg.bottom=${lastMsg.bottom} dock.y=${dock.y}`);

  // Real click on the reasoning toggle.
  await clickSel('#msgs .reason-toggle');
  ok('reasoning expands on a real click', await evalJs(`!document.querySelector('#msgs .reason-body').hidden`));
  ok('reasoning toggle reports expanded', await evalJs(`document.querySelector('#msgs .reason-toggle').getAttribute('aria-expanded') === 'true'`));
  await shot('ui-04-reasoning-1440.png');
  await clickSel('#msgs .reason-toggle');
  ok('reasoning collapses on a second real click', await evalJs(`document.querySelector('#msgs .reason-body').hidden`));

  // Real click on the code copy button.
  const copyBefore = await evalJs(`document.querySelector('#msgs .copy-btn').textContent`);
  await clickSel('#msgs .copy-btn');
  await sleep(120);
  const copyAfter = await evalJs(`document.querySelector('#msgs .copy-btn').textContent`);
  ok('copy button reacts to a real click', copyAfter !== copyBefore, `${copyBefore} -> ${copyAfter}`);

  // Real typing + Enter sends a second turn; Shift+Enter must not.
  await clickSel('#input');
  await send('Input.insertText', { text: '第二条消息' });
  ok('typing enables send', await evalJs(`!document.getElementById('send').disabled`));
  const turnsBefore = await evalJs(`window.__APP__.getState().turns.length`);
  await key('keyDown', 'Enter', 'Enter', 13, 8);   // Shift+Enter: newline, no send
  await key('keyUp', 'Enter', 'Enter', 13, 8);
  await sleep(80);
  ok('Shift+Enter does not send', await evalJs(`window.__APP__.getState().turns.length`) === turnsBefore);
  await key('keyDown', 'Enter', 'Enter', 13, 0);
  await key('keyUp', 'Enter', 'Enter', 13, 0);
  await sleep(150);
  ok('Enter sends a second turn', await evalJs(`window.__APP__.getState().turns.length`) > turnsBefore);
  await waitFor(`(() => { const s = window.__APP__.getState(); return !s.busy && s.pending === 0; })()`, 20000);
  await shot('ui-05-two-turns-1440.png');

  // Real click on new chat.
  await clickSel('#btn-new');
  ok('new chat clears the transcript', await evalJs(`window.__APP__.getState().turns.length === 0`));
  ok('new chat restores the empty state', await evalJs(`!document.getElementById('empty').hidden`));

  // Narrow viewport: the layout must not overflow or collide.
  await send('Emulation.setDeviceMetricsOverride', { width: 420, height: 900, deviceScaleFactor: 1, mobile: false });
  await sleep(150);
  await clickSel('.chip');
  await waitFor(`(() => { const s = window.__APP__.getState(); return !s.busy && s.pending === 0; })()`, 20000);
  await sleep(150);
  ok('narrow: no horizontal overflow', await evalJs(`document.documentElement.scrollWidth <= window.innerWidth + 1`));
  const nThread = await box('#thread'), nDock = await box('#dock'), nHdr = await box('#hdr');
  ok('narrow: header and dock do not overlap', nHdr.bottom <= nDock.y);
  ok('narrow: column fits the viewport', nThread.right <= 420 + 1 && nThread.x >= -1, JSON.stringify(nThread));
  const nTitle = await box('#hdr-title'), nBadge = await box('#badge'), nBtn = await box('#btn-new');
  ok('narrow: title does not overprint the badge', nTitle.right <= nBadge.x + 0.5, `title.right=${nTitle.right} badge.x=${nBadge.x}`);
  ok('narrow: title clears the new-chat button', nTitle.x >= nBtn.right - 0.5, `title.x=${nTitle.x} btn.right=${nBtn.right}`);
  await shot('ui-06-narrow-420.png');

  console.log(`exceptions: ${exceptions.length}`);
  console.log(`consoleErrors: ${consoleErrors.length}`);
  console.log(`logErrors: ${logErrors.length}`);
  if (exceptions.length) console.log(exceptions.slice(0, 5).join('\n---\n'));
  if (consoleErrors.length) console.log(consoleErrors.slice(0, 5).join('\n---\n'));
  if (logErrors.length) console.log(logErrors.slice(0, 5).join('\n---\n'));
  const failed = results.filter(r => !r.ok).length;
  console.log(`ui-drive: ${results.length - failed}/${results.length} passed`);
  if (failed || exceptions.length || consoleErrors.length || logErrors.length) process.exitCode = 1;
} catch (err) {
  console.error('UI DRIVE FAILED: ' + err.message);
  process.exitCode = 1;
} finally {
  try { socket?.close(); } catch { /* ignore */ }
  try { browser?.kill(); } catch { /* ignore */ }
  try { rmSync(profile, { recursive: true, force: true }); } catch { /* ignore */ }
}