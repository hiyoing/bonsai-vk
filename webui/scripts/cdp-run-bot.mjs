#!/usr/bin/env node
import { spawn } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const argv = process.argv.slice(2);
const VALUE_FLAGS = new Set([
  '--url', '--expr', '--ready', '--timeout', '--settle', '--screenshot', '--window', '--viewport', '--browser',
]);
const KNOWN_FLAGS = new Set([...VALUE_FLAGS, '--summary', '--full', '--help']);

function take(name, fallback = '') {
  const i = argv.indexOf(name);
  if (i === -1) return fallback;
  const v = argv[i + 1];
  return v === undefined || (v.startsWith('--') && KNOWN_FLAGS.has(v)) ? fallback : v;
}
function flag(name) {
  return argv.includes(name);
}

const urlInput = take('--url', '');
if (!urlInput) {
  console.error('Usage: node cdp-run-bot.mjs --url <file-or-url> [--expr <js>] [--ready <js>] [--viewport 1440x900] [--screenshot <png>] [--summary|--full]');
  process.exit(2);
}

const expr = take('--expr', 'window.__selfTest ? window.__selfTest() : null');
const ready = take('--ready', "document.readyState === 'complete'");
const timeout = Math.max(5000, Number(take('--timeout', '60000')) || 60000);
const settle = Math.max(0, Number(take('--settle', '300')) || 0);
const screenshot = take('--screenshot', '');
const windowSize = take('--window', '1440,900').replace(/x/i, ',');
const viewportSize = take('--viewport', '1440,900').replace(/x/i, ',');
const full = flag('--full');

let targetUrl = urlInput;
if (!/^(https?:|file:|data:|about:)/i.test(targetUrl)) {
  targetUrl = pathToFileURL(resolve(targetUrl)).href;
}

const browserCandidates = [
  take('--browser', ''),
  process.env.BROWSER_PATH || '',
  'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
  'C:/Program Files/Microsoft/Edge/Application/msedge.exe',
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
  'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe',
  '/usr/bin/google-chrome',
  '/usr/bin/chromium',
  '/usr/bin/chromium-browser',
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
].filter(Boolean);

const browserPath = [...new Set(browserCandidates)].find(candidate => existsSync(candidate));
if (!browserPath) {
  console.error('No Edge/Chrome executable found. Set --browser <path> or BROWSER_PATH.');
  process.exit(2);
}

const sleep = ms => new Promise(resolveSleep => setTimeout(resolveSleep, ms));
const profile = mkdtempSync(join(tmpdir(), 'cdp-bot-'));
const port = 9300 + Math.floor(Math.random() * 400);
const browserArgs = [
  '--headless=new',
  '--disable-gpu',
  '--hide-scrollbars',
  '--allow-file-access-from-files',
  '--no-first-run',
  '--no-default-browser-check',
  `--remote-debugging-port=${port}`,
  `--user-data-dir=${profile}`,
  `--window-size=${windowSize}`,
  'about:blank',
];
if (process.env.CDP_NO_SANDBOX === '1') browserArgs.unshift('--no-sandbox');

let browser = null;
let socket = null;
let spawnError = null;

async function waitJson(url, waitMs) {
  const end = Date.now() + waitMs;
  while (Date.now() < end) {
    if (spawnError) throw spawnError;
    try {
      const response = await fetch(url);
      if (response.ok) return await response.json();
    } catch {
      // Browser is still starting.
    }
    await sleep(200);
  }
  throw new Error(`CDP endpoint not ready: ${url}`);
}

try {
  browser = spawn(browserPath, browserArgs, { stdio: 'ignore', windowsHide: true });
  browser.on('error', err => {
    spawnError = err;
  });

  await waitJson(`http://127.0.0.1:${port}/json/version`, 15000);
  const targets = await waitJson(`http://127.0.0.1:${port}/json/list`, 10000);
  const page = targets.find(target => target.type === 'page' && target.webSocketDebuggerUrl);
  if (!page) throw new Error('No page target found');

  socket = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((resolveOpen, rejectOpen) => {
    socket.onopen = resolveOpen;
    socket.onerror = () => rejectOpen(new Error('WebSocket connection failed'));
  });

  let nextId = 1;
  const pending = new Map();
  const exceptions = [];
  const consoleErrors = [];
  const logErrors = [];

  function send(method, params = {}, sendMs = 15000) {
    return new Promise((resolveSend, rejectSend) => {
      const id = nextId++;
      const timer = setTimeout(() => {
        pending.delete(id);
        rejectSend(new Error(`CDP timeout: ${method}`));
      }, sendMs);
      pending.set(id, { resolve: resolveSend, reject: rejectSend, timer });
      socket.send(JSON.stringify({ id, method, params }));
    });
  }

  socket.onmessage = event => {
    const raw = typeof event.data === 'string' ? event.data : Buffer.from(event.data).toString('utf8');
    const message = JSON.parse(raw);
    if (message.id && pending.has(message.id)) {
      const entry = pending.get(message.id);
      pending.delete(message.id);
      clearTimeout(entry.timer);
      if (message.error) entry.reject(new Error(message.error.message));
      else entry.resolve(message.result);
      return;
    }
    if (message.method === 'Runtime.exceptionThrown') {
      const details = message.params.exceptionDetails;
      exceptions.push(details.exception?.description || details.text || 'unknown exception');
    } else if (message.method === 'Runtime.consoleAPICalled' && message.params.type === 'error') {
      consoleErrors.push(message.params.args.map(arg => arg.value ?? arg.description ?? '').join(' '));
    } else if (message.method === 'Log.entryAdded' && message.params.entry.level === 'error') {
      logErrors.push(message.params.entry.text || message.params.entry.url || 'unknown log error');
    }
  };

  await send('Page.enable');
  await send('Runtime.enable');
  await send('Log.enable');

  const [viewportWidth, viewportHeight] = viewportSize.split(',').map(Number);
  if (!viewportWidth || !viewportHeight) throw new Error(`invalid --viewport: ${viewportSize}`);
  await send('Emulation.setDeviceMetricsOverride', {
    width: viewportWidth,
    height: viewportHeight,
    deviceScaleFactor: 1,
    mobile: false,
  });

  const navigation = await send('Page.navigate', { url: targetUrl }, 30000);
  if (navigation.errorText) throw new Error(`Navigation failed: ${navigation.errorText}`);

  async function waitForExpression(expression, waitMs) {
    const end = Date.now() + waitMs;
    let lastError = '';
    while (Date.now() < end) {
      try {
        const result = await send('Runtime.evaluate', {
          expression,
          awaitPromise: true,
          returnByValue: true,
        }, 5000);
        if (result.exceptionDetails) {
          lastError = result.exceptionDetails.exception?.description || result.exceptionDetails.text || '';
        } else if (result.result?.value) {
          return;
        }
      } catch (err) {
        lastError = err.message;
      }
      await sleep(120);
    }
    throw new Error(`Timed out waiting for: ${expression}${lastError ? ` (last error: ${lastError})` : ''}`);
  }

  if (ready) await waitForExpression(ready, timeout);
  if (settle) await sleep(settle);

  const evaluation = await send('Runtime.evaluate', {
    expression: expr,
    awaitPromise: true,
    returnByValue: true,
    userGesture: true,
  }, Math.max(15000, timeout));

  if (evaluation.exceptionDetails) {
    throw new Error(evaluation.exceptionDetails.exception?.description || evaluation.exceptionDetails.text || 'Evaluation failed');
  }

  const value = evaluation.result?.value;

  if (screenshot) {
    const capture = await send('Page.captureScreenshot', { format: 'png', fromSurface: true }, 20000);
    const outputPath = resolve(screenshot);
    mkdirSync(dirname(outputPath), { recursive: true });
    writeFileSync(outputPath, Buffer.from(capture.data, 'base64'));
    console.log(`screenshot: ${outputPath}`);
  }

  if (value && typeof value === 'object' && 'ok' in value) {
    console.log(`ok: ${value.ok}`);
    if (value.durationMs != null) console.log(`duration: ${value.durationMs}ms`);
    if (value.totals) console.log(`totals: ${JSON.stringify(value.totals)}`);
    if (Array.isArray(value.results)) {
      for (const result of value.results) {
        console.log(`${result.ok ? 'PASS' : 'FAIL'} | ${result.name} | ${result.summary ?? ''}`);
        if (!result.ok && result.errors?.length) {
          console.log(`  errors: ${JSON.stringify(result.errors).slice(0, 1200)}`);
        }
      }
    }
    if (value.runtimeErrors?.length) console.log(`runtimeErrors: ${value.runtimeErrors.length}`);
    if (full) console.log(JSON.stringify(value, null, 2));
  } else if (value === undefined || value === null) {
    console.log('result: null');
  } else {
    console.log(`result: ${typeof value === 'string' ? value : JSON.stringify(value)}`);
    if (full) console.log(JSON.stringify(value, null, 2));
  }

  console.log(`exceptions: ${exceptions.length}`);
  console.log(`consoleErrors: ${consoleErrors.length}`);
  console.log(`logErrors: ${logErrors.length}`);
  if (exceptions.length) console.log(exceptions.slice(0, 5).join('\n---\n'));
  if (consoleErrors.length) console.log(consoleErrors.slice(0, 5).join('\n---\n'));
  if (logErrors.length) console.log(logErrors.slice(0, 5).join('\n---\n'));

  const reportOk = value && typeof value === 'object' && 'ok' in value
    ? value.ok !== false
    : value !== undefined && value !== null;
  const runtimeErrorCount = value && typeof value === 'object' && Array.isArray(value.runtimeErrors)
    ? value.runtimeErrors.length
    : 0;
  if (!reportOk || runtimeErrorCount > 0 || exceptions.length || consoleErrors.length || logErrors.length) {
    process.exitCode = 1;
  }
} catch (err) {
  console.error(`BOT RUN FAILED: ${err.message}`);
  process.exitCode = 1;
} finally {
  try {
    socket?.close();
  } catch {
    // Ignore cleanup failures.
  }
  try {
    browser?.kill();
  } catch {
    // Ignore cleanup failures.
  }
  try {
    rmSync(profile, { recursive: true, force: true });
  } catch {
    // Browser may still hold the profile on Windows; temp cleanup is best effort.
  }
}
