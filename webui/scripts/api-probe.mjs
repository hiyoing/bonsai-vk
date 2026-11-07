#!/usr/bin/env node
// Connectivity + contract probe for the Bonsai HTTP shell. Zero npm dependencies (node:http only).
//
//   $ node webui/scripts/api-probe.mjs [--port 8080] [--max-tokens 48]
//
// Assumes a server is already listening (runtime_vk/batch_webui_serve.sh starts one). Every claim is
// an observed status code / header / parsed field, never an assumption. The serve-mode tok/s is read
// from the engine's own /v1/status last_tok_s after a real streamed completion, and is reported next
// to the client's wall-clock rate so prefill is never silently folded into a "tok/s" headline.
import http from 'node:http';

const argv = process.argv.slice(2);
function take(name, fallback) {
  const i = argv.indexOf(name);
  return i === -1 ? fallback : argv[i + 1];
}
const port = Number(take('--port', '8080'));
const maxTokens = Number(take('--max-tokens', '48'));
const host = '127.0.0.1';

const results = [];
const ok = (name, cond, detail = '') => {
  results.push({ name, ok: !!cond, detail });
  console.log(`${cond ? 'PASS' : 'FAIL'} | ${name}${detail ? ' | ' + detail : ''}`);
};

function request(method, path, { body, headers = {} } = {}) {
  return new Promise((resolve, reject) => {
    const payload = body === undefined ? null : (typeof body === 'string' ? body : JSON.stringify(body));
    const req = http.request({
      host, port, method, path,
      headers: Object.assign({
        'Content-Type': 'application/json',
        ...(payload ? { 'Content-Length': Buffer.byteLength(payload) } : {}),
      }, headers),
    }, res => {
      const chunks = [];
      res.on('data', c => chunks.push(c));
      res.on('end', () => resolve({
        status: res.statusCode,
        headers: res.headers,
        text: Buffer.concat(chunks).toString('utf8'),
      }));
    });
    req.on('error', reject);
    req.setTimeout(60000, () => { req.destroy(new Error('client timeout')); });
    if (payload) req.write(payload);
    req.end();
  });
}

// Streamed completion: parse the SSE framing exactly as a browser would, and time it wall-clock.
function streamCompletion(body) {
  return new Promise((resolve, reject) => {
    const started = Date.now();
    const payload = JSON.stringify(body);
    const state = { content: '', reasoning: '', finish: null, usage: null, role: null, done: false, events: 0, wallMs: 0 };
    let buf = '';
    const req = http.request({
      host, port, method: 'POST', path: '/v1/chat/completions',
      headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(payload), Accept: 'text/event-stream' },
    }, res => {
      if (res.statusCode !== 200) {
        const chunks = [];
        res.on('data', c => chunks.push(c));
        res.on('end', () => resolve({ status: res.statusCode, state, text: Buffer.concat(chunks).toString('utf8') }));
        return;
      }
      res.setEncoding('utf8');
      res.on('data', chunk => {
        buf += chunk;
        let idx;
        while ((idx = buf.search(/\r\n\r\n|\n\n|\r\r/)) !== -1) {
          const block = buf.slice(0, idx);
          buf = buf.slice(idx + (buf[idx] === '\r' && buf[idx + 1] === '\n' ? 4 : 2));
          const dataLines = block.split(/\r\n|\n|\r/).filter(l => l.startsWith('data:')).map(l => l.slice(5).trim());
          if (!dataLines.length) continue;
          const data = dataLines.join('\n');
          if (data === '[DONE]') { state.done = true; state.wallMs = Date.now() - started; continue; }
          let ev; try { ev = JSON.parse(data); } catch { continue; }
          state.events++;
          const ch = ev.choices && ev.choices[0];
          if (!ch) continue;
          const d = ch.delta || {};
          if (d.role) state.role = d.role;
          if (typeof d.content === 'string') state.content += d.content;
          if (typeof d.reasoning_content === 'string') state.reasoning += d.reasoning_content;
          if (ch.finish_reason) state.finish = ch.finish_reason;
          if (ev.usage) state.usage = ev.usage;
        }
      });
      res.on('end', () => { if (!state.wallMs) state.wallMs = Date.now() - started; resolve({ status: res.statusCode, state, headers: res.headers }); });
    });
    req.on('error', reject);
    req.setTimeout(120000, () => req.destroy(new Error('stream timeout')));
    req.write(payload);
    req.end();
  });
}

const chatBody = (content, extra = {}) => Object.assign({
  model: 'bonsai', stream: false, max_tokens: maxTokens,
  messages: [{ role: 'user', content }],
}, extra);

try {
  // --- static routes
  const root = await request('GET', '/');
  ok('GET / serves the WebUI', root.status === 200 && /text\/html/.test(root.headers['content-type'] || '') && root.text.includes('Project Bonsai'), `status=${root.status} bytes=${root.text.length}`);
  ok('no CORS header on /', !root.headers['access-control-allow-origin']);
  const idx = await request('GET', '/index.html');
  ok('GET /index.html serves the same page', idx.status === 200 && idx.text === root.text, `status=${idx.status}`);
  const fav = await request('GET', '/favicon.ico');
  ok('GET /favicon.ico is 204', fav.status === 204, `status=${fav.status}`);
  const models = await request('GET', '/v1/models');
  let modelsJson = null; try { modelsJson = JSON.parse(models.text); } catch { /* reported below */ }
  ok('GET /v1/models is an OpenAI list', models.status === 200 && modelsJson && modelsJson.object === 'list' && modelsJson.data[0].id, `status=${models.status} id=${modelsJson && modelsJson.data[0].id}`);
  const status0 = await request('GET', '/v1/status');
  let st = null; try { st = JSON.parse(status0.text); } catch { /* reported below */ }
  ok('GET /v1/status reports the measured engine', status0.status === 200 && st && st.engine_bytes > 0 && st.cap > 0, `status=${status0.status} engine_bytes=${st && st.engine_bytes} cap=${st && st.cap}`);
  console.log(`INFO | device=${st && st.device} engine_kib=${st && st.engine_kib} cap=${st && st.cap} v_quant=${st && st.v_quant} reference_tok_s=${st && st.reference_tok_s} thinking=${st && st.thinking}`);
  const health = await request('GET', '/health');
  let hj = null; try { hj = JSON.parse(health.text); } catch { /* below */ }
  ok('GET /health aliases status', health.status === 200 && hj && hj.engine_bytes > 0, `status=${health.status}`);
  const nope = await request('GET', '/definitely-not-a-route');
  ok('unknown route is 404', nope.status === 404, `status=${nope.status}`);
  const opt = await request('OPTIONS', '/v1/chat/completions');
  ok('OPTIONS is 405 (no CORS)', opt.status === 405 && !opt.headers['access-control-allow-origin'], `status=${opt.status}`);
  const getChat = await request('GET', '/v1/chat/completions');
  ok('GET on completions is 405', getChat.status === 405, `status=${getChat.status}`);
  const rebinding = await request('GET', '/', { headers: { Host: 'rebinding.example' } });
  ok('non-loopback Host is refused', rebinding.status === 400, `status=${rebinding.status}`);

  // --- unary completion
  const unary = await request('POST', '/v1/chat/completions', { body: chatBody('Say hello in one word.', { stream: false, max_tokens: 16 }) });
  let uj = null; try { uj = JSON.parse(unary.text); } catch { /* below */ }
  ok('unary completion returns a message', unary.status === 200 && uj && uj.choices && typeof uj.choices[0].message.content === 'string', `status=${unary.status} content=${uj && JSON.stringify(uj.choices[0].message.content).slice(0, 60)}`);
  ok('unary completion carries usage', !!(uj && uj.usage && uj.usage.completion_tokens > 0), `usage=${uj && JSON.stringify(uj.usage)}`);
  // With thinking on (the engine default), a 16-token budget is still inside the reasoning phase,
  // so the visible content is legitimately empty and the text lives in reasoning_content. Assert on
  // the ACTIVE channel, which /v1/status reports, instead of assuming content is always non-empty.
  const thinking = !!(st && st.thinking);
  const uMsg = uj && uj.choices && uj.choices[0].message;
  const uText = uMsg ? ((uMsg.content || '') + (uMsg.reasoning_content || '')) : '';
  ok(`unary completion produced text (${thinking ? 'thinking on' : 'thinking off'})`, uText.length > 0, `content_chars=${uMsg ? (uMsg.content || '').length : -1} reasoning_chars=${uMsg ? (uMsg.reasoning_content || '').length : -1}`);

  // --- streamed completion + the engine's own measured rate
  const sse = await streamCompletion(chatBody('用一句话解释三值量化。', { stream: true, max_tokens: maxTokens }));
  const s = sse.state;
  ok('SSE stream opens with the assistant role', sse.status === 200 && s.role === 'assistant', `status=${sse.status} role=${s.role}`);
  if (thinking) {
    // Same reasoning-phase fact as the unary leg: at this budget every delta is reasoning_content.
    // An empty content field here is the contract, not a regression.
    ok('SSE stream delivers reasoning deltas (thinking on)', s.reasoning.length > 0, `reasoning_chars=${s.reasoning.length} content_chars=${s.content.length} events=${s.events}`);
  } else {
    ok('SSE stream delivers content deltas (thinking off)', s.content.length > 0, `content_chars=${s.content.length} events=${s.events}`);
  }
  ok('SSE stream delivers text in the active channel', s.content.length + s.reasoning.length > 0, `content=${s.content.length} reasoning=${s.reasoning.length} events=${s.events}`);
  ok('SSE stream ends with finish_reason and DONE', !!s.finish && s.done, `finish=${s.finish} done=${s.done}`);
  ok('SSE stream carries usage on the finish chunk', !!(s.usage && s.usage.completion_tokens > 0), `usage=${JSON.stringify(s.usage)}`);
  ok('SSE headers disable buffering', /no-cache/.test(sse.headers?.['cache-control'] || '') && (sse.headers?.['x-accel-buffering'] === 'no'), `cache-control=${sse.headers?.['cache-control']}`);
  const wallRate = s.usage && s.wallMs ? (s.usage.completion_tokens * 1000 / s.wallMs) : 0;
  const status1 = await request('GET', '/v1/status');
  let st1 = null; try { st1 = JSON.parse(status1.text); } catch { /* below */ }
  const engineRate = st1 && Number(st1.last_tok_s) > 0 ? Number(st1.last_tok_s) : 0;
  console.log(`MEASURED | serve-mode engine last_tok_s=${engineRate} over last_ms=${st1 && st1.last_ms} (n_gen/ms_gen, excludes prefill)`);
  console.log(`MEASURED | client wall rate=${wallRate.toFixed(2)} tok/s over ${s.wallMs} ms for ${s.usage && s.usage.completion_tokens} tokens (includes prefill + network)`);
  ok('the engine reports a measured serve rate', engineRate > 0, `last_tok_s=${engineRate}`);

  // --- concurrency: one reply at a time
  const [a, b] = await Promise.all([
    request('POST', '/v1/chat/completions', { body: chatBody('并发甲', { max_tokens: 24 }) }),
    request('POST', '/v1/chat/completions', { body: chatBody('并发乙', { max_tokens: 24 }) }),
  ]);
  const codes = [a.status, b.status].sort();
  ok('a second concurrent completion gets 429', codes.includes(429) && codes.includes(200), `statuses=${codes.join(',')}`);
  const statusWhileBusy = await request('GET', '/v1/status');
  ok('status stays live while the GPU is busy', statusWhileBusy.status === 200, `status=${statusWhileBusy.status}`);

  // --- request validation
  const badJson = await request('POST', '/v1/chat/completions', { body: '{not json' });
  ok('malformed JSON body is 400', badJson.status === 400, `status=${badJson.status}`);
  const emptyMsgs = await request('POST', '/v1/chat/completions', { body: { model: 'bonsai', messages: [] } });
  ok('empty messages array is 400', emptyMsgs.status === 400, `status=${emptyMsgs.status}`);
  const cap = (st && st.cap) || 2048;
  const overflow = await request('POST', '/v1/chat/completions', { body: chatBody('token '.repeat(cap + 512), { max_tokens: 8 }) });
  ok('a prompt beyond the KV cap is 400', overflow.status === 400, `status=${overflow.status} cap=${cap}`);

  const failed = results.filter(r => !r.ok).length;
  console.log(`api-probe: ${results.length - failed}/${results.length} passed`);
  console.log(`SUMMARY | serve_engine_tok_s=${engineRate} client_wall_tok_s=${wallRate.toFixed(2)} cap=${cap} engine_bytes=${st && st.engine_bytes} thinking=${thinking ? 'on' : 'off'}`);
  if (failed) process.exitCode = 1;
} catch (err) {
  console.error('API PROBE FAILED: ' + err.message);
  process.exitCode = 1;
}