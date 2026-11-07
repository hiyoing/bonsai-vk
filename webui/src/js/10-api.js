// ===== 10-api.js =====
// The wire layer. Two routes matter: GET /v1/status for the badge, POST /v1/chat/completions for the
// reply. Paths are relative on purpose - the page is served by the engine itself, so there is no
// origin to configure and no CORS header to ask for.
//
// The SSE parser is a pure incremental state machine, separate from fetch, because the one thing
// that breaks naive stream readers is a chunk boundary landing inside "data:", inside a JSON string,
// or inside a multi-byte UTF-8 character. The self-test feeds the same byte stream one character at a
// time and compares the event list to the single-shot parse; they must be identical.

'use strict';

function ApiError(status, type, message) {
  const e = new Error(message);
  e.name = 'ApiError';
  e.status = status;
  e.type = type;
  return e;
}

// One SSE event block -> payload lines joined the way the spec says, then either [DONE] or JSON.
// Comment lines (starting with ':') and the event/id/retry fields are ignored: this server sends none
// of them, and honouring them would only add ways to be wrong.
function sseBlockToEvent(block) {
  const lines = block.split(/\r\n|\n|\r/);
  const data = [];
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (line.charAt(0) === ':') continue;
    const colon = line.indexOf(':');
    const field = colon === -1 ? line : line.slice(0, colon);
    if (field !== 'data') continue;
    let value = colon === -1 ? '' : line.slice(colon + 1);
    if (value.charAt(0) === ' ') value = value.slice(1);
    data.push(value);
  }
  if (!data.length) return null;
  const payload = data.join('\n');
  if (payload === '[DONE]') return { done: true, data: null };
  let obj;
  try {
    obj = JSON.parse(payload);
  } catch (err) {
    return { malformed: true, raw: payload, data: null };
  }
  return { done: false, data: obj };
}

function makeSseParser(onEvent) {
  let buf = '';
  const sep = /\r\n\r\n|\n\n|\r\r/;
  function drain(final) {
    for (;;) {
      const m = sep.exec(buf);
      if (!m) break;
      const block = buf.slice(0, m.index);
      buf = buf.slice(m.index + m[0].length);
      const ev = sseBlockToEvent(block);
      if (ev) onEvent(ev);
    }
    if (final && buf.trim()) {
      const ev = sseBlockToEvent(buf);
      buf = '';
      if (ev) onEvent(ev);
    }
  }
  return {
    feed: function (text) { buf += text; drain(false); },
    end: function () { drain(true); },
    pending: function () { return buf; }
  };
}

// Pull the pieces the UI cares about out of one chat.completion.chunk. Returns null for anything
// that is not a usable delta, so a keep-alive or a malformed line cannot corrupt the transcript.
function chunkParts(ev) {
  if (!ev || ev.done || ev.malformed || !ev.data) return null;
  const d = ev.data;
  const out = { content: '', reasoning: '', finish: null, usage: null, id: null, role: null };
  if (typeof d.id === 'string') out.id = d.id;
  if (d.usage && typeof d.usage === 'object') out.usage = d.usage;
  const ch = Array.isArray(d.choices) ? d.choices[0] : null;
  if (!ch) return out;
  if (typeof ch.finish_reason === 'string') out.finish = ch.finish_reason;
  const delta = ch.delta || {};
  if (typeof delta.role === 'string') out.role = delta.role;
  if (typeof delta.content === 'string') out.content = delta.content;
  if (typeof delta.reasoning_content === 'string') out.reasoning = delta.reasoning_content;
  return out;
}

async function readErrorBody(res) {
  let text = '';
  try { text = await res.text(); } catch (err) { text = ''; }
  let msg = text.slice(0, 300);
  let type = 'server_error';
  try {
    const j = JSON.parse(text);
    if (j && j.error) {
      if (typeof j.error.message === 'string') msg = j.error.message;
      if (typeof j.error.type === 'string') type = j.error.type;
    }
  } catch (err) { /* a non-JSON body keeps its raw text as the message */ }
  return ApiError(res.status, type, msg || ('HTTP ' + res.status));
}

const realTransport = {
  name: 'fetch',

  async status(signal) {
    const res = await fetch('/v1/status', { signal: signal, cache: 'no-store', credentials: 'omit' });
    if (!res.ok) throw await readErrorBody(res);
    return await res.json();
  },

  // onEvent receives {done} / {data} / {malformed}. Resolves when the stream ends, rejects on a
  // non-2xx response or a transport failure. An AbortError is re-thrown untouched so the caller can
  // tell "the user pressed stop" apart from "the network died".
  async completeStream(body, onEvent, signal) {
    const res = await fetch('/v1/chat/completions', {
      method: 'POST',
      signal: signal,
      credentials: 'omit',
      headers: { 'Content-Type': 'application/json', 'Accept': 'text/event-stream' },
      body: JSON.stringify(body)
    });
    if (!res.ok) throw await readErrorBody(res);
    if (!res.body) throw ApiError(0, 'transport_error', 'this browser exposed no response body to stream');
    const reader = res.body.getReader();
    const dec = new TextDecoder('utf-8');
    const parser = makeSseParser(onEvent);
    try {
      for (;;) {
        const step = await reader.read();
        if (step.done) break;
        parser.feed(dec.decode(step.value, { stream: true }));
      }
      parser.feed(dec.decode());
      parser.end();
    } finally {
      try { reader.releaseLock(); } catch (err) { /* already released */ }
    }
  },

  async completeUnary(body, signal) {
    const res = await fetch('/v1/chat/completions', {
      method: 'POST',
      signal: signal,
      credentials: 'omit',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    if (!res.ok) throw await readErrorBody(res);
    return await res.json();
  }
};

function transport() {
  return S.transport || realTransport;
}

// The transcript the server renders. Only the three roles its no-tools template accepts are sent;
// error notes and UI chrome are never part of it.
function buildMessages(turns, systemPrompt) {
  const out = [];
  if (systemPrompt) out.push({ role: 'system', content: systemPrompt });
  for (let i = 0; i < turns.length; i++) {
    const t = turns[i];
    if (t.role !== 'user' && t.role !== 'assistant') continue;
    if (t.role === 'assistant' && t.error) continue;   // a failed reply must not poison the next prompt
    if (!t.text && !t.reasoning) continue;
    out.push({ role: t.role, content: t.text || '' });
  }
  return out;
}

function requestBody(messages, opts) {
  const b = {
    model: (S.status && S.status.model) || 'bonsai',
    messages: messages,
    stream: true,
    max_tokens: opts && opts.maxTokens ? opts.maxTokens : 1024
  };
  if (opts && typeof opts.temperature === 'number') b.temperature = opts.temperature;
  if (opts && typeof opts.topP === 'number') b.top_p = opts.topP;
  if (opts && typeof opts.seed === 'number') b.seed = opts.seed;
  return b;
}