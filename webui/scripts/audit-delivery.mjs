#!/usr/bin/env node
import { readFileSync, statSync } from 'node:fs';
import { resolve } from 'node:path';

const argv = process.argv.slice(2);
const VALUE_FLAGS = new Set(['--file', '--accent', '--accent-token', '--allow-accent', '--allow-url', '--max-kb']);
const KNOWN_FLAGS = new Set([...VALUE_FLAGS, '--strict-accent', '--json', '--verbose', '--help']);

function take(name, fallback = '') {
  const i = argv.indexOf(name);
  if (i === -1) return fallback;
  const v = argv[i + 1];
  return v === undefined || (v.startsWith('--') && KNOWN_FLAGS.has(v)) ? fallback : v;
}
function flag(name) {
  return argv.includes(name);
}
function positionalFile() {
  for (let i = 0; i < argv.length; i++) {
    if (argv[i].startsWith('--')) {
      if (VALUE_FLAGS.has(argv[i])) i++;
      continue;
    }
    return argv[i];
  }
  return '';
}

const file = resolve(take('--file', positionalFile() || 'index.html'));
const accentInput = take('--accent', '#d97757');
const accentTokens = take('--accent-token', '').split(',').map(s => s.trim()).filter(Boolean);
const allowAccent = take('--allow-accent', 'selected|attack|target|hover|focus|active');
const allowUrls = take('--allow-url', 'www.w3.org').split(',').map(s => s.trim()).filter(Boolean);
const maxKb = Number(take('--max-kb', '0')) || 0;
const strictAccent = flag('--strict-accent');
const jsonOut = flag('--json');
const verbose = flag('--verbose');

let html;
try {
  html = readFileSync(file, 'utf8');
} catch (err) {
  console.error(`FAIL cannot read ${file}: ${err.message}`);
  process.exit(2);
}

const sizeKb = Number((statSync(file).size / 1024).toFixed(1));
const lines = html.split(/\r?\n/);
const lineStarts = [0];
for (let i = 0; i < html.length; i++) {
  if (html[i] === '\n') lineStarts.push(i + 1);
}
function lineOf(index) {
  let lo = 0;
  let hi = lineStarts.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (lineStarts[mid] <= index) lo = mid;
    else hi = mid - 1;
  }
  return lo + 1;
}
function lineText(n) {
  return (lines[n - 1] || '').trim().slice(0, 180);
}
function snippet(index, len = 140) {
  const start = Math.max(0, html.lastIndexOf('\n', index) + 1);
  const end = html.indexOf('\n', index);
  return html.slice(start, end === -1 ? Math.min(html.length, start + len) : Math.min(end, start + len)).trim();
}

const passes = [];
const failures = [];
const warnings = [];
function pass(msg) {
  passes.push(msg);
}
function fail(msg) {
  failures.push(msg);
}
function warn(msg) {
  warnings.push(msg);
}
function check(cond, passMsg, failMsg = passMsg) {
  if (cond) pass(passMsg);
  else fail(failMsg);
}
function failMatches(matches, label) {
  for (const m of matches) {
    fail(`${label} at line ${lineOf(m.index)}: ${snippet(m.index)}`);
  }
  if (!matches.length) pass(`no ${label}`);
}

if (maxKb && sizeKb > maxKb) fail(`file is ${sizeKb} KB, exceeds --max-kb ${maxKb}`);
else if (maxKb) pass(`file size ${sizeKb} KB <= ${maxKb} KB`);

failMatches([...html.matchAll(/<canvas\b[^>]*>/gi)], '<canvas>');
failMatches([...html.matchAll(/<img\b[^>]*>/gi)], '<img>');

const scriptSrc = [...html.matchAll(/<script\b[^>]*\bsrc\s*=\s*["']?([^"'\s>]+)/gi)];
for (const m of scriptSrc) fail(`external <script src> at line ${lineOf(m.index)}: ${m[0].slice(0, 160)}`);
if (!scriptSrc.length) pass('no external <script src>');

const linkHref = [...html.matchAll(/<link\b[^>]*\bhref\s*=\s*["']?([^"'\s>]+)/gi)];
let externalLinks = 0;
for (const m of linkHref) {
  const href = m[1];
  if (/^(data:|#|about:)/i.test(href)) continue;
  externalLinks++;
  fail(`external <link href> at line ${lineOf(m.index)}: ${href.slice(0, 120)}`);
}
if (!externalLinks) pass('no external <link href>');

const imports = [...html.matchAll(/@import\b[^;]*/gi)];
failMatches(imports, '@import');

const urlValues = [...html.matchAll(/url\(\s*(['"]?)([^)'"]+)\1\s*\)/gi)];
let externalUrls = 0;
for (const m of urlValues) {
  const value = m[2].trim();
  if (/^(data:|#|about:|blob:|var\()/i.test(value)) continue;
  externalUrls++;
  fail(`external url() at line ${lineOf(m.index)}: ${value.slice(0, 120)}`);
}
if (!externalUrls) pass('no external url()');

const styleTags = [...html.matchAll(/<style\b[^>]*>([\s\S]*?)<\/style>/gi)];
const scriptTags = [...html.matchAll(/<script\b[^>]*>([\s\S]*?)<\/script>/gi)];
check(styleTags.length >= 1, 'inline <style> present', 'no inline <style>');
check(scriptTags.length >= 1, 'inline <script> present', 'no inline <script>');
if (styleTags.length > 1) warn(`${styleTags.length} <style> blocks; single-file delivery usually uses one`);
if (scriptTags.length > 1) warn(`${scriptTags.length} <script> blocks; single-file delivery usually uses one`);

const css = styleTags.map(m => m[1]).join('\n');
const js = scriptTags.map(m => m[1]).join('\n');
check(Boolean(css.trim()), 'inline CSS is non-empty', 'inline CSS is empty');
check(Boolean(js.trim()), 'inline JS is non-empty', 'inline JS is empty');

const markerRe = /\b(?:__CSS__|__JS__|__HTML__)\b/g;
const markers = [...html.matchAll(markerRe)];
failMatches(markers, 'unresolved build marker');

const placeholderPatterns = [
  { re: /\b(?:TODO|FIXME|TBD)\b/g, label: 'unresolved marker' },
  { re: /(?:\/\/|\/\*|<!--)\s*(?:省略|待补充|待实现|此处省略|\.\.\.|…)/g, label: 'placeholder comment' },
  { re: /\b(?:lorem ipsum|not implemented)\b/gi, label: 'stub text' },
];
let placeholderCount = 0;
for (const { re, label } of placeholderPatterns) {
  for (const m of html.matchAll(re)) {
    placeholderCount++;
    fail(`${label} at line ${lineOf(m.index)}: ${snippet(m.index)}`);
  }
}
if (!placeholderCount) pass('no placeholder markers');

function hostAllowed(host) {
  const h = host.toLowerCase();
  return allowUrls.some(a => h === a.toLowerCase() || h.endsWith('.' + a.toLowerCase()));
}
const httpUrls = [...html.matchAll(/https?:\/\/[^\s"'<>)]+/gi)];
let externalHosts = 0;
for (const m of httpUrls) {
  let host = '';
  try {
    host = new URL(m[0]).hostname;
  } catch {
    externalHosts++;
    fail(`unparseable external URL at line ${lineOf(m.index)}: ${m[0].slice(0, 160)}`);
    continue;
  }
  if (hostAllowed(host)) continue;
  externalHosts++;
  fail(`external URL at line ${lineOf(m.index)}: ${m[0].slice(0, 160)}`);
}
if (!externalHosts) pass('no external URLs outside allowlist');

function normalizeHex(input) {
  let h = String(input).trim().toLowerCase();
  if (!h.startsWith('#')) h = '#' + h;
  if (/^#[0-9a-f]{3}$/.test(h)) h = '#' + h[1] + h[1] + h[2] + h[2] + h[3] + h[3];
  if (!/^#[0-9a-f]{6}$/.test(h)) throw new Error(`invalid --accent value: ${input}`);
  return h;
}
function hexToRgb(hex) {
  return {
    r: parseInt(hex.slice(1, 3), 16),
    g: parseInt(hex.slice(3, 5), 16),
    b: parseInt(hex.slice(5, 7), 16),
  };
}
function escapeRegExp(value) {
  return String(value).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}
function stripCssComments(value) {
  return value.replace(/\/\*[\s\S]*?\*\//g, '');
}
function skipTrivia(value, index) {
  let i = index;
  while (i < value.length) {
    if (/\s/.test(value[i])) {
      i++;
      continue;
    }
    if (value[i] === '/' && value[i + 1] === '*') {
      const end = value.indexOf('*/', i + 2);
      i = end === -1 ? value.length : end + 2;
      continue;
    }
    break;
  }
  return i;
}
function parseCssRules(source) {
  const rules = [];
  const stack = [];
  let i = 0;
  let start = 0;
  while (i < source.length) {
    if (source[i] === '/' && source[i + 1] === '*') {
      const end = source.indexOf('*/', i + 2);
      i = end === -1 ? source.length : end + 2;
      continue;
    }
    if (source[i] === '{') {
      const rawPrelude = stripCssComments(source.slice(start, i)).trim();
      if (rawPrelude.startsWith('@')) {
        stack.push(rawPrelude);
        start = i + 1;
        i++;
        continue;
      }
      let depth = 1;
      let j = i + 1;
      while (j < source.length && depth > 0) {
        if (source[j] === '/' && source[j + 1] === '*') {
          const end = source.indexOf('*/', j + 2);
          j = end === -1 ? source.length : end + 2;
          continue;
        }
        if (source[j] === '{') depth++;
        else if (source[j] === '}') depth--;
        j++;
      }
      rules.push({
        selector: [...stack, rawPrelude].filter(Boolean).join(' '),
        rawSelector: rawPrelude,
        body: source.slice(i + 1, Math.max(i + 1, j - 1)),
        selectorIndex: skipTrivia(source, start),
      });
      start = j;
      i = j;
      continue;
    }
    if (source[i] === '}') {
      stack.pop();
      start = i + 1;
      i++;
      continue;
    }
    i++;
  }
  return rules;
}
function splitDeclarations(body) {
  const out = [];
  let current = '';
  let depth = 0;
  let quote = null;
  for (const ch of body) {
    if (quote) {
      current += ch;
      if (ch === quote) quote = null;
      continue;
    }
    if (ch === '"' || ch === "'") {
      quote = ch;
      current += ch;
      continue;
    }
    if (ch === '(') depth++;
    else if (ch === ')') depth = Math.max(0, depth - 1);
    if (ch === ';' && depth === 0) {
      if (current.trim()) out.push(current.trim());
      current = '';
      continue;
    }
    current += ch;
  }
  if (current.trim()) out.push(current.trim());
  return out;
}

let accentHex;
try {
  accentHex = normalizeHex(accentInput);
} catch (err) {
  console.error(`FAIL ${err.message}`);
  process.exit(2);
}
const rgb = hexToRgb(accentHex);
const accentPatterns = [
  new RegExp(escapeRegExp(accentHex), 'i'),
  new RegExp(`rgba?\\(\\s*${rgb.r}\\s*,\\s*${rgb.g}\\s*,\\s*${rgb.b}\\b`, 'i'),
];
for (const token of accentTokens) {
  accentPatterns.push(new RegExp(`var\\(\\s*${escapeRegExp(token)}\\s*\\)`, 'i'));
}
function matchesAccent(value) {
  return accentPatterns.some(re => re.test(value));
}
const allowAccentRe = allowAccent ? new RegExp(allowAccent, 'i') : null;
const accentUsages = [];
for (const styleTag of styleTags) {
  const contentStart = styleTag.index + styleTag[0].indexOf('>') + 1;
  for (const rule of parseCssRules(styleTag[1])) {
    for (const declaration of splitDeclarations(stripCssComments(rule.body))) {
      if (!matchesAccent(declaration)) continue;
      if (rule.rawSelector.trim() === ':root' && /^--[\w-]+\s*:/.test(declaration)) continue;
      const line = lineOf(contentStart + rule.selectorIndex);
      const allowed = Boolean(allowAccentRe && allowAccentRe.test(rule.selector));
      if (allowed) {
        accentUsages.push({ status: 'ok', line, selector: rule.selector, declaration });
        pass(`accent allowed at line ${line}: ${rule.selector}`);
      } else if (strictAccent) {
        accentUsages.push({ status: 'fail', line, selector: rule.selector, declaration });
        fail(`accent not in --allow-accent at line ${line}: ${rule.selector} | ${declaration.slice(0, 140)}`);
      } else {
        accentUsages.push({ status: 'warn', line, selector: rule.selector, declaration });
        warn(`accent needs review at line ${line}: ${rule.selector} | ${declaration.slice(0, 140)}`);
      }
    }
  }
}
for (const m of html.matchAll(/\bstyle\s*=\s*(["'])([\s\S]*?)\1/gi)) {
  if (!matchesAccent(m[2])) continue;
  accentUsages.push({ status: 'warn', line: lineOf(m.index), selector: '[inline style]', declaration: m[2].slice(0, 140) });
  warn(`accent in inline style at line ${lineOf(m.index)}: ${m[2].slice(0, 140)}`);
}
for (const scriptTag of scriptTags) {
  if (!matchesAccent(scriptTag[1])) continue;
  const offset = scriptTag[1].search(accentPatterns[0]);
  const index = scriptTag.index + (offset >= 0 ? offset : 0);
  accentUsages.push({ status: 'warn', line: lineOf(index), selector: '[script]', declaration: 'accent value in JS requires manual review' });
  warn(`accent value in JS near line ${lineOf(index)}; confirm it is only used for allowed semantics`);
}

if (!accentUsages.length) pass('no accent-color usage found');

if (jsonOut) {
  console.log(JSON.stringify({
    file,
    sizeKb,
    accent: accentHex,
    accentTokens,
    allowAccent,
    passed: passes.length,
    failed: failures.length,
    warnings: warnings.length,
    failures,
    warnings,
    accentUsages,
    checks: passes,
  }, null, 2));
} else {
  console.log(`File: ${file}`);
  console.log(`Size: ${sizeKb} KB`);
  if (verbose) {
    for (const msg of passes) console.log(`PASS  ${msg}`);
  }
  for (const msg of warnings) console.log(`WARN  ${msg}`);
  for (const msg of failures) console.log(`FAIL  ${msg}`);
  if (accentUsages.length) {
    console.log('Accent usages:');
    for (const item of accentUsages) {
      console.log(`  ${item.status.toUpperCase().padEnd(4)} line ${item.line} ${item.selector} -> ${item.declaration.slice(0, 100)}`);
    }
  }
  console.log(`Summary: ${passes.length} passed, ${failures.length} failed, ${warnings.length} warning(s)`);
}

process.exit(failures.length ? 1 : 0);
