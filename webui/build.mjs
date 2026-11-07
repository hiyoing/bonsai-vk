// Single-file build for the Bonsai WebUI. Zero npm dependencies: Node built-ins only.
//
//   $ node webui/build.mjs
//
// Output goes to ../runtime_vk/index.html because serve_loop() resolves the page relative to the
// executable's directory (the same CWD rule that makes shaders/*.spv mandatory next to the exe).
// The dist assembly copies that file; nothing is ever hand-edited in the output.
import { readFileSync, writeFileSync, statSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const root = dirname(fileURLToPath(import.meta.url));
const read = p => readFileSync(join(root, p), 'utf8').replace(/^﻿/, '');

const MODULES = (process.env.BUILD_MODULES ||
  '00-core.js,10-api.js,20-markdown.js,30-ui.js,40-selftest.js')
  .split(',').map(s => s.trim()).filter(Boolean);

const OUT = process.env.BUILD_OUT || '../runtime_vk/index.html';

function build() {
  const template = read('src/template.html');
  const css = read('src/style.css');
  const js = MODULES
    .map(name => `// ===== ${name} =====\n${read(`src/js/${name}`)}`)
    .join('\n\n');

  // --- markers: a leftover marker means the page would ship a blank style/script block
  if (!template.includes('/*__CSS__*/')) throw new Error('template missing /*__CSS__*/ marker');
  if (!template.includes('//__JS__')) throw new Error('template missing //__JS__ marker');
  if (!css.trim()) throw new Error('style.css is empty');
  if (!js.trim()) throw new Error('JS bundle is empty');

  // --- the one sequence that would terminate the inline <script> early
  if (/<\/script/i.test(js)) throw new Error('JS payload contains a closing script tag');
  if (/<\/script/i.test(css)) throw new Error('CSS payload contains a closing script tag');

  const out = template
    .replace('/*__CSS__*/', () => css)
    .replace('//__JS__', () => js);

  if (out.includes('/*__CSS__*/') || out.includes('//__JS__')) throw new Error('marker replacement failed');

  // --- delivery contract, enforced at build time so a bad edit cannot reach the browser
  const forbidden = [
    [/<canvas[\s>]/i, 'canvas element'],
    [/<img[\s>]/i, 'img element'],
    [/<script[^>]+src=/i, 'external script'],
    [/<link[^>]+href=/i, 'external stylesheet'],
    [/@import\s/i, 'CSS @import'],
    [/\bTODO\b|\bFIXME\b|\bTBD\b/, 'placeholder token'],
    [/(?:\/\/|\/\*)[^\n]*(?:省略|待补充)/, 'placeholder comment'],
    [/\bnot implemented\b/i, 'placeholder prose'],
    [/lorem ipsum/i, 'placeholder prose']
  ];
  for (const [re, what] of forbidden) {
    const m = out.match(re);
    if (m) throw new Error(`forbidden ${what}: ${JSON.stringify(m[0])}`);
  }
  // Inline SVG is the only graphics path allowed; url() must not reach outside the file.
  for (const m of out.matchAll(/url\(([^)]*)\)/gi)) {
    const v = m[1].replace(/['"]/g, '').trim();
    if (v && !v.startsWith('#') && !v.startsWith('data:')) {
      throw new Error(`external url() reference: ${v}`);
    }
  }

  const dest = join(root, OUT);
  writeFileSync(dest, out, 'utf8');
  const bytes = statSync(dest).size;
  console.log(`built ${OUT} · ${bytes} B · ${(bytes / 1024).toFixed(1)} KiB · ${MODULES.length} modules · ${css.length} css chars · ${js.length} js chars`);
}

try {
  build();
} catch (err) {
  console.error(`BUILD FAILED: ${err.message}`);
  process.exit(1);
}