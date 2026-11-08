// ===== 20-markdown.js =====
// A hand-written Markdown renderer. There is no library here and there is no innerHTML anywhere in
// this file: every node is created with createElement and every piece of model text goes in through
// textContent. That is the whole XSS story - a reply containing a raw image tag with an onerror
// handler renders as the literal characters, because nothing ever parses them as markup.
//
// It is also written to be fed a PREFIX of a document, because that is what streaming delivers. An
// unclosed code fence renders as a code block containing what has arrived so far; an unclosed ** is
// literal asterisks until its partner shows up. Re-rendering the growing prefix is therefore stable
// instead of flickering between two interpretations.
//
// Images are deliberately not supported: the delivery contract forbids the image element. ![alt](url)
// becomes a link whose text is the alt, so the information survives without fetching anything.

'use strict';

const MD_MAX_DEPTH = 12;

function mdIsPunct(c) {
  return /[!-/:-@[-`{-~]/.test(c);
}

function mdRunLen(text, i, ch) {
  let n = 0;
  while (i + n < text.length && text.charAt(i + n) === ch) n++;
  return n;
}

// The next run of `ch` at or after `from` whose length is at least `want`. Returns -1 if there is
// none, which is the streaming case: the delimiter stays literal.
function mdFindDelim(text, from, ch, want) {
  let i = from;
  while (i < text.length) {
    if (text.charAt(i) === ch) {
      const n = mdRunLen(text, i, ch);
      if (n >= want) return i;
      i += n;
      continue;
    }
    // A code span can contain delimiter characters; skipping it keeps ** inside `code` literal.
    if (text.charAt(i) === '`') {
      const n = mdRunLen(text, i, '`');
      let j = i + n;
      let closed = false;
      while (j < text.length) {
        if (text.charAt(j) === '`') {
          const m = mdRunLen(text, j, '`');
          if (m === n) { j += m; closed = true; break; }
          j += m;
          continue;
        }
        j++;
      }
      if (closed) { i = j; continue; }
    }
    i++;
  }
  return -1;
}

function safeHref(raw) {
  const s = String(raw).trim().replace(/[\u0000-\u001f\u007f]/g, '');
  if (!s) return null;
  const m = /^([a-zA-Z][a-zA-Z0-9+.-]*):/.exec(s);
  if (m) {
    const p = m[1].toLowerCase();
    if (p !== 'http' && p !== 'https' && p !== 'mailto') return null;
  }
  return s;
}

function mdCodeSpan(raw) {
  let s = raw.replace(/\r\n|\n|\r/g, ' ');
  if (s.length > 1 && s.charAt(0) === ' ' && s.charAt(s.length - 1) === ' ' && /[^ ]/.test(s)) s = s.slice(1, -1);
  const n = mk('code');
  n.textContent = s;
  return n;
}

function mdLink(text, href, depth) {
  const url = safeHref(href);
  if (url === null) {
    const span = mk('span');
    appendAll(span, mdInline(text, depth + 1));
    return span;
  }
  const a = mk('a', 'md-link');
  a.setAttribute('href', url);
  a.setAttribute('rel', 'noopener noreferrer nofollow');
  a.setAttribute('target', '_blank');
  appendAll(a, mdInline(text, depth + 1));
  return a;
}

function appendAll(parent, nodes) {
  for (let i = 0; i < nodes.length; i++) parent.appendChild(nodes[i]);
}

// [text](href) and ![alt](href). The bracket part is matched by counting, so nested brackets in the
// label work; the href stops at the first unescaped ')' which is what the model actually emits.
function mdTryLink(text, i, depth) {
  const isImage = text.charAt(i) === '!';
  const open = isImage ? i + 1 : i;
  if (text.charAt(open) !== '[') return null;
  let d = 0;
  let j = open;
  for (; j < text.length; j++) {
    const c = text.charAt(j);
    if (c === '\\') { j++; continue; }
    if (c === '[') d++;
    else if (c === ']') { d--; if (d === 0) break; }
  }
  if (j >= text.length || d !== 0) return null;
  if (text.charAt(j + 1) !== '(') return null;
  let k = j + 2;
  let href = '';
  while (k < text.length && text.charAt(k) !== ')') {
    if (text.charAt(k) === '\\' && k + 1 < text.length) { href += text.charAt(k + 1); k += 2; continue; }
    href += text.charAt(k);
    k++;
  }
  if (k >= text.length) return null;
  // An optional "title" is accepted and dropped: there is nowhere to put it that would not be a
  // hover tooltip carrying model-controlled text.
  const tm = /\s+"[^"]*"$/.exec(href) || /\s+'[^']*'$/.exec(href);
  if (tm) href = href.slice(0, tm.index);
  const label = text.slice(open + 1, j);
  return { node: mdLink(isImage ? (label || href) : label, href, depth), end: k + 1 };
}

function mdInline(text, depth) {
  const nodes = [];
  if (depth > MD_MAX_DEPTH) return [document.createTextNode(text)];
  let buf = '';
  let i = 0;
  function flush() {
    if (buf) { nodes.push(document.createTextNode(buf)); buf = ''; }
  }
  while (i < text.length) {
    const c = text.charAt(i);

    if (c === '\\' && i + 1 < text.length && mdIsPunct(text.charAt(i + 1))) {
      buf += text.charAt(i + 1);
      i += 2;
      continue;
    }

    if (c === '`') {
      const n = mdRunLen(text, i, '`');
      let j = i + n;
      let close = -1;
      while (j < text.length) {
        if (text.charAt(j) === '`') {
          const m = mdRunLen(text, j, '`');
          if (m === n) { close = j; break; }
          j += m;
          continue;
        }
        j++;
      }
      if (close >= 0) {
        flush();
        nodes.push(mdCodeSpan(text.slice(i + n, close)));
        i = close + n;
        continue;
      }
    }

    if (c === '[' || (c === '!' && text.charAt(i + 1) === '[')) {
      const got = mdTryLink(text, i, depth);
      if (got) { flush(); nodes.push(got.node); i = got.end; continue; }
    }

    if (c === '<') {
      const m = /^<([a-zA-Z][a-zA-Z0-9+.-]*:[^<>\s]+|[^<>\s]+@[^<>\s]+)>/.exec(text.slice(i));
      if (m) {
        const raw = m[1];
        const href = raw.indexOf('@') > 0 && raw.indexOf(':') === -1 ? ('mailto:' + raw) : raw;
        if (safeHref(href) !== null) {
          flush();
          nodes.push(mdLink(raw, href, depth));
          i += m[0].length;
          continue;
        }
      }
    }

    if (c === '~') {
      const n = mdRunLen(text, i, '~');
      if (n >= 2) {
        const close = mdFindDelim(text, i + 2, '~', 2);
        if (close > i + 2) {
          flush();
          const del = mk('del');
          appendAll(del, mdInline(text.slice(i + 2, close), depth + 1));
          nodes.push(del);
          i = close + 2;
          continue;
        }
      }
    }

    if (c === '*' || c === '_') {
      const n = mdRunLen(text, i, c);
      // An underscore inside a word is not emphasis (snake_case must survive intact).
      const intraword = c === '_' && i > 0 && /[A-Za-z0-9]/.test(text.charAt(i - 1));
      if (!intraword) {
        const want = n >= 3 ? 3 : (n === 2 ? 2 : 1);
        const close = mdFindDelim(text, i + want, c, want);
        if (close > i + want) {
          const inner = text.slice(i + want, close);
          if (inner.trim()) {
            flush();
            let node;
            if (want === 3) {
              node = mk('strong');
              const em = mk('em');
              appendAll(em, mdInline(inner, depth + 1));
              node.appendChild(em);
            } else {
              node = mk(want === 2 ? 'strong' : 'em');
              appendAll(node, mdInline(inner, depth + 1));
            }
            nodes.push(node);
            i = close + want;
            continue;
          }
        }
      }
    }

    buf += c;
    i++;
  }
  flush();
  return nodes;
}

// A paragraph's soft line breaks become spaces, except the two hard-break forms Markdown defines.
function mdParagraph(lines, depth) {
  const p = mk('p');
  for (let i = 0; i < lines.length; i++) {
    let line = lines[i];
    let hard = false;
    if (/\\$/.test(line)) { line = line.slice(0, -1); hard = true; }
    else if (/ {2,}$/.test(line)) { line = line.replace(/ +$/, ''); hard = true; }
    appendAll(p, mdInline(line, depth));
    if (i < lines.length - 1) p.appendChild(hard ? mk('br') : document.createTextNode(' '));
  }
  return p;
}

const RE_FENCE = /^ {0,3}(`{3,}|~{3,})[ \t]*([^`~]*)$/;
const RE_HEAD = /^ {0,3}(#{1,6})[ \t]+(.*)$/;
const RE_HR = /^ {0,3}(?:(?:\*[ \t]*){3,}|(?:-[ \t]*){3,}|(?:_[ \t]*){3,})$/;
const RE_QUOTE = /^ {0,3}>[ \t]?(.*)$/;
const RE_ITEM = /^([ \t]*)([*+-]|\d{1,9}[.)])[ \t]+(.*)$/;
const RE_TABLE_SEP = /^[ \t]*\|?[ \t]*:?-+:?[ \t]*(\|[ \t]*:?-+:?[ \t]*)+\|?[ \t]*$/;

function mdIndentWidth(s) {
  let w = 0;
  for (let i = 0; i < s.length; i++) {
    if (s.charAt(i) === ' ') w += 1;
    else if (s.charAt(i) === '\t') w += 4 - (w % 4);
    else break;
  }
  return w;
}

function mdSplitRow(line) {
  const cells = [];
  let cur = '';
  let inCode = false;
  for (let i = 0; i < line.length; i++) {
    const c = line.charAt(i);
    if (c === '\\' && i + 1 < line.length && line.charAt(i + 1) === '|') { cur += '|'; i++; continue; }
    if (c === '`') inCode = !inCode;
    if (c === '|' && !inCode) { cells.push(cur); cur = ''; continue; }
    cur += c;
  }
  cells.push(cur);
  let a = 0;
  let b = cells.length;
  if (cells.length && cells[0].trim() === '' && /^[ \t]*\|/.test(line)) a = 1;
  if (b > a && cells[b - 1].trim() === '' && /\|[ \t]*$/.test(line)) b--;
  return cells.slice(a, b).map(function (s) { return s.trim(); });
}

function mdTableAlign(sepCells) {
  return sepCells.map(function (s) {
    const l = s.charAt(0) === ':';
    const r = s.charAt(s.length - 1) === ':';
    return l && r ? 'center' : (r ? 'right' : (l ? 'left' : ''));
  });
}

function mdStartsBlock(line) {
  return RE_FENCE.test(line) || RE_HEAD.test(line) || RE_HR.test(line) ||
         RE_QUOTE.test(line) || RE_ITEM.test(line);
}

function mdToFragment(src, depth) {
  const d = depth || 0;
  const frag = document.createDocumentFragment();
  if (d > MD_MAX_DEPTH) {
    frag.appendChild(mdParagraph([String(src)], d));
    return frag;
  }
  const lines = String(src).replace(/\r\n|\r/g, '\n').split('\n');
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) { i++; continue; }

    const fence = RE_FENCE.exec(line);
    if (fence) {
      const marker = fence[1].charAt(0);
      const want = fence[1].length;
      const lang = (fence[2] || '').trim().split(/\s+/)[0] || '';
      const body = [];
      i++;
      let closed = false;
      while (i < lines.length) {
        const m = /^ {0,3}(`{3,}|~{3,})[ \t]*$/.exec(lines[i]);
        if (m && m[1].charAt(0) === marker && m[1].length >= want) { closed = true; i++; break; }
        body.push(lines[i]);
        i++;
      }
      frag.appendChild(mdCodeBlock(body.join('\n'), lang, closed));
      continue;
    }

    if (RE_HR.test(line)) { frag.appendChild(mk('hr')); i++; continue; }

    const head = RE_HEAD.exec(line);
    if (head) {
      const level = head[1].length;
      let text = head[2].replace(/[ \t]+#+[ \t]*$/, '').replace(/#+$/, '');
      const h = mk('h' + level);
      appendAll(h, mdInline(text.trim(), d));
      frag.appendChild(h);
      i++;
      continue;
    }

    if (RE_QUOTE.test(line)) {
      const body = [];
      while (i < lines.length) {
        const m = RE_QUOTE.exec(lines[i]);
        if (m) { body.push(m[1]); i++; continue; }
        // Lazy continuation: a plain line right after a quote line still belongs to the quote.
        if (lines[i].trim() && !mdStartsBlock(lines[i]) && body.length && body[body.length - 1].trim()) {
          body.push(lines[i]);
          i++;
          continue;
        }
        break;
      }
      const q = mk('blockquote');
      q.appendChild(mdToFragment(body.join('\n'), d + 1));
      frag.appendChild(q);
      continue;
    }

    const item = RE_ITEM.exec(line);
    if (item) {
      const got = mdList(lines, i, d);
      frag.appendChild(got.node);
      i = got.end;
      continue;
    }

    if (line.indexOf('|') !== -1 && i + 1 < lines.length && RE_TABLE_SEP.test(lines[i + 1]) &&
        lines[i + 1].indexOf('-') !== -1) {
      const headCells = mdSplitRow(line);
      const sepCells = mdSplitRow(lines[i + 1]);
      if (headCells.length === sepCells.length && headCells.length > 0) {
        const align = mdTableAlign(sepCells);
        const table = mk('table');
        const thead = mk('thead');
        const hr = mk('tr');
        for (let c = 0; c < headCells.length; c++) {
          const th = mk('th');
          if (align[c]) th.setAttribute('style', 'text-align:' + align[c]);
          appendAll(th, mdInline(headCells[c], d));
          hr.appendChild(th);
        }
        thead.appendChild(hr);
        table.appendChild(thead);
        const tbody = mk('tbody');
        i += 2;
        while (i < lines.length && lines[i].trim() && lines[i].indexOf('|') !== -1 && !mdStartsBlock(lines[i])) {
          const cells = mdSplitRow(lines[i]);
          const tr = mk('tr');
          for (let c = 0; c < headCells.length; c++) {
            const td = mk('td');
            if (align[c]) td.setAttribute('style', 'text-align:' + align[c]);
            appendAll(td, mdInline(cells[c] === undefined ? '' : cells[c], d));
            tr.appendChild(td);
          }
          tbody.appendChild(tr);
          i++;
        }
        table.appendChild(tbody);
        frag.appendChild(table);
        continue;
      }
    }

    const para = [];
    while (i < lines.length && lines[i].trim() && !mdStartsBlock(lines[i])) {
      // A table delimiter row ends the paragraph above it and starts a table on the next pass.
      if (para.length && RE_TABLE_SEP.test(lines[i]) && para[para.length - 1].indexOf('|') !== -1) break;
      para.push(lines[i]);
      i++;
    }
    if (!para.length) { para.push(lines[i]); i++; }
    frag.appendChild(mdParagraph(para, d));
  }
  return frag;
}

function mdCodeBlock(code, lang, closed) {
  const wrap = mk('div', 'codeblock');
  const bar = mk('div', 'codebar');
  const label = mk('span', 'code-lang', lang || 'text');
  const btn = mk('button', 'copy-btn', i18n('copy'));
  btn.type = 'button';
  btn.setAttribute('data-copy', '1');
  bar.appendChild(label);
  bar.appendChild(btn);
  const pre = mk('pre');
  const c = mk('code');
  c.textContent = code + (closed ? '' : '\n');
  pre.appendChild(c);
  wrap.appendChild(bar);
  wrap.appendChild(pre);
  if (!closed) wrap.setAttribute('data-partial', '1');
  return wrap;
}

// Lists are built from indent widths rather than a fixed two-space rule, because the model mixes
// them. Each item's content is dedented by its own marker width and parsed recursively, which is what
// makes a nested list, a code fence inside an item, and a loose multi-paragraph item all work.
function mdList(lines, start, depth) {
  const first = RE_ITEM.exec(lines[start]);
  const ordered = /\d/.test(first[2].charAt(0));
  const baseIndent = mdIndentWidth(first[1]);
  const markerWidth = first[1].length + first[2].length + 1;
  const node = mk(ordered ? 'ol' : 'ul');
  if (ordered) {
    const n = parseInt(first[2], 10);
    if (isFinite(n) && n !== 1) node.setAttribute('start', String(n));
  }
  let i = start;
  let loose = false;
  const items = [];
  while (i < lines.length) {
    const m = RE_ITEM.exec(lines[i]);
    if (m) {
      const ind = mdIndentWidth(m[1]);
      const isOrdered = /\d/.test(m[2].charAt(0));
      if (ind < baseIndent || (ind === baseIndent && isOrdered !== ordered)) break;
      if (ind > baseIndent + 1) {
        // A deeper item belongs to the previous item's content, not to this list.
        if (!items.length) break;
        items[items.length - 1].body.push(lines[i]);
        i++;
        continue;
      }
      items.push({ head: m[3], body: [], width: m[1].length + m[2].length + 1 });
      i++;
      continue;
    }
    if (!lines[i].trim()) {
      // A blank line only ends the list if the next non-blank line is not list content.
      let j = i + 1;
      while (j < lines.length && !lines[j].trim()) j++;
      if (j >= lines.length) { i = j; break; }
      const nm = RE_ITEM.exec(lines[j]);
      const nInd = mdIndentWidth(lines[j]);
      if ((nm && mdIndentWidth(nm[1]) >= baseIndent) || nInd > baseIndent) {
        loose = true;
        if (items.length) items[items.length - 1].body.push('');
        i++;
        continue;
      }
      i = j;
      break;
    }
    const ind2 = mdIndentWidth(lines[i]);
    if (ind2 > baseIndent && items.length) {
      items[items.length - 1].body.push(lines[i]);
      i++;
      continue;
    }
    if (items.length && !mdStartsBlock(lines[i]) && items[items.length - 1].body.length === 0) {
      // Lazy continuation of the item's first line.
      items[items.length - 1].head += ' ' + lines[i].trim();
      i++;
      continue;
    }
    break;
  }
  for (let k = 0; k < items.length; k++) {
    const it = items[k];
    const li = mk('li');
    const dedent = it.width || markerWidth;
    const body = it.body.map(function (s) {
      let w = 0;
      let p = 0;
      while (p < s.length && w < dedent) {
        if (s.charAt(p) === ' ') { w += 1; p++; }
        else if (s.charAt(p) === '\t') { w += 4 - (w % 4); p++; }
        else break;
      }
      return s.slice(p);
    });
    const content = [it.head].concat(body).join('\n');
    const hasBlocks = /\n[ \t]*\n/.test(content) || RE_FENCE.test(content) ||
                      content.split('\n').some(function (s, idx) { return idx > 0 && mdStartsBlock(s); });
    if (loose || hasBlocks) {
      li.appendChild(mdToFragment(content, depth + 1));
    } else {
      appendAll(li, mdInline(content.replace(/\n/g, ' '), depth + 1));
    }
    node.appendChild(li);
  }
  if (!items.length) {
    const li = mk('li');
    appendAll(li, mdInline(first[3], depth + 1));
    node.appendChild(li);
    i = start + 1;
  }
  return { node: node, end: i };
}

// The public entry point. Returns a fragment; the caller replaces the message body with it.
function renderMarkdown(src) {
  return mdToFragment(src, 0);
}