import io, re, sys

PIPE = re.compile(r"(?<!\\)\|")
files = ['README.md', 'README_zh.md', 'BENCHMARK.md', 'ROADMAP.md', 'ANNOUNCEMENTS.md',
         'HANDOVER.md', 'KV_P0P1P2_MEASURED.md', 'FAITHFUL_REPRO.md', 'ATTENTION_ALIGN_FINAL.md',
         'END_TO_END_MEASURED.md', 'DEEPINFER_EVAL.md', 'STAGE3_RUNTIME.md', 'STAGE0_REPORT.md',
         'STAGE1_KERNEL.md', 'STAGE2_CUSTOM.md', 'PROJECT_BONSAI_FINAL_REPORT.md',
         'PROJECT_BONSAI_FULL_DOSSIER.md', 'PROJECT_BONSAI_WEBUI_REPORT.md']
bad_total = 0
for f in files:
    try:
        lines = io.open(f, encoding='utf-8').read().splitlines()
    except IOError:
        continue
    hdr = None
    bad = []
    incode = False
    for i, l in enumerate(lines, 1):
        if l.strip().startswith('```'):
            incode = not incode
            hdr = None
            continue
        if incode:
            hdr = None
            continue
        if l.strip().startswith('|'):
            n = len(PIPE.findall(l))
            if hdr is None:
                hdr = n
            elif n != hdr:
                bad.append((i, n, hdr, l[:70]))
        else:
            hdr = None
    bad_total += len(bad)
    print('%-28s mismatch rows: %d' % (f, len(bad)))
    for b in bad[:8]:
        print('    %d: %d vs %d :: %s' % b)
print('TOTAL', bad_total)
