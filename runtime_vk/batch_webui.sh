#!/bin/sh
# Batch WEBUI: the regression that has to pass after the HTTP shell + the cross-platform port.
# It is the HANDOVER §4.5 sequence, narrowed to what this change could have broken:
#   * the five align_* tools (the port touched align_27b.cpp's seek and bonsai_weights.h's five seeks,
#     so the dequant/27B legs are the ones that would catch a wrapped offset);
#   * BONSAI_QCHECK=1 on the DEFAULT config, 48 tokens: the same command that produced
#     runs/regression_v14.log:94-96 (gate: PASS, 14.16 tok/s, 70.6 ms/token steady state).
# The serve path is NOT exercised here - it gets its own log, because a server does not exit.
#
# One GPU process at a time, clear_field before every leg, no log is overwritten (the _vN rule).
#
#   $ sh batch_webui.sh v1
cd "$(dirname "$0")"
TAG=${1:-v1}
M="${BONSAI_MODEL:-models/Ternary-Bonsai-2-9B-PTQ1_0.gguf}"
OUT=../runs/webui_regression_${TAG}.log
if [ -e "$OUT" ]; then echo "[!] $OUT exists - pick a new tag" >&2; exit 1; fi

clear_field() {
  powershell -NoProfile -ExecutionPolicy Bypass -File ../tools/clear_field.ps1 >/dev/null 2>&1
}

{
  echo "=== post-WebUI / post-cross-platform regression, $(date) ==="
  echo "--- binary: $(wc -c < bonsai_vk.exe) B"
  echo "--- align_tokenizer 32 probes"
  clear_field; ./align_tokenizer.exe "$M" ../runs/tok_probes.txt ../runs/tok_ref_32.jsonl 2>&1
  echo "rc=$?"
  echo "--- align_tokenizer kr set"
  ./align_tokenizer.exe "$M" ../runs/tok_probes_kr.txt ../runs/tok_ref_kr.jsonl 2>&1
  echo "rc=$?"
  echo "--- align_dequant: 1000 real blocks = 128000 weights (regression_v14's count, NOT the 100 default)"
  clear_field; ./align_dequant.exe "$M" 1000 2>&1
  echo "rc=$?"
  echo "--- align_f16"
  clear_field; ./align_f16.exe 2>&1
  echo "rc=$?"
  echo "--- align_attn 64 256 (the regression_v14 parameters, not the defaults)"
  clear_field; ./align_attn.exe 64 256 2>&1
  echo "rc=$?"
  echo "--- align_27b"
  clear_field; ./align_27b.exe 2>&1
  echo "rc=$?"
  echo "--- ignition + bitwise gate: BONSAI_QCHECK=1, DEFAULT config, cap 256, 48 tokens"
  clear_field
  # The tok/s number is only interpretable next to the machine state it was taken in. regression_v14
  # got 14.16 on an idle desktop; a browser/WeChat/remote-viewer holding 1-3 GiB of VRAM moves it by
  # ~1-2% and can make vkCreateComputePipelines fail with -2 outright. Record it, do not hide it.
  echo "--- environment at ignition time (report-only clear_field, no -Kill: same as HANDOVER §4.5)"
  powershell -NoProfile -ExecutionPolicy Bypass -File ../tools/clear_field.ps1 2>&1 | grep -E "AVAILABLE_MB|KILLED="
  powershell -NoProfile -Command "\$c=Get-Counter '\GPU Process Memory(*)\Dedicated Usage' -ErrorAction SilentlyContinue; 'VRAM_DEDICATED_TOTAL_MiB={0:N1}' -f ((\$c.CounterSamples|Measure-Object CookedValue -Sum).Sum/1MB); \$c.CounterSamples|Where-Object{\$_.CookedValue -gt 104857600}|Sort-Object CookedValue -Descending|Select-Object -First 6|ForEach-Object{ \$p=\$_.InstanceName -replace '^pid_(\d+).*','\$1'; \$n=(Get-Process -Id \$p -ErrorAction SilentlyContinue).ProcessName; '  VRAM {0,-22} pid={1,-7} {2,8:N1} MiB' -f \$n,\$p,(\$_.CookedValue/1MB)}" 2>&1
  BONSAI_QCHECK=1 ./bonsai_vk.exe run "$M" 32 48 2>&1
  echo "rc_qcheck=$?"
} > "$OUT" 2>&1

tr -d '\r' < "$OUT" | grep -E "byte-identical|bit-exact|wrong|bit for bit|score|ALL CHECKS|gate:|generated tokens|steady-state|^rc|max.*REF"
echo "--- full log: $OUT"