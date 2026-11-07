#!/bin/sh
# Batch SERVE-ATTRIB: where does the serve-mode rate difference come from?
#
# Observation (runs/webui_api_v2.log): serve at its DEFAULT cap 2048 measures 13.74-13.82 tok/s
# while the CLI ignition at cap 256 measures 14.0-14.3 in the same era. Two suspects, one at a time:
#   leg A  CLI  cap 256   - the control (the regression_v14 shape)
#   leg B  CLI  cap 2048  - same command, only BONSAI_CAP differs => isolates the arena-size effect
#   leg C  serve cap 256  - BONSAI_CAP=256 explicit => isolates the HTTP-shell effect at equal arena
# If B ~= C < A, the difference is the arena, not the shell. If C < B, the shell costs something and
# the per-token emit path is the suspect. Environment (RAM + VRAM holders) is recorded per leg because
# a busy desktop moves these numbers by 1-2% on its own (HANDOVER §4.5).
#
# One GPU process at a time; clear_field before every leg; no log overwritten (the _vN rule).
#
#   $ sh batch_serve_attrib.sh v1
cd "$(dirname "$0")"
TAG=${1:-v1}
M="${BONSAI_MODEL:-models/Ternary-Bonsai-2-9B-PTQ1_0.gguf}"
OUT=../runs/webui_attrib_${TAG}.log
if [ -e "$OUT" ]; then echo "[!] $OUT exists - pick a new tag" >&2; exit 1; fi

clear_field() {
  powershell -NoProfile -ExecutionPolicy Bypass -File ../tools/clear_field.ps1 > "$1" 2>&1
}
env_snapshot() {
  powershell -NoProfile -Command "\$c=Get-Counter '\GPU Process Memory(*)\Dedicated Usage' -ErrorAction SilentlyContinue; 'VRAM_DEDICATED_TOTAL_MiB={0:N1}' -f ((\$c.CounterSamples|Measure-Object CookedValue -Sum).Sum/1MB); \$c.CounterSamples|Where-Object{\$_.CookedValue -gt 104857600}|Sort-Object CookedValue -Descending|Select-Object -First 6|ForEach-Object{ \$p=\$_.InstanceName -replace '^pid_(\d+).*','\$1'; \$n=(Get-Process -Id \$p -ErrorAction SilentlyContinue).ProcessName; '  VRAM {0,-22} pid={1,-7} {2,8:N1} MiB' -f \$n,\$p,(\$_.CookedValue/1MB)}" 2>&1
}

{ echo "=== serve-rate attribution, $(date) ==="; echo "binary: $(wc -c < bonsai_vk.exe) B"; } > "$OUT" 2>&1

# --- leg A: CLI cap 256 (control)
F=../runs/webui_attrib_${TAG}_field_A.txt
clear_field "$F"
{ echo "=== leg A: CLI cap 256 (control), $(date) ==="
  grep -E "AVAILABLE_MB" "$F"; env_snapshot
  BONSAI_QCHECK=1 BONSAI_CAP=256 ./bonsai_vk.exe run "$M" 32 48 2>&1
  echo "rc_A=$?"; } >> "$OUT" 2>&1

# --- leg B: CLI cap 2048 (arena effect)
F=../runs/webui_attrib_${TAG}_field_B.txt
clear_field "$F"
{ echo "=== leg B: CLI cap 2048 (arena effect), $(date) ==="
  grep -E "AVAILABLE_MB" "$F"; env_snapshot
  BONSAI_QCHECK=1 BONSAI_CAP=2048 ./bonsai_vk.exe run "$M" 32 48 2>&1
  echo "rc_B=$?"; } >> "$OUT" 2>&1

# --- leg C: serve cap 256 (shell effect at equal arena)
F=../runs/webui_attrib_${TAG}_field_C.txt
clear_field "$F"
SRV=../runs/webui_attrib_${TAG}_server_C.log
{ echo "=== leg C: serve cap 256 (shell effect), $(date) ==="
  grep -E "AVAILABLE_MB" "$F"; env_snapshot; } >> "$OUT" 2>&1
BONSAI_NO_BROWSER=1 BONSAI_QCHECK=1 BONSAI_CAP=256 BONSAI_THINKING=0 ./bonsai_vk.exe serve "$M" > "$SRV" 2>&1 &
PID=$!
i=0
while [ $i -lt 90 ]; do
  if grep -q "listening   http://" "$SRV" 2>/dev/null; then break; fi
  if ! kill -0 "$PID" 2>/dev/null; then echo "[!] server died during bring-up" >> "$OUT"; break; fi
  sleep 1
  i=$((i+1))
done
node ../webui/scripts/api-probe.mjs --port 8080 --max-tokens 48 >> "$OUT" 2>&1
echo "rc_C=$?" >> "$OUT"
kill "$PID" 2>/dev/null
wait "$PID" 2>/dev/null

tr -d '\r' < "$OUT" | grep -E "^=== leg|AVAILABLE_MB|VRAM_DEDICATED_TOTAL|generated tokens|steady-state|gate:|MEASURED|SUMMARY|api-probe:|^rc_"
echo "--- full log: $OUT"