#!/bin/sh
# Serve-mode connectivity + contract regression. The companion to batch_webui.sh (which proves the
# CLI path did not regress); this one proves the HTTP shell actually answers, and measures the
# serve-mode tok/s at the serve default cap (2048), which is NOT the CLI's 256 and must not be
# reported as the 14.16 CLI baseline.
#
# Two legs, because the engine default is thinking=on: at a 48-token budget every delta is then
# reasoning_content and visible content is legitimately empty (v1's single FAIL was exactly this
# misassertion, not a server bug). Leg 1 proves the reasoning channel; leg 2 sets BONSAI_THINKING=0
# and proves real content deltas. The probe reads /v1/status.thinking and asserts the ACTIVE channel.
#
# One GPU process at a time: clear_field before EACH leg (both measure tok/s), server in background,
# probe over loopback, then kill + wait before the next leg. No log is overwritten (the _vN rule).
#
#   $ sh batch_webui_serve.sh v2
cd "$(dirname "$0")"
TAG=${1:-v1}
M="${BONSAI_MODEL:-models/Ternary-Bonsai-2-9B-PTQ1_0.gguf}"
OUT=../runs/webui_api_${TAG}.log
if [ -e "$OUT" ]; then echo "[!] $OUT exists - pick a new tag" >&2; exit 1; fi

clear_field() {
  powershell -NoProfile -ExecutionPolicy Bypass -File ../tools/clear_field.ps1 >/dev/null 2>&1
}

# $1 = leg label, $2 = BONSAI_THINKING value, $3 = server log path
run_leg() {
  LEG=$1; THINK=$2; SRV=$3
  clear_field
  # BONSAI_NO_BROWSER=1: an automated probe must not pop a browser window (focus + VRAM). A real
  # double-click leaves it unset and gets the auto-open behaviour.
  BONSAI_NO_BROWSER=1 BONSAI_QCHECK=1 BONSAI_THINKING="$THINK" ./bonsai_vk.exe serve "$M" > "$SRV" 2>&1 &
  PID=$!

  # Wait for the listen banner; cold weight load is ~14 s on this card, allow generous headroom.
  i=0
  while [ $i -lt 90 ]; do
    if grep -q "listening   http://" "$SRV" 2>/dev/null; then break; fi
    if ! kill -0 "$PID" 2>/dev/null; then echo "[!] server died during bring-up ($LEG)" >&2; break; fi
    sleep 1
    i=$((i+1))
  done

  echo "=== leg: $LEG (BONSAI_THINKING=$THINK), $(date) ===" >> "$OUT"
  echo "--- bring-up (server log head)" >> "$OUT"
  tr -d '\r' < "$SRV" | grep -E "listening|model |device |engine |WebUI |context |BONSAI_CAP|gate:|budget|arena" | head -40 >> "$OUT"
  echo "--- probe" >> "$OUT"
  node ../webui/scripts/api-probe.mjs --port 8080 --max-tokens 48 >> "$OUT" 2>&1
  echo "rc_probe=$?" >> "$OUT"

  kill "$PID" 2>/dev/null
  wait "$PID" 2>/dev/null
}

{ echo "=== serve-mode connectivity probe, $(date) ==="; } > "$OUT" 2>&1
run_leg thinking-on  1 ../runs/webui_api_${TAG}_server_on.log
run_leg thinking-off 0 ../runs/webui_api_${TAG}_server_off.log

tr -d '\r' < "$OUT" | grep -E "^=== leg|^PASS|^FAIL|^INFO|^MEASURED|^SUMMARY|api-probe:|rc_probe"
echo "--- full log: $OUT"