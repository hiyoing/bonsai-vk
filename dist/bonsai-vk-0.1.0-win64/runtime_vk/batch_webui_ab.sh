#!/bin/sh
# Batch WEBUI-AB: does the HTTP/WebUI shell cost decode throughput?
#
# The frozen baseline is runs/regression_v14.log:95 (14.16 tok/s, steady 70.6 ms) but that was measured
# on a different day with a different set of background apps holding VRAM. Comparing today's number to
# last week's number cannot separate "the shell is slower" from "the machine is busier", so this script
# alternates the two binaries IN THE SAME SESSION, same command, same env, clear_field between every
# run, and reports the pairs. NEW = runtime_vk/bonsai_vk.exe (with bonsai_http.h + bonsai_platform.h).
# OLD = the shipped pre-shell dist binary, 552178 B, which has neither.
#
# Both are started from a directory that contains shaders/, because make_pipeline reads CWD-relative
# shaders/*.spv (HANDOVER §15.8). The dist package root has its own copy; the SPIR-V is unchanged by
# this work, so the two runs execute the same kernels.
#
#   $ sh batch_webui_ab.sh v1 [pairs]      default 3 pairs = 6 runs
cd "$(dirname "$0")"
TAG=${1:-v1}
PAIRS=${2:-3}
M="${BONSAI_MODEL:-models/Ternary-Bonsai-2-9B-PTQ1_0.gguf}"
NEWDIR=$(pwd)
OLDDIR=$(cd ../dist/bonsai-vk-0.1.0-win64 && pwd)
OUT=../runs/webui_ab_${TAG}.log
SUM=../runs/webui_ab_${TAG}_summary.txt
if [ -e "$OUT" ]; then echo "[!] $OUT exists - pick a new tag" >&2; exit 1; fi

{
  echo "=== WebUI shell A/B, $(date) ==="
  echo "NEW $NEWDIR/bonsai_vk.exe  $(wc -c < "$NEWDIR/bonsai_vk.exe") B"
  echo "OLD $OLDDIR/bonsai_vk.exe  $(wc -c < "$OLDDIR/bonsai_vk.exe") B"
  echo "command: BONSAI_QCHECK=1 bonsai_vk.exe run <model> 32 48   (the regression_v14 parameters)"
  echo "clear_field is REPORT-ONLY (no -Kill), matching HANDOVER §4.5 and every existing batch script"
} > "$OUT" 2>&1
: > "$SUM"

i=1
while [ "$i" -le "$PAIRS" ]; do
  for side in NEW OLD; do
    if [ "$side" = NEW ]; then d=$NEWDIR; else d=$OLDDIR; fi
    log=../runs/webui_ab_${TAG}_${side}${i}.log
    powershell -NoProfile -ExecutionPolicy Bypass -File ../tools/clear_field.ps1 > ../runs/webui_ab_${TAG}_field_${side}${i}.txt 2>&1
    avail=$(grep -oE "AVAILABLE_MB_BEFORE=[0-9]+" ../runs/webui_ab_${TAG}_field_${side}${i}.txt | cut -d= -f2)
    ( cd "$d" && BONSAI_QCHECK=1 ./bonsai_vk.exe run "$M" 32 48 ) > "$log" 2>&1
    rc=$?
    tr -d '\r' < "$log" > "$log.tmp" && mv "$log.tmp" "$log"
    rate=$(grep -oE "[0-9]+ generated tokens in [0-9.]+ ms  ->  [0-9.]+ tok/s" "$log" | tail -1)
    steady=$(grep -oE "steady-state: [0-9.]+ ms/token" "$log" | tail -1)
    gate=$(grep -oE "gate: (PASS|FAIL)[^)]*\)" "$log" | tail -1)
    ded=$(grep -oE "dedicated +[0-9.]+ MiB" "$log" | tail -1)
    printf "%s%-2s rc=%s availRAM=%sMB | %s | %s | %s | %s\n" \
      "$side" "$i" "$rc" "$avail" "$rate" "$steady" "$gate" "$ded" | tee -a "$SUM"
    { echo "--- $side$i rc=$rc availRAM=${avail}MB log=$log"; cat "$log"; } >> "$OUT" 2>&1
  done
  i=$((i + 1))
done
echo "=== full logs concatenated in $OUT ===" >> "$SUM"
cat "$SUM"