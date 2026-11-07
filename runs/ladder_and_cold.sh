#!/bin/bash
# runs/ladder_and_cold.sh - two things the report still needs, both measured, not derived:
#  (1) the real 4 GB wall: at which context size does -ngl 99 stop fitting on the RX 550X
#  (2) a genuinely cold mechanical-disk load: copy the model, evict the standby cache with a
#      12 GiB junk write, then load, with the WDDM VRAM + PhysicalDisk counters sampled at 1 Hz
#
# Paths are desensitised for the public repo. R/P are derived from this script's own location, so
# they carry no machine-specific prefix. The two external locations are env-overridable:
#   PRISM_BUILD  the official llama.cpp/PrismML fork build dir that has llama-cli.exe (default below)
#   MODEL_DIR    the directory holding Ternary-Bonsai-2-9B-PTQ1_0.gguf            (default below)
# Set them to your own checkout before running; nothing here hardcodes a personal absolute path.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../runs (absolute, derived not hardcoded)
REPO="$(dirname "$HERE")"                              # repo root
R="$HERE"
P="$REPO/tools/prompt.txt"
PRISM="${PRISM_BUILD:-$REPO/../prism-vk/prism}"
MDIR="${MODEL_DIR:-$REPO/models}"
M="$MDIR/Ternary-Bonsai-2-9B-PTQ1_0.gguf"
cd "$PRISM" || exit 9
echo "### LADDER $(date -Is)"
for c in 8192 16384 32768 49152 65536; do
  echo "--- ctx $c"
  ./llama-cli.exe -m "$M" -dev Vulkan0 -ngl 99 -c "$c" -n 1 -s 42 -f "$P" -st --no-display-prompt --simple-io -lv 3 \
      > "$R/ladder_c$c.log" 2>&1
  echo "    rc=$?"
  grep -hoE "KV buffer size = +[0-9.]+ MiB|size = +[0-9.]+ MiB \( +[0-9]+ cells|Vulkan0 model buffer size = +[0-9.]+ MiB|Vulkan0 compute buffer size = +[0-9.]+ MiB|RS buffer size = +[0-9.]+ MiB|free device memory|projected to use [0-9]+ MiB|will leave [0-9]+|[A-Za-z ]*(out of memory|failed to allocate|VK_ERROR_OUT_OF_DEVICE_MEMORY)[A-Za-z0-9 :]*" "$R/ladder_c$c.log" | sed 's/^/    /' | head -12
  grep -hE "memory breakdown" -A2 "$R/ladder_c$c.log" | grep "Vulkan0 (RX" | sed 's/^/    /'
done
echo "### COLD LOAD $(date -Is)"
rm -f "$MDIR/_cold_probe.gguf" "$REPO/_flush.bin"
t0=$(date +%s)
cp "$MDIR/Ternary-Bonsai-2-9B-PTQ1_0.gguf" "$MDIR/_cold_probe.gguf"
echo "copy rc=$? seconds=$(( $(date +%s) - t0 ))"
ls -l "$MDIR/_cold_probe.gguf"
dd if=/dev/zero of="$REPO/_flush.bin" bs=4M count=3072 status=none
echo "flush rc=$? (12 GiB written to evict the standby cache)"
rm -f "$REPO/_flush.bin"
sync
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "tools\vram_sample.ps1" \
    -Name llama -Out "runs\vram_cold_real.tsv" -Seconds 240 > "$R/sampler3.out" 2>&1 &
SP=$!
sleep 2
{ time ./llama-cli.exe -m "$MDIR/_cold_probe.gguf" -dev Vulkan0 -ngl 99 -c 4096 -n 1 -s 42 -f "$P" \
      -st --no-display-prompt --simple-io -lv 3 ; } > "$R/run_cold_real.log" 2>&1
echo "cold load rc=$?"
kill $SP 2>/dev/null
rm -f "$MDIR/_cold_probe.gguf"
echo "### DONE $(date -Is)"