#!/bin/sh
# Batch B: the tiled-Kernel long-context ladder, measured with the synthetic KV fixture.
# One GPU process at a time, clear_field between every run, no log is overwritten.
cd "$(dirname "$0")"
M=models/Ternary-Bonsai-2-9B-PTQ1_0.gguf
for cap in 2048 4096 8192 16384; do
  for tile in 1024 2048 4096; do
    # A tile wider than the row is the single-tile case with a bigger LDS allocation. That is not a
    # bug and it is worth exactly one measurement (cap 2048 / QTILE 4096): it prices the LDS footprint
    # alone, with the instruction stream held fixed, which is the occupancy half of the question.
    if [ "$tile" -gt "$cap" ] && [ "$cap" -ne 2048 ]; then continue; fi
    name=kvfx_c${cap}_q${tile}
    out=../runs/${name}_v1.log
    if [ -e "$out" ]; then out=../runs/${name}_v2.log; fi
    powershell -NoProfile -ExecutionPolicy Bypass -File ../tools/clear_field.ps1 >/dev/null 2>&1
    BONSAI_CAP=$cap BONSAI_KVFILL=$cap BONSAI_QKERNEL=3 BONSAI_QK_TILE=$tile \
      ./bonsai_vk.exe run "$M" 32 48 > "$out" 2>&1
    rc=$?
    st=$(grep -oE "steady-state: [0-9.]+ ms/token" "$out" | tail -1)
    printf "%-28s rc=%s  %s\n" "$name" "$rc" "$st"
  done
done

# ---- the two wide references, same fixture, so the ladder has a kernel-mix A/B at the widest row
# the wide kernel can address (cap 2048) and a second cross-validation point against the real
# 1792-token ladder run (runs/ladder_cap2048_q8_0_v1.log: 96.9 ms/token).
for spec in "2048 2048 wide_c2048" "2048 1816 wide_x1816"; do
  set -- $spec
  cap=$1; nfill=$2; name=$3
  out=../runs/kvfx_${name}_v1.log
  if [ -e "$out" ]; then out=../runs/kvfx_${name}_v2.log; fi
  powershell -NoProfile -ExecutionPolicy Bypass -File ../tools/clear_field.ps1 >/dev/null 2>&1
  BONSAI_CAP=$cap BONSAI_KVFILL=$nfill BONSAI_QKERNEL=2 \
    ./bonsai_vk.exe run "$M" 32 48 > "$out" 2>&1
  rc=$?
  st=$(grep -oE "steady-state: [0-9.]+ ms/token" "$out" | tail -1)
  printf "%-28s rc=%s  %s\n" "kvfx_$name" "$rc" "$st"
done
