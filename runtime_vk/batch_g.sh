#!/bin/sh
# Batch G: the GQA 4x-redundancy merge ladder, measured against the best non-merged lineage kernel
# (tiled q2048) in the SAME binary, plus the attribution split (QK half vs PV half) and the
# short-context regression. One GPU process at a time, clear_field between every run, no log is
# overwritten (the _v1/_v2 suffix rule from batch_b is kept).
#
#   $1 = tag suffix for the log names (default v1)
cd "$(dirname "$0")"
TAG=${1:-v1}
M=models/Ternary-Bonsai-2-9B-PTQ1_0.gguf
SUM=../runs/gqa_ladder_${TAG}_summary.txt
: > "$SUM"

run() {   # name  env...
  name=$1; shift
  out=../runs/${name}_${TAG}.log
  if [ -e "$out" ]; then out=../runs/${name}_${TAG}b.log; fi
  powershell -NoProfile -ExecutionPolicy Bypass -File ../tools/clear_field.ps1 >/dev/null 2>&1
  env "$@" ./bonsai_vk.exe run "$M" 32 48 > "$out" 2>&1
  rc=$?
  st=$(grep -oE "steady-state: [0-9.]+ ms/token" "$out" | tail -1)
  tk=$(grep -oE "\-\->?  ?[0-9.]+ tok/s|-> [0-9.]+ tok/s" "$out" | tail -1)
  printf "%-34s rc=%s  %-34s %s\n" "$name" "$rc" "$st" "$tk" | tee -a "$SUM"
}

# ---- the ladder: merged (auto, HPK=4, grid 16 -> 4) at QTILE 512, the default
for c in 4096 8192 16384 32768; do
  run gqa_on_c${c}_q512 BONSAI_CAP=$c BONSAI_KVFILL=$c BONSAI_GQA_OPT=1 BONSAI_GQA_TILE=512
done

# ---- the controls: GQA off, tiled q2048 = the best non-merged kernel from §9.4(b), same binary
for c in 4096 8192 16384 32768; do
  run gqa_off_c${c}_tile2048 BONSAI_CAP=$c BONSAI_KVFILL=$c BONSAI_GQA_OPT=0 BONSAI_QKERNEL=3 BONSAI_QK_TILE=2048
done

# ---- the merged tile width: does a bigger LDS-per-workgroup (16.4 KiB) pay at the widest row?
for c in 4096 32768; do
  run gqa_on_c${c}_q1024 BONSAI_CAP=$c BONSAI_KVFILL=$c BONSAI_GQA_OPT=1 BONSAI_GQA_TILE=1024
done

# ---- attribution: which half of the surgery carries the gain (bit0 = QK gather, bit1 = V block read)
run gqa_part_qk_only_c4096  BONSAI_CAP=4096 BONSAI_KVFILL=4096 BONSAI_GQA_OPT=1 BONSAI_GQA_PART=1 BONSAI_GQA_TILE=512
run gqa_part_pv_only_c4096  BONSAI_CAP=4096 BONSAI_KVFILL=4096 BONSAI_GQA_OPT=1 BONSAI_GQA_PART=2 BONSAI_GQA_TILE=512

# ---- short-context regression: cap 256, the default prompt, merged vs control
run short_merged_gqa_on  BONSAI_CAP=256 BONSAI_GQA_OPT=1 BONSAI_GQA_TILE=256
run short_merged_gqa_off BONSAI_CAP=256 BONSAI_GQA_OPT=0
echo "done" >> "$SUM"
