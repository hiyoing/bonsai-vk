#!/bin/sh
# Batch H: the GQA merge A/B measured PAIRWISE. Batch G showed a monotone upward drift in the control
# column across the batch (+0.5 % at 8 K, +2.7 % at 32 K, then a 33 % swing on the short-context pair),
# which is what sustained load on a passively cooled RX 550X looks like: clocks sag as the card heats, and
# a ladder that measures all the controls first and all the candidates after compares the kernel to the
# thermal state instead of to the other kernel. So every configuration is measured next to its own
# control, the short-context pair runs FIRST while the card is cold, and the short control is measured
# AGAIN at the end - that last number is the drift estimate for the whole batch, in one line.
#
# Configurations:
#   off   BONSAI_GQA_OPT=0 + QKERNEL=3 QK_TILE=2048   = the best non-merged lineage kernel (tiled),
#           and the frozen attn_pv_q8v. This is the §9.4(b) incumbent.
#   on1   GQA_OPT=1 GQA_PART=1 (bit0)                  = merged QK only, frozen PV.
#   on3   GQA_OPT=1 GQA_PART=3                         = merged QK and merged PV.
# One GPU process at a time, clear_field between runs, no log is overwritten.
cd "$(dirname "$0")"
TAG=${1:-v1}
M=models/Ternary-Bonsai-2-9B-PTQ1_0.gguf
SUM=../runs/gqa_pair_${TAG}_summary.txt
: > "$SUM"

run() {
  name=$1; shift
  out=../runs/${name}_${TAG}.log
  if [ -e "$out" ]; then out=../runs/${name}_${TAG}b.log; fi
  powershell -NoProfile -ExecutionPolicy Bypass -File ../tools/clear_field.ps1 >/dev/null 2>&1
  env "$@" ./bonsai_vk.exe run "$M" 32 48 > "$out" 2>&1
  rc=$?
  st=$(grep -oE "steady-state: [0-9.]+ ms/token" "$out" | tail -1)
  tk=$(grep -oE "\-> +[0-9.]+ tok/s" "$out" | tail -1)
  printf "%-30s rc=%s  %-32s %s\n" "$name" "$rc" "$st" "$tk" | tee -a "$SUM"
}

# ---- the short-context pair, cold, before the card has any reason to sag
run short_c256_off  BONSAI_CAP=256 BONSAI_GQA_OPT=0
run short_c256_on1  BONSAI_CAP=256 BONSAI_GQA_OPT=1 BONSAI_GQA_PART=1 BONSAI_GQA_TILE=256
run short_c256_on3  BONSAI_CAP=256 BONSAI_GQA_OPT=1 BONSAI_GQA_PART=3 BONSAI_GQA_TILE=256

# ---- the long-context triples, each cap measured inside one thermal window
for c in 4096 8192 16384 32768; do
  sleep 20
  run c${c}_off  BONSAI_CAP=$c BONSAI_KVFILL=$c BONSAI_GQA_OPT=0 BONSAI_QKERNEL=3 BONSAI_QK_TILE=2048
  run c${c}_on1  BONSAI_CAP=$c BONSAI_KVFILL=$c BONSAI_GQA_OPT=1 BONSAI_GQA_PART=1 BONSAI_GQA_TILE=512
  run c${c}_on3  BONSAI_CAP=$c BONSAI_KVFILL=$c BONSAI_GQA_OPT=1 BONSAI_GQA_PART=3 BONSAI_GQA_TILE=512
done

# ---- the drift probe: the same config as run 1, at the end of the batch, on a now-warm card
sleep 20
run short_c256_off_repeat BONSAI_CAP=256 BONSAI_GQA_OPT=0
echo "done" >> "$SUM"
