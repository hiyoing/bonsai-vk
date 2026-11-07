#!/usr/bin/env python3
"""probe_hadamard.py - measure the activation-side Hadamard convention from the oracle dump.

PART 1 (this file; log only, no model, no GPU)
  The dump prints, for every node, a full-precision sum over the whole tensor plus the first 3
  and last 3 elements of every column.  For a sandwich  y = H x  with H a normalised Sylvester
  matrix in SOME row/column order, exactly one column of H is all +1/sqrt(N), every other column
  sums to zero, and the same holds for rows.  Two linear functionals are therefore observable at
  every single MUL_MAT(prism.hadamard.4096, x) node:

      (C)  sum_k (H x)_k       =  sqrt(N) * sum_c x[0][c]      column 0 of H is the all-ones one
      (R)  sum_c (H x)[0][c]   =  sum(x) / sqrt(N)             row 0 of H is the all-ones one

  Both are also read at index N-1 instead of 0 as a negative control.  sqrt(4096) = 64, so these
  two tests also confirm the 1/sqrt(N) normalisation of the matrix.

PART 2 writes runs/probe_targets.txt: the printed GET_ROWS(token_embd.weight) / MUL_MAT(hadamard)
  fingerprints of both passes.  runtime_vk/probe_hadamard.cpp uses them to (a) find the token ids
  by fingerprint and (b) invert WHICH natural Sylvester row sits at output positions 1, 2 and
  4093/4094/4095 - that is what distinguishes natural, bit-reversed and sequency order.
"""

import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(os.path.dirname(HERE), "runs")
LOG = os.path.join(RUNS, "dbg_full_real.log")
OUT = os.path.join(RUNS, "probe_targets.txt")
SCALE = 64.0
DECODE_FIRST = 32046
N = 4096

HDR = re.compile(r"common_debug_cb_eval:\s*(.*?)=\s*\(f32\)\s+([A-Z_0-9]+)\((.*?)\)\s*=\s*\{([0-9, ]+)\}")
FLOAT = re.compile(r"[-+]?\d+\.\d+(?:[eE][-+]?\d+)?")
