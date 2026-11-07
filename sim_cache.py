"""Streaming expert-cache simulation driven by a synthetic routing trace.

Purpose: turn 'hit rate' from an assumption into a function of routing skew and
cache policy, then feed it into the measured cost model.  The skew parameter is
NOT measured on the real model; this is a sensitivity tool, not a prediction.
"""
import numpy as np

LAYERS, EXPERTS, TOPK = 48, 128, 8
EXPERT_MB = 1.18            # params 4.72M * 2bit
TOTAL_EXPERTS = LAYERS * EXPERTS
BW_GB_S = 6.5               # measured PCIe 3.0 x8
A_MS = 0.190                # attention per layer, 4-bit resident
C_FUSED = 0.424             # measured probe_gemv8
C_SEQ = 0.912               # measured probe_gemv8, one dispatch per expert
HEAD_MS = 2.3               # lm_head 4-bit
FIXED_MS = 3.0              # dispatch encoding, router, sampling


def make_trace(tokens, s, seed=0):
    rng = np.random.default_rng(seed)
    ranks = np.arange(1, EXPERTS + 1)
    w = 1.0 / ranks ** s
    w /= w.sum()
    probs = np.stack([w[rng.permutation(EXPERTS)] for _ in range(LAYERS)])
    trace = []
    for _ in range(tokens):
        tok = []
        for l in range(LAYERS):
            es = rng.choice(EXPERTS, TOPK, replace=False, p=probs[l])
            tok.extend((l, int(e)) for e in es)
        trace.append(tok)
    return trace


def hit_rate(trace, cap, policy):
    cache, hits, total = {}, 0, 0
    static = None
    if policy == "static":
        warm = {}
        for tok in trace[:30]:
            for k in tok:
                warm[k] = warm.get(k, 0) + 1
        static = set(sorted(warm, key=warm.get, reverse=True)[:cap])
    for t, tok in enumerate(trace):
        for key in tok:
            total += 1
            if policy == "static":
                hits += key in static
                continue
            if key in cache:
                hits += 1
                cache[key] = (t, cache[key][1] + 1)
            else:
                if len(cache) >= cap:
                    if policy == "lru":
                        victim = min(cache, key=lambda k: cache[k][0])
                    else:
                        victim = min(cache, key=lambda k: (cache[k][1], cache[k][0]))
                    del cache[victim]
                cache[key] = (t, 1)
    return hits / total


def token_ms(h):
    miss_mb = (1.0 - h) * TOPK * EXPERT_MB
    tran = miss_mb / (BW_GB_S * 1000.0) * 1000.0
    fused = LAYERS * (A_MS + tran + C_FUSED) + HEAD_MS + FIXED_MS
    pipe = LAYERS * (A_MS + max(tran, C_SEQ)) + HEAD_MS + FIXED_MS
    return tran, fused, pipe


def main():
    print(f"{LAYERS}L x {EXPERTS}E top{TOPK} | expert {EXPERT_MB}MB | "
          f"PCIe {BW_GB_S} GB/s | fused {C_FUSED}ms seq {C_SEQ}ms per layer\n")
    print(f"{'skew s':>7s} {'policy':>7s} {'cap':>5s} {'bytes':>7s} {'hit':>6s} "
          f"{'miss ms':>8s} {'fused t/s':>10s} {'pipe t/s':>9s}")
    for s in (0.0, 0.4, 0.8, 1.2):
        trace = make_trace(300, s)
        for f in (0.20, 0.33, 0.40):
            cap = int(f * TOTAL_EXPERTS)
            for pol in ("lru", "lfu"):
                h = hit_rate(trace, cap, pol)
                tran, fu, pi = token_ms(h)
                print(f"{s:7.1f} {pol:>7s} {f:5.2f} {cap*EXPERT_MB/1024:6.2f}G {h:6.3f} "
                      f"{tran:8.2f} {1000/fu:10.1f} {1000/pi:9.1f}")
        # static is policy-independent of the run, do it once per skew
        f = 0.33
        cap = int(f * TOTAL_EXPERTS)
        h = hit_rate(trace, cap, "static")
        tran, fu, pi = token_ms(h)
        print(f"{s:7.1f} {'static':>7s} {f:5.2f} {cap*EXPERT_MB/1024:6.2f}G {h:6.3f} "
              f"{tran:8.2f} {1000/fu:10.1f} {1000/pi:9.1f}")
        print()

    print("upper bound, everything resident (no PCIe):",
          f"{1000/(LAYERS*(A_MS+C_FUSED)+HEAD_MS+FIXED_MS):.1f} tok/s fused, "
          f"{1000/(LAYERS*(A_MS+C_SEQ)+HEAD_MS+FIXED_MS):.1f} tok/s seq")
    print("ideal overlap floor (pipe model, transfer hidden):",
          f"{1000/(LAYERS*(A_MS+C_SEQ)+HEAD_MS+FIXED_MS):.1f} tok/s")


if __name__ == "__main__":
    main()
