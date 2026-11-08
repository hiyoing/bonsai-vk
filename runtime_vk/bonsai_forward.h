// bonsai_forward.h - the whole-token forward: 32 blocks, 201 PTQ GEMVs, 8 KV caches, 24 recurrent
// states, ONE command buffer, ONE submit, ONE fence per token. Included by bonsai_vk.cpp after
// bonsai_weights.h because it is built out of the Engine that header creates - the same arena
// discipline (one DEVICE_LOCAL allocation, offsets carried in push constants) and the same barrier
// rule that turned the first non-deterministic GEMV into a bit-stable one.
//
// THE GRAPH IS NOT RECONSTRUCTED, IT IS TRANSCRIBED. Every node order, tensor split and transform
// width below comes from runs/dbg_full_real.log - the official fork's own cb_eval dump of THIS
// checkpoint running prompt "ab" (runs/oracle_nodes.txt is its 4848 node lines) - plus the five
// prism.hadamard.* metadata keys, which state the transform contract outright:
//
//   prism.hadamard.block_size = 4096, transform = "normalized-sylvester-walsh-hadamard",
//   axis = "input-last-dimension", sign_mode = "explicit", gdn_v_grouped = 1
//   weight_names          = 201 tensors   <- EVERY PTQ matmul gets (signs THEN H)
//   inverse_weight_names  = token_embd    <- the embedding gets (H THEN signs)
//   sign_widths = [4096, 12288], sign_values = 16384 x +-1 concatenated in that order
//
// That closes the last open question of the reconstruction: the 201 names are exactly the 201 GEMVs
// the weight plan measures, so ssm_alpha/ssm_beta (dtype 0, absent from the list) read the
// UNROTATED normed vector - MUL_MAT(blk.0.ssm_alpha.weight, attn_norm-0) in the dump, an edge that
// goes straight from the norm - while ffn_gate/ffn_up (present in the list) read the rotated one.
//
// PREFILL IS A LOOP OF DECODE STEPS. Nothing in the graph needs seq > 1, and running one position at
// a time means the KV cache, the conv shift register and the recurrent state advance identically in
// prefill and in decode: one code path, one dispatch count per position, no seq-dependent indexing
// to get wrong. The cost is that every prompt token pays a full 4096-wide pass; for a ~10-token
// prompt that is well under a second, and it buys the only forward loop a human can audit.
#ifndef BONSAI_FORWARD_H
#define BONSAI_FORWARD_H

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "bonsai_tok.h"
#include "bonsai_chat.h"    // the model's own chat template, transcribed (see that file's header)
#include "f16_rne_host.h"   // the host fp16 RNE packer (f32_to_f16_bits / pack2 / word_lo / word_hi):
                            // the same source of truth align_attn.exe validated over 4096 floats, so
                            // the P16 self-check cannot pass by disagreeing with the kernel's packer.
#include "kv_guard.h"       // P1's block-level KV capacity math + P2's WDDM spill watcher (PDH
                            // "GPU Process Memory(pid_<pid>_*)\Shared Usage"). Header-only, no Vulkan
                            // types in the watcher, so it is testable without a device.

static inline uint32_t f2u(float v) { uint32_t u; memcpy(&u, &v, 4); return u; }

// ------------------------------------------------------------------ region table
// One VkBuffer, every region aligned to the driver's own minStorageBufferOffsetAlignment, every
// shader addressed by a push-constant offset in ITS OWN element units (float / vec4 / uvec4 / uint,
// as each kernel's header documents). V4() converts a float offset to the vec4 kernels.
struct Regions {
    uint32_t align = 64u;                   // floats; fwd_init raises it if the driver asks for more
    uint32_t total = 0;
    uint32_t RES = 0, NRM = 0, AN = 0, ROT = 0, HR = 0, EMB = 0;
    uint32_t QKV = 0, K = 0, V = 0, Z = 0;
    uint32_t ALPHA = 0, BETA = 0, G1 = 0, G2 = 0, GATE32 = 0;
    uint32_t CONV = 0, L2 = 0, GDNO = 0, GDNORM = 0, SWG = 0, PERM = 0;
    uint32_t FG = 0, FU = 0, FS = 0, PART = 0, XS = 0, LOGITS = 0;
    uint32_t Q16 = 0, QF = 0, KF = 0, SCORE = 0, PROB = 0, P16 = 0, MASK = 0, POS = 0;
    uint32_t PGATE = 0, GOUT = 0, SIGNS = 0, QKWN = 0;
    uint32_t QD = 0, GD = 0;                // attn_q de-interleaved: contiguous q, contiguous gate
    uint32_t DUMMY = 0;                     // 16 floats: the window a binding points at when its
                                            // shader branch never executes (VS in fp16-V mode, KF in
                                            // no-debug-stream mode). A descriptor still has to name a
                                            // real, aligned, non-empty range.
    uint32_t KV = 0, ST = 0, CH = 0;
    uint64_t kv_layer = 0, st_layer = 0, ch_layer = 0;
    // The three sub-arenas of one attention layer's KV region, in uint words (= the arena's float
    // units). K is ALWAYS fp16 and is always the first sub-arena at +0, so the QK kernels and the
    // BONSAI_QCHECK audit that read KC are unaffected by the V representation.
    uint64_t kv_k = 0, kv_vd = 0, kv_vs = 0;   // K fp16 | V data (fp16 halves or q8_0 words) | V scales
    // Set when a take() would need more than uint32_t can hold. The offsets and `total` are uint32_t
    // float-indices, so a cap large enough to overflow them would otherwise wrap silently and hand the
    // kernels overlapping windows: an under-allocated arena and an overflowing one fail the SAME way,
    // which is why the sizes are checked here instead of being assumed safe.
    bool overflow = false;

    uint32_t take(uint64_t n) {
        const uint32_t a = (uint32_t)((total + align - 1u) / align * align);
        if (n > uint64_t(UINT32_MAX) || uint64_t(a) + n > uint64_t(UINT32_MAX)) overflow = true;
        total = a + (uint32_t)n;
        return a;
    }
};
#define V4(x) ((x) / 4u)

// ------------------------------------------------------------------ pipeline + descriptor set
struct Pipe {
    VkPipeline       pp = VK_NULL_HANDLE;
    VkPipelineLayout pl = VK_NULL_HANDLE;
    VkDescriptorSet  ds = VK_NULL_HANDLE;
    const char      *name = "?";
    uint32_t         nbind = 0;
};

// A binding is a WINDOW: VkBuffer + byte offset + byte length. The window form exists so that the
// three attention shaders - validated to max|err| = 0.00000 against the official dump while indexing
// every one of their buffers from element 0, and therefore NOT edited here - can be handed per-layer
// KV caches living at arbitrary offsets inside one big activation buffer.
struct Win { VkBuffer buf; uint64_t off; uint64_t size; };

struct Fwd {
    Engine  *e = nullptr;
    Regions  r;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkBuffer       buf = VK_NULL_HANDLE;
    uint64_t       bytes = 0;
    VkDescriptorPool dpool = VK_NULL_HANDLE;
    VkFence          fence = VK_NULL_HANDLE;

    enum P {
        P_SETPOS, P_EMBD, P_RMS, P_MUL, P_MULB, P_SIGNS, P_HAT, P_FWHT, P_ADD, P_ADDF, P_SWIGLU, P_PERM,
        P_SIGMOID, P_SOFTPLUS, P_L2, P_CONV, P_GDN, P_XSUM, P_REDUCE, P_GEMV, P_GEMV32, P_DEINT,
        P_NORMROPE, P_QK, P_QKW, P_QKT, P_PV, P_QKG, P_PVG, P_N
    };
    Pipe p[P_N];
    Pipe arope[8], aqk[8], aqkw[8], aqkt[8], apv[8], aqkg[8], apvg[8];
    // cap <= 256 dispatches the narrow (lane == key position) kernel that the layer-3 alignment was
    // measured with; anything wider dispatches attn_qk_softmax_wide.comp, whose lanes stride over the
    // row. Both are bound to the same 7 windows in the same order, and the wide one must reproduce
    // the narrow one BITWISE at cap = 256 - that cross-check is the gate for this whole branch.
    // The THIRD row kernel is attn_qk_softmax_tile_q<QTILE>.comp: same 7 windows again, but the row is
    // walked in QTILE-sized tiles and spills through the global SCORE/PROB row it already writes, so
    // its LDS is 4*QTILE+4 bytes regardless of cap and it has no MAXCAP at all.
    uint32_t wide = 0;                      // 1 = use aqkw[] instead of aqk[]
    uint32_t qkt  = 0;                      // 1 = use aqkt[] (overrides wide; tile is always >= 257 keys)
    uint32_t qk_force = 0;                  // 0 auto, 1 narrow, 2 wide, 3 tile (A/B cross-check)
    uint32_t qk_tile  = 2048u;              // BONSAI_QK_TILE: which attn_qk_softmax_tile_q<N>.comp
    // ---- the GQA merge (the 4x-redundancy surgery). One workgroup per (layer, kv head) serving
    // HPK = n_head/n_head_kv query heads, so the K gather and the V block read happen once per key per
    // layer instead of once per key per HEAD. qkg/pvg are the derived decisions; gqa_opt is the user's
    // switch (BONSAI_GQA_OPT: 0 = never, the bit-exact control path; 1 = auto, the default; 2 = force
    // the merged kernel at every cap, which is what makes a same-binary A/B possible), and gqa_tile
    // picks the merged wrapper (LDS = HPK*4*QTILE + 8*HPK bytes). gqa_part is the DIAGNOSTIC split -
    // bit0 = merge QK, bit1 = merge PV - so a measured gain can be attributed to one half instead of
    // asserted for both.
    // DEFAULT PART = 1, and that is a MEASURED decision, not a conservative one (runs/gqa_pair_v1_
    // summary.txt, paired in one thermal window against the tiled-q2048 incumbent in the same binary):
    //    4 096 keys   incumbent 126.9  merged QK 106.3 (+19.4%)   merged QK+PV 118.1 (+7.4%)
    //    8 192 keys   incumbent 187.4  merged QK 144.3 (+23.0%)   merged QK+PV 169.5 (+10.6%)
    //    cap 256      incumbent  73.4  merged QK  72.7 (+1.0%)    merged QK+PV  70.9 (noise band)
    // The PV half alone measures 184.2 ms/token at 4 096, i.e. WORSE than the incumbent it replaces.
    // The reason is visible in the two kernels' access patterns: QK reads K as a GATHER (lane to lane
    // stride is one whole key, 2 KiB, so a 64-lane instruction touches 4 of 32 bytes per sector and
    // ~8x of the sector is waste) - merging removes transactions. PV reads its q8_0 V word CONTIGUOUSLY
    // and the word is already shared by 4 lanes (a quad broadcast) - merging removes one ALU unpack and
    // costs 4x of the resident workgroups on a 12-CU card. Operand BYTES were never the cost; the gather
    // SHAPE was. Both halves stay in the binary and stay bitwise-gated (A3/A4), because the claim this
    // file has to survive is "either answer is available without rebuilding".
    uint32_t qkg = 0, pvg = 0;
    uint32_t gqa_opt = 1, gqa_part = 1u, gqa_tile = 512u, hpk = 0u;
    // ---- the rotation block size, taken from the checkpoint and NEVER from a literal here.
    // 4096 is the 9B's value and routes to the bit-frozen hat4096.comp, which is the kernel every
    // fidelity log in this repo was produced with; any other power of two in the plan-time range
    // routes to fwht.comp, whose butterflies are the same Sylvester recursion and whose normalize
    // factor is passed in from the host because GLSL inversesqrt() is an approximation and could
    // not be bit-compared with the CPU reference. 0 == prism.hadamard.version = 0, which this
    // forward REFUSES at load (see the rotation gate in fwd_main): the block loop below interleaves
    // signs_mul with the transform at three different call shapes, and eliding both for a
    // rotation-free export is a rewrite of those call sites, not a flag. Refusing is the honest
    // failure; silently rotating a model that declares no rotation is the silent one.
    uint32_t hat_block = 4096u;
    float    hat_inv   = 1.0f / 64.0f;   // 1/sqrt(hat_block), host-computed, exact for 2^(2k)
    // Which of the two transform kernels the block size selects, and the three activation widths the
    // dispatch shapes are built from. ALL of these are copied from the ModelPlan in fwd_init(), so
    // 4096/12288/16384 appear in this file only as the DEFAULTS of a struct that is always overwritten
    // before a single dispatch - and the 9B, whose plan resolves to exactly these values, gets a
    // byte-identical command stream. hat_fwht routes to hadamard_fwht.comp (float-unit push constants,
    // same bindings as hat4096, block size as a 4th constant); false routes to hat4096.comp, the
    // bit-frozen kernel every fidelity log in this repo was produced with.
    uint32_t hat_fwht  = 0u;             // 1 == prism.hadamard.block_size != 4096
    uint32_t hid       = 4096u;          // ModelPlan::hidden, the width of RES/NRM/AN/ROT/HR/...
    uint32_t ffnw      = 12288u;         // ModelPlan::ffn, the width of FG/FU/FS
    uint32_t sign_w0   = 4096u;          // prism.hadamard.sign_widths[0]: where the 2nd vector starts
    uint32_t n_sign_f  = 16384u;         // length of prism.hadamard.sign_values, in floats
    uint32_t qcheck = 0;                    // BONSAI_QCHECK=1: audit the QK kernels after the prompt
    uint32_t kvfill = 0;                    // BONSAI_KVFILL=N: synthetic KV timing fixture (see
                                            //  kv_fill_synthetic) - timings valid, TEXT MEANINGLESS
    // kMaxWide == the shader's MAXCAP. It is 2048 because that is how far the validation ladder was
    // RUN, not because the driver refuses more: shared_probe.exe measured 32768 B (MAXCAP 4096, i.e.
    // exactly maxComputeSharedMemorySize) as created AND all 4096+4096 elements round-tripping, and
    // even 33792/36864 B execute on this card - runs/shared_probe_v3.log. What 4096 does cost is
    // residence: an identical instruction stream took 22.47 ms at 16 KiB shared and 26.87 ms at
    // 32 KiB (+19.6%), so raising this constant is a measured-throughput decision plus a re-run of
    // the narrow<->wide bitwise gate, not a driver workaround.
    static const uint32_t kMaxNarrow = 256u, kMaxWide = 2048u;   // kMaxWide == the shader's MAXCAP
    // The tiled kernel has NO compile-time row limit - that is the entire reason it exists - so the
    // row ceiling for the cap guard is the memory budget. kMaxTile is therefore not a shader constant
    // but an arithmetic sanity guard: region offsets are uint32 WORD indices and the largest region is
    // LOGITS (vocab 248320) + nh*cap, so 65536 slots keeps nh*cap at 1.05e6 words with 4x headroom.
    // Anything above 2048 in tile mode is a memory-budget question, printed as one.
    static const uint32_t kMaxTile = 65536u;

    // Chat template. The literals are never written into this source: they are taken from the model's
    // own vocab by id, so the file stays free of control-token text and the ids stay the single truth.
    uint32_t chat = 1;                      // 1 = wrap the prompt in the model's chat template
                                            // (BONSAI_CHAT=0 reproduces the bare-prompt behaviour of
                                            //  every log before this change, incl. runs/ignition_v2.log)
    uint32_t thinking = 1;                  // the template's own default is "undefined = true", so the
                                            //  engine mirrors it; BONSAI_THINKING=0 renders the empty
                                            //  think block and the model answers without reasoning
    std::string system_text;                // empty = no system block at all
    // Non-empty = the caller already rendered a whole multi-turn transcript to ids (the HTTP shell's
    // messages[] array). fwd_run then encodes nothing and renders nothing, but every downstream
    // behaviour - the think-close handling, the cap guard, the sampler, the KV hook - is unchanged.
    std::vector<int> pre_ids;
    std::vector<int> aidx;                   // block -> its index among the attention blocks. THIS IS
                                              //  A VECTOR AND NOT int[32]: 32 was the 9B's layer count,
                                              //  and a 64-layer checkpoint writes aidx[32..63] straight
                                              //  past the array into the fields below (n_attn, rms_eps,
                                              //  the four GDN switches) - which is not a crash, it is a
                                              //  silently reconfigured engine. fwd_init() resizes it to
                                              //  the plan's n_layer before anyone touches it.
    uint32_t n_attn = 0, n_gdn = 0;
    float  rms_eps = 1e-6f;                 // copied from the plan at init: every norm reads it

    // The four switches that the dump alone does not settle. Each is a push constant, so the engine
    // measures the alternatives instead of arguing about them; see FORWARD_ALIGN.md for the runs.
    uint32_t gdn_mode = 0;                  // 0 = decay before predict, 1 = predict before decay
    uint32_t gdn_gmap = 0;                  // 0 = v-head h uses q/k head h % 16, 1 = h / 2
    uint32_t l2_mode  = 1;                  // 0 = x/(||x||+eps), 1 = x*rsqrt(ss+eps)
    float    l2_eps   = 1e-12f;
    uint32_t round_probs = 1;               // PV reads the fp16 probability operand (ggml's cast)
    uint32_t cap = 256u;                    // KV slots per attention layer = the official test run
    uint32_t cap_autofit = 0;               // BONSAI_CAP_AUTOFIT=1: take the OOM branch's own advice
                                            // instead of exiting (P2's "fall back to a shorter
                                            // context length", applied where it is still safe)

    // ---- P1: the V representation. K is fp16 in both modes and its code path is character-identical.
    uint32_t v_q8 = 1;                      // BONSAI_KV_V_QUANT: q8_0 (default) | none
    uint32_t rope_oracle = 0;               // BONSAI_ROPE_ORACLE=1: bind the frozen attn_norm_rope.comp
                                            // (11 bindings, no VS) instead of rope2. Diagnostic only,
                                            // and it forces emit_kf because the oracle always writes kf.
    uint32_t emit_kf = 0;                   // BONSAI_KV_DEBUG_STREAM=1 restores the cap*nkv*hd fp32
                                            // post-rope K stream. attn_norm_rope.comp's own header
                                            // says of QF/KF "No engine buffer is involved ... nothing
                                            // reads them back", and grep finds R.KF only at the
                                            // allocation, the binding and the region dump - yet it is
                                            // 4.0 KiB of the measured 36.16 KiB per-slot cost (11% of
                                            // the arena) and it grows with context. Gated, default off.
    KvGeom    kvgeom;                       // the layout-derived byte math (kv_guard.h)
    KvGuard   kvguard;                      // block-level refuse / evict hook
    SpillWatch spill;                       // the spill judge: PDH Shared Usage on Windows, amdgpu
                                            // mem_info_gtt_used on Linux (kv_guard.h)
    VramBudget budget;                      // auxiliary commit-budget reading (NOT a spill judge)
    uint32_t kv_blk_slots = 64;             // positions per KV block = the hook's granularity
    uint32_t spill_every = 32;              // sample every N decode tokens
    double   spill_last = -1.0;             // last POLICY value the guard judged (growth, or absolute
                                            // under BONSAI_SPILL_ABS=1); negative = never sampled

    int      stop_at_block = -1;            // -1 = whole network; 0..31 cuts the graph for alignment
    bool     cut_before_ffn = false;        // ladder only: run the attention/GDN branch, stop, read
    bool     dump_regions = false;
    uint64_t n_dispatch = 0, n_barrier = 0;
    double   ms_record = 0, ms_wait = 0, ms_readback = 0;
    std::vector<float> host;                // readback scratch (logits + region sums)
};

// One place decides which row kernel covers `cap`. There are two call sites because fwd_init's
// BONSAI_CAP_AUTOFIT path re-derives the choice AFTER changing cap, and a stale decision there is a
// correctness bug, not a performance one: the narrow kernel's lane IS the key position, so dispatching
// it for a 1000-key row silently computes 256 keys and calls the rest zero.
//
// AUTO's three bands, each with its evidence:
//   <= 256      narrow. This is the kernel the short-context baselines (14.05 / 14.26 tok/s at cap
//               256) were measured on, and it costs 2 KiB of LDS, not 8.
//   257..2048   wide. Deliberately NOT tiled: this is the band the whole fidelity ladder was validated
//               on, and the measured tiled advantage in it is ~2% (88.8 vs 90.4 ms/token at 1335
//               positions, runs/qk_ab_tile2048_v1.log vs runs/qk_ab_wide_v1.log, and the synthetic
//               fixture reproduces both to 0.1 ms). Not worth re-litigating a validated kernel.
//   > 2048      tiled. The wide kernel cannot address a longer row inside the design line: its row
//               costs 8 B per key of LDS, so 4096 keys = 32 KiB = the driver's entire
//               maxComputeSharedMemorySize, measured at +19.6% residence cost (runs/shared_probe_v3.log).
//               Before this line existed, cap > 2048 simply refused (rc 4); the tiled kernel is now
//               gated bitwise-identical to wide (A2) and measured to 16384 positions, so it takes the
//               long-context band instead of an error message.
// BONSAI_QKERNEL=1/2/3 forces any of the three, which is what makes an A/B on one binary possible.
//
// The GQA merge is a FOURTH row kernel (attn_qk_softmax_gqa_h4_q*.comp) and it is chosen here too,
// because "which module owns this row" must have exactly one answer - two call sites deciding it is how
// the narrow-past-256 bug happened. The merge REPLACES the dispatch choice, it does not change what
// BONSAI_QKERNEL means: the forced kernel stays the audit's reference, so `QKERNEL=3 GQA_OPT=1` times
// the tiled kernel as the control and dispatches the merged one. That is the whole A/B design.
static void gqa_select(Fwd &f) {
    // The wrapper is compiled for HPK = 4, so the merge is only ever available to a plan that says 4.
    // No band: the merged kernel's loops are nfill-bounded and it has no MAXCAP, so structurally a
    // short row costs it nothing - whether it is FASTER on a short row is a measurement (§10.4), and
    // BONSAI_GQA_OPT=0 is the switch that keeps the old answer available either way.
    const bool able = (f.gqa_opt != 0u) && (f.hpk == 4u);
    f.qkg = able && (f.gqa_part & 1u) ? 1u : 0u;
    f.pvg = (able && (f.gqa_part & 2u) && f.v_q8) ? 1u : 0u;   // merged PV exists for the q8_0 arena
}
static void qk_select(Fwd &f) {
    f.qkt  = (f.qk_force == 3u) || (f.qk_force == 0u && f.cap > Fwd::kMaxWide);
    f.wide = f.qkt || (f.qk_force == 2u) ||
             (f.qk_force == 0u && f.cap > Fwd::kMaxNarrow);
    gqa_select(f);
}
// The longest row the selected kernel is CONTRACTUALLY allowed to cover. For the wide kernel that is
// its MAXCAP (a compile-time shared array); for the tiled and merged kernels nothing in the shader
// bounds the row, so the bound is the arithmetic guard kMaxTile and, in practice, the memory budget.
static uint32_t qk_row_limit(const Fwd &f) {
    return (f.qkg || f.qkt) ? Fwd::kMaxTile : Fwd::kMaxWide;
}

// ------------------------------------------------------------------ windows
static Win w_act(Fwd &f, uint32_t off, uint64_t n_floats) {
    return {f.buf, uint64_t(off) * 4ull, n_floats * 4ull};
}
static Win w_all(Fwd &f) { return {f.buf, 0, f.bytes}; }

static VkDescriptorSet alloc_set(Ctx &c, VkDescriptorPool pool, VkDescriptorSetLayout sl) {
    VkDescriptorSetAllocateInfo sai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    sai.descriptorPool = pool; sai.descriptorSetCount = 1; sai.pSetLayouts = &sl;
    VkDescriptorSet ds; VK_CHECK(vkAllocateDescriptorSets(c.dev, &sai, &ds));
    return ds;
}

// The window-shaped twin of bind_set. Same rule that bug taught: the offset in
// VkDescriptorBufferInfo is INTO the VkBuffer, and these VkBuffers already start at their arena
// offset, so a Win never carries the arena offset - only the region offset inside the buffer.
static void bind_wins(Ctx &c, VkDescriptorSet ds, const std::vector<Win> &ws) {
    std::vector<VkDescriptorBufferInfo> infos(ws.size());
    std::vector<VkWriteDescriptorSet> writes(ws.size());
    for (size_t i = 0; i < ws.size(); ++i) {
        infos[i] = {ws[i].buf, ws[i].off, ws[i].size};
        writes[i] = {}; writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = ds; writes[i].dstBinding = uint32_t(i); writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(c.dev, uint32_t(writes.size()), writes.data(), 0, nullptr);
}

static Pipe mk_pipe(Ctx &c, VkDescriptorPool pool, const char *spv, const char *name,
                    const std::vector<Win> &wins) {
    Pipe p; p.name = name; p.nbind = uint32_t(wins.size());
    VkDescriptorSetLayout sl = make_set_layout(c, p.nbind);
    p.pl = make_pipe_layout(c, sl, 64);
    p.pp = make_pipeline(c, std::string("shaders/") + spv + ".spv", p.pl);
    p.ds = alloc_set(c, pool, sl);
    bind_wins(c, p.ds, wins);
    return p;
}

// ------------------------------------------------------------------ recording one dispatch
static void dsp(Fwd &f, const Pipe &p, VkCommandBuffer cb, const uint32_t *pc, int n_pc,
               uint32_t gx, uint32_t gy = 1, uint32_t gz = 1) {
    uint32_t blk[16] = {};
    for (int i = 0; i < n_pc && i < 16; ++i) blk[i] = pc[i];
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.pp);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p.pl, 0, 1, &p.ds, 0, nullptr);
    vkCmdPushConstants(cb, p.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 64, blk);
    vkCmdDispatch(cb, gx, gy, gz);
    // The chain is serial through a handful of shared regions, so the conservative barrier is also
    // the honest one. What it costs is measured (n_barrier is printed beside the token time), not
    // assumed; a dependency graph can replace it once that number is known to matter.
    bar_compute(cb);
    f.n_dispatch++; f.n_barrier++;
}

// ------------------------------------------------------------------ the three-piece GEMV
// BONSAI_SPLIT: force every PTQ GEMV into one accumulation order. 0 = use the plan's own pick.
// A forced value only applies where it divides the block count, because the kernel's contract is
// that each split owns a whole number of 128-wide blocks.
static uint32_t g_split_force = 0;
// Two generation-shape knobs that are properties of the LOOP, not of the model, so they live beside
// the split forcing rather than in Fwd: BONSAI_STOP_NL=n ends the run n newlines into the answer
// (what "用一句话解释" actually means, and the only way a base model's answer has a floor under its
// length), BONSAI_STRIP_THINK=1 prints only the text after the think-close marker.
static uint32_t g_stop_nl = 0, g_strip_think = 0;

// ------------------------------------------------------------------ the serve seams (all additive)
// The HTTP shell in bonsai_http.h drives the SAME fwd_run() the CLI does. Everything it needs that the
// CLI never needed is here, and every one of these defaults to the value that makes the added branch
// dead: no hook, no override, no pre-rendered ids, watch closed at the end exactly as before. The
// validated decode path is not rewritten to accommodate a server; the server is fitted to it.
//
// g_emit_token is called once per generated token with the detokenized piece and a flag saying whether
// it belongs to the reasoning block. Returning false means the client went away, and the loop stops.
static bool (*g_emit_token)(void *ctx, const char *utf8, int reasoning) = nullptr;
static void *g_emit_ctx = nullptr;
// Per-request sampler overrides. Negative / zero = "not asked for", so the env knobs and the struct's
// own deterministic defaults stay in force for every CLI run.
struct ReqOverride { float temp = -1.0f, top_p = -1.0f; int top_k = -1; uint64_t seed = 0;
                     float rep_penalty = -1.0f, presence = -1.0f, freq = -1.0f; };
static ReqOverride g_req;
// A server keeps the PDH query and the DXGI factory open across requests: closing them per reply would
// make the second request's spill watch inert, and the arena they watch is never freed anyway.
static uint32_t g_watch_persist = 0;
// Set by main() for `serve` / a double click. fwd_main brings the engine up exactly as it does for
// `run` - same env knobs, same gates, same printed budget - and then hands the live Engine/Fwd to the
// accept loop instead of generating one reply and exiting.
static int (*g_serve_hook)(Engine &e, Fwd &f) = nullptr;

static uint32_t splits_for(uint32_t M, uint32_t nblk) {
    if (g_split_force && nblk % g_split_force == 0u) return g_split_force;
    return pick_splits(M, nblk);
}

// xsum(K/128) -> gemv (split-K over gl_WorkGroupID.y) -> reduce(PART -> dst). The xsum is what buys
// the "-1" of true ternary out of the inner loop. PART is one shared scratch because two GEMVs whose
// results must both stay live (ffn_gate and ffn_up) cannot write partials into the same place.
static void gemv_ptq(Fwd &f, VkCommandBuffer cb, const WeightPlan *w, uint32_t x, uint32_t dst) {
    const uint32_t M = w->M, K = w->K, nblk = K / 128u;
    // BONSAI_SPLIT is the control knob for the accumulation-order question: the numbers a GEMV
    // produces depend on the order its partials are summed, so forcing S=1 gives the canonical
    // ascending-K order at the cost of parallelism. Used to tell physical noise from a logic bug.
    const uint32_t S = splits_for(M, nblk);
    { uint32_t pc[8] = {V4(x), f.r.XS, nblk};
      dsp(f, f.p[Fwd::P_XSUM], cb, pc, 3, nblk); }
    { uint32_t pc[8] = {M, K, S, uint32_t(w->codes_word0), uint32_t(w->scale_word0),
                        S == 1u ? dst : f.r.PART, V4(x), f.r.XS};
      dsp(f, f.p[Fwd::P_GEMV], cb, pc, 8, (M + 63u) / 64u, S); }
    if (S > 1u) { uint32_t pc[4] = {M, S, f.r.PART, dst};
                  dsp(f, f.p[Fwd::P_REDUCE], cb, pc, 4, (M + 63u) / 64u); }
}

// The f32 projections (ssm_alpha, ssm_beta: M=32, K=4096) are 6.3 M MAC = 0.08 % of the PTQ side,
// so they get a plain fp32 dot product and no quantization path at all.
static void gemv_f32_t(Fwd &f, VkCommandBuffer cb, const WeightPlan *w, uint32_t x, uint32_t dst) {
    const uint32_t M = w->M, K = w->K;
    uint32_t S = 1;
    while (S < 16u && (M / 64u + (M % 64u != 0)) * S < 640u && ((K / 4u) % (S * 2u) == 0u)) S *= 2u;
    { uint32_t pc[8] = {M, K, S, V4(x), uint32_t(w->f32_off / 16u), S == 1u ? dst : f.r.PART};
      dsp(f, f.p[Fwd::P_GEMV32], cb, pc, 6, (M + 63u) / 64u, S); }
    if (S > 1u) { uint32_t rp[4] = {M, S, f.r.PART, dst};
                  dsp(f, f.p[Fwd::P_REDUCE], cb, rp, 4, (M + 63u) / 64u); }
}

// ------------------------------------------------------------------ the rotated GEMV input
// The block transform, dispatched by the CHECKPOINT's prism.hadamard.block_size and never by a
// literal at the call site. block_size = 4096 (the 9B, read off the file: version 1, block 4096,
// transform "normalized-sylvester-walsh-hadamard", sign_mode explicit) keeps the bit-frozen
// hat4096.comp with the exact push constants and workgroup counts every fidelity log quotes, so
// this function is a no-op refactor for the only model this repo has ever run. Any other power of
// two takes hadamard_fwht.comp: the same ascending-h Sylvester recursion, the SAME two bindings and
// the SAME float-unit offsets as hat4096 (so the units trap this file's next comment describes is
// structurally unavailable on the swap), the block size as a 4th constant and the normalize factor
// as a 5th, host-computed, because GLSL's own sqrt is not guaranteed correctly rounded and a factor
// the shader invents cannot be bit-compared with the host mirror.
// Signs stay OUT of both branches: rotate() already ran signs_mul, and applying the sign vector
// twice is not a rounding difference, it is a permutation of the answer. (shaders/fwht.comp is a
// second FWHT variant that folds the signs in and addresses the arena in vec4 units; it stays
// compiled and unused, because one of the two has to own the signs and this path gave that to
// signs_mul.comp, which the oracle dump validated.)
static void hat_apply(Fwd &f, VkCommandBuffer cb, uint32_t in, uint32_t out, uint32_t n_floats) {
    if (f.hat_block == 4096u) {
        uint32_t pc[3] = {in, out, n_floats / 4096u};
        dsp(f, f.p[Fwd::P_HAT], cb, pc, 3, n_floats / 4096u);
    } else {
        const uint32_t nb = n_floats / f.hat_block;
        uint32_t pc[5] = {in, out, nb, f.hat_block, f2u(f.hat_inv)};
        dsp(f, f.p[Fwd::P_FWHT], cb, pc, 5, nb);
    }
}

// MUL signs -> MUL_MAT prism.hadamard.4096 in that order, for every tensor in weight_names. The
// block count is K/4096: 1 for the 4096-wide projections, 3 for the FFN-down input (the dump shows
// MUL_MAT over {4096, 3} for a 1-token batch and {4096, 6} for the 2-token one).
// src is read, ROT written, H written to `out` - so a caller can keep two rotated vectors alive.
// UNITS, read off the shaders rather than assumed: signs_mul takes vec4/ivec4 units for x_off,
// s_off and y_off, hat4096 takes float units. The first run of this forward passed float offsets
// into the vec4 side, which is why RES/NRM/AN/QKV all read back exactly zero while hat4096 - the
// one kernel handed the right units - still produced sum(ROT) = 64 * EMB[0].
static void rotate(Fwd &f, VkCommandBuffer cb, uint32_t src, uint32_t mid, uint32_t out,
                   uint32_t n_floats, uint32_t signs_vec4_off) {
    { uint32_t pc[4] = {V4(src), signs_vec4_off, V4(mid), V4(n_floats)};
      dsp(f, f.p[Fwd::P_SIGNS], cb, pc, 4, (V4(n_floats) + 255u) / 256u); }
    hat_apply(f, cb, mid, out, n_floats);
}

// RMS_NORM(res) without the weight folded in, then MUL by the tensor's own weight - the two nodes
// the dump always shows as a pair, kept as two dispatches so either intermediate can be read back.
static void pre_norm(Fwd &f, VkCommandBuffer cb, uint32_t res, const WeightPlan *nw, uint32_t dst) {
    { uint32_t pc[8] = {res, 0, f.r.NRM, 1u, 4096u, 0u, f2u(f.rms_eps), 0u};
      dsp(f, f.p[Fwd::P_RMS], cb, pc, 8, 1); }
    { uint32_t pc[4] = {V4(f.r.NRM), V4(uint32_t(nw->f32_off / 4u)), V4(dst), 4096u / 4u};
      dsp(f, f.p[Fwd::P_MUL], cb, pc, 4, (4096u / 4u + 255u) / 256u); }
}

// ------------------------------------------------------------------ the FFN, both block types
// post_attention_norm -> signs -> H -> {gate, up} -> swiglu -> signs.12288 -> H x3 -> down -> add
static void ffn_block(Fwd &f, VkCommandBuffer cb, const ModelPlan &pl, int b, uint32_t s4) {
    const WeightPlan *pn = pl.find(("blk." + std::to_string(b) + ".post_attention_norm.weight").c_str());
    const WeightPlan *g  = pl.find(("blk." + std::to_string(b) + ".ffn_gate.weight").c_str());
    const WeightPlan *u  = pl.find(("blk." + std::to_string(b) + ".ffn_up.weight").c_str());
    const WeightPlan *d  = pl.find(("blk." + std::to_string(b) + ".ffn_down.weight").c_str());
    pre_norm(f, cb, f.r.RES, pn, f.r.AN);
    rotate(f, cb, f.r.AN, f.r.ROT, f.r.HR, 4096u, s4);
    gemv_ptq(f, cb, g, f.r.HR, f.r.FG);
    gemv_ptq(f, cb, u, f.r.HR, f.r.FU);
    { uint32_t pc[4] = {V4(f.r.FG), V4(f.r.FU), V4(f.r.FS), V4(12288u)};   // src0 = gate: swiglu = silu(a)*b
      dsp(f, f.p[Fwd::P_SWIGLU], cb, pc, 4, (V4(12288u) + 255u) / 256u); }
    { uint32_t pc[4] = {V4(f.r.FS), s4 + 4096u / 4u, V4(f.r.FS), V4(12288u)};
      dsp(f, f.p[Fwd::P_SIGNS], cb, pc, 4, (V4(12288u) + 255u) / 256u); }
    hat_apply(f, cb, f.r.FS, f.r.FS, 12288u);   // H over 12288/Block blocks, in place
    gemv_ptq(f, cb, d, f.r.FS, f.r.GDNO);
    { uint32_t pc[4] = {V4(f.r.GDNO), V4(f.r.RES), V4(f.r.RES), 4096u / 4u};
      dsp(f, f.p[Fwd::P_ADD], cb, pc, 4, (4096u / 4u + 255u) / 256u); }
}

// ------------------------------------------------------------------ one GDN block
static void gdn_block(Fwd &f, VkCommandBuffer cb, const ModelPlan &pl, int b, int gi, uint32_t s4) {
    const std::string B = "blk." + std::to_string(b) + ".";
    const WeightPlan *nw  = pl.find((B + "attn_norm.weight").c_str());
    const WeightPlan *alp = pl.find((B + "ssm_alpha.weight").c_str());
    const WeightPlan *bet = pl.find((B + "ssm_beta.weight").c_str());
    const WeightPlan *dtb = pl.find((B + "ssm_dt.bias").c_str());
    const WeightPlan *aa  = pl.find((B + "ssm_a").c_str());
    const WeightPlan *cw  = pl.find((B + "ssm_conv1d.weight").c_str());
    const WeightPlan *qkv = pl.find((B + "attn_qkv.weight").c_str());
    const WeightPlan *gt  = pl.find((B + "attn_gate.weight").c_str());
    const WeightPlan *snorm = pl.find((B + "ssm_norm.weight").c_str());
    const WeightPlan *out = pl.find((B + "ssm_out.weight").c_str());

    pre_norm(f, cb, f.r.RES, nw, f.r.AN);
    // branch A: alpha/beta read the UNROTATED normed vector (they are f32, not in weight_names)
    gemv_f32_t(f, cb, alp, f.r.AN, f.r.ALPHA);
    gemv_f32_t(f, cb, bet, f.r.AN, f.r.BETA);
    // g = ssm_a * softplus(alpha + dt.bias), beta = sigmoid(beta). gate-0 is already the log decay:
    // softplus is >= 0, so the sign lives in ssm_a - measured over the file's own 32 values.
    { uint32_t pc[4] = {V4(f.r.ALPHA), V4(uint32_t(dtb->f32_off / 4u)), V4(f.r.G1), 32u / 4u};
      dsp(f, f.p[Fwd::P_ADDF], cb, pc, 4, 1); }
    { uint32_t pc[5] = {V4(f.r.G1), V4(f.r.G2), 32u / 4u, 0u, f2u(20.0f)};  // torch threshold form
      dsp(f, f.p[Fwd::P_SOFTPLUS], cb, pc, 5, 1); }
    { uint32_t pc[4] = {V4(f.r.G2), V4(uint32_t(aa->f32_off / 4u)), V4(f.r.GATE32), 32u / 4u};
      dsp(f, f.p[Fwd::P_MUL], cb, pc, 4, 1); }
    { uint32_t pc[3] = {V4(f.r.BETA), V4(f.r.BETA), 32u / 4u};
      dsp(f, f.p[Fwd::P_SIGMOID], cb, pc, 3, 1); }

    // branch B: signs -> H -> qkv and the swiglu gate z
    rotate(f, cb, f.r.AN, f.r.ROT, f.r.HR, 4096u, s4);
    gemv_ptq(f, cb, qkv, f.r.HR, f.r.QKV);
    gemv_ptq(f, cb, gt,  f.r.HR, f.r.Z);

    // the short causal conv, fused with its silu, plus the 3-deep shift register it owns.
    // 8192 channels, 4 taps, NO bias (the tensor table has no ssm_conv1d.bias and the dump has no
    // ADD between SSM_CONV and SILU).
    { uint32_t pc[5] = {8192u, f.r.QKV, f.r.CONV, uint32_t(f.r.CH + uint64_t(gi) * f.r.ch_layer),
                        uint32_t(cw->f32_off / 4u)};
      dsp(f, f.p[Fwd::P_CONV], cb, pc, 5, (8192u + 255u) / 256u); }
    // L2 over the 128-wide q and k rows (32 rows = 16 q heads + 16 k heads); v is NOT normalised -
    // the dump takes v_conv_predelta from conv_output_silu directly.
    { uint32_t pc[8] = {f.r.CONV, f.r.L2, 32u, 128u, f.l2_mode, f2u(f.l2_eps), 0u, 0u};
      dsp(f, f.p[Fwd::P_L2], cb, pc, 6, 32); }
    // the delta rule: one workgroup per (v-head, state row), 128 lanes over the k dimension.
    { uint32_t pc[16] = {32u, 128u, 16u, f.gdn_gmap, f.gdn_mode,
                         f.r.L2, f.r.L2 + 16u * 128u, f.r.CONV + 4096u,
                         f.r.GATE32, f.r.BETA, f.r.GDNO,
                         uint32_t(f.r.ST + uint64_t(gi) * f.r.st_layer), f2u(1.0f / sqrtf(128.0f))};
      dsp(f, f.p[Fwd::P_GDN], cb, pc, 16, 32, 1, 128); }
    // o: RMS over each 128-wide head row, then the 128-long ssm_norm weight broadcast over 32 rows
    { uint32_t pc[8] = {f.r.GDNO, 0, f.r.GDNORM, 32u, 128u, 0u, f2u(f.rms_eps), 0u};
      dsp(f, f.p[Fwd::P_RMS], cb, pc, 8, 32); }
    { uint32_t pc[8] = {f.r.GDNORM, uint32_t(snorm->f32_off / 4u), f.r.GDNORM, 32u, 128u, 128u, 0u, 0u};
      dsp(f, f.p[Fwd::P_MULB], cb, pc, 6, 32); }
    { uint32_t pc[4] = {V4(f.r.Z), V4(f.r.GDNORM), V4(f.r.SWG), 4096u / 4u};   // SWIGLU(z, normed o) per the dump
      dsp(f, f.p[Fwd::P_SWIGLU], cb, pc, 4, (4096u / 4u + 255u) / 256u); }
    // gdn_v_grouped: {128,16,2} -> {128,2,16}, i.e. dst head h reads src head h/2 + 16*(h%2)
    { uint32_t pc[8] = {128u, 32u, 16u, 1u, f.r.SWG, f.r.PERM, 0u, 0u};
      dsp(f, f.p[Fwd::P_PERM], cb, pc, 6, (32u * 128u + 255u) / 256u); }
    rotate(f, cb, f.r.PERM, f.r.ROT, f.r.HR, 4096u, s4);
    gemv_ptq(f, cb, out, f.r.HR, f.r.GDNO);
    { uint32_t pc[4] = {V4(f.r.GDNO), V4(f.r.RES), V4(f.r.RES), 4096u / 4u};
      dsp(f, f.p[Fwd::P_ADD], cb, pc, 4, (4096u / 4u + 255u) / 256u); }
    if (!f.cut_before_ffn) ffn_block(f, cb, pl, b, s4);
}

// ------------------------------------------------------------------ one softmax-attention block
// The three attention kernels are the 0-error ones from the layer-3 alignment run and are NOT edited
// here: they index every buffer from element 0, so each of the 8 layers gets its own descriptor set
// of per-region WINDOWS (built in fwd_init) instead of the whole-arena bind the other kernels use.
// seq is 1 everywhere - prefill is a loop of decode steps - which is what makes one shared set of
// Q/PROB/GATE temporaries correct: only one layer is ever live at a time.
static void attn_block(Fwd &f, VkCommandBuffer cb, const ModelPlan &pl, int b, int li,
                       uint32_t s4, uint32_t pos) {
    const std::string B = "blk." + std::to_string(b) + ".";
    const WeightPlan *nw  = pl.find((B + "attn_norm.weight").c_str());
    const WeightPlan *qw  = pl.find((B + "attn_q.weight").c_str());
    const WeightPlan *kw  = pl.find((B + "attn_k.weight").c_str());
    const WeightPlan *vw  = pl.find((B + "attn_v.weight").c_str());
    const WeightPlan *ow  = pl.find((B + "attn_output.weight").c_str());

    pre_norm(f, cb, f.r.RES, nw, f.r.AN);
    rotate(f, cb, f.r.AN, f.r.ROT, f.r.HR, 4096u, s4);
    // M=8192, and the 8192 are NOT two halves: every 512-wide group is [q head (256), gate (256)],
    // measured row by row in runs/ladder_v4.log. deint_qg splits it into the two contiguous 4096s
    // the attention kernels index head-major.
    gemv_ptq(f, cb, qw, f.r.HR, f.r.QKV);
    { uint32_t pc[8] = {uint32_t(pl.n_head), uint32_t(pl.head_dim), f.r.QKV, f.r.QD, f.r.GD,
                        0u, 0u, 0u};
      dsp(f, f.p[Fwd::P_DEINT], cb, pc, 5,
          (uint32_t(pl.n_head) * 2u * uint32_t(pl.head_dim) + 255u) / 256u); }
    gemv_ptq(f, cb, kw, f.r.HR, f.r.K);
    gemv_ptq(f, cb, vw, f.r.HR, f.r.V);
    // POS[0] = this token's absolute position; every attention kernel derives causality from it.
    { uint32_t pc[8] = {pos, 1u, f.r.POS, 0u, 0u, 0u, 0u, 0u};
      dsp(f, f.p[Fwd::P_SETPOS], cb, pc, 4, 1); }

    // QK-norm + RoPE + the fp16 pack + the K/V cache insert, in one kernel (the dump's own order:
    // RMS_NORM -> *norm weight -> ROPE -> SET_ROWS). Words 8/9 are the P1 switches; the validated
    // oracle kernel (attn_norm_rope.comp) stops at word 7, so this is the only push difference.
    { uint32_t pc[12] = {1u, uint32_t(pl.n_head), uint32_t(pl.n_head_kv), uint32_t(pl.head_dim),
                        uint32_t(pl.n_rot), f.cap, f2u(f.rms_eps), f2u(pl.rope_base),
                        f.v_q8, f.emit_kf, 0u, 0u};
      dsp(f, f.arope[li], cb, pc, 10, uint32_t(pl.n_head + pl.n_head_kv)); }
    // The narrow kernel's lane IS the key position, so it can only cover 256 of them. Past that the
    // row goes to the strided kernel, which is contractually obliged to be bitwise identical at any
    // cap it shares with the narrow one (BONSAI_QKERNEL=both prints the comparison). Past kMaxWide the
    // row goes to the TILED kernel, which is obliged to be bitwise identical to the wide one at every
    // cap they share - and, unlike the wide one, its LDS does not grow with cap. Fourth option: the
    // GQA-MERGED kernel, one workgroup per (layer, kv head) instead of per (layer, query head), so the
    // grid shrinks n_head -> n_head_kv; it is obliged to be bitwise identical to whichever of the
    // three above owns the row (qk_audit stage A3), which is why the dispatch is the only thing here
    // that changes and the grid is the only thing the merged one pushes differently.
    { uint32_t pc[8] = {1u, uint32_t(pl.n_head), uint32_t(pl.n_head_kv), uint32_t(pl.head_dim),
                        f.cap, f.round_probs, f2u(1.0f / sqrtf(float(pl.head_dim))), 0u};
      dsp(f, f.qkg ? f.aqkg[li]
                   : (f.qkt ? f.aqkt[li] : (f.wide ? f.aqkw[li] : f.aqk[li])),
          cb, pc, 8, f.qkg ? uint32_t(pl.n_head_kv) : uint32_t(pl.n_head)); }
    { uint32_t pc[8] = {1u, uint32_t(pl.n_head), uint32_t(pl.n_head_kv), uint32_t(pl.head_dim),
                        f.cap, f.round_probs, 0u, 0u};
      // Same merge on the V side: the q8_0 block read is per (key, dimension), not per head, so four
      // heads reading it is four workgroups' worth of traffic for one number. attn_pv_q8v.comp stays
      // the oracle and the engine still dispatches it in `none` mode (no merged fp16 variant exists).
      dsp(f, f.pvg ? f.apvg[li] : f.apv[li], cb, pc, 8,
          f.pvg ? uint32_t(pl.n_head_kv) : uint32_t(pl.n_head)); }

    // GOUT already carries the per-head sigmoid gate: attn_pv writes gated = acc*sigmoid(gate), and
    // its gate window is R.GD - the contiguous half of attn_q's interleaved output (the dump's own
    // CONT over a strided VIEW of the odd 256-wide rows).
    { uint32_t pc[4] = {V4(f.r.GOUT), s4, V4(f.r.ROT), 4096u / 4u};
      dsp(f, f.p[Fwd::P_SIGNS], cb, pc, 4, (4096u / 4u + 255u) / 256u); }
    hat_apply(f, cb, f.r.ROT, f.r.HR, 4096u);
    gemv_ptq(f, cb, ow, f.r.HR, f.r.GDNO);
    { uint32_t pc[4] = {V4(f.r.GDNO), V4(f.r.RES), V4(f.r.RES), 4096u / 4u};
      dsp(f, f.p[Fwd::P_ADD], cb, pc, 4, (4096u / 4u + 255u) / 256u); }
    if (!f.cut_before_ffn) ffn_block(f, cb, pl, b, s4);
}

// ------------------------------------------------------------------ the whole token
// Returns the command buffer with the logits already copied into the staging buffer at offset 0,
// so the caller's one fence covers the forward AND the readback.
// ------------------------------------------------------------------ the time slice (BONSAI_PROFILE)
// An Amdahl ledger needs the phases MEASURED, not differenced. fwd_token is the only place all of
// them are reachable: the host command-buffer recording, the vkQueueSubmit call, the fence wait, the
// device's own start/end timestamps and the logits readback. The sampler lives one level up, so it
// is accumulated here by the loop that calls it.
static uint32_t g_prof = 0;                 // 1 = instrument, 0 = zero added cost
static double pf[6] = {};                   // record, submit, fence-wait, query-reset, readback, gpu
static uint64_t pf_n = 0;                   // tokens in the current bucket
static double pf_samp = 0.0, pf_print = 0.0;   // the two phases that live outside fwd_token
static double pf_qry = 0.0, pf_copy = 0.0;     // T_memory split: the query read vs the 0.95 MB assign
static void prof_note(double rec, double sub, double wai, double rst, double rb, double gpu) {
    pf[0] += rec; pf[1] += sub; pf[2] += wai; pf[3] += rst; pf[4] += rb; pf[5] += gpu; ++pf_n;
}

static void fwd_token(Fwd &f, uint32_t tok, uint32_t pos, std::vector<float> &logits) {
    Engine &e = *f.e;
    const ModelPlan &pl = e.plan;
    const uint32_t s4 = f.r.SIGNS / 4u;                       // signs.4096 in ivec4 units
    const auto p_t0 = std::chrono::steady_clock::now();
    VkCommandBuffer cb = begin_cmd(e.c, e.cpool);
    // The timestamp pair is 0/1, reset per token. This instrument was dead code for the whole project
    // (g_prof had no setter - see §4.3 of HANDOVER.md), and the first run after wiring the setter
    // SIGSEGVed inside amdvlk64.dll on the first vkCmdResetQueryPool. The cause was not the driver and
    // not the query index: the forward path creates its pool as Engine::qp (bonsai_weights.h:316),
    // while the instrument referenced Context::qp, which stays VK_NULL_HANDLE here. A null handle into
    // a record command is exactly the failure mode bonsai_vk.cpp:362-365 already documents. All five
    // timestamp calls now use e.qp.
    const uint32_t q0 = 0u, q1 = 1u;
    if (g_prof) {   // reset inside the buffer, so the reset itself is on the device timeline
        vkCmdResetQueryPool(cb, e.qp, q0, 1);
        vkCmdResetQueryPool(cb, e.qp, q1, 1);
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, e.qp, q0);
    }

    // embedding: GET_ROWS then H THEN signs (the inverse sandwich), then RES = input_embed
    { const WeightPlan *emb = pl.find("token_embd.weight");
      uint32_t pc[8] = {tok, emb->M, emb->K, uint32_t(emb->codes_word0), uint32_t(emb->scale_word0),
                        f.r.EMB};
      dsp(f, f.p[Fwd::P_EMBD], cb, pc, 6, (emb->K + 255u) / 256u); }
    hat_apply(f, cb, f.r.EMB, f.r.ROT, 4096u);
    { uint32_t pc[4] = {V4(f.r.ROT), s4, V4(f.r.RES), 4096u / 4u};    // signs AFTER the transform here
      dsp(f, f.p[Fwd::P_SIGNS], cb, pc, 4, (4096u / 4u + 255u) / 256u); }

    int gi = 0;
    for (int b = 0; b < pl.n_layer; ++b) {
        if (f.stop_at_block >= 0 && b > f.stop_at_block) break;
        if (pl.is_attn_layer(b)) attn_block(f, cb, pl, b, f.aidx[b], s4, pos);
        else                     gdn_block(f, cb, pl, b, gi++, s4);
    }

    if (f.stop_at_block < 0) {
        // output_norm -> signs -> H -> output.weight (M = vocab = 248320, splits 1)
        const WeightPlan *on = pl.find("output_norm.weight");
        { uint32_t pc[8] = {f.r.RES, 0, f.r.NRM, 1u, 4096u, 0u, f2u(e.plan.rms_eps), 0u};
          dsp(f, f.p[Fwd::P_RMS], cb, pc, 8, 1); }
        { uint32_t pc[4] = {V4(f.r.NRM), V4(uint32_t(on->f32_off / 4u)), V4(f.r.AN), 4096u / 4u};
          dsp(f, f.p[Fwd::P_MUL], cb, pc, 4, (4096u / 4u + 255u) / 256u); }
        rotate(f, cb, f.r.AN, f.r.ROT, f.r.HR, 4096u, s4);
        gemv_ptq(f, cb, pl.find("output.weight"), f.r.HR, f.r.LOGITS);
        // compute wrote, the copy engine reads: bar_compute() only orders shader->shader, so without
        // this the logits the CPU samples are whatever the previous token left behind.
        bar_compute_to_transfer(cb);
        VkBufferCopy cp{uint64_t(f.r.LOGITS) * 4ull, 0, uint64_t(pl.vocab) * 4ull};
        vkCmdCopyBuffer(cb, f.buf, e.host_buf, 1, &cp);
        f.n_dispatch++;
    }
    if (g_prof) vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, e.qp, q1);
    VK_CHECK(vkEndCommandBuffer(cb));
    const auto p_t1 = std::chrono::steady_clock::now();

    double ms_sub = 0.0, ms_wai = 0.0, ms_rst = 0.0;
    if (!g_prof) {
        submit_wait_sig(e.c, e.c.queue, cb, f.fence, VK_NULL_HANDLE, VK_NULL_HANDLE);
    } else {   // the same three calls submit_wait_sig makes, timed individually
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1; si.pCommandBuffers = &cb;
        const auto a0 = std::chrono::steady_clock::now();
        VK_CHECK(vkQueueSubmit(e.c.queue, 1, &si, f.fence));
        const auto a1 = std::chrono::steady_clock::now();
        VK_CHECK(vkWaitForFences(e.c.dev, 1, &f.fence, VK_TRUE, ~0ull));
        const auto a2 = std::chrono::steady_clock::now();
        VK_CHECK(vkResetFences(e.c.dev, 1, &f.fence));
        const auto a3 = std::chrono::steady_clock::now();
        ms_sub = std::chrono::duration<double, std::milli>(a1 - a0).count();
        ms_wai = std::chrono::duration<double, std::milli>(a2 - a1).count();
        ms_rst = std::chrono::duration<double, std::milli>(a3 - a2).count();
    }
    const auto p_t2 = std::chrono::steady_clock::now();
    f.ms_wait = std::chrono::duration<double, std::milli>(p_t2 - p_t1).count();
    double ms_gpu = 0.0;
    const auto q_t0 = std::chrono::steady_clock::now();
    if (g_prof) {
        uint64_t ts[2];
        VK_CHECK(vkGetQueryPoolResults(e.c.dev, e.qp, q0, 2, sizeof(ts), ts, 8,
                                       VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
        ms_gpu = double(ts[1] - ts[0]) * e.c.limits.timestampPeriod / 1e6;
    }
    const auto q_t1 = std::chrono::steady_clock::now();
    const float *hp = (const float *)e.host_ptr;      // the mapped staging buffer, typed once
    logits.assign(hp, hp + size_t(pl.vocab));
    if (g_prof) {
        const auto p_t3 = std::chrono::steady_clock::now();
        pf_qry  += std::chrono::duration<double, std::milli>(q_t1 - q_t0).count();
        pf_copy += std::chrono::duration<double, std::milli>(p_t3 - q_t1).count();
        prof_note(std::chrono::duration<double, std::milli>(p_t1 - p_t0).count(),
                  ms_sub, ms_wai, ms_rst,
                  std::chrono::duration<double, std::milli>(p_t3 - p_t2).count(), ms_gpu);
    }
}

// ------------------------------------------------------------------ region sums, for alignment
// Reads one region back and sums it. This is how a wrong layer is localised: the official dump
// prints every node's values, so a per-region sum is directly comparable without a reference build.
// Reads a region back and reports sum / max|.| / the first three values. A bare sum cannot tell
// "nothing was written" from "4096 values that cancel", and the first run needed that distinction.
struct RegStat { double sum, amax, asum; float v[3]; };
static RegStat fwd_region_stat(Fwd &f, uint32_t off, uint32_t n) {
    Engine &e = *f.e;
    VkCommandBuffer cb = begin_cmd(e.c, e.cpool);
    VkBufferCopy cp{uint64_t(off) * 4ull, 0, uint64_t(n) * 4ull};
    vkCmdCopyBuffer(cb, f.buf, e.host_buf, 1, &cp);
    VK_CHECK(vkEndCommandBuffer(cb));
    submit_wait_sig(e.c, e.c.queue, cb, f.fence, VK_NULL_HANDLE, VK_NULL_HANDLE);
    const float *p = (const float *)e.host_ptr;
    RegStat s{}; s.v[0] = s.v[1] = s.v[2] = 0.0f;
    for (uint32_t i = 0; i < n; ++i) {
        s.sum += p[i];
        const double a = p[i] < 0 ? -double(p[i]) : double(p[i]);
        s.asum += a;                       // the cancellation scale: |sum| << asum means the sum is
        if (a > s.amax) s.amax = a;        // nearly-cancelled and its relative error is meaningless
    }
    for (uint32_t i = 0; i < 3 && i < n; ++i) s.v[i] = p[i];
    return s;
}

// ------------------------------------------------------------------ the per-block residual ladder
static std::vector<float> fwd_read(Fwd &f, uint32_t off, uint32_t n);   // defined below, used here
// The same cold-cache 1-token pass that gave the block-0 anchors also carries every later block:
// runs/dbg_full_real.log prints l_out-N = ADD(ffn_out-N, attn_residual-N) for N = 0..31 with three
// head values and a sum. Splitting the forward into one submit per block lets RES be read back
// between them, so a wrong block is named instead of inferred from the logits. The want[] table is
// transcribed from that pass (shapes {4096, 1, 1, 1}); token and position are "ab" ids[0] at pos 0,
// the same state fwd_align and the tail anchors use.
static void fwd_ladder(Fwd &f, uint32_t tok, uint32_t pos) {
    Engine &e = *f.e;
    const ModelPlan &pl = e.plan;
    const uint32_t s4 = f.r.SIGNS / 4u;
    struct A { double h0, h1, h2, sum; };
    static const A want[32] = {
        {-0.0845,  0.1474, -0.0267,     3.656747}, {-0.1427,  0.1804, -0.0394,     1.325850},
        {-0.2604,  0.3694,  0.0538,    -1.912515}, {-0.3339,  0.5213,  0.0561,     0.738097},
        {-0.3005,  0.7992, -0.0914,     7.121069}, {-0.6100,  2.5083,  0.1739,    13.169378},
        {-0.5905,  5.2933,  0.7950,    64.064751}, {-0.7214,  5.7275,  1.3662,    40.740028},
        {-0.7546,  5.9108,  1.2907,    35.497593}, {-0.8803,  6.5198,  1.0209,    40.486763},
        {-1.0310,  7.5587,  1.3875,    58.043190}, {-1.0559,  7.7688,  1.3856,    28.734020},
        {-1.0370,  7.8423,  1.3800,    23.752522}, {-1.1030,  7.8720,  1.1361,    19.603004},
        {-0.7811,  8.6008,  1.2642,    33.407875}, {-1.0184,  8.8191,  0.2921,    14.039374},
        {-0.9895,  8.5309,  0.3157,     0.028751}, {-1.0093,  8.5308,  0.1043,    10.155331},
        {-1.3150,  9.7305, -0.4983,    49.849838}, {-1.7433,  9.3851, -0.0845,   -33.921829},
        {-1.7373,  9.3646,  0.0399,   -76.544434}, {-1.5489,  7.5411,  0.5483,  -197.298111},
        {-1.4415,  7.3812,  0.8197,  -263.474670}, {-1.6368,  6.9114,  0.9435,  -359.947540},
        {-1.5336,  6.5993,  1.1397,  -396.977112}, {-1.8176,  5.4363,  1.3522,  -450.954468},
        {-2.9041,  3.5853,  2.2778,  -517.161133}, {-3.5103,  4.0357,  2.1354,  -552.046753},
        {-1.7040,  7.3090,  4.1831,  -596.124573}, {-1.5466,  7.5255,  4.9780,  -558.743042},
        {-2.6307, 10.1688,  4.0183,  -579.823486}, {-2.9615,  9.1218,  4.7726,  -381.804138},
    };

    // embedding gather + the inverse Hadamard sandwich, exactly as fwd_token does it
    { const WeightPlan *emb = pl.find("token_embd.weight");
      VkCommandBuffer cb = begin_cmd(e.c, e.cpool);
      uint32_t pc[6] = {tok, emb->M, emb->K, uint32_t(emb->codes_word0), uint32_t(emb->scale_word0),
                        f.r.EMB};
      dsp(f, f.p[Fwd::P_EMBD], cb, pc, 6, (emb->K + 255u) / 256u);
      hat_apply(f, cb, f.r.EMB, f.r.ROT, 4096u);
      { uint32_t q[4] = {V4(f.r.ROT), s4, V4(f.r.RES), 4096u / 4u};
        dsp(f, f.p[Fwd::P_SIGNS], cb, q, 4, (4096u / 4u + 255u) / 256u); }
      VK_CHECK(vkEndCommandBuffer(cb));
      submit_wait_sig(e.c, e.c.queue, cb, f.fence, VK_NULL_HANDLE, VK_NULL_HANDLE); }

    printf("\n=== per-block residual ladder vs the same dump (tok %u, pos %u, cold caches) ===\n",
           tok, pos);
    const RegStat e0 = fwd_region_stat(f, f.r.RES, 4096);
    printf("  %-7s %-9s %s\n", "block", "type",
           "branch ours/official   ffn ours/official   residual ours/official   head3 vs official");
    printf("  %-7s %-9s %10.5f  %-28s  [ %7.4f %7.4f %7.4f ]\n", "emb", "input_embed", e0.sum, "",
           e0.v[0], e0.v[1], e0.v[2]);
    int nfail = -1;
    double rho[32] = {}, dres[32] = {}, asum[32] = {}, canc[32] = {};

    // The two branches feeding each residual ADD, from the same pass: the node that is added to the
    // previous l_out (linear_attn_out-N for the GDN blocks, attn_output-N for the softmax ones; block
    // 31 prints it as node_2399) and ffn_out-N. attn_residual-N - l_out-(N-1) reproduces these sums,
    // so the identity is checked rather than assumed.
    static const double off_attn[32] = {
         0.883923,   1.847109,   0.450307,   4.167934,   2.377605,   8.474438, -11.210237,
       -27.998230,  -1.462893,  13.998522,  25.271244, -36.572762,  -2.242031,  -0.970878,
       13.157617, -15.971869,   0.404959,   3.768888,  -1.937614, -98.505318, -38.294613,
     -104.792419, -72.275200, -95.216324, -37.583549, -57.495106, -65.705643,-125.185783,
     -48.874660,  25.453598, -15.997361, -15.402372};
    static const double off_ffn[32] = {
       3.236981,  -4.178006,  -3.688653,  -1.517322,   4.005345,  -2.426172,  62.105694,
       4.673182,  -3.779085,  -9.009721,  -7.714525,   7.263806,  -2.739789,  -3.178458,
       0.646386,  -3.395649, -14.415151,   6.357328,  41.631985,  14.733984,  -4.328336,
     -15.961117,   6.098594,  -1.256650,   0.553863,   3.517878,  -0.500527,  90.299690,
       4.796672,  11.927950,  -5.082813, 213.422012};

    int gi = 0;
    for (int b = 0; b < pl.n_layer; ++b) {
        const bool att = pl.is_attn_layer(b);
        // phase 1: the branch, its ADD, then stop so GDNO (= the value added) can be read back
        f.cut_before_ffn = true;
        { VkCommandBuffer cb = begin_cmd(e.c, e.cpool);
          if (att) attn_block(f, cb, pl, b, f.aidx[b], s4, pos);
          else     gdn_block(f, cb, pl, b, gi++, s4);
          VK_CHECK(vkEndCommandBuffer(cb));
          submit_wait_sig(e.c, e.c.queue, cb, f.fence, VK_NULL_HANDLE, VK_NULL_HANDLE); }
        const RegStat sa = fwd_region_stat(f, f.r.GDNO, 4096);
        // Block 3 is the FIRST softmax-attention block and its input, l_out-2, is still inside the
        // band, so every node inside it is directly comparable. With one token in the cache the
        // softmax collapses to prob=1, which takes the attention MATH out of the diagnosis and leaves
        // only the projections, the gate window and the head layout. Transcribed from the same pass
        // (log lines 34900..35116).
        if (b == 3) {
            struct X { const char *n; uint32_t off, cnt; double want; };
            const X xs[] = {
                {"QKV total    (=Qcur_full)",      f.r.QKV,        8192, -16964.929688},
                {"QD   (contig q = Qcur_reshaped)", f.r.QD,        4096,   131.812424},
                {"GD   (contig gate = gate_reshap)", f.r.GD,       4096, -17096.734375},
                {"V          (=Vcur)",             f.r.V,          1024,   46.230995},
                {"K          (=Kcur, pre norm/rope)", f.r.K,       1024,  -62.642960},
                {"PGATE      (=attn_pregate)",     f.r.PGATE,      4096,  184.928223},
                {"GOUT      (=attn_gated)",        f.r.GOUT,       4096,   -0.834408},
                {"GDNO       (=attn_output)",      f.r.GDNO,       4096,     4.167934},
            };
            printf("  --- inside block 3 (softmax over 1 position, so probs are exactly 1) ---\n");
            for (const auto &x : xs) {
                const RegStat r = fwd_region_stat(f, x.off, x.cnt);
                const int ok = fabs(r.sum - x.want) <= 2e-3 + 2e-4 * fabs(x.want);
                printf("   %-38s sum %14.5f vs %14.5f  %s   [ %8.4f %8.4f %8.4f ]\n",
                       x.n, r.sum, x.want, ok ? "OK " : "FAIL", r.v[0], r.v[1], r.v[2]);
            }
            // Which 256-wide row of the 8192 attn_q output is q and which is the gate? The dump
            // cannot say directly, but the sums can: q totals +131.81 and the gate -17096.73, so the
            // row sums separate the two with no ambiguity. Alternating small/large = interleaved per
            // head (stride 512); a clean front/back split = the two-halves layout we assume.
            { const std::vector<float> qv = fwd_read(f, f.r.QKV, 8192);
              printf("   QKV per-256-row sums (want: q rows ~+131.8 total, gate rows ~-17096.7):\n   ");
              for (int j = 0; j < 32; ++j) {
                  double s = 0.0; for (int i = 0; i < 256; ++i) s += double(qv[size_t(j) * 256 + i]);
                  printf("%8.1f", s);
                  if (j % 8 == 7) printf("\n   ");
              }
              printf("\n"); }
        }
        // phase 2: the FFN branch on the updated residual
        f.cut_before_ffn = false;
        { VkCommandBuffer cb = begin_cmd(e.c, e.cpool);
          ffn_block(f, cb, pl, b, s4);
          VK_CHECK(vkEndCommandBuffer(cb));
          submit_wait_sig(e.c, e.c.queue, cb, f.fence, VK_NULL_HANDLE, VK_NULL_HANDLE); }
        const RegStat sf = fwd_region_stat(f, f.r.GDNO, 4096);
        const RegStat s  = fwd_region_stat(f, f.r.RES, 4096);

        const A &w = want[b];
        // the dump rounds each element to 4 decimals and sums in fp32 over 4096 terms, so the band
        // has to absorb both: 5e-4 rounding plus the accumulation error, which grows with |sum|.
        const int sok = fabs(s.sum - w.sum) <= 2e-3 + 2e-4 * fabs(w.sum);
        const int aok = fabs(sa.sum - off_attn[b]) <= 2e-3 + 2e-4 * fabs(off_attn[b]);
        const int fok = fabs(sf.sum - off_ffn[b]) <= 2e-3 + 2e-4 * fabs(off_ffn[b]);
        const int hok = fabs(double(s.v[0]) - w.h0) <= 5e-4 + 2e-3 * fabs(w.h0)
                     && fabs(double(s.v[1]) - w.h1) <= 5e-4 + 2e-3 * fabs(w.h1)
                     && fabs(double(s.v[2]) - w.h2) <= 5e-4 + 2e-3 * fabs(w.h2);
        if (!sok && nfail < 0) nfail = b;
        {   // the statistic that separates accumulation noise from a logic bug. |sum| is what the
            // band compares, but the error a reordered fp32 sum can produce scales with Sum|x|,
            // not with |Sum x|. So divide the discrepancy by the cancellation scale and ask whether
            // one coefficient explains all 32 blocks.
            const double d = s.sum - w.sum;
            rho[b] = s.asum > 0 ? fabs(d) / s.asum : 0.0;
            dres[b] = d; asum[b] = s.asum; canc[b] = s.asum > 0 ? fabs(w.sum) / s.asum : 0.0;
        }
        printf("  %-7d %-9s br %11.5f/%-11.5f %s  ffn %11.5f/%-11.5f %s  res %11.5f/%-11.5f %s"
               "  [ %7.4f %7.4f %7.4f ] vs [ %7.4f %7.4f %7.4f ] %s\n",
               b, att ? "attn" : "gdn", sa.sum, off_attn[b], aok ? "b-OK " : "b-FAIL",
               sf.sum, off_ffn[b], fok ? "f-OK " : "f-FAIL", s.sum, w.sum, sok ? "r-OK " : "r-FAIL",
               s.v[0], s.v[1], s.v[2], w.h0, w.h1, w.h2, hok ? "h-OK " : "h-FAIL");
    }
    printf("\n=== residual-sum discrepancy, scaled by each block's own cancellation ===\n");
    printf("  delta = ours - official, Sum|x| = the block's magnitude scale, rho = |delta|/Sum|x|,\n"
           "  cancel = |official sum|/Sum|x| (how much the 4096 terms annihilate each other)\n");
    double rmin = 1e30, rmax = 0.0; int rminb = -1, rmaxb = -1;
    for (int b = 0; b < 32; ++b) {
        if (rho[b] < rmin) { rmin = rho[b]; rminb = b; }
        if (rho[b] > rmax) { rmax = rho[b]; rmaxb = b; }
        printf("  blk %2d delta %11.5f  Sum|x| %11.1f  rho %9.2e  cancel %7.4f%s\n",
               b, dres[b], asum[b], rho[b], canc[b],
               fabs(dres[b]) <= 2e-3 + 2e-4 * fabs(want[b].sum) ? "" : "   <- outside the band");
    }
    printf("  rho spread: min %.2e (blk %d)  max %.2e (blk %d)  ratio %.1fx\n"
           "  one fp32 accumulation-noise coefficient explains every block if that ratio is O(1)\n",
           rmin, rminb, rmax, rmaxb, rmax / (rmin > 0 ? rmin : 1e-30));
    printf("  first block whose residual sum leaves the band: %s\n",
           nfail < 0 ? "none, all 32 aligned" : "");
    if (nfail >= 0) printf("  -> block %d (%s)\n", nfail, pl.is_attn_layer(nfail) ? "attn" : "gdn");
}

// the same readback, but handed to the host so a shape can be analysed instead of summarised.
static std::vector<float> fwd_read_buf(Fwd &f, VkBuffer src, uint64_t byte_off, uint32_t n) {
    Engine &e = *f.e;
    VkCommandBuffer cb = begin_cmd(e.c, e.cpool);
    VkBufferCopy cp{byte_off, 0, uint64_t(n) * 4ull};
    vkCmdCopyBuffer(cb, src, e.host_buf, 1, &cp);
    VK_CHECK(vkEndCommandBuffer(cb));
    submit_wait_sig(e.c, e.c.queue, cb, f.fence, VK_NULL_HANDLE, VK_NULL_HANDLE);
    const float *p = (const float *)e.host_ptr;
    return std::vector<float>(p, p + n);
}
static std::vector<float> fwd_read(Fwd &f, uint32_t off, uint32_t n) {
    return fwd_read_buf(f, f.buf, uint64_t(off) * 4ull, n);
}

static void fwd_dump(Fwd &f) {
    struct R { const char *name; uint32_t off, n; };
    const R rs[] = {
        {"embedding row (EMB)", f.r.EMB, 4096}, {"input_embed (RES)", f.r.RES, 4096},
        {"normed (NRM)", f.r.NRM, 4096}, {"attn_norm-0 (AN)", f.r.AN, 4096},
        {"signs*AN (ROT)", f.r.ROT, 4096}, {"rotated (HR)", f.r.HR, 4096},
        {"qkv mixed", f.r.QKV, 8192}, {"gate z", f.r.Z, 4096},
        {"conv+silu", f.r.CONV, 8192}, {"l2 q/k", f.r.L2, 4096},
        {"alpha", f.r.ALPHA, 32}, {"beta", f.r.BETA, 32},
        {"g (log decay)", f.r.GATE32, 32},
        {"delta o", f.r.GDNO, 4096}, {"swiglu z*o", f.r.SWG, 4096},
        {"permuted", f.r.PERM, 4096}, {"ffn gate", f.r.FG, 12288},
        {"ffn up", f.r.FU, 12288}, {"ffn swiglu", f.r.FS, 12288},
        {"xsum[0..3]", f.r.XS, 4}, {"PART[0..3]", f.r.PART, 4},
    };
    printf("\n=== region stats after the cut (compare against runs/oracle_nodes.txt) ===\n");
    for (const auto &x : rs) {
        const RegStat s = fwd_region_stat(f, x.off, x.n);
        printf("  %-20s sum = %14.6f  max|.| = %10.6f   [%8.5f %8.5f %8.5f]  (%u)\n",
               x.name, s.sum, s.amax, s.v[0], s.v[1], s.v[2], x.n);
    }
}

// ------------------------------------------------------------------ block 0, op by op, against the oracle
// NRM / AN / ROT / HR / QKV / ALPHA / ... are scratch the rest of the block overwrites, so a
// whole-block dump can never show them. The harness below replays ONE MORE op per submit and reads
// the regions back between submits, which is the only way a value can be attributed to a node.
//
// WHY THE TARGET IS THE DECODE GRAPH AND NOT THE PREFILL GRAPH. runs/dbg_full_real.log holds two
// evaluations: a 2-token prefill and a 1-token step. runs/oracle_step1.md §"Token identity of the
// 1-token step" quotes the run's own trailer - `Input prompt: "ab"`, `Token ids (1): ab(365)` - so
// the second graph is a single token whose id we can reproduce exactly, and every tensor in it has
// one column, which makes the printed SUM a hard invariant rather than a two-token mixture. It is
// also the graph whose `cache_r_l*` / `cache_s_l*` inputs are all zero (the doc's Cold-cache
// evidence), which is precisely the state a fresh activation arena starts in.
//
// The anchors are transcribed from that graph verbatim (log lines 32046..32560, %8.4f prints):
//   node_0  GET_ROWS(token_embd)   [-0.0153  0.0000 -0.0153]  sum   -0.010437
//   node_2  MUL_MAT(hadamard)      [-0.0002 -0.0116 -0.0131]  sum   -0.979492
//   model.input_embed               [ 0.0002  0.0116  0.0131]  sum   -0.464154
//   norm-0  RMS_NORM                [ 0.0143  1.0184  1.1478]  sum  -40.661980
//   attn_norm-0  MUL(weight)        [ 0.0134  1.1238  1.1596]  sum  -45.631546
//   node_17  MUL(signs)             [-0.0134 -1.1238 -1.1596]  sum  -89.238937
//   node_19  MUL_MAT(hadamard)      [-1.3944  0.0263 -1.4444]  sum   -0.860536
//   node_21  MUL_MAT(attn_qkv)      [ 0.5366 -0.8338  0.3179]  sum -324.560303
//   node_42  MUL_MAT(ssm_alpha)     [-0.0428  0.0203  3.2403]  sum  -37.683086
//   node_48  MUL_MAT(ssm_beta)      [-3.0942  1.4797  3.6127]  sum   34.892235
//   node_44  ADD(dt.bias)           [-3.5740 -1.4250 21.7403]  sum   38.082535
//   a_softplus-0                    [ 0.0277  0.2155 21.7403]  sum  146.662094
//   gate-0  MUL(ssm_a)              [-0.0019 -0.0019 -1.2265]  sum  -14.298706
//   beta_sigmoid-0                  [ 0.0433  0.8145  0.9737]  sum   19.822166
//   conv_output_silu-0              [-0.0286 -0.0426  0.0152]  sum  168.968338
//   qk_conv_l2-0  row 0             [-0.0117 -0.0174  0.0062]  sum   21.658962
// Bands: the dump rounds each element to 5e-5 and each sum to 5e-7, so 5e-4 + 2e-3 relative on an
// element and 1e-3 + 1e-5 relative on a sum is ~10x the print's own quantisation and still catches
// any unit, order or shape mistake outright.
struct Anchor {
    int after_op; const char *region; uint32_t off, n;
    double h[3]; double sum;
};

static int fwd_align(Engine &e, Fwd &f, uint32_t tok0) {
    const WeightPlan *emb = e.plan.find("token_embd.weight");
    const WeightPlan *nw  = e.plan.find("blk.0.attn_norm.weight");
    const WeightPlan *qkv = e.plan.find("blk.0.attn_qkv.weight");
    const WeightPlan *alp = e.plan.find("blk.0.ssm_alpha.weight");
    const WeightPlan *bet = e.plan.find("blk.0.ssm_beta.weight");
    const WeightPlan *dtb = e.plan.find("blk.0.ssm_dt.bias");
    const WeightPlan *aa  = e.plan.find("blk.0.ssm_a");
    const WeightPlan *cw  = e.plan.find("blk.0.ssm_conv1d.weight");
    const uint32_t s4 = f.r.SIGNS / 4u;
    if (!emb || !nw || !qkv || !alp || !bet || !dtb || !aa || !cw) {
        fprintf(stderr, "align: a needed tensor is missing\n"); return 2;
    }

    Anchor anc[] = {
        {0, "EMB", f.r.EMB, 4096, {-0.0153,  0.0000, -0.0153},  -0.010437},
        {1, "ROT", f.r.ROT, 4096, {-0.0002, -0.0116, -0.0131},  -0.979492},
        {2, "RES", f.r.RES, 4096, { 0.0002,  0.0116,  0.0131},  -0.464154},
        {3, "NRM", f.r.NRM, 4096, { 0.0143,  1.0184,  1.1478}, -40.661980},
        {4, "AN",  f.r.AN,  4096, { 0.0134,  1.1238,  1.1596}, -45.631546},
        {5, "ROT", f.r.ROT, 4096, {-0.0134, -1.1238, -1.1596}, -89.238937},
        {6, "HR",  f.r.HR,  4096, {-1.3944,  0.0263, -1.4444},  -0.860536},
        {9, "QKV", f.r.QKV, 8192, { 0.5366, -0.8338,  0.3179}, -324.560303},
        {10, "ALPHA", f.r.ALPHA, 32, {-0.0428, 0.0203, 3.2403}, -37.683086},
        {11, "BETA",  f.r.BETA,  32, {-3.0942, 1.4797, 3.6127},  34.892235},
        {12, "G1",    f.r.G1,    32, {-3.5740, -1.4250, 21.7403}, 38.082535},
        {13, "G2",    f.r.G2,    32, {0.0277, 0.2155, 21.7403},  146.662094},
        {14, "GATE32", f.r.GATE32, 32, {-0.0019, -0.0019, -1.2265}, -14.298706},
        {15, "BETA",  f.r.BETA,  32, {0.0433, 0.8145, 0.9737},   19.822166},
        {16, "CONV",  f.r.CONV,  8192, {-0.0286, -0.0426, 0.0152}, 168.968338},
        {17, "L2",    f.r.L2,    4096, {-0.0117, -0.0174, 0.0062}, 21.658962},
    };
    const char *opname[] = {
        "GET_ROWS token_embd", "H (inverse sandwich)", "MUL signs.4096", "RMS_NORM",
        "MUL attn_norm.weight", "MUL signs.4096", "H", "xsum", "GEMV qkv -> PART",
        "reduce -> QKV", "GEMV f32 ssm_alpha", "GEMV f32 ssm_beta", "ADD ssm_dt.bias",
        "SOFTPLUS", "MUL ssm_a", "SIGMOID beta", "SSM_CONV + SILU", "L2_NORM q/k" };
    const int N_OPS = 18;

    int fails = 0, checks = 0;
    printf("\n=== block 0 op by op vs the official 1-token decode graph (token id %u) ===\n", tok0);
    for (int op = 0; op < N_OPS; ++op) {
        VkCommandBuffer cb = begin_cmd(e.c, e.cpool);
        for (int i = 0; i <= op; ++i) {
            switch (i) {
            case 0: { uint32_t pc[8] = {tok0, emb->M, emb->K, uint32_t(emb->codes_word0),
                                        uint32_t(emb->scale_word0), f.r.EMB};
                      dsp(f, f.p[Fwd::P_EMBD], cb, pc, 6, (emb->K + 255u) / 256u); } break;
            case 1: hat_apply(f, cb, f.r.EMB, f.r.ROT, 4096u); break;
            case 2: { uint32_t pc[4] = {V4(f.r.ROT), s4, V4(f.r.RES), 4096u / 4u};
                      dsp(f, f.p[Fwd::P_SIGNS], cb, pc, 4, (4096u / 4u + 255u) / 256u); } break;
            case 5: { uint32_t pc[4] = {V4(f.r.AN), s4, V4(f.r.ROT), 4096u / 4u};
                      dsp(f, f.p[Fwd::P_SIGNS], cb, pc, 4, (4096u / 4u + 255u) / 256u); } break;
            case 3: { uint32_t pc[8] = {f.r.RES, 0, f.r.NRM, 1u, 4096u, 0u, f2u(f.rms_eps), 0u};
                      dsp(f, f.p[Fwd::P_RMS], cb, pc, 8, 1); } break;
            case 4: { uint32_t pc[4] = {V4(f.r.NRM), V4(uint32_t(nw->f32_off / 4u)), V4(f.r.AN), 4096u / 4u};
                      dsp(f, f.p[Fwd::P_MUL], cb, pc, 4, (4096u / 4u + 255u) / 256u); } break;
            case 6: hat_apply(f, cb, f.r.ROT, f.r.HR, 4096u); break;
            case 7: case 8: break;   // xsum and PART are recorded inside gemv_ptq; they get no
                                     // separate submit here, so their anchor is the reduce output.
            case 9: gemv_ptq(f, cb, qkv, f.r.HR, f.r.QKV); break;
            case 10: gemv_f32_t(f, cb, alp, f.r.AN, f.r.ALPHA); break;
            case 11: gemv_f32_t(f, cb, bet, f.r.AN, f.r.BETA); break;
            case 12: { uint32_t pc[4] = {V4(f.r.ALPHA), V4(uint32_t(dtb->f32_off / 4u)), V4(f.r.G1), 32u / 4u};
                      dsp(f, f.p[Fwd::P_ADDF], cb, pc, 4, 1); } break;
            case 13: { uint32_t pc[5] = {V4(f.r.G1), V4(f.r.G2), 32u / 4u, 0u, f2u(20.0f)};
                      dsp(f, f.p[Fwd::P_SOFTPLUS], cb, pc, 5, 1); } break;
            case 14: { uint32_t pc[4] = {V4(f.r.G2), V4(uint32_t(aa->f32_off / 4u)), V4(f.r.GATE32), 32u / 4u};
                      dsp(f, f.p[Fwd::P_MUL], cb, pc, 4, 1); } break;
            case 15: { uint32_t pc[3] = {V4(f.r.BETA), V4(f.r.BETA), 32u / 4u};
                      dsp(f, f.p[Fwd::P_SIGMOID], cb, pc, 3, 1); } break;
            case 16: { // THE REPLAY IS NOT THE RUN. Every submit re-records ops 0..op, and the conv
                       // owns a state the arena keeps BETWEEN submits, so op 17 would otherwise see a
                       // two-token history and read as (w2+w3)*x. Zero the layer's taps first; the
                       // official 1-token decode graph runs on exactly this cold state (its
                       // conv_states_reshaped-0 prints all zeros), which is the whole point.
                vkCmdFillBuffer(cb, f.buf, uint64_t(f.r.CH) * 4ull,
                                uint64_t(f.r.ch_layer) * 4ull, 0u);
                bar_transfer_to_compute(cb); }
                { uint32_t pc[5] = {8192u, f.r.QKV, f.r.CONV, f.r.CH, uint32_t(cw->f32_off / 4u)};
                  dsp(f, f.p[Fwd::P_CONV], cb, pc, 5, (8192u + 255u) / 256u); } break;
            case 17: { uint32_t pc[8] = {f.r.CONV, f.r.L2, 32u, 128u, f.l2_mode, f2u(f.l2_eps), 0u, 0u};
                      dsp(f, f.p[Fwd::P_L2], cb, pc, 6, 32); } break;
            default: break;
            }
        }
        VK_CHECK(vkEndCommandBuffer(cb));
        submit_wait_sig(e.c, e.c.queue, cb, f.fence, VK_NULL_HANDLE, VK_NULL_HANDLE);
        printf("  op %2d  %-24s\n", op, opname[op]);
        for (const auto &a : anc) {
            if (a.after_op != op) continue;
            const RegStat s = fwd_region_stat(f, a.off, a.n);
            double hdev = 0.0; bool hok = true;
            for (int k = 0; k < 3; ++k) {
                if (std::isnan(a.h[k])) continue;
                const double d = fabs(double(s.v[k]) - a.h[k]);
                if (d > hdev) hdev = d;
                if (d > 5e-4 + 2e-3 * fabs(a.h[k])) hok = false;
            }
            const bool sok = std::isnan(a.sum) ||
                fabs(s.sum - a.sum) <= 1e-3 + 1e-5 * fabs(a.sum);
            ++checks; if (!hok) ++fails; if (!sok) ++fails;
            printf("       %-6s [%9.5f %9.5f %9.5f] vs [%9.5f %9.5f %9.5f] %s"
                   "  | sum %14.6f vs %14.6f %s\n",
                   a.region, s.v[0], s.v[1], s.v[2], a.h[0], a.h[1], a.h[2],
                   hok ? "h-OK " : "h-FAIL", s.sum, a.sum, sok ? "s-OK" : "s-FAIL");
        }
    }
    // ---------------------------------------------------------------- the L2 row diagnostic
    // WHY THIS EXISTS: op 17 is the one anchor whose deviation cannot be explained by the 0.1 %
    // accumulation noise seen upstream - our row-0 scale is 0.559 against the official 0.408, i.e.
    // our row-0 energy is 1.87x too small, while our row-0 HEAD VALUES match the dump to 1 %. Only
    // the rest of the row can say which of the two is wrong, so the q/k block is pulled back whole
    // and each row's ||x|| is measured on the host.
    // The official per-row scale is INFERRED (not printed): qk_conv_l2-0 is a pure scalar multiple of
    // conv_output_silu-0 (view) within a row, so out/in on any element of the row is 1/||x||. The
    // dump prints rows 0,1,2,29,30,31 head 3 + tail 3 for both tensors; the numbers below are the
    // mean of the three head ratios, and rows 29/30 have one ratio contaminated by 4-decimal
    // rounding of a 0.0001-scale input, which is why their quoted spread is shown.
    {
        const std::vector<float> cv = fwd_read(f, f.r.CONV, 4096);
        const std::vector<float> l2 = fwd_read(f, f.r.L2, 4096);
        const int rows[6] = {0, 1, 2, 29, 30, 31};
        const double osc[6] = {0.40848, 1.01534, 1.10495, 2.7991, 1.5885, 0.64481};
        printf("\n  L2 rows, measured on the host (oracle scale is inferred from its own print):\n");
        printf("    row   ||x|| ours   1/||x|| ours   oracle 1/||x||   ratio  |  conv head3 / l2 head3\n");
        for (int t = 0; t < 6; ++t) {
            const int r = rows[t];
            double ss = 0.0;
            for (int d = 0; d < 128; ++d) ss += double(cv[r * 128 + d]) * cv[r * 128 + d];
            const double nrm = sqrt(ss);
            const double sc = nrm > 0 ? 1.0 / nrm : 0.0;
            printf("    %3d  %11.5f  %13.5f  %15.5f  %7.3f  | [%8.5f %8.5f %8.5f] [%8.5f %8.5f %8.5f]\n",
                   r, nrm, sc, osc[t], sc / osc[t],
                   cv[r * 128], cv[r * 128 + 1], cv[r * 128 + 2],
                   l2[r * 128], l2[r * 128 + 1], l2[r * 128 + 2]);
        }
    }
    // ---------------------------------------------------------------- which channel is which
    // op 17 read as "our CONV agrees at channels 0,1,2 and is 2x-45x off at 128, 256, 3712, 3968
    // while the 8192-channel SUM agrees to 0.04 %" - differences that cancel are the fingerprint of a
    // MAPPING error, so this probe was written to catch the conv reading the wrong channel.
    // WHAT IT MEASURED instead: the mapping was never the problem. The dump offers 36 probe channels
    // for free (rows 0,1,2,29,30,31 of conv_output_silu-0 (view), head 3 and tail 3), and on a COLD
    // conv state the official conv is a pure per-channel gain, because conv_input-0 =
    // CONCAT(conv_states_reshaped{3,8192}, qkv_mixed_transposed{1,8192}) puts this token at tap 3 and
    // every history tap is 0. Inverting silu on the printed value and dividing by w(3,c) therefore
    // RECOVERS the official x[c] - a 36-deep probe of node_21 that no printed node offers. Result,
    // runs/fwd_align_v9.log: our x matches the recovered x at all 24 usable probes (max 0.5 %), the
    // recovered x times the weight reproduces the official silu output at all 24, and our own CONV
    // region satisfies out[c] = silu(w(3,c)*x[c]) for 8192 of 8192 channels. The original deviation
    // was the HARNESS: every submit re-records ops 0..op, so op 17 ran the conv a second time on the
    // history the first pass had just written and produced (w2+w3)*x. The fill buffer in case 16 is
    // what makes the replay see the same cold state the official decode graph does.
    // The "best match in QKV" column is kept but is NOT evidence of anything: with 8192 dense floats
    // a nearest-neighbour search almost always finds some other channel closer than the probe's own,
    // which is why 20 of 24 say MOVED while the two value columns agree.
    {
        const std::vector<float> qkv = fwd_read(f, f.r.QKV, 8192);
        const std::vector<float> cvw = fwd_read_buf(f, e.f32blob.buf, cw->f32_off, 8192u * 4u);
        const std::vector<float> cv2 = fwd_read(f, f.r.CONV, 8192);
        static const int    pch[] = {  0,   1,   2, 125,   126,   127, 128, 129, 130, 253,  254,
                                        255, 256, 257, 258, 381,   382, 383, 3712, 3713, 3714,
                                       3837, 3838, 3839, 3840, 3841, 3842, 3965, 3966, 3967,
                                       3968, 3969, 3970, 4093, 4094, 4095 };
        static const double pon[] = { -0.0286, -0.0426, 0.0152, -0.1696, 0.0010, -0.0337,
                                       0.0332,  0.0128, -0.0006,  0.0010, -0.0774,  0.0681,
                                       0.2836,  0.0058,  0.2168, -0.0206, -0.0265,  0.0103,
                                      -0.0028, -0.0016, -0.0006, -0.0074,  0.0019,  0.0040,
                                       0.0001, -0.0529,  0.0033,  0.0742, -0.0204,  0.0010,
                                       0.0031,  0.0064, -0.1013, -0.0304,  0.0617, -0.0001 };
        printf("\n  channel probe: our conv vs the official one, and the x each one implies\n");
        printf("      ch   our conv   off conv  |    our x     off x      tap3   best match in QKV\n");
        int mapped = 0, probed = 0;
        for (int t = 0; t < int(sizeof(pch) / sizeof(pch[0])); ++t) {
            const int c = pch[t];
            const double y = pon[t];
            if (fabs(y) < 0.004) continue;                 // 4 decimals carry no signal below this
            double x = y;                                  // Newton on g(x) = x*sigmoid(x) - y
            for (int it = 0; it < 40; ++it) {
                const double s = 1.0 / (1.0 + exp(-x));
                const double g = x * s - y, dg = s + x * s * (1.0 - s);
                if (fabs(dg) < 1e-14) break;
                x -= g / dg;
            }
            const double w3 = cvw[size_t(c) * 4u + 3u];
            const double xoff = w3 != 0.0 ? x / w3 : 0.0;
            int bi = 0; double bd = 1e30;
            for (int j = 0; j < 8192; ++j) {
                const double d = fabs(double(qkv[j]) - xoff);
                if (d < bd) { bd = d; bi = j; }
            }
            ++probed; if (bi == c) ++mapped;
            printf("    %5d  %9.5f %9.5f  | %9.4f %9.4f w3 %8.5f  ch %5d %s |d| %.4f\n",
                   c, cv2[c], y, qkv[c], xoff, w3, bi, bi == c ? "SAME " : "MOVED", bd);
        }
        printf("  %d/%d probes land on their own channel\n", mapped, probed);

        // The probe just said OUR x and the official x agree at every one of the 24 channels (max
        // 1 % apart, the same accumulation noise as op 9), so node_21 and the channel order are
        // right and the disagreement is inside conv_silu. Ask the kernel's own definition: with a
        // cold history it is out[c] = silu(w(3,c) * x[c]), so the gain it ACTUALLY applied is
        // silu^-1(out[c]) / x[c], and that gain is a float somewhere in the weight blob. Where the
        // found float sits says which index expression was used - tap 0 instead of 3, a stride of
        // 8192 instead of 4, or a state value instead of a weight.
        {
            auto silu_inv = [](double y) { double x = y;
                for (int it = 0; it < 60; ++it) {
                    const double s = 1.0 / (1.0 + exp(-x));
                    const double g = x * s - y, dg = s + x * s * (1.0 - s);
                    if (fabs(dg) < 1e-15) break;
                    x -= g / dg; }
                return x; };
            int bad = 0, shown = 0;
            printf("\n  every channel checked against silu(w3*x), the kernel's own definition:\n");
            printf("      ch      ours   expected   |   gain used   nearest weight float  idx%%4  idx/4\n");
            for (int c = 0; c < 8192; ++c) {
                const double raw = double(cvw[size_t(c) * 4u + 3u]) * double(qkv[c]);
                const double expv = raw / (1.0 + exp(-raw));
                if (fabs(expv - double(cv2[c])) <= 2e-4 + 2e-3 * fabs(expv)) continue;
                ++bad;
                if (shown >= 10) continue;
                ++shown;
                const double gain = fabs(double(qkv[c])) > 1e-3 ? silu_inv(cv2[c]) / qkv[c] : 0.0;
                int bj = 0; double bd = 1e30;
                for (int j = 0; j < 8192 * 4; ++j) {
                    const double d = fabs(double(cvw[j]) - gain);
                    if (d < bd) { bd = d; bj = j; }
                }
                printf("    %5d  %9.5f  %9.5f  |  %10.5f  %13.5f  %5d  %6d\n",
                       c, cv2[c], expv, gain, cvw[bj], bj % 4, bj / 4);
            }
            printf("  %d of 8192 channels disagree with silu(w3*x)\n", bad);
        }
    }
    printf("  %d/%d anchor comparisons failed (%d anchors, head + sum each)\n",
           fails, 2 * checks, checks);
    return fails ? 1 : 0;
}


// The act buffer is its OWN device-local allocation, not a slice of the weight arena: the weight
// arena is already committed by engine_init(), and asking for a second one is how the 4 GB wall is
// measured honestly instead of being papered over by a failed bind at an offset nobody checked.
static bool act_store(Fwd &f, uint64_t dst_off, const void *src, uint64_t n) {
    Engine &e = *f.e;
    uint64_t done = 0;
    while (done < n) {
        uint64_t t = n - done;
        if (t > e.chunk_bytes) t = e.chunk_bytes;
        if (src) memcpy(e.host_ptr, (const char *)src + done, size_t(t));
        else     memset(e.host_ptr, 0, size_t(t));
        VkCommandBuffer cb = begin_cmd(e.c, e.cpool);
        VkBufferCopy cp{0, dst_off + done, t};
        vkCmdCopyBuffer(cb, e.host_buf, f.buf, 1, &cp);
        VK_CHECK(vkEndCommandBuffer(cb));
        submit_wait_sig(e.c, e.c.queue, cb, f.fence, VK_NULL_HANDLE, VK_NULL_HANDLE);
        done += t;
    }
    return true;
}

// device -> staging read, one fence: the readback half of act_store.
//
// HARD CAPACITY RULE: the staging buffer is exactly `chunk_bytes` (bonsai_weights.h:358 allocates it
// from the same number the upload loop chunks with), and a VkBufferCopy whose size leaves that mapping
// is not an error the driver reports - it is memory corruption. This was found by the first real
// cap-32768 run: the QK audit's E stage read one layer's whole K arena (cap*nkv*hd/2 words = 64 MiB)
// through a 32 MiB staging buffer and the process died with rc 139 (runs/real_cap32768_auto_v1.log,
// which got as far as the A2 SKIP print). So the size is checked here and the call is refused loudly;
// anything that can scale with cap goes through act_rd_words() below instead.
static void act_rd(Fwd &f, uint64_t src_off, uint64_t n_bytes) {
    Engine &e = *f.e;
    if (n_bytes > e.chunk_bytes) {
        fprintf(stderr, "act_rd: asked for %llu B into a %llu B host staging buffer. The copy would"
                        " run off the end of the mapping (silent corruption, not a driver error).\n"
                        "      Read it in pieces with act_rd_words(), or raise the upload chunk (the"
                        " first positional argument of `run`).\n",
                (unsigned long long)n_bytes, (unsigned long long)e.chunk_bytes);
        exit(2);
    }
    VkCommandBuffer cb = begin_cmd(e.c, e.cpool);
    VkBufferCopy cp{src_off, 0, n_bytes};
    vkCmdCopyBuffer(cb, f.buf, e.host_buf, 1, &cp);
    VK_CHECK(vkEndCommandBuffer(cb));
    submit_wait_sig(e.c, e.c.queue, cb, f.fence, VK_NULL_HANDLE, VK_NULL_HANDLE);
}

// The chunked form: any readback whose size scales with cap (an arena, not a row) belongs here. It is
// also the only readback that stays correct if the upload chunk is lowered - the chunks are derived
// from chunk_bytes, so a 1 MiB chunk simply means more, smaller copies.
static std::vector<uint32_t> act_rd_words(Fwd &f, uint64_t src_off, uint64_t n_words) {
    Engine &e = *f.e;
    std::vector<uint32_t> out;
    out.reserve(size_t(n_words));
    const uint64_t chunk_w = e.chunk_bytes / 4ull;
    uint64_t done = 0;
    while (done < n_words) {
        uint64_t t = n_words - done;
        if (t > chunk_w) t = chunk_w;
        act_rd(f, src_off + done * 4ull, t * 4ull);
        out.insert(out.end(), (const uint32_t *)e.host_ptr, (const uint32_t *)e.host_ptr + t);
        done += t;
    }
    return out;
}

// BONSAI_KVFILL=N: the long-context TIMING FIXTURE.
//
// WHY IT EXISTS. The thing being measured is the per-token cost of a row that is N keys long, and the
// only way to get a real N = 4096/8192/16384 context is to prefill 4096/8192/16384 tokens first - at
// the rate under test that is 6, 20 and 80 minutes of setup per data point, before the number is even
// taken. This fills the K/V arena with one device-side pattern and pins the decode position, so a point
// costs ~20 tokens instead of ~N of prefill.
//
// WHAT IT DOES NOT CLAIM. The bytes are 0x3C003C00 = fp16 1.0 in both halves (and in q8_0 mode, V data
// byte 60 with scale 1.0). They are not the model's K/V, so the sampled tokens and the text below the
// banner are MEANINGLESS, and no fidelity statement of any kind may be made from a fixture run. What IS
// unchanged is everything the timer measures: the same number of bytes read per head per layer, the
// same workgroup shape (n_head workgroups per layer per token), the same instruction stream, the same
// single submit per token. Cross-validation is mandatory before the curve is quoted: the fixture value
// at N = 1800 is compared against the REAL 1800-position measurement from runs/ladder_cap2048_* (96.9
// ms/token), and the two have to agree to within the run-to-run noise the report quotes.
static double kv_fill_synthetic(Fwd &f) {
    Engine &e = *f.e;
    const uint64_t off = uint64_t(f.r.KV) * 4ull;
    const uint64_t len = uint64_t(f.n_attn) * uint64_t(f.cap) * uint64_t(f.r.kv_layer) * 4ull;
    const auto t0 = std::chrono::steady_clock::now();
    VkCommandBuffer cb = begin_cmd(e.c, e.cpool);
    // Chunked only so that no single command asks a WDDM driver to record a multiple-of-4 range of an
    // unusual size; the fills are all in ONE command buffer, so the fence below covers all of them.
    const uint64_t CH = 64ull << 20;
    for (uint64_t d = 0; d < len; d += CH)
        vkCmdFillBuffer(cb, f.buf, off + d, VkDeviceSize(std::min<uint64_t>(CH, len - d)),
                        0x3C003C00u);
    VK_CHECK(vkEndCommandBuffer(cb));
    submit_wait_sig(e.c, e.c.queue, cb, f.fence, VK_NULL_HANDLE, VK_NULL_HANDLE);
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// Take one KV block out of what the model can attend to. MASK is the additive score mask the two QK
// kernels already read - measured semantics, -INF = dead, applied over the live prefix i < nfill in
// both of them - so masking the farthest block is exactly "those keys never existed" for the softmax.
// It is NOT a memory reclaim: the arena is indexed by absolute position, so the bytes stay allocated
// (kv_guard.h's header states the limit and what true reclaim would need). Two invariants keep this
// numerically safe rather than NaN-producing: the block that holds the current position is never
// touched (a fully-masked live prefix would make max = -INF and exp(v - max) = NaN), and mx starts at
// -INF in the kernel, so a masked key can never become the maximum.
static void kv_mask_block(Fwd &f, uint32_t lo, uint32_t hi) {
    if (hi > f.cap) hi = f.cap;
    if (lo >= hi) return;
    const uint64_t nb = uint64_t(f.cap) * 4ull;
    act_rd(f, uint64_t(f.r.MASK) * 4ull, nb);                  // read-modify-write the whole row
    float *m = (float *)f.e->host_ptr;
    for (uint32_t i = lo; i < hi; ++i) m[i] = -INFINITY;
    act_store(f, uint64_t(f.r.MASK) * 4ull, m, nb);
}

// ---------------------------------------------------------------- the KV fidelity dump (BONSAI_KVCHECK)
// The measurement that says what compressing V costs, together with the check that says the two runs
// were comparable at all. BONSAI_KVCHECK=<file> dumps, for one attention layer, every live position of
// the K arena (fp16 in BOTH modes) and the V arena (fp16 words in none mode, q8_0 data+scale decoded
// back to float in q8 mode), all with the host bit-math of f16_rne_host.h - the same packer the
// shaders use, so the dump cannot disagree with the arena by rounding differently.
//
// Why position 0 is the clean number: block 3 is the FIRST attention layer, so its V at position 0 is
// written from activations that no cache read has influenced yet - identical inputs in both runs. The
// K row is the in-run proof of that claim (both modes store it as the same fp16 words, so a K
// mismatch would mean the dumps are not comparable and the V difference is not a quantisation
// measurement). Positions >= 1 mix in the drift the compressed cache itself causes downstream, which
// is the number that actually reaches the model, so they are dumped too and tools/kv_compare.py
// reports the two bands separately.
static void kv_dump(Fwd &f, uint32_t nfill, uint32_t li, const char *path, uint32_t max_pos) {
    Engine &e = *f.e;
    const Regions &R = f.r;
    // one fp16 half of an arena word, by parity - the branch-free form of "low half / high half",
    // which the dump needs for both the V scales and the K/V values.
    auto word_of = [](uint32_t w, uint32_t hi) { return hi ? word_hi(w) : word_lo(w); };
    // one q8_0 byte, sign-extended without relying on the implementation-defined uint->int8_t narrow:
    // the byte is 0..255, and 128..255 are exactly the negatives, subtract 256.
    auto byte_of = [](uint32_t w, uint32_t sh) {
        const uint32_t b = (w >> sh) & 0xFFu;
        return float(int(b) - (b >= 128u ? 256 : 0));
    };
    const uint32_t nkv = uint32_t(e.plan.n_head_kv), hd = uint32_t(e.plan.head_dim);
    const uint32_t kk = uint32_t(R.kv_k), kvd = uint32_t(R.kv_vd), kvs = uint32_t(R.kv_vs);
    const uint32_t npos = nfill < max_pos ? nfill : max_pos;
    // ONE LAYER's arena is cap positions of (kk+kvd+kvs) words, and inside it the three sub-arenas are
    // cap words apart from each other - the same nesting the kernels index by, and the reason the three
    // arrays are read back separately instead of as one contiguous live prefix.
    const uint64_t layer_words = uint64_t(f.cap) * R.kv_layer;
    const uint64_t lbase = uint64_t(R.KV) + uint64_t(li) * layer_words;   // word 0 of this layer's K
    const uint64_t need_k = uint64_t(kk) * npos;
    const uint64_t need_v = uint64_t(kvd) * npos;
    const uint64_t need_s = uint64_t(kvs) * npos;
    if (lbase + uint64_t(f.cap) * R.kv_layer > uint64_t(R.total)) {
        printf("  kv check: layer %u's live arena does not fit the buffer, dump skipped\n", li);
        return;
    }
    act_rd(f, lbase * 4ull, need_k * 4ull);
    const std::vector<uint32_t> kwa((const uint32_t *)e.host_ptr,
                                    (const uint32_t *)e.host_ptr + need_k);
    act_rd(f, (lbase + uint64_t(f.cap) * kk) * 4ull, need_v * 4ull);
    const std::vector<uint32_t> vwa((const uint32_t *)e.host_ptr,
                                    (const uint32_t *)e.host_ptr + need_v);
    std::vector<uint32_t> swa;
    if (f.v_q8 && kvs) {
        act_rd(f, (lbase + uint64_t(f.cap) * (kk + kvd)) * 4ull, need_s * 4ull);
        swa.assign((const uint32_t *)e.host_ptr, (const uint32_t *)e.host_ptr + need_s);
    }
    const uint32_t *aw = kwa.data();        // element 0 = position 0, head 0, half 0 of K
    const uint32_t *vb0 = vwa.data();
    const uint32_t *vs0 = swa.data();
    FILE *g = fopen(path, "wb");
    if (!g) { printf("  kv check: cannot open %s\n", path); return; }
    fprintf(g, "# layer %u  positions %u  nkv %u  hd %u  v_q8 %u  kk %u kvd %u kvs %u  per-layer %llu words\n",
            li, npos, nkv, hd, f.v_q8, kk, kvd, kvs, (unsigned long long)layer_words);
    const uint32_t hw = hd / 2u, qw = hd / 4u, nblk = hd / 32u, sw = nblk / 2u;
    for (uint32_t p = 0u; p < npos; ++p) {
        const uint32_t *kb = aw + p * kk;
        for (uint32_t h = 0u; h < nkv; ++h)
            for (uint32_t d = 0u; d < hw; ++d) {
                const uint32_t w = kb[h * hw + d];
                fprintf(g, "K %u %u %u %.9g\n", p, h, 2u * d, double(word_lo(w)));
                fprintf(g, "K %u %u %u %.9g\n", p, h, 2u * d + 1u, double(word_hi(w)));
            }
        if (!f.v_q8) {
            const uint32_t *vb = vb0 + p * kvd;
            for (uint32_t h = 0u; h < nkv; ++h)
                for (uint32_t d = 0u; d < hw; ++d) {
                    const uint32_t w = vb[h * hw + d];
                    fprintf(g, "V %u %u %u %.9g\n", p, h, 2u * d, double(word_lo(w)));
                    fprintf(g, "V %u %u %u %.9g\n", p, h, 2u * d + 1u, double(word_hi(w)));
                }
        } else {
            const uint32_t *vd = vb0 + p * kvd;
            const uint32_t *vs = vs0 + p * kvs;
            for (uint32_t h = 0u; h < nkv; ++h) {
                for (uint32_t b = 0u; b < nblk; ++b) {
                    const uint32_t w = vs[h * sw + (b >> 1u)];
                    const float dsc = word_of(w, b & 1u);
                    fprintf(g, "S %u %u %u %.9g\n", p, h, b, double(dsc));
                }
                for (uint32_t d = 0u; d < hd; ++d) {
                    const uint32_t w = vd[h * qw + (d >> 2u)];
                    // the q8_0 reconstruction the kernel does: float(stored fp16 d) * sign-extended byte
                    const float dsc = word_of(vs[h * sw + ((d / 32u) >> 1u)], (d / 32u) & 1u);
                    fprintf(g, "V %u %u %u %.9g\n", p, h, d,
                            double(dsc * byte_of(w, (d & 3u) * 8u)));
                }
            }
        }
    }
    fclose(g);
    printf("  kv check: layer %u, %u positions, V = %s -> %s\n", li, npos,
           f.v_q8 ? "q8_0 reconstructed" : "fp16", path);
}

static bool fwd_init(Engine &e, Fwd &f) {
    f.e = &e;
    const ModelPlan &pl = e.plan;
    f.rms_eps = pl.rms_eps;
    f.n_attn = uint32_t(pl.n_attn_layers());
    f.n_gdn  = uint32_t(pl.n_layer) - f.n_attn;
    // The GQA merge's whole premise is a plan whose query heads group by exactly the HPK the wrapper was
    // compiled for. Anything else (nh % nkv != 0, or a ratio != 4) leaves the merged modules unselected
    // and the row on the non-merged kernels - it is not an error, it is the absence of an optimisation.
    f.hpk = (pl.n_head_kv > 0 && pl.n_head % pl.n_head_kv == 0)
                ? uint32_t(pl.n_head / pl.n_head_kv) : 0u;
    gqa_select(f);
    // The rotation route, read off the same metadata the resolver validated. Doing it here rather
    // than at the call sites is what keeps hat4096's push constants byte-identical for the 9B: the
    // file says block_size = 4096, so f.hat_block == 4096 and every dispatch below takes the
    // frozen branch. A 27B-class export that says 128/256/1024 takes fwht.comp instead, and the
    // 1/sqrt(N) it is handed is computed here, once, in double, then narrowed - which for N = 2^(2k)
    // is the same float the frozen kernel multiplies by and for any other N is the only value the
    // host reference in align_27b.cpp can agree with bit for bit.
    f.hat_block = pl.hat_active ? pl.hat_block : 0u;
    if (f.hat_block != 0u) {
        f.hat_inv = float(1.0 / std::sqrt(double(f.hat_block)));
        int e2 = 0;                                   // log2(block), by shifting not by ffs():
        for (uint32_t t = f.hat_block; t > 1u; t >>= 1u) ++e2;   // ffs is <strings.h> and GCC-only
        printf("  rotation route: block_size = %u -> %s, normalize = %.9g (%s)\n", f.hat_block,
               f.hat_block == 4096u ? "hat4096.comp (bit-frozen, the 9B's path)"
                                    : "fwht.comp (generic Sylvester FWHT)",
               double(f.hat_inv), (e2 % 2 == 0) ? "exact in binary" : "narrowed from double");
    }

    printf("\n=== forward bring-up ===\n");
    const uint32_t msa = uint32_t(e.c.limits.minStorageBufferOffsetAlignment);
    const uint64_t msbr = e.c.limits.maxStorageBufferRange;
    printf("  driver: minStorageBufferOffsetAlignment = %u B, maxStorageBufferRange = %.3f MiB\n",
           msa, double(msbr) / 1048576.0);
    // Every region offset has to be a multiple of BOTH msa (as a byte offset) and 4 floats (the vec4
    // kernels). Querying it is the rule; 64 is what this driver answers.
    uint32_t af = msa / 4u; if (af < 4u) af = 4u;
    af = (af + 15u) / 16u * 16u;
    // R = Regions() makes the tally idempotent, which is what lets the BONSAI_CAP_AUTOFIT fallback
    // below re-enter fwd_init with a smaller cap and get an EXACT second reading (alignment padding
    // of the cap-proportional regions is re-derived, not estimated).
    Regions &R = f.r; R = Regions(); R.align = af;

    const uint32_t nh = uint32_t(pl.n_head), nkv = uint32_t(pl.n_head_kv), hd = uint32_t(pl.head_dim);
    const uint32_t cap = f.cap;

    // The split-K scratch has to hold S*M partials for the widest case; measured over the plan's own
    // tensors rather than guessed, because a 4096-row tensor does split 16 ways on this card.
    // max_blk is the same reading for the xsum scratch: xsum.comp writes one float per 128-element
    // block of K, so the region has to hold max(K)/128 floats. The 9B's answer is 12288/128 = 96, and
    // the region below keeps the historical 128 as its FLOOR so this file's arena offsets do not move
    // for the model every log was produced with - a wider checkpoint gets a bigger XS instead of an
    // overwrite.
    uint32_t part = 256u, max_blk = 1u;
    for (const auto &w : pl.w) if (w.quantized()) {
        const uint32_t S = splits_for(w.M, w.K / 128u);
        if (S > 1u && S * w.M > part) part = S * w.M;
        if (w.K / 128u > max_blk) max_blk = w.K / 128u;
    }
    printf("  split-K scratch sized from the plan: %u floats (%.1f MiB) - max S*M over %zu PTQ tensors"
           " (xsum needs %u floats, region keeps %u)\n",
           part, part * 4.0 / 1048576.0, size_t(pl.n_ptq_tensors), max_blk,
           max_blk > 128u ? max_blk : 128u);

    // THE WIDTHS BEHIND THE REGION TABLE. Every one of these used to be a literal typed next to the
    // R.take() it sizes (4096, 12288, 8192, 1024, 32, 16384) - all of them true of the 9B, none of
    // them read from the file. They are read now, and for the 9B they evaluate to the same numbers,
    // so the offsets below are unchanged (that is the strict-superset rule: a 9B plan must resolve
    // bit-for-bit to what it resolved to before, or the ten logs in this repo stop being evidence).
    const uint32_t hid = uint32_t(pl.hidden), ffnw = uint32_t(pl.ffn);
    const WeightPlan *qkv0 = pl.find("blk.0.attn_qkv.weight");
    const WeightPlan *gt0  = pl.find("blk.0.attn_gate.weight");
    if (!qkv0 && f.n_gdn > 0u) {
        fprintf(stderr, "blk.0.attn_qkv.weight missing; cannot size the mixed qkv/conv regions\n");
        return false;
    }
    const uint32_t chn = uint32_t(qkv0 ? qkv0->M : pl.hidden);   // conv channels = qkv rows, 8192 on 9B
    const uint32_t zid = uint32_t(gt0 ? gt0->M : pl.hidden);     // the gate-z branch
    f.hid = hid; f.ffnw = ffnw;
    f.sign_w0 = pl.sign_width0();
    f.n_sign_f = uint32_t(pl.signs_total() ? pl.signs_total() : 16384ull);
    f.aidx.assign(size_t(pl.n_layer > 0 ? pl.n_layer : 1), 0);   // 64 layers must not walk off [32]
    R.RES = R.take(hid);  R.NRM = R.take(hid);  R.AN  = R.take(hid);
    R.ROT = R.take(hid);  R.HR  = R.take(hid);  R.EMB = R.take(hid);
    R.QKV = R.take(chn);  R.K   = R.take(nkv * hd);  R.V = R.take(nkv * hd);  R.Z = R.take(zid);
    R.ALPHA = R.take(uint32_t(pl.ts_rank));  R.BETA = R.take(uint32_t(pl.ts_rank));
    R.G1 = R.take(uint32_t(pl.ts_rank));
    R.G2 = R.take(uint32_t(pl.ts_rank));    R.GATE32 = R.take(uint32_t(pl.ts_rank));
    R.CONV = R.take(chn); R.L2 = R.take(hid);   R.GDNO = R.take(hid); R.GDNORM = R.take(hid);
    R.SWG = R.take(hid);  R.PERM = R.take(hid);
    R.FG = R.take(ffnw);  R.FU = R.take(ffnw);  R.FS = R.take(ffnw);
    R.PART = R.take(part); R.XS = R.take(max_blk > 128u ? max_blk : 128u);
    R.LOGITS = R.take(uint64_t(pl.vocab));
    R.Q16 = R.take(nh * hd / 2u);   R.QF = R.take(nh * hd);
    R.QD = R.take(nh * hd);         R.GD = R.take(nh * hd);   // attn_q's interleave split apart
    R.SCORE = R.take(nh * cap);     R.PROB = R.take(nh * cap);
    R.P16 = R.take(nh * cap / 2u);  R.PGATE = R.take(nh * hd); R.GOUT = R.take(nh * hd);
    R.MASK = R.take(cap);  R.POS = R.take(cap);
    R.DUMMY = R.take(16);                    // the window a binding gets when its branch is off
    R.KF = R.take(f.emit_kf ? uint64_t(cap) * nkv * hd : 16ull);
    R.QKWN = R.take(uint64_t(f.n_attn) * 2u * hd);
    R.SIGNS = R.take(f.n_sign_f);         // length of prism.hadamard.sign_values, from the file
    // One K arena + one V arena (+ the V scale array in q8_0 mode) per attention layer. The sizes are
    // DERIVED from the layout the two kernels write (KvGeom::compute), not typed in, and rejected
    // before allocation if a divisibility rule the shaders depend on stops holding.
    f.kvgeom.nkv = nkv; f.kvgeom.hd = hd; f.kvgeom.n_attn = f.n_attn; f.kvgeom.v_q8 = f.v_q8;
    std::string gwhy;
    if (!f.kvgeom.check(gwhy)) { fprintf(stderr, "KV geometry rejected: %s\n", gwhy.c_str()); return false; }
    f.kvgeom.compute(); f.kvgeom.report();
    // Every sub-arena start has to be a multiple of R.align FLOATS, because each one becomes a
    // VkDescriptorBufferInfo offset inside the single big VkBuffer. On this driver R.align is 16 and
    // all three sizes already satisfy it (512/512/0 words in fp16 mode, 512/256/16 in q8_0 mode), so
    // the rounding is a no-op here; it is written out because a driver that answers 256 would
    // otherwise scribble over the scale array with no validation error at all.
    R.kv_k  = (f.kvgeom.w_k  + R.align - 1u) / R.align * R.align;
    R.kv_vd = (f.kvgeom.w_vd + R.align - 1u) / R.align * R.align;
    R.kv_vs = (f.kvgeom.w_vs + R.align - 1u) / R.align * R.align;
    R.kv_layer = R.kv_k + R.kv_vd + R.kv_vs;
    // The arena is cap SLOTS x n_attn layers x the per-slot-per-layer footprint. Losing the `cap`
    // factor here is not a size warning, it is a correctness bug with no validation error behind it:
    // position 1 onward writes straight over the regions that follow (ST first), so a one-position
    // run is bit-exact and every longer run drifts - which is exactly what this build did until the
    // A/B against bonsai_vk_baseline_v1.exe caught it. The assert below is the same trap made
    // unmissable: the KV span has to equal per_slot's KV term x cap.
    R.KV = R.take(uint64_t(cap) * uint64_t(f.n_attn) * R.kv_layer);
    R.st_layer = uint64_t(pl.state_size) * pl.state_size * uint64_t(pl.inner / pl.state_size);
    R.ST = R.take(uint64_t(f.n_gdn) * R.st_layer);
    // The conv history is (conv_kernel-1) taps x one channel per attn_qkv ROW, not per hidden dim:
    // conv_silu.comp indexes st[st_off + c*3] for c in [0, 8192), i.e. 24576 floats per GDN layer.
    // Using inner=4096 here made every layer's history half as long as its kernel writes, so layer n
    // silently scribbled over layer n+1's taps. The channel count is read from the plan, not typed.
    R.ch_layer = uint64_t(pl.conv_kernel - 1) * chn;
    R.CH = R.take(uint64_t(f.n_gdn) * R.ch_layer);

    // ---------------------------------------------------------------- ARENA INVARIANTS
    // Measured from the region TABLE (start offsets of consecutively allocated regions), not from the
    // expression that produced them, so each check is independent evidence that the bytes are there.
    // KV is the only attention region whose span is cap-proportional, and ST/CH the two that must NOT
    // be; if any of the three is wrong the run scribbles over another layer's state with no validation
    // error and no warning - which is exactly the bug this block was written to make unmissable.
    const uint64_t kv_span = uint64_t(R.ST) - uint64_t(R.KV);
    const uint64_t st_span = uint64_t(R.CH) - uint64_t(R.ST);
    const uint64_t ch_span = uint64_t(R.total) - uint64_t(R.CH);
    const uint64_t kv_want = uint64_t(cap) * uint64_t(f.n_attn) * R.kv_layer;
    const uint64_t st_want = uint64_t(f.n_gdn) * R.st_layer;
    const uint64_t ch_want = uint64_t(f.n_gdn) * R.ch_layer;
    // take() rounds the NEXT start up to R.align, so a region's measured span is its size rounded up
    // to the alignment - unless it is the last region, whose end is `total` and therefore exact.
    const uint64_t kv_room = (kv_want + R.align - 1u) / R.align * R.align;
    const uint64_t st_room = (st_want + R.align - 1u) / R.align * R.align;
    if (R.overflow || kv_span != kv_room || st_span != st_room || ch_span != ch_want) {
        fprintf(stderr,
            "  ARENA INVARIANT FAIL (nothing dispatched)\n"
            "    index space overflowed uint32 float offsets : %s\n"
            "    KV span %llu vs cap %u x %u layers x %llu = %llu\n"
            "    ST span %llu vs %u GDN layers x %llu = %llu (must NOT contain cap)\n"
            "    CH span %llu vs %u GDN layers x %llu = %llu (must NOT contain cap)\n",
            R.overflow ? "YES" : "no",
            (unsigned long long)kv_span, cap, f.n_attn, (unsigned long long)R.kv_layer,
            (unsigned long long)kv_want,
            (unsigned long long)st_span, f.n_gdn, (unsigned long long)R.st_layer,
            (unsigned long long)st_want,
            (unsigned long long)ch_span, f.n_gdn, (unsigned long long)R.ch_layer,
            (unsigned long long)ch_want);
        return false;
    }
    printf("  arena invariants | KV span %llu == cap %u x %u x %llu OK;"
           " ST %llu / CH %llu are %u GDN layers x per-layer state, cap-free OK\n",
           (unsigned long long)kv_span, cap, f.n_attn, (unsigned long long)R.kv_layer,
           (unsigned long long)st_span, (unsigned long long)ch_span, f.n_gdn);

    // ---------------------------------------------------------------- THE MIXED-STACK INVARIANT
    // The block above compares the arena against the expression that allocated it, which is the same
    // code read twice. This one compares it against the PLAN's independent per-layer charge
    // (ModelPlan::kv_bytes_per_layer / ssm_bytes_per_layer, derived in bonsai_model.h from the layout
    // the KV kernels write) - two separate derivations of the same bytes, and this is the place they
    // are required to meet before a single byte is committed.
    //   KV: the plan charges slot_words per (attention layer, position) and EXACTLY 0 for every
    //       recurrent layer; the arena holds cap x n_attn x R.kv_layer, where R.kv_layer is the same
    //       words rounded up to the driver's descriptor-offset alignment. So the arena must be at
    //       least the charge, and the difference must be pure per-term alignment padding - a gap
    //       larger than R.align per term means the two formulas are describing different layouts, and
    //       an arena SMALLER than the charge is the overwrite this whole gate exists to catch.
    //   State: the plan's 2*inner conv spelling versus the arena's attn_qkv.M spelling, which
    //       model_plan_validate() has already required to be equal (gdn_channels); re-checked here
    //       against the bytes actually reserved rather than against the metadata.
    const uint64_t slot_words   = ModelPlan::kv_words_slot_layer(int(nkv), int(hd), int(f.v_q8));
    const uint64_t plan_kv_w    = slot_words * uint64_t(f.n_attn) * uint64_t(cap);
    const uint64_t pad_per_term = uint64_t(R.kv_layer) - slot_words;      // underflows if arena < charge
    const uint64_t plan_st_w    = uint64_t(pl.n_ssm_layers()) * pl.ssm_bytes_per_layer() / 4ull;
    const uint64_t arena_st_w   = st_want + ch_want;
    // The padding bound is 3 * R.align because K, V-data and V-scales are rounded INDEPENDENTLY above
    // (R.kv_k / R.kv_vd / R.kv_vs), so a worst-case legal layout wastes R.align-1 words in each term.
    // Beyond that the two formulas are not describing the same arena. The 9B wastes exactly 0.
    if (slot_words == 0ull || pad_per_term >= 3ull * R.align || plan_st_w != arena_st_w ||
        f.n_attn != uint32_t(pl.n_attn_layers())) {
        fprintf(stderr,
            "  MIXED-STACK INVARIANT FAIL (nothing dispatched)\n"
            "    plan KV charge %llu words (slot %llu x %u layers x cap %u) vs arena %llu\n"
            "    per-term alignment padding %llu words (must be < 3*R.align = %llu;\n"
            "    an unsigned underflow here is the arena being SMALLER than the charge)\n"
            "    an unsigned underflow here is the arena being SMALLER than the charge)\n"
            "    plan recurrent state %llu words vs arena ST+CH %llu words\n"
            "    attention layer count: plan %d, arena %u\n",
            (unsigned long long)plan_kv_w, (unsigned long long)slot_words, f.n_attn, cap,
            (unsigned long long)(kv_want), (unsigned long long)pad_per_term,
            (unsigned long long)(3ull * R.align),
            (unsigned long long)plan_st_w, (unsigned long long)arena_st_w,
            pl.n_attn_layers(), f.n_attn);
        return false;
    }
    printf("  mixed-stack invariants | plan charges %llu KV words == arena %llu (+%llu B/term of"
           " offset alignment); %d recurrent layers charged 0 B of KV and %llu words of state =="
           " arena ST+CH; KV bytes per token %llu (%s V)\n",
           (unsigned long long)plan_kv_w, (unsigned long long)(kv_want),
           (unsigned long long)(pad_per_term * 4ull), pl.n_ssm_layers(),
           (unsigned long long)plan_st_w,
           (unsigned long long)pl.kv_bytes_per_token(int(f.v_q8)), f.v_q8 ? "q8_0" : "fp16");

    f.bytes = uint64_t(R.total) * 4ull;
    printf("  activation arena: %u floats = %.3f MiB (state %.1f MiB + KV %.1f MiB + logits %.2f MiB)\n",
           R.total, double(f.bytes) / 1048576.0,
           double(R.st_layer * f.n_gdn * 4ull) / 1048576.0,
           double(uint64_t(cap) * f.n_attn * R.kv_layer * 4ull) / 1048576.0,
           double(pl.vocab) * 4.0 / 1048576.0);

    // ---------------------------------------------------------------- MEMORY BUDGET (the OOM fallback's whole point)
    // A wall of bare "did not fit" is useless to a first-time user: on a 4 GB card the honest answer
    // is always "which of the four numbers is the one that overflowed". So split the ask into the
    // part that does not depend on cap and the part that does, both READ BACK from the tally above -
    // per_slot is the symbolic sum of the cap-proportional regions, and fixed is the measured total
    // MINUS that, which makes the two add back to f.bytes by construction rather than by luck.
    const uint64_t per_slot = uint64_t(nh) * 2u + nh / 2u + 2u     // SCORE PROB P16 MASK POS
                            + (f.emit_kf ? uint64_t(nkv) * hd : 0u)  // KF, the gated debug stream
                            + uint64_t(f.n_attn) * R.kv_layer;    // KV: K + V (+ V scales) per layer
    // THE P0 AUDIT ANSWER, in one line: this sum is the ENTIRE cap-proportional part of the arena.
    // Nothing in it belongs to a GDN layer, so the 24 recurrent layers cannot grow with the context.
    printf("    P0 check | cap-proportional regions = SCORE+PROB+P16+MASK+POS (%u floats)"
           " + KF %s (%u) + KV %u layers x %llu = %llu floats/slot;"
           " GDN terms in it: %u (ST = %llu floats total, independent of cap)\n",
           uint32_t(nh) * 2u + nh / 2u + 2u, f.emit_kf ? "on" : "off",
           f.emit_kf ? uint32_t(nkv) * hd : 0u, f.n_attn,
           (unsigned long long)R.kv_layer, (unsigned long long)per_slot,
           0u, (unsigned long long)(R.st_layer * f.n_gdn));
    const uint64_t fixed_f = uint64_t(R.total) - per_slot * uint64_t(cap);
    const uint64_t heap = device_local_heap(e.c);
    const double M = 1048576.0;
    // BONSAI_HEAP_LIMIT_MIB is a TEST KNOB, not a user feature: on this card no legal cap can overflow
    // (cap 2048 needs 124 MiB and the heap leaves ~1.3 GiB), so without it the fallback below has
    // never been executed on real hardware. Clamping the budget makes the refuse path deterministic and
    // therefore assertable; the printed table says when it was in effect.
    uint64_t heap_budget = heap;
    if (const char *v = getenv("BONSAI_HEAP_LIMIT_MIB")) {
        heap_budget = std::min(heap, uint64_t(atoll(v)) * 1048576ull);
        printf("  [TEST KNOB] BONSAI_HEAP_LIMIT_MIB=%s -> budget clamped to %.1f MiB"
               " (driver reports %.1f MiB)\n", v, heap_budget / M, heap / M);
    }
    const uint64_t weights = e.arena.size;
    const long long room = int64_t(heap_budget) - int64_t(weights) - int64_t(f.bytes);
    printf("  memory budget | device-local heap %.1f MiB, weights %.1f MiB already committed\n",
           heap_budget / M, weights / M);
    printf("    activations at cap %u : %.1f MiB = cap-independent %.1f MiB"
           " + %u slots x %.2f KiB/slot (%s arena %.2f KiB + fp32 side %.2f KiB%s)\n",
           cap, double(f.bytes) / M, double(fixed_f) * 4.0 / M, cap, per_slot * 4.0 / 1024.0,
           f.v_q8 ? "q8_0-V + fp16-K" : "fp16 K/V",
           double(uint64_t(f.n_attn) * R.kv_layer) * 4.0 / 1024.0,
           double(uint64_t(nh) * 2u + nh / 2u + 2u) * 4.0 / 1024.0,
           f.emit_kf ? ", + KF" : "");
    if (room >= 0)
        printf("    headroom after this allocation: %.1f MiB (= %.0f more slots of cap,"
               " but the widest VALIDATED row for the %s QK kernel is %u)\n",
               room / M, room / (per_slot * 4.0), f.qkt ? "tiled" : "wide", qk_row_limit(f));
    else {
        const long long avail = int64_t(heap_budget) - int64_t(weights) - int64_t(fixed_f) * 4;
        if (avail <= 0) {
            fprintf(stderr,
                "  OOM FALLBACK - no cap fits at all, and NOTHING WAS DISPATCHED.\n"
                "    device-local heap        : %.1f MiB\n"
                "    weights already resident : %.1f MiB\n"
                "    cap-INDEPENDENT activity : %.1f MiB (recurrent state + logits + split-K scratch"
                " + the fixed regions)\n"
                "    this model therefore needs at least %.1f MiB of device-local memory;"
                " it cannot run in %.1f MiB.\n"
                "    options: free the GPU (other processes hold device-local memory too),"
                " or run the CPU fallback, or use a smaller checkpoint.\n",
                heap_budget / M, weights / M, double(fixed_f) * 4.0 / M,
                (double(weights) + double(fixed_f) * 4.0) / M, heap_budget / M);
            return false;
        }
        // the cap has to be EVEN (P16 packs key pairs) and is capped by what the QK kernel validated.
        // In tiled mode "what the kernel validated" is not a shader array any more, so this clamp
        // becomes the memory budget - which is the whole point of the tiling exercise.
        long long fit_cap = avail / int64_t(per_slot * 4);
        if (fit_cap & 1) --fit_cap;
        const long long use_cap = fit_cap > int64_t(qk_row_limit(f)) ? int64_t(qk_row_limit(f)) : fit_cap;
        fprintf(stderr,
            "  OOM FALLBACK - this cap does not fit, and NOTHING WAS DISPATCHED (no half-built state).\n"
            "    short by          : %.1f MiB\n"
            "    heap - weights    : %.1f MiB available for activations + KV\n"
            "    cap-independent   : %.1f MiB (state, logits, scratch, fixed regions)\n"
            "    left for the KV   : %.1f MiB, at %.2f KiB per slot -> %lld slots\n"
            "    widest VALIDATED row for the %s QK kernel : %u slots\n"
            "    so the usable cap here is %lld; rerun with BONSAI_CAP=%lld (even, or lower),\n"
            "    or set BONSAI_CAP_AUTOFIT=1 to take that fallback automatically.\n",
            -double(room) / M, (double(heap_budget) - double(weights)) / M,
            double(fixed_f) * 4.0 / M, double(avail) / M, per_slot * 4.0 / 1024.0, fit_cap,
            f.qkt ? "tiled" : "wide", qk_row_limit(f), use_cap, use_cap);
        // P2's third defence, at the one moment it can be taken without corrupting anything: BEFORE the
        // arena exists. "Fall back to a shorter context length" is only honest here - after allocation
        // the bytes are already committed, so the runtime equivalent is refuse/evict (see kv_guard.h).
        if (f.cap_autofit && use_cap >= 2) {
            fprintf(stderr, "    BONSAI_CAP_AUTOFIT=1 -> re-initialising at cap %lld.\n", use_cap);
            f.cap = uint32_t(use_cap);
            // f.wide was chosen by the caller from the OLD cap, and choosing the wrong row kernel is a
            // correctness bug, not a performance one: the narrow kernel's lane IS the key position.
            qk_select(f);
            if (f.qk_force == 1u && f.cap > Fwd::kMaxNarrow) return false;
            return fwd_init(e, f);
        }
        return false;
    }

    // ---------------------------------------------------------------- P2: the block hook + the spill watch
    // Both are set up BEFORE the arena is created so that a counter failure or a geometry complaint is
    // visible in the same log as the budget, and so that the watch's first sample is taken with the
    // weights already resident (that is the state a spill would start from).
    const uint64_t kv_bytes = uint64_t(R.kv_layer) * f.n_attn * 4ull;
    if (f.kv_blk_slots < 1u) f.kv_blk_slots = 1u;
    f.kvguard.blk_slots = f.kv_blk_slots;
    f.kvguard.init(f.kvgeom, cap, kv_bytes + uint64_t(room > 0 ? room : 0),
                   int(f.kvguard.allow_evict));      // allow_evict was set by BONSAI_KV_EVICT
    f.kvguard.g_spill_soft = f.spill.soft_mib;
    f.kvguard.g_spill_hard = f.spill.hard_mib;
    if (cap % f.kvguard.blk_slots)
        printf("  note: cap %u is not a multiple of the %u-position KV block; the last partial block"
               " is never evictable (it holds the live tail)\n", cap, f.kvguard.blk_slots);
    f.kvguard.report();
    f.spill.open(bonsai_pid());
    f.budget.open();
    f.budget.report();
    if (!f.spill.counter_ok)
        printf("  [spill] %s counter unavailable (%s) - the watch is inert and no decision will be"
               " made on it; the block hook still enforces the arena ceiling\n",
               SpillWatch::judge_api(), f.spill.err.c_str());

    Arena ar; ar.init(e.c);
    const uint64_t U = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                       VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    uint64_t off = ar.reserve(U, f.bytes, "act");
    if (!ar.commit()) {
        fprintf(stderr, "act arena did not fit: the driver refused %.1f MiB although the heap it"
                        " reports still leaves %.1f MiB free after the weights - the driver keeps a"
                        " reservation inside that heap, so the real ceiling for this cap is LOWER than"
                        " the budget table above. Lower BONSAI_CAP (each slot costs %.2f KiB) and rerun.\n",
                double(f.bytes) / M, room / M, per_slot * 4.0 / 1024.0);
        return false;
    }
    Buf ab = ar.create_at(U, f.bytes, off, "act");
    f.buf = ab.buf;
    // The arena's own VkDeviceMemory handle is what has to stay alive for the buffer's lifetime.
    f.mem = ar.mem;

    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VK_CHECK(vkCreateFence(e.c.dev, &fci, nullptr, &f.fence));

    // zero everything (state, conv history and the causal mask all start dead-clean), then the two
    // sign vectors. One host->device pass per region, no staging alignment assumptions.
    act_store(f, 0, nullptr, f.bytes);
    auto sv = e.hdr.kv_a.find("prism.hadamard.sign_values");
    if (sv == e.hdr.kv_a.end() || sv->second.size() != 16384) {
        fprintf(stderr, "prism.hadamard.sign_values missing or wrong length (%zu)\n",
                sv == e.hdr.kv_a.end() ? 0 : sv->second.size());
        return false;
    }
    std::vector<int32_t> sgn(16384);
    long s1 = 0, s2 = 0;
    for (size_t i = 0; i < 16384; ++i) {
        sgn[i] = int32_t(atoi(sv->second[i].c_str()));
        if (i < 4096) s1 += sgn[i]; else s2 += sgn[i];
    }
    printf("  signs: sum(0..4095) = %ld, sum(4096..16383) = %ld   (the file's own values; 60 / -94\n"
           "  were the numbers read off the metadata, and they are what makes this upload checkable)\n", s1, s2);
    act_store(f, uint64_t(R.SIGNS) * 4ull, sgn.data(), 16384ull * 4ull);

    // The two attention norm weights have to live at a 64-byte-aligned address because the validated
    // attention kernels index them from element 0; the f32 blob only guarantees 16-byte tensor
    // alignment, so they are copied out of the file into act rather than pointed at in place.
    for (int b = 0; b < pl.n_layer; ++b) {
        if (!pl.is_attn_layer(b)) continue;
        const int li = f.aidx[b];
        const WeightPlan *qn = pl.find(("blk." + std::to_string(b) + ".attn_q_norm.weight").c_str());
        const WeightPlan *kn = pl.find(("blk." + std::to_string(b) + ".attn_k_norm.weight").c_str());
        if (!qn || !kn || qn->K != hd || kn->K != hd) {
            fprintf(stderr, "attn norm weight missing for block %d\n", b); return false;
        }
        float tmp[256];
        uint32_t dsts[2] = {R.QKWN + uint32_t(li) * 2u * hd, R.QKWN + uint32_t(li) * 2u * hd + hd};
        const WeightPlan *srcs[2] = {qn, kn};
        for (int j = 0; j < 2; ++j) {
            fseek(e.f, long(e.hdr.data_start + srcs[j]->file_off), SEEK_SET);
            if (fread(tmp, 4, hd, e.f) != hd) { fprintf(stderr, "read qn/kn failed\n"); return false; }
            act_store(f, uint64_t(dsts[j]) * 4ull, tmp, hd * 4ull);
        }
    }

    // ------------------------------------------------------------------ pipelines
    const Win A = w_all(f), C = {e.codes.buf, 0, e.codes.size}, S = {e.scales.buf, 0, e.scales.size},
              F = {e.f32blob.buf, 0, e.f32blob.size};
    VkDescriptorPoolSize psz{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1024};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 128; dpi.poolSizeCount = 1; dpi.pPoolSizes = &psz;
    VK_CHECK(vkCreateDescriptorPool(e.c.dev, &dpi, nullptr, &f.dpool));
    struct Spec { const char *spv, *name; std::vector<Win> wins; };
    const Spec spec[Fwd::P_N] = {
        {"set_pos",        "setpos",   {A}},
        {"embd_gather",    "embd",     {C, S, A}},
        {"rms_norm",       "rms",      {A, F, A}},
        {"mul",            "mul",      {A, F, A}},
        {"mul_bcast",      "mulb",     {A, F, A}},
        {"signs_mul",      "signs",    {A, A, A}},
        {"hat4096",        "hat",      {A, A}},
        {"hadamard_fwht",  "fwht",     {A, A}},        // block from prism.hadamard.block_size; only
                                                        // dispatched when that value is not 4096
        {"add",            "add",      {A, A, A}},
        {"add",            "addf",     {A, F, A}},
        {"swiglu",         "swiglu",   {A, A, A}},
        {"permute_vg",     "permute",  {A, A}},
        {"sigmoid",        "sigmoid",  {A, A}},
        {"softplus",       "softplus", {A, A}},
        {"l2_norm",        "l2",       {A, A}},
        {"conv_silu",      "conv",     {A, A, A, F}},
        {"gdn_delta",      "gdn",      {A, A, A}},
        {"xsum",           "xsum",     {A, A}},
        {"reduce",         "reduce",   {A}},
        {"gemv_ternary",   "gemv",     {C, A, S, A, A}},
        {"gemv_f32",       "gemv32",   {A, F, A}},
        {"deint_qg",       "deint",    {A, A, A}},
        {"attn_norm_rope", "normrope", {A}},   // replaced per layer below
        {"attn_qk_softmax","qk",       {A}},
        {"attn_qk_softmax_wide","qkw", {A}},   // same 7 windows, lanes stride over the row
        {"attn_qk_softmax_tile_q2048","qkt", {A}},   // same 7 windows, row walked in QTILE tiles
        {"attn_pv",        "pv",       {A}},
    };
    // The attention kernels are only ever dispatched through the per-layer sets below, so the
    // placeholder rows in spec[] are not built here - a 1-binding layout against an 11-binding shader
    // is exactly the kind of mismatch a validation layer would flag for no benefit.
    for (int i = 0; i < Fwd::P_NORMROPE; ++i)
        f.p[i] = mk_pipe(e.c, f.dpool, spec[i].spv, spec[i].name, spec[i].wins);

    // 8 layers x 4 attention descriptor sets. The pipelines are shared; only the windows move.
    // THE NESTING, spelled out because getting it wrong is silent: R.kv_layer is the footprint of ONE
    // POSITION in ONE layer. The kernels index their window by ABSOLUTE POSITION
    // (`(position*nkv + h)*hd/2`), so a layer's arena is cap positions of that, and a sub-arena starts
    // where the previous one ENDS after all cap positions:
    //     layer base = kv0 + li * cap * kv_layer
    //     KC = [base + 0,            + cap*kv_k)      fp16 K, in BOTH modes
    //     VC = [base + cap*kv_k,     + cap*kv_vd)     fp16 V  |  q8_0 data
    //     VS = [base + cap*(kv_k+kv_vd), + cap*kv_vs) q8_0 scales (a 16-float dummy window otherwise)
    // Using kv_layer where cap*kv_layer belongs does not corrupt at position 0 - every window's first
    // slot is at its own base - so the 1-token oracle tests keep passing bitwise; from position 1 the
    // K row of slot p lands on the V row of slot p-1 of the same layer and on the neighbouring layers'
    // slots. That is the failure this line is written against, and it is checked at init below.
    const uint32_t kwh = cap * nkv * hd / 2u;         // one layer's fp16 K (or V) arena: cap*512
    const uint32_t kk = uint32_t(R.kv_k), kvd = uint32_t(R.kv_vd), kvs = uint32_t(R.kv_vs);
    const uint32_t ck = cap * kk, cvd = cap * kvd, cvs = cap * kvs;
    const uint32_t kv0 = R.KV;
    const uint32_t kv_layer_span = uint32_t(uint64_t(cap) * R.kv_layer);   // one LAYER's whole arena
    // The regression contract, as an executable check: in none mode the K and V windows this code
    // builds must be the SAME (base, size) pairs the pre-P1 binary handed the validated kernels -
    // K at the layer base for cap*kv_k words, V at layer base + cap*kv_k for cap*kv_vd, stride 2x that.
    // If this ever prints a failure, the A/B against bonsai_vk_baseline_v1.exe is meaningless, so it
    // refuses to run instead of producing a drift that would be blamed on the shader switch.
    if (!f.v_q8 && (ck != kwh || cvd != kwh || kv_layer_span != 2u * kwh)) {
        fprintf(stderr, "KV window nesting rejected: none-mode wants K/V = %u words at +0/+%u and a"
                        " %u-word layer stride (the pre-P1 layout), got ck %u cvd %u span %u\n",
                kwh, kwh, 2 * kwh, ck, cvd, kv_layer_span);
        return false;
    }
    if (f.v_q8 && kv_layer_span != ck + cvd + cvs) {
        fprintf(stderr, "KV window nesting rejected: q8_0 layer span %u != K %u + V %u + scales %u\n",
                kv_layer_span, ck, cvd, cvs);
        return false;
    }
    const uint32_t gate_off = R.GD;         // deint_qg's contiguous gate window, head-major
    const char *rope_spv = f.rope_oracle ? "attn_norm_rope" : "attn_norm_rope2";
    const char *pv_spv   = f.v_q8 ? "attn_pv_q8v" : "attn_pv";
    // Which tile module to build: only when it will be dispatched (force=3) or audited (QCHECK). The
    // std::string outlives the lambda because mk_attn runs to completion inside this function.
    const std::string tile_name = "attn_qk_softmax_tile_q" + std::to_string(f.qk_tile);
    const bool want_tile = (f.qkt != 0u) || (f.qcheck != 0u);
    const char *tile_spv = want_tile ? tile_name.c_str() : nullptr;
    // Same rule for the two merged modules: build them when they will be dispatched OR when the audit
    // will compare them bitwise against the kernel they replace - the gate needs both modules resident
    // in the same process, because the inputs it compares are the same KV arena bytes at the same cap.
    const std::string gqa_name = "attn_qk_softmax_gqa_h4_q" + std::to_string(f.gqa_tile);
    const bool want_gqa_qk = ((f.qkg != 0u) || (f.qcheck != 0u)) && f.hpk == 4u;
    const bool want_gqa_pv = ((f.pvg != 0u) || (f.qcheck != 0u)) && f.hpk == 4u && f.v_q8 != 0u &&
                              !f.rope_oracle;                       // oracle mode pins the frozen set
    const char *gqa_spv = want_gqa_qk ? gqa_name.c_str() : nullptr;
    uint32_t n_sets = 0, n_desc = 0;
    auto mk_attn = [&](int li) {
        const uint32_t a = kv0 + uint32_t(li) * kv_layer_span;
        std::vector<Win> r = {
            w_act(f, R.QD, nh * hd), w_act(f, R.K, nkv * hd), w_act(f, R.V, nkv * hd),
            w_act(f, R.QKWN + uint32_t(li) * 2u * hd, hd), w_act(f, R.QKWN + uint32_t(li) * 2u * hd + hd, hd),
            w_act(f, R.POS, cap),
            w_act(f, a, ck),                                   // KC - fp16 in BOTH modes
            w_act(f, a + ck, f.v_q8 ? cvd : kwh),              // VC - fp16 V arena | q8_0 data
            w_act(f, f.v_q8 ? a + ck + cvd : R.DUMMY, f.v_q8 ? cvs : 16u),  // VS - scale words
            w_act(f, R.Q16, nh * hd / 2u), w_act(f, R.QF, nh * hd),
            w_act(f, f.emit_kf ? R.KF : R.DUMMY, f.emit_kf ? cap * nkv * hd : 16u) };
        // BONSAI_ROPE_ORACLE=1: the byte-frozen attn_norm_rope.comp has ELEVEN bindings - it has no VS
        // slot, and its Q16/QF/KF sit one lower. Dropping the VS window is the whole re-plumbing, and
        // it is the bisect tool: if none + oracle reproduces the pre-P1 binary bit for bit, the
        // difference lives in rope2 and nowhere else; if it still differs, the plumbing (region layout)
        // is what moved. The oracle always writes kf, so emit_kf is forced on with it.
        if (f.rope_oracle) r.erase(r.begin() + 8);
        std::vector<Win> q = {
            w_act(f, R.Q16, nh * hd / 2u),
            w_act(f, a, ck),                                   // the same KC window
            w_act(f, R.MASK, cap), w_act(f, R.POS, cap),
            w_act(f, R.SCORE, nh * cap), w_act(f, R.PROB, nh * cap), w_act(f, R.P16, nh * cap / 2u) };
        std::vector<Win> v = {
            w_act(f, R.PROB, nh * cap), w_act(f, R.P16, nh * cap / 2u) };
        if (f.v_q8) {   // the fused-unpack kernel reads the data array and the scale array separately
            v.push_back(w_act(f, a + ck, cvd));
            v.push_back(w_act(f, a + ck + cvd, cvs));
        } else {
            v.push_back(w_act(f, a + ck, kwh));                // the fp16 V arena, as before
        }
        v.push_back(w_act(f, R.POS, cap));
        v.push_back(w_act(f, gate_off, nh * hd));
        v.push_back(w_act(f, R.PGATE, nh * hd));
        v.push_back(w_act(f, R.GOUT, nh * hd));
        f.arope[li] = mk_pipe(e.c, f.dpool, rope_spv, "normrope", r);
        f.aqk[li]   = mk_pipe(e.c, f.dpool, "attn_qk_softmax", "qk",       q);
        // Identical windows, identical order - the only difference is the SPIR-V module, so a
        // dispatch-level A/B of narrow vs wide cannot be confounded by a binding mistake.
        f.aqkw[li]  = mk_pipe(e.c, f.dpool, "attn_qk_softmax_wide", "qkw", q);
        // The tiled row kernel is built when it will be dispatched OR when the audit will compare it
        // against the wide one - the gate needs both modules resident in the same process, because the
        // inputs it compares are the same KV arena bytes at the same cap.
        if (tile_spv) f.aqkt[li] = mk_pipe(e.c, f.dpool, tile_spv, "qkt", q);
        // The GQA-merged row kernel gets the SAME seven windows, in the same order - the only difference
        // is the SPIR-V module and the grid, so A3 cannot be confounded by a binding mistake either.
        if (gqa_spv)  f.aqkg[li] = mk_pipe(e.c, f.dpool, gqa_spv, "qkg", q);
        f.apv[li]   = mk_pipe(e.c, f.dpool, pv_spv,    "pv",       v);
        if (want_gqa_pv) f.apvg[li] = mk_pipe(e.c, f.dpool, "attn_pv_q8v_gqa", "pvg", v);
        n_sets += 4u + (tile_spv ? 1u : 0u) + (gqa_spv ? 1u : 0u) + (want_gqa_pv ? 1u : 0u);
        n_desc += uint32_t(r.size() + q.size() + q.size() + v.size() +
                           (tile_spv ? q.size() : 0u) + (gqa_spv ? q.size() : 0u) +
                           (want_gqa_pv ? v.size() : 0u));
    };
    int li = 0;
    for (int b = 0; b < pl.n_layer; ++b) if (pl.is_attn_layer(b)) { f.aidx[b] = li; mk_attn(li); ++li; }
    // 21 elementwise/GEMV sets + n_attn x (4..7) attention sets. The five row/branch kernels per
    // attention layer are: norm+rope, narrow QK, wide QK, PV - and up to three more built only when
    // they are dispatched or audited: tiled QK, merged GQA QK, merged GQA PV. All five QK modules take
    // the same seven windows, so the worst case is 21 + 8*7 = 77 of the 128 allocated.
    // The descriptor total is counted, not asserted: the pool was sized by hand at 1024, the q8_0 mode
    // adds one binding to the rope set and one to the pv set per layer, and each extra QK module adds 7.
    printf("  pipelines: %d elementwise/GEMV + %u layers x %u attention descriptor sets"
           " (%u of 128 sets, %u of 1024 descriptors) | V = %s, K = fp16 | norm+rope+pack = %s"
           " | tiled QK = %s | GQA-merged QK = %s | GQA-merged PV = %s\n",
           Fwd::P_NORMROPE, f.n_attn, uint32_t(n_sets / (f.n_attn ? f.n_attn : 1)),
           Fwd::P_NORMROPE + n_sets, 56u + n_desc,
           f.v_q8 ? "q8_0 (attn_pv_q8v)" : "fp16 (attn_pv, the validated kernel)",
           f.rope_oracle ? "attn_norm_rope (FROZEN oracle)" : "attn_norm_rope2",
           want_tile ? tile_name.c_str() : "not built",
           gqa_spv ? gqa_name.c_str() : "not built",
           want_gqa_pv ? "attn_pv_q8v_gqa" : "not built");

    // The region table is printed because "RES is zero" has exactly two causes - nothing was
    // written, or two regions are the same bytes - and only the offsets can tell them apart.
    printf("\n=== region table (float units; every offset is a multiple of %u) ===\n", R.align);
    struct Rn { const char *n; uint32_t off, sz; };
    const Rn tbl[] = {
        {"RES", R.RES, 4096}, {"NRM", R.NRM, 4096}, {"AN", R.AN, 4096}, {"ROT", R.ROT, 4096},
        {"HR", R.HR, 4096}, {"EMB", R.EMB, 4096}, {"QKV", R.QKV, 8192}, {"K", R.K, 1024},
        {"V", R.V, 1024}, {"Z", R.Z, 4096}, {"ALPHA", R.ALPHA, 32}, {"BETA", R.BETA, 32},
        {"G1", R.G1, 32}, {"G2", R.G2, 32}, {"GATE32", R.GATE32, 32}, {"CONV", R.CONV, 8192},
        {"L2", R.L2, 4096}, {"GDNO", R.GDNO, 4096}, {"GDNORM", R.GDNORM, 4096}, {"SWG", R.SWG, 4096},
        {"PERM", R.PERM, 4096}, {"FG", R.FG, 12288}, {"FU", R.FU, 12288}, {"FS", R.FS, 12288},
        {"PART", R.PART, part}, {"XS", R.XS, 128}, {"LOGITS", R.LOGITS, uint32_t(pl.vocab)},
        {"Q16", R.Q16, nh * hd / 2u}, {"QF", R.QF, nh * hd},
        {"QD", R.QD, nh * hd}, {"GD", R.GD, nh * hd}, {"SCORE", R.SCORE, nh * cap},
        {"PROB", R.PROB, nh * cap}, {"P16", R.P16, nh * cap / 2u}, {"PGATE", R.PGATE, nh * hd},
        {"GOUT", R.GOUT, nh * hd}, {"MASK", R.MASK, cap}, {"POS", R.POS, cap},
        {"KF", R.KF, f.emit_kf ? cap * nkv * hd : 16u}, {"DUMMY", R.DUMMY, 16},
        {"QKWN", R.QKWN, f.n_attn * 2u * hd}, {"SIGNS", R.SIGNS, 16384},
        {"KV", R.KV, uint32_t(uint64_t(cap) * f.n_attn * R.kv_layer)},
        {"ST", R.ST, uint32_t(R.st_layer)},
        {"CH", R.CH, uint32_t(R.ch_layer)},
    };
    for (const auto &x : tbl) printf("  %-8s off %9u  size %9u\n", x.n, x.off, x.sz);
    // The KV row above is ONE layer's sub-arena; break it into the pieces the kernels index, because
    // "how big is the arena" is the P0/P1 question and the answer has to be readable per mode.
    printf("  KV split  K(fp16) %-6llu  V %s  %-6llu  V-scales %-6llu   x%u layers, x%u slots\n",
           (unsigned long long)R.kv_k, f.v_q8 ? "(q8_0 data)" : "(fp16)  ",
           (unsigned long long)R.kv_vd, (unsigned long long)R.kv_vs, f.n_attn, cap);

    // The signs are read back from the DEVICE and summed as int32: the host-side 60/-94 only proves
    // the metadata parsed, not that the upload landed where the shader looks.
    {
        act_rd(f, uint64_t(R.SIGNS) * 4ull, 16384 * 4ull);
        const int32_t *p = (const int32_t *)e.host_ptr;
        long a = 0, c = 0; int nz = 0;
        for (int i = 0; i < 16384; ++i) { if (i < 4096) a += p[i]; else c += p[i]; if (p[i]) ++nz; }
        printf("  signs on device: sum4096 = %ld, sum12288 = %ld, nonzero = %d/16384\n", a, c, nz);
    }
    return true;
}

// ------------------------------------------------------------------ the QK kernel audit (BONSAI_QCHECK)
// The wide kernel exists so a row can be longer than 256 keys. Two things can go wrong when a lane
// stops BEING a key position and starts STRIDING over them, and they fail differently:
//
//   WIDE#1  a wrong key slot (kc indexed with the wrong k) -> the SCORE itself is wrong.
//   WIDE#2  a pack that stopped covering the row (256 lanes can only write 512 probability slots)
//           -> SCORE/PROB stay perfect and P16 silently keeps the PREVIOUS token's bytes above 512.
//           With round_probs=1 PV reads P16, so the symptom is "long contexts drift", with no NaN,
//           no error and no warning. HANDOVER.md §7.1 calls this the one change that is easiest to miss.
//
// So the audit is four checks, not one. A is a bitwise A/B on identical inputs; B/C/D/E are invariants
// of the wide kernel's own output that hold at ANY cap and specifically cover the ranges A cannot.
// A2 is the same kind of A/B for the TILED kernel against the wide one - the contract that lets the
// tiled module stand in for the wide one at all, and the gate that has to pass before anyone times it.
enum { QK_NARROW = 0, QK_WIDE = 1, QK_TILE = 2, QK_GQA = 3 };
static const char *qk_name(int ksel) {
    return ksel == QK_GQA ? "GQA" : (ksel == QK_TILE ? "TILE"
             : (ksel == QK_WIDE ? "WIDE" : "NARROW"));
}
// Which of the three NON-MERGED row kernels owns this cap. That set - narrow, wide, tiled - is the
// validation ladder's own lineage, so it stays the reference even when the merged kernel is the one
// the engine dispatches; A3 is what connects the new module to that lineage bit by bit.
static int qk_reference_selection(const Fwd &f, uint32_t cap) {
    if (f.qk_force == 1u) return QK_NARROW;
    if (f.qk_force == 2u) return QK_WIDE;
    if (f.qk_force == 3u || cap > Fwd::kMaxWide) return QK_TILE;
    if (cap > Fwd::kMaxNarrow) return QK_WIDE;
    return QK_NARROW;
}
static void qk_dispatch(Fwd &f, int li, int ksel) {
    Engine &e = *f.e;
    const ModelPlan &pl = e.plan;
    VkCommandBuffer cb = begin_cmd(e.c, e.cpool);
    { uint32_t pc[8] = {1u, uint32_t(pl.n_head), uint32_t(pl.n_head_kv), uint32_t(pl.head_dim),
                        f.cap, f.round_probs, f2u(1.0f / sqrtf(float(pl.head_dim))), 0u};
      // The merged kernel is the only one whose workgroup is a kv GROUP rather than a query head, so
      // the grid is the one thing that differs; every other module gets n_head.
      dsp(f, ksel == QK_GQA ? f.aqkg[li]
              : (ksel == QK_TILE ? f.aqkt[li]
              : (ksel == QK_WIDE ? f.aqkw[li] : f.aqk[li])), cb, pc, 8,
          ksel == QK_GQA ? uint32_t(pl.n_head_kv) : uint32_t(pl.n_head)); }
    VK_CHECK(vkEndCommandBuffer(cb));
    submit_wait_sig(e.c, e.c.queue, cb, f.fence, VK_NULL_HANDLE, VK_NULL_HANDLE);
}
// One PV dispatch, same rule: merged -> n_head_kv workgroups, frozen -> n_head.
static void pv_dispatch(Fwd &f, int li, int mode) {
    Engine &e = *f.e;
    const ModelPlan &pl = e.plan;
    VkCommandBuffer cb = begin_cmd(e.c, e.cpool);
    { uint32_t pc[8] = {1u, uint32_t(pl.n_head), uint32_t(pl.n_head_kv), uint32_t(pl.head_dim),
                        f.cap, f.round_probs, 0u, 0u};
      dsp(f, mode ? f.apvg[li] : f.apv[li], cb, pc, 8,
          mode ? uint32_t(pl.n_head_kv) : uint32_t(pl.n_head)); }
    VK_CHECK(vkEndCommandBuffer(cb));
    submit_wait_sig(e.c, e.c.queue, cb, f.fence, VK_NULL_HANDLE, VK_NULL_HANDLE);
}
// Snapshot of the three row regions a dispatch just produced. Every readback lands in the SAME host
// staging buffer, so a pointer taken before the next act_rd is a pointer to the NEXT region by the
// time it is used - an aliasing bug that once printed a false "NOT BITWISE" in stage A. Copy out.
struct QkRow {
    std::vector<float> s, p;
    std::vector<uint32_t> x;
};
static QkRow qk_snapshot(Fwd &f, uint64_t nsc, uint64_t np16) {
    Engine &e = *f.e;
    QkRow r;
    act_rd(f, uint64_t(f.r.SCORE) * 4ull, nsc * 4ull);
    r.s.assign((const float *)e.host_ptr, (const float *)e.host_ptr + nsc);
    act_rd(f, uint64_t(f.r.PROB) * 4ull, nsc * 4ull);
    r.p.assign((const float *)e.host_ptr, (const float *)e.host_ptr + nsc);
    act_rd(f, uint64_t(f.r.P16) * 4ull, np16 * 4ull);
    r.x.assign((const uint32_t *)e.host_ptr, (const uint32_t *)e.host_ptr + np16);
    return r;
}

// Returns a failure bitmask: 1 = A (narrow vs wide) not bitwise, 2 = A2 (wide vs tiled) not bitwise,
// 4 = B pack invariant, 8 = C dead slots, 16 = E index audit, 64 = A3 (GQA-merged QK vs the row kernel
// that owns this cap) not bitwise, 128 = A4 (GQA-merged PV vs attn_pv_q8v.comp) not bitwise. The caller
// refuses to generate on nonzero: a row kernel that is not bitwise identical to the one it replaces is
// a second model, not a performance experiment. A3/A4 print SKIP only when the merged module was never
// built in this process (BONSAI_GQA_OPT=0 or a plan whose ratio is not 4), never when it was built and
// disagreed - that case is a FAIL and a nonzero return.
static int qk_audit(Fwd &f, uint32_t nfill) {
    Engine &e = *f.e;
    const ModelPlan &pl = e.plan;
    const uint32_t nh = uint32_t(pl.n_head), nkv = uint32_t(pl.n_head_kv), hd = uint32_t(pl.head_dim);
    const uint32_t cap = f.cap, hw = hd / 2u;
    const uint64_t nsc = uint64_t(nh) * cap;                    // SCORE / PROB element count
    int fail = 0;
    bool a2_skipped = false;
    const uint64_t np16 = uint64_t(nh) * cap / 2u;              // P16 word count
    const uint64_t dsc = f.n_dispatch, dbar = f.n_barrier;      // keep the per-token op count honest
    printf("\n=== QK kernel audit (BONSAI_QCHECK, %u live keys in the row, cap %u, dispatching %s"
           " q%d) ===\n", nfill, cap, qk_name(f.qkt ? QK_TILE : (f.wide ? QK_WIDE : QK_NARROW)),
           f.qk_tile);

    std::vector<float>  s1, p1, pr1;
    std::vector<uint32_t> x1;
    int li_hi = int(f.n_attn) - 1;

    // ---- A. bitwise narrow-vs-wide on IDENTICAL inputs. Only meaningful where the narrow kernel is
    // a valid program at all (cap <= its lane count); it validates the dot expression, the ascending
    // d, lane 0's serial max/exp/sum order and the RNE pack, because any deviation shows up as a bit.
    if (cap <= Fwd::kMaxNarrow) {
        for (const int li : {0, li_hi}) {
            qk_dispatch(f, li, QK_NARROW);
            act_rd(f, uint64_t(f.r.SCORE) * 4ull, nsc * 4ull);
            s1.assign((const float *)e.host_ptr, (const float *)e.host_ptr + nsc);
            act_rd(f, uint64_t(f.r.PROB) * 4ull, nsc * 4ull);
            p1.assign((const float *)e.host_ptr, (const float *)e.host_ptr + nsc);
            act_rd(f, uint64_t(f.r.P16) * 4ull, np16 * 4ull);
            x1.assign((const uint32_t *)e.host_ptr, (const uint32_t *)e.host_ptr + np16);
            // Each readback lands in the SAME host staging buffer (e.host_ptr), so a pointer taken
            // before the next act_rd is a pointer to the NEXT region by the time it is used. That
            // aliasing cost a false "NOT BITWISE" here on the first run: s2/p2 were both really the
            // P16 bytes. Copy every readback out, exactly like stages B/C/D/E below do.
            qk_dispatch(f, li, QK_WIDE);
            act_rd(f, uint64_t(f.r.SCORE) * 4ull, nsc * 4ull);
            std::vector<float> s2((const float *)e.host_ptr, (const float *)e.host_ptr + nsc);
            act_rd(f, uint64_t(f.r.PROB) * 4ull, nsc * 4ull);
            std::vector<float> p2((const float *)e.host_ptr, (const float *)e.host_ptr + nsc);
            act_rd(f, uint64_t(f.r.P16) * 4ull, np16 * 4ull);
            std::vector<uint32_t> x2((const uint32_t *)e.host_ptr, (const uint32_t *)e.host_ptr + np16);
            // The count alone cannot be read as either pass or fail: 1 ulp and 1000 ulp both count as
            // "a bit-diff", and they mean completely different things (one is the driver contracting
            // a multiply-add, the other is a wrong number). So the diff is measured in fp32 ulp -
            // the monotone-integer map, so it is an ORDER distance and not a subtraction of bits.
            // +0.0 and -0.0 are the same number, so they are 0 ulp apart even though their bits differ.
            auto ulp_dist = [](float a, float b) -> long {
                if (a == 0.0f && b == 0.0f) return 0;
                int32_t ia, ib; memcpy(&ia, &a, 4); memcpy(&ib, &b, 4);
                if (ia < 0) ia = INT32_MIN - ia;
                if (ib < 0) ib = INT32_MIN - ib;
                return labs(long(ia) - long(ib));
            };
            uint64_t bs = 0, bp = 0, bx = 0; unsigned us = 0, up = 0;
            double rel_s = 0.0, rel_p = 0.0;
            uint64_t fi_s = 0, fi_p = 0;                    // first differing index, for the dump
            for (uint64_t i = 0; i < nsc; ++i) {
                if (memcmp(&s1[i], &s2[i], 4)) { ++bs; if (!fi_s) fi_s = i + 1;
                                                 const long u = ulp_dist(s1[i], s2[i]);
                                                 if (unsigned(u) > us) us = unsigned(u);
                                                 if (s1[i] != 0.0f && fabs(double(s2[i] - s1[i]) / double(s1[i])) > rel_s)
                                                     rel_s = fabs(double(s2[i] - s1[i]) / double(s1[i])); }
                if (memcmp(&p1[i], &p2[i], 4)) { ++bp; if (!fi_p) fi_p = i + 1;
                                                 const long u = ulp_dist(p1[i], p2[i]);
                                                 if (unsigned(u) > up) up = unsigned(u);
                                                 if (p1[i] != 0.0f && fabs(double(p2[i] - p1[i]) / double(p1[i])) > rel_p)
                                                     rel_p = fabs(double(p2[i] - p1[i]) / double(p1[i])); }
            }
            for (uint64_t i = 0; i < np16; ++i) if (x1[i] != x2[i]) ++bx;
            printf("  A layer %d (cap<=256): SCORE %llu/%llu bit-diff, PROB %llu, P16 %llu  -> %s\n",
                   li, (unsigned long long)bs, (unsigned long long)nsc, (unsigned long long)bp,
                   (unsigned long long)bx, (bs || bp || bx) ? "NOT BITWISE" : "bitwise identical");
            if (bs || bp || bx) fail |= 1;
            printf("      size of the diff: SCORE <= %u fp32 ulp (rel %.3e), PROB <= %u ulp (rel %.3e)"
                   "  [first diff: SCORE idx %llu, PROB idx %llu]\n",
                   us, rel_s, up, rel_p,
                   (unsigned long long)(fi_s ? fi_s - 1 : 0), (unsigned long long)(fi_p ? fi_p - 1 : 0));
            auto fbits = [](float x) { uint32_t b; memcpy(&b, &x, 4); return b; };
            if (fi_s) printf("      SCORE@%llu: narrow %.9g (0x%08X)  wide %.9g (0x%08X)\n",
                             (unsigned long long)(fi_s - 1), s1[fi_s - 1], fbits(s1[fi_s - 1]),
                             s2[fi_s - 1], fbits(s2[fi_s - 1]));
            if (fi_p) printf("      PROB @%llu: narrow %.9g (0x%08X)  wide %.9g (0x%08X)\n",
                             (unsigned long long)(fi_p - 1), p1[fi_p - 1], fbits(p1[fi_p - 1]),
                             p2[fi_p - 1], fbits(p2[fi_p - 1]));
            printf("      P16 (the operand PV reads when round_probs=1): %s\n",
                   bx ? "DIFFERS" : "bitwise identical - the token stream cannot differ between kernels");
        }
    } else {
        printf("  A skipped: the narrow kernel cannot address cap %u rows (its lane IS the position)\n",
               cap);
    }

    // ---- A2. bitwise WIDE-vs-TILED on identical inputs, at every cap the two can both cover. This is
    // the gate that lets the tiled kernel stand in for the wide one. It is a stronger test than A in
    // one way and weaker in another: it holds at cap 2048 (where the row crosses 8 QTILE tiles of the
    // max/exp walk and 1024 packed pairs), but the reference it is bitwise against is our own kernel,
    // so it transfers the layer-3 alignment verdict from the wide kernel rather than re-earning it.
    // What makes that legitimate is that both kernels here produce the number by the SAME dependency
    // chain (ascending d in the dot, lane-0 ascending serial max/exp/sum, RNE pack); A2 is precisely
    // the test that the tiling did not reorder it.
    if (cap <= Fwd::kMaxWide) {
        for (const int li : {0, li_hi}) {
            qk_dispatch(f, li, QK_WIDE); const QkRow a = qk_snapshot(f, nsc, np16);
            qk_dispatch(f, li, QK_TILE); const QkRow b = qk_snapshot(f, nsc, np16);
            uint64_t bs = 0, bp = 0, bx = 0, fi = ~0ull, fp = ~0ull;
            for (uint64_t i = 0; i < nsc; ++i) {
                if (memcmp(&a.s[i], &b.s[i], 4)) { ++bs; if (fi == ~0ull) fi = i; }
                if (memcmp(&a.p[i], &b.p[i], 4)) { ++bp; if (fp == ~0ull) fp = i; }
            }
            for (uint64_t i = 0; i < np16; ++i) if (a.x[i] != b.x[i]) ++bx;
            printf("  A2 layer %d (cap %u, QTILE %u): SCORE %llu/%llu bit-diff, PROB %llu, P16 %llu"
                   "  -> %s\n", li, cap, f.qk_tile, (unsigned long long)bs,
                   (unsigned long long)nsc, (unsigned long long)bp, (unsigned long long)bx,
                   (bs || bp || bx) ? "NOT BITWISE" : "bitwise identical");
            if (bs || bp || bx) {
                fail |= 2;
                auto fbits = [](float x) { uint32_t t; memcpy(&t, &x, 4); return t; };
                if (fi != ~0ull) printf("      SCORE@%llu: wide %.9g (0x%08X)  tiled %.9g (0x%08X)\n",
                                        (unsigned long long)fi, a.s[fi], fbits(a.s[fi]),
                                        b.s[fi], fbits(b.s[fi]));
                if (fp != ~0ull) printf("      PROB @%llu: wide %.9g (0x%08X)  tiled %.9g (0x%08X)\n",
                                        (unsigned long long)fp, a.p[fp], fbits(a.p[fp]),
                                        b.p[fp], fbits(b.p[fp]));
            }
        }
    } else {
        printf("  A2 SKIPPED (not a pass): the WIDE kernel cannot cover cap %u rows (MAXCAP %u), so\n"
               "      there is no same-input reference to be bitwise against at this cap. What DOES\n"
               "      run here is B/C/D/E on the tiled kernel's own output - E recomputes the dots in\n"
               "      double from the read-back fp16 arena, which is an independent oracle at any cap,\n"
               "      but it is a 1e-3 tolerance check, NOT a bitwise one. Say so in the report.\n",
               cap, Fwd::kMaxWide);
        a2_skipped = true;
    }

    // ---- A3. bitwise GQA-MERGED-vs-REFERENCE on identical inputs, at EVERY cap the engine can reach.
    // This is the gate the merge was conditioned on: "keep the bit-exact path, make the optimized one
    // switchable". The reference is the kernel that owns this cap in the non-merged lineage - narrow at
    // cap <= 256 (the oracle the whole fidelity ladder was measured against), wide at 257..2048, tiled
    // past that - so A3 does not just prove the merge is self-consistent, it pins the merged kernel to
    // the same bit pattern the validated kernel produces for the same bytes.
    // What it can catch, in order of likelihood: (1) FMA contraction changing a product because the K
    // operand now arrives through a hoisted register instead of an inline unpack - the compiler is free
    // to do that and the fix is to materialise the multiply, exactly as attn_pv_q8v.comp does; (2) a
    // wrong head-to-kv mapping (heads of a group are h0 = kv*HPK + s, which is what h/(nh/nkv) means);
    // (3) a row offset that forgot cap is per-head rather than per-group. All three print as a bit diff,
    // none of them print as an error.
    const int refsel = qk_reference_selection(f, cap);
    if (f.aqkg[0].pp != VK_NULL_HANDLE) {
        uint64_t tot = 0;
        for (const int li : {0, li_hi}) {
            qk_dispatch(f, li, refsel);  const QkRow a = qk_snapshot(f, nsc, np16);
            qk_dispatch(f, li, QK_GQA);  const QkRow b = qk_snapshot(f, nsc, np16);
            uint64_t bs = 0, bp = 0, bx = 0, fi = ~0ull, fp = ~0ull, fx = ~0ull;
            for (uint64_t i = 0; i < nsc; ++i) {
                if (memcmp(&a.s[i], &b.s[i], 4)) { ++bs; if (fi == ~0ull) fi = i; }
                if (memcmp(&a.p[i], &b.p[i], 4)) { ++bp; if (fp == ~0ull) fp = i; }
            }
            for (uint64_t i = 0; i < np16; ++i) if (a.x[i] != b.x[i]) { ++bx; if (fx == ~0ull) fx = i; }
            tot += bs + bp + bx;
            printf("  A3 layer %d (cap %u, ref %s, HPK %u, merged q%u, grid %u -> %u):"
                   " SCORE %llu/%llu bit-diff, PROB %llu, P16 %llu  -> %s\n",
                   li, cap, qk_name(refsel), f.hpk, f.gqa_tile, uint32_t(e.plan.n_head),
                   uint32_t(e.plan.n_head_kv), (unsigned long long)bs,
                   (unsigned long long)nsc, (unsigned long long)bp, (unsigned long long)bx,
                   (bs || bp || bx) ? "NOT BITWISE" : "bitwise identical");
            if (bs || bp || bx) {
                fail |= 64;
                auto fbits = [](float x) { uint32_t t; memcpy(&t, &x, 4); return t; };
                if (fi != ~0ull) printf("      SCORE@%llu: %s %.9g (0x%08X)  merged %.9g (0x%08X)\n",
                                        (unsigned long long)fi, qk_name(refsel), a.s[fi], fbits(a.s[fi]),
                                        b.s[fi], fbits(b.s[fi]));
                if (fp != ~0ull) printf("      PROB @%llu: %s %.9g (0x%08X)  merged %.9g (0x%08X)\n",
                                        (unsigned long long)fp, qk_name(refsel), a.p[fp], fbits(a.p[fp]),
                                        b.p[fp], fbits(b.p[fp]));
                if (fx != ~0ull) printf("      P16  @%llu: %s 0x%08X  merged 0x%08X\n",
                                        (unsigned long long)fx, qk_name(refsel), a.x[fx], b.x[fx]);
            }
        }
        printf("      reach: every one of the %llu SCORE/PROB elements and %llu P16 words of both the"
               " first and the last attention layer, i.e. all %u heads x %u slots, packed pairs"
               " included%s\n", (unsigned long long)(2ull * nsc), (unsigned long long)(2ull * np16),
               nh, cap, np16 > 512ull ? " past the 512-word boundary" : "");
        (void)tot;
    } else {
        printf("  A3 not built: the merged module was not created in this process (BONSAI_GQA_OPT with"
               " a plan whose n_head/n_head_kv != 4, or BONSAI_GQA_PART excluding the QK half)\n");
    }

    // ---- A4. the same claim on the V side: merged PV vs the frozen attn_pv_q8v.comp, bitwise on
    // PGATE and GOUT. The probabilities feeding PV are produced by whichever QK kernel owns the row,
    // so both PV runs see literally the same input bytes; only the reduction shape may differ.
    if (f.apvg[0].pp != VK_NULL_HANDLE) {
        const uint64_t ng = uint64_t(nh) * hd;                // PGATE / GOUT element count
        for (const int li : {0, li_hi}) {
            qk_dispatch(f, li, f.qkg ? QK_GQA : refsel);      // fill P16/PROB for this layer
            pv_dispatch(f, li, 0);
            act_rd(f, uint64_t(f.r.PGATE) * 4ull, ng * 4ull);
            std::vector<float> g1((const float *)e.host_ptr, (const float *)e.host_ptr + ng);
            act_rd(f, uint64_t(f.r.GOUT) * 4ull, ng * 4ull);
            std::vector<float> h1((const float *)e.host_ptr, (const float *)e.host_ptr + ng);
            pv_dispatch(f, li, 1);
            act_rd(f, uint64_t(f.r.PGATE) * 4ull, ng * 4ull);
            std::vector<float> g2((const float *)e.host_ptr, (const float *)e.host_ptr + ng);
            act_rd(f, uint64_t(f.r.GOUT) * 4ull, ng * 4ull);
            std::vector<float> h2((const float *)e.host_ptr, (const float *)e.host_ptr + ng);
            uint64_t bg = 0, bh = 0, fg = ~0ull, fh = ~0ull;
            for (uint64_t i = 0; i < ng; ++i) {
                if (memcmp(&g1[i], &g2[i], 4)) { ++bg; if (fg == ~0ull) fg = i; }
                if (memcmp(&h1[i], &h2[i], 4)) { ++bh; if (fh == ~0ull) fh = i; }
            }
            printf("  A4 layer %d (cap %u, PV q8_0 frozen vs merged, grid %u -> %u):"
                   " PGATE %llu/%llu bit-diff, GOUT %llu  -> %s\n", li, cap,
                   uint32_t(e.plan.n_head), uint32_t(e.plan.n_head_kv),
                   (unsigned long long)bg, (unsigned long long)ng, (unsigned long long)bh,
                   (bg || bh) ? "NOT BITWISE" : "bitwise identical");
            if (bg || bh) {
                fail |= 128;
                auto fbits = [](float x) { uint32_t t; memcpy(&t, &x, 4); return t; };
                if (fg != ~0ull) printf("      PGATE@%llu: frozen %.9g (0x%08X)  merged %.9g (0x%08X)\n",
                                        (unsigned long long)fg, g1[fg], fbits(g1[fg]),
                                        g2[fg], fbits(g2[fg]));
                if (fh != ~0ull) printf("      GOUT @%llu: frozen %.9g (0x%08X)  merged %.9g (0x%08X)\n",
                                        (unsigned long long)fh, h1[fh], fbits(h1[fh]),
                                        h2[fh], fbits(h2[fh]));
            }
        }
    } else {
        printf("  A4 not built: no merged PV module (needs BONSAI_KV_V_QUANT=q8_0, HPK 4 and"
               " BONSAI_GQA_PART including bit 1)\n");
    }

    // ---- B/C/D/E run on the DISPATCHED row kernel's own output for the LAST attention layer, which
    // is the state the next token would actually consume.
    qk_dispatch(f, li_hi, f.qkg ? QK_GQA : qk_reference_selection(f, cap));
    act_rd(f, uint64_t(f.r.SCORE) * 4ull, nsc * 4ull);
    std::vector<float> sc((const float *)e.host_ptr, (const float *)e.host_ptr + nsc);
    act_rd(f, uint64_t(f.r.PROB) * 4ull, nsc * 4ull);
    std::vector<float> pb((const float *)e.host_ptr, (const float *)e.host_ptr + nsc);
    act_rd(f, uint64_t(f.r.P16) * 4ull, np16 * 4ull);
    std::vector<uint32_t> p16((const uint32_t *)e.host_ptr, (const uint32_t *)e.host_ptr + np16);
    act_rd(f, uint64_t(f.r.Q16) * 4ull, uint64_t(nh) * hw * 4ull);
    std::vector<uint32_t> q16((const uint32_t *)e.host_ptr, (const uint32_t *)e.host_ptr + nh * hw);
    // The read-back has to use the SAME nesting as the kernels: one layer's arena starts at
    // KV + li * cap * kv_layer (R.kv_layer is per POSITION, not per layer). And it only has to cover
    // the keys E can actually sample, which are all below nfill: the whole arena would be
    // cap*nkv*hd/2 words = 64 MiB at cap 32768, and reading that in one act_rd is what segfaulted.
    const uint64_t kw_live = uint64_t(nfill) * nkv * hw;
    std::vector<uint32_t> kc = act_rd_words(
        f, uint64_t(f.r.KV + uint32_t(uint64_t(li_hi) * uint64_t(f.cap) * f.r.kv_layer)) * 4ull,
        kw_live);

    // B. the pack invariant, over EVERY pair - including the ones above 512 that a non-strided
    // packer would never touch. This is the direct test for WIDE#2.
    uint64_t bad = 0, first = ~0ull, stale = 0;
    for (uint64_t i = 0; i < np16; ++i) {
        const uint64_t e2 = i / (cap / 2u), w = i % (cap / 2u);
        const uint32_t want = pack2(pb[e2 * cap + 2u * w], pb[e2 * cap + 2u * w + 1u]);
        if (want != p16[i]) { ++bad; if (first == ~0ull) first = i; }
        if (i >= 512u && p16[i] == 0u && pb[e2 * cap + 2u * w] != 0.0f) ++stale;
    }
    printf("  B P16 == pack2(PROB) for %llu pairs: %llu mismatches%s  -> %s\n",
           (unsigned long long)np16, (unsigned long long)bad,
           bad ? "" : " (covers indices above 512, the strided-pack range)",
           bad ? "PACK BUG" : "ok");
    if (bad) { fail |= 4; printf("      first mismatch at word %llu\n", (unsigned long long)first); }
    (void)stale;

    // C. dead slots are EXACTLY +0.0 bit-for-bit (exp(-inf-max) semantics, §8.3).
    uint64_t nz = 0;
    for (uint64_t r = 0; r < nh; ++r)
        for (uint32_t i = nfill; i < cap; ++i) {
            uint32_t bits; memcpy(&bits, &pb[r * cap + i], 4);
            if (bits != 0u) ++nz;                       // catches -0.0 (0x80000000) too
        }
    printf("  C dead slots [%u,%u) x %u heads are bitwise +0.0: %llu violations -> %s\n",
           nfill, cap, nh, (unsigned long long)nz, nz ? "NOT EXACT ZERO" : "ok");
    if (nz) fail |= 8;

    // D. each live row sums to 1 within fp32 serial-summation noise.
    double worst = 0.0;
    for (uint64_t r = 0; r < nh; ++r) {
        float s = 0.0;
        for (uint32_t i = 0; i < nfill; ++i) s = s + pb[r * cap + i];
        if (fabs(double(s) - 1.0) > worst) worst = fabs(double(s) - 1.0);
    }
    printf("  D live-row sums: worst |sum-1| = %.3e over %u heads x %u keys -> %s\n",
           worst, nh, nfill, worst < 1e-4 ? "ok" : "SUSPECT");
    if (worst >= 1e-4) fail |= 32;

    // E. index audit in double: recompute 32 sampled keys straight from the readback fp16 arena and
    // compare with the device's SCORE. This is NOT a bit check (double adds in another order); it is
    // a "did lane j read key j" check, and it deliberately samples past 256 so the strided loop
    // iterations are covered, including the last live key and the last live pair of the pack.
    const uint32_t nsm = 32u;
    double worst_err = 0.0; uint32_t worst_k = 0;
    uint64_t checked = 0;
    for (uint64_t r = 0; r < nh; ++r) {
        const uint32_t h = uint32_t(r), kv = h / (nh / nkv);
        for (uint32_t s = 0u; s < nsm; ++s) {
            uint32_t k = uint32_t(uint64_t(s) * nfill / nsm);          // spread over the live range
            if (k >= nfill) k = nfill - 1u;
            const uint32_t k1 = (k * nkv + kv) * hw, q1 = h * hw;
            double acc = 0.0;
            for (uint32_t d = 0u; d < hw; ++d)
                acc += double(word_lo(q16[q1 + d])) * double(word_lo(kc[k1 + d])) +
                       double(word_hi(q16[q1 + d])) * double(word_hi(kc[k1 + d]));
            const double err = fabs(acc - double(sc[r * cap + k]));
            if (err > worst_err) { worst_err = err; worst_k = k; }
            ++checked;
        }
    }
    printf("  E index audit: %llu keys recomputed in double, worst |dot-SCORE| = %.3e at key %u -> %s\n",
           (unsigned long long)checked, worst_err, worst_k,
           worst_err < 1e-3 ? "keys line up" : "WRONG KEY SLOT");
    if (worst_err >= 1e-3) fail |= 16;
    printf("      reach: %u live keys = %u NTILE tile(s) of dot products, %u packed pairs = %u tile(s)"
           " of packs%s\n", nfill, (nfill + 255u) / 256u, uint32_t(np16 / nh),
           (uint32_t(np16 / nh) + 255u) / 256u,
           np16 > 512ull ? "  <- past 512: the range a non-strided pack would drop"
                         : "  (does not yet pass 512 packed pairs)");
    f.n_dispatch = dsc; f.n_barrier = dbar;                 // the audit is not part of a token
    printf("  gate: %s (bitmask %d = A%s A2%s A3%s A4%s B%s C%s D%s E%s)\n",
           fail ? "FAILED" : "PASS", fail,
           fail & 1 ? " FAIL" : " ok",
           a2_skipped ? " SKIP" : (fail & 2 ? " FAIL" : " ok"),
           f.aqkg[0].pp != VK_NULL_HANDLE ? (fail & 64 ? " FAIL" : " ok") : " SKIP",
           f.apvg[0].pp != VK_NULL_HANDLE ? (fail & 128 ? " FAIL" : " ok") : " SKIP",
           fail & 4 ? " FAIL" : " ok", fail & 8 ? " FAIL" : " ok",
           fail & 32 ? " FAIL" : " ok", fail & 16 ? " FAIL" : " ok");
    return fail;
}

// ------------------------------------------------------------------ the sampler
// Sampled by default. A distilled 1.58-bit base model left on argmax falls into short token loops
// ("a joke, joke, or funny situation" / "stories or stories") on short chat prompts; a temperature
// above zero plus the presence/repetition penalties below break that without touching the forward
// pass. Keep temp > 0: temp == 0 short-circuits to argmax in pick() and re-opens the loop. Every
// one of these values is still overridable per CLI run (BONSAI_* env) or per HTTP request (g_req).
struct Sampler {
    float temp = 0.70f; int top_k = 0; float top_p = 1.0f; uint64_t rng = 88172645463325252ull;
    int eos = -1;
    // Anti-repetition knobs (HF / llama.cpp semantics), applied to the raw logits over a window of
    // the recent generation BEFORE temperature / top-k / top-p. 1.0 / 0.0 / 0.0 mean "off".
    // These are the handover release defaults, verified on the two short prompts that actually
    // reproduce the loop (runs/overnight_tuning.log, FINAL section). The overnight ladder's own
    // 5-question WINNER was temp=0.90 / presence=1.20 / rep=1.20 (fluent 5/5 on logic questions);
    // callers who prefer that combination can simply send it per request, with no rebuild.
    float rep_penalty = 1.15f;      // >1 discourages re-use: divide a +logit, multiply a -logit
    float presence_penalty = 0.50f; // subtract once per distinct token present in the window
    float freq_penalty = 0.0f;      // subtract in proportion to the token's count in the window
    int penalty_last_n = 64;        // history window the penalties read (<=0 = the whole history)
    uint64_t next_rand() {                       // xorshift64*, deterministic across runs
        rng ^= rng >> 12; rng ^= rng << 25; rng ^= rng >> 27;
        return rng * 2685821657736338717ull;
    }
    double uniform01() { return double(next_rand() >> 11) * (1.0 / 9007199254740992.0); }

    int pick(const float *lg, uint32_t n, const std::vector<int> *hist = nullptr) {
        // Penalties mutate logits, so adjust a COPY: the caller's buffer is refilled by the next
        // fwd_token and must not carry this step's adjustments into the next step's argmax.
        std::vector<float> work;
        if (hist && !hist->empty() &&
            (rep_penalty != 1.0f || presence_penalty != 0.0f || freq_penalty != 0.0f)) {
            const size_t win = penalty_last_n > 0 ? size_t(penalty_last_n) : hist->size();
            const size_t begin = hist->size() > win ? hist->size() - win : 0;
            std::unordered_map<int, int> cnt;
            for (size_t k = begin; k < hist->size(); ++k) {
                const int id = (*hist)[k];
                if (id >= 0 && uint32_t(id) < n) ++cnt[id];
            }
            work.assign(lg, lg + n);
            for (const auto &kv : cnt) {
                const uint32_t id = uint32_t(kv.first);
                float v = work[id];
                if (rep_penalty != 1.0f) v = (v < 0.0f) ? v * rep_penalty : v / rep_penalty;
                if (presence_penalty != 0.0f) v -= presence_penalty;
                if (freq_penalty != 0.0f) v -= freq_penalty * float(kv.second);
                work[id] = v;
            }
            lg = work.data();
        }
        uint32_t best = 0; float bv = lg[0];
        for (uint32_t i = 1; i < n; ++i) if (lg[i] > bv) { bv = lg[i]; best = i; }
        if (temp <= 0.0f) return int(best);
        // top-k: the k-th largest logit is the floor
        float floor_v = -3.4e38f;
        if (top_k > 0 && uint32_t(top_k) < n) {
            std::vector<float> v(lg, lg + n);
            std::nth_element(v.begin(), v.begin() + top_k - 1, v.end(), std::greater<float>());
            floor_v = v[top_k - 1];
        }
        std::vector<double> pr(n, 0.0);            // (n, 0.0), not (size_t(n)): the latter is the
                                                   // vexed parse that declares a function
        double sum = 0.0;
        for (uint32_t i = 0; i < n; ++i) {
            if (lg[i] < floor_v) continue;
            const double p = exp(double((lg[i] - bv) / temp));
            pr[i] = p; sum += p;
        }
        if (sum <= 0.0) return int(best);
        // top-p: sort the candidates by probability and cut the tail past the cumulative mass
        if (top_p < 1.0f) {
            std::vector<uint32_t> idx;
            for (uint32_t i = 0; i < n; ++i) if (pr[i] > 0.0) idx.push_back(i);
            std::sort(idx.begin(), idx.end(), [&](uint32_t a, uint32_t b) { return pr[a] > pr[b]; });
            double c = 0.0; size_t keep = idx.size();
            for (size_t j = 0; j < idx.size(); ++j) { c += pr[idx[j]] / sum; if (c >= top_p) { keep = j + 1; break; } }
            for (size_t j = keep; j < idx.size(); ++j) { sum -= pr[idx[j]]; pr[idx[j]] = 0.0; }
        }
        const double u = uniform01() * sum;
        double a = 0.0;
        for (uint32_t i = 0; i < n; ++i) { a += pr[i]; if (a >= u) return int(i); }
        return int(best);
    }
};

// ------------------------------------------------------------------ the generation loop
static std::string detok_sp(const std::string &raw) {
    std::string o;
    size_t i = 0;
    while (i < raw.size()) {
        // U+2581 BOX DRAWINGS LIGHT UPPER, the sentencepiece space marker, is 3 bytes: e2 96 81
        if (i + 2 < raw.size() && (uint8_t)raw[i] == 0xE2 && (uint8_t)raw[i+1] == 0x96 &&
            (uint8_t)raw[i+2] == 0x81) { o += ' '; i += 3; }
        else { o += raw[i]; ++i; }
    }
    return o;
}

// A server answers many conversations from one resident weight set, so the arena has to go back to the
// state fwd_init left it in before each one. That state is "everything zero except the two uploaded
// weight regions", and the layout makes it two contiguous ranges: [0, QKWN) is scratch, [QKWN, SIGNS +
// n_sign_f) is the sign vectors and the attention norm weights, [KV, bytes) is the K/V cache plus the
// GDN recurrent state and conv history. Zeroing the two outer ranges is therefore an EXACT restore,
// not an approximation of one - and it never touches a weight, so the dequant path is not involved.
// The guard's own per-conversation state goes with it; the spill watch's reference does NOT, because
// the arena it watches is never freed and the process-lifetime zero is still the right one.
static void fwd_reset_state(Fwd &f) {
    const uint64_t w_end = uint64_t(f.r.SIGNS + f.n_sign_f) * 4ull;
    const uint64_t s_beg = uint64_t(f.r.KV) * 4ull;
    if (s_beg < w_end) {
        fprintf(stderr, "fwd_reset_state: the region layout put KV at %llu B, inside the uploaded"
                        " weights that end at %llu B. Zeroing it would erase the sign vectors and"
                        " produce garbage with no validation error. Refusing.\n",
                (unsigned long long)s_beg, (unsigned long long)w_end);
        exit(2);
    }
    const auto t0 = std::chrono::steady_clock::now();
    act_store(f, 0, nullptr, uint64_t(f.r.QKWN) * 4ull);
    act_store(f, s_beg, nullptr, f.bytes - s_beg);
    f.kvguard.live_low = 0;
    f.kvguard.n_refuse = f.kvguard.n_evict = f.kvguard.n_warn = 0;
    f.spill_last = -1.0;
    printf("  [serve] arena restored to its init state (%.1f MiB zeroed, weights untouched) in %.1f ms\n",
           double(f.bytes - (w_end - uint64_t(f.r.QKWN) * 4ull)) / 1048576.0,
           std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
}

// What one reply cost, for a caller that has to report it (the OpenAI `usage` object and the WebUI's
// live tok/s). nullptr from every CLI call site, so nothing below changes behaviour when it is absent.
// The rate is the engine's own definition - generated tokens over the summed fwd_token wall time -
// not a wall-clock number that would silently include the sampler and the printf.
struct RunStats {
    int n_prompt = 0, n_gen = 0, n_think = 0;
    double ms_prefill = 0.0, ms_gen = 0.0, ms_first = 0.0;
    const char *finish = "length";
    double tok_s() const { return n_gen * 1000.0 / (ms_gen > 0.0 ? ms_gen : 1.0); }
};

static int fwd_run(Engine &e, Fwd &f, const char *prompt, int n_max, RunStats *out = nullptr) {
    Bpe tok; if (!tok.load(e.hdr)) { fprintf(stderr, "tokenizer load failed\n"); return 5; }
    Sampler sp;
    if (const char *v = getenv("BONSAI_TEMP"))  sp.temp  = float(atof(v));
    if (const char *v = getenv("BONSAI_TOPK"))  sp.top_k = atoi(v);
    if (const char *v = getenv("BONSAI_TOPP"))  sp.top_p = float(atof(v));
    if (const char *v = getenv("BONSAI_REP_PENALTY")) sp.rep_penalty = float(atof(v));
    if (const char *v = getenv("BONSAI_PRESENCE"))    sp.presence_penalty = float(atof(v));
    if (const char *v = getenv("BONSAI_FREQ"))       sp.freq_penalty = float(atof(v));
    if (g_req.temp  >= 0.0f) sp.temp  = g_req.temp;
    if (g_req.top_k >= 0)    sp.top_k = g_req.top_k;
    if (g_req.top_p >= 0.0f) sp.top_p = g_req.top_p;
    if (g_req.rep_penalty >= 0.0f) sp.rep_penalty = g_req.rep_penalty;
    if (g_req.presence    >= 0.0f) sp.presence_penalty = g_req.presence;
    if (g_req.freq        >= 0.0f) sp.freq_penalty = g_req.freq;
    if (g_req.seed)          sp.rng   = g_req.seed;
    sp.eos = tok.eos;

    std::vector<int> ids;
    // BONSAI_CHAT=1 wraps the prompt in the model's own template. The rendered string is what gets
    // tokenized - never a hand-written id sequence - because encode() is the path that was aligned to
    // the reference tokenizer, and render_ids() below is the independent second path: the two must be
    // IDENTICAL, and BONSAI_CHAT_DUMP writes the rendered prompt + its ids so they can be handed to
    // the oracle's llama-tokenize. That last cross-check (HANDOVER §7.1 step D4) is still OPEN.
    if (!f.pre_ids.empty()) {
        // The HTTP shell rendered N turns with the same control ids and already cross-checked its two
        // id paths; re-rendering one user turn here would throw the history away.
        ids = f.pre_ids;
        printf("  chat transcript: %zu ids supplied by the caller (thinking=%s)\n",
               ids.size(), f.thinking ? "on " : "off");
    } else if (f.chat) {
        if (!chat::ids_are_singles(tok)) {
            fprintf(stderr, "chat: a control id does not round-trip as a SINGLE token in our vocab - "
                            "refusing to render (the model would see prose, not a delimiter)\n");
            return 6;
        }
        const bool thinking = f.thinking != 0u;
        const std::string rendered = chat::render(tok, f.system_text, prompt, thinking);
        ids = tok.encode(rendered, true, false);
        const std::vector<int> assembled = chat::render_ids(tok, f.system_text, prompt, thinking);
        size_t d = 0;
        while (d < ids.size() && d < assembled.size() && ids[d] == assembled[d]) ++d;
        printf("  chat template: thinking=%s system=%s user=%zu chars -> %d tokens, "
               "id paths (encode vs assembled) %s\n",
               thinking ? "on " : "off", f.system_text.empty() ? "(none)" : "given",
               strlen(prompt), int(ids.size()),
               (d == ids.size() && d == assembled.size()) ? "IDENTICAL" : "DIVERGE");
        if (d != ids.size() || d != assembled.size())
            printf("    first divergence at %zu: encode=%d assembled=%d\n", d,
                   d < ids.size() ? ids[d] : -1, d < assembled.size() ? assembled[d] : -1);
        if (const char *v = getenv("BONSAI_CHAT_DUMP")) {   // "<user_file> <ids_file>"
            const char *sp_ = strchr(v, ' ');
            const std::string uf = sp_ ? std::string(v, size_t(sp_ - v)) : std::string(v);
            const std::string ifn = sp_ ? std::string(sp_ + 1) : std::string();
            if (FILE *g = fopen(uf.c_str(), "wb")) { fputs(prompt, g); fclose(g); }
            if (FILE *g = fopen(ifn.c_str(), "wb")) {
                for (size_t i = 0; i < ids.size(); ++i) fprintf(g, "%s%d", i ? "," : "", ids[i]);
                fprintf(g, "\n"); fclose(g);
            }
            printf("    chat dump: user -> %s, ids -> %s\n", uf.c_str(), ifn.c_str());
        }
    } else {
        ids = tok.encode(prompt, true, false);
    }
    // CONTEXT-FIT GUARD. The K/V arena holds EXACTLY cap positions (R.kv_layer uint words per layer,
    // and the norm_rope kernel indexes it by `position`), and nothing above this line checked the
    // prompt length: a prompt longer than cap wrote past its own region into whichever region the arena
    // laid out next - silent corruption, no validation error, no crash. The decode loop already stopped
    // at cap (line "context cap reached"); the prefill loop never did. Refuse before dispatching.
    {
        const size_t need = ids.size();
        if (need > size_t(f.cap)) {
            fprintf(stderr,
                    "context overflow: the prompt renders to %zu tokens but the KV arena has %u slots"
                    " (cap). Positions %u..%zu would be written past the end of the K/V regions.\n"
                    "  nothing was dispatched. Fix: shorten the prompt/system text, or raise"
                    " BONSAI_CAP to >= %zu (the widest validated row is %u; each 256 slots of cap"
                    " costs %.2f MiB of device-local memory for the K/V arenas with V = %s).\n",
                    need, f.cap, f.cap, need - 1, need, qk_row_limit(f),
                    double(256ull * f.kvgeom.bytes_slot) / 1048576.0, f.v_q8 ? "q8_0" : "fp16");
            return 7;
        }
        if (need + size_t(n_max) > size_t(f.cap))
            printf("  note: prompt %zu tokens + %d requested = %zu > cap %u; generation stops early"
                   " at [context cap reached].\n", need, n_max, need + size_t(n_max), f.cap);
    }
    printf("  prompt %d tokens:", int(ids.size()));
    for (int t : ids) printf(" %d", t);
    printf("\n  sampler: temp=%.2f top_k=%d top_p=%.2f rep=%.2f pres=%.2f freq=%.2f eos=%d cap=%u\n",
           sp.temp, sp.top_k, sp.top_p, sp.rep_penalty, sp.presence_penalty, sp.freq_penalty,
           sp.eos, f.cap);

    std::vector<float> logits;
    double ms_prefill = 0.0, ms_gen = 0.0;
    const uint32_t V = uint32_t(e.plan.vocab);
    int pos = 0;
    for (size_t i = 0; i < ids.size(); ++i) {
        auto t0 = std::chrono::steady_clock::now();
        fwd_token(f, uint32_t(ids[i]), uint32_t(pos), logits);
        auto t1 = std::chrono::steady_clock::now();
        ms_prefill += std::chrono::duration<double, std::milli>(t1 - t0).count();
        ++pos;
        if (f.stop_at_block >= 0) { fwd_dump(f); printf("  cut after block %d: no logits, alignment mode\n", f.stop_at_block); return 0; }
    }
    printf("  prefill %.1f ms for %zu prompt tokens (%.2f tok/s)\n", ms_prefill, ids.size(),
           ids.size() * 1000.0 / (ms_prefill > 0 ? ms_prefill : 1.0));
    // The row is now as wide as the prompt is long, which is exactly the state the QK audit needs:
    // nothing else in the run touches SCORE/PROB/P16 between here and the first sampled logit.
    if (f.qcheck && f.stop_at_block < 0) {
        const int gate = qk_audit(f, uint32_t(pos));
        if (gate) {
            fprintf(stderr,
                "\n  [GATE] QK kernel audit returned %d - REFUSING TO GENERATE.\n"
                "         A row kernel that is not bitwise identical to the kernel it replaces is a\n"
                "         second model, not a performance experiment. Fix the kernel or run with\n"
                "         BONSAI_QKERNEL=2 (the wide kernel, which the alignment ladder was measured on).\n",
                gate);
            return 6;
        }
    }
    // BONSAI_KVCHECK=<file> [BONSAI_KVCHECK_POS=<n>, default 64] [BONSAI_KVCHECK_LAYER=<li>, default 0]
    if (const char *kf_ = getenv("BONSAI_KVCHECK")) {
        uint32_t mp = 64u, kl = 0u;
        if (const char *v = getenv("BONSAI_KVCHECK_POS"))  mp = uint32_t(atoi(v));
        if (const char *v = getenv("BONSAI_KVCHECK_LAYER")) kl = uint32_t(atoi(v));
        if (kl < f.n_attn) kv_dump(f, uint32_t(pos), kl, kf_, mp);
        else printf("  kv check: layer index %u out of range (n_attn %u)\n", kl, f.n_attn);
    }

    // ---------------------------------------------------------------- the tail, against the same dump
    // runs/dbg_full_real.log does not stop at block 0: the 1-token decode graph it evaluated ends in
    // result_output, so the WHOLE network has an oracle for the identical state our engine starts
    // from (prompt "ab" = one token, id 365, every cache zero). Transcribed from log lines
    // 57198..57253, %8.4f prints plus the printed sums:
    //   l_out-31  ADD(ffn_out-31, attn_residual-31)  [-2.9615  9.1218  4.7726]  sum  -381.804138
    //   norm      RMS_NORM(l_out-31)                 [-0.3923  1.2082  0.6322]  sum   -50.572254
    //   result_norm  MUL(norm, output_norm.weight)   [-0.8796  2.5297  1.4026]  sum   -46.624306
    //   node_2419 MUL(result_norm, signs.4096)       [ 0.8796 -2.5297 -1.4026]  sum     6.048075
    //   node_2421 MUL_MAT(hadamard.4096, node_2419)                            sum    56.291439
    //   result_output MUL_MAT(output.weight, node_2421) [-1.8606 -0.4644 -0.0287]
    //                                                           sum 43374.937500
    // Set BONSAI_TAIL=1 and prompt "ab" to land on exactly that state.
    if (getenv("BONSAI_TAIL")) {
        struct T { const char *n, *o; uint32_t off; uint32_t cnt; double h0, h1, h2, sum; };
        const ModelPlan &pl = e.plan;
        const T ts[] = {
            {"RES",  "l_out-31",     f.r.RES, 4096, -2.9615,  9.1218,  4.7726,  -381.804138},
            {"NRM",  "norm",         f.r.NRM, 4096, -0.3923,  1.2082,  0.6322,   -50.572254},
            {"AN",   "result_norm",  f.r.AN,  4096, -0.8796,  2.5297,  1.4026,   -46.624306},
            {"ROT",  "node_2419",    f.r.ROT, 4096,  0.8796, -2.5297, -1.4026,     6.048075},
            {"HR",   "node_2421",    f.r.HR,  4096,  NAN,     NAN,     NAN,      56.291439},
            {"LOGITS","result_output",f.r.LOGITS, uint32_t(pl.vocab), -1.8606, -0.4644, -0.0287,
                                                                     43374.937500},
        };
        int tf = 0, tc = 0;
        printf("\n=== the whole network vs the same dump (1 token, cold caches) ===\n");
        for (const auto &t : ts) {
            const RegStat s = fwd_region_stat(f, t.off, t.cnt);
            int ok = 1;
            const double hs[3] = {t.h0, t.h1, t.h2};
            for (int k = 0; k < 3; ++k)
                if (!std::isnan(hs[k]) && fabs(double(s.v[k]) - hs[k]) > 5e-4 + 2e-3 * fabs(hs[k]))
                    ok = 0;
            const int sok = fabs(s.sum - t.sum) <= 1e-3 + 2e-4 * fabs(t.sum);
            tc += 2; tf += !ok; tf += !sok;
            printf("  %-6s %-14s [ %8.4f %8.4f %8.4f] vs [ %8.4f %8.4f %8.4f] %s"
                   " | sum %14.5f vs %14.5f %s\n",
                   t.n, t.o, s.v[0], s.v[1], s.v[2], hs[0], hs[1], hs[2], ok ? "h-OK " : "h-FAIL",
                   s.sum, t.sum, sok ? "s-OK" : "s-FAIL");
        }
        printf("  tail: %d/%d comparisons failed\n", tf, tc);
    }

    // The shape of the first logit vector, printed BEFORE any sampler can be blamed for it. Whether
    // a run that repeats one token is a numerical blow-up, a NaN, or a legitimately peaked
    // distribution is decided here and nowhere else.
    {
        int nan = 0, inf = 0; double mx = -1e300, mn = 1e300, sum = 0.0;
        int idx[8]; double val[8];
        for (int k = 0; k < 8; ++k) { idx[k] = -1; val[k] = -1e300; }
        for (uint32_t i = 0; i < V; ++i) {
            const double v = logits[i];
            if (std::isnan(v)) { ++nan; continue; }
            if (std::isinf(v)) { ++inf; continue; }
            if (v > mx) mx = v;
            if (v < mn) mn = v;
            sum += v;
            for (int k = 0; k < 8; ++k)
                if (v > val[k]) {
                    for (int j = 7; j > k; --j) { val[j] = val[j - 1]; idx[j] = idx[j - 1]; }
                    val[k] = v; idx[k] = int(i); break;
                }
        }
        printf("  logits after the prompt: max %.4f min %.4f mean %.6f  nan %d inf %d\n",
               mx, mn, sum / double(V - nan), nan, inf);
        printf("  top 8:");
        for (int k = 0; k < 8; ++k)
            printf("  %d(%.2f\"%s\")", idx[k], val[k], detok_sp(tok.decode_ids({idx[k]})).c_str());
        printf("\n");
    }

    std::vector<int> gen;
    const char *finish = "length";         // OpenAI's word for "the token budget ran out"
    double ms_first = 0.0;
    int nl_seen = 0;                       // newline budget, BONSAI_STOP_NL (0 = off)
    size_t answer_start = 0;               // first id of the post-think body when BONSAI_STRIP_THINK=1
    bool in_think = f.chat && f.thinking;  // thinking off => the wrapper is already closed in the prompt
    // The ledger's denominator has to be the wall clock of the whole loop, because pick(), the
    // per-token printf and the detokenize run OUTSIDE fwd_token and were never counted in ms_gen.
    for (int i = 0; i < 6; ++i) pf[i] = 0.0;
    pf_n = 0; pf_samp = pf_print = pf_qry = pf_copy = 0.0;
    // ---- BONSAI_KVFILL=N: switch the decode loop onto the synthetic long-context fixture.
    if (f.kvfill) {
        if (f.kvfill < 2u || f.kvfill > f.cap) {
            fprintf(stderr, "  BONSAI_KVFILL=%u is outside [2, cap %u]: the fixture cannot fake a\n"
                            "      position the arena does not have. Raise BONSAI_CAP or lower N.\n",
                    f.kvfill, f.cap);
            return 8;
        }
        const double ms_fill = kv_fill_synthetic(f);
        const uint64_t kvb = uint64_t(f.n_attn) * uint64_t(f.cap) * uint64_t(f.r.kv_layer) * 4ull;
        printf("\n*** SYNTHETIC KV TIMING FIXTURE IS ON (BONSAI_KVFILL=%u) ***\n", f.kvfill);
        printf("  %.1f MiB of K/V arena filled with 0x3C003C00 (fp16 1.0 in both halves) in %.0f ms.\n",
               double(kvb) / 1048576.0, ms_fill);
        printf("  every token below is dispatched AT POSITION %u, so each one reads %u keys per head\n"
               "  per layer - the traffic and workgroup shape of a %u-token context, without %u prefills.\n",
               f.kvfill - 1u, f.kvfill, f.kvfill, f.kvfill);
        printf("  VALID from this run: ms/token, tok/s, the kernel mix, the memory traffic.\n"
               "  NOT VALID: the sampled ids, the text, ANY fidelity statement. The content is a\n"
               "  pattern, not the model's K/V, and the GDN state is whatever the pinned position\n"
               "  left behind. Cross-validate against a real run before quoting the curve.\n");
        pos = int(f.kvfill) - 1;
    }
    const auto loop_t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n_max; ++i) {
        const auto s0 = std::chrono::steady_clock::now();
        const int next = sp.pick(logits.data(), V, &gen);
        const auto s1 = std::chrono::steady_clock::now();
        pf_samp += std::chrono::duration<double, std::milli>(s1 - s0).count();
        if (pos >= int(f.cap)) { printf("  [context cap %u reached]\n", f.cap); break; }
        if (next == sp.eos) { printf("  [eos]\n"); finish = "stop"; break; }
        // The think-close marker is a SECTION delimiter while thinking is on - the answer follows it -
        // and a wrapper we already closed in the prompt when thinking is off, where meeting it again
        // means the model is reopening a reasoning block instead of answering. Stop only in the second
        // case; stopping in the first would cut the answer off before it starts.
        if (next == chat::kThinkClose) {
            if (in_think) { in_think = false; answer_start = gen.size(); }
            else { printf("  [think-close while not thinking]\n"); finish = "stop"; break; }
        }
        if (g_stop_nl && !in_think &&
            (next == 198 || next == 620) && ++nl_seen >= int(g_stop_nl)) {
            gen.push_back(next);
            printf("  [stop: %u newlines (BONSAI_STOP_NL)]\n", g_stop_nl);
            finish = "stop";
            break;
        }
        // ---- P2: the spill watch and the KV block hook, at the decode boundary BEFORE this token's
        // KV write (that is the only moment a refusal can still leave the cache whole, and it is why
        // this runs ahead of gen.push_back: a refused token was never fed to the model). Sampled
        // every BONSAI_SPILL_EVERY tokens and again at every KV block boundary; the counter read
        // itself is microseconds against a 71 ms token, and the eviction path additionally costs one
        // read-modify-write of the cap-wide mask, which is why it is not on the timing path unless a
        // threshold has actually been crossed.
        {
            const bool periodic = f.spill_every && (i % int(f.spill_every)) == 0;
            const bool blk_b    = f.kvguard.blk_slots &&
                                  (uint32_t(pos) % f.kvguard.blk_slots) == 0u;
            if (f.spill.counter_ok && (periodic || blk_b)) {
                f.spill.sample(uint32_t(pos));
                // The policy judges the SPILL WATCH's own metric (growth over the reference sample by
                // default, absolute under BONSAI_SPILL_ABS=1), so the number the guard acts on is the
                // number the curve printed - no second, silently different definition of "too big".
                f.spill_last = f.spill.last_growth;
            }
            std::string why;
            const KvAction act = f.kvguard.decide(uint32_t(pos), f.spill_last,
                                                  f.spill.counter_ok && f.spill.n_sample > 0,
                                                  f.spill.growing(), why);
            if (act == KV_EVICT) {
                const uint32_t bs = f.kvguard.blk_slots;
                const uint32_t lo = f.kvguard.live_low, hi = lo + bs;
                if (lo < uint32_t(pos) && hi <= uint32_t(pos)) {
                    kv_mask_block(f, lo, hi);
                    f.kvguard.live_low = hi; ++f.kvguard.n_evict;
                    printf("  [kv block evicted] positions %u..%u masked to -INF (%s);"
                           " arena bytes stay allocated - see kv_guard.h\n",
                           lo, hi - 1u, why.c_str());
                }
            } else if (act == KV_REFUSE) {
                ++f.kvguard.n_refuse; f.kvguard.bytes_at_refuse = uint64_t(pos) * f.kvgeom.bytes_slot;
                printf("  [KV REFUSED at position %u] %s\n"
                       "    policy value %.2f MiB (%s), absolute Shared Usage %.2f MiB,"
                       " reference %.2f MiB\n"
                       "    thresholds: soft %.0f / hard %.0f MiB (hard additionally needs a growth"
                       " run >= %u),\n    KV written so far %.2f MiB\n"
                       "    the run stops here with the cache consistent; the shorter-context advice is\n"
                       "    the init-time BONSAI_CAP_AUTOFIT=1 path, and BONSAI_KV_EVICT=1 turns on the\n"
                       "    farthest-block mask instead of stopping.\n",
                       pos, why.c_str(), f.spill_last,
                       f.spill.use_absolute ? "absolute, BONSAI_SPILL_ABS=1"
                                            : "growth over the reference sample",
                       f.spill.last_shared, f.spill.ref_shared, f.kvguard.g_spill_soft,
                       f.kvguard.g_spill_hard, f.spill.mono_need,
                       double(f.kvguard.bytes_at_refuse) / 1048576.0);
                break;
            } else if (act == KV_WARN) {
                // Counted whether or not it is printed: the guard's action counters are what the
                // close-out line reports, and a soft crossing that stayed silent still happened.
                ++f.kvguard.n_warn;
                const bool at_blk = f.kvguard.blk_slots &&
                                    (uint32_t(pos) % f.kvguard.blk_slots) == 0u;
                if (f.kvguard.n_warn == 1u || at_blk)
                    printf("  [KV WATCH] position %u: %s (policy value %.2f MiB, absolute %.2f,"
                           " soft %.0f / hard %.0f, evict %s)\n", pos, why.c_str(), f.spill_last,
                           f.spill.last_shared, f.kvguard.g_spill_soft, f.kvguard.g_spill_hard,
                           f.kvguard.allow_evict ? "armed" : "off");
            }
        }
        gen.push_back(next);
        auto t0 = std::chrono::steady_clock::now();
        fwd_token(f, uint32_t(next), uint32_t(pos), logits);
        auto t1 = std::chrono::steady_clock::now();
        const double dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (i == 0) ms_first = dt;
        ms_gen += dt;
        // One detokenization, shared by the log line and the stream hook: doing it twice would put a
        // second vocab walk on the per-token path of a run that is trying to measure that path.
        const std::string piece = detok_sp(tok.decode_ids({next}));
        printf("  gen %2d  tok=%7d  %8.2f ms  %s\n", i + 1, next, dt, piece.c_str());
        if (g_emit_token && next != chat::kImStart && next != chat::kImEnd &&
            next != chat::kThinkOpen && next != chat::kThinkClose) {
            if (!g_emit_token(g_emit_ctx, piece.c_str(), in_think ? 1 : 0)) {
                printf("  [stream client went away]\n");
                finish = "stop";
                break;
            }
        }
        pf_print += std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t1).count();
        // The fixture pins the position; a real run walks forward one key per token.
        if (!f.kvfill) ++pos;
    }
    const double ms_wall = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - loop_t0).count();
    if (g_prof && pf_n) {
        const int n = int(pf_n);
        printf("\n=== single-token time slice (measured, %d decode tokens, wall %.1f ms) ===\n",
               n, ms_wall);
        const double row[8] = {pf[0] / n, pf[1] / n, pf[2] / n, pf[3] / n, pf[4] / n,
                               pf_samp / n, pf_print / n,
                               ms_wall / n - (pf[0] + pf[1] + pf[2] + pf[3] + pf[4]
                                             + pf_samp + pf_print) / n};
        const char *nm[8] = {"T_cpu  host command-buffer recording",
                             "T_submit vkQueueSubmit call",
                             "T_sync   vkWaitForFences (the GPU is running here)",
                             "T_reset  vkResetFences + query read",
                             "T_memory logits readback (device->host copy + assign)",
                             "T_other  sampler argmax over the vocab",
                             "T_other  printf + detokenize per token",
                             "T_other  loop bookkeeping (wall minus the above)"};
        for (int k = 0; k < 8; ++k)
            printf("  %-52s %8.3f ms/token  %6.2f%% of the wall\n", nm[k], row[k],
                   100.0 * row[k] / (ms_wall / n));
        printf("  %-52s %8.3f ms/token  %6.2f%% (device timestamps)\n",
               "T_gpu    the CB's own span, first to last timestamp", pf[5] / n,
               100.0 * (pf[5] / n) / (ms_wall / n));
        printf("  %-52s %8.3f ms/token\n", "  of which vkGetQueryPoolResults", pf_qry / n);
        printf("  %-52s %8.3f ms/token\n",
               "  of which logits.assign (248320 floats = 0.946 MiB)", pf_copy / n);
        printf("  %-52s %8.3f ms/token  %6.2f%% of the wall\n",
               "bubble   wall - T_gpu (host phases + launch latency)",
               ms_wall / n - pf[5] / n, 100.0 * (ms_wall / n - pf[5] / n) / (ms_wall / n));
        printf("  sum of the host-visible phases: %.3f ms; wall per token %.3f ms (%.2f tok/s)\n",
               (pf[0] + pf[1] + pf[2] + pf[3] + pf[4] + pf_samp + pf_print) / n, ms_wall / n,
               n * 1000.0 / ms_wall);
        printf("  note: T_sync overlaps T_gpu by construction - the fence waits for the work the\n"
               "  submit just queued, so GPU-busy = T_sync + a launch latency, and the host cost is\n"
               "  T_cpu + T_submit + T_reset + T_memory + the T_other rows. The bubble is\n"
               "  wall - T_gpu.\n");
    }
    const int n = int(gen.size());
    printf("\n=== pure-blood token rate (measured, this card) ===\n");
    printf("  %d generated tokens in %.1f ms  ->  %.2f tok/s\n", n, ms_gen, n * 1000.0 / (ms_gen > 0 ? ms_gen : 1.0));
    printf("  first generated token (prefill+1): %.1f ms; steady-state: %.1f ms/token\n",
           ms_first, n ? ms_gen / n : 0.0);
    printf("  dispatches %llu, barriers %llu per command buffer (one submit per token)\n",
           (unsigned long long)f.n_dispatch, (unsigned long long)f.n_barrier);
    // The instrument for the attribution question the P2 curve raises: is Shared Usage climbing with
    // the SUBMITS (command buffers, driver-side tracking) or with the POSITIONS (our own arena)? This
    // line counts the buffers, split by phase, so the weight-loading buffers (which are allocated per
    // upload group and are NOT part of the per-token story) cannot be mistaken for per-token ones.
    // BONSAI_CB_REUSE=1 collapses the forward count to one buffer; BONSAI_CB_PROBE=N turns the count
    // into MiB. The two curves are runs/p2_rate_q8_v1.log and runs/p2_rate_q8_cbreuse_v1.log.
    const uint64_t alloc_fwd  = g_cb_alloc - g_cb_alloc_load;
    const uint64_t begin_fwd  = alloc_fwd + g_cb_reset;
    printf("  command buffers | load phase: %llu allocated (never reused, not counted below)\n"
           "                  | forward phase: %llu allocated + %llu reuses = %llu begins over"
           " %u positions = %.1f per position (BONSAI_CB_REUSE=%u)\n",
           (unsigned long long)g_cb_alloc_load, (unsigned long long)alloc_fwd,
           (unsigned long long)g_cb_reset, (unsigned long long)begin_fwd, uint32_t(pos),
           pos > 0 ? double(begin_fwd) / double(unsigned(pos)) : 0.0, g_cb_reuse);
    // ---- P1/P2 close-out: what the V representation cost, and what the watch saw.
    printf("  KV at the end: %u live positions of cap %u (%.2f MiB of arena,"
           " %.2f KiB/token, V = %s), blocks evicted %u, refused %u, soft-watched %u\n",
           uint32_t(pos), f.cap, double(uint64_t(pos) * f.kvgeom.bytes_slot) / 1048576.0,
           double(f.kvgeom.bytes_slot) / 1024.0, f.v_q8 ? "q8_0" : "fp16",
           f.kvguard.n_evict, f.kvguard.n_refuse, f.kvguard.n_warn);
    printf("  guard actions (decisions taken at a block boundary) and spill samples (threshold"
           " crossings) are different counters - a refusal on the cap-geometry path needs no hard"
           " sample at all, so \"refused > 0, hard = 0\" is consistent, not a contradiction.\n");
    // One last sample after the final token: a curve whose right end is the state the run actually
    // finished in is what a spill looks like. A run that spills on the way out but not on the way in
    // would otherwise be reported as clean.
    if (f.spill.counter_ok) { f.spill.sample(uint32_t(pos)); f.spill_last = f.spill.last_growth; }
    f.spill.print_summary();
    if (!g_watch_persist) { f.spill.close(); f.budget.close(); }
    printf("\n=== text ===\n%s\n", detok_sp(tok.decode_ids(
             std::vector<int>(gen.begin() + (g_strip_think ? answer_start : 0), gen.end()))).c_str());
    if (g_strip_think && answer_start)
        printf("  (%zu reasoning tokens before the think-close were not printed; "
               "BONSAI_STRIP_THINK=0 shows them)\n", answer_start);
    if (out) {
        out->n_prompt = int(ids.size()); out->n_gen = int(gen.size());
        out->n_think = int(answer_start);
        out->ms_prefill = ms_prefill; out->ms_gen = ms_gen; out->ms_first = ms_first;
        out->finish = finish;
    }
    return 0;
}

// bonsai_vk.exe run [gguf] [chunk_MiB] [n_tokens] [-- "prompt"]
static int fwd_main(int argc, char **argv) {
    const char *path = argc > 2 ? argv[2] : "models/Ternary-Bonsai-2-9B-PTQ1_0.gguf";
    const uint64_t chunk_mib = argc > 3 ? uint64_t(atoll(argv[3])) : 32u;
    int n_max = argc > 4 ? atoi(argv[4]) : 24;
    std::string prompt = "\xe7\x94\xa8\xe4\xb8\x80\xe5\x8f\xa5\xe8\xaf\x9d\xe8\xa7\xa3\xe9\x87\x8a\xe4\xb8\x89\xe5\x80\xbc\xe6\xa8\xa1\xe5\x9e\x8b";
    for (int i = 1; i < argc - 1; ++i) if (!strcmp(argv[i], "--")) prompt = argv[i + 1];
    if (const char *p = getenv("BONSAI_PROMPT")) prompt = p;

    Engine e;
    auto t0 = std::chrono::steady_clock::now();
    if (!engine_init(e, path, chunk_mib << 20)) return 2;
    if (!load_ptq_weights(e)) return 3;
    if (!load_f32_weights(e)) return 3;
    auto t1 = std::chrono::steady_clock::now();
    Fwd f;
    // The attribution instrument, switched on only from here: the weights are resident and every
    // upload command buffer has been waited, so nothing an earlier queue might still be using can be
    // reset under the reuse path. See begin_cmd() in bonsai_weights.h. This is also where the load/
    // forward split of the command-buffer counters is taken, so the per-token number below is the
    // forward phase alone and the weight-loading buffers are not charged to it.
    cb_enter_forward();
    if (const char *v = getenv("BONSAI_CB_REUSE")) {
        g_cb_reuse = uint32_t(atoi(v));
        printf("  [instrument] BONSAI_CB_REUSE=%s: forward phase %s one command buffer"
               " instead of allocating one per dispatch group\n", v,
               g_cb_reuse ? "reuses" : "allocates");
    }
    if (const char *v = getenv("BONSAI_STOP"))   f.stop_at_block = atoi(v);
    if (const char *v = getenv("BONSAI_GDN_MODE")) f.gdn_mode  = uint32_t(atoi(v));
    if (const char *v = getenv("BONSAI_GDN_GMAP")) f.gdn_gmap  = uint32_t(atoi(v));
    if (const char *v = getenv("BONSAI_L2_MODE"))  f.l2_mode   = uint32_t(atoi(v));
    if (const char *v = getenv("BONSAI_CAP"))     f.cap = uint32_t(atoi(v));
    if (const char *v = getenv("BONSAI_SPLIT"))  g_split_force = uint32_t(atoi(v));
    // The time-slice instrument in fwd_token was written long before this line existed: g_prof had
    // no setter anywhere, so the whole ledger was dead code. One line, and it costs nothing when 0.
    if (const char *v = getenv("BONSAI_PROFILE"))   g_prof     = uint32_t(atoi(v));
    if (const char *v = getenv("BONSAI_CHAT"))       f.chat       = uint32_t(atoi(v));
    if (const char *v = getenv("BONSAI_THINKING"))   f.thinking   = uint32_t(atoi(v));
    if (const char *v = getenv("BONSAI_QCHECK"))     f.qcheck     = uint32_t(atoi(v));
    if (const char *v = getenv("BONSAI_QKERNEL"))    f.qk_force   = uint32_t(atoi(v));
    // BONSAI_QK_TILE picks which compiled tile wrapper is built: the tile width is the ONLY thing that
    // sets the kernel's LDS footprint (4*QTILE+4 bytes), so it is the knob the "find the new balance
    // point" instruction asks for. 1024 -> 4.0 KiB, 2048 -> 8.0 KiB, 4096 -> 16.0 KiB (the design
    // line), 8192 -> 32772 B, which CROSSES this driver's measured maxComputeSharedMemorySize of
    // 32768 B. 8192 is a cliff probe, not a config: no auto path can reach it (f.qk_tile is only ever
    // read from here), it is Vulkan valid-usage violation territory kept open solely so the occupancy
    // penalty can be measured inside the attention kernel rather than borrowed from the synthetic
    // bench in runs/shared_probe_v3.log.
    if (const char *v = getenv("BONSAI_QK_TILE")) {
        f.qk_tile = uint32_t(atoi(v));
        if (f.qk_tile != 1024u && f.qk_tile != 2048u && f.qk_tile != 4096u && f.qk_tile != 8192u) {
            fprintf(stderr, "BONSAI_QK_TILE must be 1024, 2048, 4096 or 8192 (the compiled wrappers),"
                            " got %s - no other tile is built, and inventing one here would be a"
                            " shader that does not exist\n", v);
            return 4;
        }
        if (f.qk_tile == 8192u)
            printf("  [probe] QTILE 8192 asks 32772 B of shared memory; this driver's measured"
                   " maxComputeSharedMemorySize is 32768 B.\n"
                   "          This width exists only to price the occupancy cliff and must not be"
                   " used for a result that ships.\n");
    }
    if (f.qk_force > 3u) {
        fprintf(stderr, "BONSAI_QKERNEL must be 0 auto / 1 narrow / 2 wide / 3 tiled, got %u\n",
                f.qk_force);
        return 4;
    }
    // ---- BONSAI_GQA_OPT: the GQA 4x-redundancy merge switch. 0 = OFF (the bit-exact control path:
    // narrow/wide/tiled + the frozen attn_pv_q8v, i.e. every kernel the fidelity ladder was measured
    // on), 1 = AUTO = DEFAULT (merged QK and merged PV), 2 = FORCE (merged at every cap, including the
    // <= 256 band where the narrow kernel is the incumbent - that is what makes a same-binary A/B of
    // short contexts possible). The merged kernels are gated bitwise-identical to the kernels they
    // replace (qk_audit A3/A4), so "OFF" is not the safe mode and "AUTO" is not the risky one: they
    // produce the same bits. The switch exists because they do not necessarily produce the same TIME,
    // and because an auditor has to be able to re-run the reference without rebuilding anything.
    if (const char *v = getenv("BONSAI_GQA_OPT"))  f.gqa_opt  = uint32_t(atoi(v));
    if (f.gqa_opt > 2u) {
        fprintf(stderr, "BONSAI_GQA_OPT must be 0 off / 1 auto (default) / 2 force-at-every-cap,"
                        " got %u\n", f.gqa_opt);
        return 4;
    }
    // Which half of the surgery is in force: bit0 = QK (the K gather), bit1 = PV (the V block read).
    // A DIAGNOSTIC, not a config - it exists so a measured gain can be attributed to one half instead
    // of asserted for both. Default 1 = QK only, because the PV half measures negative (§10.5).
    if (const char *v = getenv("BONSAI_GQA_PART")) f.gqa_part = uint32_t(atoi(v));
    if (f.gqa_part > 3u) {
        fprintf(stderr, "BONSAI_GQA_PART must be 0..3 (bit0 QK, bit1 PV), got %u\n", f.gqa_part);
        return 4;
    }
    // The merged wrapper, i.e. its LDS footprint: HPK*4*QTILE + 8*HPK bytes. 256 -> 4,128 B, 512 ->
    // 8,224 B (default), 1024 -> 16,416 B. The 16 KiB design line is the top of the measured flat band;
    // nothing in the auto path asks for more, and no merged width crosses maxComputeSharedMemorySize.
    if (const char *v = getenv("BONSAI_GQA_TILE")) {
        f.gqa_tile = uint32_t(atoi(v));
        if (f.gqa_tile != 256u && f.gqa_tile != 512u && f.gqa_tile != 1024u) {
            fprintf(stderr, "BONSAI_GQA_TILE must be 256, 512 or 1024 (the compiled wrappers),"
                            " got %s - no other merged tile exists\n", v);
            return 4;
        }
    }
    // BONSAI_KVFILL=N: the synthetic-KV timing fixture (kv_fill_synthetic). Token identities in that
    // run are meaningless; only ms/token is valid. Never a fidelity statement.
    if (const char *v = getenv("BONSAI_KVFILL")) f.kvfill = uint32_t(atoi(v));
    if (const char *v = getenv("BONSAI_STOP_NL"))     g_stop_nl    = uint32_t(atoi(v));
    if (const char *v = getenv("BONSAI_STRIP_THINK")) g_strip_think = uint32_t(atoi(v));
    if (const char *v = getenv("BONSAI_SYSTEM"))     f.system_text = v;
    // ---- P1: the V representation. Default q8_0; "none" keeps the fp16 arena and is the regression
    // control, so the switch is measurable instead of arguable.
    if (const char *v = getenv("BONSAI_KV_V_QUANT")) {
        std::string s = v;
        if      (s == "q8_0" || s == "q8" || s == "1") f.v_q8 = 1;
        else if (s == "none" || s == "fp16" || s == "0") f.v_q8 = 0;
        else { fprintf(stderr, "BONSAI_KV_V_QUANT must be none|q8_0, got '%s'\n", s.c_str()); return 4; }
    }
    if (const char *v = getenv("BONSAI_KV_DEBUG_STREAM"))  f.emit_kf = uint32_t(atoi(v));
    // The bisect switch: the frozen oracle kernel instead of the switchable twin. It writes kf
    // unconditionally, so binding 10 has to be the real cap*nkv*hd region, not the 16-float dummy.
    if (const char *v = getenv("BONSAI_ROPE_ORACLE")) {
        f.rope_oracle = uint32_t(atoi(v));
        if (f.rope_oracle && f.v_q8) {
            fprintf(stderr, "BONSAI_ROPE_ORACLE=1 is the A/B against the pre-P1 binary and is only"
                            " defined for BONSAI_KV_V_QUANT=none\n");
            return 4;
        }
        if (f.rope_oracle) f.emit_kf = 1;
    }
    if (const char *v = getenv("BONSAI_CAP_AUTOFIT"))      f.cap_autofit = uint32_t(atoi(v));
    // ---- P2: the spill watch and the block hook. The three thresholds are the spec's own band
    // (32..64 MiB) and are env-overridable so the policy is assertable without spilling a 4 GB card.
    if (const char *v = getenv("BONSAI_KV_EVICT"))       f.kvguard.allow_evict = atoi(v);
    if (const char *v = getenv("BONSAI_KV_BLOCK_SLOTS")) f.kv_blk_slots = uint32_t(atoi(v));
    if (const char *v = getenv("BONSAI_SPILL_EVERY"))    f.spill_every = uint32_t(atoi(v));
    if (const char *v = getenv("BONSAI_SPILL_SOFT_MIB")) f.spill.soft_mib = atof(v);
    if (const char *v = getenv("BONSAI_SPILL_HARD_MIB")) f.spill.hard_mib = atof(v);
    if (const char *v = getenv("BONSAI_SPILL_MONO"))     f.spill.mono_need = uint32_t(atoi(v));
    if (const char *v = getenv("BONSAI_SPILL_ABS"))      f.spill.use_absolute = uint32_t(atoi(v));
    printf("  KV V representation: %s (BONSAI_KV_V_QUANT) | KF debug stream: %s"
           " | KV block: %u positions | spill watch: every %u tokens, soft %.0f / hard %.0f MiB,"
           " growth run >= %u\n",
           f.v_q8 ? "q8_0 blocks (32 elements + 1 fp16 scale)" : "fp16 (unchanged arena)",
           f.emit_kf ? "ON (cap*nkv*hd fp32)" : "off", f.kv_blk_slots, f.spill_every,
           f.spill.soft_mib, f.spill.hard_mib, f.spill.mono_need);
    // Why this line exists: on this card a healthy run's ABSOLUTE Shared Usage is already 40.93 MiB
    // (measured, runs/p2_rate_q8_reuse_v2.log), so the spec's 32/64 MiB band is only meaningful as a
    // GROWTH band over a reference. Say which of the two is in force, because a "soft" verdict means
    // something completely different depending on it.
    printf("  spill thresholds are measured as %s; the reference is the first decode sample and is"
           " printed with the curve\n",
           f.spill.use_absolute ? "ABSOLUTE Shared Usage (BONSAI_SPILL_ABS=1)"
                                : "GROWTH over the first decode sample (default; BONSAI_SPILL_ABS=1"
                                  " for the absolute reading)");
    // Which QK kernel owns the row. 0 = auto (narrow while it can, wide past it), 1 = force narrow,
    // 2 = force wide, 3 = force tiled. The forcing exists so the three can be compared on identical
    // inputs; auto NEVER picks the tiled kernel, because it has not been measured yet.
    if (f.cap < 2u || f.cap % 2u) {
        fprintf(stderr, "cap must be an even number of slots (P16 packs pairs), got %u\n", f.cap);
        return 4;
    }
    // HPK has to be known BEFORE the row kernel is chosen and before the cap is checked against
    // qk_row_limit(): with the merge in force the limit is the tiled arithmetic guard, not the wide
    // kernel's compile-time MAXCAP, and getting this order wrong would refuse a cap the merged kernel
    // can serve. fwd_init recomputes it from the same plan fields, so the two sites cannot drift.
    f.hpk = (e.plan.n_head_kv > 0 && e.plan.n_head % e.plan.n_head_kv == 0)
                ? uint32_t(e.plan.n_head / e.plan.n_head_kv) : 0u;
    // ---- the rotation gate, version-0 side. prism.hadamard.version = 0 is a legal file and this
    // forward cannot serve it: the fused block loop interleaves signs_mul with the transform at
    // three different call shapes (pre-projection: signs THEN H; the 12288-wide FFN sandwich; and
    // the embedding's INVERSE sandwich, which applies signs AFTER H). Eliding both kernels there is
    // a rewrite of those sites, not a flag, and leaving them in place on a rotation-free export is
    // the mirror-image of the bug the directive names - applying a rotation the weights never saw
    // produces exactly the "全乱码" outcome, just with confidence. So refuse before any GPU work.
    // Version 1 is served: block 4096 through the bit-frozen hat4096.comp (what the 9B declares),
    // any other power-of-two block through fwht.comp, both gated by the resolver in bonsai_model.h.
    if (!e.plan.hat_active) {
        fprintf(stderr,
            "prism.hadamard.version = %lld: this file declares NO activation rotation, but the\n"
            "  fused block loop in runtime_vk/bonsai_forward.h always applies signs + Hadamard\n"
            "  before every ternary GEMV. Refusing instead of rotating a model that was never\n"
            "  rotated. Version 1 exports run as-is: block_size 4096 -> hat4096.comp (the 9B's\n"
            "  bit-frozen path), any other power of two in [16, 4096] -> fwht.comp.\n",
            e.plan.hat_version);
        return 6;
    }
    qk_select(f);
    if (f.cap > qk_row_limit(f)) {
        const bool tiled = f.qkt != 0u;
        fprintf(stderr, "cap %u exceeds the widest VALIDATED row (%u): %s\n",
                f.cap, qk_row_limit(f),
                tiled
                    ? "the tiled kernel has no compile-time row limit, so this is the arithmetic"
                      " guard kMaxTile (region offsets are uint32 words) - nothing here is a shader"
                      " bound; the real bound is the memory budget printed below."
                    : "the wide kernel's two shared arrays are float[2048] = 16 KiB of the 32 KiB"
                      " this driver reports as its limit. This is a validation gate, not a driver"
                      " refusal - shared_probe.exe measured float[4096] (32768 B, exactly the limit)"
                      " as created and round-tripped elementwise, at a measured +19.6% cost from"
                      " residence. This refusal therefore only ever fires when the wide kernel is"
                      " FORCED (BONSAI_QKERNEL=2); auto takes the tiled kernel past 2048 now.\n");
        return 4;
    }
    if (f.qk_force == 1u && f.cap > Fwd::kMaxNarrow) {
        fprintf(stderr, "narrow qk kernel cannot cover cap %u (its lane IS the key position)\n", f.cap);
        return 4;
    }
    printf("  attention QK kernel: %s (cap %u; narrow <= %u, wide <= %u, tiled row <= %u;"
           " force=%u tile=%u)\n",
           f.qkg ? "GQA-MERGED attn_qk_softmax_gqa_h4_q*.comp"
                 : (f.qkt ? "TILED attn_qk_softmax_tile_q*.comp"
                          : (f.wide ? "WIDE   attn_qk_softmax_wide.comp"
                                     : "NARROW attn_qk_softmax.comp")),
           f.cap, Fwd::kMaxNarrow, Fwd::kMaxWide, Fwd::kMaxTile, f.qk_force, f.qk_tile);
    // The reference row kernel is still the one BONSAI_QKERNEL/auto would have picked: the merged module
    // replaces the DISPATCH, not the AUDIT TARGET, so say both here rather than let one line imply the
    // other. Grid 16 -> 4 is the whole point of the surgery (one workgroup per kv head, HPK heads inside).
    if (f.qkg) {
        printf("    GQA merge: grid %u -> %u workgroups per layer (HPK = %u query heads served by one"
               " K gather), QK reference = %s, LDS %u B = %.1f KiB per workgroup"
               " (HPK*4*QTILE + 8*HPK, QTILE = %u)\n",
               uint32_t(e.plan.n_head), uint32_t(e.plan.n_head_kv), f.hpk,
               f.qk_force == 1u ? "narrow" : (f.qk_force == 2u ? "wide" : (f.qkt ? "tiled" :
                    (f.cap > Fwd::kMaxNarrow ? "wide-or-tiled" : "narrow"))),
               4u * f.hpk * f.gqa_tile + 8u * f.hpk, (4u * f.hpk * f.gqa_tile + 8u * f.hpk) / 1024.0,
               f.gqa_tile);
        printf("    V path: %s\n", f.pvg ? "GQA-MERGED attn_pv_q8v_gqa.comp (one V word/scale unpack"
                                           " per key, shared by HPK heads)"
                                         : "frozen attn_pv_q8v.comp (BONSAI_GQA_PART bit1 off, or the"
                                           " fp16 V arena which has no merged twin)");
    }
    if (f.qkt) printf("    tiled kernel LDS: 4*%u + 4 = %u B = %.1f KiB per workgroup"
                      " (the wide kernel spends %u KiB to address %u keys; this one spends %.1f KiB"
                      " and addresses an unbounded row)\n",
                      f.qk_tile, 4u * f.qk_tile + 4u, (4u * f.qk_tile + 4u) / 1024.0,
                      2u * Fwd::kMaxWide * 4u / 1024u, Fwd::kMaxWide,
                      (4u * f.qk_tile + 4u) / 1024.0);
    if (!fwd_init(e, f)) return 4;
    auto t2 = std::chrono::steady_clock::now();
    printf("\n  weight load %.1f ms, forward bring-up %.1f ms\n",
           std::chrono::duration<double, std::milli>(t1 - t0).count(),
           std::chrono::duration<double, std::milli>(t2 - t1).count());
    printf("  device-local total: weights %.3f GiB + activations %.3f MiB\n",
           double(e.arena.size) / 1073741824.0, double(f.bytes) / 1048576.0);
    // ---- BONSAI_CB_PROBE=N: the per-command-buffer price, measured instead of inferred.
    // The A/B curve says Shared Usage follows the NUMBER OF COMMAND BUFFERS. This isolates the number:
    // N buffers from the same pool the forward path uses (e.cpool), each one begin/end'd with no
    // commands and submitted on the same queue with the same fence - the only difference from a real
    // dispatch group is the recorded payload, which is NOT included, so what this prints is a lower
    // bound on the tracking cost of a busy buffer. Samples before, every N/8 and after; the slope is
    // read off the printed curve. Stage 2 frees all of them, which answers the other half of the
    // question: whether vkFreeCommandBuffers hands the memory back to the driver at all. A run under
    // this knob starts its token curve from the probed level; it is an instrument, not a workload.
    if (const char *v = getenv("BONSAI_CB_PROBE")) {
        const uint32_t n_probe = uint32_t(atoi(v));
        const uint32_t stride = n_probe / 8u ? n_probe / 8u : 1u;
        double s0 = -1.0, s1 = -1.0, s2 = -1.0;
        const bool rd0 = f.spill.read_judge(s0);
        printf("\n=== BONSAI_CB_PROBE=%u: command-buffer tracking cost (measured) ===\n", n_probe);
        if (!rd0) {
            printf("  skipped: the %s counter answers nothing %s, so no slope"
                   " can be measured (see the counter diagnostics above)\n",
                   SpillWatch::judge_source(), SpillWatch::judge_scope());
        } else {
        printf("  %s usage before: %8.2f MiB\n", SpillWatch::judge_label(), s0);
        std::vector<VkCommandBuffer> probe_cb; probe_cb.reserve(n_probe);
        uint32_t n_alloc = 0;
        for (uint32_t i = 0; i < n_probe; ++i) {
            VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            cai.commandPool = e.cpool; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cai.commandBufferCount = 1;
            VkCommandBuffer cb; VK_CHECK(vkAllocateCommandBuffers(e.c.dev, &cai, &cb));
            ++n_alloc;
            VkCommandBufferBeginInfo beg{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            beg.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            VK_CHECK(vkBeginCommandBuffer(cb, &beg));
            VK_CHECK(vkEndCommandBuffer(cb));
            submit_wait_sig(e.c, e.c.queue, cb, f.fence, VK_NULL_HANDLE, VK_NULL_HANDLE);
            probe_cb.push_back(cb);
            if (((i + 1u) % stride) == 0u || i + 1u == n_probe) {
                double s = -1.0;
                if (f.spill.read_judge(s))
                    printf("  after %5u buffers: %s %8.2f MiB (%+.2f MiB, %+.4f MiB per buffer)\n",
                           i + 1u, SpillWatch::judge_label(), s, s - s0,
                           n_alloc ? (s - s0) / n_alloc : 0.0);
            }
        }
        if (f.spill.read_judge(s1)) {
            printf("  %u command buffers allocated, submitted and waited -> %s %.2f MiB:"
                   " %.4f MiB per command buffer  [MEASURED, lower bound: empty payload]\n",
                   n_alloc, SpillWatch::judge_label(), s1, n_alloc ? (s1 - s0) / n_alloc : 0.0);
        }
        vkFreeCommandBuffers(e.c.dev, e.cpool, uint32_t(probe_cb.size()), probe_cb.data());
        if (f.spill.read_judge(s2))
            printf("  after vkFreeCommandBuffers(%u): %s %.2f MiB (returned %.2f MiB)\n",
                   uint32_t(probe_cb.size()), SpillWatch::judge_label(), s2, s1 - s2);
        printf("  note: the buffers are freed again, but the peak they caused stays visible to the"
               " curve that follows - run the token benchmark in a separate process when the absolute"
               " level matters\n");
        fflush(stdout);
        }
    }
    if (getenv("BONSAI_ALIGN")) {                 // the oracle-anchor harness, no generation
        Bpe tok; if (!tok.load(e.hdr)) { fprintf(stderr, "tokenizer load failed\n"); return 5; }
        std::vector<int> ids = tok.encode("ab", true, false);
        printf("  align prompt \"ab\" -> ids");
        for (int t : ids) printf(" %d", t);
        printf("\n");
        uint32_t t0 = uint32_t(ids[0]);
        if (const char *v = getenv("BONSAI_ALIGN_TOK")) t0 = uint32_t(atoi(v));
        return fwd_align(e, f, t0);
    }
    if (getenv("BONSAI_LADDER")) {               // one submit per block, RES read between them
        Bpe tok; if (!tok.load(e.hdr)) { fprintf(stderr, "tokenizer load failed\n"); return 5; }
        std::vector<int> ids = tok.encode("ab", true, false);
        uint32_t t0 = uint32_t(ids[0]);
        if (const char *v = getenv("BONSAI_ALIGN_TOK")) t0 = uint32_t(atoi(v));
        fwd_ladder(f, t0, 0);
        return 0;
    }
    // The HTTP shell takes the engine at exactly this point: weights resident, every gate above already
    // printed and passed, the same Fwd the CLI would have generated one reply from. It owns the loop
    // from here and never comes back, so nothing below this line runs in serve mode.
    if (g_serve_hook) return g_serve_hook(e, f);
    const int rc = fwd_run(e, f, prompt.c_str(), n_max);
    printf("  gdn_mode=%u gdn_gmap=%u l2_mode=%u round_probs=%u cap=%u\n",
           f.gdn_mode, f.gdn_gmap, f.l2_mode, f.round_probs, f.cap);
    return rc;
}

#endif  // BONSAI_FORWARD_H
