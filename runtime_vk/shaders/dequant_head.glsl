// dequant_head.glsl - the prelude the official PrismML/llama.cpp vulkan-shaders/dequant_ptq1_0.comp
// expects to be prepended for it at build time (ggml-vulkan generates it from its own templates).
//
// We write it ourselves for ONE reason: to be able to compile the OFFICIAL shader body verbatim and
// run it on this card as the reference oracle for align_dequant (the dequant-fidelity test). This
// file is not part of the runtime path - bonsai_vk.exe never loads it. It only supplies:
//   * the 28-byte PTQ1_0 block struct (128 weights, 1.75 bpw),
//   * the output element type D_TYPE,
//   * the push constant the body reads (p.nel = number of *weights*, not blocks).
// It deliberately contains no arithmetic, so it cannot bias what the test measures.
#ifndef BONSAI_DEQUANT_HEAD_GLSL
#define BONSAI_DEQUANT_HEAD_GLSL

#extension GL_EXT_control_flow_attributes : require   // for the [[unroll]] the official body uses
// uint8_t / uint16_t are not keywords in plain Vulkan GLSL: the official struct in ggml-vulkan is
// generated with explicit 8/16-bit scalar types enabled. Without these two lines glslang stops at
// "unexpected IDENTIFIER" on the struct body.
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
// The scale is a REAL 16-bit float type, not its bits. Measured, not guessed: with uint16_t the
// official body's `const float d = float(data_a[ib].d)` widens the *bit pattern* instead of the
// value, and the fidelity test reported max|err| 10916.9 (an fp16 bit pattern of that magnitude,
// while every real block scale here is 0.0091..0.0519) on exactly the non-zero weights, with only
// the 4194 zero weights agreeing. float16_t is also what the fork itself uses for half in shaders -
// ggml-vulkan.cpp line 1483: "FLOAT_TYPE in the shader is float16_t with fp16 support, otherwise float".
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

// float is what ggml asks for when it builds the f32 dequant pipeline
// (pipeline_dequant[GGML_TYPE_PTQ1_0], "dequant_ptq1_0", 2 bindings, 5*4 B push constants).
#ifndef D_TYPE
#define D_TYPE float
#endif

// Ground truth from the fork's own kernel comments (mul_mat_vec_ptq1_0.comp) and confirmed against
// the file bytes by our GGUF reader: one block = 7 x 32-bit words = 28 B for 128 weights.
//   qs[0..15]   words 0..3 : byte j, trit t -> element 16*t + j
//   qs[16..23]  words 4..5 : byte j, trit t -> element 80 + 8*t + (j-16)
//   qh[0..1]    word 6     : byte h, trit t -> element 120 + 2*t + h
//   d           word 6 upper half : fp16 block scale, bytes 26..27
struct block_ptq1_0 {
    uint8_t   qs[24];
    uint8_t   qh[2];
    float16_t d;        // the fp16 block scale AS a half value; the body floats it (see above)
};

layout (push_constant) uniform Push {
    uint nel;              // total weights in this buffer
    uint pad0, pad1, pad2, pad3;   // keeps the 5 x u32 push-constant size the fork's host uses
} p;

#endif
