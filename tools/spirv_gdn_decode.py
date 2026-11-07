#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spirv_gdn_decode.py -- extract and decode the Gated Delta Net Vulkan kernel.

WHY THIS EXISTS
---------------
The GDN kernel we must re-implement is shipped by the official Vulkan backend as an
embedded SPIR-V byte array.  This tool locates that array, validates it, writes raw
.spv files and prints a decoder listing good enough to read the *math* (order of
operations, where each scalar enters, how each array is indexed).

TWO INTAKE PATHS (auto-detected per --source file)
--------------------------------------------------
1. C-array text  : `static const uint32_t  foo_data[] = {0x07230203, ...};`
                   also handles `uint8_t`/`char` byte arrays (4 bytes -> 1 LE word)
                   and `\\xNN` string-literal blobs.
2. Binary blob   : PE / ELF / raw dump.  Scans for the SPIR-V magic word
                   0x07230203 and accepts a candidate only if it parses as a module.

NOTE ON THIS REPO (measured, not assumed): the checked-in `_ggml-vulkan.cpp` is only
the host half of the backend.  It *references* `gated_delta_net_f32_{,nocluster_,shmem_}data`
(lines ~3761-3768) but never *defines* them -- the generated shader half is absent.  The
byte arrays are therefore taken out of the built backend binary (default `--source`
below), which is where they exist on this machine.

The host half is still the authority on *how* a pipeline is parameterised, and it says:
one SPIR-V module per reduction strategy (clustered / nocluster / shmem), with
S_V (head dim), kda, subgroup size and lanes_per_column passed as **specialization
constants** -- i.e. `gated_delta_net_f32_d128` and `..._d128_kda` are two pipeline
layouts over the SAME module, not two arrays.  `--fingerprint` reports the spec
constants so this can be confirmed from the binary instead of trusted.

USAGE
-----
  PYTHONIOENCODING=utf-8 python tools/spirv_gdn_decode.py                 # default run
  PYTHONIOENCODING=utf-8 python tools/spirv_gdn_decode.py --list          # just identify
  PYTHONIOENCODING=utf-8 python tools/spirv_gdn_decode.py --spv X.spv     # decode a file
  PYTHONIOENCODING=utf-8 python tools/spirv_gdn_decode.py --verify450 Y.spv  # print every
          OpExtInst index with its GLSL.std.450 name (calibrate against glslang.exe)

OUTPUT
------
  runs/gdn_d128_<tag>.spv       raw SPIR-V, one per candidate module
  runs/gdn_d128_disasm.txt      full decoded listing of the selected module(s)

Exit code 0 on success, 1 if no GDN-shaped module was found.
"""

from __future__ import annotations

import argparse
import os
import re
import struct
import subprocess
import sys
import tempfile

SPIRV_MAGIC = 0x07230203

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SOURCES = [
    os.path.join(REPO, "_ggml-vulkan.cpp"),                      # C-array path (absent here)
    r"llama.cpp/ggml-vulkan.dll",                          # built backend binary
]
HEADER_ARRAY_HINT = "gated_delta_net"


# ---------------------------------------------------------------------------
# enum tables.  Authoritative when glslang's generated spirv header is present
# on disk (we parse it); a hand-written subset is the fallback so the tool stays
# runnable on a machine without the SDK.
# ---------------------------------------------------------------------------
SPIRV_HPP_CANDIDATES = [
    r"<toolchain>/glslang/include/glslang/SPIRV/spirv.hpp11",
]

FALLBACK_OPS = {
    "OpNop": 0, "OpUndef": 1, "OpSourceContinued": 2, "OpSource": 3,
    "OpSourceExtension": 4, "OpName": 5, "OpMemberName": 6, "OpString": 7,
    "OpLine": 8, "OpExtension": 10, "OpExtInstImport": 11, "OpExtInst": 12,
    "OpMemoryModel": 14, "OpEntryPoint": 15, "OpExecutionMode": 16,
    "OpCapability": 17, "OpTypeVoid": 19, "OpTypeBool": 20, "OpTypeInt": 21,
    "OpTypeFloat": 22, "OpTypeVector": 23, "OpTypeMatrix": 24, "OpTypeImage": 25,
    "OpTypeSampler": 26, "OpTypeSampledImage": 27, "OpTypeArray": 28,
    "OpTypeRuntimeArray": 29, "OpTypeStruct": 30, "OpTypeOpaque": 31,
    "OpTypePointer": 32, "OpTypeFunction": 33, "OpConstantTrue": 41,
    "OpConstantFalse": 42, "OpConstant": 43, "OpConstantComposite": 44,
    "OpConstantNull": 45, "OpSpecConstantTrue": 48, "OpSpecConstantFalse": 49,
    "OpSpecConstant": 50, "OpSpecConstantComposite": 51, "OpSpecConstantOp": 52,
    "OpFunction": 54, "OpFunctionParameter": 55, "OpFunctionEnd": 56,
    "OpFunctionCall": 57, "OpVariable": 59, "OpLoad": 64, "OpStore": 65,
    "OpAccessChain": 67, "OpInBoundsAccessChain": 68, "OpPtrAccessChain": 69,
    "OpArrayLength": 70, "OpDecorate": 71, "OpMemberDecorate": 72,
    "OpVectorShuffle": 79, "OpCompositeConstruct": 80, "OpCompositeExtract": 81,
    "OpCompositeInsert": 82, "OpCopyObject": 83, "OpTranspose": 84,
    "OpConvertFToU": 109, "OpConvertFToS": 110, "OpConvertSToF": 111,
    "OpConvertUToF": 112, "OpUConvert": 113, "OpSConvert": 114, "OpFConvert": 115,
    "OpQuantizeToF16": 116, "OpConvertPtrToU": 117, "OpSatConvertSToU": 118,
    "OpSatConvertUToS": 119, "OpConvertUToPtr": 120, "OpPtrCastToGeneric": 121,
    "OpGenericCastToPtr": 122, "OpBitcast": 124, "OpSNegate": 125, "OpFNegate": 126,
    "OpIAdd": 127, "OpFAdd": 128, "OpISub": 129, "OpFSub": 130, "OpIMul": 131,
    "OpFMul": 132, "OpUDiv": 133, "OpSDiv": 134, "OpFDiv": 136, "OpUMod": 137,
    "OpSRem": 138, "OpSMod": 139, "OpFRem": 140, "OpVectorTimesScalar": 141,
    "OpDot": 148, "OpAny": 146, "OpAll": 145, "OpLogicalNot": 147,
    "OpSelect": 150, "OpFOrdEqual": 151, "OpFUnordEqual": 152,
    "OpFOrdNotEqual": 153, "OpFUnordNotEqual": 154, "OpFOrdLessThan": 155,
    "OpFUnordLessThan": 156, "OpFOrdGreaterThan": 157, "OpFUnordGreaterThan": 158,
    "OpFOrdLessThanEqual": 159, "OpFUnordLessThanEqual": 160,
    "OpFOrdGreaterThanEqual": 161, "OpFUnordGreaterThanEqual": 162,
    "OpISub": 129, "OpShiftRightLogical": 194, "OpShiftRightArithmetic": 195,
    "OpShiftLeftLogical": 196, "OpBitwiseOr": 197, "OpBitwiseXor": 198,
    "OpBitwiseAnd": 199, "OpNot": 200, "OpBitCount": 202,
    "OpControlBarrier": 225, "OpMemoryBarrier": 226,
    "OpLogicalEqual": 169, "OpLogicalNotEqual": 170, "OpLogicalOr": 171,
    "OpLogicalAnd": 172, "OpIEqual": 173, "OpINotEqual": 174,
    "OpUGreaterThan": 175, "OpSGreaterThan": 176, "OpUGreaterThanEqual": 177,
    "OpSGreaterThanEqual": 178, "OpULessThan": 179, "OpSLessThan": 180,
    "OpULessThanEqual": 181, "OpSLessThanEqual": 182, "OpPhi": 245,
    "OpLoopMerge": 246, "OpSelectionMerge": 247, "OpLabel": 248,
    "OpBranch": 249, "OpBranchConditional": 250, "OpSwitch": 251,
    "OpKill": 252, "OpReturn": 253, "OpReturnValue": 254, "OpUnreachable": 255,
    "OpVectorExtractDynamic": 77, "OpVectorInsertDynamic": 78,
    "OpGroupNonUniformElect": 333, "OpGroupNonUniformAll": 334,
    "OpGroupNonUniformAny": 335, "OpGroupNonUniformAllEqual": 336,
    "OpGroupNonUniformBroadcast": 337, "OpGroupNonUniformShuffle": 342,
    "OpGroupNonUniformShuffleXor": 343, "OpGroupNonUniformShuffleUp": 344,
    "OpGroupNonUniformShuffleDown": 345, "OpGroupNonUniformFAdd": 347,
    "OpGroupNonUniformFSub": 348, "OpGroupNonUniformFMul": 349,
    "OpGroupNonUniformSMin": 353, "OpGroupNonUniformUMin": 354,
    "OpGroupNonUniformFMin": 356, "OpGroupNonUniformSMax": 357,
    "OpGroupNonUniformUMax": 358, "OpGroupNonUniformFMax": 360,
    "OpGroupNonUniformClusteredAdd": 379, "OpGroupNonUniformClusteredMul": 381,
    "OpGroupNonUniformClusteredMin": 385, "OpGroupNonUniformClusteredMax": 389,
    "OpGroupNonUniformClusteredShuffleXor": 377,
    "OpTerminateInvocation": 4417, "OpIgnoreIntersection": 4401,
    "OpNoLine": 324, "OpModuleProcessed": 330, "OpExecutionModeId": 361,
    "OpDecorateId": 332, "OpSpecConstantIntegerEXT": 5901,
}

# GLSL.std.450 extended-instruction numbers.  Verified on this machine with
# --verify450 against a glslang.exe-compiled probe (see verify_450()); the
# subset we actually depend on is EXP=50, LOG=51, EXP2=52, LOG2=53, SQRT=54,
# INVSQRT=55, FMAX=63, FMIN=60, FMA=25, TANH=44, POW=49.
GLSL_450 = {
    0: "Bad", 1: "Round", 2: "RoundEven", 3: "Trunc", 4: "FAbs", 5: "FSign",
    6: "FOrdEqual", 7: "FUnordEqual", 8: "FOrdNotEqual", 9: "FUnordNotEqual",
    10: "FOrdLessThan", 11: "FUnordLessThan", 12: "FOrdGreaterThan",
    13: "FUnordGreaterThan", 14: "FOrdLessThanEqual", 15: "FUnordLessThanEqual",
    16: "FOrdGreaterThanEqual", 17: "FUnordGreaterThanEqual", 18: "Length",
    19: "Distance", 20: "Cross", 21: "Normalize", 22: "FaceForward",
    23: "Reflect", 24: "Refract", 25: "Fma", 26: "Mix", 27: "Step",
    28: "SmoothStep", 29: "Min", 30: "Max", 31: "Clamp", 32: "FastNormalize",
    33: "Synthesize", 34: "Degrees", 35: "Radians", 36: "Sin", 37: "Cos",
    38: "Tan", 39: "Asin", 40: "Acos", 41: "Atan", 42: "Sinh", 43: "Cosh",
    44: "Tanh", 45: "Asinh", 46: "Acosh", 47: "Atanh", 48: "Atan2", 49: "Pow",
    50: "Exp", 51: "Log", 52: "Exp2", 53: "Log2", 54: "Sqrt", 55: "InverseSqrt",
    56: "Determinant", 57: "MatrixInverse", 58: "Modf", 59: "ModfStruct",
    60: "FMin", 61: "UMin", 62: "SMin", 63: "FMax", 64: "UMax", 65: "SMax",
    66: "FClamp", 67: "UClamp", 68: "SClamp", 69: "FMix", 70: "IMix",
    71: "Step", 72: "SmoothStep", 73: "FindILsb", 74: "FindSMsb", 75: "FindUMsb",
    76: "InterpolateAtCentroid", 77: "NCoarse", 78: "PackSNorm", 79: "UnpackSNorm",
    80: "PackUnorm", 81: "UnpackUnorm", 82: "Float2Bits16x2?", 83: "Frexp",
    84: "Ldexp", 85: "LogarithmBase2?", 86: "Clamp?", 87: "Inverse?",
    88: "Sign?", 89: "SmoothStep?", 90: "FAbs?", 91: "UnpackHalf2x16",
    92: "PackHalf2x16", 93: "UnpackSnorm3x10_1x2?", 94: "Radians?2",
}

DECORATIONS = {
    0: "RelaxedPrecision", 1: "SpecId", 2: "Block", 3: "BufferBlock", 4: "RowMajor",
    5: "ColMajor", 6: "ArrayStride", 7: "MatrixStride", 8: "GLSLShared",
    9: "GLSLPacked", 10: "CPacked", 11: "BuiltIn", 12: "NoPrototype",
    13: "Invariant", 14: "Restricted", 15: "Uniform", 16: "Constant",
    17: "Coherent", 18: "Nonvolatile", 19: "Volatile", 20: "Relaxed",
    21: "Generic", 22: "Patch", 23: "ByteOffset", 24: "Aligned",
    25: "Compacted", 26: "UserSemantic", 27: "Register", 28: "InputAttachmentIndex",
    29: "Alignment", 30: "MaxTileExtents", 31: "MaxPerWorkgroupCOUNT_NV",
    32: "MaxPerTaskNV", 33: "Overlay", 34: "Literal", 35: "Offset",
    36: "Flat", 37: "NoInline", 38: "Centroid", 39: "Sample",
    40: "Invariant?", 41: "RestrictPointer", 42: "AliasedPointer",
    43: "ReferencedIndirectly", 44: "Bank", 45: "Tail?", 46: "NoContraction",
    50: "Input", 51: "Output", 52: "UniformId", 53: "SaturatedConversion",
    54: "Stream", 55: "Location", 56: "Component", 57: "Index",
    58: "Binding", 59: "DescriptorSet", 60: "Offset?", 61: "Alignment?",
}

BUILTINS = {
    0: "VertexId", 1: "InstanceId", 2: "VertexIndex", 3: "PrimitiveId",
    4: "InvocationId", 5: "Concurrency?", 6: "PatchVertices", 7: "GeomInvocationCount",
    8: "InstanceIndex", 9: "SubgroupSize?", 10: "NumSubgroups?", 11: "SubgroupID?",
    12: "SubgroupSize", 13: "NumSubgroups", 14: "SubgroupID", 15: "SubgroupLocalInvocationId",
    16: "VertexIndex2", 17: "TessLevelOuter", 18: "TessLevelInner", 19: "TessCoordU",
    20: "TessCoordV", 21: "TessCoordW", 22: "Position", 23: "PointCoord",
    24: "VertexIndex?", 25: "PrimitiveId?", 26: "Layer", 27: "ViewIndex",
    28: "FragCoord", 29: "PointCoord?", 30: "FrontFacing", 31: "SampleId",
    32: "SamplePosition", 33: "SampleMask", 34: "SampleCoverage?", 35: "FragDepth",
    36: "HelperInvocation", 37: "NumWorkgroups", 38: "WorkgroupSize", 39: "WorkgroupId",
    40: "LocalInvocationId", 41: "GlobalInvocationId", 42: "LocalInvocationIndex",
    43: "Workdimension", 44: "VertexIndex3", 45: "BaseVertex", 46: "BaseInstance",
    47: "DrawIndex", 48: "DeviceIndex",
}

STORAGE_CLASSES = {
    0: "UniformConstant", 1: "Input", 2: "Uniform", 3: "Output", 4: "Workgroup",
    5: "CrossWorkgroup", 6: "Private", 7: "Function", 8: "Generic", 9: "PushConstant",
    10: "AtomicCounter", 11: "Image", 12: "StorageBuffer",
}

CAPABILITIES = {
    0: "Matrix", 1: "Shader", 2: "Geometry", 3: "Tessellation", 4: "Addresses",
    5: "Linkage", 6: "Kernel", 7: "Vector16", 8: "Float16Buffer", 9: "Float16",
    10: "Float64", 11: "Int64", 12: "Int64Atomics", 13: "ImageBasic", 14: "ImageReadWrite",
    15: "ImageMipmap", 17: "Pipes", 18: "Groups", 19: "DeviceEnqueue", 20: "LiteralSampler",
    21: "AtomicStorage", 22: "Int16", 23: "TessellationPointSize", 24: "GeometryPointSize",
    25: "ImageGatherExtended", 27: "StorageImageMultisample", 31: "SampleRateShading",
    32: "ImageRect", 33: "SampledRect", 35: "InputAttachment", 36: "SparseResidency",
    37: "MinLod", 39: "Sampled1D", 40: "Image1D", 41: "SampledCubeArray",
    42: "SampledBuffer", 47: "ImageMSArray", 48: "StorageImageExtendedFormats",
    49: "ImageQuery", 50: "DerivativeControl", 51: "InterpolationFunction",
    52: "TransformFeedback", 53: "GeometryStreams", 54: "StorageImageReadWithoutFormat",
    55: "StorageImageWriteWithoutFormat", 56: "MultiViewport", 57: "SubgroupDispatch",
    58: "NamedBarrier", 59: "PipeStorage", 60: "GroupNonUniform", 61: "GroupNonUniformVote",
    62: "GroupNonUniformArithmetic", 63: "GroupNonUniformBallot",
    64: "GroupNonUniformShuffle", 65: "GroupNonUniformClustered",
    66: "GroupNonUniformQuad?", 67: "ShaderLayer", 68: "ShaderViewportIndex",
}

EXECUTION_MODELS = {0: "Vertex", 4: "Fragment", 5: "GLCompute"}
EXECUTION_MODES = {
    0: "Invocations", 1: "SpacingEquality", 2: "SpacingFractionalEven",
    3: "SpacingFractionalOdd", 4: "VertexOrderCw", 5: "VertexOrderCcw",
    6: "PixelCenterInteger", 7: "OriginUpperLeft", 8: "PixelInterlockOperation",
    14: "LocalSize", 15: "LocalSizeHint", 16: "InputPoints", 17: "InputLines",
    35: "OutputVertices", 38: "LocalSizeId",
}
SCOPES = {0: "Invocations", 1: "Subgroup", 2: "Workgroup", 3: "Device", 4: "QueueFamily", 5: "Queue"}


def load_tables():
    """Parse the authoritative opcode table if the glslang header is on disk."""
    ops = dict(FALLBACK_OPS)
    for path in SPIRV_HPP_CANDIDATES:
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                txt = f.read()
        except OSError:
            continue
        m = re.search(r"enum class Op\s*:\s*unsigned\s*\{(.*?)\n\};", txt, re.S)
        if not m:
            continue
        found = {}
        for name, value in re.findall(r"(Op\w+)\s*=\s*(\d+)", m.group(1)):
            found[name] = int(value)
        if len(found) > 60:
            ops = found
            return ops, path
    return ops, "(built-in fallback table)"


OPS, OPS_SOURCE = load_tables()
OP_BY_NUM = {v: k for k, v in OPS.items()}


# ---------------------------------------------------------------------------
# opcode operand signatures.  tokens:
#   R result id | T result type id | I id | L literal | S literal string (to end)
#   V rest, printed as id-or-literal heuristically | P pairs of (id,id) | Q pairs (lit,id)
# unknown opcodes fall back to a raw word dump and are counted so the reader can
# see whether the listing lost anything.
# ---------------------------------------------------------------------------
SIGS = {
    "OpSource": "L L I S", "OpSourceContinued": "S", "OpSourceExtension": "S",
    "OpExtension": "S", "OpName": "I S", "OpMemberName": "I L S", "OpString": "R S",
    "OpLine": "I L L", "OpNoLine": "", "OpModuleProcessed": "S",
    "OpCapability": "L", "OpExtInstImport": "R S", "OpMemoryModel": "L L",
    "OpEntryPoint": "L I S V", "OpExecutionMode": "I L V", "OpExecutionModeId": "I L V",
    "OpDecorate": "I L V", "OpDecorateId": "I L V", "OpMemberDecorate": "I L L V",
    # NB: type instructions carry no result *type*, so their first operand IS the
    # result id -- the opposite of every "T R ..." instruction below.
    "OpTypeVoid": "R", "OpTypeBool": "R", "OpTypeInt": "R L L", "OpTypeFloat": "R L",
    "OpTypeVector": "R T L", "OpTypeMatrix": "R T L", "OpTypeArray": "R T V",
    "OpTypeRuntimeArray": "R T", "OpTypeStruct": "R V", "OpTypeOpaque": "R S",
    "OpTypePointer": "R L T", "OpTypeFunction": "R T V", "OpTypeSampledImage": "R T",
    "OpTypeImage": "R T L L L L L V",
    "OpTypeForwardPointer": "I L",
    "OpConstantTrue": "T R", "OpConstantFalse": "T R", "OpConstantNull": "T R",
    "OpConstant": "T R V", "OpConstantComposite": "T R V",
    "OpSpecConstantTrue": "T R", "OpSpecConstantFalse": "T R",
    "OpSpecConstant": "T R V", "OpSpecConstantComposite": "T R V",
    "OpSpecConstantOp": "T R V", "OpUndef": "T R",
    "OpVariable": "T R L V", "OpLoad": "T R I V", "OpStore": "I I V",
    "OpAccessChain": "T R I V", "OpInBoundsAccessChain": "T R I V",
    "OpPtrAccessChain": "T R I I V", "OpArrayLength": "T R I L",
    "OpFunction": "T R L I", "OpFunctionParameter": "T R", "OpFunctionEnd": "",
    "OpFunctionCall": "T R I V",
    "OpLabel": "R", "OpBranch": "I", "OpBranchConditional": "I I V",
    "OpSwitch": "I I Q", "OpReturn": "", "OpReturnValue": "I", "OpUnreachable": "",
    "OpKill": "", "OpTerminateInvocation": "", "OpIgnoreIntersection": "",
    "OpPhi": "T R P", "OpSelectionMerge": "I L", "OpLoopMerge": "I I L",
    "OpCompositeConstruct": "T R V", "OpCompositeExtract": "T R I V",
    "OpCompositeInsert": "T R I I V", "OpVectorShuffle": "T R I I V",
    "OpVectorExtractDynamic": "T R I I", "OpVectorInsertDynamic": "T R I I I",
    "OpCopyObject": "T R I", "OpTranspose": "T R I", "OpDot": "T R I I",
    "OpVectorTimesScalar": "T R I I", "OpMatrixTimesScalar": "T R I I",
    "OpConvertFToU": "T R I", "OpConvertFToS": "T R I", "OpConvertSToF": "T R I",
    "OpConvertUToF": "T R I", "OpUConvert": "T R I", "OpSConvert": "T R I",
    "OpFConvert": "T R I", "OpQuantizeToF16": "T R I", "OpBitcast": "T R I",
    "OpConvertPtrToU": "T R I", "OpConvertUToPtr": "T R I",
    "OpPtrCastToGeneric": "T R I", "OpGenericCastToPtr": "T R I",
    "OpSatConvertSToU": "T R I", "OpSatConvertUToS": "T R I",
    "OpSNegate": "T R I", "OpFNegate": "T R I", "OpNot": "T R I",
    "OpBitCount": "T R I", "OpAny": "T R I", "OpAll": "T R I",
    "OpLogicalNot": "T R I", "OpIsNull": "T R I", "OpIsNotNull": "T R I",
    "OpIsNan": "T R I", "OpIsInf": "T R I",
}
# OpSNegate / OpISub are re-declared below by the binary-ops loop; dict literal
# duplicates are harmless, and the loop is the authoritative form.

# binary (two-operand) arithmetic / logic / comparison
for _n in ("OpIAdd", "OpFAdd", "OpISub", "OpFSub", "OpIMul", "OpFMul", "OpUDiv",
           "OpSDiv", "OpFDiv", "OpUMod", "OpSRem", "OpSMod", "OpFRem", "OpDot",
           "OpLogicalEqual", "OpLogicalNotEqual", "OpLogicalOr", "OpLogicalAnd",
           "OpIEqual", "OpINotEqual", "OpUGreaterThan", "OpSGreaterThan",
           "OpUGreaterThanEqual", "OpSGreaterThanEqual", "OpULessThan", "OpSLessThan",
           "OpULessThanEqual", "OpSLessThanEqual", "OpFOrdEqual", "OpFUnordEqual",
           "OpFOrdNotEqual", "OpFUnordNotEqual", "OpFOrdLessThan", "OpFUnordLessThan",
           "OpFOrdGreaterThan", "OpFUnordGreaterThan", "OpFOrdLessThanEqual",
           "OpFUnordLessThanEqual", "OpFOrdGreaterThanEqual", "OpFUnordGreaterThanEqual",
           "OpShiftRightLogical", "OpShiftRightArithmetic", "OpShiftLeftLogical",
           "OpBitwiseOr", "OpBitwiseXor", "OpBitwiseAnd", "OpSelect2"):
    SIGS[_n] = "T R I I"
SIGS.pop("OpSelect2", None)
SIGS["OpSelect"] = "T R I I I"

# scope-carrying subgroup ops: result, type, scope literal, then ids/literals
for _n in ("OpGroupNonUniformAll", "OpGroupNonUniformAny", "OpGroupNonUniformAllEqual",
           "OpGroupNonUniformBroadcast", "OpGroupNonUniformShuffle",
           "OpGroupNonUniformShuffleXor", "OpGroupNonUniformShuffleUp",
           "OpGroupNonUniformShuffleDown", "OpGroupNonUniformSAdd",
           "OpGroupNonUniformUAdd", "OpGroupNonUniformFAdd", "OpGroupNonUniformSSub",
           "OpGroupNonUniformUSub", "OpGroupNonUniformFSub", "OpGroupNonUniformSMul",
           "OpGroupNonUniformUMul", "OpGroupNonUniformFMul", "OpGroupNonUniformSMin",
           "OpGroupNonUniformUMin", "OpGroupNonUniformFMin", "OpGroupNonUniformSMax",
           "OpGroupNonUniformUMax", "OpGroupNonUniformFMax",
           "OpGroupNonUniformBitwiseAnd", "OpGroupNonUniformBitwiseOr",
           "OpGroupNonUniformBitwiseXor", "OpGroupNonUniformLogicalAnd",
           "OpGroupNonUniformLogicalOr", "OpGroupNonUniformLogicalXor",
           "OpGroupNonUniformQuadBroadcast", "OpGroupNonUniformQuadSwap",
           "OpGroupNonUniformClusteredAdd", "OpGroupNonUniformClusteredMul",
           "OpGroupNonUniformClusteredMin", "OpGroupNonUniformClusteredMax",
           "OpGroupNonUniformClusteredBitwiseXor",
           "OpGroupNonUniformClusteredShuffleXor"):
    SIGS[_n] = "T R L V"
SIGS["OpGroupNonUniformElect"] = "T R"
SIGS["OpGroupNonUniformBallot"] = "T R L I"
SIGS["OpGroupNonUniformInverseBallot"] = "T R L I"
SIGS["OpGroupNonUniformBallotBitExtract"] = "T R L I L"
SIGS["OpGroupNonUniformBallotBitCount"] = "T R L L I"
SIGS["OpGroupNonUniformBallotFindLSB"] = "T R L I"
SIGS["OpGroupNonUniformBallotFindMSB"] = "T R L I"
SIGS["OpExtInst"] = "T R I L V"
SIGS["OpControlBarrier"] = "L L L"
SIGS["OpMemoryBarrier"] = "L L V"


# ---------------------------------------------------------------------------
# instruction walking
# ---------------------------------------------------------------------------
def decode_string(words, i):
    out = []
    while i < len(words):
        w = words[i]
        b = struct.pack("<I", w)
        z = b.find(b"\0")
        if z >= 0:
            out.append(b[:z].decode("utf-8", "replace"))
            return "".join(out), i + 1
        out.append(b.decode("utf-8", "replace").rstrip("\0"))
        i += 1
    return "".join(out), i


def walk_module(words, start=0, with_ops=True, min_instrs=1):
    """Walk a word stream instruction by instruction using each instruction's own
    word count (the module header's instruction-count field is 0 in every embedded
    module on this machine -- measured -- so the length has to come from the walk).

    Returns (instrs, clean).  clean == True when the walk stopped because it met a
    word that is not a plausible instruction header, i.e. it found the real end of
    the module; clean == False means the provided window ran out first.

    with_ops=False is the cheap pass used to pre-filter the ~1.5k modules inside
    the built backend: it records only (pos, name, wc) so no operand lists are
    built for modules that are then thrown away."""
    if start + 5 > len(words):
        return [], False
    if words[start] != SPIRV_MAGIC:
        return [], False
    version = words[start + 1]
    if (version >> 16) not in (1, 2):
        return [], False
    pos = start + 5
    first = OP_BY_NUM.get(words[pos] & 0xFFFF) if pos < len(words) else None
    if first not in ("OpCapability", "OpExtension", "OpSource", "OpName"):
        return [], False
    instrs = []
    nwords = len(words)
    while pos < nwords:
        hw = words[pos]
        wc = hw >> 16
        op = hw & 0xFFFF
        if wc == 0 or pos + wc > nwords:
            return instrs, True
        name = OP_BY_NUM.get(op)
        if name is None:
            # unknown opcode: real modules are built by glslang against the table we
            # loaded, so this means we have run past the end into unrelated data.
            return instrs, True
        if with_ops:
            instrs.append((pos, name, wc, words[pos + 1:pos + wc]))
        else:
            instrs.append((pos, name, wc))
        pos += wc
    return instrs, False


class Module:
    def __init__(self, data, offset, words, instrs):
        self.data = data
        self.offset = offset
        self.words = words
        self.instrs = instrs
        self.bound = words[3]
        self.version = words[1]
        self.generator = words[2]
        # word 4 is the instruction-count field; llama.cpp's embedded modules
        # carry 0 there (measured), so length comes from the walk.
        self.word_count = words[4]
        self.end_words = 5 + sum(w for _, _, w, _ in instrs)
        self.names = {}
        self.decs = {}          # id -> list of (decoration, literals)
        self.member_decs = {}   # (id, member) -> (decoration, literals)
        self.strings = {}       # OpString id -> text
        self.types = {}         # id -> (name, operands)
        self.consts = {}        # id -> decoded literal / (composite, [ids])
        self.spec_ids = {}      # id -> SpecId literal
        self.builtins = {}      # id -> BuiltIn literal
        self.bindings = {}      # id -> (set, binding)
        self.offsets = {}       # (struct, member) -> byte offset
        self.entry = None       # (model, id, name, interface)
        self.local_size = None
        self.local_size_id = None
        self.execution_modes = []
        self.memory_model = None
        self.ext_imports = {}   # id -> set name
        self.variables = []     # (id, type, storage, init)
        self.functions = []
        self.unknown_ops = set()

    # -------------------------------------------------------------------
    def analyze(self):
        for pos, name, wc, ops in self.instrs:
            try:
                self._one(pos, name, wc, ops)
            except IndexError:
                self.unknown_ops.add(name + "(parse-error)")
        # second pass: bindings need DescriptorSet+Binding merged
        return self

    def _one(self, pos, name, wc, ops):
        if name == "OpName":
            self.names[ops[0]] = decode_string(self.words, pos + 2)[0]
        elif name == "OpMemberName":
            s = decode_string(self.words, pos + 3)[0]
            self.names["%d::%d" % (ops[0], ops[1])] = s
        elif name == "OpString":
            self.strings[ops[0]] = decode_string(self.words, pos + 2)[0]
        elif name == "OpModuleProcessed":
            pass
        elif name == "OpSource":
            txt = decode_string(self.words, pos + 4)[0]
            self.strings.setdefault("source", txt)
        elif name == "OpDecorate":
            dec = ops[1]
            lits = ops[2:]
            self.decs.setdefault(ops[0], []).append((DECORATIONS.get(dec, str(dec)), lits))
            if dec == 1:
                self.spec_ids[ops[0]] = lits[0] if lits else None
            elif dec == 11:
                self.builtins[ops[0]] = lits[0] if lits else None
            elif dec == 58:
                self.bindings.setdefault(ops[0], [None, None])[1] = lits[0] if lits else None
            elif dec == 59:
                self.bindings.setdefault(ops[0], [None, None])[0] = lits[0] if lits else None
        elif name == "OpMemberDecorate":
            dec = ops[2]
            lits = ops[3:]
            self.member_decs[(ops[0], ops[1])] = (DECORATIONS.get(dec, str(dec)), lits)
            if dec == 35:
                self.offsets[(ops[0], ops[1])] = lits[0]
        elif name.startswith("OpType"):
            self.types[ops[0]] = (name, ops[1:])
        elif name in ("OpConstant", "OpConstantTrue", "OpConstantFalse",
                      "OpConstantComposite", "OpConstantNull",
                      "OpSpecConstant", "OpSpecConstantTrue", "OpSpecConstantFalse",
                      "OpSpecConstantComposite", "OpSpecConstantOp", "OpUndef"):
            # word1 = result type, word2 = result id, rest = literals / constituents
            self.consts[ops[1]] = (name, ops[0], ops[2:])
        elif name == "OpCapability":
            self.execution_modes.append(("capability", ops[0]))
        elif name == "OpExtInstImport":
            self.ext_imports[ops[0]] = decode_string(self.words, pos + 2)[0]
        elif name == "OpMemoryModel":
            self.memory_model = ops[:2]
        elif name == "OpEntryPoint":
            eid = ops[1]
            ename = decode_string(self.words, pos + 3)[0]
            iface = ops[2 + (len(ename) + 4) // 4:]
            self.entry = (ops[0], eid, ename, list(iface))
        elif name == "OpExecutionMode":
            mode = ops[1]
            self.execution_modes.append(("mode", mode, list(ops[2:])))
            if mode == 14:
                self.local_size = list(ops[2:5])
            elif mode == 38:
                self.local_size_id = list(ops[2:5])
        elif name == "OpVariable":
            storage = ops[2] if len(ops) > 2 else 7
            self.variables.append((ops[1], ops[0], storage,
                                   ops[3] if len(ops) > 3 else None))
        elif name == "OpFunction":
            self.functions.append((ops[1], ops[0], ops[3]))

    # -------------------------------------------------------------------
    def fmt_type(self, tid):
        if tid not in self.types:
            return "?t%d" % tid
        name, ops = self.types[tid]
        if name == "OpTypeVoid":
            return "void"
        if name == "OpTypeBool":
            return "bool"
        if name == "OpTypeInt":
            return "i%d" % ops[0] if not (len(ops) > 1 and ops[1] == 0) else "u%d" % ops[0]
        if name == "OpTypeFloat":
            return "f%d" % ops[0]
        if name == "OpTypeVector":
            return "%s<%d>" % (self.fmt_type(ops[0]), ops[1])
        if name == "OpTypeMatrix":
            return "mat<%d,%s>" % (ops[1], self.fmt_type(ops[0]))
        if name == "OpTypeArray":
            n = self.fmt_len(ops[1])
            return "%s[%s]" % (self.fmt_type(ops[0]), n)
        if name == "OpTypeRuntimeArray":
            return "%s[]" % self.fmt_type(ops[0])
        if name == "OpTypePointer":
            return "ptr<%s,%s>" % (STORAGE_CLASSES.get(ops[0], ops[0]), self.fmt_type(ops[1]))
        if name == "OpTypeFunction":
            return "fn(%s,%s)" % (self.fmt_type(ops[0]), ",".join(self.fmt_type(x) for x in ops[1:]))
        if name == "OpTypeStruct":
            return "struct"
        return name

    def fmt_len(self, w):
        if w in self.consts:
            kind, tid, vals = self.consts[w]
            if vals:
                return str(vals[0])
        if w in self.spec_ids:
            return "SpecId%d" % self.spec_ids[w]
        return str(w)

    def fmt_const(self, cid):
        if cid not in self.consts:
            return None
        kind, tid, vals = self.consts[cid]
        ty = self.fmt_type(tid)
        spec = self.spec_ids.get(cid)
        pre = "SpecId%d " % spec if spec is not None else ""
        if kind == "OpConstantTrue":
            return "%s:true" % ty
        if kind == "OpConstantFalse":
            return "%s:false" % ty
        if kind == "OpConstantNull":
            return "%s:null" % ty
        if kind == "OpUndef":
            return "%s:undef" % ty
        if kind in ("OpConstantComposite", "OpSpecConstantComposite"):
            return "%s%s{%s}" % (pre, ty, ",".join("%%%d" % v for v in vals))
        if kind == "OpSpecConstantOp":
            inner = OP_BY_NUM.get(vals[0]) if vals else None
            return "%sspecop<%s %s %s>" % (pre, ty, inner or vals[0] if vals else "",
                                           " ".join("%%%d" % v for v in vals[1:]))
        if kind == "OpConstant":
            if not vals:
                return "%s:?" % ty
            raw = vals[0]
            if ty.startswith("f"):
                v = struct.unpack("<f", struct.pack("<I", raw))[0]
                return "%s:%.9g" % (ty, v)
            if ty.startswith("i"):
                return "%s:%d" % (ty, struct.unpack("<i", struct.pack("<I", raw))[0])
            if ty.startswith("u"):
                return "%s:%u" % (ty, raw)
            return "%s:0x%x" % (ty, raw)
        if kind == "OpSpecConstant":
            raw = vals[0] if vals else 0
            if ty.startswith("f"):
                d = "%.9g" % struct.unpack("<f", struct.pack("<I", raw))[0]
            else:
                d = str(struct.unpack("<i", struct.pack("<I", raw))[0])
            return "SPEC<%s default=%s>" % (ty, d)
        return "%s:?" % ty

    # -------------------------------------------------------------------
    def disasm(self):
        lines = []
        A = lines.append
        A("; ============================================================")
        A("; SPIR-V module @ file offset 0x%x (%d bytes, %d instructions)"
          % (self.offset, self.end_words * 4, len(self.instrs)))
        A("; version %d.%d  generator 0x%x  id bound %d  header inst-count word %d"
          % (self.version >> 16, (self.version >> 8) & 0xFF, self.generator,
             self.bound, self.word_count))
        A("; ============================================================")
        for pos, name, wc, ops in self.instrs:
            A(self.render(name, ops, pos))
        return "\n".join(lines)

    def render(self, name, ops, pos):
        """One listing line:  [ %R = ] OpName <type> <operands...>  ; annotation"""
        sig = SIGS.get(name)
        if sig is None:
            self.unknown_ops.add(name)
            return "%-24s ; raw words %s" % (name + " [UNSUPPORTED]",
                                             " ".join("0x%08x" % x for x in ops))
        toks = sig.split()
        out = []
        ann = []
        i = 0
        res = None
        ri = 0                       # position of the result id inside `out`
        for k, t in enumerate(toks):
            if i >= len(ops) and t in ("R", "T", "I", "L"):
                self.unknown_ops.add(name + "(short)")
                break
            if t == "R":
                res = ops[i]
                ri = len(out)
                out.append("%%%d" % ops[i])
                i += 1
            elif t == "T":
                out.append(self.fmt_type(ops[i]))
                i += 1
            elif t == "I":
                w = ops[i]
                out.append(self.fmt_ref(w))
                c = self.fmt_const(w)
                if c:
                    ann.append("%%%d=%s" % (w, c))
                i += 1
            elif t == "L":
                v = ops[i]
                i += 1
                if name == "OpCapability":
                    out.append(CAPABILITIES.get(v, str(v)))
                elif name == "OpExecutionMode":
                    out.append(EXECUTION_MODES.get(v, str(v)))
                elif name == "OpDecorate" and k == 1:
                    out.append(DECORATIONS.get(v, str(v)))
                elif name == "OpMemberDecorate" and k == 2:
                    out.append(DECORATIONS.get(v, str(v)))
                elif name == "OpExtInst" and k == 3:
                    out.append("450::" + GLSL_450.get(v, "UNKNOWN_%d" % v))
                elif name == "OpVariable":
                    out.append(STORAGE_CLASSES.get(v, str(v)))
                elif name == "OpEntryPoint":
                    out.append(EXECUTION_MODELS.get(v, str(v)))
                elif name == "OpGroupNonUniformShuffleXor" or name.startswith("OpGroupNonUniformClustered"):
                    out.append("mask/cluster=%s" % v)
                else:
                    out.append("%d/0x%x" % (v, v))
            elif t == "S":
                s, nxt = decode_string(self.words, pos + 1 + i)
                out.append('"%s"' % s)
                i = nxt - (pos + 1)
                if name in ("OpEntryPoint",):
                    rest = ops[i:]
                    i = len(ops)
                    if rest:
                        out.append("interface:" + " ".join(self.fmt_ref(x) for x in rest))
            elif t == "V":
                rest = ops[i:]
                i = len(ops)
                if name == "OpConstant":
                    ty = self.fmt_type(ops[0])
                    vals = []
                    for x in rest:
                        if ty.startswith("f"):
                            vals.append("%.9g" % struct.unpack("<f", struct.pack("<I", x))[0])
                        elif ty.startswith("i"):
                            vals.append(str(struct.unpack("<i", struct.pack("<I", x))[0]))
                        else:
                            vals.append("%u/0x%x" % (x, x))
                    out.extend(vals)
                elif name in ("OpAccessChain", "OpInBoundsAccessChain", "OpPtrAccessChain"):
                    out.extend(self.fmt_ref(x) for x in rest)
                    chain = []
                    for x in rest:
                        c = self.fmt_const(x)
                        chain.append(c if c else self.fmt_ref(x))
                    ann.append("CHAIN[" + " ".join(chain) + "]")
                elif name == "OpCompositeExtract":
                    out.extend([self.fmt_ref(rest[0])] + [str(x) for x in rest[1:]])
                    ann.append("extract[%s]" % ",".join(str(x) for x in rest[1:]))
                elif name == "OpVectorShuffle":
                    out.extend([self.fmt_ref(rest[0]), self.fmt_ref(rest[1])]
                               + [str(x) for x in rest[2:]])
                elif name == "OpExtInst":
                    out.extend(self.fmt_ref(x) for x in rest)
                    args = []
                    for x in rest:
                        c = self.fmt_const(x)
                        args.append(c if c else self.fmt_ref(x))
                    ann.append("450args[" + " ".join(args) + "]")
                elif name == "OpPhi":
                    out.append(" ".join("%s->%%%d" % (self.fmt_ref(rest[j + 1]), rest[j])
                                        for j in range(0, len(rest) - 1, 2)))
                elif name == "OpVariable":
                    out.extend("init=" + self.fmt_ref(x) for x in rest)
                else:
                    out.extend(self.fmt_ref(x) for x in rest)
            elif t == "P":
                rest = ops[i:]
                i = len(ops)
                out.append(" ".join("%s:%%%d" % (self.fmt_ref(rest[j]), rest[j + 1])
                                    for j in range(0, len(rest) - 1, 2)))
            elif t == "Q":
                rest = ops[i:]
                i = len(ops)
                out.append(" ".join("%s:%%%d" % (rest[j], rest[j + 1])
                                    for j in range(0, len(rest) - 1, 2)))
        head = ("%5s =" % out[0]) if res is not None else "        "
        body = "%s %-24s %s" % (head, name, " ".join(str(o) for o in out[1:]))
        if ann:
            body += "   ; " + "  ".join(ann)
        return body

    def fmt_ref(self, w):
        return "%%%d" % w


# ---------------------------------------------------------------------------
# intake: C arrays and raw binaries
# ---------------------------------------------------------------------------
C_ARRAY_RE = re.compile(
    r"(?:static\s+)?const\s+(?:unsigned\s+)?(?:uint32_t|uint8_t|char)\s+"
    r"(?P<name>\w+)\s*\[(?:\s*(?P<len>\d*)\s*)?\]\s*=\s*\{(?P<body>[^;]*?)\}\s*;",
    re.S)
C_STRING_RE = re.compile(
    r"(?:static\s+)?const\s+(?:unsigned\s+)?char\s+(?P<name>\w+)\s*\[\s*\]\s*=\s*\"(?P<body>(?:\\.|[^\"\\])*)\"\s*;")


def extract_c_arrays(text, hint):
    """Return {name: words} for every embedded array in `text` whose name matches
    `hint` (or any array that starts with the SPIR-V magic)."""
    found = {}
    for m in C_ARRAY_RE.finditer(text):
        name = m.group("name")
        if hint and hint not in name:
            continue
        body = m.group("body")
        nums = re.findall(r"0x([0-9a-fA-F]{1,8})|(?<![\w.])(\d+)(?![\w.])", body)
        vals = [int(a or b, 16 if a else 10) for a, b in nums]
        if not vals:
            continue
        if "8_t" in m.group(0) or "char" in m.group(0).split(name)[0]:
            vals = [v & 0xFF for v in vals]
            words = [struct.unpack("<I", bytes(vals[i:i + 4]))[0]
                     for i in range(0, len(vals) - 3, 4)]
            if words and words[0] == SPIRV_MAGIC:
                found[name] = words
                continue
        if vals[0] == SPIRV_MAGIC or (vals[0] & 0xFF) == 0x03:
            found[name] = vals
    for m in C_STRING_RE.finditer(text):
        name = m.group("name")
        if hint and hint not in name:
            continue
        raw = m.group("body").encode().decode("unicode_escape").encode("latin1")
        if len(raw) >= 4 and struct.unpack("<I", raw[:4])[0] == SPIRV_MAGIC:
            words = list(struct.unpack("<%dI" % (len(raw) // 4), raw[:len(raw) - len(raw) % 4]))
            found[name] = words
    return found


MAX_MODULE_WORDS = 400_000          # 1.6 MB -- far above any ggml shader


def _header_ok(data, off):
    """Cheap gate: magic + SPIR-V 1.x version + the first instruction after the
    header is one of the section-leading opcodes."""
    if off + 24 > len(data):
        return False
    ver = struct.unpack_from("<I", data, off + 4)[0]
    if (ver >> 16) not in (1, 2):
        return False
    first = struct.unpack_from("<H", data, off + 20)[0]
    return first in (OPS.get("OpCapability", 17), OPS.get("OpExtension", 10),
                    OPS.get("OpSource", 3), OPS.get("OpName", 5))


def module_starts(data):
    return [m.start() for m in re.finditer(rb"\x03\x02\x23\x07", data)]


def find_modules_in_binary(data, min_instrs=25, verbose=False):
    """Two-pass scan of a PE/ELF/raw dump.

    Pass 1 walks with word counts only (no operand lists) over a window bounded by
    the *next* magic offset, so the 1.5k embedded modules cost a few MB of work
    instead of 1.5k full-file unpacks.  A module is accepted when the walk stops on
    its own (first word that is not a plausible instruction header) and ends on
    OpFunctionEnd -- glslang-emitted modules always do.  If the walk instead runs
    into the next magic, the window was too small (a module body can legitimately
    contain the magic word as a literal), so widen it.

    Returns [(offset, nwords, quick_instrs)] -- re-walk with with_ops=True for the
    handful of survivors.
    """
    starts = module_starts(data)
    out = []
    for idx, off in enumerate(starts):
        if not _header_ok(data, off):
            continue
        windows = []
        for j in range(idx + 1, min(idx + 5, len(starts))):
            windows.append((starts[j] - off) // 4)
        windows.append(min((len(data) - off) // 4, MAX_MODULE_WORDS))
        chosen = None
        for w in windows:
            if w < 32:
                continue
            words = struct.unpack_from("<%dI" % w, data, off)
            quick, clean = walk_module(words, with_ops=False, min_instrs=min_instrs)
            if len(quick) < min_instrs:
                break
            if not quick:
                continue
            ends_ok = quick[-1][1] == "OpFunctionEnd"
            if clean or ends_ok:
                chosen = (off, w, quick)
                break
        if chosen is None:
            continue
        off, w, quick = chosen
        if not any(q[1] == "OpEntryPoint" for q in quick):
            continue
        out.append((off, w, quick))
    if verbose:
        print("  pass1: %d/%d magic hits accepted as complete modules"
              % (len(out), len(starts)))
    return out


def rewalk(data, off, nwords):
    words = struct.unpack_from("<%dI" % nwords, data, off)
    instrs, clean = walk_module(words, with_ops=True)
    return Module(data, off, words, instrs), clean


# ---------------------------------------------------------------------------
# fingerprinting / GDN identification
# ---------------------------------------------------------------------------
def struct_of(mod, tid):
    """Follow OpTypePointer to its pointee; return the struct id if it is one."""
    seen = 0
    while tid in mod.types and seen < 4:
        name, ops = mod.types[tid]
        if name == "OpTypePointer":
            tid = ops[1]
        elif name == "OpTypeStruct":
            return tid
        else:
            return None
        seen += 1
    return None


def fingerprint(mod: Module):
    mod.analyze()
    pc_members = 0
    pc_struct = None
    buffers = []
    workgroup_vars = []
    spec_consts = []
    for vid, ty, storage, init in mod.variables:
        if storage == 9:  # PushConstant
            pc_struct = struct_of(mod, ty)
        elif storage in (12, 2, 3):  # StorageBuffer / Uniform
            b, ds = mod.bindings.get(vid), None
            if vid in mod.decs:
                for d, lits in mod.decs[vid]:
                    if d == "DescriptorSet":
                        ds = lits[0] if lits else None
            buffers.append((vid, storage, ds, b[1] if isinstance(b, list) else b,
                            mod.fmt_type(ty)))
        elif storage == 4:  # Workgroup (shared memory)
            workgroup_vars.append((vid, mod.fmt_type(ty)))
    if pc_struct is not None:
        pc_members = sum(1 for (sid, _m), (d, _l) in mod.member_decs.items()
                         if sid == pc_struct and d == "Offset")
    for cid, (kind, ops) in mod.consts.items():
        if kind.startswith("OpSpec") and kind != "OpSpecConstantComposite":
            spec_consts.append((mod.spec_ids.get(cid), cid, mod.fmt_const(cid)))
    caps = [x[1] for x in mod.execution_modes if x[0] == "capability"]
    counts = {}
    for _p, name, _wc, _ops in mod.instrs:
        counts[name] = counts.get(name, 0) + 1
    return {
        "offset": mod.offset,
        "bytes": mod.end_words * 4,
        "instrs": len(mod.instrs),
        "entry": mod.entry[2] if mod.entry else None,
        "buffers": len(buffers),
        "buffer_detail": sorted(buffers, key=lambda x: (x[2] is None, x[3] is None,
                                                        x[2] or 0, x[3] or 0)),
        "pc_struct": pc_struct,
        "pc_members": pc_members,
        "spec_consts": sorted(spec_consts, key=lambda x: (x[0] is None, x[0])),
        "workgroup_vars": workgroup_vars,
        "local_size": mod.local_size,
        "local_size_id": [mod.fmt_len(x) if isinstance(x, int) else x
                          for x in (mod.local_size_id or [])],
        "caps": sorted(CAPABILITIES.get(c, str(c)) for c in caps),
        "ext450": sorted({GLSL_450.get(o[3], "?") for _p, n, _wc, o in mod.instrs
                          if n == "OpExtInst" and len(o) > 3}),
        "n_ext": counts.get("OpExtInst", 0),
        "opcounts": counts,
        "arith": {k: counts.get(k, 0) for k in
                  ("OpFMul", "OpFAdd", "OpFSub", "OpFDiv", "OpAccessChain", "OpLoad",
                   "OpStore", "OpPhi", "OpSelect", "OpFOrdLessThan", "OpDot",
                   "OpGroupNonUniformFAdd", "OpGroupNonUniformFMul",
                   "OpGroupNonUniformClusteredAdd", "OpControlBarrier", "OpVectorShuffle",
                   "OpCompositeInsert", "OpFunctionCall")},
        "unknown": sorted(mod.unknown_ops),
        "mod": mod,
    }


def looks_like_gdn(fp):
    """Structural signature taken from the host code (see module docstring):
    7 descriptor buffers + a 17-member push-constant struct (H, n_tokens, n_seqs,
    s_off, 3x3 source strides, neq1, rq3, scale, K)."""
    return fp["buffers"] == 7 and fp["pc_members"] == 17


def variant_tag(fp):
    """Which of the three reduction strategies the host code would have picked for
    this module (gated_delta_net_f32 / _nocluster / _shmem), read off the module
    itself rather than assumed."""
    caps = set(fp["caps"])
    has_smem = bool(fp["workgroup_vars"])
    clustered = "GroupNonUniformClustered" in caps
    arith = ("GroupNonUniformArithmetic" in caps) or ("GroupNonUniform" in caps)
    if has_smem and not arith:
        return "shmem"
    if has_smem and clustered:
        return "clustered+smem"
    if clustered:
        return "clustered"
    if arith:
        return "nocluster"
    return "sequential"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--source", action="append", default=None,
                    help="file(s) to search (C array source or binary); default: %s"
                         % ", ".join(DEFAULT_SOURCES))
    ap.add_argument("--hint", default=HEADER_ARRAY_HINT,
                    help="array-name substring for the C-array path")
    ap.add_argument("--spv", help="decode one already-extracted .spv file")
    ap.add_argument("--out-dir", default=os.path.join(REPO, "runs"))
    ap.add_argument("--prefix", default="gdn_d128")
    ap.add_argument("--list", action="store_true", help="only print fingerprints")
    ap.add_argument("--all", action="store_true", help="dump every GDN-shaped module")
    ap.add_argument("--expect-sv", type=int, default=128,
                    help="S_V specialization value expected (default 128)")
    ap.add_argument("--verify450", metavar="FILE.spv",
                    help="decode FILE and print all OpExtInst indices with names")
    args = ap.parse_args(argv)

    if args.verify450:
        with open(args.verify450, "rb") as f:
            data = f.read()
        words = list(struct.unpack("<%dI" % (len(data) // 4), data[:len(data) - len(data) % 4]))
        instrs, ok = walk_module(words)
        m = Module(data, 0, words, instrs).analyze()
        print("GLSL.std.450 indices present in %s (ok=%s):" % (args.verify450, ok))
        for _p, n, _wc, o in instrs:
            if n == "OpExtInst":
                print("  idx=%-3d -> %-14s result-type=%s args=%s"
                      % (o[3], GLSL_450.get(o[3], "UNKNOWN"), m.fmt_type(o[1]),
                         " ".join("%%%d" % x for x in o[4:])))
        return 0

    if args.spv:
        with open(args.spv, "rb") as f:
            data = f.read()
        words = list(struct.unpack("<%dI" % (len(data) // 4), data[:len(data) - len(data) % 4]))
        instrs, ok = walk_module(words)
        mod = Module(data, 0, words, instrs).analyze()
        print(mod.disasm())
        print("; unsupported opcodes:", sorted(mod.unknown_ops))
        return 0

    os.makedirs(args.out_dir, exist_ok=True)
    sources = args.source or DEFAULT_SOURCES
    cands = []      # (source, tag, Module)
    notes = []

    for src in sources:
        if not os.path.isfile(src):
            notes.append("MISSING source: %s" % src)
            continue
        with open(src, "rb") as f:
            data = f.read()
        text = None
        # try C-array text intake
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = None
        if text is not None and ("_data" in text or "spirv" in text.lower()):
            arrays = extract_c_arrays(text, args.hint)
            if arrays:
                for name, words in arrays.items():
                    instrs, ok = walk_module(words)
                    if not ok:
                        notes.append("array %s in %s does not parse (%s)" % (name, src, ok))
                        continue
                    cands.append((src, name, Module(data, 0, words, instrs)))
                notes.append("C-array intake: %d array(s) matched in %s: %s"
                             % (len(arrays), src, ", ".join(sorted(arrays))))
                continue
        # binary intake: two passes, the first without operand lists
        quick = find_modules_in_binary(data, verbose=True)
        notes.append("binary intake: %d magic hit(s) -> %d self-terminating module(s) "
                     "in %s (%d bytes)" % (len(module_starts(data)), len(quick), src, len(data)))
        survivors = []
        for off, nwords, q in quick:
            n_md = 0
            n_var = 0
            for item in q:
                if item[1] == "OpMemberDecorate":
                    n_md += 1
                elif item[1] == "OpVariable":
                    n_var += 1
            if n_md >= 14 and n_var >= 8:
                survivors.append((off, nwords))
        notes.append("prefilter (>=14 OpMemberDecorate and >=8 OpVariable) kept %d "
                     "module(s) for full decode" % len(survivors))
        for off, nwords in survivors:
            mod, clean = rewalk(data, off, nwords)
            if not clean:
                notes.append("module @0x%x did not self-terminate on rewalk" % off)
            cands.append((src, "off_0x%x" % off, mod))

    if not cands:
        print("FATAL: no SPIR-V modules found."); [print(" -", n) for n in notes]; return 1

    fps = []
    for src, tag, mod in cands:
        try:
            fps.append((src, tag, mod, fingerprint(mod)))
        except Exception as e:      # a broken module must not kill the run
            notes.append("fingerprint failed for %s: %r" % (tag, e))

    gdn = [x for x in fps if looks_like_gdn(x[3])]
    near = [x for x in fps if x[3]["buffers"] == 7 or x[3]["pc_members"] == 17]

    print("== notes ==")
    for n in notes:
        print(" -", n)
    print("== modules parsed: %d ; GDN-shaped (7 buffers + 17-member push-constant "
          "struct): %d ; near-miss: %d ==" % (len(fps), len(gdn), len(near)))

    def show(fp, label=""):
        print("--- %s ---" % label)
        print("  variant=%s  bytes=%d instrs=%d entry=%r buffers=%d pc_members=%d"
              % (variant_tag(fp), fp["bytes"], fp["instrs"], fp["entry"],
                 fp["buffers"], fp["pc_members"]))
        print("  buffers (id, storage, set, binding, type)=")
        for b in fp["buffer_detail"]:
            print("      %s" % (b,))
        print("  local_size=%s local_size_id=%s" % (fp["local_size"], fp["local_size_id"]))
        print("  spec_consts (SpecId, id, default)=%s" % fp["spec_consts"])
        print("  workgroup_vars=%s" % fp["workgroup_vars"])
        print("  caps=%s" % fp["caps"])
        print("  ext450=%s (n_ext=%d)" % (fp["ext450"], fp["n_ext"]))
        print("  arith=%s" % fp["arith"])
        print("  unknown_opcodes=%s" % fp["unknown"])

    if args.list:
        for src, tag, mod, fp in sorted(gdn or near, key=lambda x: -x[3]["bytes"])[:12]:
            show(fp, "%s @ %s" % (tag, src))
        return 0 if gdn else 1

    sel = (gdn or near)
    sel = sorted(sel, key=lambda x: -x[3]["bytes"])
    if not args.all:
        sel = sel[:1]
    listing = []
    for src, tag, mod, fp in sel:
        show(fp, "%s [%s] @ %s" % (tag, variant_tag(fp), src))
        spv_path = os.path.join(args.out_dir, "%s_%s_%s.spv"
                                % (args.prefix, variant_tag(fp).replace("+", "_"), tag))
        with open(spv_path, "wb") as f:
            f.write(struct.pack("<%dI" % mod.end_words, *mod.words[:mod.end_words]))
        print("  wrote %s (%d bytes, self-reported header count=%d)"
              % (spv_path, mod.end_words * 4, mod.word_count))
        listing.append(mod.disasm())
        listing.append("")
    dis_path = os.path.join(args.out_dir, "%s_disasm.txt" % args.prefix)
    with open(dis_path, "w", encoding="utf-8") as f:
        f.write("\n".join(listing))
    print("wrote listing:", dis_path)
    return 0 if gdn else 1


if __name__ == "__main__":
    sys.exit(main())
