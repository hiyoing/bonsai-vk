# Licenses and provenance

**The engine's own licence is Apache-2.0.** The full text ships as [`LICENSE`](../LICENSE) at the
package root (and at the repository root); the copyright line reads `2026 The Project Bonsai
Authors`. Apache-2.0 was chosen over MIT because its explicit patent grant matters for an inference
runtime. Everything written from scratch for this project is covered by it; the third-party pieces
below keep their own terms.

## What is in this package and where it came from

| Component | Origin | Licence |
|---|---|---|
| `bonsai_vk.exe`, `bonsai_fleet.exe`, `align_*.exe`, `shared_probe.exe` | written from scratch for this project (see the `*.md` docs for the line-by-line provenance claims) | Apache-2.0 |
| `shaders/*.spv` | compiled by `glslang` 16.6.0 from this repository's own `.comp` / `.glslh` sources; **51 of 52 are ours**. The exception is `dequant_ptq1_0_ref.spv`, which is the official PrismML reference kernel compiled verbatim as a measuring stick — it is used only by `fidelity/align_dequant.exe`, never by the engine's forward path | reference kernel: PrismML upstream terms |
| `vulkan-1.dll` | **not shipped.** Comes from the installed AMD display driver / Vulkan loader | driver licence, not ours to redistribute |
| `libgcc_s_seh-1.dll`, `libstdc++-6.dll`, `libwinpthread-1.dll` | MinGW-w64 runtime (WinLibs GCC 16.2 UCRT) | GCC runtime library exception (linking exception), so shipping them next to the exe is permitted |
| model weights (`Ternary-Bonsai-2-9B-PTQ1_0.gguf`) | **not shipped** — download from the public PrismML Ternary Bonsai release and pass the path as the first argument | the model's own release terms, which are not this repository's to restate |

## Third-party code in the runtime

The forward pass, tokenizer, GGUF reader, dequant path, attention kernels and command-buffer
orchestration contain **no code copied from any inference framework**. The reference arithmetic was
read, transcribed into a C++ mirror and independently tested; where a reference kernel is executed
at all (`dequant_ptq1_0_ref.spv`) it is executed only as an oracle inside a fidelity tool.

## Chat-template note

`bonsai_chat.h` transcribes the chat template string that ships **inside the GGUF metadata** of the
checkpoint. It is the model's own text, reproduced as data so the engine needs no template parser;
the four control ids are the numbers the checkpoint declares.
