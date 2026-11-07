#!/bin/sh
# build_linux.sh - the Linux/macOS counterpart of build_mingw.bat. Same eight hosts, same shaders,
# same -Wall, none of the Windows import libraries.
#
# STATUS, stated before anything else because this project does not report unrun checks as passed:
# THIS SCRIPT HAS NOT BEEN EXECUTED. The development machine is Windows and its WSL has no
# distribution installed, so there is no Linux compiler here to run it with. It is written against
# the documented toolchain and the POSIX branch of bonsai_platform.h / kv_guard.h. The claim is
# "Linux 编译支持已就绪，待实机验证" - compile support is in place, awaiting a real machine.
#
# What a Linux box needs (all of it in the distro's own repositories, nothing vendored):
#   g++ with C++17, glslang (glslangValidator on older packages), the Vulkan headers and loader:
#     Debian/Ubuntu  apt install g++ glslang-tools libvulkan-dev vulkan-tools python3
#     Fedora         dnf install gcc-c++ glslang vulkan-headers vulkan-loader-devel python3
#     Arch           pacman -S gcc glslang vulkan-headers vulkan-icd-loader python
#   plus a driver: RADV (mesa) for AMD, which is the stack the Linux branch was written for.
#
# Differences from the Windows build, and why each one is here:
#   -lvulkan        not -lvulkan-1: the Linux loader's import name has no suffix.
#   -pthread        not -lpdh -ldxgi -lws2_32 -lshell32: the spill judge reads amdgpu sysfs, the
#                   budget reads the same files, the socket layer is POSIX and the browser is opened
#                   with posix_spawn. There is nothing to import.
#   -D_FILE_OFFSET_BITS=64
#                   MANDATORY. bonsai_seek() uses fseeko(off_t) on this side, and the checkpoint is
#                   ~2 GB with tensor offsets added on top. Without it off_t is 32 bits on a 32-bit
#                   userland and the seek wraps silently past 2 GiB - the exact bug align_dequant.cpp
#                   already documents. On a 64-bit userland it is a no-op, which is why it is always on.
#   no -m64         the native ABI is already 64-bit on x86_64/aarch64, and -m64 would break an ARM
#                   build for no gain.
#
# The script does NOT run anything on the GPU. A build that silently grabs the card is a surprise on
# a shared Linux box, and every timing number in this project is taken separately with the field
# cleared. The commands to run afterwards are printed at the end.
#
#   $ BONSAI_MODEL=/path/to/model.gguf ./build_linux.sh
#   $ ./bonsai_vk serve "$BONSAI_MODEL"        # WebUI + OpenAI-compatible API on 127.0.0.1:8080
#
# Override the toolchain/headers without editing this file:
#   CXX=g++-13 GLSLANG=glslang VK_INCLUDE=/opt/Vulkan-Headers/include ./build_linux.sh

set -eu
cd "$(dirname "$0")"

CXX=${CXX:-g++}
PYTHON=${PYTHON:-python3}
VK_INCLUDE=${VK_INCLUDE:-}
MODEL=${BONSAI_MODEL:-}

# glslang is called glslangValidator in the older distro packages and glslang in the newer ones (and
# in the Windows toolchain this project measured with). Accept either rather than guessing.
if [ -n "${GLSLANG:-}" ]; then
  :
elif command -v glslang >/dev/null 2>&1; then
  GLSLANG=glslang
elif command -v glslangValidator >/dev/null 2>&1; then
  GLSLANG=glslangValidator
else
  echo "[!] no glslang / glslangValidator on PATH - install glslang-tools (or set GLSLANG=...)" >&2
  exit 1
fi

command -v "$CXX" >/dev/null 2>&1 || { echo "[!] $CXX not found" >&2; exit 1; }
command -v "$PYTHON" >/dev/null 2>&1 || { echo "[!] $PYTHON not found" >&2; exit 1; }

INC=""
if [ -n "$VK_INCLUDE" ]; then
  [ -f "$VK_INCLUDE/vulkan/vulkan.h" ] || { echo "[!] $VK_INCLUDE/vulkan/vulkan.h missing" >&2; exit 1; }
  INC="-I$VK_INCLUDE"
fi

# -Wall is on purpose: the two most-vexing-parse warnings it produced on Windows were real bugs.
FLAGS="-std=c++17 -O2 -Wall -D_FILE_OFFSET_BITS=64 $INC"

echo "== 0/3 reference oracle: inline the official dequant head =="
"$PYTHON" ../tools/gen_dequant_ref_comp.py

echo "== 1/3 shaders: GLSL -> SPIR-V =="
mkdir -p shaders/spv
for f in shaders/*.comp; do
  echo "    $f"
  "$GLSLANG" -V -O "$f" -o "shaders/$(basename "$f" .comp).spv"
done

echo "== 2/3 hosts: $CXX -> bonsai_vk + the alignment tests + the fleet driver =="
# The engine is the only host that includes kv_guard.h and bonsai_http.h, so it is the only one that
# needs the thread library. On Windows the same line carries -lpdh -ldxgi -lws2_32 -lshell32.
# shellcheck disable=SC2086
$CXX $FLAGS bonsai_vk.cpp -o bonsai_vk -lvulkan -pthread
echo "    bonsai_vk OK"
# shellcheck disable=SC2086
$CXX $FLAGS -Wno-unused-variable align_tokenizer.cpp -o align_tokenizer
echo "    align_tokenizer OK"
for t in align_dequant align_f16 shared_probe align_27b bonsai_fleet; do
  # shellcheck disable=SC2086
  $CXX $FLAGS "$t.cpp" -o "$t" -lvulkan
  echo "    $t OK"
done
# shellcheck disable=SC2086
$CXX $FLAGS -Wno-unused-variable align_attn.cpp -o align_attn -lvulkan
echo "    align_attn OK"

echo "== 3/3 done =="
echo
echo "SPIR-V is loaded from CWD-relative shaders/<name>.spv, so run from this directory."
echo "The spill judge on this side is /sys/class/drm/cardN/device/mem_info_gtt_used (device-wide,"
echo "not per-process); BONSAI_DRM_CARD pins the card. vram_used/vram_total are printed beside it."
echo
if [ -n "$MODEL" ]; then
  echo "  ./bonsai_vk serve \"$MODEL\"     # WebUI + /v1/chat/completions on 127.0.0.1:8080"
  echo "  ./bonsai_vk run   \"$MODEL\" 32 48"
else
  echo "  ./bonsai_vk serve /path/to/Ternary-Bonsai-2-9B-PTQ1_0.gguf"
  echo "  ./bonsai_vk run   /path/to/Ternary-Bonsai-2-9B-PTQ1_0.gguf 32 48"
fi
echo "  ./bonsai_vk 12288 4096 4 100   # the kernel benchmark (no checkpoint needed)"
echo
echo "NOT VERIFIED: this script and the POSIX branch have not been compiled or run by the author."