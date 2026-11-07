@echo off
REM build_mingw.bat - compile the whole runtime with nothing but %DEV%.
REM No Visual Studio, no Vulkan SDK, no admin, no C: writes. Run from anywhere; paths are absolute.
REM
REM   %DEV%\mingw64          WinLibs GCC 16.2 UCRT (ships g++ 16.2 + cmake 4.4.4 + ninja + dlltool)
REM   %DEV%\vkheaders\...     Vulkan-Headers 1.4.309 (the only header set we need)
REM   %DEV%\vklib\libvulkan-1.a   import lib generated from C:\Windows\System32\vulkan-1.dll:
REM                                gendef vulkan-1.dll  ->  vulkan-1.def
REM                                dlltool --input-def vulkan-1.def --output-lib libvulkan-1.a --dllname vulkan-1.dll
REM                              (a MinGW import lib is NOT shipped by mingw-w64 and NOT by the driver;
REM                               if this file is missing the link step fails with "cannot find -lvulkan-1")
REM   %DEV%\glslang\bin\glslang.exe   glslang 16.6.0; note it is called glslang.exe, NOT glslangValidator.exe
REM
setlocal
if not defined DEV (
  echo [!] Set DEV to your toolchain root: the folder holding mingw64\, vkheaders\, vklib\, glslang\.
  exit /b 1
)
set MINGW=%DEV%\mingw64\bin
set HDR=%DEV%\vkheaders\Vulkan-Headers-1.4.309\include
set VKLIB=%DEV%\vklib
set GSLANG=%DEV%\glslang\bin\glslang.exe
set HERE=%~dp0

if not exist "%MINGW%\g++.exe"       echo [!] %MINGW%\g++.exe missing - toolchain not unpacked & exit /b 1
if not exist "%HDR%\vulkan\vulkan.h" echo [!] %HDR%\vulkan\vulkan.h missing & exit /b 1
if not exist "%VKLIB%\libvulkan-1.a" echo [!] %VKLIB%\libvulkan-1.a missing - rebuild it with gendef+dlltool & exit /b 1
if not exist "%GSLANG%"              echo [!] %GSLANG% missing & exit /b 1
set PATH=%MINGW%;%PATH%

pushd "%HERE%"
if not exist shaders\spv mkdir shaders\spv 2>nul

echo == 0/2 reference oracle: inline the official dequant head ==
REM   shaders\dequant_ptq1_0_ref.comp = official _dequant_ptq1_0.comp body, verbatim, with our
REM   dequant_head.glsl inlined (glslang will not honour the fork's build-time #include).
REM   This shader is the ORACLE for the dequant fidelity test - see FAITHFUL_REPRO.md.
python ..\tools\gen_dequant_ref_comp.py || (echo [!] oracle source generation failed & popd & exit /b 1)

echo == 1/2 shaders: GLSL -^> SPIR-V ==
for %%f in (shaders\*.comp) do (
  echo     %%~nxf
  "%GSLANG%" -V -O "%%f" -o "shaders\%%~nf.spv" || (echo [!] glslang failed on %%f & popd & exit /b 1)
)

echo == 2/2 host: g++ -^> bonsai_vk.exe + the alignment tests + the fleet driver ==
REM -Wall is on purpose: the two most vexing parse warnings in this file were real bugs.
g++ -std=c++17 -O2 -m64 -Wall bonsai_vk.cpp -I"%HDR%" -o bonsai_vk.exe -L"%VKLIB%" -lvulkan-1 -lpdh -ldxgi -lws2_32 -lshell32 || (echo [!] compile failed & popd & exit /b 1)
REM   -lpdh = kv_guard.h's SpillWatch (PDH "GPU Process Memory(pid_*)\Shared Usage" - the WDDM spill
REM   judge).  -ldxgi = the auxiliary IDXGIAdapter3::QueryVideoMemoryInfo budget reading.
REM   -lws2_32 = bonsai_http.h's socket layer (Winsock).  -lshell32 = ShellExecuteA, "open the default
REM   browser" on double-click. All four import libs ship in %DEV%\mingw64; the four align_*.exe and
REM   bonsai_fleet.exe include neither kv_guard.h nor bonsai_http.h and are not relinked.
REM   On Linux the same engine needs -lvulkan -lpthread and NONE of these four: build_linux.sh.
echo     bonsai_vk.exe OK
REM   faithful-reproduction tests (FAITHFUL_REPRO.md). align_dequant runs the OFFICIAL dequant
REM   shader, so it needs the .spv from step 1/2 and must be started from runtime_vk\.
g++ -std=c++17 -O2 -m64 -Wall -Wno-unused-variable align_tokenizer.cpp -I"%HDR%" -o align_tokenizer.exe || (echo [!] align_tokenizer failed & popd & exit /b 1)
echo     align_tokenizer.exe OK
g++ -std=c++17 -O2 -m64 -Wall align_dequant.cpp -I"%HDR%" -o align_dequant.exe -L"%VKLIB%" -lvulkan-1 || (echo [!] align_dequant failed & popd & exit /b 1)
echo     align_dequant.exe OK
g++ -std=c++17 -O2 -m64 -Wall align_f16.cpp -I"%HDR%" -o align_f16.exe -L"%VKLIB%" -lvulkan-1 || (echo [!] align_f16 failed & popd & exit /b 1)
echo     align_f16.exe OK
REM   层3: the attention core. attn_vk.h is the CPU mirror AND the ideal legs, the three
REM   attn_*.comp kernels are the GPU side, so align_attn.exe measures kernel vs reference on the
REM   card at the real layer-3 shapes (head_dim 256, 16/4 heads, fp16 arena in our own buffer).
g++ -std=c++17 -O2 -m64 -Wall -Wno-unused-variable align_attn.cpp -I"%HDR%" -o align_attn.exe -L"%VKLIB%" -lvulkan-1 || (echo [!] align_attn failed & popd & exit /b 1)
echo     align_attn.exe OK
REM   shared_mem_probe: does this driver create AND correctly run a pipeline whose static shared
REM   allocation is exactly maxComputeSharedMemorySize (32768 B)? That is the 4 K question of
REM   attn_qk_softmax_wide.comp's MAXCAP, measured instead of derived. See shared_probe.cpp.
g++ -std=c++17 -O2 -m64 -Wall shared_probe.cpp -I"%HDR%" -o shared_probe.exe -L"%VKLIB%" -lvulkan-1 || (echo [!] shared_probe failed & popd & exit /b 1)
echo     shared_probe.exe OK
REM   align_27b: the 27B adapter. PTQ1_0's 5-trit packing measured against a closed form, the decode
REM   triangulated over all 65 536 fp16 scales AND against the real checkpoint's own scale bytes, the
REM   16-attn/48-ssm KV ledger (an SSM layer must charge exactly 0), the prism.hadamard.version gate,
REM   and hadamard_fwht.comp compared BITWISE to a host Sylvester FWHT at every block size the
REM   resolver accepts. CPU legs need no card; the rotation leg does and skips on --no-gpu.
g++ -std=c++17 -O2 -m64 -Wall align_27b.cpp -I"%HDR%" -o align_27b.exe -L"%VKLIB%" -lvulkan-1 || (echo [!] align_27b failed & popd & exit /b 1)
echo     align_27b.exe OK
REM   bonsai_fleet: the multi-card / iGPU driver for gpu_fleet.h. probe = topology + the zero-copy
REM   decision the engine would make; split = the layer-pipeline schedule for 64 blocks over N cards;
REM   ring = the pinned ring's index arithmetic across a wrap; selftest = the legs that need no
REM   second device. What this host CANNOT exercise (one discrete GPU, no iGPU) is printed as such.
g++ -std=c++17 -O2 -m64 -Wall bonsai_fleet.cpp -I"%HDR%" -o bonsai_fleet.exe -L"%VKLIB%" -lvulkan-1 || (echo [!] bonsai_fleet failed & popd & exit /b 1)
echo     bonsai_fleet.exe OK

echo.
echo == run: one token-op of the hidden 4096 / ffn 12288 projection set ==
REM   bonsai_vk.exe  M  K  splits  iterations
REM   BONSAI_DIAG=1      reads the arena back and checks upload / x / xsum / scales / codes
REM   BONSAI_PARTIAL=1   one quant block per split slot, no reduce -> per-block error table
REM   BONSAI_RANDSCALE=1 random fp16 per-block scales (exercises unpackHalf2x16, d != 1.0)
REM   BONSAI_XFER=1      memory types, bulk + scattered copies, transfer rates on the COMPUTE queue
REM   BONSAI_SYNC=1      per-token CPU/GPU bubble in four submit shapes (wall/gpu/host/bubble)
REM   BONSAI_OVERLAP=1   the TRANSFER-ONLY queue: can a copy run under the GEMV, H2D and D2H
REM   BONSAI_PIPE=1      five submit shapes for a WHOLE token (201 ops + 993280 B logits D2H + host
REM                      argmax): A per-op+fence, A2 per-op/1 fence, B 1 CB inline D2H, C 1 CB + SDMA,
REM                      D depth-2 pipeline.  This is the 37%% submission claim turned into measurements.
REM   BONSAI_TXBARRIER=1 probe only: does a lone vkCmdPipelineBarrier on the transfer-only queue survive
REM   BONSAI_API12=1     create the instance at VK_API_VERSION_1_2 so vkGetDeviceProcAddr returns a real
REM                      vkResetQueryPool (under 1.1 it returns NULL - that NULL jump caused the amdvlk SIGSEGV)
REM                      (raw output of all of these is archived in ..\stage3_measured.txt and ..\runs\pipe_measured.txt)
set BONSAI_RANDSCALE=1
bonsai_vk.exe 12288 4096 4 100
popd
endlocal
