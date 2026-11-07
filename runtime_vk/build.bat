@echo off
REM build.bat - compile the GLSL compute shaders and the Vulkan host.
REM Free toolchain: Vulkan SDK (glslangValidator + headers + vulkan-1.lib) + Visual Studio
REM Build Tools (cl.exe), or MinGW-w64 (g++). Run from runtime_vk\.

if defined VULKAN_SDK call "%VULKAN_SDK%\Bin\setupEnv.bat"

echo == compiling shaders ==
if not exist shaders\spv mkdir shaders\spv
for %%f in (shaders\*.comp) do (
  echo    %%~nxf
  glslangValidator -V -O "%%f" -o "shaders\%%~nf.spv" || exit /b 1
)

echo == compiling host ==
where cl >nul 2>nul
if %errorlevel%==0 (
  cl /nologo /std:c++17 /O2 /EHsc bonsai_vk.cpp /link vulkan.lib bonsai_vk.exe || exit /b 1
) else (
  echo no cl.exe, trying g++ (MinGW-w64)
  g++ -std=c++17 -O2 -m64 bonsai_vk.cpp -o bonsai_vk.exe -lvulkan-1 || exit /b 1
)

echo == run ==
REM   bonsai_vk.exe  M  K  splits  iterations
REM   shapes of a hidden=4096 / ffn=12288 dense 9B:
REM     12288 4096 2     (gate+up)      expect ~150 Gop/s
REM      4096 4096 4     (q/o)          expect ~124 Gop/s
REM      4096 12288 8    (down)         expect ~144 Gop/s
set VK_VALIDATION=1
bonsai_vk.exe 12288 4096 2 100
