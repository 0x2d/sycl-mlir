# Building sycl-mlir for AMD GPUs on the DTK Cluster

This document describes how to build the `sycl-mlir` compiler with HIP/AMD backend support on this cluster (DTK 25.04.1, gfx906 / DCU). It captures the workarounds for several environment quirks that the stock `buildbot/configure.py` does not handle.

## Prerequisites

| Component | Path / Version |
|-----------|----------------|
| GCC ≥ 7.4 | `/public/home/liuying/Tools/gcc/11.2.0/bin/{gcc,g++}` (system `/usr/bin/gcc` is 4.8.5, too old) |
| Python 3.6+ | conda env `oy_dev` at `/public/home/liuying/anaconda3/envs/oy_dev` |
| DTK (ROCm fork) | `/public/software/compiler/dtk/dtk-25.04.1/` (loaded via `module load compiler/rocm/dtk/25.04.1`) |
| CMake, Ninja | provided by the conda env |

## Why the stock build.sh isn't enough

Three issues bite on this cluster:

1. **Non-interactive shells can't `conda activate`** — `~/.bashrc` only initializes conda for interactive shells. Build scripts must source `conda.sh` explicitly.
2. **`buildbot/configure.py` hardcodes `/usr/bin/gcc`** — it ignores `$CC`/`$CXX`, so `setgcc.sh` alone isn't enough. You must pass `--build-compiler-c` / `--build-compiler-cpp` explicitly.
3. **DTK's library and header layout differ from stock ROCm** — `libamdhip64.so` is in `lib/`, `libamd_comgr.so` is in `lib64/`, and `amd_comgr.h` is in `include/` (flat) instead of `include/amd_comgr/`. The SYCL HIP plugin can't find them out of the box.

Additionally, the original `build.sh` used `-DSYCL_BUILD_PI_HIP_AMD_LIBRARY=...` — that's not a real CMake variable; CMake silently warned and fell back to `/opt/rocm/lib/libamdhip64.so`. The correct variable is `SYCL_BUILD_PI_HIP_LIB_DIR` (a *directory*).

## One-time setup

These shim directories let the HIP plugin find DTK's libraries and headers without modifying the read-only DTK install.

```bash
BUILD=/public/home/liuying/sycl-mlir/build
DTK=/public/software/compiler/dtk/dtk-25.04.1

# Merged lib dir: DTK splits libs across lib/ and lib64/
mkdir -p $BUILD/dtk-libs
ln -sf $DTK/lib/libamdhip64.so   $BUILD/dtk-libs/
ln -sf $DTK/lib/libamdhip64.so.4 $BUILD/dtk-libs/
ln -sf $DTK/lib/libamdhip64.so.5 $BUILD/dtk-libs/
ln -sf $DTK/lib64/libamd_comgr.so       $BUILD/dtk-libs/
ln -sf $DTK/lib64/libamd_comgr.so.2     $BUILD/dtk-libs/
ln -sf $DTK/lib64/libamd_comgr.so.2.4.0 $BUILD/dtk-libs/

# Header shim: DTK ships include/amd_comgr.h flat, plugin expects amd_comgr/amd_comgr.h
mkdir -p $BUILD/dtk-include/amd_comgr
ln -sf $DTK/include/amd_comgr.h $BUILD/dtk-include/amd_comgr/amd_comgr.h
```

## CMakeLists patches

The HIP plugin's CMake doesn't add the `amd_comgr` include path on its own. The plugin in `sycl/plugins/hip/CMakeLists.txt` exposes a cache variable `SYCL_BUILD_PI_HIP_EXTRA_INCLUDE_DIRS` for this — pass it at configure time (see the build script below). No source edit is needed there.

The unified-runtime adapter under `build/_deps/` does still need a one-line edit:

**`build/_deps/unified-runtime-src/source/adapters/hip/CMakeLists.txt`** — extend the `target_include_directories(${TARGET_NAME} PRIVATE ...)` call:

```cmake
target_include_directories(${TARGET_NAME} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../"
    "/public/home/liuying/sycl-mlir/build/dtk-include"
)
```

> ⚠️  This file lives under `build/_deps/`, so it will be regenerated if FetchContent re-fetches the unified-runtime sources (e.g. after a clean rebuild that re-pulls the dependency). Reapply the edit if that happens, or upstream it.

## Build script

`build/build.sh` is the canonical entry point. Key points to copy if you write your own:

```bash
#!/bin/bash
set -e

# 1. Modules
module unload compiler/devtoolset/7.3.1
module unload mpi/hpcx/2.11.0/gcc-7.3.1
module unload compiler/rocm/dtk/22.10.1
module load compiler/rocm/dtk/25.04.1

# 2. Conda — MUST source profile.d/conda.sh in non-interactive shells
source /public/home/liuying/anaconda3/etc/profile.d/conda.sh
conda activate oy_dev

# 3. GCC 11.2 (system gcc is 4.8.5)
source ~/Tools/setgcc.sh

# 4. Configure — note --build-compiler-c/cpp and SYCL_BUILD_PI_HIP_LIB_DIR
python /public/home/liuying/sycl-mlir/buildbot/configure.py \
    --hip \
    --build-compiler-c=/public/home/liuying/Tools/gcc/11.2.0/bin/gcc \
    --build-compiler-cpp=/public/home/liuying/Tools/gcc/11.2.0/bin/g++ \
    --cmake-opt=-DSYCL_BUILD_PI_HIP_INCLUDE_DIR=/public/software/compiler/dtk/dtk-25.04.1/hip/include \
    --cmake-opt=-DSYCL_BUILD_PI_HIP_HSA_INCLUDE_DIR=/public/software/compiler/dtk/dtk-25.04.1/hsa/include \
    --cmake-opt=-DSYCL_BUILD_PI_HIP_LIB_DIR=/public/home/liuying/sycl-mlir/build/dtk-libs \
    --cmake-opt=-DSYCL_BUILD_PI_HIP_EXTRA_INCLUDE_DIRS=/public/home/liuying/sycl-mlir/build/dtk-include

# 5. Build
python /public/home/liuying/sycl-mlir/buildbot/compile.py -j 32
```

Run with logging:

```bash
cd /public/home/liuying/sycl-mlir/build
bash build.sh > build.log 2>&1
```

Expect ~10–15 min for configure (large CMake graph on this filesystem) and ~1–2 hr for the full compile on 32 cores. The build target is `deploy-sycl-toolchain` which also runs `install`.

## Build sequence (if a step fails)

If a particular step needs a re-run without redoing configure, drive ninja directly with the right env:

```bash
source /public/home/liuying/anaconda3/etc/profile.d/conda.sh
conda activate oy_dev
source ~/Tools/setgcc.sh
ninja -C /public/home/liuying/sycl-mlir/build deploy-sycl-toolchain -j 32
```

There is one known build-graph race: `clang/lib/Driver/Driver.cpp` includes `llvm/SYCLLowerIR/DeviceConfigFile.inc` (tablegen-generated) without an explicit dependency. If you hit `fatal error: llvm/SYCLLowerIR/DeviceConfigFile.inc: No such file or directory`, build that target first then resume:

```bash
ninja -C build DeviceConfigFile -j 32
ninja -C build deploy-sycl-toolchain -j 32
```

## Verification

After a successful build:

```bash
$ source ~/Tools/setgcc.sh   # needed for libstdc++ at runtime
$ /public/home/liuying/sycl-mlir/build/install/bin/clang++ --version
clang version 18.0.0git ...
Target: x86_64-unknown-linux-gnu

$ ls /public/home/liuying/sycl-mlir/build/install/lib/libpi_hip.so
$ ls /public/home/liuying/sycl-mlir/build/install/lib/clc/libspirv-amdgcn--amdhsa.bc
$ /public/home/liuying/sycl-mlir/build/install/bin/cgeist --version
```

## Compiling SYCL code for gfx906

Once the toolchain is built, compile a SYCL program targeting the cluster's DCU (gfx906):

```bash
module unload compiler/devtoolset/7.3.1
module unload mpi/hpcx/2.11.0/gcc-7.3.1
module unload compiler/rocm/dtk/22.10.1
module load compiler/rocm/dtk/25.04.1
source ~/Tools/setgcc.sh
export PATH=/public/home/liuying/sycl-mlir/build/install/bin:$PATH
export LD_LIBRARY_PATH=/public/home/liuying/sycl-mlir/build/install/lib:$LD_LIBRARY_PATH

clang++ -fsycl \
        --gcc-toolchain=/public/home/liuying/Tools/gcc/11.2.0 \
        -fsycl-targets=amdgcn-amd-amdhsa-syclmlir \
        -Xsycl-target-backend --offload-arch=gfx906 \
        your_code.cpp -o your_code
```

At runtime, `libpi_hip.so` (under `build/install/lib/`) loads DTK's `libamdhip64.so`, so the DTK module must be loaded.

## Run SYCL code on ORISE

```bash
#!/bin/bash
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --ntasks-per-node=1
#SBATCH --gres=dcu:1

module unload compiler/devtoolset/7.3.1 2>/dev/null
module unload mpi/hpcx/2.11.0/gcc-7.3.1 2>/dev/null
module unload compiler/rocm/dtk/22.10.1 2>/dev/null
module load compiler/rocm/dtk/25.04.1

source ~/Tools/setgcc.sh
unset CPLUS_INCLUDE_PATH C_INCLUDE_PATH

export DTK_HOME=/public/software/compiler/dtk/dtk-25.04.1
export LD_LIBRARY_PATH=/public/home/liuying/sycl-mlir/build/install/lib:$DTK_HOME/lib:$DTK_HOME/lib64:$DTK_HOME/hip/lib:$DTK_HOME/llvm/lib:/public/home/liuying/Tools/gcc/11.2.0/lib64:$LD_LIBRARY_PATH
export PATH=$DTK_HOME/bin:$DTK_HOME/hip/bin:$PATH
export OMP_NUM_THREADS=6

echo "=== hostname ==="; hostname
echo "=== sycl-ls ==="
ONEAPI_DEVICE_SELECTOR=hip:* /public/home/liuying/sycl-mlir/build/install/bin/sycl-ls 2>&1 || true
echo "=== Run a.out ==="
ONEAPI_DEVICE_SELECTOR=hip:* SYCL_PI_TRACE=2 ./a.out
echo "EXIT=$?"
```

## Build vs install gotcha

Per the project's CLAUDE.md: `ninja cgeist` rebuilds `build/bin/cgeist` only — there is no `install-cgeist` target. After iterating on cgeist (or `clang++`, `opt`, `llvm-spirv`, `llvm-dis`, `FileCheck`), copy the freshly-built binary into `install/bin/` before running downstream benchmarks, otherwise you'll silently test a stale binary:

```bash
cp build/bin/cgeist build/install/bin/cgeist
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Run 'conda init' before 'conda activate'` | non-interactive shell | source `/public/home/liuying/anaconda3/etc/profile.d/conda.sh` first |
| `'module' object has no attribute 'quote'` | fell through to system Python 2.7 | conda activation failed — see above |
| `Host GCC version must be at least 7.4, your version is 4.8.5` | configure.py used `/usr/bin/gcc` | pass `--build-compiler-c=...gcc-11.2.0/bin/gcc --build-compiler-cpp=...g++` |
| `Manually-specified variables were not used by the project: SYCL_BUILD_PI_HIP_AMD_LIBRARY` | bogus cmake var | use `-DSYCL_BUILD_PI_HIP_LIB_DIR=<dir>` instead (a directory, not a file) |
| `ninja: error: '/opt/rocm/lib/libamdhip64.so', needed by 'lib/libpi_hip.so', missing` | plugin fell back to default ROCm path | same as above — `SYCL_BUILD_PI_HIP_LIB_DIR` is missing or wrong |
| `libamd_comgr.so ... missing and no known rule to make it` | DTK keeps it in `lib64/`, not `lib/` | use the merged `dtk-libs/` shim dir |
| `fatal error: amd_comgr/amd_comgr.h: No such file or directory` | DTK ships header flat at `include/amd_comgr.h` | use the `dtk-include/amd_comgr/` shim and add the include path to both CMakeLists |
| `clang++: ... GLIBCXX_3.4.29 not found` at runtime | running with system libstdc++ | `source ~/Tools/setgcc.sh` before invoking the built binaries |
| `fatal error: llvm/SYCLLowerIR/DeviceConfigFile.inc: No such file or directory` | tablegen build-graph race | `ninja DeviceConfigFile` first, then resume |
