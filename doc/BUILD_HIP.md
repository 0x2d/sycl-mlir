# Building sycl-mlir for AMD GPUs on the DTK Cluster

This document describes how to build the `sycl-mlir` compiler with HIP/AMD backend support on this cluster (DTK 25.04.1, gfx906 / DCU). It captures the workarounds for several environment quirks that the stock `buildbot/configure.py` does not handle.

## Prerequisites

| Component | Path / Version |
|-----------|----------------|
| GCC ≥ 7.4 | `/public/home/liuying/Tools/gcc/11.2.0/bin/{gcc,g++}` (system `/usr/bin/gcc` is 4.8.5, too old) |
| Python 3.6+ | conda env `oy_dev` at `/public/home/liuying/anaconda3/envs/oy_dev` |
| DTK (ROCm fork) | `/public/software/compiler/dtk/dtk-25.04.1/` (loaded via `module load compiler/rocm/dtk/25.04.1`) |
| CMake, Ninja | provided by the conda env |

由于OpenCL-Headers最新Commit与sycl-mlir不兼容，因此需要手动选择旧版本，修改 OpenCL-Headers.git/refs/heads/main 内容为 8275634cf9ec31b6484c2e6be756237cb583999d

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

module unload compiler/devtoolset/7.3.1
module unload mpi/hpcx/2.11.0/gcc-7.3.1
module unload compiler/rocm/dtk/22.10.1
module load compiler/rocm/dtk/25.04.1
source ~/Tools/setgcc.sh

export LD_LIBRARY_PATH=/public/home/liuying/sycl-mlir/build/install/lib:$LD_LIBRARY_PATH

ONEAPI_DEVICE_SELECTOR=hip:* ./a.out
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

---

# `SYCL_AMDGCN_OPT` / `SYCL_AMDGCN_OPT_FLAGS` / `SYCL_AMDGCN_LLC`

This section describes three environment variables added to the clang SYCL
driver for the `amdgcn-amd-amdhsa-syclmlir` target: what they do, why they
exist, and how to use them (in particular, to drive an external DTK `opt`/`llc`
from a normal cmake build).

## TL;DR

These variables let a cmake-driven SYCL-AMDGCN build swap two pipeline tools for
externally-provided ones — without editing any build script and without
rebuilding the toolchain. All three are **off by default**: when unset, the
driver behaves exactly as before.

| Variable | Purpose | Default |
|---|---|---|
| `SYCL_AMDGCN_OPT` | Path to an external `opt`. **Presence enables** a new `opt` pass step over each split per-kernel device bitcode file. | unset (no opt step) |
| `SYCL_AMDGCN_OPT_FLAGS` | Pass arguments handed to that `opt`. Whitespace-split. | `-O2` |
| `SYCL_AMDGCN_LLC` | Path to an external `llc` used to compile the host offload-wrapper `.bc` into a host object. | unset → the `llc` next to the `clang` driver binary |

```bash
# Point a cmake build at the DTK opt/llc:
export SYCL_AMDGCN_OPT=/public/software/compiler/dtk/dtk-25.04.2/dcc/bin/opt
export SYCL_AMDGCN_LLC=/public/software/compiler/dtk/dtk-25.04.2/dcc/bin/llc
export SYCL_AMDGCN_OPT_FLAGS="-O2"     # optional; this is the default
cmake ... -DCMAKE_CXX_COMPILER=/public/home/liuying/sycl-mlir/build/install/bin/clang++ ...
```

## What they do

### `SYCL_AMDGCN_OPT` — insert an `opt` pass over device bitcode

The SYCL-AMDGCN device pipeline (after `sycl-post-link -split=auto` and
`file-table-tform -extract=Code`) produces a **file list** of split per-kernel
device `.bc` files, which `llvm-foreach` then feeds one-by-one to
`clang-18 -cc1 -x ir`. When `SYCL_AMDGCN_OPT` is set, the driver inserts an
**additional** `llvm-foreach`-wrapped `opt` step that runs once per split `.bc`,
between the extract step and the backend `clang-18 -cc1 -x ir`:

```
... → file-table-tform -extract=Code  (emits file list of *_0.bc, *_1.bc, …)
   → llvm-foreach -- <opt> <flags> <in.bc> -o <out.bc>      ← NEW (per kernel)
   → llvm-foreach -- clang-18 -cc1 -x ir <opt-out.bc>       (device backend)
   → llvm-foreach -- lld ...                                (device link)
   → ...
```

Because the new action lives inside the existing `ForEachWrappingAction`, it
inherits the per-split-file iteration for free — `opt` runs once per kernel
module, exactly as if you had written the `llvm-foreach -- opt …` line by hand.

The value of the variable is the **opt binary path**; it is passed straight to
the spawned command. (If the variable is set but empty, the tool falls back to
`GetProgramPath("opt")`, i.e. `-B`/`PATH` lookup.)

### `SYCL_AMDGCN_OPT_FLAGS` — opt pass arguments

The arguments handed to `opt` between the binary and the input file. Split on
whitespace (spaces/tabs). Defaults to `-O2` when the variable is unset. The
resulting command is:

```
<SYCL_AMDGCN_OPT> <SYCL_AMDGCN_OPT_FLAGS...> <in.bc> -o <out.bc>
```

This is the knob for swapping in a different optimization level or a custom pass
pipeline (e.g. `"-passes=default<O2>,some-pass"`) without recompiling clang.

### `SYCL_AMDGCN_LLC` — the host offload-wrapper `llc`

The SYCL flow wraps the linked device binary into a host object via
`clang-offload-wrapper` (producing a host-side `wrapper.bc`) followed by `llc`:

```
clang-offload-wrapper -o=wrapper.bc ... table
llc -filetype=obj -o a-...o wrapper.bc -relocation-model=pic
```

Historically this `llc` was hardcoded to `<driver-dir>/llc` (the `llc` sitting
next to `clang`), which bypassed `GetProgramPath` and could not be redirected
with `-B` or `PATH`. When `SYCL_AMDGCN_LLC` is set, that path is replaced by its
value. When unset, the original default is used unchanged.

## Why environment variables

- **Off by default.** Existing AMDGCN users (and the SPIR/NVPTX/native-CPU SYCL
  targets, which are not touched at all) see no change. This was verified: with
  all three variables unset, `clang++ -###` emits a pipeline byte-identical to
  the pre-change expansion (no `opt` step, install-dir `llc`).
- **No rebuild to reconfigure.** A cmake project can flip between the in-tree
  `opt`/`llc` and an external DTK pair by exporting/clearing the variables.
- **No new driver flags.** The variables are read with
  `llvm::sys::Process::GetEnv` at the points where the commands are built, so
  there is no `Options.td` surface to maintain and no flag-forwarding plumbing
  through `-Xsycl-target-backend` etc.

## How to use it

### From a cmake build

Export the variables in the cmake environment (or a toolchain file) so they are
inherited by every `clang++` invocation:

```bash
module load compiler/rocm/dtk/25.04.2
source ~/Tools/setgcc.sh
export SYCL_AMDGCN_OPT=/public/software/compiler/dtk/dtk-25.04.2/dcc/bin/opt
export SYCL_AMDGCN_LLC=/public/software/compiler/dtk/dtk-25.04.2/dcc/bin/llc
export SYCL_AMDGCN_OPT_FLAGS="-O2"

cmake -S . -B build \
      -DCMAKE_CXX_COMPILER=/public/home/liuying/sycl-mlir/build/install/bin/clang++ \
      ...
cmake --build build
```

Only `SYCL_AMDGCN_OPT` need be set to get the opt step; `SYCL_AMDGCN_LLC` is
independent and can be set alone if you only want to swap the wrapper `llc`.

### Inspecting what the driver emits

Use `-###` to dry-run the pipeline and confirm the tools are picked up:

```bash
export SYCL_AMDGCN_OPT=.../dcc/bin/opt
export SYCL_AMDGCN_LLC=.../dcc/bin/llc
clang++ -fsycl -fsycl-targets=amdgcn-amd-amdhsa-syclmlir \
        -Xsycl-target-backend --offload-arch=gfx906 app.cpp -o app -###
```

You should see, in order:

```
".../file-table-tform" "-extract=Code" ...
".../llvm-foreach" ... "--" "<SYCL_AMDGCN_OPT>" "<flags>" "<in.bc>" "-o" "<out.bc>"
".../llvm-foreach" ... "--" ".../clang-18" "-cc1" ... "-x" "ir" "<out.bc>"
...
"<SYCL_AMDGCN_LLC>" "-filetype=obj" "-o" "..." "wrapper.bc" "-relocation-model=pic"
```

With the variables unset, the `llvm-foreach -- opt` line is absent and the final
`llc` is `<driver-dir>/llc`.

### Custom opt passes

```bash
# New-PM pipeline instead of -O2:
export SYCL_AMDGCN_OPT_FLAGS="-passes=default<O2>"
# Or a single pass with an option:
export SYCL_AMDGCN_OPT_FLAGS="--mem2reg"
```

## Where it lives in the code

The implementation follows the existing custom-SYCL-tool pattern
(`SYCLPostLink`, `FileTableTform`, …): a new `JobAction` subclass dispatched
through `ToolChain::getTool`, plus a new `Tool` subclass.

- **`OptJobAction`** — `clang/include/clang/Driver/Action.h:834` (enum entry
  `OptJobClass` at `Action.h:84`), defined in `clang/lib/Driver/Action.cpp:531`
  (name string `"opt"` at `Action.cpp:57`).
- **`Opt` tool** — `clang/lib/Driver/ToolChains/Clang.h:248`,
  `Opt::ConstructJob` at `clang/lib/Driver/ToolChains/Clang.cpp:10415`. Reads
  `SYCL_AMDGCN_OPT` (binary path) and `SYCL_AMDGCN_OPT_FLAGS` (pass args,
  default `-O2`); emits `opt <flags> <in.bc> -o <out.bc>`.
- **`ToolChain::getOpt`** — `clang/include/clang/Driver/ToolChain.h:188` /
  `clang/lib/Driver/ToolChain.cpp:549`, dispatched from `getTool` at
  `ToolChain.cpp:634`.
- **Pipeline insertion** — `clang/lib/Driver/Driver.cpp:4879`, inside
  `finalizeAMDGCNDependences`. Gated on `SYCL_AMDGCN_OPT`:
  ```cpp
  if (llvm::sys::Process::GetEnv("SYCL_AMDGCN_OPT"))
    Input = C.MakeAction<OptJobAction>(Input, types::TY_LLVM_BC);
  ```
  The new action becomes the head of the per-kernel device action chain, which
  `ForEachWrappingAction` (`Driver.cpp:5724`) runs once per split `.bc`.
- **`llc` redirect** — `clang/lib/Driver/ToolChains/Clang.cpp:9850` (inside
  `OffloadWrapper::ConstructJob`). Reads `SYCL_AMDGCN_LLC`; falls back to
  `<driver-dir>/llc` when unset.

Only the AMDGCN branch is modified. NVPTX, SPIR, and SYCL-native-CPU targets are
unchanged.

## Notes / caveats

- **Toolchain mismatch.** An external `opt`/`llc` (e.g. the DTK pair) is built
  against its own LLVM, while the surrounding pipeline uses this fork's LLVM 18.
  The IR is generally compatible at `-O2` / object emission for AMDGCN, but if
  `opt` rejects unknown metadata or `llc` rejects the wrapper module, narrow
  `SYCL_AMDGCN_OPT_FLAGS` (or unset `SYCL_AMDGCN_OPT` to drop the opt step)
  rather than widening the pipeline.
- **`opt` runs on the post-split device IR** (after `sycl-post-link` lowering
  and splitting), not on the pre-split linked module. This matches the
  hand-validated `llvm-foreach -- opt` placement in `build-hip-dtk.sh`.
- **The wrapper `llc` compiles host IR** (the `clang-offload-wrapper` output is
  x86_64 host code that embeds the device binary), so `SYCL_AMDGCN_LLC` should
  point at an `llc` capable of emitting a host object — the DTK `llc` works.
- There is a separate, **dead** `SYCL::Linker::constructLlcCommand`
  (`clang/lib/Driver/ToolChains/SYCL.cpp:495`) left over from the legacy
  SPIR/native-CPU path; it has no callers and is not affected by
  `SYCL_AMDGCN_LLC`.

## Build / iteration note

These variables are read by the **clang driver** (`clang-18`), so after editing
the driver sources you must rebuild and refresh the installed `clang-18`:

```bash
ninja -C build clang && cp build/bin/clang-18 build/install/bin/clang-18
```

(See `CLAUDE.md` → "Build vs install". `cgeist` and the other tools are not
touched by this change.)

## Not to be confused with

- `-Xsycl-target-backend` forwards extra args to the AOT backend compiler; it is
  **not** a tool-path override and does not interact with these variables.
- `ROCM_PATH` / `HIP_PATH` / `HIP_DEVICE_LIB_PATH` (read in
  `clang/lib/Driver/ToolChains/AMDGPU.cpp`) govern device-library *bitcode*
  discovery, not the `opt`/`llc` binaries.
- The `opt` inserted here is a **driver pipeline step** over device bitcode. It
  is unrelated to cgeist's own optimization PM (see
  [doc/sycl_fusion_option.md](sycl_fusion_option.md)) and to the MLIR pass
  pipelines run inside cgeist/polygeist-opt.
