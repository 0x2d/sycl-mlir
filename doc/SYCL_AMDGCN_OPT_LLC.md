# `SYCL_AMDGCN_OPT` / `SYCL_AMDGCN_OPT_FLAGS` / `SYCL_AMDGCN_LLC`

This document describes three environment variables added to the clang SYCL
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
