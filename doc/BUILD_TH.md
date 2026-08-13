# Building sycl-mlir for the MT3K-DSP (mocl/POCL) host

This document covers building the `sycl-mlir` compiler for the AArch64 host
that targets the `MT3K-DSP` device via the mocl3.1.3 / POCL OpenCL backend,
plus the things that affect actually compiling and launching a SYCL program
on the DSP:

1. **`LLVM_ROOT` must be exported** — mocl3.1.3's `env.sh` leaves it unset, and
   POCL's `pocl_dat_gen` aborts at kernel launch without it
   (`basic_string::_M_construct null not valid`). See
   [Fix 1 — `LLVM_ROOT`](#fix-1--llvm_root-must-be-exported).
2. **Kernel-name length** *(mocl3.1.2 only)* — `func _pocl_kernel_…_workgroup
   not found / Set function error`, caused by the mocl3.1.2 MT3X DSP host
   runtime's 63-char symbol-name limit. **This no longer reproduces under
   mocl3.1.3** (verified: a 76-char SPIR-V entry name → 99-char workgroup
   symbol launches correctly); the workaround is retained for the 3.1.2
   fallback only. See
   [Fix 2](#fix-2--dsp-kernel-launch-workgroup-symbol-name-limit).

> **mocl3.1.3 vs mocl3.1.2.** mocl3.1.3 is now the working runtime and
> supersedes the old mocl3.1.2 setup, which needed four post-build hacks:
> a copied `libOpenCL.so.1` into the install tree, `LD_PRELOAD`, a patch
> to `unified-runtime .../opencl/program.cpp` for the `-3`
> (`PI_ERROR_COMPILER_NOT_AVAILABLE`) false-negative guard, **and** the
> short-kernel-name workaround (Fix 2). Under mocl3.1.3 **none** of those are
> needed: the DSP device advertises `cl_khr_il_program` (so the `-3` guard
> passes), the standard ICD architecture (`libpocl.so` + `pocl.icd` + the
> prebuilt `opencl-icd-loader`) replaces the `LD_PRELOAD`/copy hacks, and the
> `libhthread_host.so` loader resolves the full long workgroup symbol names
> (so the 63-char limit does not bite). The only new requirement is
> `LLVM_ROOT` (Fix 1). The mocl3.1.2 hacks are retained at the end as a
> fallback.

## OpenCL-Headers pin

`sycl-mlir/build/_deps/unified-runtime-src/source/adapters/opencl/CMakeLists.txt`
fetches the OpenCL headers via `FetchContent`. The latest `OpenCL-Headers` commit
is incompatible with this sycl-mlir revision, so pin it manually to
`8275634cf9ec31b6484c2e6be756237cb583999d`:

```cmake
FetchContent_Declare(OpenCL-Headers
    GIT_REPOSITORY  "https://github.com/KhronosGroup/OpenCL-Headers.git"
    GIT_TAG         8275634cf9ec31b6484c2e6be756237cb583999d
)
```

由于OpenCL-Headers最新Commit与sycl-mlir不兼容，因此需要手动选择Commit 8275634cf9ec31b6484c2e6be756237cb583999d

## Build

```bash
module load python/3.8.6

python /thfs1/home/ouyyc/sycl-mlir/buildbot/configure.py \
    -o /thfs1/home/ouyyc/sycl-mlir/build \
    --host-target AArch64 \
    --cmake-opt=-DLLVM_DEFAULT_TARGET_TRIPLE=aarch64-unknown-linux-gnu
python /thfs1/home/ouyyc/sycl-mlir/buildbot/compile.py \
    -o /thfs1/home/ouyyc/sycl-mlir/build -j16
```

The canonical entry point is `build/build.sh`.

## Compile a SYCL program

```bash
source /thfs1/software/sycl/mocl3.1.3/env.sh
export LLVM_ROOT=/thfs1/software/llvm/llvm17-mt          # see Fix 1
export MOCL_CORE_NUMS=24
export LD_LIBRARY_PATH=/thfs1/software/sycl/opencl-icd-loader/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/thfs1/home/ouyyc/sycl-mlir/build/install/lib:$LD_LIBRARY_PATH

/thfs1/home/ouyyc/sycl-mlir/build/install/bin/clang++ -fsycl \
    -fsycl-targets=spir64-unknown-unknown-syclmlir \
    simple-sycl-app.cpp -o a.out
```

## Run a SYCL program on the DSP

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=thmt1
#SBATCH --cpus-per-task=4

source /thfs1/software/sycl/mocl3.1.3/env.sh
export LLVM_ROOT=/thfs1/software/llvm/llvm17-mt          # see Fix 1
export MOCL_CORE_NUMS=24
export OCL_ICD_VENDORS=${MOCL3_ROOT}/etc/OpenCL/vendors
export LD_LIBRARY_PATH=/thfs1/software/sycl/opencl-icd-loader/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/thfs1/home/ouyyc/sycl-mlir/build/install/lib:$LD_LIBRARY_PATH

/thfs1/home/ouyyc/Workspace/sycl-test/a.out
```

Submit with `yhbatch run.sh`. Expected output (the `Error opening file:
Permission denied` line is non-fatal — see below):

```
...
Error opening file: Permission denied
...
Selected device: MT3K-DSP Device #0
2345678910111213141516171819202122232425262728293031323334353637383940414243
```

### How the standard ICD path resolves

- `mocl3.1.3/env.sh` puts `libpocl.so` (the POCL implementation), `hthreads`,
  and `hwloc` on `LD_LIBRARY_PATH`, and exports `MOCL3_ROOT`. **It does NOT
  export `LLVM_ROOT` (commented out)** — see Fix 1.
- `OCL_ICD_VENDORS=${MOCL3_ROOT}/etc/OpenCL/vendors` points the ICD loader at
  `pocl.icd`, whose single line names `libpocl.so.2.12.0`; the loader `dlopen`s
  it. No `LD_PRELOAD`.
- The ICD loader itself (`libOpenCL.so.1`, exports `OPENCL_2.2` + `OPENCL_3.0`,
  honors `OCL_ICD_VENDORS`) comes from the prebuilt
  `/thfs1/software/sycl/opencl-icd-loader/lib`. mocl3.1.3 ships only
  `libpocl.so` and does not ship its own `libOpenCL.so.1`.
- The self-built `libpi_opencl.so` (in `build/install/lib`) `NEEDED`s
  `libOpenCL.so.1`; it resolves to the prebuilt loader, which dispatches to
  `libpocl.so`. Nothing is copied into the install tree.

### Non-fatal warnings you will see (do not fix these)

Two `Permission denied` errors print on every run but are tolerated — the
program completes normally:

1. `[MT3X] failed to create lock file. / failed to lock dsp clusters.` — POCL
   tries to `flock("/tmp/pocl_lock.lock", O_RDWR|O_CREAT)` for inter-process
   DSP serialization. A stale `/tmp/pocl_lock.lock` owned by another user
   (mode `0644`, sticky `/tmp`) blocks the open; POCL logs the error and
   proceeds **without** the lock. `/tmp/pocl_lock.lock` is a soft mutual-
   exclusion aid for concurrent processes, not a hard gate — a single job runs
   fine without it.
2. `Error opening file: Permission denied` — POCL's `pocl_dat_gen` (mt3x
   driver, `pocl-dat-gen.cc`) writes a debug dump `parallel_before_scalarizer.ll`
   to a build-time hardcoded path under the mocl author's home
   (`/thfs1/home/penglin_jianbin/.cache/pocl/kcache/`). That dir is foreign and
   unwritable; the dump write fails, POCL logs and continues.

Both are confirmed non-fatal: the stock `sycl_test` (built with the prebuilt
dpcpp) and our sycl-mlir `a.out` both print them and both run to completion.
Do **not** chase these; chasing the lock was a red herring that cost a
mis-attribution (see history below). The real abort was Fix 1.

---

# Fix 1 — `LLVM_ROOT` must be exported

## Symptom (without `LLVM_ROOT`)

The program selects the device, ingests the SPIR-V, creates the kernel, then
aborts at `clEnqueueNDRangeKernel` time:

```
Preparing kernel _ZTS1K ... group sizes 42 x 1 x 1...
Error opening file: Permission denied               ← non-fatal (see above)
############ before pocl_dat_gen : .../parallel.bc
terminate called after throwing an instance of 'std::logic_error'
  what():  basic_string::_M_construct null not valid
Aborted
```

`pocl_dat_gen` is called (the "before" log) but never reaches the "after" log;
no `parallel.bc.dat` is produced; the process throws and aborts.

## Root cause

mocl3.1.3's `env.sh` leaves `LLVM_ROOT` **unset** (lines 23–26 are commented
out — they reference a build-time `/home/lb/...` path). POCL's
`pocl_dat_gen` / LLVM backend in `libpocl-devices-mt3x.so` consults
`getenv("LLVM_ROOT")` to locate the LLVM tools (`llvm-link`/`opt`/`llc`/…)
it shells out to during SPIR-V→workgroup lowering. When the env var is unset,
that code passes a null pointer to a `std::string` constructor, which throws
`basic_string::_M_construct null not valid` → SIGABRT.

The prebuilt dpcpp `example.sh` sets
`export LLVM_ROOT=/thfs1/software/llvm/llvm17-mt`, so it never hit the abort —
which is why the mocl3.1.3 path appeared to work for the dpcpp install but
"mysteriously" failed for sycl-mlir.

## Fix

Export `LLVM_ROOT` (and `MOCL_CORE_NUMS`, matching `example.sh`) in both
`build.sh` and `run.sh`, after sourcing `mocl3.1.3/env.sh`:

```bash
source /thfs1/software/sycl/mocl3.1.3/env.sh
export LLVM_ROOT=/thfs1/software/llvm/llvm17-mt
export MOCL_CORE_NUMS=24
```

Both LLVM trees exist on this host (`/thfs1/software/llvm/llvm17` and
`llvm17-mt`); `llvm17-mt` is the one `example.sh` uses. Verified: with
`LLVM_ROOT` exported, sycl-mlIR `a.out` prints `Selected device: MT3K-DSP
Device #0` + `23456789…4243` and exits 0.

## How this was diagnosed (so the next person doesn't re-chase the lock)

The abort coincides with two `Permission denied` log lines (the lock and the
scalarizer dump). It is tempting to blame the lock. Do not. The decisive
evidence:

- Stock `sycl_test` (prebuilt dpcpp) under mocl3.1.3 prints the **same** two
  errors and **completes** (`0 1 2 … 15`, exit 0) — so both errors are
  non-fatal.
- `strace` of the sycl-mlIR abort showed `pocl_dat_gen` reads `parallel.bc`
  then SIGABRTs **in-process** before writing `parallel.bc.dat`; the stock run
  forks the LLVM-tools child, writes `parallel.bc.dat` + `parallel.bc.out`,
  opens `/dev/mt_debug`, and runs. The divergence is a `std::string(null)`
  throw, classic for an unset env var.
- Diffing the two `program.bc`s showed the same external calls (`get_global_id`,
  `__itt_offload_wi_*`, …) — i.e. not an IR problem. The only material env
  difference was `LLVM_ROOT` (example.sh sets it; mocl3.1.3/env.sh does not).
- Setting `export LLVM_ROOT=/thfs1/software/llvm/llvm17-mt` made the sycl-mlIR
  run complete.

## Why the old mocl3.1.2 fixes (the `-3` patch, `LD_PRELOAD`, the
`libOpenCL.so.1` copy) are obsolete under mocl3.1.3

- **`-3` / `program.cpp` patch.** mocl3.1.3's `libpocl-devices-mt3x.so`
  advertises `cl_khr_il_program` on the DSP device (mocl3.1.2 advertised only
  `cl_khr_spir`). The unified-runtime `urProgramCreateWithIL` guard therefore
  passes and `clCreateProgramWithIL` is called normally — no patch needed. (An
  install still carrying the patched `libpi_opencl.so` works fine on 3.1.3; the
  patch is a no-op there.)
- **`LD_PRELOAD` + copied `libOpenCL.so.1`.** mocl3.1.3 uses the standard ICD
  architecture; the prebuilt `opencl-icd-loader/lib/libOpenCL.so.1` (exports
  `OPENCL_2.2`/`3.0`) on `LD_LIBRARY_PATH` satisfies `libpi_opencl.so`'s
  `NEEDED`, and `OCL_ICD_VENDORS` discovers POCL. Nothing is copied into
  `build/install/lib`.

---

# Fix 2 — DSP kernel-launch (workgroup symbol-name limit)

> **Status under mocl3.1.3: NOT NEEDED.** The 63-char limit documented below
> does **not** reproduce on the mocl3.1.3 `libhthread_host.so`. Verified
> 2026-07-28 on MT3K-DSP: a kernel with a 76-char SPIR-V entry name
> (`_ZTS70ThisIsAReallyLongKernelNameThatExceedsSixtyThreeCharactersLimitForDSPx`)
> → 99-char `_pocl_kernel_…_workgroup` symbol launches correctly — correct
> output (`23456789`), `pocl_dat_gen` before/after both present, no
> `func … not found` / `Set function error`. The 3.1.3 loader keeps the same
> `0x48`-stride / name-at-`0x40` record layout but resolves the full long
> name, so the truncation that broke 3.1.2 no longer happens. **You can keep
> unnamed lambdas under mocl3.1.3.** This whole section is retained only as
> the diagnostic + fallback for the mocl3.1.2 runtime.

This is the change that *used to* make a SYCL-MLIR program run to completion on
the `MT3K-DSP` device under mocl3.1.2. Fix 1 (`LLVM_ROOT`) clears the
`pocl_dat_gen` abort; on 3.1.2 the remaining blocker was at kernel launch and
is a hardware/runtime constraint of the 3.1.2 vendor libs. On 3.1.3 this
constraint is gone.

## Symptom

The program selects the DSP device and builds the SPIR-V program, but fails
at `clEnqueueNDRangeKernel` time with:

```
func _pocl_kernel__ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_EUlNS0_2idILi1EEEE__workgroup not found
Set function error
Selected device: MT3K-DSP Device #0
```

No kernel executes, so the expected output (`23456789…4243`) is not printed.

## Root cause

The launch path is:

1. SYCL-MLIR emits the kernel into SPIR-V with `OpEntryPoint` name = the
   Itanium-mangled name of the SYCL kernel-name type (computed in
   `clang/lib/Sema/SemaSYCL.cpp::constructKernelName` /
   `SetSYCLKernelNames`, consumed verbatim by cgeist at
   `polygeist/tools/cgeist/Lib/clang-mlir.cc:2782`).
2. POCL takes that SPIR-V, translates it to LLVM bitcode with the external
   `/thfs1/software/llvm/llvm-spirv/bin/llvm-spirv`, and generates a
   workgroup-driver function named `_pocl_kernel_<entry-name>_workgroup`
   (`pocl_llvm_generate_workgroup_function` in the POCL core
   `libOpenCL.so.2.12.0`).
3. The MT3X DSP device driver (`libpocl-devices-mt3x.so`) calls the DSP host
   runtime (`mt3x_toolkit/hthreads-old/lib/libhthread_host.so`) to load the
   compiled DSP image (`parallel.bc.dat` — a **text** symbol map of
   `<name>  <addr>`) and resolve the workgroup function by name via
   `mt_search_func` / `mt_set_func`.

The bug is in step 3. `libhthread_host.so` stores each loaded function symbol
in a fixed 72-byte record (`func_tables`): a **64-byte name field at offset 0**
(63 chars + NUL) and the 8-byte address at offset `0x40`. This is visible in
the disassembly of `mt_load_dat` (`mov x22, #0x40`; record stride
`add x20, x20, #0x48`) and `mt_search_func` (iterates records with
`add x19, x19, #0x48`, compares via `strcmp(entry, name)`, reads the address
from `[entry, #0x40]`).

Because the `.dat` is text it *contains* the full symbol name, but the
in-memory record **truncates it to 63 characters**. For the original lambda
kernel the names were:

| symbol | length |
|---|---|
| kernel name (`_ZTSZZ4mainENK…EEEE_`) | 63 |
| `_pocl_kernel_<name>` | 76 |
| `_pocl_kernel_<name>_workgroup` (looked up at launch) | **86** |

`mt_search_func("_pocl_kernel_…_workgroup")` therefore `strcmp`-fails against
the truncated 63-char stored name → `func … not found` / `Set function error`.

> Note: a naive `strncmp(_, _, 63)` patch on `mt_search_func` does **not**
> work, because the per-WI kernel symbol `_pocl_kernel_<name>` (76 chars) and
> the workgroup symbol `_pocl_kernel_<name>_workgroup` (86 chars) share their
> first 76 characters; both truncate to the same 63-char prefix and would be
> indistinguishable. The only safe fix is to keep the **whole** workgroup name
> ≤ 63 chars, which means a kernel name ≤ 40 chars.

The vendor libraries (`libOpenCL.so.2.12.0`, `libpocl-devices-mt3x.so`,
`libhthread_host.so`) are shipped only as `.so` and cannot be rebuilt here, so
the fix is applied on the SYCL-MLIR side by keeping kernel entry names short.

## Fix

SYCL lets you name a kernel explicitly via the first template argument of
`parallel_for` / `single_task`. The kernel-name type is what gets mangled into
the SPIR-V entry name and the `sycl::detail::KernelInfo::getName()` string
used for `clCreateKernel` (`sycl/include/sycl/kernel.hpp:47-63`,
`SemaSYCL.cpp:4152`). There is no hash/flag; the name is exactly what the
supplied type mangles to.

`simple-sycl-app.cpp` declares an empty global-scope name tag and passes it to
`parallel_for`:

```cpp
struct K {};                                  // global scope -> mangles to _ZTS1K

// ...
cgh.parallel_for<K>(NumOfWorkItems, [=](sycl::id<1> WIid) {
    c[WIid] = 2 * a[WIid] + b[WIid];
});
```

`K` is declared at **namespace/global scope** deliberately: a `struct K`
declared *inside* the submit lambda (e.g. `parallel_for<struct K>`) is a local
type and mangles with the full enclosing context
(`_ZTSZZ4mainENK…E1K`, 47 chars), which is still too long. At global scope it
mangles to `_ZTS1K` (6 chars), giving a workgroup symbol
`_pocl_kernel__ZTS1K_workgroup` (29 chars) — well under the 63-char limit and
distinct from the per-WI kernel symbol.

The kernel body is unchanged; only the kernel *name* changes. The program
output is identical.

## Verification

```bash
# build (Workspace/sycl-test/build.sh — mocl3.1.3 + LLVM_ROOT path)
source /thfs1/software/sycl/mocl3.1.3/env.sh
export LLVM_ROOT=/thfs1/software/llvm/llvm17-mt
export LD_LIBRARY_PATH=/thfs1/software/sycl/opencl-icd-loader/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/thfs1/home/ouyyc/sycl-mlir/build/install/lib:$LD_LIBRARY_PATH
/thfs1/home/ouyyc/sycl-mlir/build/install/bin/clang++ -fsycl \
    -fsycl-targets=spir64-unknown-unknown-syclmlir \
    simple-sycl-app.cpp -o a.out

# confirm the SPIR-V entry name is short
llvm-objcopy --dump-section __CLANG_OFFLOAD_BUNDLE__sycl-spir64=dev.bundle a.out
llvm-spirv -to-text dev.bundle -o dev.spv.txt
grep EntryPoint dev.spv.txt        # -> ... EntryPoint 6 72 "_ZTS1K" ...
```

Submit and run:

```bash
yhbatch run.sh
```

`slurm-<jobid>.out` (the `Error opening file` line is the non-fatal
scalarizer dump; see "Non-fatal warnings"):

```
Selected device: MT3K-DSP Device #0
2345678910111213141516171819202122232425262728293031323334353637383940414243
```

## How to diagnose a recurrence

- Clear the POCL cache before re-testing a kernel-name change
  (`rm -rf ~/.cache/pocl/kcache /tmp/pocl/kcache`) — POCL caches the generated
  `parallel.bc` / `parallel.bc.dat` / `parallel.bc.out` per (program-hash,
  local-size) and will otherwise reuse a stale image.
- The `func _pocl_kernel_<X>_workgroup not found` / `Set function error`
  message comes from `libhthread_host.so` (`mt_search_func`), printed by the
  mt3x driver during `pocl_mt3x_submit_kernel`.
- To check the workgroup name length: extract the SPIR-V (above), read the
  `OpEntryPoint` name, and compute `13 + len(name) + 10`
  (`_pocl_kernel_` + name + `_workgroup`). It must be ≤ 63.

## Limitation / guidance for SYCL-MLIR users on this DSP

> Applies to the **mocl3.1.2** runtime only. Under **mocl3.1.3** there is no
> kernel-name length limit (see the banner at the top of this section) and
> unnamed lambdas work as-is.

On mocl3.1.2, kernels whose mangled entry name exceeds ~40 characters cannot be
launched on `MT3K-DSP` with the 3.1.2 vendor mocl/POCL/MT3X runtime. In
practice this means **unnamed lambdas** (`-fsycl-unnamed-lambda`, the DPC++
default) almost always hit the limit on 3.1.2, because the lambda-closure
mangling embeds the full enclosing context (`main` → handler → `operator()`
→ lambda).

Workarounds (3.1.2 only), in order of preference:

1. **Name the kernel with a global-scope type** (`struct K {};` +
   `parallel_for<K>(…)`) — minimal, keeps the lambda body.
2. Use a named functor type at global scope.
3. Avoid deep nesting of the `parallel_for` call site (the deeper the call
   context, the longer the mangled lambda name).

A general compiler-side fix (hashing long kernel names into a fixed short
identifier in `SemaSYCL.cpp::constructKernelName`/`SetSYCLKernelNames`, with
matching `KernelInfo::getName()`) is possible but not implemented here: it
would change kernel names for **all** backends (regressing readability and
profiling on targets that handle long names fine) and is the wrong layer for a
single vendor runtime's 63-char limit. The proper long-term fix belongs in the
MT3X host runtime (enlarge the `func_tables` name field / use a string heap) —
which the 3.1.3 vendor libs appear to have done.

## Files changed (fix 2)

> Only relevant to the mocl3.1.2 fallback path. Under mocl3.1.3 the test
> program is left with its natural unnamed lambda (no `struct K` tag) and
> launches fine.

For the 3.1.2 fallback:

- `Workspace/sycl-test/simple-sycl-app.cpp` — added global `struct K {}` name
  tag; `parallel_for` → `parallel_for<K>`. (Kernel body and output unchanged.)
- `Workspace/sycl-test/build.sh` — use absolute paths for the source and
  output so the script works regardless of the directory it is invoked from
  (Slurm's default working directory is the submit directory); mocl3.1.3 path
  with `LLVM_ROOT`.
- `Workspace/sycl-test/run.sh` — run `a.out` by absolute path for the same
  reason; mocl3.1.3 path with `LLVM_ROOT` + `OCL_ICD_VENDORS`.

No SYCL-MLIR source, build, or runtime files were modified for this fix; the
compiler already emits a correct SPIR-V. Under 3.1.3 no change is needed at
all; under 3.1.2 the change is purely in the test program's kernel naming and
the test-driver scripts.

---

# Fallback — mocl3.1.2 (not recommended)

If mocl3.1.3 is unavailable, the old mocl3.1.2 runtime still works but needs
four post-build hacks that mocl3.1.3 makes unnecessary:

1. Copy the build's ICD loader into the install tree (the mocl3.1.2 loader is
   a non-standard monolithic `libOpenCL.so.2`):
   ```bash
   cp build/lib/libOpenCL.so.1.2 build/install/lib/libOpenCL.so.1.2
   ln -sfn libOpenCL.so.1.2 build/install/lib/libOpenCL.so.1
   ln -sfn libOpenCL.so.1   build/install/lib/libOpenCL.so
   ```
2. Patch `build/_deps/unified-runtime-src/source/adapters/opencl/program.cpp`
   — remove both `if (!Supported) return ...COMPILER_NOT_AVAILABLE;` guards in
   `urProgramCreateWithIL` (the mocl3.1.2 DSP device does not advertise
   `cl_khr_il_program`, tripping a false-negative guard that returns `-3`).
3. Rebuild the two adapters:
   ```bash
   ninja -C build lib/libpi_opencl.so lib/libur_adapter_opencl.so
   cp build/lib/libpi_opencl.so               build/install/lib/libpi_opencl.so
   cp build/lib/libur_adapter_opencl.so.0.9.0 build/install/lib/libur_adapter_opencl.so.0.9.0
   ```
   and run with `LD_PRELOAD=/thfs1/software/opencl/mocl3.1.2/lib/libOpenCL.so.2`.
4. Apply the short-kernel-name workaround from [Fix 2](#fix-2--dsp-kernel-launch-workgroup-symbol-name-limit)
   — name each kernel with a global-scope `struct K {};` +
   `parallel_for<K>(…)` so the workgroup symbol stays ≤ 63 chars. (Not needed
   on 3.1.3, whose `libhthread_host.so` resolves full long names.)

Durability: a `ninja install` regenerates the install tree without the copied
`libOpenCL.so.1` (re-apply step 1); a clean re-fetch of `unified-runtime-src`
clobbers the step-2 patch (make it durable via a local commit or a post-fetch
patch script). None of this applies to the mocl3.1.3 path, which is why 3.1.3
is preferred.
