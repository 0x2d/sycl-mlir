# The `-sycl-fusion` Compile Option

This document describes the `-sycl-fusion` command-line option added to the
cgeist / SYCL-MLIR toolchain: what it does, why it exists, and how to use it.

## TL;DR

`-sycl-fusion=<bool>` is a **cgeist** command-line switch that toggles the SYCL
kernel **FusionPass** on or off during the cgeist optimization pipeline. It
defaults to **on** (`-sycl-fusion=true`). It is provided as an escape hatch so
the (benchmark-specific) fusion pass can be turned off for sources whose
kernel-body shape does not match what the pass expects.

```bash
# Default behavior (fusion enabled) — no flag needed:
clang++ -fsycl -fsycl-targets=... app.cpp -o app

# Disable the fusion pass through the SYCL driver (-Xcgeist forwards to cgeist):
clang++ -fsycl -fsycl-targets=... -Xcgeist -sycl-fusion=false app.cpp -o app
```

## What it is

The option is a plain LLVM `cl::opt<bool>` registered by the cgeist driver:

```cpp
// polygeist/tools/cgeist/Options.h:161
static llvm::cl::opt<bool>
    EnableFusionPass("sycl-fusion", llvm::cl::init(true),
                     llvm::cl::desc("Run the SYCL kernel fusion pass"));
```

It is consumed in `driver.cc`, where the SYCL FusionPass (plus a trailing
canonicalize + CSE cleanup) is added to the optimization PM only when the flag
is set:

```cpp
// polygeist/tools/cgeist/driver.cc:510
if (EnableFusionPass) {
  PM.addPass(sycl::createFusionPass());
  PM.addPass(mlir::createCanonicalizerPass(CanonicalizerConfig, {}, {}));
  PM.addPass(mlir::createCSEPass());
}
```

> Note the **single-dash** spelling (`-sycl-fusion=...`). This is cgeist's own
> `cl::opt`, not an MLIR pass flag. The registered MLIR pass that does the
> actual work is named `sycl-fusion` too, and is invoked through the MLIR
> standard double-dash form, `--sycl-fusion`, when run standalone — see
> [Running the pass standalone](#running-the-pass-standalone).

## What the pass does

The pass (`mlir-sycl/lib/Dialect/SYCL/Transforms/FusionPass.cpp`, registered as
`Pass<"sycl-fusion", "::mlir::ModuleOp">` in `Passes.td`) performs **kernel
fusion** for the 126.ge (Gaussian Elimination) launch pattern. It is
**benchmark-specific**:

1. Walks the module's `gpu::GPUFuncOp`s and finds a `Fan1` and a `Fan2` kernel
   by demangled-name match (`llvm::demangle(...).find("Fan1")` etc.). If either
   is missing it prints `Fusion Pass: No candidate functions.` and returns.
2. Locates the kernel-body call inside each kernel. After the SYCL inliner, the
   expected shape is an `scf.if` dispatching between a `.specialized` callee and
   the generic kernel-body callee. The pass finds the non-`.specialized`
   `func.call` whose sibling region holds a `.specialized` call, hoists it out
   of the `scf.if`, and erases the `if`.
3. Pairs `Fan1`/`Fan2` arguments by *role* (`i32` scalar, accessor, `nd_item`,
   other) and clones `Fan2`'s body into `Fan1`, bridging any type divergence
   with `unrealized_conversion_cast`.
4. Redirects the cloned `Fan2` `get_global_id(0)` to `Fan1`'s
   `get_global_id(0)` result, and rewrites `Fan2`'s `get_global_id(1)` into an
   `scf.for` induction variable (the loop bounds are extracted from the
   surrounding `arith.cmpi slt/sgt`), effectively serializing Fan2's second
   dimension into a loop inside the fused kernel.
5. Fuses the two surviving `scf.if`s when their conditions are operationally
   equivalent, runs `RegisterPromotion`, and finally empties `Fan2` (leaving a
   stub `gpu.return` for API compatibility).

The pass is intentionally narrow and pattern-driven; the runtime kill-switch is
what makes it safe to ship enabled by default.

## Why it exists (the escape hatch)

Because the pass hardcodes the `Fan1`/`Fan2` name match, the argument-pairing
swap, and a specific post-inliner `scf.if` dispatch shape, it only does
something sensible for the 126.ge pattern. On other SYCL sources the pass will
either (a) find no `Fan1`/`Fan2` candidates and print
`Fusion Pass: No candidate functions.` — harmless — or (b) hit a kernel-body
shape it does not recognize and bail/misbehave. `-sycl-fusion=false` lets you
disable the pass entirely without rebuilding, so a non-matching source compiles
cleanly.

History: the flag was added in commit `4c96259a` ("Fix kernel-body lookup in SYCL
FusionPass") alongside a tightening of the kernel-body discovery heuristic, so
that the pass could be turned off at runtime if the kernel-body shape did not
match in some other source.

## How to use it

### From the SYCL driver (clang++ / swsyclmlir)

The clang driver does **not** know cgeist-specific flags directly. It forwards
arguments to cgeist through the `-Xcgeist <arg>` escape hatch
(`clang/include/clang/Driver/Options.td:974`; consumed at
`clang/lib/Driver/ToolChains/Clang.cpp:10838` via `AddAllArgValues`).

```bash
# Disable fusion when compiling a .cpp through the SYCL toolchain
clang++ -fsycl -fsycl-targets=spir64-unknown-unknown-syclmlir \
        -Xcgeist -sycl-fusion=false \
        app.cpp -o app
```

`-Xcgeist` is a *Separate* option, so the value follows as its own token. Both
of these work:

```bash
-Xcgeist -sycl-fusion=false      # separate form (recommended)
-Xcgeist=-sycl-fusion=false      # joined form
```

For the Sunway driver path (`swsyclmlir --target=athread -fsycl ...`), pass the
same `-Xcgeist -sycl-fusion=false` — it is just a wrapped clang invocation.

### From cgeist directly

If you invoke `cgeist` yourself (e.g. when iterating on the front-end), the flag
is a top-level option:

```bash
build/bin/cgeist -emit-llvm -S -fsycl -fsycl-targets=... \
                 -sycl-fusion=false app.cpp -o app.ll
```

With fusion left at its default (`true`) no extra flag is required.

### Running the pass standalone

The FusionPass is registered (via `sycl::registerSYCLPasses()`) in both
`sycl-mlir-opt` and `polygeist-opt`, so it can be applied to an existing MLIR
module with the standard MLIR double-dash form (note: **no `=`**, and there are
no pass options):

```bash
build/bin/polygeist-opt --sycl-fusion  in.mlir -o out.mlir
build/bin/sycl-mlir-opt --sycl-fusion  in.mlir -o out.mlir
```

This is the convenient way to inspect the fused kernel in isolation.

## Diagnostics

The pass is chatty on `llvm::dbgs()` under `DEBUG_TYPE = "sycl-fusion-pass"`:

```
Fusion Pass: Find candidate functions ... and ...
Fusion Pass: Performing kernel fusion on ... and ...
Fusion Pass: RegisterPromotion pattern converged
Fusion Pass: Two kernels
Fusion Pass: Completed
```

or, when it bails:

```
Fusion Pass: No candidate functions.
Fusion Pass: Cannot find kernel functions.
Fusion Pass: Cannot pair Fan1/Fan2 arguments by role.
```

To see these lines, run cgeist with debug output. As noted in `CLAUDE.md`, the
clang driver does **not** propagate MLIR debug flags, so invoke cgeist directly
(e.g. capture the `--args` line the driver emits, then run `cgeist` yourself
with `--debug-only=sycl-fusion-pass`).

## Build / iteration note

After editing `FusionPass.cpp` or cgeist sources and rebuilding, `ninja cgeist`
only refreshes `build/bin/cgeist`. Downstream consumers resolve cgeist out of
`build/install/bin/`, so copy the rebuilt binary across before re-running
benchmarks, otherwise you will silently test a stale binary:

```bash
ninja -C build cgeist && cp build/bin/cgeist build/install/bin/cgeist
```

(The same caveat applies to `clang++`, `opt`, `llvm-spirv`, `llvm-dis`,
`FileCheck`, `count`, `not`, `llvm-lit` — see `CLAUDE.md` → "Build vs install".)

## Not to be confused with

- The **`sycl-fusion/`** directory is a *runtime* kernel-fusion component
  (built via `buildbot/configure.py` as the `sycl-fusion` external project).
  It is unrelated to this compile-time `mlir::sycl::createFusionPass` pass and
  to the `-sycl-fusion=` cgeist flag, despite the shared name.
- The flag is cgeist's own. There is **no** `-fsycl-fusion=` clang driver flag;
  reach cgeist through `-Xcgeist`.
