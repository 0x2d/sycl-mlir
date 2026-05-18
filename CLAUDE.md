# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is the Intel LLVM-based DPC++ compiler project (intel/llvm fork, sycl branch), implementing compiler and runtime support for the SYCL language. The repository contains the full LLVM toolchain with SYCL, MLIR, and related projects.

## Build System

The project uses CMake via buildbot scripts:

```bash
# Configure
python buildbot/configure.py -o build

# Build (default target: deploy-sycl-toolchain)
python buildbot/compile.py -o build -j<threads>

# Build specific targets
python buildbot/compile.py -o build -t clang -j8

# Or drive ninja directly once configured
ninja -C build cgeist polygeist-opt mlir-opt
```

Available configure flags:
- `--werror` - Treat warnings as errors
- `--cuda` - Enable CUDA backend
- `--hip` - Enable HIP backend
- `--shared-libs` - Build shared libraries
- `--cgeist-allow-undefined-sycl-types` - Let cgeist accept undefined SYCL types by default (useful when iterating on Polygeist front-end work)
- `--enable-all-llvm-targets` - Build NVPTX and AMDGPU targets in addition to the host
- `-t {Debug|Release}` - Build type
- `-o <path>` - Build directory

The build uses Ninja by default. Existing build directory is at `build/`. Built tools land in `build/bin/` — the ones used most for MLIR-SYCL work are `cgeist`, `polygeist-opt`, `mlir-opt`, `mlir-translate`, `clang`, and `llvm-spirv`.

**Build vs install:** downstream consumers (e.g. `sycl-bench/build-mlir`) point `CMAKE_CXX_COMPILER` at `build/install/bin/clang++`, which resolves cgeist relative to itself (`build/install/bin/cgeist`). `ninja cgeist` rebuilds `build/bin/cgeist` only — there is no `install-cgeist` target. After iterating on cgeist, copy the freshly-built binary into `install/bin/` (`cp build/bin/cgeist build/install/bin/cgeist`) before re-running downstream benchmarks, otherwise you'll silently test a stale binary. Same caveat applies to `clang++`, `opt`, `llvm-spirv`, `llvm-dis`, `FileCheck`, `count`, `not`, `llvm-lit`.

## Key Directories

| Directory | Purpose |
|-----------|---------|
| `llvm/` | LLVM core (compiler infrastructure) |
| `clang/` | C/C++/SYCL frontend |
| `sycl/` | SYCL runtime, headers, and tests |
| `mlir/` | MLIR core infrastructure |
| `mlir-sycl/` | SYCL-specific MLIR dialect and passes |
| `polygeist/` | C/C++ to MLIR conversion (Cgeist) |
| `sycl-fusion/` | Runtime kernel fusion |
| `llvm-spirv/` | SPIR-V translation |
| `buildbot/` | Build configuration scripts |

## Project Architecture

The SYCL compilation flow:
1. `clang` parses SYCL/C++ code → LLVM IR
2. `mlir-sycl` dialect provides SYCL-specific MLIR operations
3. `polygeist/cgeist` lowers C to MLIR
4. `llvm-spirv` translates to SPIR-V
5. Runtime handles execution

Key MLIR-SYCL components:
- `mlir-sycl/lib/Dialect/SYCL/` - SYCL dialect definition
- `mlir-sycl/lib/Conversion/` - Lowering passes (`SYCLToLLVM`, `SYCLToSPIRV`, `SYCLToGPU`, `SYCLToMath`)
- `mlir-sycl/lib/Transforms/` and `mlir-sycl/lib/Analysis/` - dialect-level transforms and analyses
- `mlir-sycl/test/{Dialect,Conversion,Transforms,Analysis}/` - lit tests, organized to mirror `lib/`
- `polygeist/tools/cgeist/` - C/C++ → MLIR driver (`cgeist`)
- `polygeist/tools/polygeist-opt/` - opt-style driver for Polygeist + SYCL dialects

### cgeist function-emission machinery

When touching call lowering, function-type construction, or return-value handling in cgeist, the relevant files form a tight cluster:

- `polygeist/tools/cgeist/Lib/CodeGenTypes.{h,cc}` — `getFunctionType` builds the MLIR `FunctionType` from clang's `CGFunctionInfo`, applying ABI lowering. The compile-time `AllowSRet`/`AllowInAllocaRet`/`AllowStructFlattening` flags gate which `ABIArgInfo` kinds are honored. `ClangToLLVMArgMapping` (defined in the header) computes the IR-arg index of the sret slot, the `this` slot, and per-clang-formal arg mappings; reuse this rather than reimplementing.
- `Lib/CGCall.cc::callHelper` — caller side. Inserts the sret alloca and arg at the right position, and decides whether to take the SYCL fast-path (`emitSYCLOps` → specific method ops or generic `sycl.call`) or fall through to `func::CallOp`. Note the asymmetry: specific SYCL method ops semantically model the return value as the op result and don't want a sret arg, while a generic `sycl.call` references the underlying `func.func` by mangled name and *must* match its signature.
- `Lib/clang-mlir.cc::MLIRScanner::init` — callee side prologue. Binds clang `ParmVarDecl`s to MLIR `Function.getArgument(...)` slots; when sret is in use the IR has one more arg than the clang formal list, so this loop must use a separate IR-index that skips the sret slot. The `ReturnVal` alloca is created only when `Function.getResultTypes()` is non-empty, so a void-returning sret-aware function gets the right prologue automatically.
- `Lib/CGStmt.cc::VisitReturnStmt` — callee side return. The `IsArrayReturn` and "normal value" branches need a sibling `IsSRet` branch that stores into the sret pointer via `ValueCategory::store` (not open-coded `memref::StoreOp` — aggregates need the store helper).
- `Lib/CGExpr.cc::emitSYCLOps` / `EmitSYCLConstructor` / `createSYCLMethodOp` — the SYCL fast-path. `createSYCLMethodOp` succeeds only when the dialect knows the method name on the receiver type (`SYCLDialect::findMethod`); otherwise the generic `SYCLCallOp` fallback fires.
- `Test/Verification/sycl/` — lit tests. FileCheck patterns here are commonly pinned to function signatures, so any change to ABI handling (sret, byval, flattening) will produce shape-only diffs that need updating.

The `getOrCreateLLVMFunction` path (`clang-mlir.cc:1495`) is a *separate* declaration cache that does NOT apply ABI lowering — it uses `CGM.getTypes().ConvertType(QT)` directly. Symbols emitted through both paths must agree on signature, otherwise phase-3 finalize will fail with `'llvm.call' op incorrect number of operands` after the func→LLVM conversion runs.

## Pipeline phases (driver.cc)

`finalize()` in `polygeist/tools/cgeist/driver.cc` runs three pass managers:

1. **Phase 1 (PM)** — early canonicalization, mem2reg, loop restructuring, affine raising. Operates on `func.func`.
2. **Phase 2 (PM2)** — SYCL host raising, dialect-level transforms.
3. **Phase 3 (PM3)** — `arith-expand`, `convert-polygeist-to-llvm` (this is where `func.func` → `llvm.func` and `func.call` → `llvm.call` happen), `reconcile-unrealized-casts`, `legalize-for-spirv`. **This is where ABI mismatches surface as verifier failures.** The error `Finalize failed (phase 3)` followed by `'llvm.call' op incorrect number of operands` is almost always a signature disagreement between a func.func and its call sites.

Use `--mlir-print-ir-before-all --mlir-disable-threading` on cgeist directly to see the IR between passes — useful for diagnosing where IR shape disagrees with what a later pass expects. The clang driver does not propagate these flags; invoke cgeist itself with the captured `--args` line.

## Testing

### In-tree LIT tests
```bash
# SYCL runtime / headers
ninja -C build check-sycl

# MLIR-SYCL dialect, conversions, transforms
ninja -C build check-mlir-sycl

# Polygeist (cgeist + polygeist-opt regression suites)
ninja -C build check-cgeist
ninja -C build check-polygeist

# Other components
ninja -C build check-llvm-spirv
ninja -C build check-clang
```

Run a single lit test by invoking `llvm-lit` directly against the file:
```bash
build/bin/llvm-lit -v mlir-sycl/test/Conversion/SYCLToLLVM/some_test.mlir
```

### E2E tests
```bash
cd sycl/test-e2e
python format.py --help  # See how to run E2E tests
```

### Code Formatting
```bash
# Check formatting (against origin/sycl)
./clang/tools/clang-format/git-clang-format $(git merge-base origin/sycl HEAD)

# Format specific files
git-clang-format -f <file>
```

## Commit Message Style

- DPC++ commits use `[SYCL]` tag: `git commit -m "[SYCL] Add feature X"`
- Additional tags: `[PI]`, `[CUDA]`, `[Doc]`, `[NFC]`
- Follow LLVM commit message conventions

## Important Notes

- `#include <iostream>` is forbidden; use `sycl/detail/iostream_proxy.hpp`
- Use `sycl::` namespace (not `cl::sycl::`)
- Tests must accompany code changes
- Maintain ABI/API stability for existing APIs
