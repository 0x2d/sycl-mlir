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
