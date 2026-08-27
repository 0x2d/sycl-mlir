# Commit History

Chronological log of branch commits starting at `9ece24a9948c` ("[SYCL-MLIR]: disable level_zero configuration"). Each entry covers what changed, where, and why. Most recent commit is at the bottom.

---

## 9ece24a9948c — [SYCL-MLIR]: disable "level_zero" configuration

- **Author:** Yucheng Ouyang
- **Date:** 2025-09-09
- **Files:** `buildbot/configure.py` (+9 / -9)

Comments out the two places in the buildbot configure script that opt the `level_zero` plugin into the build:

1. The non-darwin branch that appends `"level_zero"` to `sycl_enabled_plugins`.
2. The block that wires `-DLEVEL_ZERO_INCLUDE_DIR` / `-DLEVEL_ZERO_LIBRARY` into the CMake invocation when `--l0-headers` / `--l0-loader` are supplied (along with the matching error if only one is provided).

Effectively skips Level Zero support locally without touching argparse declarations — the flags are still accepted but become no-ops. Useful when the developer's environment lacks the Level Zero loader and they don't want the plugin pulled in.

---

## ea607f5f9a1a — add toy pass

- **Author:** Yucheng Ouyang
- **Date:** 2025-10-15
- **Files:**
  - `mlir-sycl/include/mlir/Dialect/SYCL/Transforms/Passes.h` (+1)
  - `mlir-sycl/include/mlir/Dialect/SYCL/Transforms/Passes.td` (+10)
  - `mlir-sycl/lib/Dialect/SYCL/Transforms/CMakeLists.txt` (+1)
  - `mlir-sycl/lib/Dialect/SYCL/Transforms/ToyPass.cpp` (new, +66)
  - `polygeist/tools/cgeist/driver.cc` (+7 / -1)

Scaffolding for an experimental SYCL pass:

- New `ToyPass` declared in `Passes.td` (`-sycl-toy`, runs on `ModuleOp`, depends on `arith`) with a matching factory `mlir::sycl::createToyPass()`.
- `ToyPass.cpp` walks the module for `gpu.func` and `func.func` ops and prints a debug message when their mangled names match two specific lambda mangled names from a particular SYCL test program. No IR is mutated.
- Registers the pass in cgeist's optimization pipeline — added to `PM` after `RaiseSCFToAffinePass`/`ReplaceAffineCFGPass` and before the SYCL inliner — and adds two commented-out `Module->dump()` debug points around the optimize phase in `createAndExecutePassPipeline`.

This is the seed for what later becomes the kernel-fusion pass; on its own it is a print-only no-op.

---

## a4277e28faa2 — fix RaiseToAffinePass

- **Author:** Yucheng Ouyang
- **Date:** 2025-11-03
- **Files:**
  - `polygeist/lib/Dialect/Polygeist/Transforms/AffineCFG.cpp` (+1 / -1)
  - `polygeist/tools/cgeist/driver.cc` (+6 / -2)

Two unrelated hunks:

1. **AffineCFG.cpp** — In `handle(...)`, the `CmpIOp → IntegerSet` lowering used by `MoveIfToAffine`. The `sgt` (strictly greater-than) branch built the constraint `expr = (lhs - rhs) + 1`, which encodes `lhs >= rhs - 1` — a *weaker* condition than `lhs > rhs` — and would let `i == rhs` enter the then-branch of an affine-if that should require `i > rhs`. Changed to `expr = (lhs - rhs) - 1`, mirroring the symmetric `slt` branch a few lines below. Soundness fix.
2. **driver.cc** — Comment-only churn: relabels two commented `Module->dump()` markers around the canonicalize/optimize/finalize boundaries ("Before canonicalize." / "Before optimize." / "Before finalize."). No behavioral effect.

The driver hunk is unrelated to the title and should arguably have been split out, but it's harmless.

---

## babfcca4fb95 — Add naive 126.ge optimization

- **Author:** Yucheng Ouyang
- **Date:** 2026-01-15
- **Files:**
  - `mlir-sycl/lib/Dialect/SYCL/Transforms/ToyPass.cpp` (+138 / -17)
  - `polygeist/tools/cgeist/driver.cc` (+2 / -5)

Turns the previously-empty `ToyPass` into a hand-coded kernel fusion pass tailored to the 126.ge benchmark (Gaussian elimination, with `Fan1` as the row-elimination factor kernel and `Fan2` as the row-update kernel):

- Locates `Fan1` / `Fan2` `gpu.func`s by exact mangled-name match.
- For each fan, walks its body, finds the wrapper `scf.if` whose else-region calls the device kernel, hoists the `func.call` above the `if`, erases the `if`, and trims away dead operations between argument producers and the call.
- Looks up the underlying `func.func` callees, builds an `IRMapping` aligning Fan2's accessor/value/range arguments with the corresponding ones on Fan1 (with arg-2/arg-3 swapped, matching how the two kernels see the matrix), and clones Fan2's body into Fan1 just before its terminator.
- Empties Fan2's `gpu.func` and replaces its body with a single `gpu.return` to keep the module valid.
- Replaces the `nd_item.get_global_id(1)` call inside the merged Fan1 body with the constant `i32 0`, then wraps the operations after that constant in an `scf.for` whose bounds come from the `slt` comparison that previously gated the y-dimension thread; replaces uses of the old constant with the loop induction variable.
- Driver pipeline: moves the toy-pass registration so that it runs *after* `sycl::createInlinePass` (instead of before), so kernels are inlined into Fan1/Fan2 before fusion runs. Drops the surrounding commented-out `Module->dump()` markers.

Naive in the literal sense: arg-mapping indices, mangled names, and the `get_global_id(1) → 0` rewrite are all hard-coded for this one benchmark.

---

## e4064e58b4b7 — Optimize ```FusionPass```

- **Author:** Yucheng Ouyang
- **Date:** 2026-03-03
- **Files:**
  - `mlir-sycl/include/mlir/Dialect/SYCL/Analysis/KernelAnalysis.h` (new, +30)
  - `mlir-sycl/lib/Dialect/SYCL/Analysis/KernelAnalysis.cpp` (new, +27)
  - `mlir-sycl/lib/Dialect/SYCL/Analysis/CMakeLists.txt` (+1)
  - `mlir-sycl/include/mlir/Dialect/SYCL/Transforms/Passes.h` (+3 / -1)
  - `mlir-sycl/include/mlir/Dialect/SYCL/Transforms/Passes.td` (+5 / -5)
  - `mlir-sycl/lib/Dialect/SYCL/Transforms/CMakeLists.txt` (+2 / -1)
  - `mlir-sycl/lib/Dialect/SYCL/Transforms/FusionPass.cpp` (new, +275)
  - `mlir-sycl/lib/Dialect/SYCL/Transforms/RegisterPromotionPattern.cpp` (new, +72)
  - `mlir-sycl/lib/Dialect/SYCL/Transforms/ToyPass.cpp` (deleted, -197)
  - `mlir-sycl/include/mlir/Dialect/SYCL/Utils/Utils.h` (+5)
  - `mlir-sycl/lib/Dialect/SYCL/Utils/Utils.cpp` (+76)
  - `polygeist/tools/cgeist/driver.cc` (+6 / -7)

Renames and generalizes the toy fusion pass into a real `FusionPass`, plus adds a register-promotion pattern and a kernel-counting analysis.

- **Pass rename** — `ToyPass` → `FusionPass` (TableGen def, header decl, CMake source list, factory). `ToyPass.cpp` is deleted; `FusionPass.cpp` carries forward the structure.
- **Kernel discovery** — drops hard-coded mangled names. Walks `gpu.func`s, demangles each name, and matches against `"Fan1"` / `"Fan2"` substrings.
- **Callee discovery** — drops the "find the `scf.if` then take its else-region call" approach; instead walks each fan and picks the first `func.call` whose callee name does **not** contain `".specialized"`, then hoists it out of any enclosing `scf.if` and erases the `if`.
- **Post-fusion cleanups added:**
  - **`if`-fusion** — if Fan1 originally had one `scf.if` and after merging there are two `scf.if`s in the merged callee with structurally-equivalent conditions (`OperationEquivalence::isEquivalentTo` using a new `checkEquivalent` helper), splice operations between them up before the first `if`, then move the second `then`-block into the first `then`-block and erase the second `if`.
  - **Register promotion** — new `RegisterPromotion` rewrite pattern in `RegisterPromotionPattern.cpp`, exposed via `populateRegisterPromotion`. For an `sycl.accessor.subscript` that feeds an `affine.load`, it scans backwards for an earlier subscript that feeds an `affine.store` to the same accessor or an equivalent offset, and short-circuits the load by replacing its result with the value that was stored. Driven by `applyPatternsAndFoldGreedily` on the merged callee.
- **`KernelAnalysis`** — new analysis that just counts `gpu.func`s in the module; queried at the end of `FusionPass::runOnOperation` purely to log "Two kernels" vs "More than two kernels".
- **`Utils`** — adds `checkEquivalent(Value, Value)` (recursive structural equality on defining ops, used by the if-fusion match) and `getOffsetFromSubscriptOp(SYCLAccessorSubscriptOp)` (walks `memref.cast` → `affine.store` → `affine.load` → `memref.cast` → `memref.memory_space_cast` → `sycl.constructor` to recover the `id` argument feeding a subscript; used by register promotion).
- **Pipeline** — driver registers `createFusionPass`, then `createCanonicalizerPass` and `createCSEPass` immediately after it. Removes the commented dump markers around the optimize phase.

Bigger picture: same fusion shape as the previous commit, but no longer keyed to one specific test; `Fan1`/`Fan2` now means "any function with that substring in its demangled name", and the post-fusion shape is canonicalized via if-fusion + register-promotion + canonicalize/CSE.

Two minor sharp edges introduced:
- `OperationEquivalence::isEquivalentTo` is invoked with `checkEquivalent` as the *value-mapping callback*, but `checkEquivalent` returns `LogicalResult` ignoring the callback contract (it should record a one-to-one mapping); operands are also traversed twice (once by `isEquivalentTo`, once by `checkEquivalent` directly), which is redundant rather than wrong.
- Both `KernelAnalysis.cpp` and `RegisterPromotionPattern.cpp` end without a trailing newline.

---

## 95de32e009a5 — Claude init

- **Author:** Yucheng Ouyang
- **Date:** 2026-05-09
- **Files:** `CLAUDE.md` (new, +102)

Initial drop of the project-level `CLAUDE.md`. Documents:

- Build invocation via `buildbot/configure.py` and `buildbot/compile.py` plus the available flags (`--werror`, `--cuda`, `--hip`, `--shared-libs`, `-t Debug/Release`, `-o`).
- Directory map (`llvm/`, `clang/`, `sycl/`, `mlir/`, `mlir-sycl/`, `polygeist/`, `sycl-fusion/`, `llvm-spirv/`, `buildbot/`).
- Compilation flow at a high level (clang → mlir-sycl/polygeist → llvm-spirv → runtime) and the SYCL-specific MLIR component layout.
- Test entry points (`check-sycl`, `check-llvm-spirv`, `check-clang`, the E2E directory, `clang-format` invocation).
- Commit-message style and a few coding rules (`<iostream>` forbidden, use `sycl::` namespace, tests required, ABI/API stability).

No code changes.

---

## 077a3565985283b07a6aed19399961d88780fb0b — Skip flag "-target-cpu"

- **Author:** Yucheng Ouyang
- **Date:** 2026-05-09
- **Files:** `polygeist/tools/cgeist/driver.cc` (+6)

Extends cgeist's command-line splitter (`Options::splitCommandLineOptions`) to recognize four cc1-style options that take a *separate* value argument: `-aux-target-feature`, `-target-feature`, `-aux-target-cpu`, `-target-cpu`. When one of these is seen, the driver skips both the flag and the next argv slot. Without this, e.g. `-target-feature +lzcnt` would leave `+lzcnt` in the residual arg list and get mistaken for an input file by later parsing logic.

Narrow, defensive fix to the argv-routing layer; no pipeline change.

---

## fffe3ddf038ddf8b7a8e28ca995c57dbd2b4e2b3 — Update CLAUDE.md

- **Author:** Yucheng Ouyang
- **Date:** 2026-05-18
- **Files:** `CLAUDE.md` (+26 / -8)

Editorial pass on `CLAUDE.md`, no code touched:

- Build section: corrects default target to `deploy-sycl-toolchain`, adds direct-`ninja` invocation example, documents `--cgeist-allow-undefined-sycl-types` and `--enable-all-llvm-targets`, notes that built tools land in `build/bin/` and lists the ones used most for MLIR-SYCL work.
- Architecture section: expands the MLIR-SYCL bullet list (adds `Transforms/`, `Analysis/`, the `test/{Dialect,Conversion,Transforms,Analysis}/` mirror, and `polygeist-opt`), formats conversion-pass names as code spans.
- Testing section: replaces the three generic targets with a categorized list (`check-sycl`, `check-mlir-sycl`, `check-cgeist`, `check-polygeist`, `check-llvm-spirv`, `check-clang`) and adds a snippet for running a single lit test via `build/bin/llvm-lit`.

---

## Support SRet aggregate returns in cgeist

- **Author:** Yucheng Ouyang
- **Date:** 2026-05-18
- **Files:**
  - `polygeist/tools/cgeist/Lib/CodeGenTypes.cc` (+15 / -86)
  - `polygeist/tools/cgeist/Lib/CodeGenTypes.h` (+68)
  - `polygeist/tools/cgeist/Lib/CGCall.cc` (+67 / -4)
  - `polygeist/tools/cgeist/Lib/CGStmt.cc` (+7)
  - `polygeist/tools/cgeist/Lib/CGExpr.cc` (+14)
  - `polygeist/tools/cgeist/Lib/clang-mlir.cc` (+23 / -3)
  - `polygeist/tools/cgeist/Lib/clang-mlir.h` (+2)
  - `polygeist/lib/Dialect/Polygeist/Transforms/Mem2Reg.cpp` (+15 / -1)
  - `polygeist/tools/cgeist/Test/Verification/sycl/sret-aggregate-return.cpp` (new, +31)
  - `CLAUDE.md` (+25)
  - `COMMIT_HISTORY.md` (new, +161)

Until now, `AllowSRet` in `CodeGenTypes.cc` was hard-coded `false`: any function whose Itanium ABI lowering classified the return as `ABIArgInfo::Indirect` (struct returned via a hidden pointer parameter) hit a `CGEIST_WARNING` and silently fell back to a direct struct return. That broke SYCL hierarchical-parallelism kernels — `h_item::get_global`, `h_item::get_local`, `Builder::createItem`, `InitializedVal::get` all return non-trivial structs that the ABI wants to lower as sret. This change wires sret end-to-end.

- **`CodeGenTypes.cc/h`** — Flips `AllowSRet` to `true`. Promotes `ClangToLLVMArgMapping` out of the file-local anonymous namespace into the header (declaration in `CodeGenTypes.h`, definition split into `CodeGenTypes.cc`) so the caller side in `CGCall.cc` and the callee prologue in `clang-mlir.cc` can both use it to compute `getSRetArgNo()`. In `getFunctionType`, the `Indirect` return now produces `NoneType` (void return) and the sret arg type is computed via `CGM.getDataLayout().getAllocaAddrSpace()` with `getPointerOrMemRefType(..., /*IsAlloc=*/false)`, replacing the previous `llvm_unreachable` placeholder that mis-used `getTargetAddressSpace(Ret.getAddressSpace())`. Adds a sanity assert that array-return and sret are never both set.
- **`CGCall.cc::callHelper`** — Caller side. Detects sret early via `RetAI.getKind() == Indirect` and stashes `SRetArgNo`. The sret alloca/arg is *not* prepended at the natural point in argument building; instead it is deferred until after `emitSYCLOps`. Reason: SYCL method ops (e.g. `sycl.range.get`) model the return value as the op result and don't want a sret pointer prepended, but the generic `sycl.call` fallback references the underlying `func.func` by name and would disagree with its sret-aware signature — so when a generic `SYCLCallOp` would be emitted we erase it and fall through to `func::CallOp`. Only on the `func::CallOp` path do we materialize the sret alloca (as `memref::AllocaOp` or `LLVM::AllocaOp` depending on `getPointerOrMemRefType`'s shape), insert it at `Args.begin() + SRetArgNo`, run `castCallerArgs`, and return `ValueCategory(Alloc, /*isReference=*/true, ElemTy)`. `castCallerArgs` is now skipped on the pre-sret Args list to avoid trying to cast against a signature whose arg count doesn't match yet.
- **`clang-mlir.cc::MLIRScanner::init`** — Callee prologue. The clang `ParmVarDecl` loop now tracks two indices: the existing `I` (clang-formal index, used to look up `FIArgs[I]`) and a new `IRIdx` (MLIR-arg index, used for `Function.getArgument(IRIdx)`). A `MaybeSkipSRet` lambda advances `IRIdx` past the sret slot whenever it lands on it; called once at entry, after the implicit `this`, and after each formal. The sret value itself is captured into the new `MLIRScanner::SRetArg` field (with `IsSRet` flag) declared in `clang-mlir.h`. The existing `ReturnVal` alloca is unchanged — it's only created when `Function.getResultTypes()` is non-empty, which is now naturally false for sret functions, so the prologue stays correct without further edits.
- **`CGStmt.cc::VisitReturnStmt`** — Adds an `IsSRet` branch alongside `IsArrayReturn` and the normal-value branch. Emits the return-value expression, then writes it through `SRetArg` via `ValueCategory(SRetArg, /*isReference=*/true, ElemTy).store(Builder, Rv, /*isArray=*/false)`. Uses the `ValueCategory::store` helper rather than open-coding `memref::StoreOp`, which is necessary for aggregates.
- **`CGExpr.cc::VisitLambdaExpr`** — Independent fix that surfaced while testing the sret path on a SYCL kernel. When a lambda captures a reference, the source pointer/memref might live in a different address space than the closure field (e.g. host-side stack `int i` in addrspace 0 vs a device-side closure field in addrspace 4). Without an explicit AS cast, `ValueCategory::store` cannot match the types and silently emits no store — the field stays uninitialized. Now, when the field is a pointer or memref, we route the source value through `castToMemSpaceOfType(Val, FieldTy)` before the field store.
- **`Mem2Reg.cpp::isPromotable`** — Independent bugfix to the alloca-promotability check, exposed by sret enabling. When traversing alloca uses, the `LLVM::StoreOp` branch already had a "captured-as-store-value" guard, but it compared against `AI` (the original alloca) rather than `val` (the value reached by following the cast chain), so a cast-then-store-as-value pattern slipped through. It also didn't `return false` after logging, so the alloca was still considered promotable. Fixed both: now compares against `val` and returns `false`. Also adds the same guard to the previously-missing `memref::StoreOp` and `affine::AffineStoreOp` branches — those simply `continue`d before, which would let an alloca whose address had escaped through one of those stores be wrongly promoted, losing writes through the captured pointer. Sret-aware lowering exercises this path because aggregate return slots get stored as values into outer storage in some cases.
- **`Test/Verification/sycl/sret-aggregate-return.cpp`** — New lit test. A minimal SYCL kernel using `parallel_for_work_group` + `parallel_for_work_item` + `h_item`, which transitively pulls in several Indirect-returning header methods. Uses `--implicit-check-not="function should return its value indirectly"` plus a positive `CHECK-NOT` to assert that the previous warning no longer fires. Drives cgeist via `clang++ -fsycl -fsycl-targets=spir64-unknown-unknown-syclmlir -fsycl-device-only -emit-llvm`.
- **Documentation** — `CLAUDE.md` gains two sections relevant to this work. (1) "Build vs install" explains that downstream consumers like `sycl-bench/build-mlir` resolve cgeist out of `build/install/bin/` while `ninja cgeist` only updates `build/bin/`, so a freshly-built cgeist must be copied across after iteration to avoid silently testing a stale binary (same caveat for the rest of the toolchain). (2) "cgeist function-emission machinery" documents the file cluster touched here — `CodeGenTypes`, `CGCall`, `clang-mlir`, `CGStmt`, `CGExpr`, `Test/Verification/sycl/` — including the asymmetry between specific SYCL method ops and generic `sycl.call` (the reason the sret arg insertion has to be deferred), the warning that `getOrCreateLLVMFunction` is a *separate* declaration cache that doesn't apply ABI lowering (signature divergence between paths surfaces as `'llvm.call' op incorrect number of operands` in phase-3 finalize), and a "Pipeline phases" section spelling out where ABI mismatches appear during finalize.
- **`COMMIT_HISTORY.md`** — New file, this very document. Chronological log of branch commits with what/where/why for each.

End-to-end effect: a void-returning sret-aware `func.func` is emitted with one extra leading or post-`this` argument (its position chosen by `ClangToLLVMArgMapping`), the callee writes its return value through that pointer, and call sites materialize an alloca, append it at the right index, and project the alloca back as a `ValueCategory` reference. Existing direct-return paths are untouched; the warning emitted by the old fallback no longer fires.

---

## 4c96259ae4b5a2122bba18089b0323eef79e6604 — Fix kernel-body lookup in SYCL FusionPass

- **Author:** Yucheng Ouyang
- **Date:** 2026-05-19
- **Files:**
  - `mlir-sycl/lib/Dialect/SYCL/Transforms/FusionPass.cpp` (+52 / -29)
  - `polygeist/tools/cgeist/Options.h` (+4)
  - `polygeist/tools/cgeist/driver.cc` (+5 / -3)

Fallout from the sret-ABI commit (`1022c039`, "Support SRet aggregate returns in cgeist"). After sret was enabled, an unrelated helper call (e.g. `sycl::detail::Builder::getElement<N>`) survives the SYCL inliner and shows up in the kernel body *before* the `scf.if` that dispatches between the `.specialized` and generic kernel-body callees. The previous heuristic — "first `func.call` in the kernel whose callee name does not contain `.specialized`" — picked that 2-argument helper, and the downstream hard-coded `getArgument(0..5)` mapping then tripped `MutableArrayRef::operator[]`'s "Invalid index!" assertion.

- **`FusionPass.cpp`** — Replaces the per-kernel `walk` with a `findKernelBodyCall(GPUFuncOp)` helper that requires the call to be (a) non-`.specialized`, (b) directly nested inside an `scf.if`, and (c) whose sibling region of that `scf.if` contains a `.specialized` call. That uniquely identifies the inliner's specialized-vs-generic dispatch shape and is independent of which other helpers remain in the kernel body. Hoisting the call out of its enclosing `scf.if` and erasing the `if` is now factored into a small loop over the two found calls, run *after* both have been located (previously each fan's hoist/erase was done inline during the walk). The "Cannot find kernel functions" / "Performing kernel fusion on …" log lines are preserved.
- **`Options.h`** — New `-sycl-fusion=<bool>` cl::opt (`EnableFusionPass`, default `true`). Escape hatch so the pass can be disabled at runtime if the kernel-body shape doesn't match in some other source.
- **`driver.cc`** — Wraps the `createFusionPass` + post-fusion canonicalize + CSE additions to `PM` in `if (EnableFusionPass)`. No reordering of any other passes.

Reproducer is `gaussianElim_kernels.cpp` from the 126.ge benchmark — previously crashed in `FusionPass`, now the pass logs `Performing kernel fusion on …_clESD_` on the actual kernel-body callees and the source compiles cleanly. Conceptually narrow: the fusion logic itself is untouched; only the kernel-body discovery is tightened, plus a runtime kill-switch.

---

## fa1ad396a9acc85bd3514ad9d5a331bd1e9ad127 — [SYCL-MLIR] Add SYCL-to-ROCDL lowering for AMDGCN device targets

- **Author:** Yucheng Ouyang
- **Date:** 2026-05-26
- **Files:**
  - `mlir-sycl/include/mlir/Conversion/SYCLPasses.h` (+1)
  - `mlir-sycl/include/mlir/Conversion/SYCLPasses.td` (+30 / -1)
  - `mlir-sycl/include/mlir/Conversion/SYCLToROCDL/SYCLToROCDL.h` (new, +33)
  - `mlir-sycl/include/mlir/Dialect/SYCL/IR/SYCLAttributes.td` (+2 / -1)
  - `mlir-sycl/lib/Conversion/CMakeLists.txt` (+1)
  - `mlir-sycl/lib/Conversion/SYCLToLLVM/CMakeLists.txt` (+2 / -1)
  - `mlir-sycl/lib/Conversion/SYCLToLLVM/DPCPP.cpp` (+7)
  - `mlir-sycl/lib/Conversion/SYCLToLLVM/SYCLToLLVM.cpp` (+1)
  - `mlir-sycl/lib/Conversion/SYCLToROCDL/CMakeLists.txt` (new, +20)
  - `mlir-sycl/lib/Conversion/SYCLToROCDL/SYCLToROCDL.cpp` (new, +422)
  - `mlir-sycl/test/Conversion/SYCLToROCDL/global-id.mlir` (new, +53)
  - `mlir-sycl/test/Conversion/SYCLToROCDL/global-offset.mlir` (new, +20)
  - `mlir-sycl/test/Conversion/SYCLToROCDL/grid-ops.mlir` (new, +59)
  - `mlir-sycl/test/Conversion/SYCLToROCDL/sub-group.mlir` (new, +46)
  - `mlir-sycl/tools/sycl-mlir-opt/CMakeLists.txt` (+1)
  - `polygeist/include/mlir/Conversion/PolygeistPasses.td` (+4 / -2)
  - `polygeist/include/mlir/Conversion/PolygeistToLLVM/PolygeistToLLVM.h` (+5)
  - `polygeist/lib/Conversion/PolygeistToLLVM/CMakeLists.txt` (+1)
  - `polygeist/lib/Conversion/PolygeistToLLVM/PolygeistToLLVM.cpp` (+30 / -2)
  - `polygeist/test/polygeist-opt/sycl/rocdl-builtins.mlir` (new, +90)
  - `polygeist/tools/cgeist/CMakeLists.txt` (+3)
  - `polygeist/tools/cgeist/Lib/clang-mlir.h` (+2)
  - `polygeist/tools/cgeist/driver.cc` (+16 / -1)
  - `polygeist/tools/polygeist-opt/polygeist-opt.cpp` (+3 / -1)
  - `clang/lib/Driver/ToolChain.cpp` (+12 / -1)
  - `sycl/plugins/hip/CMakeLists.txt` (+3 / -1)
  - `BUILD_AMD.md` (new, +214)

End-to-end SYCL-MLIR support for `amdgcn-amd-amdhsa-syclmlir`: a new `-convert-sycl-to-rocdl` dialect-conversion pass, the wiring needed to drive it from `convert-polygeist-to-llvm` and from cgeist's pass pipeline, the clang-driver routing fix that lets the AMDGCN backend job run, an HIP-plugin cmake var, and a build recipe doc.

- **New `LoweringTarget::ROCDL` enum value** — added in `SYCLAttributes.td` (`I32EnumAttrCase` "ROCDL" = 1) and exposed as a `clEnumValN` choice on the `sycl-target` option of both `convert-sycl-to-llvm` and `convert-polygeist-to-llvm`. Selected from the SYCL device triple in cgeist's `getSYCLTargetFromTriple`: `amdgcn-amd-amdhsa` returns `ROCDL`, any other amdgcn variant errors with "amdgcn requires amd-amdhsa vendor/OS".
- **`SYCLToROCDL` pass** (`mlir-sycl/lib/Conversion/SYCLToROCDL/SYCLToROCDL.cpp`) — operates on each `gpu.module` inside the top-level `ModuleOp`. Patterns:
  - `NDGridOpPattern<OpTy>` template-instantiated for `SYCLLocalIDOp`, `SYCLWorkGroupIDOp`, `SYCLWorkGroupSizeOp`, `SYCLNumWorkGroupsOp`. Mapping picked through a `RocdlGridKind` enum + `rocdl_kind_of<OpTy>` traits + a switch in `buildRocdlGridDim` that emits the correct ROCDL op for each (kind, dim) pair.
  - `GlobalIDOpPattern` — synthesizes `global_id_i = workgroup_id_i * workgroup_dim_i + workitem_id_i` per dimension (no single AMDGCN intrinsic).
  - `NumWorkItemsOpPattern` — `num_work_items_i = workgroup_dim_i * grid_dim_i`.
  - `GlobalOffsetOpPattern` — fills the result with `i64 0` per dim (HIP/AMDGCN has no kernel-launch global offset).
  - `SubGroup1DPattern<OpTy>` — emits an `LLVM::LLVMFuncOp` declaration for an `__ockl_get_*` extern (`sub_group_size`, `max_sub_group_size`, `sub_group_id`, `num_sub_groups`, `sub_group_local_id`) and replaces the SYCL op with an `LLVM::CallOp` to it. The HIP device libraries supply the implementation at link time.
  - **Result-type handling:** `workitem.id`/`workgroup.id` ROCDL ops are constructed as `i32` while `workgroup.dim`/`grid.dim` are constructed as `i64`, matching the LLVM intrinsics each translates to (the `dim` ones lower to the `__ockl_get_local_size` / `__ockl_get_num_groups` ockl calls). Producing the wrong MLIR type would still verify but `translateModuleToLLVMIR` later asserts when an arith op consumes the value, so an explicit `convertScalarToDtype` widens the i32 results before the multiply-add in `GlobalIDOpPattern`.
  - **Dim mirroring:** SYCL id/range with N dimensions stores values such that index 0 is slowest-varying; AMDGCN exposes dim 0 (X) as fastest-varying. `mirrorIndex<N>` (specialized for N=1/2/3) maps SYCL i to AMDGCN dim — visible in the lit tests as e.g. SYCL dim 0 → `rocdl.workitem.id.z`.
  - **Pattern benefit = 2** — the catch-all `LLVMOpLowering` in `convert-polygeist-to-llvm` runs at benefit 1 and would otherwise eagerly rebuild SYCL grid ops with a converted (LLVM struct) result type before our patterns can match. The `resultIsSYCLIdOrRange` guard at the top of each pattern returns `failure()` when the pattern fires on a clone whose result type has already been remapped.
  - **Element accessors:** `createGetOp` switches between `SYCLIDGetOp` / `SYCLRangeGetOp` based on the result's element type; the result of each grid op is materialized as a stack `memref::AllocaOp` of the SYCL id/range type, written per-dim via the get-op + `memref::StoreOp`, then reloaded.
- **`convert-polygeist-to-llvm` plumbing** (`PolygeistToLLVM.cpp`) — when `syclTarget == ROCDL`:
  - Skip `populateSPIRVToLLVMConversionPatterns` / `populateSPIRVToLLVMTypeConversion` (no SPIR-V on the AMDGCN path).
  - Call `populateSYCLToROCDLConversionPatterns` instead of `populateSYCLToSPIRVConversionPatterns`.
  - Mark the twelve SYCL grid/sub-group ops handled by SYCLToROCDL `addIllegalOp` so the dialect-conversion driver actually invokes our patterns; without this it leaves them legal and they survive as `unrealized_conversion_cast` operands that fail reconciliation.
  - `addLegalDialect<ROCDL::ROCDLDialect>()` so the driver doesn't roll the rewrite back when no further legalization pattern exists for the just-emitted `rocdl.*` ops.
  - `dependentDialects` on the pass td gains `ROCDL::ROCDLDialect` and `memref::MemRefDialect`; `PolygeistToLLVM.h` re-exports a few dialect headers for downstream consumers.
- **`SYCLToLLVM/DPCPP.cpp`** — `populateSYCLToLLVMConversionPatterns` gets a `case LoweringTarget::ROCDL` arm that, for now, reuses the SPIR populate (`populateSYCLToLLVMSPIRConversionPatterns`) — the accessor/range/id struct layout is the same and `targetToAddressSpace` already returns AMDGPU-compatible 1/1/3. Comment notes the place to specialise as divergences surface.
- **cgeist driver** (`driver.cc`):
  - On `SYCLIsDevice`, also load `ROCDLDialect` and `AMDGPUDialect`, and register `ROCDLDialectTranslation` so MLIR→LLVM-IR translation knows how to lower the new ops.
  - In `finalize` phase 3, gate `polygeist::createLegalizeForSPIRVPass()` so it runs on host or SPIR device only — the AMDGCN device path skips it.
  - `getSYCLTargetFromTriple` extended for `Triple::amdgcn` (returns `ROCDL` when vendor/OS is `amd/amdhsa`, `createStringError` otherwise).
  - `cgeist/CMakeLists.txt` links `MLIRROCDLDialect`, `MLIRAMDGPUDialect`, and `MLIRROCDLToLLVMIRTranslation`; `clang-mlir.h` includes their headers.
- **`polygeist-opt` registration** — adds `ROCDL::ROCDLDialect` to the registry so the lit tests under `polygeist-opt --convert-polygeist-to-llvm="sycl-target=rocdl"` can parse and emit ROCDL ops.
- **Clang-driver routing fix** (`clang/lib/Driver/ToolChain.cpp`) — `SelectTool`'s SYCLMLIR branch was unconditionally returning `getCgeist()` whenever the input was LLVM IR/BC. That broke the AMDGCN device backend (LLC) job, which takes LLVM IR → assembly/object and must run through clang's own backend. Now: route to `mlir-translate` only when the *output* is `TY_MLIR_IR` (the source→MLIR step). When the input is `TY_LLVM_IR`/`TY_LLVM_BC` and the output is not MLIR, return `getClang()` so the LLC-equivalent runs through clang.
- **HIP plugin cmake var** (`sycl/plugins/hip/CMakeLists.txt`) — new `SYCL_BUILD_PI_HIP_EXTRA_INCLUDE_DIRS` cache string (semicolon-separated), threaded into the plugin's `target_include_directories`. Lets the build inject extra header paths (e.g. the `amd_comgr` shim documented in `BUILD_AMD.md`) without editing source. Marked advanced.
- **`BUILD_AMD.md`** — new top-level doc capturing the cluster-specific recipe for building `sycl-mlir` against DTK 25.04.1 / gfx906: prerequisite paths, why the stock build script doesn't work (non-interactive conda, hardcoded `/usr/bin/gcc`, DTK lib/include layout), the symlink shim setup for `dtk-libs/`+`dtk-include/amd_comgr/`, the canonical `build/build.sh` invocation, the `DeviceConfigFile.inc` build-graph race workaround, a `clang++ -fsycl-targets=amdgcn-amd-amdhsa-syclmlir` invocation, a Slurm submission template for ORISE, and a troubleshooting table.
- **lit tests:**
  - `mlir-sycl/test/Conversion/SYCLToROCDL/{grid-ops,global-id,global-offset,sub-group}.mlir` — unit tests of `sycl-mlir-opt -convert-sycl-to-rocdl` covering the ND grid patterns (1D/2D/3D pinning the dim mirroring), the multiplied-add `global_id`, the multiplied `num_work_items`, the all-zero `global_offset` (with `CHECK-NOT: rocdl.`), and the five sub-group extern-call patterns.
  - `polygeist/test/polygeist-opt/sycl/rocdl-builtins.mlir` — end-to-end via `polygeist-opt --convert-polygeist-to-llvm="sycl-target=rocdl"`, asserting that after the full lowering the funcs become `llvm.func` and the right `rocdl.*` / `__ockl_*` calls appear inside.

End-to-end effect: a SYCL kernel compiled with `-fsycl-targets=amdgcn-amd-amdhsa-syclmlir -Xsycl-target-backend --offload-arch=gfx906` now lowers cleanly through cgeist (source → MLIR with SYCL grid ops), `convert-polygeist-to-llvm sycl-target=rocdl` (SYCL grid ops → ROCDL intrinsics + ockl extern calls in LLVM dialect), MLIR→LLVM-IR translation, and clang's AMDGCN backend → object / a.out, with the HIP plugin loading at runtime. Two sharp edges worth flagging for future work: (1) `SYCLToROCDL.cpp` instantiates a fresh `TypeConverter` whose only conversion is identity, which is enough for the current grid-op patterns but means SYCL types reaching this pass cannot be remapped here; (2) the unified-runtime adapter `CMakeLists.txt` under `build/_deps/` still needs a manual one-line edit (documented in `BUILD_AMD.md`) and is not handled by the new `SYCL_BUILD_PI_HIP_EXTRA_INCLUDE_DIRS` because that var only feeds the SYCL HIP plugin, not the unified-runtime fetch-content target.

---

## 2523e7e1cf29 — [SYCL-MLIR] Fix FusionPass Fan1/Fan2 arg pairing for AMDGCN nd_item mismatch

- **Author:** Yucheng Ouyang
- **Date:** 2026-05-26
- **Files:**
  - `mlir-sycl/lib/Dialect/SYCL/Transforms/FusionPass.cpp` (+135 / -8)
  - `COMMIT_HISTORY.md` (+89)

Fallout from the AMDGCN device-target work (`fa1ad39`). Under the SPIR64 target, both `Fan1` and `Fan2` callees in the 126.ge benchmark take a 1D `nd_item` argument and the previous hard-coded `mapper.map(fan2->getArg(k), fan1->getArg(perm[k]))` permutation lined up. Under `amdgcn-amd-amdhsa-syclmlir`, `Fan2` ends up with a 2D `nd_item` (it indexes into the matrix in 2D) while `Fan1` keeps a 1D `nd_item`, so the verbatim arg map produced an `IRMapping` entry whose source and target types disagreed and the cloned Fan2 body failed to verify. Hard-coded indices were also brittle in general.

- **Argument classification** — new `ArgRole` enum (`I32Scalar` / `Accessor` / `NDItem` / `Other`) with a `classifyArg(Type)` helper. Looks at the top-level type, and for `MemRefType` looks at the element type, recognizing `i32`, `sycl::AccessorType` / `sycl::LocalAccessorType`, and `sycl::NdItemType`. Anything else returns `Other` and aborts the pass with a `dbgs()` message.
- **Role-based pairing** — replaces the four hard-coded `mapper.map` calls with a pass over Fan2's args: each Fan2 arg is matched against the next available Fan1 arg of the same role. Two cursors (`nextI32`, `nextAcc`) advance independently for `I32Scalar` and `Accessor`; `NDItem` always pairs with Fan1's first (and only) `NDItem`. The 126.ge-specific Fan1 arg-2/arg-3 swap is preserved by overwriting `pairing[2]`/`pairing[3]` after the role pass — same end result as before for the existing benchmark, but no longer load-bearing on the exact signature.
- **Type bridging** — for each `(j, i)` pair, if `fan2.arg(j).getType() == fan1.arg(i).getType()` the mapper is set up directly; otherwise an `UnrealizedConversionCastOp` is inserted at the start of Fan1's entry block casting Fan1's value to Fan2's expected type, and the mapper points at the cast result. This is what unblocks the 1D-vs-2D `nd_item` case: the cloned Fan2 ops see a "2D nd_item" SSA value whose definer is an unrealized cast from Fan1's 1D nd_item, so verification passes even though there is no real 2D nd_item in scope.
- **Cloned `get_global_id(0)` redirect** — the bridging cast leaves dead `nd_item` plumbing in Fan1, but only one consumer of the cloned Fan2 nd_item actually matters semantically: `nd_item.get_global_id(0)`, which under fusion should be the same induction variable that Fan1 uses. So before cloning, walk Fan1 for an `SYCLNDItemGetGlobalIDOp` whose `index` is the constant `0` and stash the op; track every op cloned from Fan2 in a `DenseSet<Operation *> clonedOps`; after cloning, re-walk Fan1 for `SYCLNDItemGetGlobalIDOp`s that are in `clonedOps` with a constant-`0` index and `replaceAllUsesWith(fan1G0Op.getRes())` + erase. The remaining cast and Fan2-cloned nd_item ops then become dead and fall away in the canonicalize/CSE pair that follows the pass.
- **Minor log-message tweak** — the `applyPatternsAndFoldGreedily` log line for the register-promotion pattern is renamed from `"Find patterns"` / `"Cannot find patterns"` to `"Find RegisterPromotion pattern"` / `"Cannot find RegisterPromotion pattern"` so it can be told apart from the (future) if-fusion pattern logging.

End-to-end: `gaussianElim_mono` from 126.ge now compiles to a gfx906 a.out under fusion, with the cloned Fan2 body sharing Fan1's induction variable and the bridging cast getting eliminated downstream. One sharp edge: the post-role-pairing overwrite of indices 2/3 still bakes in the 126.ge swap, so a Fan1/Fan2 pair from a different benchmark whose accessor argument order doesn't match this pattern would silently get the wrong mapping; future work is to drive that swap from a structural cue (e.g. how each accessor is indexed) rather than its ordinal.

---

## 5d965598669e — [SYCL-MLIR] Make FusionPass RegisterPromotion fire under AMDGCN

- **Author:** Yucheng Ouyang
- **Date:** 2026-05-27
- **Files:**
  - `polygeist/tools/cgeist/Lib/CGExpr.cc` (+35)
  - `mlir-sycl/include/mlir/Dialect/SYCL/Utils/Utils.h` (+1 / -1)
  - `mlir-sycl/lib/Dialect/SYCL/Utils/Utils.cpp` (+33 / -28)
  - `mlir-sycl/lib/Dialect/SYCL/Transforms/RegisterPromotionPattern.cpp` (+8 / -6)
  - `mlir-sycl/lib/Dialect/SYCL/Transforms/FusionPass.cpp` (+2 / -2)
  - `COMMIT_HISTORY.md` (+20)

Follow-on to the AMDGCN device-target enablement (`fa1ad39`) and the FusionPass arg-pairing fix (`2523e7e1`). Under SPIR64 the index operand to `sycl.accessor.subscript` arrives as a `memref.cast` of an alloca of `!sycl.id` (i.e. the SPIR64 by-pointer ABI for small aggregates). Under `amdgcn-amd-amdhsa-syclmlir` the Direct ABI passes `sycl::id` by-value, so the same operand arrives as a load of an `!sycl.id` value — which the typed `sycl.accessor.subscript` op refuses to verify (its index is constrained to `memref<?x!sycl.id>`). Two consequences: (a) the typed op was never built on AMDGCN, so `sycl.accessor.subscript` ops did not appear in the kernel and the `RegisterPromotion` pattern had nothing to match; (b) `getOffsetFromSubscriptOp`'s SPIR64-shaped chain walk (`memref.cast → affine.store → affine.load → memref.cast → memref.memory_space_cast → sycl.constructor`) does not exist on AMDGCN even when the typed op is present.

- **`CGExpr.cc`** — New file-local helper `rewriteIDLoadToMemRefCast(Builder, V)` and a corresponding loop inserted into `createSYCLMethodOp`. The helper checks whether `V` has type `sycl::IDType` and was defined by an `affine.load` / `memref.load`; if so, it materializes a `memref::CastOp` from the source memref to `memref<?x!sycl.id>` (`ShapedType::kDynamic`) in the source memory space and returns the cast result. `createSYCLMethodOp` runs this over every operand from index 1 onward (right after the existing `abstractCasts` on operand 0). Operand 0 is the receiver (the accessor) and is left untouched. The rewrite restores the SPIR64-shape index operand on AMDGCN so the typed subscript op verifies; on SPIR64 the helper short-circuits (the operand is not produced by a load of `!sycl.id`) and behavior is unchanged.
- **`Utils.h` / `Utils.cpp`** — `getOffsetFromSubscriptOp` gains a second parameter `StringAttr &tripleAttr` and is split into two arms keyed on its value:
  - `"spir64-unknown-unknown-syclmlir"` — preserves the existing six-step chain walk verbatim.
  - `"amdgcn-amd-amdhsa-syclmlir"` — walks `op.getIndex().getUsers()` directly looking for the `sycl::SYCLConstructorOp` whose first arg is the offset. Shorter because the AMDGCN-side index is a value (the loaded `sycl::id`) directly consumed by both the constructor that built it and the (newly-inserted) `memref.cast` feeding the subscript, rather than being routed through an alloca.
  - Anything else falls through with `offset` left default-constructed; a future non-SPIR/AMDGCN target would need a third arm.
- **`RegisterPromotionPattern.cpp`** — Threads the triple. Adds includes for `LLVM::LLVMDialect` and `BuiltinAttributes`, reads `llvm.target_triple` off the parent `ModuleOp` via `LLVM::LLVMDialect::getTargetTripleAttrName()`, and passes it into both `getOffsetFromSubscriptOp` calls (the load-side and the back-scan store-side). Deletes the local `curAcc`/`curIndex` shadows and uses `loadOp.getAcc()` directly in the equivalence check. Adds a `dbgs() << "Fusion Pass: Find RegisterPromotion pattern\n"` print *inside* the success branch (i.e. only when an actual `replaceAllUsesWith` happened), distinct from the per-`applyPatternsAndFoldGreedily` log in `FusionPass.cpp`.
- **`FusionPass.cpp`** — Flips the polarity of the post-`applyPatternsAndFoldGreedily` log lines, which were inverted: `failed(...)` now prints `"RegisterPromotion pattern not converge"` and the success arm prints `"RegisterPromotion pattern converged"`. Previously the failure path printed the find-message and vice versa, which had been mildly misleading since the pattern was introduced.

End-to-end: under `-fsycl-targets=amdgcn-amd-amdhsa-syclmlir`, cgeist now emits typed `sycl.accessor.subscript` ops with a `memref.cast`-shaped index (matching SPIR64), `getOffsetFromSubscriptOp` resolves the offset on the AMDGCN-shaped def-use graph, and `RegisterPromotion` fires on `gaussianElim_mono` exactly as it did on SPIR64. Two sharp edges worth flagging: (1) `Utils.cpp` still ends without a trailing newline (introduced in `e4064e58` and not addressed here); (2) `getOffsetFromSubscriptOp` takes `StringAttr &tripleAttr` by non-const reference even though it only reads it — should be `StringAttr` by value or `const StringAttr &` if a future non-null-or-default precondition is added.

---

## 75978e743de7 — [SYCL-MLIR] Route ROCDL transcendentals through ocml device library

- **Author:** Yucheng Ouyang
- **Date:** 2026-05-27
- **Files:**
  - `mlir/lib/Conversion/GPUCommon/OpToFuncCallLowering.h` (+2 / -2)
  - `polygeist/lib/Conversion/PolygeistToLLVM/PolygeistToLLVM.cpp` (+60 / -2)
  - `polygeist/test/polygeist-opt/rocdl-math-ops.mlir` (new, +59)
  - `mlir-sycl/test/Conversion/SYCLToROCDL/global-id.mlir` (+6 / -2)
  - `COMMIT_HISTORY.md` (+26)

AMDGPU has no f64 ISel pattern for the transcendental LLVM intrinsics (`llvm.intr.sin`, `cos`, `exp`, `log`, `pow`, …), so once `convert-polygeist-to-llvm` reached `MathToLLVM` on an AMDGCN target, any kernel using `math.*` on f64 would lower to `llvm.intr.*` and then fail in the AMDGPU backend at codegen. Fix: re-route those ops to the AMD ocml device library (linked at HIP-driver finalize time via `ocml.bc`) on the ROCDL `sycl-target` only, leaving the SPIR-V path untouched.

- **`OpToFuncCallLowering.h`** — Upstream MLIR `GPUCommon/OpToFuncCallLowering` (the template that rewrites a `math::*Op` to `llvm.call @<f32_or_f64_func>` based on operand type) was hard-wired to default `PatternBenefit`. Adds an optional `PatternBenefit benefit = 1` ctor parameter and forwards it to `ConvertOpToLLVMPattern<SourceOp>`'s base. Tiny upstream-shaped change, default-compatible.
- **`PolygeistToLLVM.cpp`** — Includes `Math/IR/Math.h` and reaches into the upstream private header via a relative include (`../../../../mlir/lib/Conversion/GPUCommon/OpToFuncCallLowering.h`) — annotated with a comment block explaining why. New file-local `addOcmlPattern<OpTy>(converter, patterns, f32, f64)` helper registers an `OpToFuncCallLowering<OpTy>` with `PatternBenefit(2)` so it wins against the default-benefit `MathToLLVM` pattern when both end up in the same `RewritePatternSet`. Inside `convert-polygeist-to-llvm`, on the existing `if (isROCDL)` arm — right after `populateSYCLToROCDLConversionPatterns` — registers the ocml redirect for sixteen ops: `math::{Sin,Cos,Tan,Atan,Atan2,Exp,Exp2,ExpM1,Log,Log2,Log10,Log1p,PowF,Tanh,Erf,Cbrt}Op` paired with `__ocml_<name>_{f32,f64}`. Notable omission: `math::SqrtOp` is intentionally not redirected — AMDGPU has a native f64 ISel pattern for `llvm.intr.sqrt`, so going through ocml would lose performance. The SPIR-V `else` branch is unchanged.
- **`polygeist/test/polygeist-opt/rocdl-math-ops.mlir`** — New lit test with two `RUN` lines (`sycl-target=rocdl` vs default SPIR-V) and matching `ROCDL` / `SPIRV` check prefixes. Covers: f64 transcendentals chain (`sin`/`cos`/`tan`/`exp`/`log`/`powf`) lower to `llvm.call @__ocml_*_f64` under ROCDL but stay as `llvm.intr.*` under SPIR-V; a separate f32 chain confirms the `_f32` suffix is picked correctly; a regression-guard function asserts `math.sqrt` lowers to `llvm.intr.sqrt` and emphatically NOT to `__ocml_sqrt_f64` even on the ROCDL path.
- **`mlir-sycl/test/Conversion/SYCLToROCDL/global-id.mlir`** — Test-only update for the i32/i64 mismatch fix already in `SYCLToROCDL.cpp` (workitem.id/workgroup.id materialize as i32, workgroup.dim as i64). The `CHECK` lines now expect `arith.extsi %BID : i32 to i64` and `arith.extsi %TID : i32 to i64` ahead of the `muli`/`addi`. Comment is added inline explaining why both extensions appear. No source change here — the test just catches up to the lowering's actual output.
- **`COMMIT_HISTORY.md`** — Backfills the entry for `5d965598669e` ("Make FusionPass RegisterPromotion fire under AMDGCN") that wasn't logged when first committed.

End-to-end effect: a SYCL kernel using f64 transcendentals (e.g. `sycl::sin`, `std::exp`, `std::pow` over `double`) now compiles cleanly under `-fsycl-targets=amdgcn-amd-amdhsa-syclmlir`. After `convert-polygeist-to-llvm sycl-target=rocdl`, `math.*Op`s become `llvm.call @__ocml_*_f64` rather than `llvm.intr.*`, and the AMDGPU backend resolves the calls against the ocml.bc bitcode that the HIP driver links at finalize. SPIR64 is byte-identical to before. One sharp edge: the relative include `../../../../mlir/lib/Conversion/GPUCommon/OpToFuncCallLowering.h` reaches into MLIR's private headers and will silently break if upstream restructures `GPUCommon/`; long-term the right move is either to upstream the header to `include/mlir/Conversion/GPUCommon/` or copy the small template into mlir-sycl.

---

## 986f9be7c9e8 — [SYCL-MLIR] Set cgeist LLVMMod DataLayout from target before CGM init

- **Author:** Yucheng Ouyang
- **Date:** 2026-05-28
- **Files:**
  - `polygeist/tools/cgeist/Lib/clang-mlir.h` (+8)
  - `polygeist/tools/cgeist/Test/Verification/sycl/accessor-local-amdgcn.cpp` (new, +42)
  - `COMMIT_HISTORY.md` (+23)

cgeist constructed its `llvm::Module LLVMMod` without a target `DataLayout`, so `CodeGenModule`'s record-layout queries (`getTypeAllocSize`, struct layout) fell back to LLVM defaults — which assume a 64-bit pointer in every address space. On AMDGCN that's wrong for `__local` storage: a pointer in address space 3 is 32-bit, so the LLVM default of 8 bytes disagreed with the AST's 4-byte layout for the `MData` member inside `sycl::accessor`, producing an apparent overlap with the following field that tripped `CGRecordLowering::clipTailPadding`'s `NoUniqueAddressAttr` assertion when cgeist lowered a `sycl::accessor<int, 1, ..., target::local>`.

- **`clang-mlir.h`** — Adds `#include "clang/Basic/TargetInfo.h"` and a `bool DataLayoutInitialized` member that sits between `LLVMMod` and `CGM` in the `MLIRScanner` member declaration order (so also in the initializer-list order). It is initialized with a comma expression `(LLVMMod.setDataLayout(AstContext.getTargetInfo().getDataLayoutString()), true)` — i.e. it calls `LLVMMod.setDataLayout(...)` *before* the `CGM` member is constructed. Member-init order matters here: `CGM` (the next member) caches data-layout info during construction, so the `DataLayout` must be set first. The `bool` result is just a vehicle to inject a side effect into the initializer list; it's never read.

- **`accessor-local-amdgcn.cpp`** — New regression lit test. A minimal SYCL kernel using a `target::local` accessor (`sycl::accessor<int, 1, access::mode::read_write, target::local>`) inside an `nd_range` `parallel_for`, driven through `clang++ -fsycl -fsycl-device-only -fsycl-targets=amdgcn-amd-amdhsa-syclmlir -Xsycl-target-backend --offload-arch=gfx906`. Run at both `-O0` and `-O2`. Uses `--implicit-check-not="{{Assertion|reused this field's tail padding}}"` to assert the `NoUniqueAddressAttr` assertion no longer fires, plus positive `CHECK` lines pinning that the emitted datalayout contains `p3:32:32` (the AMDGCN 32-bit local-pointer rule) and that the kernel signature takes the local accessor as `ptr addrspace(3)`. Comment block documents the root cause.

- **`COMMIT_HISTORY.md`** — Backfills the entry for `75978e7` (the ocml-transcendentals commit) that hadn't been logged when it was committed.

End-to-end effect: the local-accessor path that crashed cgeist on AMDGCN now lowers cleanly, and the AMDGCN data layout (`p3:32:32`, etc.) is visible in the emitted module. No SPIR64 impact — the SPIR64 data layout is also applied the same way (the fix is target-agnostic: it just reads whatever target `ASTContext` was configured with). One sharp edge: the comma-expression-in-initializer-list trick is unusual and easy to misread as dead code; a future cleanup could move `setDataLayout` into a small `LLVMMod`-initializing helper, but member-init ordering makes a plain statement-form fix awkward since `LLVMMod` is constructed by its own initializer.

---

## c54f8b85500c — [SYCL-MLIR] Enable struct-flattening ABI for AMDGCN cgeist codegen

- **Author:** Yucheng Ouyang
- **Date:** 2026-06-30
- **Files:**
  - `polygeist/tools/cgeist/Lib/CodeGenTypes.h` (+13 / -3)
  - `polygeist/tools/cgeist/Lib/CodeGenTypes.cc` (+28 / -25)
  - `polygeist/tools/cgeist/Lib/CGCall.cc` (+65 / -5)
  - `polygeist/tools/cgeist/Lib/clang-mlir.cc` (+105 / -13)
  - `polygeist/tools/cgeist/Lib/clang-mlir.h` (+17 / -7)
  - `polygeist/tools/cgeist/Test/Verification/packedstruct.c` (+12 / -4)
  - `polygeist/tools/cgeist/Test/Verification/sycl/struct-flatten-amdgcn.cpp` (new, +78)
  - `polygeist/tools/cgeist/Test/Verification/sycl/h_item-flatten-amdgcn.cpp` (new, +45)

Until this commit, `AllowStructFlattening` in `CodeGenTypes.cc` was hard-coded `false` with a "Need to revisit" TODO, and any aggregate whose Itanium ABI lowering classified it as `Direct` + `canBeFlattened` hit a `CGEIST_WARNING` ("struct should be flattened but MLIR codegen cannot yet handle it") and was passed as a single pointer. On SPIR64 that's tolerable because the SPIR calling convention passes small aggregates by-pointer. On AMDGCN, the AMDGPU calling convention *flattens* small aggregates into multiple scalar IR args — so the cgeist `func.func` (single aggregate arg) disagreed with its call sites (multiple scalar args) and phase-3 finalize aborted with `'llvm.call' op incorrect number of operands`. This commit turns flattening on with a guard and wires it through caller and callee symmetrically.

- **`CodeGenTypes.cc` / `CodeGenTypes.h`** — Flips `AllowStructFlattening` to `true`. The two `CGEIST_WARNING` blocks ("struct should be flattened but MLIR codegen cannot yet handle it") are deleted — flattening is now honored, so the warning is stale. The central guard, mirrored in both `ClangToLLVMArgMapping::construct` and `getFunctionType`, is: only flatten when the *MLIR-side* aggregate is an `LLVM::LLVMStructType` whose `getBody().size()` equals the LLVM `CoerceToType` element count. This is needed because `ClangToLLVMArgMapping` reasons about the LLVM CoerceToType while the MLIR arg type is built from the declared C++ type, and the two can disagree in two directions:
  - **SYCL dialect types** (e.g. `!sycl.nd_item<N>`, `!sycl.item<N>`) — LLVM coerces the underlying struct to N fields, but the MLIR type is a single opaque dialect type, not an `LLVMStructType`. Stay as one MLIR arg.
  - **SysV i64-coerced small structs** — the SysV ABI coerces a small struct (e.g. two i32s) to a single `i64`, so `NumIRArgs == 1`, but the MLIR type is still a 2-field `LLVMStructType`. The `NumIRArgs > 1` half of the guard rejects this case so it isn't wrongly flattened.
  To make `ClangToLLVMArgMapping` able to consult the MLIR type, the constructor and `construct` gain a `CodeGenTypes &Types` parameter (no longer defaulted); the header gains a forward declaration of `CodeGenTypes` and a doc comment explaining the consultation is load-bearing. All three call sites (`getFunctionType`, `constructAttributeList`, and the two in `CGCall.cc`/`clang-mlir.cc`) pass `*this` / `Glob.getTypes()`.

- **`CGCall.cc::callHelper`** (caller side) — Builds a single `ClangToLLVMArgMapping` up front (reused for the sret lookup) instead of constructing a throwaway one inside the `if (IsSRet)` block. Adds an `ABIArgInfo::Ignore` short-circuit (no IR arg corresponds to an ignored clang arg, so skip it). Changes the "too many arguments" guard from an `assert(false)` to a `return ValueCategory()` with a `mlir::emitError`, and corrects the bound check from `I >= FnType.getInputs().size()` to `Args.size() >= ...` (I is the clang-formal index, Args is the MLIR-arg index — with flattening they diverge). The new flatten block: when `Val` is an `LLVMStructType` and `ArgMapping.getIRArgs(I)` reports `NumIRArgs > 1` matching `STy.getBody().size()`, materialize `Val` into an `LLVM::AllocaOp` scratch slot (in `getAllocaAddrSpace()`) at `AllocationScope`, store the whole aggregate via `ValueCategory::store`, then `GEP` each field and `LoadOp` it, pushing N scalar args. A comment notes the deliberate AS asymmetry: the caller scratch slot lives in AllocaAS (AMDGPU private AS 5) and its GEP/Load are in AllocaAS, while the callee prologue casts its slot back to flat AS 0 — the caller slot is local to the call sequence, the callee slot is the parameter's stack slot accessed in flat AS by the body; "don't normalize one to match the other."

- **`clang-mlir.cc::MLIRScanner::init`** (callee side) — Symmetric counterpart. Builds the `ClangToLLVMArgMapping` once (hoisted out of the `if (IsSRet)` block so it's in scope). For each clang formal whose MLIR type is an `LLVMStructType` that the ABI flattened into `NumIRArgs > 1` matching fields, allocates the struct via `createAllocOp(..., /*LLVMABI=*/true)` and `GEP`+`StoreOp`s each of the N incoming IR args into its field slot, then advances `IRIdx = FirstIRArg + NumIRArgs`, calls `MaybeSkipSRet()`, and `continue`s the formal loop. Critically uses **field-wise GEP+store**, *not* `llvm.insertvalue` — because the struct's fields can be SYCL dialect types (e.g. `h_item<1>` lowers to `!llvm.struct<(!sycl.item, !sycl.item, !sycl.item)>`), and `llvm.insertvalue` requires primitive LLVM operands.

- **`clang-mlir.cc::createAllocOp`** — Routs all alloca paths through `getAllocaAddrSpace()`. Computes `StorageAS = (MemSpace == 0) ? AllocaAS : MemSpace` so a flat-AS (`MemSpace == 0`) request lands in the target's alloca AS (AMDGPU private AS 5) on the actual `LLVM::AllocaOp`/`memref::AllocaOp`, then bridges back to the requested `MemSpace` via `LLVM::AddrSpaceCastOp` when they differ. Applied symmetrically across all three branches: the `LLVMABI` scalar/VLA branch (raw alloca in StorageAS, optional addrspacecast to MemSpace), the memref branch (build the memref in StorageAS, then `Memref2PointerOp` → `AddrSpaceCastOp` → `Pointer2MemrefOp` to get a flat-AS memref), and the array-shaped memref branch (same Memref2Pointer/AddrSpaceCast/Pointer2Memref bridge). The previous code hardcoded flat AS for the alloca and only cast when `MemSpace != 0`; on AMDGCN that produced an `i64 = FrameIndex` SDNode matched against an addrspace(0) pointer, aborting in ISel.

- **`clang-mlir.h`** — `Bufs` changes from `map<const void*, vector<LLVM::AllocaOp>>` to `map<const void*, vector<Value>>` and `allocateBuffer` returns `Value`, because the cached entry may now be an `AddrSpaceCastOp` result rather than the bare `AllocaOp`. `allocateBuffer` computes `StorageTy` in `AllocaAS`, allocates there, and addrspacecasts back to the requested pointer type `T` when they differ.

- **`packedstruct.c`** — Existing lit test, updated to the new flatten shape. The `compute` caller now allocas the aggregate, stores the by-pointer arg into it, GEPs each field, loads, and passes the flattened scalars to `@run`. The `@run` call signature changes from `(!llvm.struct<(i64, i8)>, i8)` to `(i64, i8, i8)`.

- **`struct-flatten-amdgcn.cpp`** (new) — Smoke test for both fixes at once: a `SobelMin` kernel whose functor captures a 16-byte `Point` aggregate by value (exercises the flatten path) and whose parameter staging allocas must land in AS 5 (exercises the alloca-AS fix). `CHECK-DAG`s `alloca {{.*}}, addrspace(5)` and `addrspacecast ptr addrspace(5) ... to ptr`; `--implicit-check-not="{{Assertion|struct should be flattened}}"`. Explicitly not a bit-exact IR test (accessor/capture layout varies with header revisions).

- **`h_item-flatten-amdgcn.cpp`** (new) — Smoke test for the callee-side flatten prologue on a non-dialect aggregate: `h_item<1>` is *not* a SYCL dialect type (hierarchical parallelism is deferred in this build), so cgeist emits `!llvm.struct<(!sycl.item, !sycl.item, !sycl.item)>`, which AMDGPU flattens, exercising the per-field GEP+store reassembly in `MLIRScanner::init`. `CHECK`s `getelementptr` / `store` appear in the kernel body; `--implicit-check-not="{{Assertion|Callsite argument mismatch}}"`.

End-to-end effect: AMDGCN kernels that pass small aggregates by value (the common case — captured accessors, `id`, `h_item`, functor captures) now compile through phase-3 finalize without the `incorrect number of operands` verifier failure, and AMDGPU private-AS stack slots are emitted in AS 5 with addrspacecasts back to flat. Two sharp edges worth flagging: (1) the `ClangToLLVMArgMapping` consultation of `Types.getMLIRType(...)` means the mapping now depends on full MLIR type construction being available at mapping time — a future target that builds MLIR types lazily would break it; (2) the deliberate caller-vs-callee address-space asymmetry in the flatten scratch slots is load-bearing and easy to "fix" into a bug.

---

## 0ec753bb8cf9 — [SYCL-MLIR] Add Sunway driver, bench scripts, and reorganize build docs

- **Author:** Yucheng Ouyang
- **Date:** 2026-07-01
- **Files:**
  - `script/swsyclmlir` (new, +496)
  - `script/run-syclbench-sw.sh` (new, +29)
  - `script/syclbench-summary.py` (new, +90)
  - `doc/BUILD_130.md` (new, +31)
  - `doc/BUILD_SW.md` (new, +27)
  - `doc/BUILD_HIP.md` (new, +206; supersedes `BUILD_AMD.md`)
  - `doc/COMMIT_HISTORY.md` (moved from `COMMIT_HISTORY.md`, +352 / −352 at old path)
  - `BUILD_AMD.md` (deleted, −214)
  - `COMMIT_HISTORY.md` (deleted at old path)

A grab-bag commit: ports the SYCL-MLIR device-compile flow to the Sunway (sw) platform via a new Python wrapper driver, adds a benchmark submission/diffing workflow, and reorganizes the build docs under `doc/`.

- **`script/swsyclmlir`** (new, ~496 lines, Python) — A standalone driver that masquerades as `clang++`: when the user invokes `swsyclmlir --target=athread -fsycl x.cpp -o a.out`, it runs the host `clang++ -fsycl -###` to capture the full device/host compile pipeline, then intercepts each step and reroutes SYCL device code through cgeist instead of the stock device compiler. Pipeline (numbered per the source comments):
  - **1.** For each `clang-18 -cc1 -fsycl-is-device` job (device compilation), rewrites `-triple spir64-unknown-unknown` → `spir64-unknown-unknown-syclmlir`, redirects `-triple`/include roots from the swcl `intel-llvm` tree to the local cgeist install, and invokes `cgeist -emit-llvm <source> -o <output> --args <cc1 args>` to produce device LLVM bitcode.
  - **3.** `llvm-foreach` → SPIR-V step: collects the IL inputs, runs `swcl --target=<athread|openmp> --static-lib` to assemble `libSWCLKernel.a`, and writes an `empty.spv` placeholder (a minimal SPIR-V magic + `OpenCL.std` capability blob, inlined as a byte literal) for each input so the downstream offload-wrapper step has something to bundle.
  - **4.1/4.2** — `clang-offload-wrapper -kind=sycl` produces wrapper bitcode; `make_wrapper_v13_compatible` post-processes it (run `opt -mtriple=spir64`, null out empty offload-entry symbols, `llvm-as`, `llvm-spirv`, then a *second* `llvm-spirv -r` round-trip through the local toolchain's llvm-spirv to get a v13-compatible `.bc`, appending a `__swcl_sycl_descriptor_reg` registration function) so the wrapper links against the sw runtime. The `llc -filetype=obj` wrapper-compile step is rerouted to `clang -c` (athread) or `llc -mtriple=<host-triple> -relocation-model=pic` (openmp) when not using pure dpcpp.
  - **6.** `ld` link step (non-dpcpp): instead of the swcl `ld`, compiles a small C constructor (`__swcl_sycl_descriptor_reg_constructor`) that calls the descriptor registration, then links via `swg++`/`clang++` with `-lsycl -lstdc++`, the file-mapped object inputs, and the `libSWCLKernel.link_args.txt` argument list.
  - `-E -MD` depfile jobs and all other jobs are run through with only a `--fsycl-disable-range-rounding` → `-D__SYCL_DISABLE_PARALLEL_FOR_RANGE_ROUNDING__=1` rewrite.
  File mapping is handled by `map_file_in_args` / `run_with_file_mapped`, which redirect every `-o`/`--output=`/`-input=` style path into a temp dir (`--save-temps-dir`, default `.swsycl`) and track the original→temp mapping so later steps pick up the rewritten files. `--target` choices are `athread` (Sunway athread slave-core runtime; uses `swg++`/`llc`/`clang` from PATH, `-mdynamic` link) and `openmp` (uses the toolchain's `clang++`, `-fopenmp=libgomp`). Has a `DEBUG_MODE` (set when run as a script, unset when frozen via PyInstaller) that toggles argparse help/options; in frozen mode only `-h` (delegated to clang++ --help) is exposed.

- **`script/run-syclbench-sw.sh`** (new) — Submits every binary in `${SCRIPT_DIR}/benchmarks/*` to the Sunway batch system via `bsub -q q_share -b -m 1 -n 1 -cgsp 64 -share_size 13000 -priv_size 16 -host_stack 1024 -cache_size 128`, each with `--output=sycl-bench.csv` and a per-benchmark size override (`scalar_prod` / `vec_add` → `--size=1024`, else `--size=1024` default). Sets `LD_LIBRARY_PATH` to the sw runtime lib dir.

- **`script/syclbench-summary.py`** (new) — Diffs the LLVM-vs-MLIR sycl-bench CSVs. Loads `sycl-bench/build/sycl-bench.csv` (LLVM/DPC++ baseline) and `sycl-bench/build-mlir/sycl-bench-mlir.csv` (MLIR path) keyed by `(Benchmark name, local-size, problem-size)`, preserves LLVM row order then appends MLIR-only rows, and writes a `sycl-bench-summary.csv` with shared columns (`device-name`) plus per-implementation `Verification` and `run-time-median` columns (`*_llvm` / `*_mlir`). Prints a common/llvm-only/mlir-only key counts summary.

- **`doc/BUILD_130.md`** (new) — Build/run recipe for the 130 cluster: `.gitconfig` `insteadOf` rewrite to reach GitHub through the cluster mirror, the OpenCL-Headers pinning workaround (manually set `OpenCL-Headers.git/refs/heads/main` to `8275634cf9ec31b6484c2e6be756237cb583999d` because the latest commit is incompatible with sycl-mlir), `configure.py`/`compile.py -j16`, and the `PATH`/`LD_LIBRARY_PATH` (with oneAPI TBB) exports for running `clang++ -fsycl -fsycl-targets=spir64-unknown-unknown-syclmlir`.

- **`doc/BUILD_SW.md`** (new) — Build/run recipe for the Sunway/sw environment: `.gitconfig` `insteadOf = "repo:"`, `source /usr/sw/swllvm/setenv-18.sh`, configure with GCC 11.2.0 as the build compiler and `--cmake-gen "Unix Makefiles"` + `-DSYCL_LIBDEVICE_GCC_TOOLCHAIN`, then compile/run via `swsyclmlir --target=athread -fsycl x.cpp` and `bsub ... a.out` with the sw share-memory/stack flags.

- **`doc/BUILD_HIP.md`** (moved from `BUILD_AMD.md`) — The AMDGCN/HIP build doc relocates under `doc/` and gains two edits: (1) the same OpenCL-Headers pinning note as `BUILD_130.md` (set `refs/heads/main` to `8275634...`); (2) the runtime-launch Slurm snippet is slimmed — the verbose `module unload`/`DTK_HOME`/`sycl-ls`/`SYCL_PI_TRACE` block is replaced with a minimal `export LD_LIBRARY_PATH=...build/install/lib` + `ONEAPI_DEVICE_SELECTOR=hip:* ./a.out`. The build recipe itself (DTK 25.04.1 against gfx906, the `amd_comgr` shim, `build/build.sh`) is otherwise unchanged.

- **`doc/COMMIT_HISTORY.md`** — This very document, moved from the repo root to `doc/` so the build docs and history live together. Pure relocation, no content change at move time (subsequent commits, including this entry, append here).

End-to-end effect: `swsyclmlir --target=athread -fsycl x.cpp -o a.out` on a Sunway node now routes the SYCL device compile through the locally-built cgeist (producing spir64-unknown-unknown-syclmlir device IR, bundling it via the swcl toolchain, and linking against the Sunway athread runtime), and `run-syclbench-sw.sh` + `syclbench-summary.py` give a turn-key A/B comparison against the LLVM/DPC++ baseline. Sharp edges: (1) `swsyclmlir` has `/home/oyyc/sycl-mlir/build/install/bin/cgeist` and the sw runtime lib path hard-coded — they're per-developer and won't resolve elsewhere without edits; (2) the `make_wrapper_v13_compatible` SPIR-V round-trip is fragile version glue between the local `llvm-spirv` and the swcl-bundled one, and will need revisiting if either side moves; (3) the `empty.spv` placeholder is a hand-encoded byte literal rather than a generated minimal SPIR-V, so a future SPIR-V validator that checks capability consistency could reject it.

---

## [SYCL-MLIR] Fix ≥2-local-accessor aliasing on AMDGCN (getArchType gate + amdgcn.annotations)

- **Author:** Yucheng Ouyang
- **Date:** 2026-08-15
- **Files:**
  - `llvm/lib/SYCLLowerIR/TargetHelpers.cpp` (+14 / -2)
  - `polygeist/tools/cgeist/driver.cc` (+147)
  - `polygeist/tools/cgeist/Test/Verification/sycl/accessor-local-amdgcn-metadata.cpp` (new, +83)

Fixes the ≥2-local-accessor mis-compilation on the MLIR/cgeist→LLVM AMDGCN path (`amdgcn-amd-amdhsa-syclmlir`): SYCL kernels with two or more `accessor<...,target::local>` scratch buffers had their LDS tiles overlap (the 2nd accessor placed at the 1st's address +4 — one float slot — instead of +host-buffer-size), so they aliased and produced wrong results. One local accessor passed (no partner to alias with). Root-caused and GPU-verified end-to-end on gfx906.

- **Root cause (two conditions, BOTH required):**
  1. `TargetHelpers::getArchType` (`llvm/lib/SYCLLowerIR/TargetHelpers.cpp`) matched AMDHSA triples by *exact* `.Case("amdgcn-amd-amdhsa", …)` / `.Case("amdgcn--amdhsa", …)`. The SYCL-MLIR device target uses the environment-suffixed triple `amdgcn-amd-amdhsa-syclmlir` (`Triple::EnvironmentType::SYCLMLIR`, `Triple.h:258`), which matched neither case → `ArchType::Unsupported` → `LocalAccessorToSharedMemoryPass` (`llvm/lib/SYCLLowerIR/LocalAccessorToSharedMemory.cpp:56`) and `GlobalOffsetPass` (`GlobalOffset.cpp:77`) — both callers of this shared `getArchType` — early-returned and no-op'd. With the pass inert, each `ptr addrspace(3)` local-accessor kernel arg was never rewritten to an `i32` offset, so `AMDGPUHSAMetadataStreamer::getValueKind` classified it `dynamic_shared_pointer`; the runtime then spaced multiple local args by the code-object pointer-slot `.size` (=4) instead of the host buffer size → the 2nd accessor's LDS tile overlapped the 1st.
  2. `populateKernels` (`TargetHelpers.cpp:61`) finds kernels by reading the module-level `!amdgcn.annotations` named metadata; cgeist's C→MLIR→LLVM path never runs clang CodeGen for device code, so it emitted no `!amdgcn.annotations`, and `populateKernels` early-returned with no kernels found → the local-accessor pass had nothing to rewrite even once the gate was opened.

- **`TargetHelpers.cpp`** — `getArchType` switches the two AMDHSA cases from `.Case` (exact) to `.StartsWith("amdgcn-amd-amdhsa", …)` / `.StartsWith("amdgcn--amdhsa", …)`, so env-suffixed variants (`-syclmlir`) classify as `AMDHSA` while the plain `amdgcn-amd-amdhsa` / `amdgcn--amdhsa` outcomes are unchanged. CUDA cases untouched. A multi-line comment explains the gate. One edit un-gates both `LocalAccessorToSharedMemoryPass` (the lever — now rewrites `ptr addrspace(3)`→`i32`→HSA `by_value`→runtime spaces by host buffer size→distinct LDS) and `GlobalOffsetPass`; the latter has a second gate (`GlobalOffset.cpp:83`, early-returns when `@llvm.amdgcn.implicit_offset` is unused, which holds for the MLIR path's nd_range offset 0) so it stays inert — only `LocalAccessorToSharedMemoryPass` activates. The AMDGPU backend itself has zero references to `syclmlir`/`SYCLMLIR` (grepped empty); the divergence was purely this SYCL-side string gate. The principled alternative (`getArch()==amdgcn && getOS()==AMDHSA`, matching the backend's own gate at `AMDGPUTargetMachine.cpp:1118`) was noted but the prefix form chosen for minimalism; sharp edge: the prefix also matches legacy env-suffixed non-syclmlir triples (`-opencl`, `-hcc`), none of which appear in this tree, so practical risk is limited to the intended `-syclmlir` suffix.

- **`driver.cc::emitSYCLKernelArgMetadata(llvm::Module&)`** — New ROCDL-only helper called in `compileModule` right after `LLVMModule->setTargetTriple`, gated on `getSYCLTargetFromTriple(Triple) == ROCDL` && `SYCLIsDevice`. Emits the SYCL kernel-arg metadata that clang CodeGen emits on the SPIR path (`CodeGenModule::GenKernelArgMetadata` + `AMDGPUTargetCodeGenInfo::addAMDGCNMetadata`) but cgeist skips. Pure LLVM-IR-type-driven (no clang AST), so it reproduces the classification from `llvm::Function` argument types alone:
  - **`!amdgcn.annotations`** (the decisive node): one `!{ptr @kernel, !"kernel", i32 1}` per `amdgpu_kernel`, mirroring clang `AMDGPU.cpp:388`. This is what feeds `populateKernels` so the local-accessor pass can find the kernels — without it the gate fix alone is inert.
  - **Per-kernel `!kernel_arg_exclusive_ptr` / `!kernel_arg_runtime_aligned`**: share one i1 vector, `true` for each accessor base pointer arg, `false` otherwise (clang emits the same node for both at `:2429`/`:2430`). An arg is classified an accessor base pointer by `isAccessorBasePtr` — `ptr` in addrspace 1 (global) or 3 (local). This empirically matches clang's `SYCLAccessorPtrAttr` rule across every SPIR device kernel inspected (the accessor base pointers are exactly the AS1/AS3 pointer args; `i64` and `ptr addrspace(4) byref(range/id)` args are not). Emitted only when ≥1 accessor is present (`AnyAccessor`).
  - **`!kernel_arg_buffer_location`**: vector of `-1`, length = arg count, emitted regardless (matches clang `:2423`).
  - **`!sycl_fixed_targets`**: empty node `!{}` (as on the SPIR path).
  - **Module-level**: `opencl.spir.version` `{1,2}`, `opencl.ocl.version` `{2,0}`, `spirv.Source` `{4,100000}` (OpenCL_CPP non-ESIMD, the value clang uses when no `sycl_explicit_simd` kernel exists), and `amdgpu_code_object_version` module flag = 400 (COV v4) via `Module::Error` behavior.
  - Iterates `amdgpu_kernel` functions by `CallingConv` (naturally excludes host helpers). Sharp edges flagged for future work: (a) `isAccessorBasePtr` keys on address space, not clang's `SYCLAccessorPtrAttr`, so a raw non-accessor AS1/AS3 pointer arg (e.g. USM) would be mis-marked `exclusive_ptr` — today harmless because the AMDGPU backend reads `kernel_arg_exclusive_ptr` from nothing (grep empty), but it's a semantic lie vs clang a future consumer could act on; (b) `addModuleFlag(Module::Error, ...)` for COV is non-idempotent — a second call on the same module (re-entrance / linked bitcode that already set it) would trip the verifier's "module flag identifiers must be unique" (clang uses `Module::Min`); (c) the helper hand-rolls the metadata nodes that clang CodeGen already emits via reachable emitters (cgeist links clang CodeGen), so the two paths can drift — a clang-side shape change won't appear on the cgeist ROCDL path.

- **`accessor-local-amdgcn-metadata.cpp`** (new lit test) — A two-local-accessor kernel (the aliasing trigger) compiled via `clang++ -fsycl -fsycl-device-only -fsycl-targets=amdgcn-amd-amdhsa-syclmlir ... -emit-llvm -S`, FileChecking every node `emitSYCLKernelArgMetadata` emits in file order: the `define amdgpu_kernel` line carrying `!kernel_arg_buffer_location !N !kernel_arg_runtime_aligned !N !kernel_arg_exclusive_ptr !N !sycl_fixed_targets !N` (shared `!N` for exclusive_ptr/runtime_aligned), the `!amdgcn.annotations` entry `!{ptr @kernel, !"kernel", i32 1}`, the module-level `opencl.spir.version`/`opencl.ocl.version`/`spirv.Source`, the `amdgpu_code_object_version` module flag `!{i32 1, !"amdgpu_code_object_version", i32 400}`, and the numbered node definitions — notably pinning the i1 vector to `{true,false,false,false,true,false,false,false,true,false,false,false}` (true at args 0/4/8 = the two AS3 locals + one AS1 global accessor base pointer), so any regression in the AS-based classification changes the vector and fails the test. Satisfies CLAUDE.md's "Tests must accompany code changes" for the 147-line emitter.

- **Build/deploy:** the `TargetHelpers.cpp` change links into `libLLVMSYCLLowerIR.a`, statically linked by `clang-18`, `cgeist`, and `lld`. The device ELF / HSA `value_kind` is produced by **`lld` LTO** (the SYCL AMDGCN fat-binary codegen step), not `clang-18 -emit-obj` and not `llc` — so all three must be rebuilt and copied to `build/install/bin/` (the standalone external DTK `opt`/`llc` via `SYCL_AMDGCN_OPT`/`SYCL_AMDGCN_LLC` do NOT run the pass or emit the ELF: external opt only does `-O2`, external llc only compiles the host offload-wrapper). External `SYCL_AMDGCN_OPT`/`SYCL_AMDGCN_LLC` do not bypass the fix.

- **Verification (gfx906, slurm 21628894):** rebuilt cgeist from staged source; `diag.mlir` reports MLIR `pb(B)-pa(A) = 1024 bytes` (was **4** — the bug; SPIR-ref unchanged at 1024); `local_acc_repro.mlir` RA (2D+2-local) and RC (1D+2-local) **PASS** (were FAIL), RB (1-local) PASS, SPIR-ref all PASS — no regression. `check-cgeist` `accessor-local-amdgcn.cpp` PASS; the new metadata test PASS; `accessor-local.cpp` failure is the pre-existing GLIBCXX host-link baseline (SPIR path, unrelated). Codegen-level (`llvm-readobj --notes`): local args now `.value_kind: by_value` (was `dynamic_shared_pointer`).

- **What this commit does NOT include:** the `sycl-fold-local-addrspacecasts` MLIR pass (`SYCLFoldLocalAddrSpaceCasts.cpp`) that was staged alongside this fix in an earlier iteration is **dropped**. It was hypothesized to fix the bug by folding AS3→flat addrspacecasts feeding `llvm.store`/`llvm.load` (matching the working SPIR path's typed AS3 stores), but GPU runs proved it **insufficient** — driving the MLIR IR to structurally match SPIR (0 AS3→flat casts, typed AS3 stores) left the kernels failing, because the actual lever is the `getArchType` gate + `amdgcn.annotations` (the aliasing is an HSA `value_kind`/LDS-spacing problem, not an IR cast problem). The pass is additive and ROCDL-gated so it was harmless, but it carried latent bugs (indirect-call retyping gap, per-function call-site-use bail leaving the bug condition, multi-return/early-return guard mismatch) and shipped 254 lines + 2 lit tests + `.td`/`.h`/CMake wiring for no verified benefit. Dropped in favor of the minimal verified fix. See `local-accessor-amdgcn-flatcast-bug.md` (project memory) for the full falsification chain.

End-to-end effect: a SYCL kernel with ≥2 local accessors compiled with `-fsycl-targets=amdgcn-amd-amdhsa-syclmlir -Xsycl-target-backend --offload-arch=gfx906` now gets distinct, non-overlapping LDS regions for each local accessor (spaced by host buffer size via HSA `by_value`), matching the working SPIR-AMDGCN path. One local accessor was already correct and is unchanged.

---

## [SYCL-MLIR] Add CovarianceLoopReorderPass (device-only loop-reorder of the polybench covar kernel)

- **Author:** Yucheng Ouyang
- **Date:** 2026-08-26
- **Files:**
  - `mlir-sycl/include/mlir/Dialect/SYCL/Transforms/Passes.h` (+1)
  - `mlir-sycl/include/mlir/Dialect/SYCL/Transforms/Passes.td` (+39)
  - `mlir-sycl/lib/Dialect/SYCL/Transforms/CMakeLists.txt` (+1 / -1)
  - `mlir-sycl/lib/Dialect/SYCL/Transforms/CovarianceLoopReorder.cpp` (new, +344)
  - `mlir-sycl/test/Transforms/covariance-loop-reorder.mlir` (new, +86)
  - `polygeist/tools/cgeist/Options.h` (+6)
  - `polygeist/tools/cgeist/driver.cc` (+6)
  - `doc/COMMIT_HISTORY.md` (this entry)

Stage-1 of the staged covariance device-only optimization plan. A new `-sycl-covariance-loop-reorder` MLIR pass that rewrites the baseline polybench `CovarianceCovar` lambda body **in place** — no signature change, no `nd_item`, no host/launch change, no work-groups, no local memory — into a register-blocked loop reorder that cuts the dominant global-memory traffic. Off by default; gated to the `amdgcn-amd-amdhsa-syclmlir` device target.

- **Why this shape.** The baseline `CovarianceCovar` kernel launches one work-item per column `j1` via a `parallel_for(range<1>(size), id<1>(1), …)` host launch. A device pass cannot change that launch *size* (the host-side `range` is the wall — see the `m4-host-launch-not-device-pass-feasible` analysis), so the full `covariance_opt` tiled-GEMM win (547×) is unreachable from a pass. What a pass *can* do is restructure the per-item serial loop: the baseline loops `j2 ∈ [j1..M]` then `i ∈ [1..N]`, reloading column `j1` ~`(M-j1+1)` times from global memory. Hoisting that column-`j1` load out of the `j2` unroll captures bandwidth, not occupancy — a modest, real speedup rather than the 547×.

- **`CovarianceLoopReorder.cpp`** — the pass. Walks the module for `func::FuncOp`s whose `sycl.kernel_func_obj` ArrayAttr references a symbol containing `"CovarianceCovar"` (the dispatching `gpu.func @_ZTS15CovarianceCovar`; the substring uniquely excludes `CovarianceMean` / `CovarianceReduce`). For each matched body it calls `rewriteBodyReorder`, which wipes the entry block and rebuilds the compute from scratch (block arguments / function args are kept). Reads the real 6-arg baseline signature directly: `arg0`/`arg1` = M/N as `memref<?xi64>`, `arg2` = `symmat` discard-write, `arg3` = `data` read, `arg4` = `symmat2` discard-write (the mirror), `arg5` = `!sycl_item_1_`. Emits:
  - `j1 = sycl.item.get_id(item, 0)` (i64 → `index_cast`), M/N via `memref.load arg{M,N}[0]` → `index_cast`. `scf.for` is half-open, so the `j2 ≤ M` / `i ≤ N` baseline bounds become `j2 < M+1` / `i < N+1`.
  - Outer `scf.for %j2b = j1 to M+1 step B` (B = `tile-size`, default 16), no iter_args.
  - Inner `scf.for %i = 1 to N+1` carrying `B` f32 `iter_args` (init 0.0). Column `j1` is loaded **once per `i`** (`data[i, j1]`, hoisted out of the `jj` unroll); each `jj ∈ [0,B)` computes `j2 = j2b+jj`, clamps the data subscript to `min(j2, M)` (so the tail of the last block never issues an OOB accessor subscript), and folds `fma(a, data[i, min(j2,M)], acc[jj])` guarded by `arith.select(j2 ≤ M, fma, acc[jj])` so the tail of the last block is skipped rather than accumulated.
  - After the `i`-loop, the `B` accumulators are stored guarded by `scf.if(j2 ≤ M)`: the upper cell `symmat[j1, j2]` via `arg2`, the mirror `symmat2[j2, j1]` via `arg4`. `j2 == j1` (the diagonal) writes the same cell twice, matching the baseline mirror of the `j2=j1` iteration.
  - Accessor subscripts are built with `mkId`/`loadData`/`storeSym` helpers (`sycl.id.constructor` + `sycl.accessor.subscript` + `memref.load`/`store`), all copied from the sibling `CovarianceTiledGEMM.cpp`'s idiom. The dead `symmat[j1,j1] = 1.0` store is dropped (overwritten by the `j2=j1` variance sum; the CPU reference `covariance()` does not set 1.0 either).
  - **Correctness:** for each `j2` the accumulator is summed over `i` in the *same* `1..N` order as the baseline — only the `j2` loop is reordered/blocked — so the result is bit-identical FP to the baseline. Whatever offset behavior the baseline has (cf. the `rocdl_global_offset_hardcoded_zero` known issue) is inherited unchanged: `j1` is still `sycl.item.get_id`, bounds still `1..N` / `j1..M`, accessor subscripts still take the 1-based logical index directly.

- **Two gates, both load-bearing.** (1) the opt-in `--sycl-covariance-loop-reorder` flag (off by default, wired in cgeist `driver.cc` where the pass is only added to the pipeline when set), and (2) the module's `llvm.target_triple` attribute must equal `amdgcn-amd-amd-hsa-syclmlir`. The in-pass target guard is necessary because the pass can also be reached via `polygeist-opt` / lit, where the driver flag is absent; on any other target the pass is a no-op. `sycl.kernel_func_obj` is a SYCL-dialect attribute (`SYCLBase.td::getKernelFuncObjAttrName`), so the pass needs no Polygeist dialect — `sycl-mlir-opt` registers only `SYCLDialect`.

- **Two gotchas that cost iteration (documented in the source):**
  1. **Body wipe must erase top-level ops in REVERSE program order** (`for (op : reverse(topOps)) op->erase()`), not `Block::getOperations().clear()` (forward-order destroy trips "operation destroyed but still has uses" on the non-trivial baseline body — only worked on the trivial `{return}` lit body) and not a `walk`-based erase (double-free: nested ops erased both explicitly and via their parent). Non-entry blocks are dropped after.
  2. **The 2-D id payload MUST be the array form `!sycl.id<[2], (!sycl.array<[2], (memref<2xi64>)>)>`** (the `!sycl_id_2_` spelling the baseline emits), not `!sycl.id<[2], (i64)>`. The `(i64)` form verifies in `sycl-mlir-opt`/lit *and* in `cgeist -emit-mlir` (the `sycl.accessor.subscript` verifier only checks id dim == accessor dim), but fails phase-3 (`-c`/binary) with `'llvm.getelementptr' op type 'i64' cannot be indexed (index #2)` / `Finalize failed (phase 3)`: `AccessorSubscriptIDIndexPattern::getLinearIndex` (DPCPP.cpp) does `Res = Res*Mem[I] + Id[I]` GEPing `Id[I]` per dim, and `IDIndexConstructorPattern` stores args via `SYCLIDGetOp` — both GEP the id payload, which only works when it is the indexable 2-element array.

- **Wiring.** `Passes.td` declares `CovarianceLoopReorderPass` (`-sycl-covariance-loop-reorder`, `ModuleOp`, `tile-size` option default 16, `num-detected`/`num-rewritten` statistics, the usual `arith`/`memref`/`affine`/`gpu`/`scf`/`math`/`func` dependent dialects); `Passes.h` declares `createCovarianceLoopReorderPass()`; `Transforms/CMakeLists.txt` adds the source (and drops a stale trailing-whitespace on the `MLIRTransforms` link line); `Options.h` adds `EnableCovarianceLoopReorder`; `driver.cc` adds the pass + a canonicalize/CSE pair to `PM`, gated on the flag, alongside the existing fusion/inline blocks.

- **`covariance-loop-reorder.mlir`** (lit) — `sycl-mlir-opt -split-input-file -sycl-covariance-loop-reorder -mlir-pass-statistics`. Two chunks exercising both gates: an `amdgcn-amd-amdhsa-syclmlir` chunk (both gates pass → 1 detected / 1 rewritten, and the rewritten body carries the reorder shape — `sycl.item.get_id`, outer `scf.for`, inner `scf.for` with 16 f32 `iter_args`, hoisted `sycl.accessor.subscript` data load, `math.fma`, `arith.select` tail guard, upper + mirror stores), and a `spir64-unknown-unknown-syclmlir` chunk with the same anchor (gate #2 fails → 0 / 0, body untouched). Type aliases are taken verbatim from the cgeist-lowered `covariance.cpp` IR.

- **GPU results (gfx906, size 1024, `--local=256 --num-runs=10`, DTK 25.04.1 runtime):** baseline (pass OFF) median 1.060 s, Verify PASS; loop-reorder (pass ON) median 0.214 s, Verify PASS → ~5×. (`covariance_opt` full tiled GEMM is 0.0019 s → 547×, unreachable via pass — host-launch wall.) Bench wiring (`sycl-bench` `COVARIANCE_LOOP_REORDER` option + `-Xcgeist --sycl-covariance-loop-reorder`) lives outside this repo and is not included here.

Two sharp edges worth flagging: (1) the per-`j2` `i`-summation order is preserved so the transform is bit-identical FP, but the `j2` loop is *blocked* (`j2b`, `j2b+1`, … `j2b+B-1` per outer iteration) rather than strictly sequential — within a block the `B` accumulators are independent by construction, so this does not change any `i`-sum, but a future stricter FP-identity requirement (strict `j2` order across blocks) would need the block stride reconsidered; (2) `tile-size` is a compile-time constant baked into the IR as `B` separate `iter_args` / unrolled `jj` ops — a large `tile-size` inflates the IR (and register pressure) linearly, and AMDGPU `promote-alloca` rejects arrays > 16 elements, which is why the default is 16 and the accumulators are emitted as distinct `iter_args` rather than a `memref<16xf32>`.

End-to-end effect: a `CovarianceCovar` kernel compiled with `-fsycl-targets=amdgcn-amd-amdhsa-syclmlir -Xcgeist --sycl-covariance-loop-reorder` now lowers the per-item body to a register-blocked loop reorder with the column-`j1` load hoisted out of the `j2` unroll, giving ~5× on gfx906 at size 1024 with bit-identical results and Verify PASS. Off by default; the SPIR64 path and the baseline AMDGCN build are byte-identical to before. Stage-2 (1-D LDS tiling via grid ops, de-risked on a tiny kernel first) is not started.

---

## [SYCL-MLIR] Add SYRK register-accumulator + paired launch-tile passes for AMDGCN

- **Author:** Yucheng Ouyang
- **Date:** 2026-08-26
- **Files:**
  - `mlir-sycl/include/mlir/Dialect/SYCL/Transforms/Passes.h` (+2)
  - `mlir-sycl/include/mlir/Dialect/SYCL/Transforms/Passes.td` (+89)
  - `mlir-sycl/lib/Dialect/SYCL/Transforms/CMakeLists.txt` (+2)
  - `mlir-sycl/lib/Dialect/SYCL/Transforms/SyrkRegisterAccumulator.cpp` (new, +356)
  - `mlir-sycl/lib/Dialect/SYCL/Transforms/SyrkRegisterTile.cpp` (new, +539)
  - `mlir-sycl/test/Transforms/syrk-register-accumulator.mlir` (new, +71)
  - `mlir-sycl/test/Transforms/syrk-register-tile.mlir` (new, +81)
  - `mlir-sycl/tools/sycl-mlir-opt/CMakeLists.txt` (+1)
  - `mlir-sycl/tools/sycl-mlir-opt/sycl-mlir-opt.cpp` (+2)
  - `polygeist/tools/cgeist/Options.h` (+18)
  - `polygeist/tools/cgeist/driver.cc` (+12)
  - `llvm/include/llvm/SYCLLowerIR/SYCLRewriteSyrkRange.h` (new, +62)
  - `llvm/lib/SYCLLowerIR/SYCLRewriteSyrkRange.cpp` (new, +197)
  - `llvm/lib/SYCLLowerIR/CMakeLists.txt` (+1)
  - `llvm/lib/Passes/PassBuilder.cpp` (+1)
  - `llvm/lib/Passes/PassRegistry.def` (+1)
  - `llvm/test/SYCLLowerIR/sycl-rewrite-syrk-range.ll` (new, +59)
  - `llvm/test/SYCLLowerIR/sycl-rewrite-syrk-range-noop.ll` (new, +30)
  - `clang/include/clang/Basic/LangOptions.def` (+1)
  - `clang/include/clang/Driver/Options.td` (+3)
  - `clang/lib/CodeGen/BackendUtil.cpp` (+11)
  - `doc/COMMIT_HISTORY.md` (this entry)

Three off-by-default passes for the baseline polybench `Syr2k2` (`polybench/syrk.cpp`) kernel under `amdgcn-amd-amdhsa-syclmlir`, staged after the `CovarianceLoopReorderPass` template. Two are device-only MLIR passes in `mlir-sycl`; the third is a host LLVM pass in `llvm/lib/SYCLLowerIR/` that pairs with the second device pass to reproduce `polybench/orise/syrk_opt.cpp`'s source-level win (an 8×8 register micro-kernel per work-item plus a `range<2>(N,N)` → `range<2>(ceil(N/8), ceil(N/8))` launch-grid shrink) without touching benchmark source.

The baseline `Syr2k2` kernel launches one work-item per output cell via `parallel_for(range<2>(N,N), item<2>(2), …)`; each item does `C[item]*=beta; for k: C[item]+=alpha*A[{i,k}]*A[{j,k}]` as a per-`k` read-modify-write on `C[item]`, with `k` kept in a stack slot (`scf.while` + `llvm.store …%slot`). A device pass cannot change the launch *size* (the host-side `range` is the wall — see the `m4-host-launch-not-device-pass-feasible` analysis), so the full `syrk_opt` win — whose arithmetic-intensity gain (0.5 → 4.0 FMA/load) depends on each item owning an 8×8 tile, which in turn depends on the shrunk grid — is unreachable from a device pass alone. The two device passes capture, respectively, the device-feasible subset (single-C-RMW promotion) and the device half of the full win (the 8×8 tile, with the host pass supplying the missing grid shrink).

### `SyrkRegisterAccumulatorPass` (`-sycl-syrk-register-accumulator`)

Stage-1, standalone device-only. Rewrites the `Syr2k2` lambda body **in place** (no signature change, no `nd_item`, no host change, no work-groups, no local memory) to a register-accumulator form: hoist `i=item.get_id(0)`, `j=item.get_id(1)`, and the `M` bound out of the loop; convert the `k` `scf.while` into an `scf.for` carrying a single `f32` accumulator seeded with `C[{i,j}]*beta`; one global read of `C` and one global write of `C` after the loop (vs the baseline's per-`k` RMW). Reads the real 2-arg baseline signature directly: `arg0` = captured struct `memref<?x!llvm.struct<(i64, rw_acc, r_acc)>>` (member 0 = M bound i64, member 1 = C accessor, member 2 = A accessor) projected via `polygeist::SubIndexOp`, `arg1` = `!sycl_item_2_`. The `k`-summation order is preserved (`0..M`) and the FMA operands are bit-identical, so the result is bit-identical FP to the baseline.

**Outcome:** GPU-verified correct on gfx906 (Verify PASS both ON/OFF) but **net-neutral** on performance (OFF median 0.116 s vs ON median 0.115 s at size 1024, within ~0.002 s stddev). The LLVM backend already promotes the per-`k` `C[item]` RMW to a register PHI, so the transform adds nothing on top — confirming the `polybench-pass-d-net-negative` finding for syrk specifically. Kept as a commitable, correctness-preserving, off-by-default scaffold.

### `SyrkRegisterTilePass` (`-sycl-syrk-register-tile`) + `SYCLRewriteSyrkRangePass` (`-fsycl-rewrite-syrk-range`)

The paired host+device optimization that actually captures the `syrk_opt` win. The device pass rewrites the `Syr2k2` body so each work-item computes the 8×8 register tile at `(bi, bj) = (item.get_id(0)*8, item.get_id(1)*8)`: seed `regC[a][b] = C[{bi+a,bj+b}]*beta` (one global read per cell), `scf.for %k=0..M` of 64 `math.fma` into `regC`, then one predicated store per cell. No LDS, no barrier, no cooperative loading — a faithful port of `syrk_opt`'s algorithm. The host pass shrinks the launch grid by rewriting the two `store i64` into the `%UserRange` alloca of `parallel_for_lambda_impl` to `ceil(V/8) = (V+7) udiv 8`.

**`SYCLRewriteSyrkRange.cpp`** (host) — a `PassInfoMixin` module pass registered as `sycl-rewrite-syrk-range`, hooked into `EmitAssemblyHelper::RunOptimizationPipeline` at `PipelineStartEP` (before AlwaysInliner/SROA, where the `__SYCL_ALWAYS_INLINE` handler code is still in its own `parallel_for_lambda_impl` function and the range alloca stores are intact), gated on `LangOpts.SYCLIsHost && LangOpts.SYCLRewriteSyrkRange`. The launch site is identified by (1) a call/invoke of `getRoundedRangeILi2EE` (the 2-D `parallel_for` range entry; `syrk.cpp` has exactly one) whose two i64 arguments are loads from a common alloca, and (2) the kernel-name anchor: a private string constant `_ZTS6Syr2k2` (from `__builtin_sycl_unique_stable_name(Syr2k2)`) somewhere in the module. At `PipelineStartEP` the call's loads typically read a temporary copy (`%agg.tmp11`) filled by a `memcpy` from the real `%UserRange` alloca; the pass traces that memcpy back to the origin and rewrites the origin's two i64 stores to `ceil(V/8)`, so every downstream consumer (`getRoundedRange` args, the range-rounding wrapper's `UserRange` member, `checkValueRange`, `MNDRDesc.set`→`GlobalSize`) sees the shrunk range — exactly source-equivalent to writing `range<2>(ceil(N/8), ceil(N/8))`. Safe no-op if either anchor does not match. Limitation: no-ops if range rounding was compiled out (`-fsycl-disable-range-rounding`, also auto-added at `-O0`); the benchmark builds at `-O3`.

**`SyrkRegisterTile.cpp`** (device) — two rewrite sites, because the cgeist frontend emits the `Syr2k2` body in two places with different consumers:
- **Site 1 — `func.func` (the wrapper path):** the top-level lambda body `(memref<?x!llvm.struct<(i64, rw_acc, r_acc)>>, !sycl_item_2_)`. `gpu.func @__pf_kernel_wrapperI6Syr2k2EE` calls it once per enumerated user item (the `RoundedRangeIDGenerator` loop), later inlined into `RoundedRangeKernel::operator()`. The ONLY path exercised when range rounding fires. The whole body is wiped (reverse-erase) and rebuilt via the shared `emitTileBody`.
- **Site 2 — `gpu.func @_ZTS6Syr2k2` (the direct kernel):** body born **inlined** from the frontend (it never calls the `func.func`). The path exercised when NO rounding is needed — the common case, and the case the paired host pass produces (`ceil(N/8)` is usually already local-size divisible). Without this site the host pass would shrink the launch while the direct kernel still ran the baseline single-cell body → 1/64 of C computed, wrong results (this was the root cause of an earlier Verify FAIL). `rewriteDirectKernel` keeps the entry-block prologue (kernel-arg → captured-struct reassembly via the `polygeist::Pointer2MemrefOp` result typed `memref<?x!llvm.struct<(i64, rw_acc, r_acc)>>` plus the accessor `__init`), locates the zero-operand `sycl.call @getItem` (the work-item), erases everything after it, and appends `emitTileBody` + `gpu.return`.

- **regC representation (the iteration-costing gotcha, documented in source):** `regC` is a `memref<8x8xf32,5>` (private AS 5) alloca at **function scope** (not inside the guard), and every regC access uses a **constant** index — the 8×8 seed / k-body / store bodies are **fully unrolled**, leaving only the `k`-loop as a loop. This mirrors the `syrk_opt` kernel's machine code (64 `v_fma`, ~80 global loads, zero private-memory accesses). A loop-nest form lowers to a `[64 x float]` alloca with loop-variable GEP indices; LLVM's `AMDGPUPromoteAlloca::tryPromoteAllocaToVector` rejects arrays > 16 elements and SROA cannot partition variable-index accesses, so the accumulators would live in scratch memory (measured: 410 buffer ops per kernel — pathological, and the cause of an earlier Verify-hang). With constant indices, SROA slices `regC` into 64 scalar registers.
- **No `reqd_work_group_size`:** each work-item is now an independent tile, so any work-group shape is valid and `LocalSize` stays auto. An earlier iteration tried annotating `reqd_work_group_size [8,1,8]` on the wrapper kernel; that path was abandoned because amdgcn HSA metadata requires a 3-element array while the SYCL runtime's `program_manager` dimension check requires the element count to match the 2-D launch — an irreconcilable conflict that gets any `reqd` value rejected at launch.
- **Fail-safe guard:** the tile computation is wrapped in `if (bi < M && bj < M)`. In paired mode (host pass fired) the launch is `ceil(N/8)×ceil(N/8)` and the guard is dead. If the host pass did not fire (launch still `N×N`), items with id ≥ `ceil(N/8)` exit immediately and every tile is still computed exactly once by the item whose id IS the tile index — correct for any launch size, merely not faster.
- **Edge handling for N % 8 ≠ 0:** loads use a clamped index (`min(idx, M-1)`, safe, garbage into discarded cells); stores are predicated per cell (`bi+a < N && bj+b < N`). Out-of-range `regC` cells are computed but never stored.

### Wiring

- **`Passes.td` / `Passes.h`** declare both device passes (`-sycl-syrk-register-accumulator`, `-sycl-syrk-register-tile`; `ModuleOp`; `num-detected`/`num-rewritten` statistics; the usual `arith`/`memref`/`affine`/`gpu`/`scf`/`math`/`func` dependent dialects) and their factories. `Transforms/CMakeLists.txt` adds both sources.
- **`sycl-mlir-opt`** gains `MLIRPolygeistDialect` (CMake link + `registry.insert` + header include) because both passes use `polygeist::SubIndexOp` / `polygeist::Pointer2MemrefOp`. (The covariance pass needs none of this — it uses only SYCL-dialect attrs/ops.)
- **`Options.h` / `driver.cc`** add `--sycl-syrk-register-accumulator` and `--sycl-syrk-register-tile` (off by default), each adding the pass + a canonicalize/CSE pair to `PM` alongside the existing fusion/inline/covariance blocks.
- **Host pass wiring:** `LangOptions.def` adds `SYCLRewriteSyrkRange`; `Options.td` adds `-fsycl-rewrite-syrk-range` (cc1-only, marshalled to that langopt); `BackendUtil.cpp` registers it at `PipelineStartEP` gated on `SYCLIsHost && SYCLRewriteSyrkRange`; `PassBuilder.cpp` + `PassRegistry.def` register `sycl-rewrite-syrk-range`; `llvm/lib/SYCLLowerIR/CMakeLists.txt` adds the source.

### Lit tests

- **`sycl-register-accumulator.mlir` / `sycl-register-tile.mlir`** (`sycl-mlir-opt -split-input-file -sycl-{...} -mlir-pass-statistics`) — lock in the `Syr2k2` detection anchor, gate #2 (the `amdgcn-amd-amdhsa-syclmlir` triple guard; a `spir64` chunk is a no-op), and the signature-validation guard (non-matching body → `num-detected=1`/`num-rewritten=0`, body preserved). The full rewrite targets the real `Syr2k2` body whose captured-struct arg is `memref<?x!llvm.struct<…>>`, which `sycl-mlir-opt`'s asm parser rejects as a memref element type ("invalid memref element type") — a parser limitation, not a pass bug; the production cgeist/clang pipeline consumes that exact type and the full `-fsycl` build links cleanly. The full rewrite is therefore verified out-of-tree via the cgeist IR dump and the gfx906 GPU run (the test headers document this). The tile test additionally pins the two rewrite sites (wrapper `func.func` + direct `gpu.func @_ZTS6Syr2k2`).
- **`sycl-rewrite-syrk-range.ll`** (positive, `opt -passes=sycl-rewrite-syrk-range -S`) — the real `PipelineStartEP` shape: the `getRoundedRangeILi2EE` call's two i64 args load from a temporary `%agg.tmp11` filled by a memcpy from `%UserRange`; asserts the pass traces the memcpy back and rewrites `%UserRange`'s two stores to `(V+7) udiv 8`. **`sycl-rewrite-syrk-range-noop.ll`** (negative) — same launch-site shape but no `_ZTS6Syr2k2` string anchor → no-op.

### GPU results (gfx906, size 1024, `--local=256 --num-runs=10`, DTK 25.04.1 runtime)

- **Register-accumulator:** OFF median 0.116 s / ON median 0.115 s → net-neutral, Verify PASS both.
- **Launch-tile (paired host+device):** OFF median 0.127 s / ON median 0.095 s → **1.34×**, Verify PASS. `syrk_opt` source-level reference measured 0.083 s (1.53×) same session → the pass pair captures ~88% of the source-level win, with zero benchmark-source changes. N%8≠0 sanity (size 1030): PASS (ceil-div launch + partial tiles + per-cell predication + rounding path all correct). Disassembly of the tile kernel: 0 private-scratch buffer ops, 64 `v_fma`, ~80 global loads — byte-shape match with `syrk_opt`'s winner. Bench wiring (`sycl-bench` `SYRK_REGISTER_ACCUMULATOR` / `SYRK_LAUNCH_TILE` options + the per-target `-Xcgeist` / `-Xclang -fsycl-rewrite-syrk-range` flags) lives outside this repo and is not included here.

Two sharp edges worth flagging: (1) the launch-tile win is **paired** — the device pass alone, with the host pass disabled, is a silent no-op on the live path at N=1024 (no rounding fires, the direct kernel runs the baseline body), so the two passes must be enabled together (`SYRK_LAUNCH_TILE` does this); enabling only the device pass is correctness-preserving (the fail-safe guard keeps every tile computed exactly once) but gives no speedup. (2) `regC` as `memref<8x8xf32,5>` is only register-promoted because every access uses a constant index (full 8×8 unroll); a future "parametrize the tile size" change that made the regC indices variable would silently regress to the 410-buffer-op scratch-memory pathology, and `AMDGPUPromoteAlloca`'s >16-element rejection caps any such parametrisation at ≤16 anyway.

End-to-end effect: a `Syr2k2` kernel compiled with `-fsycl-targets=amdgcn-amd-amdhsa-syclmlir -Xcgeist --sycl-syrk-register-tile -Xclang -fsycl-rewrite-syrk-range` now runs an 8×8 register-tiled body on a `ceil(N/8)×ceil(N/8)` launch, giving 1.34× on gfx906 at size 1024 with Verify PASS and bit-identical results; the register-accumulator pass is available standalone as a correctness-preserving net-neutral rewrite. Both off by default; the SPIR64 path and the baseline AMDGCN build are byte-identical to before.

---

## [SYCL-MLIR] Add SYR2K launch-tile pass pair for AMDGCN (device register-tile + host launch shrink)

- **Author:** Yucheng Ouyang
- **Date:** 2026-08-26
- **Files:**
  - `mlir-sycl/include/mlir/Dialect/SYCL/Transforms/Passes.h` (+1)
  - `mlir-sycl/include/mlir/Dialect/SYCL/Transforms/Passes.td` (+49)
  - `mlir-sycl/lib/Dialect/SYCL/Transforms/CMakeLists.txt` (+1)
  - `mlir-sycl/lib/Dialect/SYCL/Transforms/Syr2kRegisterTile.cpp` (new, +583)
  - `mlir-sycl/test/Transforms/syr2k-register-tile.mlir` (new, +84)
  - `polygeist/tools/cgeist/Options.h` (+12)
  - `polygeist/tools/cgeist/driver.cc` (+6)
  - `llvm/include/llvm/SYCLLowerIR/SYCLRewriteSyr2kRange.h` (new, +66)
  - `llvm/lib/SYCLLowerIR/SYCLRewriteSyr2kRange.cpp` (new, +197)
  - `llvm/lib/SYCLLowerIR/CMakeLists.txt` (+1)
  - `llvm/lib/Passes/PassBuilder.cpp` (+1)
  - `llvm/lib/Passes/PassRegistry.def` (+1)
  - `llvm/test/SYCLLowerIR/sycl-rewrite-syr2k-range.ll` (new, +59)
  - `llvm/test/SYCLLowerIR/sycl-rewrite-syr2k-range-noop.ll` (new, +30)
  - `clang/include/clang/Basic/LangOptions.def` (+1)
  - `clang/include/clang/Driver/Options.td` (+3)
  - `clang/lib/CodeGen/BackendUtil.cpp` (+10)
  - `doc/COMMIT_HISTORY.md` (this entry)

Second instance of the paired host-LLVM/device-MLIR launch-tile pattern (the first being the SYRK pair, previous entry), applied to the baseline polybench `Syr2k1` kernel (`polybench/syr2k.cpp`) to reproduce `polybench/orise/syr2k_opt.cpp`'s source-level win without touching benchmark source: `Syr2kRegisterTilePass` (`-sycl-syr2k-register-tile`, device MLIR, cgeist pipeline) + `SYCLRewriteSyr2kRangePass` (`-fsycl-rewrite-syr2k-range`, host LLVM, PipelineStartEP). The two pass pairs are distinguished purely by kernel anchor — `Syr2k1`/`_ZTS6Syr2k1` here vs `Syr2k2`/`_ZTS6Syr2k2` in SYRK (syrk.cpp's tag class is confusingly also named `Syr2k2`) — so they never cross-fire; the passes are named after the benchmark to avoid ambiguity.

The baseline `syr2k.cpp` kernel is `C = alpha*(A*B^T + B*A^T) + beta*C` over `range<2>(N,N)`: one work-item per output cell, `C[item] *= BETA` then a `k`-loop doing `C[item] += ALPHA*A[{i,k}]*B[{j,k}] + ALPHA*B[{i,k}]*A[{j,k}]` as a per-`k` global RMW. The source-level `syr2k_opt` shrinks the launch to `range<2>(ceil(N/8), ceil(N/8))` and gives each work-item an 8×8 register tile: per `k`, 16 loads (rows `bi+a` of A and B, columns `bj+b` of A and B) feed 2 FMAs per cell (the A·Bᵀ term and the B·Aᵀ term) — arithmetic intensity ~8 FLOP/load vs the baseline's ~0.5, and 64× less C traffic (one read to seed, one guarded write).

### What differs from the SYRK clone (the real content of this commit)

- **Four-member captured struct.** The `syr2k.cpp` lambda captures three accessors — `memref<?x!llvm.struct<(i64, C rw_acc, A r_acc, B r_acc)>>` (member 2 = A, member 3 = B; the A/B order verified from the baseline body's subscript patterns in a live cgeist IR dump). Validation requires exactly this shape (`getBody().size() == 4`, i64 first) via a shared `isSyr2kStruct` helper, so a malformed match aborts cleanly with the body preserved rather than mis-indexing member 3. (SYRK's check is `>= 3`.) A safety property unique to syr2k: members 2/3 are type-identical, and even a hypothetical A/B capture-order swap would be result-preserving (the two k-terms swap).
- **`emitTileBody` k-loop structure.** Per `k`: 16 hoisted loads (`avA[a]=A[{clamp(bi+a),k}]`, `avB[a]=B[{clamp(bi+a),k}]`, `bvA[b]=A[{clamp(bj+b),k}]`, `bvB[b]=B[{clamp(bj+b),k}]`), an alpha pre-scale (folds away — see below), then 64 cells × 2 chained `math.fma`: `regC[a][b] = fma(avB[a], bvA[b], fma(avA[a], bvB[b], regC[a][b]))` — matching `syr2k_opt`'s load-then-reuse loop shape exactly. Live register set at the k-body: 64 regC accumulators + 16 loaded scalars ≈ 80 VGPRs — same pressure as the proven SYRK kernel, fine on gfx906's 256.
- **Constants.** `ALPHA`/`BETA` are `constexpr 1` in `syr2k.cpp` (unlike syrk's 123/14512), so the emitted `* alpha` / `* beta` are dead multiplications kept only to stay a mechanical sibling; the canonicalizer+CSE pair that follows the pass in `driver.cc` folds them.

Everything else transfers verbatim from `SyrkRegisterTile.cpp`: the **two mandatory rewrite sites** (site 1: the top-level lambda-body `func.func` — the wrapper path exercised when range rounding fires; site 2: the direct kernel `gpu.func @_ZTS6Syr2k1`, body born inlined, whose `polygeist::Pointer2MemrefOp` captured-struct prologue + 3× accessor `__init` calls are preserved and everything after the zero-operand `sycl.call @getItem` is erased in reverse order and replaced — without this site the shrunk launch would compute 1/64 of C); the `regC` `memref<8x8xf32,5>` function-scope alloca with fully-unrolled constant indices (so SROA slices it into 64 scalar registers — the AMDGPUPromoteAlloca >16-element rejection and the 16KB-private-segment/lld-HSA-metadata failure modes are documented in source); the fail-safe `scf.if (bi < M && bj < M)` guard; clamped loads (`min(idx, M-1)`) + per-cell predicated stores for N%8≠0; the item→tile-origin idiom (`sycl.item.get_id` with the array-form id payload `!sycl.id<[2], (!sycl.array<[2], (memref<2xi64>)>)>` that phase-3 GEP lowering requires); no `reqd_work_group_size` (the amdgcn-HSA-3-element vs program_manager-2-D-dim-count conflict — LocalSize stays auto 256); reverse-order entry-block erase; and both gates (opt-in cgeist flag + `llvm.target_triple == amdgcn-amd-amdhsa-syclmlir`).

### Host pass

`SYCLRewriteSyr2kRange.cpp` is byte-for-byte the SYRK host-pass logic with only the kernel anchor changed to `_ZTS6Syr2k1` (the debug-type/statistic prefixes renamed to `syr2k`). Anchors on the private string constant from `__builtin_sycl_unique_stable_name(Syr2k1)`; finds the single `getRoundedRangeILi2EE` call whose two i64 args are loads from a common alloca (tracing a filling memcpy back to the origin `%UserRange` alloca when the loads read a temporary copy — the syr2k host IR at PipelineStartEP reads the origin directly, no memcpy); rewrites every i64 store into that alloca to `(V+7) udiv 8`. Verified in the emitted host IR: both `%UserRange.i` stores become `(V+7) >> 3` (the optimizer legalizes the udiv to lshr for the positive range values), so `getRoundedRange`, the rounding wrapper's `UserRange` member, `checkValueRange`, and `MNDRDesc.set`→`GlobalSize` all see `ceil(N/8)`. Wired exactly as SYRK: `LangOptions.def` (`SYCLRewriteSyr2kRange`), `Options.td` (`-fsycl-rewrite-syr2k-range`), `BackendUtil.cpp` (PipelineStartEP, gated on `SYCLIsHost && SYCLRewriteSyr2kRange`, before AlwaysInliner/SROA where the range alloca stores are intact), `PassRegistry.def`/`PassBuilder.cpp` (`sycl-rewrite-syr2k-range` for `opt -passes=`), `SYCLLowerIR/CMakeLists.txt`. Same limitation: no-op when range rounding is compiled out (`-fsycl-disable-range-rounding`, auto at -O0) — the `getRoundedRange` anchor is then absent; benchmarks build at -O3, and the device fail-safe guard keeps results correct (each tile computed exactly once by the item whose id IS the tile index) even then, merely not faster.

### Lit tests

Mirrors of the SYRK pair, adapted to the syr2k anchors. `syr2k-register-tile.mlir` (`sycl-mlir-opt -split-input-file -sycl-syr2k-register-tile -mlir-pass-statistics`): positive `amdgcn-amd-amdhsa-syclmlir` chunk with `sycl.kernel_func_obj = [@_ZTS6Syr2k1]` and a `memref<?xi64>` stand-in arg0 (the asm parser still rejects `memref<?x!llvm.struct<…>>` element types — the full rewrite is verified out-of-tree via the cgeist IR dump and the GPU run; the test header documents this) → `1 num-detected / 0 num-rewritten`, body preserved verbatim; negative `spir64` chunk → no detection. `sycl-rewrite-syr2k-range.ll` (positive) reproduces the real PipelineStartEP shape (memcpy'd `%agg.tmp11` traced back to `%UserRange`) and CHECKs the `(V+7) udiv 8` store rewrite; `sycl-rewrite-syr2k-range-noop.ll` (negative) CHECK-NOTs `udiv` when the `_ZTS6Syr2k1` anchor is absent. All four pass; the SYRK siblings are unregressed.

### Verification (gfx906, size 1024, `--num-runs=10`, DTK 25.04.2 runtime)

- **IR:** out-of-tree cgeist dump confirms both rewrite sites fire — the wrapper `func.func` body carries the 4-member struct unpack (`polygeist::SubIndexOp` at indices 0-3), `sycl.item.get_id`-derived tile origin, `memref<8x8xf32,5>` regC, the fail-safe guard, 64 clamped seed loads, and the single `scf.for` k-loop with 16 loads + 128 `math.fma` per iteration (64 cells × 2 terms); the direct `gpu.func @_ZTS6Syr2k1` keeps its 3× `__init` prologue and has the post-`getItem` tail replaced. Host `-emit-llvm -S` shows both `%UserRange.i` stores rewritten.
- **GPU:** baseline (passes off) median 0.2495 s, Verify PASS; launch-tile ON median 0.1698 s, Verify PASS → **1.47×** (syr2k does 2× the FMAs per cell vs SYRK, so the higher win over SYRK's 1.34× is expected). N%8≠0 sanity (size 1030, ceil-div launch + partial tiles + per-cell predication + rounding path): Verify PASS. Bench wiring (`sycl-bench` `SYR2K_LAUNCH_TILE` CMake option + the per-target `-Xcgeist --sycl-syr2k-register-tile` / `-Xclang -fsycl-rewrite-syr2k-range` flags, `build-ab-syr2k.sh`, `syr2k-launch-tile-verify.sbatch`) lives outside this repo and is not included here.

Two sharp edges carried over from SYRK, both still load-bearing: (1) the win is **paired** — the device pass alone on the live path (no rounding fires at N=1024, the direct kernel runs) is correctness-preserving but a silent no-op, so the passes must be enabled together; (2) the regC constant-index full-unroll is what gets the accumulators into registers — any future variable-index form regresses to the scratch-memory pathology. A new one specific to this pair: the third accessor raises the k-body live set to ~80 f32 values; a future BM/BN retune downward (e.g. 4×4) is the occupancy fallback if a different gfx target clips VGPRs.

End-to-end effect: a `Syr2k1` kernel compiled with `-fsycl-targets=amdgcn-amd-amdhsa-syclmlir -Xcgeist --sycl-syr2k-register-tile -Xclang -fsycl-rewrite-syr2k-range` now runs an 8×8 register-tiled body (2 FMAs per cell per k) on a `ceil(N/8)×ceil(N/8)` launch, giving 1.47× on gfx906 at size 1024 with Verify PASS; results within the benchmark's 0.05 percentDiff threshold. Off by default; the SPIR64 path and the baseline AMDGCN build are byte-identical to before.

---

## [SYCL-MLIR] Add 2mm local-tile pass pair for AMDGCN (device LDS-tile + host launch pad + SYCL_FORCE_LOCAL_SIZE)

- **Author:** Yucheng Ouyang
- **Date:** 2026-08-26
- **Files:**
  - `mlir-sycl/include/mlir/Dialect/SYCL/Transforms/Passes.h` (+1)
  - `mlir-sycl/include/mlir/Dialect/SYCL/Transforms/Passes.td` (+67)
  - `mlir-sycl/lib/Dialect/SYCL/Transforms/CMakeLists.txt` (+2)
  - `mlir-sycl/lib/Dialect/SYCL/Transforms/TwoMmLocalTile.cpp` (new, +648)
  - `mlir-sycl/test/Transforms/2mm-local-tile.mlir` (new, +114)
  - `polygeist/tools/cgeist/Options.h` (+15)
  - `polygeist/tools/cgeist/driver.cc` (+7)
  - `llvm/include/llvm/SYCLLowerIR/SYCLRewrite2mmRange.h` (new, +69)
  - `llvm/lib/SYCLLowerIR/SYCLRewrite2mmRange.cpp` (new, +226)
  - `llvm/lib/SYCLLowerIR/CMakeLists.txt` (+1)
  - `llvm/lib/Passes/PassBuilder.cpp` (+1)
  - `llvm/lib/Passes/PassRegistry.def` (+1)
  - `llvm/test/SYCLLowerIR/sycl-rewrite-2mm-range.ll` (new, +125)
  - `llvm/test/SYCLLowerIR/sycl-rewrite-2mm-range-noop.ll` (new, +40)
  - `clang/include/clang/Basic/LangOptions.def` (+1)
  - `clang/include/clang/Driver/Options.td` (+3)
  - `clang/lib/CodeGen/BackendUtil.cpp` (+10)
  - `sycl/source/detail/config.def` (+1)
  - `sycl/source/detail/config.hpp` (+58)
  - `sycl/source/detail/scheduler/commands.cpp` (+48)
  - `doc/COMMIT_HISTORY.md` (this entry)

Third instance of the paired host-LLVM/device-MLIR launch-tile pattern (after SYRK and SYR2K), applied to the baseline polybench 2mm benchmark — but with a **different tile technology** and a **new runtime hook**: the earlier register-tile form of this pass (8×8 regC per item on a shrunk `ceil(N/8)` launch) was GPU-verified CORRECT but **2.2× SLOWER** than baseline (0.0328 vs 0.0148 s at N=1024) while the source-level `2mm_opt` (LDS-tiled) measured 4.6× FASTER (0.0032 s); the pass was replaced in place by the LDS design. The commit therefore contains three coordinated pieces:

1. **Device MLIR pass `sycl-2mm-local-tile`** (`TwoMmLocalTilePass`, cgeist, off by default): a faithful MLIR transcription of `2mm_opt`'s `submitTiledGemm`. TS = TK = 16. Each 16×16 work-group (256 work-items) cooperatively stages a 16×16 tile of A (rows) and B (columns) into two private module-scope `memref.global` 16×16 f32 tiles in the SYCL local address space (AS 3, same construction as LoopInternalization's WGLocalMem; per-kernel names `_2mm_k{1,2}_{a,b}Tile`) — ONE coalesced global load of each per work-item per k-chunk — then every work-item runs a 16-FMA micro-kernel on the staged tiles into a SCALAR accumulator (an `scf.for` iter_arg — a register; NO accumulator memref anywhere) and stores its single output cell (cell = item id). ONE instance rewrites BOTH benchmark kernels (`Polybench_2mm_1` `C += A*B` seeded from the existing read_write C; `Polybench_2mm_2` `E = C*D` discard_write seeded from zero) at BOTH rewrite sites — the wrapper-path lambda-body `func.func` and the born-inlined direct `gpu.func @_ZTS15Polybench_2mm_{1,2}` (the path that always runs; lesson from SYRK's `_ZTS6Syr2k2`). Edge handling for N%16≠0: loads clamped (`min(idx, M-1)`), final store predicated (`irow < M && jcol < M`), and out-of-range TAIL K-LANES store 0.0 into the tile via a branch-free `arith.select` (clamping alone would double-count the last k row/column — a latent bug in `2mm_opt` itself, exposed by this pass's N=1030 verify round 1 and fixed here).
2. **Host LLVM pass `SYCLRewrite2mmRange`** (`-fsycl-rewrite-2mm-range`, PipelineStartEP, `SYCLIsHost`-gated): anchors on the private string constant from `__builtin_sycl_unique_stable_name(Polybench_2mm_N)`; finds each 2-D `getRoundedRangeILi2EE` call whose two i64 range args load from a common alloca; traces a filling memcpy back to the origin `%UserRange` alloca when needed; rewrites every i64 store into that alloca to `((V+15) udiv 16) * 16` — the FULL PADDED grid, one work-item per output CELL (not the old ceil(N/8) shrink). Divisibility by 16 satisfies the UR launch validator for the explicit local size, and ceil16 values never trigger range rounding (multiples of the rounding MinFactor 16), so the direct-kernel path is the one that runs.
3. **Runtime env `SYCL_FORCE_LOCAL_SIZE=16,16`** (new in this tree, first of the family): `SYCLConfig<SYCL_FORCE_LOCAL_SIZE>` parse-once support in `config.hpp` (per-dim "a,b,c", trailing dims default 1, nullopt on unset/malformed/zero) + a `getForcedLocalSize()` helper in `scheduler/commands.cpp` applied at BOTH enqueue sites (regular `SetKernelParamsAndLaunch` and the command-buffer path) in the `else` branch after `COMPILE_WORK_GROUP_SIZE` enforcement. Compatibility rules keep unrelated kernels unaffected: only launches WITHOUT an explicit local size (plain range `parallel_for`, never `nd_range`), whose every global dimension is divisible by the forced per-dim size, and whose total does not exceed the device's max work-group size, are overridden. NECESSARY because the UR HIP adapter's `simpleGuessLocalWorkSize` sizes only dim 0 (auto local is `(L,1)`), `UR_KERNEL_GROUP_INFO_COMPILE_WORK_GROUP_SIZE` is stubbed to `{0,0,0}` so `reqd_work_group_size` can never be enforced, and `program_manager.cpp` rejects a 3-element reqd against a 2-D launch anyway.

**Barrier op: `rocdl.barrier`** — the only barrier that survives this pipeline: no sycl.barrier op exists; `gpu.barrier` is explicitly illegal in ConvertPolygeistToLLVM; `spirv.ControlBarrier` has no ROCDL lowering. The ROCDL dialect is legal in the phase-3 target and its LLVMIR translation emits `fence(workgroup)` + `llvm.amdgcn.s.barrier`. This is why `MLIRROCDLDialect` joins the Transforms CMake LINK_LIBS and the pass declares a ROCDL dependent dialect.

**Lit tests:** `2mm-local-tile.mlir` (positive amdgcn chunk with `memref<?xi64>` stand-in arg0 → `num-detected` statistics; negative spir64 chunk; the asm parser's `memref<?x!llvm.struct<...>>` rejection is documented in-file — the full rewrite is verified via the cgeist IR dump and the GPU run); `sycl-rewrite-2mm-range.ll` (positive; CHECKs the `+15 udiv 16 * 16` padded store rewrite on the real memcpy'd PipelineStartEP shape) and `sycl-rewrite-2mm-range-noop.ll` (negative; CHECK-NOT `udiv` with no 2mm anchor). All three pass.

**Verification (gfx906, size 1024, DTK 25.04.x runtime, job 21762797):** device ELF (2-step unbundle of the nested hipv4-amdgcn bundle) shows 8 `s_barrier` (2 per k-body × 4 rewrite sites) and 56 `ds_read`/`ds_write`; `_ZTS15Polybench_2mm_1` alone carries exactly 2 `s_barrier` + 14 LDS ops. GPU: baseline 0.0153 s Verify PASS; local-tile 0.0033 s Verify PASS → **4.61× speedup, BEATS the source-level 2mm_opt (0.0037 s)**. N=1030 (N%16≠0) 0.0055 s Verify PASS — the tail-chunk select-zero fix confirmed (round 1 at N=1030 FAILed on the clamp double-count). Verify PASS is itself proof the forced (16,16) local took effect (wrong work-group shape ⇒ garbage tiles ⇒ FAIL). Bench wiring (`sycl-bench` `TWO_MM_LAUNCH_TILE` CMake option + per-target flags, sbatch exporting `SYCL_FORCE_LOCAL_SIZE=16,16` and `SYCL_DISABLE_PARALLEL_FOR_RANGE_ROUNDING=1`) lives outside this repo and is not included.

Sharp edges: (1) the win is **paired + env-dependent** — all three pieces must be enabled together (`-Xcgeist --sycl-2mm-local-tile -Xclang -fsycl-rewrite-2mm-range` + `SYCL_FORCE_LOCAL_SIZE=16,16`); the device pass alone is correctness-preserving but a silent no-op, and without the env var the padded launch gets auto `(L,1)` groups and produces garbage. (2) The env var is process-global: any concurrently-run range-launch kernel whose global dims happen to be divisible by 16×16 and within the WG limit would also be forced — benign for the benchmark suites run under it, but a reason to keep it out of default environments. (3) The scalar-accumulator-as-iter_arg shape is what keeps the accumulator in a register; reintroducing an accumulator memref regresses to the scratch-memory pathology.

End-to-end effect: a 2mm benchmark compiled with `-fsycl-targets=amdgcn-amd-amdhsa-syclmlir -Xcgeist --sycl-2mm-local-tile -Xclang -fsycl-rewrite-2mm-range` and run with `SYCL_FORCE_LOCAL_SIZE=16,16` now runs both GEMM kernels as 16×16 LDS-tiled bodies on the ceil16(N)×ceil16(N) padded grid, giving 4.61× on gfx906 at size 1024 with Verify PASS and beating the source-level 2mm_opt. Off by default; the SPIR64 path and the baseline AMDGCN build are byte-identical to before.

---
