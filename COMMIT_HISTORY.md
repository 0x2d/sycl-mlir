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
