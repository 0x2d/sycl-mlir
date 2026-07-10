# Compilation Optimization Techniques in the SYCL-MLIR (cgeist) Pipeline

**Scope:** Optimization passes executed by `polygeist/tools/cgeist/driver.cc` across the four pipeline stages (`canonicalize` → `optimize` → `optimizeCUDA`/`finalizeCUDA` → `finalize`), with pass defaults drawn from `polygeist/tools/cgeist/Options.h`.

---

## 1. Executive Summary

The cgeist MLIR pipeline implements a substantial, **manually staged** optimization strategy. Rather than a single parameter-driven pass manager (as in LLVM's `-O2`), cgeist interleaves general-purpose MLIR cleanups with SYCL-specific transforms across four explicit pipeline stages. The majority of optimizations are **on by default** and gated by `cl::opt` flags. A key architectural property: optimization is **target-bifurcated** — SPIR/AMDGCN device code and CUDA-lowered code follow different optimization subsequences, with the AMDGCN path (this fork's focus) receiving a leaner optimization set than CUDA.

Optimization falls into six functional categories: cleanup/simplification, SSA & scalar promotion, loop transformations, interprocedural optimization, SYCL-dialect-specific transforms, and final lowering/legalization cleanup. Each non-trivial transform is bookended by `Canonicalizer + CSE + Mem2Reg` cycles so that simplification opportunities exposed by one pass are reaped before the next runs.

---

## 2. Pipeline Orchestration

Entry point: `createAndExecutePassPipeline()` (driver.cc:918), which sequences:

| Stage | Function | Trigger | Purpose |
|---|---|---|---|
| 1 | `canonicalize()` | always (O0-gated intensity) | Early restructuring of `func.func` IR |
| 2 | `optimize()` | `OptLevel != O0` | SYCL host raising, IPO, device opts |
| 3a | `optimizeCUDA()` | `-cuda-lower` (default off) | CUDA-specific loop/unroll/barrier opts |
| 3b | `finalizeCUDA()` | `-cuda-lower` | CUDA post-opt cleanup |
| 4 | `finalize()` | always | Lowering affine→SCF, func→llvm, SPIR/legalize |

Target selection happens in `finalize()` via `getSYCLTargetFromTriple()` (driver.cc:749), dispatching `LoweringTarget::SPIR` or `AMDGCN` (AMDGCN requiring the device library).

`OptLevel` is parsed from the optimization level argument (driver.cc:1220); **at O0, stage 2 is skipped entirely**, falling through from minimal canonicalization straight to lowering.

---

## 3. Optimization Techniques by Category

### 3.1 Cleanup & Simplification (running throughout)

| Technique | Implementation | Role |
|---|---|---|
| Common Subexpression Elimination | `mlir::createCSEPass` | Eliminates redundant computations; invoked ~15+ times across stages |
| Canonicalization | `mlir::createCanonicalizerPass` | Folding, dedup, pattern-based simplification; the pipeline's most frequent pass (~20 invocations) |
| Trivial-use removal | `polygeist::createRemoveTrivialUsePass` | Removes trivial indirect uses exposed after mem2reg |
| Dead symbol elimination | `mlir::createSymbolDCEPass` | Removes unreachable functions/globals (finalize stage) |

These are **not optimizations in isolation but enablers**: each is run after every transformative pass to keep IR in a canonical shape for the next.

### 3.2 SSA & Scalar Promotion

| Technique | Implementation | Default | Notes |
|---|---|---|---|
| Memory-to-SSA promotion | `polygeist::createMem2RegPass` | on | Polygeist's memref→SSA promoter, analog of LLVM `mem2reg`/`SROA`; run after most transforms |
| Scalar replacement | `createFuncPass` (scalar rep) | on (`ScalarReplacement`, Options.h:150) | Promotes aggregate components to scalars |
| Argument promotion | `polygeist::createArgumentPromotionPass` | on | Interprocedural: promotes aggregate `byval` args to scalar copies at call sites (optimize stage) |

### 3.3 Loop Optimizations

| Technique | Implementation | Default | Path |
|---|---|---|---|
| Loop-invariant code motion | `polygeist::createLICMPass` + `mlir::createLoopInvariantCodeMotionPass` | on (`EnableLICM`, Options.h:127) | all |
| Loop restructuring | `polygeist::createLoopRestructurePass` | on | all |
| Loop internalization | `polygeist::createLoopInternalizationPass` | on (`EnableLoopInternalization`, Options.h:135) | device — folds loop setup into kernel entry |
| Affine raising | `polygeist::createRaiseSCFToAffinePass` | gated (`RaiseToAffine`) | all — enables polyhedral analysis |
| Affine CFG replacement | `polygeist::createReplaceAffineCFGPass` | on | restructures CFG for affine analysis |
| Canonicalize-for | `polygeist::createCanonicalizeForPass` | on | normalizes loop forms |
| Loop unrolling | `affine::createLoopUnrollPass` | on (`LoopUnroll`, Options.h:153) | **CUDA path only** |

**⚠️ Notable gap:** Loop unrolling runs *only* under `optimizeCUDA`/`finalizeCUDA`. The SPIR/AMDGCN path does not unroll in MLIR — it defers to the LLVM backend / device library. For AMDGCN-bound kernels, this is a missed structural optimization at the MLIR level.

### 3.4 Interprocedural Optimization (IPO)

| Technique | Implementation | Default | Role |
|---|---|---|---|
| SYCL-aware inlining | `sycl::createInlinePass({InlineMode::Simple})` | on | Inlines SYCL method/accessor ops; runs in optimize stage |
| Kernel disjoint specialization | `polygeist::createKernelDisjointSpecializationPass` | on | Specializes kernels on disjoint access-range sets |
| Argument promotion (IPO form) | `polygeist::createArgumentPromotionPass` | on | Cross-function aggregate→scalar |

### 3.5 SYCL-Dialect-Specific Transforms

These are the optimizations unique to this fork:

| Technique | Implementation | Default | Role |
|---|---|---|---|
| SYCL host raising | `polygeist::createSYCLHostRaisingPass` | on | Raises host-side SYCL constructs from lowered IR |
| Constant propagation | `sycl::createConstantPropagationPass` | gated (`EnableSYCLConstantPropagation`, Options.h:130) | SYCL-aware constant folding across host/device |
| Reduction detection | `polygeist::createDetectReductionPass` | on (`DetectReduction`, Options.h:157) | Recognizes reduction idioms for later lowering |
| Kernel fusion | `sycl::createFusionPass` | on (`EnableFusionPass`, Options.h:161) | Fuses adjacent kernels / access ranges (see `126_ge_fusion_plan`) |
| Parallel lowering | `polygeist::createParallelLowerPass` | on (CUDA path) | Lowers parallel constructs |
| Barrier handling | `polygeist::createBarrierRemovalContinuation`, `createCPUifyPass` | CUDA path | Barrier/codegen reshaping |

### 3.6 Lowering & Legalization (finalize / Phase 3)

| Technique | Implementation | Role |
|---|---|---|
| Affine lowering | `mlir::createLowerAffinePass` | affine dialect → SCF |
| Arithmetic expansion | `arith::createArithExpandOpsPass` | expands composite arith ops before LLVM lowering (PM3) |
| Polygeist→LLVM conversion | `createConvertPolygeistToLLVM` | `func.func`→`llvm.func`, `func.call`→`llvm.call` (**PM3** — where ABI mismatches surface) |
| Cast reconciliation | `createReconcileUnrealizedCastsPass` | cleans up type bridges |
| SPIR-V legalization | `polygeist::createLegalizeForSPIRVPass` | SPIR path only |
| OpenMP lowering/opt | `createConvertSCFToOpenMPPass`, `createOpenMPOptPass` | host OpenMP path |

---

## 4. Structural Characteristics

1. **Explicit, alternating clean-pass interleaving.** No single "optimizer" pass manager; every transform is wrapped in `Canonicalizer + CSE + Mem2Reg`. This maximizes simplification reach but inflates compile time. O0 disables most of stage 2.

2. **Host/device separation.** Host OpenMP code and device code are optimized by distinct OpPassManagers; `eraseHostCode` is called between stages (driver.cc:123, 459, 785) to prune after host raising and before device lowering.

3. **Target bifurcation.** SPIR/AMDGCN share a leaner optimization subsequence; CUDA (`-cuda-lower`) adds unrolling, barrier removal, CPUify, and parallel lowering. The fork's primary target (AMDGCN) therefore relies more heavily on downstream LLVM backend optimization than the CUDA path does.

4. **Defaults are aggressively on.** Only `-cuda-lower` defaults to false; all SYCL-specific and loop flags default true, so a standard device build already exercises host raising, constant propagation, fusion, reduction detection, LICM, internalization, and inlining.

---

## 5. Gaps & Observations

| Gap | Detail |
|---|---|
| No SCCP | No sparse conditional constant propagation at MLIR level beyond the SYCL-specific ConstantPropagation. Relies on LLVM backend. |
| Limited DCE | Only `SymbolDCE`; per-instruction dead-code elimination deferred to LLVM. |
| No GVN / dead-arg-elim | Not present in MLIR stages. |
| **Loop unrolling missing on AMDGCN** | Unroll runs only on the CUDA path; AMDGCN kernels do not get MLIR-level unrolling. |
| High compile-time | Alternating canonicalize/CSE/mem2reg is intentional but costly. |

---

## 6. Conclusion

The pipeline is **optimization-rich and SYCL-specialized**, but its optimization is structurally different from a conventional LLVM pipeline: it is explicitly staged, clean-pass-interleaved, and target-bifurcated. The most distinguishing techniques relative to stock MLIR are the SYCL-dialect passes — host raising, kernel disjoint specialization, reduction detection, and kernel fusion — supplemented by interprocedural argument promotion and SYCL-aware inlining. The main structural weakness is the **asymmetric AMDGCN path**, which forgoes loop unrolling and several CUDA-only transforms, deferring that work to the LLVM backend.
