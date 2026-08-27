//===-- Passes.td - SYCL pass definition file --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SYCL_TRANSFORMS_PASSES
#define MLIR_DIALECT_SYCL_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def FusionPass
    : Pass<"sycl-fusion", "::mlir::ModuleOp"> {
  let summary = "SYCL fusion pass";
  let description = [{
    Perform kernel fusion.
  }];
  let constructor = "mlir::sycl::createFusionPass()";
  let dependentDialects = ["arith::ArithDialect"];
}

def ConstantPropagationPass
    : Pass<"sycl-constant-propagation", "::mlir::ModuleOp"> {
  let summary = "SYCL host-device constant propagation";
  let description = [{
    This pass implements SYCL host-device constant propagation. In order to do
    so, it tracks constant values, e.g., those defined by `arith.constant`,
    passed as an argument to `sycl.host.schedule_kernel`, and replaces the
    corresponding `gpu.func` argument with the constant value.

    This pass is also capable of propagating constant members of
    `sycl::accessor` arguments, i.e., offset and range.

    No further change is needed to the `sycl.host.schedule_kernel` operation or
    the kernel, as the Dead Argument Elimination SYCL LLVM pass (`DAESYCL`) will
    remove the kernel arguments that become unused as the result of this pass.
  }];
  let options = [
    Option<"relaxedAliasing", "relaxed-aliasing", "bool",
            /*default=*/"false",
            "Whether to use strict aliasing (i.e. type-based aliasing) rules or not">,
  ];
  let constructor = "mlir::sycl::createConstantPropagationPass()";
  let dependentDialects = ["arith::ArithDialect", "memref::MemRefDialect"];

  let statistics = [
    Statistic<"NumPropagatedConstants", "num-propagated-constants",
              "Number of propagated constants">,
    Statistic<"NumReplacedImplicitArguments", "num-replaced-implicit-args",
              "Number of replaced implicit arguments">,
    Statistic<"NumReplacedExplicitArguments", "num-replaced-explicit-args",
              "Number of replaced explicit arguments">
  ];
}

def InlinePass : Pass<"inliner"> {
  let summary = "Inline function calls";
  let description = [{
    Replace a function call operation with the body of the callee.
  }];
  let constructor = "mlir::sycl::createInlinePass()";
  let options = [
    Option<"InlineMode", "mode", "enum InlineMode", 
      "mlir::sycl::InlineMode::Simple", "Inline mode to attempt", 
      "llvm::cl::values("
        "clEnumValN(mlir::sycl::InlineMode::AlwaysInline, "
          "\"alwaysinline\", \"Inline callees with 'alwaysinline' attribute\"), "
        "clEnumValN(mlir::sycl::InlineMode::Simple, "
          "\"simple\", \"Inline callees using simple heuristics\"), "
        "clEnumValN(mlir::sycl::InlineMode::Aggressive, "
          "\"aggressive\", \"Inline callees using aggressive heuristics\"))">,
    Option<"RemoveDeadCallees", "remove-dead-callees", "bool", /*default=*/"true",
           "Remove callees that become unreachable after inlining them">,
    Option<"MaxIterationCount", "max-num-iters", "unsigned", /*default=*/"3",
           "Maximum number of inlining iterations for each SCC">,
  ];
  let dependentDialects = ["memref::MemRefDialect"];

  let statistics = [
    Statistic<"NumInlinedCalls", "num-inlined-calls", "Number of inlined calls">
  ];
}

def CovarianceLoopReorderPass
    : Pass<"sycl-covariance-loop-reorder", "::mlir::ModuleOp"> {
  let summary = "Reorder the polybench covariance covar kernel (loop-reorder + column-j1 cache)";
  let description = [{
    A device-only optimization of the baseline polybench `CovarianceCovar`
    lambda body. The baseline launches one work-item per column `j1` (a
    `range<1>` host launch that a device pass cannot change) and each item
    serially loops `j2` then `i`, reloading column `j1` ~`(M-j1+1)` times from
    global memory. This pass rewrites the body in place (no signature change, no
    `nd_item`, no host change) to a register-blocked loop reorder: an outer
    `j2`-block loop of width `tile-size` carrying `tile-size` f32 accumulators,
    with the `i` loop inside and the column-`j1` load hoisted out of the `j2`
    unroll, cutting column-`j1` global traffic by ~`tile-size`x. The per-`j2`
    `i`-summation order is preserved, so the result is bit-identical to the
    baseline. The dead `symmat[j1,j1]=1.0` store is dropped (overwritten by the
    variance sum); the mirror store through `symmat2` is preserved.

    Gated on two conditions, both of which must hold: (1) the opt-in
    `--sycl-covariance-loop-reorder` flag (off by default), and (2) the
    module's `llvm.target_triple` attribute must equal
    `amdgcn-amd-amdhsa-syclmlir`. On any other target the pass is a no-op.
  }];
  let options = [
    Option<"TileSize", "tile-size", "unsigned", /*default=*/"16",
           "j2-block width B (number of register accumulators)">,
  ];
  let constructor = "mlir::sycl::createCovarianceLoopReorderPass()";
  let dependentDialects = ["arith::ArithDialect", "memref::MemRefDialect",
                            "affine::AffineDialect", "gpu::GPUDialect",
                            "scf::SCFDialect", "math::MathDialect",
                            "func::FuncDialect"];
  let statistics = [
    Statistic<"NumDetected", "num-detected",
              "Number of CovarianceCovar bodies detected">,
    Statistic<"NumRewritten", "num-rewritten",
              "Number of CovarianceCovar bodies rewritten to reordered form">,
  ];
}

def SyrkRegisterAccumulatorPass
    : Pass<"sycl-syrk-register-accumulator", "::mlir::ModuleOp"> {
  let summary = "Rewrite the polybench Syr2k2 kernel to a register accumulator (single C read/write, scf.while->scf.for, hoist i/j/M)";
  let description = [{
    A device-only optimization of the baseline polybench Syr2k2 lambda body.
    The baseline launches one work-item per output cell (a range<2>(N,N) host
    launch whose size a device pass cannot change -- see memory
    m4-host-launch-not-device-pass-feasible) and each item does
    C[item]*=beta; for k: C[item]+=alpha*A[i,k]*A[j,k] as a per-k
    read-modify-write on C[item] with k kept in a stack slot (scf.while +
    llvm.store ...%slot).

    This pass rewrites the body in place (no signature change, no nd_item,
    no host change, no work-groups, no local memory) to a register-accumulator
    form: it hoists i=item.get_id(0), j=item.get_id(1), and the M bound out
    of the loop, converts the k scf.while into an scf.for carrying a single
    f32 accumulator seeded with C[i,j]*beta, and stores the result once
    after the loop (one global read of C, one global write of C). This is the
    device-feasible subset of what the source-level polybench/orise/syrk_opt.cpp
    register-blocking rewrite does; the remaining syrk_opt win (the 8x8 register
    micro-kernel) requires changing the host launch grid, which a device pass
    cannot do. The LLVM backend already promotes the per-k C[item] RMW to a
    register PHI, so this transform is expected to be correctness-preserving
    and roughly net-neutral on performance (it may expose backend vectorization
    that the scf.while/stack-k form inhibits). The k-summation order is
    preserved (0..M), so the result is bit-identical FP to the baseline.

    Gated on two conditions, both of which must hold: (1) the opt-in
    --sycl-syrk-register-accumulator flag (off by default), and (2) the
    module's llvm.target_triple attribute must equal
    amdgcn-amd-amdhsa-syclmlir. On any other target the pass is a no-op.
  }];
  let constructor = "mlir::sycl::createSyrkRegisterAccumulatorPass()";
  let dependentDialects = ["arith::ArithDialect", "memref::MemRefDialect",
                            "affine::AffineDialect", "gpu::GPUDialect",
                            "scf::SCFDialect", "math::MathDialect",
                            "func::FuncDialect"];
  let statistics = [
    Statistic<"NumDetected", "num-detected",
              "Number of Syr2k2 bodies detected">,
    Statistic<"NumRewritten", "num-rewritten",
              "Number of Syr2k2 bodies rewritten to register-accumulator form">,
  ];
}

def SyrkRegisterTilePass
    : Pass<"sycl-syrk-register-tile", "::mlir::ModuleOp"> {
  let summary = "Rewrite the polybench Syr2k2 kernel so each work-item computes the 8x8 register tile at (item_id*8, item_id*8) (no LDS/barrier/reqd_work_group_size); pair with the host LLVM pass SYCLRewriteSyrkRange which shrinks the launch to ceil(N/8) x ceil(N/8)";
  let description = [{
    Device-side half of the launch-grid-shrinking SYRK optimization. The
    source-level polybench/orise/syrk_opt.cpp win is an 8x8 register
    micro-kernel per work-item plus a host-side launch-grid shrink
    range<2>(N,N) -> range<2>(ceil(N/8), ceil(N/8)). This pass rewrites the
    device body so each work-item computes the 8x8 regC tile whose origin is
    its own item id (bi = item.get_id(0)*8, bj = item.get_id(1)*8): seed
    C*beta (one global read per cell), k loop of 64 FMAs, single guarded
    store per cell. No LDS, no barrier, no cooperative loading, no
    reqd_work_group_size (LocalSize stays auto) -- a faithful port of
    syrk_opt's algorithm. The launch-grid shrink itself is host-side and is
    performed by the paired LLVM pass SYCLRewriteSyrkRange
    (-fsycl-rewrite-syrk-range); a fail-safe guard (bi < M && bj < M) keeps
    the device body correct even if the host pass does not fire (launch still
    N x N: every tile is still computed exactly once, merely not faster).
    TWO rewrite sites are needed because the cgeist frontend emits the Syr2k2
    body twice: (1) the top-level lambda body func.func, called per
    enumerated user item by the __pf_kernel_wrapper path (the only path that
    runs when range rounding fires), and (2) the DIRECT kernel
    gpu.func @_ZTS6Syr2k2 whose body is born inlined from the frontend (the
    path that runs when no rounding is needed -- the common case for the
    shrunk launch). The 8x8 seed/fma/store bodies are fully unrolled with
    constant regC indices so that SROA promotes the 64 accumulator cells to
    scalar registers (AMDGPU promote-alloca rejects arrays > 16 elements, so
    a loop-nest form leaves the accumulators in scratch memory).
    Gated on (1) --sycl-syrk-register-tile (off) and (2) llvm.target_triple
    == amdgcn-amd-amdhsa-syclmlir.
  }];
  let constructor = "mlir::sycl::createSyrkRegisterTilePass()";
  let dependentDialects = ["arith::ArithDialect", "memref::MemRefDialect",
                            "affine::AffineDialect", "gpu::GPUDialect",
                            "scf::SCFDialect", "math::MathDialect",
                            "func::FuncDialect"];
  let statistics = [
    Statistic<"NumDetected", "num-detected",
              "Number of Syr2k2 bodies detected">,
    Statistic<"NumRewritten", "num-rewritten",
              "Number of Syr2k2 bodies rewritten to register-tile form">,
  ];
}

def Syr2kRegisterTilePass
    : Pass<"sycl-syr2k-register-tile", "::mlir::ModuleOp"> {
  let summary = "Rewrite the polybench Syr2k1 kernel so each work-item computes the 8x8 register tile at (item_id*8, item_id*8) (no LDS/barrier/reqd_work_group_size); pair with the host LLVM pass SYCLRewriteSyr2kRange which shrinks the launch to ceil(N/8) x ceil(N/8)";
  let description = [{
    Device-side half of the launch-grid-shrinking SYR2K optimization (a
    per-benchmark sibling of sycl-syrk-register-tile, distinguished purely by
    the kernel anchor -- Syr2k1 here vs Syr2k2 there -- so the two passes
    never cross-fire). The source-level polybench/orise/syr2k_opt.cpp win is
    an 8x8 register micro-kernel per work-item plus a host-side launch-grid
    shrink range<2>(N,N) -> range<2>(ceil(N/8), ceil(N/8)). This pass
    rewrites the device body so each work-item computes the 8x8 regC tile
    whose origin is its own item id (bi = item.get_id(0)*8, bj =
    item.get_id(1)*8): seed C*beta (one global read per cell), k loop loading
    16 values per iteration (rows bi+a of A and B, columns bj+b of A and B)
    feeding 2 FMAs per cell (the A*B^T term and the B*A^T term), single
    guarded store per cell. No LDS, no barrier, no cooperative loading, no
    reqd_work_group_size (LocalSize stays auto) -- a faithful port of
    syr2k_opt's algorithm. The captured struct has FOUR members (i64 M_,
    C read_write, A read, B read). The launch-grid shrink itself is
    host-side and is performed by the paired LLVM pass SYCLRewriteSyr2kRange
    (-fsycl-rewrite-syr2k-range); a fail-safe guard (bi < M && bj < M) keeps
    the device body correct even if the host pass does not fire (launch
    still N x N: every tile is still computed exactly once, merely not
    faster). TWO rewrite sites are needed because the cgeist frontend emits
    the Syr2k1 body twice: (1) the top-level lambda body func.func, called
    per enumerated user item by the __pf_kernel_wrapper path (the only path
    that runs when range rounding fires), and (2) the DIRECT kernel
    gpu.func @_ZTS6Syr2k1 whose body is born inlined from the frontend (the
    path that runs when no rounding is needed -- the common case for the
    shrunk launch). The 8x8 seed/fma/store bodies are fully unrolled with
    constant regC indices so that SROA promotes the 64 accumulator cells to
    scalar registers (AMDGPU promote-alloca rejects arrays > 16 elements, so
    a loop-nest form leaves the accumulators in scratch memory).
    Gated on (1) --sycl-syr2k-register-tile (off) and (2)
    llvm.target_triple == amdgcn-amd-amdhsa-syclmlir.
  }];
  let constructor = "mlir::sycl::createSyr2kRegisterTilePass()";
  let dependentDialects = ["arith::ArithDialect", "memref::MemRefDialect",
                            "affine::AffineDialect", "gpu::GPUDialect",
                            "scf::SCFDialect", "math::MathDialect",
                            "func::FuncDialect"];
  let statistics = [
    Statistic<"NumDetected", "num-detected",
              "Number of Syr2k1 bodies detected">,
    Statistic<"NumRewritten", "num-rewritten",
              "Number of Syr2k1 bodies rewritten to register-tile form">,
  ];
}

def TwoMmLocalTilePass
    : Pass<"sycl-2mm-local-tile", "::mlir::ModuleOp"> {
  let summary = "Rewrite BOTH polybench 2mm kernels (Polybench_2mm_1 C += A*B seeded from C, Polybench_2mm_2 E = C*D seeded from zero) to the 2mm_opt form: 16x16 LDS tiles cooperatively loaded by 16x16 work-groups (rocdl.barrier between stages), one output cell and a scalar accumulator per work-item; pair with the host LLVM pass SYCLRewrite2mmRange which pads BOTH launches to ceil16(N) x ceil16(N) and the runtime env SYCL_FORCE_LOCAL_SIZE=16,16 which fixes the work-group shape";
  let description = [{
    Device-side half of the 2MM optimization, transcribing the LDS-tiled
    micro-kernel of the source-level polybench/orise/2mm_opt.cpp (4.6x over
    the baseline on gfx906; the earlier 8x8 register-tile form of this pass
    measured 2.2x SLOWER than baseline and was replaced). ONE instance of
    this pass rewrites BOTH 2mm kernels -- the benchmark launches two
    back-to-back GEMMs sharing the shape out[i,j] = seed(i,j) +
    sum_k in1[i,k]*in2[k,j] and the captured-struct layout (i64 size_, OUT
    accessor, IN1 read, IN2 read); they differ only in the seed (kernel 1
    accumulates into the existing read_write C, kernel 2 writes
    discard_write E from zero). Tile shape TS = TK = 16, exactly 2mm_opt's:
    each 16x16 work-group (256 work-items) cooperatively stages a 16x16
    tile of A (rows) and B (columns) per k-chunk -- ONE global load of each
    per work-item, coalesced -- then every work-item runs a 16-FMA
    micro-kernel on the staged tiles into a SCALAR accumulator (an scf.for
    iter_arg -- a register; no accumulator memref anywhere) and stores its
    single output cell (cell = item id). The two LDS tiles are private
    module-scope memref.globals with the SYCL local address space (AS 3),
    created in the enclosing gpu.module (same construction as
    LoopInternalization's WGLocalMem); phase-3 lowers them to internal
    addrspace(3) LLVM globals which the AMDGPU backend allocates into
    static LDS (group_segment_fixed_size) -- no kernel-signature or
    host-side plumbing. Barriers are rocdl.barrier ops (the only barrier op
    that survives this pipeline: gpu.barrier is explicitly illegal in
    phase 3 and spirv.ControlBarrier has no ROCDL lowering); the ROCDL
    dialect is legal in the phase-3 target and its LLVMIR translation
    produces fence(workgroup) + llvm.amdgcn.s.barrier. The launch must
    provide 16x16 work-groups: the paired HOST LLVM pass
    SYCLRewrite2mmRange (-fsycl-rewrite-2mm-range) rewrites BOTH range<2>
    launches to (ceil16(N), ceil16(N)) -- the FULL padded grid, one
    work-item per output CELL -- and the SYCL runtime env var
    SYCL_FORCE_LOCAL_SIZE=16,16 (this tree) supplies the explicit local
    size (neither auto local -- UR's simpleGuessLocalWorkSize sizes only
    dim 0 -- nor reqd_work_group_size -- UR HIP stubs
    COMPILE_WORK_GROUP_SIZE to {0,0,0} and the runtime rejects 3-element
    reqd against a 2-D launch -- can provide it). With local (16,16)
    guaranteed, all of 2mm_opt's index math reduces to arithmetic on the
    item id: ty = irow % 16, tx = jcol % 16. Edge handling for N % 16 != 0:
    the padded launch creates up to 15 out-of-range rows/cols of
    work-items; their loads are clamped (min(idx, M-1)) and their final
    store is predicated (irow < M && jcol < M). TWO rewrite sites are
    needed per kernel because the cgeist frontend emits each body twice:
    (1) the top-level lambda body func.func, called per enumerated user
    item by the __pf_kernel_wrapper path (only runs when range rounding
    fires, which ceil16 values never trigger -- ceil16(N) is a multiple of
    the rounding MinFactor 16), and (2) the DIRECT kernels
    gpu.func @_ZTS15Polybench_2mm_{1,2} whose bodies are born inlined from
    the frontend (the path that always runs). Gated on (1)
    --sycl-2mm-local-tile (off) and (2) llvm.target_triple ==
    amdgcn-amd-amdhsa-syclmlir.
  }];
  let constructor = "mlir::sycl::createTwoMmLocalTilePass()";
  let dependentDialects = ["arith::ArithDialect", "memref::MemRefDialect",
                            "affine::AffineDialect", "gpu::GPUDialect",
                            "scf::SCFDialect", "math::MathDialect",
                            "func::FuncDialect", "ROCDL::ROCDLDialect"];
  let statistics = [
    Statistic<"NumDetected", "num-detected",
              "Number of 2mm bodies detected">,
    Statistic<"NumRewritten", "num-rewritten",
              "Number of 2mm bodies rewritten to local-tile form">,
  ];
}

#endif // MLIR_DIALECT_SYCL_TRANSFORMS_PASSES
