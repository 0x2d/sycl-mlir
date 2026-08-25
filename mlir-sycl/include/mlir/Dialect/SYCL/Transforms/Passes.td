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

#endif // MLIR_DIALECT_SYCL_TRANSFORMS_PASSES
