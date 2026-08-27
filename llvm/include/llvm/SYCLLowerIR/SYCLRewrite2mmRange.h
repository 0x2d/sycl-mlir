//===- SYCLRewrite2mmRange.h - 2mm launch-range rewrite --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Host-side half of the 2MM LDS-tile optimization (the device half is the
// MLIR pass `sycl-2mm-local-tile`, the runtime half is the env var
// SYCL_FORCE_LOCAL_SIZE). The source-level `2mm_opt.cpp` win over the
// baseline `2mm.cpp` is a 16x16 local-memory-tiled micro-kernel driven by
// an nd_range launch with 16x16 work-groups over the FULL padded grid
// `range<2>(ceil16(N), ceil16(N))`. The launch size lives in host code, so
// the padding must happen here, on the host x86-64 module.
//
// 2mm launches TWO GEMM kernels back-to-back (C += A*B then E = C*D), and
// BOTH launches must be padded. Unlike SYCLRewriteSyrkRange (one launch,
// one kernel-name anchor), this pass identifies each launch site
// per-FUNCTION: at PipelineStartEP (before AlwaysInliner/SROA) the handler
// code is still in its own `parallel_for_lambda_impl` instantiation, and
// each instantiation's mangled name embeds its kernel tag class:
//
//   @_ZN4sycl3_V17handler24parallel_for_lambda_implI15Polybench_2mm_1ZZN13Polybench_2mm...
//   @_ZN4sycl3_V17handler24parallel_for_lambda_implI15Polybench_2mm_2ZZN13Polybench_2mm...
//
// (verified from a -O3 -Xclang -disable-llvm-passes -emit-llvm dump of the
// 2mm.cpp host TU). Each instantiation contains its own UserRange alloca,
// the two `store i64` of the by-value range arguments, a memcpy into a
// temporary (`%agg.tmp11`) and the `getRoundedRangeILi2EE` call whose two
// i64 arguments load that temporary. Rewriting the two stores into the
// UserRange alloca to `ceil(V/16)*16` makes the module behave exactly as if
// the source had said `range<2>(ceil16(N), ceil16(N))`: getRoundedRange
// (a no-op on multiples of 16 -- the range-rounding wrapper never fires),
// checkValueRange, and MNDRDesc.set -> GlobalSize all read that single
// alloca. The padded grid gives one work-item per output CELL (the device
// pass's mapping), and its divisibility by 16 satisfies the UR launch
// validator for the explicit local size (16,16) injected from
// SYCL_FORCE_LOCAL_SIZE.
//
// The module-level gate is a private string constant equal to either
// kernel's mangled name (`_ZTS15Polybench_2mm_1` / `_ZTS15Polybench_2mm_2`,
// from `__builtin_sycl_unique_stable_name`) somewhere in the module. If
// neither matches, the pass is a safe no-op.
//
// Limitation: no-ops if range rounding was compiled out
// (`-fsycl-disable-range-rounding`, also auto-added at -O0) -- the
// `getRoundedRange` anchor is then absent. The benchmark builds at -O3.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYCLLOWERIR_SYCLREWRITE2MMRANGE_H
#define LLVM_SYCLLOWERIR_SYCLREWRITE2MMRANGE_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class SYCLRewrite2mmRangePass : public PassInfoMixin<SYCLRewrite2mmRangePass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);
  static StringRef getPassName() {
    return "Rewrite SYCL 2mm launch ranges to ceil16(N) per dim";
  }
};

} // end namespace llvm

#endif // LLVM_SYCLLOWERIR_SYCLREWRITE2MMRANGE_H
