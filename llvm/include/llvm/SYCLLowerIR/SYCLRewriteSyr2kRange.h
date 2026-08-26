//===- SYCLRewriteSyr2kRange.h - SYR2K launch-range rewrite -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Host-side half of the launch-grid-shrinking SYR2K optimization (the device
// half is the MLIR pass `sycl-syr2k-register-tile`). The source-level
// `syr2k_opt.cpp` win over the baseline `syr2k.cpp` is an 8x8 register
// micro-kernel per work-item PLUS a launch-grid shrink
// `range<2>(N,N)` -> `range<2>(ceil(N/8), ceil(N/8))`. The launch size lives
// in host code, so the shrink must happen here, on the host x86-64 module.
//
// This is a per-benchmark sibling of SYCLRewriteSyrkRange (which handles the
// syrk benchmark's Syr2k2 kernel); the two are distinguished purely by the
// kernel-name anchor and never cross-fire.
//
// The pass runs at PipelineStartEP (before AlwaysInliner/SROA), where the
// `__SYCL_ALWAYS_INLINE` handler code is still in its own
// `parallel_for_lambda_impl` function: the by-value range arguments are
// stored into ONE alloca (`%UserRange`), and every downstream consumer -- the
// `getRoundedRange` call (directly or via a memcpy'd temporary), the
// range-rounding wrapper's `UserRange` member, `checkValueRange` (by-ref),
// and `MNDRDesc.set` -> `GlobalSize` -- reads that single alloca. Rewriting
// the two stores into it to `ceil(V/8)` is therefore exactly
// source-equivalent to writing `range<2>(ceil(N/8), ceil(N/8))` in the
// source: if range rounding fires on the new range, the wrapper enumerates
// tile indices correctly; if not, the direct kernel launches with the
// shrunk GlobalSize.
//
// The launch site is identified by (1) a call/invoke of
// `getRoundedRangeILi2EE` (the 2-D parallel_for range entry; syr2k.cpp has
// exactly one) whose two i64 arguments are loads from a common alloca
// (tracing a filling memcpy back to the origin range object when the loads
// read a temporary copy), and (2) the kernel-name anchor: a private string
// constant `_ZTS6Syr2k1` (from `__builtin_sycl_unique_stable_name(Syr2k1)`)
// somewhere in the module. If either does not match, the pass is a safe
// no-op.
//
// Limitation: no-ops if range rounding was compiled out
// (`-fsycl-disable-range-rounding`, also auto-added at -O0) -- the
// `getRoundedRange` anchor is then absent. The benchmark builds at -O3.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYCLLOWERIR_SYCLREWRITESYR2KRANGE_H
#define LLVM_SYCLLOWERIR_SYCLREWRITESYR2KRANGE_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class SYCLRewriteSyr2kRangePass
    : public PassInfoMixin<SYCLRewriteSyr2kRangePass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);
  static StringRef getPassName() {
    return "Rewrite SYCL Syr2k1 launch range to ceil(N/8) per dim";
  }
};

} // end namespace llvm

#endif // LLVM_SYCLLOWERIR_SYCLREWRITESYR2KRANGE_H
