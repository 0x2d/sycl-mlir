// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SYCL_ANALYSIS_KERNELANALYSIS_H
#define MLIR_DIALECT_SYCL_ANALYSIS_KERNELANALYSIS_H

#include "mlir/IR/Operation.h"

namespace mlir {
namespace sycl {

/// Analyze the number of SYCL device kernels.
class KernelAnalysis {
public:
  KernelAnalysis(Operation *op) {
    KernelAnalysisImpl(op);
  }
  int getNumKernels();
private:
  int numKernels;
  void KernelAnalysisImpl(Operation *op);
};

} // namespace sycl
} // namespace mlir

#endif // MLIR_DIALECT_SYCL_ANALYSIS_KERNELANALYSIS_H
