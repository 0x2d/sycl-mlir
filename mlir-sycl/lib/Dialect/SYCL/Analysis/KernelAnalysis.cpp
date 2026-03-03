// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SYCL/Analysis/KernelAnalysis.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SYCL/IR/SYCLDialect.h"
#include "mlir/Dialect/SYCL/IR/SYCLOps.h"
#include "llvm/IR/Attributes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "kernel-analysis"

using namespace mlir;

void sycl::KernelAnalysis::KernelAnalysisImpl(Operation *op) {
  numKernels = 0;
  op->walk([&](gpu::GPUFuncOp op) {
    numKernels++;
  });
}

int sycl::KernelAnalysis::getNumKernels() {
  return numKernels;
}