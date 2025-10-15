// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SYCL/Transforms/Passes.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SYCL/Analysis/SYCLAccessorAnalysis.h"
#include "mlir/Dialect/SYCL/Analysis/SYCLIDAndRangeAnalysis.h"
#include "mlir/Dialect/SYCL/Analysis/SYCLNDRangeAnalysis.h"
#include "mlir/Dialect/SYCL/IR/SYCLOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Support/LLVM.h"

#include <llvm/ADT/TypeSwitch.h>
#include <llvm/ADT/identity.h>
#include <llvm/Support/Debug.h>

#include <numeric>

namespace mlir {
namespace sycl {
#define GEN_PASS_DEF_TOYPASS
#include "mlir/Dialect/SYCL/Transforms/Passes.h.inc"
} // namespace sycl
} // namespace mlir

#define DEBUG_TYPE "sycl-toy-pass"

using namespace mlir;
using namespace mlir::sycl;

namespace {

class ToyPass : public mlir::sycl::impl::ToyPassBase<ToyPass> {
public:
  void runOnOperation() final;
};

} // namespace

void ToyPass::runOnOperation() {
  getOperation()->walk([&](gpu::GPUFuncOp op) {
    if (static_cast<SymbolOpInterface>(op).getName() ==
        "_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_EUlNS0_2idILi1EEEE_") {
      llvm::dbgs() << "Performing toy pass on gpu function '"
                   << static_cast<SymbolOpInterface>(op).getName() << "'\n";
    }
  });
  getOperation()->walk([&](func::FuncOp op) {
    if (static_cast<SymbolOpInterface>(op).getName() ==
        "_ZZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_ENKUlNS0_2idILi1EEEE_"
        "clES5_") {
      llvm::dbgs() << "Performing toy pass on func function '"
                   << static_cast<SymbolOpInterface>(op).getName() << "'\n";
    }
  });
}

std::unique_ptr<Pass> sycl::createToyPass() {
  return std::make_unique<ToyPass>();
}
