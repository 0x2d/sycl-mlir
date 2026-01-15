// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SYCL/Transforms/Passes.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SYCL/Analysis/SYCLAccessorAnalysis.h"
#include "mlir/Dialect/SYCL/Analysis/SYCLIDAndRangeAnalysis.h"
#include "mlir/Dialect/SYCL/Analysis/SYCLNDRangeAnalysis.h"
#include "mlir/Dialect/SYCL/IR/SYCLOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Support/LLVM.h"

#include <llvm/ADT/identity.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/TypeSwitch.h>
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
  gpu::GPUFuncOp Fan1, Fan2;
  ModuleOp module = getOperation();

  module.walk([&](gpu::GPUFuncOp op) {
    if (op.getName() ==
        "_ZTSZZ4Fan1RN4sycl3_V15queueERNS0_6bufferIfLi1ENS0_6detail17aligned_allocatorIfEEvEES8_S8_iiENKUlRNS0_7handlerEE_clESA_EUlNS0_7nd_itemILi1EEEE_") {
      Fan1 = op;
    }    
    if (op.getName() ==
        "_ZTSZZ4Fan2RN4sycl3_V15queueERNS0_6bufferIfLi1ENS0_6detail17aligned_allocatorIfEEvEES8_S8_iiENKUlRNS0_7handlerEE_clESA_EUlNS0_7nd_itemILi2EEEE_") {
      Fan2 = op;
    }
  });
  
  if (!Fan1 || !Fan2) {
    llvm::dbgs() << "No candidate functions.\n";
    return;
  } else {
    llvm::dbgs() << "Performing kernel fusion on "
                << Fan1.getName()  << " and " << Fan2.getName() << "\n";    
  }

  IRRewriter rewriter(&getContext());
  llvm::StringRef fan1CalleeName, fan2CalleeName;
  for (auto fan : {Fan1, Fan2}) {
    func::CallOp fanCall;
    fan.walk([&](scf::IfOp ifOp) {
      Block &elseBlock = ifOp.getElseRegion().front();
      elseBlock.walk([&](func::CallOp callOp) {
        callOp->moveBefore(ifOp);
        fanCall = callOp;
      });
      rewriter.eraseOp(ifOp);
    });
    if (fan == Fan1) {
      fan1CalleeName = fanCall.getCallee();
    } else if (fan == Fan2) {
      fan2CalleeName = fanCall.getCallee();
    }
    auto operands = fanCall.getArgOperands();
    llvm::SmallVector<Value> fanCallParams(operands.begin(), operands.end());
    Operation *bottomOp = &(fan.front().front());
    for (Value arg : fanCallParams) {
      Operation *definingOp = arg.getDefiningOp();
      if (bottomOp->isBeforeInBlock(definingOp)) {
        bottomOp = definingOp;
      }
    }
    auto startIt = bottomOp->getIterator();
    auto endIt = --(fanCall->getIterator());
    while (startIt != endIt) {
      auto oldIt = endIt;
      endIt--;
      oldIt->erase();
    }
  }

  func::FuncOp fan1CalleeOp, fan2CalleeOp;
  module.walk([&](func::FuncOp op) {
    if (op.getName() == fan1CalleeName) {
      fan1CalleeOp = op;
    }    
    if (op.getName() == fan2CalleeName) {
      fan2CalleeOp = op;
    }
  });
  IRMapping mapper;
  unsigned int fan1CalleeNumArguments = fan1CalleeOp.getNumArguments();
  unsigned int fan2CalleeNumArguments = fan2CalleeOp.getNumArguments();
  mapper.map(fan2CalleeOp.getArgument(0), fan1CalleeOp.getArgument(0));
  mapper.map(fan2CalleeOp.getArgument(1), fan1CalleeOp.getArgument(1));
  mapper.map(fan2CalleeOp.getArgument(2), fan1CalleeOp.getArgument(3));
  mapper.map(fan2CalleeOp.getArgument(3), fan1CalleeOp.getArgument(2));
  mapper.map(fan2CalleeOp.getArgument(fan2CalleeNumArguments-1), 
             fan1CalleeOp.getArgument(fan1CalleeNumArguments-1));
  Block &f1Entry = fan1CalleeOp.front();
  Block &f2Entry = fan2CalleeOp.front();
  rewriter.setInsertionPoint(f1Entry.getTerminator()); 
  for (auto &op : f2Entry.without_terminator()) {
    rewriter.clone(op, mapper);
  }

  Fan2.eraseBody();
  Block *entryBlock = Fan2.addEntryBlock();
  rewriter.setInsertionPointToStart(entryBlock);
  rewriter.create<gpu::ReturnOp>(rewriter.getUnknownLoc());

  sycl::SYCLNDItemGetGlobalIDOp gIdyop;
  fan1CalleeOp.walk([&](sycl::SYCLNDItemGetGlobalIDOp op) {
    auto indexValue = op.getIndex();
    auto indexOp = llvm::dyn_cast<arith::ConstantOp>(indexValue.getDefiningOp());
    auto indexAttr = indexOp.getValue().dyn_cast<mlir::IntegerAttr>();
    if (indexAttr.getInt() == 1) {
      gIdyop = op;
      return;
    }
  });
  rewriter.setInsertionPoint(gIdyop); 
  Type typei32 = rewriter.getI32Type();
  TypedAttr value0i32 = rewriter.getI32IntegerAttr(0);
  auto c0i32 = rewriter.create<arith::ConstantOp>(rewriter.getUnknownLoc(), typei32, value0i32);
  auto gIdy = gIdyop.getRes();
  if (gIdy.hasOneUse()) {
    auto user = *(gIdy.getUsers().begin());
    user->replaceAllUsesWith(c0i32);
    user->erase();
    gIdyop->erase();
  } else {
    gIdy.dump();
    llvm::dbgs() << "Has more then 1 user.\n";
    return;
  }

  rewriter.setInsertionPoint(*(++(c0i32->getUsers().begin())));
  TypedAttr value1i32 = rewriter.getI32IntegerAttr(1);
  auto c1i32 = rewriter.create<arith::ConstantOp>(rewriter.getUnknownLoc(), typei32, value1i32);
  Value lb, ub;
  if (auto op = llvm::dyn_cast<arith::CmpIOp>(*(c0i32->getUsers().begin()))) {
    if (op.getPredicate() == mlir::arith::CmpIPredicate::slt) {
      lb = op.getLhs();
      ub = op.getRhs();
    }
  } else {
    llvm::dbgs() << "Not CmpIOp.\n";
    return;
  }
  auto forOp = rewriter.create<scf::ForOp>(rewriter.getUnknownLoc(), lb, ub, c1i32);
  Value iv = forOp.getInductionVar();
  Block *loopBody = forOp.getBody();
  Block *parentBlock = forOp->getBlock();
  auto startIt = ++(Block::iterator(forOp));
  auto endIt = Block::iterator(parentBlock->getTerminator());
  loopBody->getOperations().splice(
      loopBody->begin(), 
      parentBlock->getOperations(), 
      startIt, endIt
  );
  Value oldValue = c0i32.getResult(); 
  for (Operation &op : loopBody->getOperations()) {
      op.replaceUsesOfWith(oldValue, iv);
  }
  llvm::dbgs() << "ToyPass complete.\n";
}

std::unique_ptr<Pass> sycl::createToyPass() {
  return std::make_unique<ToyPass>();
}
