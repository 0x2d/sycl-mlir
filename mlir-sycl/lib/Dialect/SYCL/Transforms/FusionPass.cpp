// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SYCL/Transforms/Passes.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SYCL/Analysis/KernelAnalysis.h"
#include "mlir/Dialect/SYCL/Analysis/SYCLAccessorAnalysis.h"
#include "mlir/Dialect/SYCL/Analysis/SYCLIDAndRangeAnalysis.h"
#include "mlir/Dialect/SYCL/Analysis/SYCLNDRangeAnalysis.h"
#include "mlir/Dialect/SYCL/IR/SYCLOps.h"
#include "mlir/Dialect/SYCL/Utils/Utils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/ADT/identity.h>
#include <llvm/Demangle/Demangle.h>
#include <llvm/Support/Debug.h>

#include <numeric>

namespace mlir {
namespace sycl {
#define GEN_PASS_DEF_FUSIONPASS
#include "mlir/Dialect/SYCL/Transforms/Passes.h.inc"
} // namespace sycl
} // namespace mlir

#define DEBUG_TYPE "sycl-fusion-pass"

using namespace mlir;
using namespace mlir::sycl;

namespace {

enum class ArgRole { I32Scalar, Accessor, NDItem, Other };

static ArgRole classifyArg(Type t) {
  if (t.isInteger(32))
    return ArgRole::I32Scalar;
  if (auto m = dyn_cast<MemRefType>(t)) {
    Type e = m.getElementType();
    if (e.isInteger(32))
      return ArgRole::I32Scalar;
    if (isa<sycl::AccessorType, sycl::LocalAccessorType>(e))
      return ArgRole::Accessor;
    if (isa<sycl::NdItemType>(e))
      return ArgRole::NDItem;
  }
  if (isa<sycl::NdItemType>(t))
    return ArgRole::NDItem;
  return ArgRole::Other;
}

class FusionPass : public mlir::sycl::impl::FusionPassBase<FusionPass> {
public:
  void runOnOperation() final;
};

} // namespace

void FusionPass::runOnOperation() {
  gpu::GPUFuncOp Fan1, Fan2;
  ModuleOp module = getOperation();
  // Find Fan1 and Fan2
  module.walk([&](gpu::GPUFuncOp op) {
    StringRef mangledName = op.getName();
    std::string demangledName = llvm::demangle(mangledName.str());
    if (demangledName.find("Fan1") != std::string::npos) {
      Fan1 = op;
    } else if (demangledName.find("Fan2") != std::string::npos) {
      Fan2 = op;
    }
  });

  if (!Fan1 || !Fan2) {
    llvm::dbgs() << "Fusion Pass: No candidate functions.\n";
    return;
  } else {
    llvm::dbgs() << "Fusion Pass: Find candidate functions" << Fan1.getName()
                 << " and " << Fan2.getName() << "\n";
  }

  MLIRContext *context = &getContext();
  IRRewriter rewriter(context);

  // Find the kernel-body call inside `kernel`. After the inliner runs the
  // expected shape is:
  //
  //   scf.if %cond {
  //     func.call @body.specialized(...)
  //   } else {
  //     func.call @body(...)
  //   }
  //
  // We want the non-specialized call. Helper calls emitted earlier in the
  // kernel (e.g. sycl::detail::Builder::getElement) must NOT be selected;
  // they happen to be non-specialized too, and picking one causes the
  // downstream argument-mapping logic to index past the helper's argument
  // list. Restrict the search to a non-specialized func.call whose direct
  // parent is an scf.if whose sibling region holds a `.specialized` call.
  auto findKernelBodyCall = [](gpu::GPUFuncOp kernel) -> func::CallOp {
    func::CallOp result;
    kernel.walk([&](func::CallOp callOp) {
      if (callOp.getCallee().contains(".specialized"))
        return WalkResult::advance();
      auto ifOp = llvm::dyn_cast<scf::IfOp>(callOp->getParentOp());
      if (!ifOp)
        return WalkResult::advance();
      Region *callRegion = callOp->getParentRegion();
      Region *siblingRegion = (callRegion == &ifOp.getThenRegion())
                                  ? &ifOp.getElseRegion()
                                  : &ifOp.getThenRegion();
      if (siblingRegion->empty())
        return WalkResult::advance();
      bool hasSiblingSpecialized = false;
      siblingRegion->walk([&](func::CallOp other) {
        if (other.getCallee().contains(".specialized"))
          hasSiblingSpecialized = true;
      });
      if (!hasSiblingSpecialized)
        return WalkResult::advance();
      result = callOp;
      return WalkResult::interrupt();
    });
    return result;
  };

  func::CallOp fan1Call = findKernelBodyCall(Fan1);
  func::CallOp fan2Call = findKernelBodyCall(Fan2);
  if (!fan1Call || !fan2Call) {
    llvm::dbgs() << "Fusion Pass: Cannot find kernel functions.\n";
    return;
  }
  llvm::StringRef fan1CalleeName = fan1Call.getCallee();
  llvm::StringRef fan2CalleeName = fan2Call.getCallee();
  for (func::CallOp callOp : {fan1Call, fan2Call}) {
    auto ifOp = llvm::cast<scf::IfOp>(callOp->getParentOp());
    callOp->moveBefore(ifOp);
    rewriter.eraseOp(ifOp);
  }

  llvm::dbgs() << "Fusion Pass: Performing kernel fusion on "
               << fan1CalleeName << " and " << fan2CalleeName << "\n";

  func::FuncOp fan1CalleeOp, fan2CalleeOp;
  module.walk([&](func::FuncOp op) {
    if (op.getName() == fan1CalleeName) {
      fan1CalleeOp = op;
    }
    if (op.getName() == fan2CalleeName) {
      fan2CalleeOp = op;
    }
  });

  // Map kernel arguments and clone fan2 to fan1
  IRMapping mapper;
  Block &fan1CalleeBlock = fan1CalleeOp.front();
  size_t fan1IfCount =
      llvm::count_if(fan1CalleeBlock.getOperations(), [](mlir::Operation &op) {
        return llvm::isa<scf::IfOp>(op);
      });

  SmallVector<std::pair<unsigned, unsigned>> pairing; // (fan2Idx, fan1Idx)

  SmallVector<ArgRole> r1, r2;
  for (Type t : fan1CalleeOp.getArgumentTypes())
    r1.push_back(classifyArg(t));
  for (Type t : fan2CalleeOp.getArgumentTypes())
    r2.push_back(classifyArg(t));

  unsigned nextI32 = 0, nextAcc = 0;
  auto findFan1 = [&](ArgRole role, unsigned &cursor) -> int {
    for (unsigned i = cursor; i < r1.size(); ++i)
      if (r1[i] == role) {
        cursor = i + 1;
        return i;
      }
    return -1;
  };
  int fan1NDItemIdx = -1;
  for (unsigned i = 0; i < r1.size(); ++i)
    if (r1[i] == ArgRole::NDItem) {
      fan1NDItemIdx = i;
      break;
    }

  bool ok = true;
  for (unsigned j = 0; j < r2.size() && ok; ++j) {
    int i = -1;
    switch (r2[j]) {
    case ArgRole::I32Scalar:
      i = findFan1(ArgRole::I32Scalar, nextI32);
      break;
    case ArgRole::Accessor:
      i = findFan1(ArgRole::Accessor, nextAcc);
      break;
    case ArgRole::NDItem:
      i = fan1NDItemIdx;
      break;
    case ArgRole::Other:
      ok = false;
      break;
    }
    if (i < 0)
      ok = false;
    else
      pairing.push_back({j, (unsigned)i});
  }
  if (!ok) {
    llvm::dbgs() << "Fusion Pass: Cannot pair Fan1/Fan2 arguments by role.\n";
    return;
  }
  // Special pairing for Fan1 and Fan2
  pairing[2] = {(unsigned)2, (unsigned)3};
  pairing[3] = {(unsigned)3, (unsigned)2};

  // Capture Fan1's get_global_id(0) op BEFORE cloning, so we can later
  // redirect the Fan2-cloned get_global_id(0) consumers to it (and let the
  // dead nd_item plumbing fall away).
  sycl::SYCLNDItemGetGlobalIDOp fan1G0Op;
  fan1CalleeOp.walk([&](sycl::SYCLNDItemGetGlobalIDOp op) {
    auto indexOp =
        llvm::dyn_cast_or_null<arith::ConstantOp>(op.getIndex().getDefiningOp());
    if (!indexOp)
      return;
    auto indexAttr = indexOp.getValue().dyn_cast<mlir::IntegerAttr>();
    if (indexAttr && indexAttr.getInt() == 0) {
      fan1G0Op = op;
      return;
    }
  });

  rewriter.setInsertionPointToStart(&fan1CalleeBlock);
  SmallVector<UnrealizedConversionCastOp> insertedCasts;
  for (auto [j, i] : pairing) {
    Value f2 = fan2CalleeOp.getArgument(j);
    Value f1 = fan1CalleeOp.getArgument(i);
    if (f2.getType() == f1.getType()) {
      mapper.map(f2, f1);
    } else {
      auto cast = rewriter.create<UnrealizedConversionCastOp>(
          fan1CalleeOp.getLoc(), f2.getType(), f1);
      mapper.map(f2, cast.getResult(0));
      insertedCasts.push_back(cast);
    }
  }

  rewriter.setInsertionPoint(fan1CalleeBlock.getTerminator());
  llvm::DenseSet<Operation *> clonedOps;
  for (auto &op : fan2CalleeOp.front().without_terminator()) {
    Operation *cloned = rewriter.clone(op, mapper);
    clonedOps.insert(cloned);
  }

  // Redirect Fan2-cloned get_global_id(0) -> Fan1's get_global_id(0) result.
  // This eliminates the only "real" consumers of the cloned nd_item plumbing
  // (the unrealized_conversion_cast we inserted to bridge nd_item type
  // divergence between Fan1's 1D and Fan2's 2D nd_item).
  if (fan1G0Op) {
    SmallVector<sycl::SYCLNDItemGetGlobalIDOp> fan2G0Cloned;
    fan1CalleeOp.walk([&](sycl::SYCLNDItemGetGlobalIDOp op) {
      if (op == fan1G0Op)
        return;
      if (!clonedOps.contains(op))
        return;
      auto indexOp = llvm::dyn_cast_or_null<arith::ConstantOp>(
          op.getIndex().getDefiningOp());
      if (!indexOp)
        return;
      auto indexAttr = indexOp.getValue().dyn_cast<mlir::IntegerAttr>();
      if (indexAttr && indexAttr.getInt() == 0)
        fan2G0Cloned.push_back(op);
    });
    for (auto op : fan2G0Cloned) {
      op.getRes().replaceAllUsesWith(fan1G0Op.getRes());
      op->erase();
    }
  }

  // Erase fan2
  Block &fan2Block = Fan2.getBody().front();
  fan2Block.clear();
  rewriter.setInsertionPointToStart(&fan2Block);
  rewriter.create<gpu::ReturnOp>(rewriter.getUnknownLoc());

  // Create constant i32 0
  rewriter.setInsertionPointToStart(&fan1CalleeBlock);
  Type typei32 = rewriter.getI32Type();
  TypedAttr value0i32 = rewriter.getI32IntegerAttr(0);
  auto c0i32 = rewriter.create<arith::ConstantOp>(rewriter.getUnknownLoc(),
                                                  typei32, value0i32);
  // Create constant i32 1
  TypedAttr value1i32 = rewriter.getI32IntegerAttr(1);
  auto c1i32 = rewriter.create<arith::ConstantOp>(rewriter.getUnknownLoc(),
                                                  typei32, value1i32);
  // Replace get_global_id(1) with 0;
  llvm::SmallVector<sycl::SYCLNDItemGetGlobalIDOp> gIdyOps;
  fan1CalleeOp.walk([&](sycl::SYCLNDItemGetGlobalIDOp op) {
    auto indexOp =
        llvm::dyn_cast<arith::ConstantOp>(op.getIndex().getDefiningOp());
    auto indexAttr = indexOp.getValue().dyn_cast<mlir::IntegerAttr>();
    if (indexAttr.getInt() == 1) {
      gIdyOps.push_back(op);
    }
  });
  for (auto op : gIdyOps) {
    Value gIdyi64 = op.getRes();
    auto gIdyi32op = *gIdyi64.getUsers().begin();
    gIdyi32op->replaceAllUsesWith(c0i32);
    gIdyi32op->erase();
    if (op->use_empty()) {
      op->erase();
    } else {
      llvm::dbgs()
          << "Fusion Pass: SYCLNDItemGetGlobalIDOp has more than 1 user\n";
      return;
    }
  }

  // Add loop
  Value lb, ub;
  auto userIter = c0i32->getUsers().begin();
  if (auto op = llvm::dyn_cast<arith::CmpIOp>(*userIter)) {
    if (op.getPredicate() == mlir::arith::CmpIPredicate::slt) {
      lb = op.getLhs();
      ub = op.getRhs();
    } else if (op.getPredicate() == mlir::arith::CmpIPredicate::sgt) {
      ub = op.getLhs();
      lb = op.getRhs();
    } else {
      llvm::dbgs() << "Fusion Pass: TODO: CmpIPredicate.\n";
      return;
    }
    if (!lb.getType().isInteger(32)) {
      llvm::dbgs() << "Fusion Pass: TODO: Type cast.\n";
      return;
    }
  } else {
    llvm::dbgs() << "Fusion Pass: First user not CmpIOp.\n";
    return;
  }
  rewriter.setInsertionPoint(*(++userIter));
  auto forOp =
      rewriter.create<scf::ForOp>(rewriter.getUnknownLoc(), lb, ub, c1i32);
  Block *loopBody = forOp.getBody();
  Block *parentBlock = forOp->getBlock();
  auto startIt = ++(Block::iterator(forOp));
  auto endIt = Block::iterator(parentBlock->getTerminator());
  loopBody->getOperations().splice(
      loopBody->begin(), parentBlock->getOperations(), startIt, endIt);
  Value oldValue = c0i32.getResult();
  Value iv = forOp.getInductionVar();
  for (Operation &op : loopBody->getOperations()) {
    op.replaceUsesOfWith(oldValue, iv);
  }

  // Fuse ifOp
  llvm::SmallVector<scf::IfOp> ifOps;
  for (auto &op : fan1CalleeBlock) {
    if (auto ifOp = llvm::dyn_cast<scf::IfOp>(op)) {
      ifOps.push_back(ifOp);
    }
  }
  if (ifOps.size() == 2 && fan1IfCount == 1) {
    Operation *cond1Op = ifOps[0].getCondition().getDefiningOp();
    Operation *cond2Op = ifOps[1].getCondition().getDefiningOp();
    if (OperationEquivalence::isEquivalentTo(
            cond1Op, cond2Op, checkEquivalent, nullptr,
            OperationEquivalence::Flags::IgnoreLocations, nullptr)) {
      // Move preIfList operations
      Block *preIfList = ifOps[1]->getBlock();
      startIt = ++(Block::iterator(ifOps[0]));
      endIt = Block::iterator(ifOps[1]);
      preIfList->getOperations().splice(Block::iterator(ifOps[0]),
                                        preIfList->getOperations(), startIt,
                                        endIt);
      // Fuse ifOps
      Block &fan1IfOpBlock = ifOps[0].getThenRegion().front();
      Block &fan2IfOpBlock = ifOps[1].getThenRegion().front();
      startIt = Block::iterator(fan2IfOpBlock.front());
      endIt = Block::iterator(fan2IfOpBlock.getTerminator());
      fan1IfOpBlock.getOperations().splice(
          fan1IfOpBlock.getTerminator()->getIterator(),
          fan2IfOpBlock.getOperations(), startIt, endIt);
      rewriter.eraseOp(ifOps[1]);
    } else {
      llvm::dbgs() << "Fusion Pass: Cannot fuse ifOps\n";
    }
  }

  // Register Promotion
  RewritePatternSet patterns(context);
  populateRegisterPromotion(patterns, context);
  if (failed(applyPatternsAndFoldGreedily(fan1CalleeOp, std::move(patterns)))) {
    llvm::dbgs() << "Fusion Pass: Find RegisterPromotion pattern\n";
  } else {
    llvm::dbgs() << "Fusion Pass: Cannot find RegisterPromotion pattern\n";
  }

  AnalysisManager am = getAnalysisManager();
  KernelAnalysis &kernelAnalysis = am.getAnalysis<KernelAnalysis>();
  if (kernelAnalysis.getNumKernels() == 2) {
    llvm::dbgs() << "Fusion Pass: Two kernels\n";
  } else {
    llvm::dbgs() << "Fusion Pass: More than two kernels\n";
  }
  llvm::dbgs() << "Fusion Pass: Completed\n";
}

std::unique_ptr<Pass> sycl::createFusionPass() {
  return std::make_unique<FusionPass>();
}
