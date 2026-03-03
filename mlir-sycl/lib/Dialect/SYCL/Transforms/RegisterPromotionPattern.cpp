#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SYCL/IR/SYCLOps.h"
#include "mlir/Dialect/SYCL/Transforms/Passes.h"
#include "mlir/Dialect/SYCL/Utils/Utils.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::sycl;

namespace mlir::sycl {
struct RegisterPromotion
    : public OpRewritePattern<sycl::SYCLAccessorSubscriptOp> {

  RegisterPromotion(MLIRContext *context)
      : OpRewritePattern<sycl::SYCLAccessorSubscriptOp>(context,
                                                        /*benefit=*/1) {}

  LogicalResult matchAndRewrite(sycl::SYCLAccessorSubscriptOp loadOp,
                                PatternRewriter &rewriter) const override {
    // Information of current load op
    Value curAcc = loadOp.getAcc();
    Value curIndex = loadOp.getIndex();

    // Match if next op is affine.load
    if (!llvm::isa<affine::AffineLoadOp>(loadOp->getNextNode())) {
      return failure();
    }

    // Get load address
    Value loadOffset = getOffsetFromSubscriptOp(loadOp);
    // Get store op
    func::FuncOp funcOp = loadOp->getParentOfType<func::FuncOp>();
    llvm::SmallVector<Operation *> opsBeforeLoad;
    funcOp->walk([&](Operation *op) {
      if (op == loadOp) {
        return WalkResult::interrupt();
      } else {
        opsBeforeLoad.push_back(op);
        return WalkResult::advance();
      }
    });

    sycl::SYCLAccessorSubscriptOp storeOp;
    for (Operation* op : llvm::reverse(opsBeforeLoad)) {
      if (auto sOp = llvm::dyn_cast<sycl::SYCLAccessorSubscriptOp>(op)) {
        if (llvm::isa<affine::AffineStoreOp>(sOp->getNextNode())) {
          Value storeOffset = getOffsetFromSubscriptOp(sOp);
          if (curAcc == sOp.getAcc() || succeeded(checkEquivalent(loadOffset, storeOffset))) {
            storeOp = sOp;
            break;
          }
        }
      }
    }
    if (storeOp) {
      auto storeNextOp = llvm::dyn_cast<affine::AffineStoreOp>(storeOp->getNextNode());
      auto loadNextOp = llvm::dyn_cast<affine::AffineLoadOp>(loadOp->getNextNode());
      Value storeTmpValue = storeNextOp.getValue();
      Value loadTmpValue = loadNextOp.getValue();
      rewriter.replaceAllUsesWith(loadTmpValue, storeTmpValue);
    }
    return success();
  }
};

void populateRegisterPromotion(RewritePatternSet &patterns,
                               MLIRContext *context) {
  patterns.add<RegisterPromotion>(context);
}
} // namespace mlir::sycl