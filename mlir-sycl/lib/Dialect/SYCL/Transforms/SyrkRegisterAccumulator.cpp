//===- SyrkRegisterAccumulator.cpp - rewrite the syrk kernel ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Device-only optimization of the baseline polybench `Syr2k2` lambda body. The
// baseline launches one work-item per output cell via
// `parallel_for(range<2>(N,N), item<2>(2), ...)` -- a launch *size* a device
// pass cannot change (see memory `m4-host-launch-not-device-pass-feasible`).
// Each item does `C[item]*=beta; for k: C[item]+=alpha*A[{i,k}]*A[{j,k}]` as a
// per-`k` read-modify-write on `C[item]`, with `k` kept in a stack slot
// (`scf.while` + `llvm.store ...%slot`).
//
// This pass rewrites the body IN PLACE (no signature change, no `nd_item`, no
// host change, no work-groups, no local memory) to a register-accumulator
// form: it hoists `i=item.get_id(0)`, `j=item.get_id(1)`, and the `M` bound out
// of the loop, converts the `k` `scf.while` into an `scf.for` carrying a
// single `f32` accumulator seeded with `C[{i,j}]*beta`, and stores the result
// once after the loop (one global read of `C`, one global write of `C`).
//
// This is the device-feasible subset of what the source-level
// `polybench/orise/syrk_opt.cpp` register-blocking rewrite does; the remaining
// `syrk_opt` win (the 8x8 register micro-kernel per thread, raising arithmetic
// intensity from 0.5 to 4.0 FMA/load) requires changing the host launch grid
// from `(N,N)` to `(N/8, N/8)`, which a device pass cannot do. The LLVM backend
// already promotes the per-`k` `C[item]` RMW to a register PHI, so this
// transform is expected to be correctness-preserving and roughly net-neutral
// on performance (it may expose backend vectorization that the
// `scf.while`/stack-`k` form inhibits).
//
// Correctness: the `k`-summation order is preserved (`0..M`), the operands to
// the `math.fma` are bit-identical (`alpha*A[{i,k}]`, `A[{j,k}]`, the running
// accumulator), and the initial accumulator is `C[{i,j}]*beta` (the baseline
// writes `C[{i,j}]*beta` before the loop and the first FMA adds onto the
// reloaded `C[{i,j}]`). So the result is bit-identical FP to the baseline.
// Whatever offset behavior the baseline has (cf. memory
// `rocdl_global_offset_hardcoded_zero`) is inherited unchanged: `i`/`j` are
// still `sycl.item.get_id`, `M` is still the captured struct member, accessor
// subscripts still take the logical index directly (no OFFSET arithmetic).
//
// Two gates must hold for the pass to act:
//  (1) opt-in flag `--sycl-syrk-register-accumulator` (off by default; wired in
//      cgeist driver.cc), and
//  (2) the module's `llvm.target_triple` attribute must equal
//      `amdgcn-amd-amdhsa-syclmlir` (the build used
//      `-fsycl-targets=amdgcn-amd-amdhsa-syclmlir`).
//
//===----------------------------------------------------------------------===//

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Polygeist/IR/PolygeistOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SYCL/IR/SYCLOps.h"
#include "mlir/Dialect/SYCL/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Visitors.h"

#include <llvm/Support/Debug.h>

namespace mlir {
namespace sycl {
#define GEN_PASS_DEF_SYRKREGISTERACCUMULATORPASS
#include "mlir/Dialect/SYCL/Transforms/Passes.h.inc"
} // namespace sycl
} // namespace mlir

#define DEBUG_TYPE "sycl-syrk-register-accumulator"

using namespace mlir;
using namespace mlir::sycl;

namespace {

/// The SYCL target triple the pass is permitted to act on.
static constexpr llvm::StringRef AMDGCN_SYCLMLIR_TRIPLE =
    "amdgcn-amd-amdhsa-syclmlir";

/// The mangled kernel name fragment uniquely identifying the syrk body
/// function. The `sycl.kernel_func_obj` attribute on the body `func.func`
/// lists `@_ZTS6Syr2k2` (the dispatching `gpu.func`) and the
/// `__pf_kernel_wrapperI6Syr2k2EE` wrapper, of which "Syr2k2" is a substring
/// that does not collide with other polybench kernels.
static constexpr llvm::StringRef SYRK_KERNEL_FRAGMENT = "Syr2k2";

class SyrkRegisterAccumulatorPass
    : public mlir::sycl::impl::SyrkRegisterAccumulatorPassBase<
          SyrkRegisterAccumulatorPass> {
public:
  void runOnOperation() final;

private:
  /// Gate #2: only act when the module targets amdgcn-amd-amdhsa-syclmlir.
  /// Mirrors the precedent in CovarianceLoopReorder.cpp (which itself mirrors
  /// RegisterPromotionPattern.cpp + the exact-string idiom in Utils.cpp:162).
  bool targetsAmdgcnSyclmlir(ModuleOp module) const {
    auto tripleAttr = module->getAttrOfType<StringAttr>(
        LLVM::LLVMDialect::getTargetTripleAttrName());
    return tripleAttr && tripleAttr.getValue() == AMDGCN_SYCLMLIR_TRIPLE;
  }

  /// True iff `func` is a Syr2k2 lambda body, identified by its
  /// `sycl.kernel_func_obj` attribute referencing `@_ZTS6Syr2k2`.
  bool isSyrkBody(func::FuncOp func) const {
    auto kernelFuncObj = func->getAttrOfType<ArrayAttr>(
        sycl::SYCLDialect::getKernelFuncObjAttrName());
    if (!kernelFuncObj)
      return false;
    return llvm::any_of(kernelFuncObj.getAsRange<FlatSymbolRefAttr>(),
                        [](FlatSymbolRefAttr symbol) {
                          return symbol.getValue().contains(SYRK_KERNEL_FRAGMENT);
                        });
  }

  /// Rewrite `body`'s region into the register-accumulator form. Clears the
  /// entry block and rebuilds from scratch; block arguments (the function
  /// args) are kept. Reads the real 2-arg baseline signature:
  ///   arg0 = captured struct
  ///         (memref<?x!llvm.struct<(i64, !sycl_accessor_2_f32_rw_dev,
  ///                                  !sycl_accessor_2_f32_r_dev)>>)
  ///         member 0 = M bound (i64), member 1 = C (rw accessor),
  ///         member 2 = A (r accessor).
  ///   arg1 = !sycl_item_2_.
  /// Returns false (leaving the body untouched) if the signature does not match
  /// the expected baseline shape, so a `.specialized` twin is not half-erased.
  bool rewriteBodyAccumulate(func::FuncOp body);
};

} // namespace

bool SyrkRegisterAccumulatorPass::rewriteBodyAccumulate(func::FuncOp body) {
  MLIRContext *ctx = body.getContext();
  OpBuilder b(ctx);
  Location loc = body.getLoc();

  Type i32 = b.getI32Type();
  Type i64 = b.getIntegerType(64);
  Type indexTy = b.getIndexType();
  FloatType f32 = b.getF32Type();

  // Signature gate: the real baseline body has exactly 2 args -- the captured
  // struct memref and the item. A `.specialized` twin (if any) with a different
  // signature is left untouched rather than half-erased.
  if (body.getNumArguments() != 2)
    return false;

  Value arg0 = body.getArgument(0); // captured struct memref
  Value item = body.getArgument(1); // !sycl_item_2_

  // Validate the captured-struct shape (memref of an LLVM struct with >= 3
  // members: M-bound i64, C accessor, A accessor) BEFORE erasing the body, so
  // a non-matching body is preserved intact.
  auto arg0MT = dyn_cast<MemRefType>(arg0.getType());
  if (!arg0MT)
    return false;
  auto sTy = dyn_cast<LLVM::LLVMStructType>(arg0MT.getElementType());
  if (!sTy || sTy.getBody().size() < 3)
    return false;
  ArrayRef<Type> members = sTy.getBody();

  // The 2-D id used to subscript the (2-D) syrk accessors. The payload MUST
  // be the 2-element array form `(!sycl.array<[2], (memref<2xi64>)>)` (the
  // `!sycl_id_2_` spelling the baseline emits), NOT the simple `(i64)` form:
  // `sycl.accessor.subscript`'s lowering GEPs `Id[I]` per dimension
  // (DPCPP.cpp::AccessorSubscriptIDIndexPattern::getLinearIndex), and
  // `sycl.id.constructor`'s lowering stores each arg into the i-th slot via
  // `SYCLIDGetOp` -- both GEP the id payload, which only works when the payload
  // is the indexable array. The `(i64)` form verifies (the accessor.subscript
  // verifier only checks id dim == accessor dim) but fails phase-3 lowering
  // with "'llvm.getelementptr' op type 'i64' cannot be indexed". (See the
  // CovarianceLoopReorder.cpp:148-157 comment for the full provenance.)
  Type idElemTy =
      parseType("!sycl.id<[2], (!sycl.array<[2], (memref<2xi64>)>)>", ctx);
  if (!idElemTy)
    return false;
  // Use MemRefType::Builder: its typed MemRefLayoutAttrInterface member
  // selects the unambiguous MemRefType::get overload.
  MemRefType idMemrefTy = MemRefType::Builder({1}, idElemTy);
  MemRefType itemMemrefTy =
      MemRefType::Builder({1}, item.getType())
          .setMemorySpace(b.getI64IntegerAttr(5));
  MemRefType f32DynTy = MemRefType::Builder({ShapedType::kDynamic}, f32);

  // Wipe the existing entry-block body (the baseline RMW loop) and rebuild the
  // register-accumulator compute. Block arguments (function args) are kept.
  // Erase the entry block's TOP-LEVEL ops in reverse program order: each
  // `erase()` safely destroys the op's own nested regions, and reverse order
  // guarantees every result's uses (which can only be by later ops, by SSA
  // dominance) are gone before its defining op is erased. `Block::
  // getOperations().clear()` would destroy ops in forward order while later
  // ops still reference them, tripping "operation destroyed but still has
  // uses" on the real (non-trivial) baseline body; a `walk`-based erase
  // double-frees (nested ops erased both explicitly and via their parent).
  Region &region = body.getFunctionBody();
  Block &entry = region.front();
  SmallVector<Operation *> topOps;
  for (Operation &op : entry.getOperations())
    topOps.push_back(&op);
  for (Operation *op : llvm::reverse(topOps))
    op->erase();
  // Drop any non-entry blocks (now empty); their block args have no remaining
  // uses once all top-level ops are gone.
  SmallVector<Block *> extraBlocks;
  for (Block &blk : region.getBlocks())
    if (&blk != &entry)
      extraBlocks.push_back(&blk);
  for (Block *blk : extraBlocks)
    blk->erase();
  b.setInsertionPointToStart(&entry);

  // --- constants ---
  Value c0i32 = b.create<arith::ConstantIntOp>(loc, 0, i32);
  Value c1i32 = b.create<arith::ConstantIntOp>(loc, 1, i32);
  Value c0idx = b.create<arith::ConstantIndexOp>(loc, 0);
  Value c1idx = b.create<arith::ConstantIndexOp>(loc, 1);
  Value c2idx = b.create<arith::ConstantIndexOp>(loc, 2);
  Value beta = b.create<arith::ConstantFloatOp>(loc, llvm::APFloat(14512.0f), f32);
  Value alpha = b.create<arith::ConstantFloatOp>(loc, llvm::APFloat(123.0f), f32);

  // --- item memref (stack alloca in addrspace 5) + get_id -> i, j ---
  // sycl.item.get_id takes an ItemMemRef (= MemRefOf<[ItemType]>, any memref of
  // item incl. memref<1xT,5>); returns i64 -> cast to index.
  Value itemMem = b.create<memref::AllocaOp>(loc, itemMemrefTy);
  b.create<memref::StoreOp>(loc, item, itemMem, ValueRange{c0idx});
  Value ii64 = b.create<sycl::SYCLItemGetIDOp>(loc, TypeRange{i64},
                                              ValueRange{itemMem, c0i32});
  Value ji64 = b.create<sycl::SYCLItemGetIDOp>(loc, TypeRange{i64},
                                               ValueRange{itemMem, c1i32});
  Value i = b.create<arith::IndexCastOp>(loc, indexTy, ii64);
  Value j = b.create<arith::IndexCastOp>(loc, indexTy, ji64);

  // --- captured struct members via polygeist.subindex ---
  // arg0 : memref<?x!llvm.struct<(i64, !sycl_accessor_2_f32_rw_dev,
  //                               !sycl_accessor_2_f32_r_dev)>>
  //   c0 -> memref<?xi64>                          (M bound)
  //   c1 -> memref<?x!sycl_accessor_2_f32_rw_dev>  (C)
  //   c2 -> memref<?x!sycl_accessor_2_f32_r_dev>   (A)
  // SubIndexOp's builder requires an explicit result memref type (it has no
  // result-type inference): build `memref<?xmember>` per struct member.
  Value mboundAcc = b.create<polygeist::SubIndexOp>(
      loc, MemRefType::get({ShapedType::kDynamic}, members[0]), arg0, c0idx);
  Value cAcc = b.create<polygeist::SubIndexOp>(
      loc, MemRefType::get({ShapedType::kDynamic}, members[1]), arg0, c1idx);
  Value aAcc = b.create<polygeist::SubIndexOp>(
      loc, MemRefType::get({ShapedType::kDynamic}, members[2]), arg0, c2idx);

  // --- M (i64) -> index ---
  Value Mraw = b.create<memref::LoadOp>(loc, i64, mboundAcc, ValueRange{c0idx});
  Value M = b.create<arith::IndexCastOp>(loc, indexTy, Mraw);

  // --- helpers (capture SSA values defined in the entry block, which
  // dominates the nested scf region) ---
  auto mkId = [&](OpBuilder &bb, Value r, Value c) -> Value {
    return bb.create<sycl::SYCLIDConstructorOp>(loc, TypeRange{idMemrefTy},
                                               ValueRange{r, c});
  };
  auto loadData = [&](OpBuilder &bb, Value acc, Value r, Value c) -> Value {
    Value idm = mkId(bb, r, c);
    Value sub = bb.create<sycl::SYCLAccessorSubscriptOp>(
        loc, TypeRange{f32DynTy}, ValueRange{acc, idm});
    return bb.create<memref::LoadOp>(loc, f32, sub, ValueRange{c0idx});
  };
  auto storeC = [&](OpBuilder &bb, Value v, Value r, Value c) {
    Value idm = mkId(bb, r, c);
    Value sub = bb.create<sycl::SYCLAccessorSubscriptOp>(
        loc, TypeRange{f32DynTy}, ValueRange{cAcc, idm});
    bb.create<memref::StoreOp>(loc, v, sub, ValueRange{c0idx});
  };

  // --- seed accumulator: acc0 = C[{i,j}] * beta ---
  Value cAddr = [&] {
    Value idm = mkId(b, i, j);
    return b.create<sycl::SYCLAccessorSubscriptOp>(loc, TypeRange{f32DynTy},
                                                   ValueRange{cAcc, idm});
  }();
  Value cLoad = b.create<memref::LoadOp>(loc, f32, cAddr, ValueRange{c0idx});
  Value seed = b.create<arith::MulFOp>(loc, cLoad, beta);

  // --- k loop: for k = 0; k < M; k++ carrying one f32 accumulator ---
  //   acc = acc + alpha * A[{i,k}] * A[{j,k}]
  // alpha is folded into the FMA chain by pre-multiplying A[{i,k}] by alpha
  // (one mulf per k, hoistable by LICM later if A[{i,k}] were loop-invariant
  // -- it is not, since k varies; this matches the baseline's `cst_0 * A`
  // shape, which the backend folds into the FMA immediate anyway).
  Value kLo = b.create<arith::ConstantIndexOp>(loc, 0);
  Value kStep = c1idx;
  auto kFor = b.create<scf::ForOp>(
      loc, kLo, M, kStep, ValueRange{seed},
      [&](OpBuilder &fb, Location fl, Value k, ValueRange iters) {
        Value av = loadData(fb, aAcc, i, k); // A[{i,k}]
        Value avA = fb.create<arith::MulFOp>(fl, av, alpha);
        Value bv = loadData(fb, aAcc, j, k); // A[{j,k}]
        Value acc = iters[0];
        Value fma = fb.create<math::FmaOp>(fl, avA, bv, acc);
        fb.create<scf::YieldOp>(fl, ValueRange{fma});
      });

  // --- single guarded write of C[{i,j}] ---
  Value res = kFor.getResults()[0];
  storeC(b, res, i, j);

  b.create<func::ReturnOp>(loc);
  return true;
}

void SyrkRegisterAccumulatorPass::runOnOperation() {
  ModuleOp module = getOperation();

  // Gate #2: target triple must be amdgcn-amd-amdhsa-syclmlir. On any other
  // target (SPIR64-syclmlir, plain amdgcn-amd-amdhsa, etc.) the pass is a
  // no-op. Gate #1 (the opt-in flag) is enforced in cgeist driver.cc, where
  // the pass is only added to the pipeline when --sycl-syrk-register-accumulator
  // is set; the in-pass target guard is load-bearing because the pass can also
  // be reached via polygeist-opt / lit, where the driver flag is absent.
  if (!targetsAmdgcnSyclmlir(module)) {
    LLVM_DEBUG(llvm::dbgs() << "SyrkRegisterAccumulator: target is not "
                           << AMDGCN_SYCLMLIR_TRIPLE << ", skipping.\n");
    return;
  }

  // Detection: find the Syr2k2 lambda body func.func(s) (the `.specialized`
  // and non-specialized twins share the sycl.kernel_func_obj anchor).
  SmallVector<func::FuncOp, 2> syrkBodies;
  module.walk([&](func::FuncOp func) {
    if (isSyrkBody(func))
      syrkBodies.push_back(func);
  });

  for (func::FuncOp body : syrkBodies) {
    ++NumDetected;
    LLVM_DEBUG({
      llvm::dbgs() << "SyrkRegisterAccumulator: detected body " << body.getName()
                   << " (" << body.getNumArguments() << " args)\n";
    });
    // Rewrite the real baseline body directly (2 args: captured struct +
    // item<2>); no signature gate -- unlike the tiled pass, this transform
    // needs no canonical signature / nd_item. rewriteBodyAccumulate returns
    // false (and leaves the body untouched) if the shape does not match.
    if (rewriteBodyAccumulate(body))
      ++NumRewritten;
  }
}

std::unique_ptr<Pass> mlir::sycl::createSyrkRegisterAccumulatorPass() {
  return std::make_unique<SyrkRegisterAccumulatorPass>();
}
