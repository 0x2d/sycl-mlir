//===- SYCLToROCDL.cpp - SYCL to ROCDL Patterns ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns to convert SYCL dialect to ROCDL dialect
// (AMDGCN intrinsics).
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/SYCLToROCDL/SYCLToROCDL.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SYCL/IR/SYCLOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTSYCLTOROCDL
#include "mlir/Conversion/SYCLPasses.h.inc"
#undef GEN_PASS_DEF_CONVERTSYCLTOROCDL
} // namespace mlir

using namespace mlir;
using namespace mlir::sycl;

namespace {

/// Per-dimension AMDGCN intrinsic kind selector. The X/Y/Z entries are
/// instantiated by the templates below.
enum class RocdlGridKind {
  WorkItemId,    // rocdl.workitem.id.{x,y,z}
  WorkGroupId,   // rocdl.workgroup.id.{x,y,z}
  WorkGroupSize, // rocdl.workgroup.dim.{x,y,z}
  NumWorkGroups  // rocdl.grid.dim.{x,y,z}
};

template <typename OpTy> struct rocdl_kind_of;

template <> struct rocdl_kind_of<SYCLLocalIDOp> {
  static constexpr RocdlGridKind value = RocdlGridKind::WorkItemId;
};
template <> struct rocdl_kind_of<SYCLWorkGroupIDOp> {
  static constexpr RocdlGridKind value = RocdlGridKind::WorkGroupId;
};
template <> struct rocdl_kind_of<SYCLWorkGroupSizeOp> {
  static constexpr RocdlGridKind value = RocdlGridKind::WorkGroupSize;
};
template <> struct rocdl_kind_of<SYCLNumWorkGroupsOp> {
  static constexpr RocdlGridKind value = RocdlGridKind::NumWorkGroups;
};

/// Build the ROCDL intrinsic for the given grid query and dim. The result type
/// matches what the ROCDL→LLVM-IR translation produces:
///   * `workitem.id` and `workgroup.id` lower to LLVM intrinsics returning i32.
///   * `workgroup.dim` and `grid.dim` lower to `__ockl_get_local_size` /
///     `__ockl_get_num_groups`, both returning i64.
/// Producing the wrong MLIR type would still verify, but
/// `mlir::translateModuleToLLVMIR` later asserts on the type mismatch when an
/// arith op consumes the value.
static Value buildRocdlGridDim(OpBuilder &builder, Location loc,
                               RocdlGridKind kind, int64_t dim) {
  Type i32 = builder.getI32Type();
  Type i64 = builder.getI64Type();
  switch (kind) {
  case RocdlGridKind::WorkItemId:
    switch (dim) {
    case 0: return builder.create<ROCDL::ThreadIdXOp>(loc, i32);
    case 1: return builder.create<ROCDL::ThreadIdYOp>(loc, i32);
    case 2: return builder.create<ROCDL::ThreadIdZOp>(loc, i32);
    }
    break;
  case RocdlGridKind::WorkGroupId:
    switch (dim) {
    case 0: return builder.create<ROCDL::BlockIdXOp>(loc, i32);
    case 1: return builder.create<ROCDL::BlockIdYOp>(loc, i32);
    case 2: return builder.create<ROCDL::BlockIdZOp>(loc, i32);
    }
    break;
  case RocdlGridKind::WorkGroupSize:
    switch (dim) {
    case 0: return builder.create<ROCDL::BlockDimXOp>(loc, i64);
    case 1: return builder.create<ROCDL::BlockDimYOp>(loc, i64);
    case 2: return builder.create<ROCDL::BlockDimZOp>(loc, i64);
    }
    break;
  case RocdlGridKind::NumWorkGroups:
    switch (dim) {
    case 0: return builder.create<ROCDL::GridDimXOp>(loc, i64);
    case 1: return builder.create<ROCDL::GridDimYOp>(loc, i64);
    case 2: return builder.create<ROCDL::GridDimZOp>(loc, i64);
    }
    break;
  }
  llvm_unreachable("Invalid (kind, dim) pair");
}

/// Returns the result of creating an operation to get a reference to an
/// element of a sycl::id or sycl::range.
static Value createGetOp(OpBuilder &builder, Location loc, Type dimMtTy,
                         Value res, Value index) {
  return TypeSwitch<Type, Value>(
             cast<MemRefType>(res.getType()).getElementType())
      .Case<IDType, RangeType>([&](auto arg) {
        using ArgTy = decltype(arg);
        using OpTy = std::conditional_t<std::is_same_v<ArgTy, IDType>,
                                        SYCLIDGetOp, SYCLRangeGetOp>;
        return builder.create<OpTy>(loc, dimMtTy, res, index);
      });
}

// SYCL id/range with N dimensions stores values such that index 0 is the
// slowest-varying dimension. AMDGCN intrinsics (and SPIR-V GlobalInvocationId)
// expose dim 0 (X) as the fastest-varying. The mirror maps SYCL i to AMDGCN dim.
template <unsigned Dimensions>
static std::enable_if_t<(1 <= Dimensions && Dimensions < 4), int64_t>
mirrorIndexCalc(int64_t index);

template <> int64_t mirrorIndexCalc<1>(int64_t index) { return 0; }
template <> int64_t mirrorIndexCalc<2>(int64_t index) { return !index; }
template <> int64_t mirrorIndexCalc<3>(int64_t index) { return 2 - index; }

template <unsigned Dimensions>
static std::enable_if_t<(1 <= Dimensions && Dimensions < 4), int64_t>
mirrorIndex(int64_t index) {
  assert(0 <= index && index < Dimensions && "Invalid index");
  return mirrorIndexCalc<Dimensions>(index);
}

static llvm::function_ref<int64_t(int64_t)> getMirror(unsigned dimensions) {
  switch (dimensions) {
  case 1: return mirrorIndex<1>;
  case 2: return mirrorIndex<2>;
  case 3: return mirrorIndex<3>;
  default: llvm_unreachable("Invalid number of dimensions");
  }
}

/// True iff `op`'s sole result is still a SYCL id/range. The dialect
/// conversion driver will sometimes invoke the pattern on a clone whose result
/// type has already been remapped to an LLVM struct (e.g. by the catch-all
/// `LLVMOpLowering` in `convert-polygeist-to-llvm`); in that case we must
/// return failure rather than indexing into `getDimensions` (which would
/// fall off the end of its TypeSwitch).
static bool resultIsSYCLIdOrRange(Operation *op) {
  return op->getNumResults() == 1 &&
         isa<IDType, RangeType>(op->getResultTypes()[0]);
}

/// Replace an n-dimensional grid op with a stack-allocated id/range whose
/// dimensions are filled with the corresponding ROCDL intrinsic results.
static LogicalResult rewriteND(Operation *op, RocdlGridKind kind,
                               ConversionPatternRewriter &rewriter) {
  if (!resultIsSYCLIdOrRange(op))
    return failure();
  const auto dimensions = getDimensions(op->getResultTypes()[0]);
  const Location loc = op->getLoc();
  const Type i32 = rewriter.getIntegerType(32);
  const Type targetIndexType = rewriter.getI64Type();
  const auto dimMtTy = MemRefType::get(dimensions, targetIndexType);
  const Type resTy = op->getResultTypes()[0];
  const Value res =
      rewriter.create<memref::AllocaOp>(loc, MemRefType::get(1, resTy));
  const Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  auto mirrorer = getMirror(dimensions);
  for (int64_t i = 0; i < dimensions; ++i) {
    const Value index = rewriter.create<arith::ConstantIntOp>(loc, i, i32);
    const Value raw = buildRocdlGridDim(rewriter, loc, kind, mirrorer(i));
    const Value val = convertScalarToDtype(rewriter, loc, raw, targetIndexType,
                                           /*isUnsignedCast=*/false);
    const Value ptr = createGetOp(rewriter, loc, dimMtTy, res, index);
    rewriter.create<memref::StoreOp>(loc, val, ptr, zero);
  }
  rewriter.replaceOpWithNewOp<memref::LoadOp>(op, res, zero);
  return success();
}

template <typename OpTy> class NDGridOpPattern : public OpConversionPattern<OpTy> {
public:
  using OpConversionPattern<OpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    return rewriteND(op, rocdl_kind_of<OpTy>::value, rewriter);
  }
};

/// `sycl.global_id` = `sycl.work_group_id * sycl.work_group_size +
///                    sycl.local_id`. Computed per-dimension on ROCDL since
/// AMDGCN has no single intrinsic for the global id.
class GlobalIDOpPattern : public OpConversionPattern<SYCLGlobalIDOp> {
public:
  using OpConversionPattern<SYCLGlobalIDOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SYCLGlobalIDOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (!resultIsSYCLIdOrRange(op))
      return failure();
    const auto dimensions = getDimensions(op->getResultTypes()[0]);
    const Location loc = op.getLoc();
    const Type i32 = rewriter.getIntegerType(32);
    const Type targetIndexType = rewriter.getI64Type();
    const auto dimMtTy = MemRefType::get(dimensions, targetIndexType);
    const Type resTy = op->getResultTypes()[0];
    const Value res =
        rewriter.create<memref::AllocaOp>(loc, MemRefType::get(1, resTy));
    const Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto mirrorer = getMirror(dimensions);
    for (int64_t i = 0; i < dimensions; ++i) {
      const Value index = rewriter.create<arith::ConstantIntOp>(loc, i, i32);
      const int64_t amdgcnDim = mirrorer(i);
      const Value blockIdRaw = buildRocdlGridDim(rewriter, loc,
                                                 RocdlGridKind::WorkGroupId,
                                                 amdgcnDim);
      const Value blockDim = buildRocdlGridDim(rewriter, loc,
                                               RocdlGridKind::WorkGroupSize,
                                               amdgcnDim);
      const Value threadIdRaw = buildRocdlGridDim(rewriter, loc,
                                                  RocdlGridKind::WorkItemId,
                                                  amdgcnDim);
      // workgroup.id and workitem.id are i32; workgroup.dim is i64. Extend the
      // i32 values so arith operands all share the i64 element type.
      const Value blockId = convertScalarToDtype(rewriter, loc, blockIdRaw,
                                                 targetIndexType,
                                                 /*isUnsignedCast=*/false);
      const Value threadId = convertScalarToDtype(rewriter, loc, threadIdRaw,
                                                  targetIndexType,
                                                  /*isUnsignedCast=*/false);
      const Value mul =
          rewriter.create<arith::MulIOp>(loc, blockId, blockDim);
      const Value sum =
          rewriter.create<arith::AddIOp>(loc, mul, threadId);
      const Value ptr = createGetOp(rewriter, loc, dimMtTy, res, index);
      rewriter.create<memref::StoreOp>(loc, sum, ptr, zero);
    }
    rewriter.replaceOpWithNewOp<memref::LoadOp>(op, res, zero);
    return success();
  }
};

/// `sycl.num_work_items` = `sycl.work_group_size * sycl.num_work_groups`.
class NumWorkItemsOpPattern : public OpConversionPattern<SYCLNumWorkItemsOp> {
public:
  using OpConversionPattern<SYCLNumWorkItemsOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SYCLNumWorkItemsOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (!resultIsSYCLIdOrRange(op))
      return failure();
    const auto dimensions = getDimensions(op->getResultTypes()[0]);
    const Location loc = op.getLoc();
    const Type i32 = rewriter.getIntegerType(32);
    const Type targetIndexType = rewriter.getI64Type();
    const auto dimMtTy = MemRefType::get(dimensions, targetIndexType);
    const Type resTy = op->getResultTypes()[0];
    const Value res =
        rewriter.create<memref::AllocaOp>(loc, MemRefType::get(1, resTy));
    const Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto mirrorer = getMirror(dimensions);
    for (int64_t i = 0; i < dimensions; ++i) {
      const Value index = rewriter.create<arith::ConstantIntOp>(loc, i, i32);
      const int64_t amdgcnDim = mirrorer(i);
      const Value blockDim = buildRocdlGridDim(rewriter, loc,
                                               RocdlGridKind::WorkGroupSize,
                                               amdgcnDim);
      const Value gridDim = buildRocdlGridDim(rewriter, loc,
                                              RocdlGridKind::NumWorkGroups,
                                              amdgcnDim);
      // Both intrinsics return i64.
      const Value mul = rewriter.create<arith::MulIOp>(loc, blockDim, gridDim);
      const Value ptr = createGetOp(rewriter, loc, dimMtTy, res, index);
      rewriter.create<memref::StoreOp>(loc, mul, ptr, zero);
    }
    rewriter.replaceOpWithNewOp<memref::LoadOp>(op, res, zero);
    return success();
  }
};

/// HIP/AMDGCN does not provide a kernel-launch global offset. Produce an
/// id<N> filled with zeros, matching the HIP plugin convention.
class GlobalOffsetOpPattern : public OpConversionPattern<SYCLGlobalOffsetOp> {
public:
  using OpConversionPattern<SYCLGlobalOffsetOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SYCLGlobalOffsetOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (!resultIsSYCLIdOrRange(op))
      return failure();
    const auto dimensions = getDimensions(op->getResultTypes()[0]);
    const Location loc = op.getLoc();
    const Type i32 = rewriter.getIntegerType(32);
    const Type targetIndexType = rewriter.getI64Type();
    const auto dimMtTy = MemRefType::get(dimensions, targetIndexType);
    const Type resTy = op->getResultTypes()[0];
    const Value res =
        rewriter.create<memref::AllocaOp>(loc, MemRefType::get(1, resTy));
    const Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    const Value zeroI64 =
        rewriter.create<arith::ConstantIntOp>(loc, 0, targetIndexType);
    for (int64_t i = 0; i < dimensions; ++i) {
      const Value index = rewriter.create<arith::ConstantIntOp>(loc, i, i32);
      const Value ptr = createGetOp(rewriter, loc, dimMtTy, res, index);
      rewriter.create<memref::StoreOp>(loc, zeroI64, ptr, zero);
    }
    rewriter.replaceOpWithNewOp<memref::LoadOp>(op, res, zero);
    return success();
  }
};

/// 1D sub-group queries — lowered to extern device-function calls; the HIP
/// device libraries (libdevice / __ockl_*) provide the implementation at link
/// time.
template <typename OpTy>
class SubGroup1DPattern : public OpConversionPattern<OpTy> {
public:
  SubGroup1DPattern(MLIRContext *ctx, StringRef ockl,
                    PatternBenefit benefit = 1)
      : OpConversionPattern<OpTy>(ctx, benefit), ocklName(ockl) {}

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    const Location loc = op.getLoc();
    Operation *symbolTableOp = SymbolTable::getNearestSymbolTable(op);
    auto i32 = rewriter.getI32Type();
    auto fnTy = LLVM::LLVMFunctionType::get(i32, {}, /*isVarArg=*/false);
    auto fn = dyn_cast_or_null<LLVM::LLVMFuncOp>(
        SymbolTable::lookupSymbolIn(symbolTableOp, ocklName));
    if (!fn) {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(&symbolTableOp->getRegion(0).front());
      fn = rewriter.create<LLVM::LLVMFuncOp>(loc, ocklName, fnTy);
    }
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, fn, ValueRange{});
    return success();
  }

private:
  std::string ocklName;
};

} // namespace

void mlir::populateSYCLToROCDLConversionPatterns(TypeConverter &typeConverter,
                                                 RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  // The catch-all `LLVMOpLowering` in `convert-polygeist-to-llvm` runs at
  // benefit 1 and would otherwise eagerly rebuild SYCL grid ops with a
  // converted (LLVM struct) result type before our patterns can match.
  // Use a higher benefit so the SYCL-specific lowering wins.
  const PatternBenefit benefit{2};
  patterns.add<NDGridOpPattern<SYCLLocalIDOp>,
               NDGridOpPattern<SYCLWorkGroupIDOp>,
               NDGridOpPattern<SYCLWorkGroupSizeOp>,
               NDGridOpPattern<SYCLNumWorkGroupsOp>, GlobalIDOpPattern,
               NumWorkItemsOpPattern, GlobalOffsetOpPattern>(typeConverter, ctx,
                                                             benefit);
  patterns.add<SubGroup1DPattern<SYCLSubGroupSizeOp>>(
      ctx, "__ockl_get_sub_group_size", benefit);
  patterns.add<SubGroup1DPattern<SYCLSubGroupMaxSizeOp>>(
      ctx, "__ockl_get_max_sub_group_size", benefit);
  patterns.add<SubGroup1DPattern<SYCLSubGroupIDOp>>(
      ctx, "__ockl_get_sub_group_id", benefit);
  patterns.add<SubGroup1DPattern<SYCLNumSubGroupsOp>>(
      ctx, "__ockl_get_num_sub_groups", benefit);
  patterns.add<SubGroup1DPattern<SYCLSubGroupLocalIDOp>>(
      ctx, "__ockl_get_sub_group_local_id", benefit);
}

namespace {
class ConvertSYCLToROCDLPass
    : public impl::ConvertSYCLToROCDLBase<ConvertSYCLToROCDLPass> {
  void runOnOperation() override;
};
} // namespace

void ConvertSYCLToROCDLPass::runOnOperation() {
  auto *context = &getContext();
  auto module = getOperation();

  module.walk([&](gpu::GPUModuleOp gpuModule) {
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    TypeConverter typeConverter;
    typeConverter.addConversion([](Type t) { return t; });

    populateSYCLToROCDLConversionPatterns(typeConverter, patterns);

    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<ROCDL::ROCDLDialect>();
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<vector::VectorDialect>();

    target.addDynamicallyLegalDialect<SYCLDialect>([](Operation *op) {
      return !isa<SYCLGlobalIDOp, SYCLLocalIDOp, SYCLWorkGroupIDOp,
                  SYCLWorkGroupSizeOp, SYCLNumWorkGroupsOp, SYCLNumWorkItemsOp,
                  SYCLGlobalOffsetOp, SYCLSubGroupSizeOp, SYCLSubGroupMaxSizeOp,
                  SYCLSubGroupIDOp, SYCLNumSubGroupsOp, SYCLSubGroupLocalIDOp>(
          op);
    });

    if (failed(applyPartialConversion(gpuModule, target, std::move(patterns))))
      signalPassFailure();
  });
}
