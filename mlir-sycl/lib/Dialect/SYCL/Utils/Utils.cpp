//===- Utils.cpp - Utilities to support the SYCL dialect ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements utilities for the SYCL dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SYCL/Utils/Utils.h"

using namespace mlir;
using namespace mlir::sycl;

SYCLIDGetOp sycl::createSYCLIDGetOp(Type resTy, TypedValue<MemRefType> id,
                                    unsigned index, OpBuilder builder,
                                    Location loc) {
  const Value indexOp = builder.create<arith::ConstantIntOp>(loc, index, 32);
  return builder.create<SYCLIDGetOp>(loc, resTy, id, indexOp);
}

SYCLRangeGetOp sycl::createSYCLRangeGetOp(Type resTy,
                                          TypedValue<MemRefType> range,
                                          unsigned index, OpBuilder builder,
                                          Location loc) {
  const Value indexOp = builder.create<arith::ConstantIntOp>(loc, index, 32);
  return builder.create<SYCLRangeGetOp>(loc, resTy, range, indexOp);
}

TypedValue<MemRefType> sycl::createSYCLIDConstructorOp(IDType idTy,
                                                       ValueRange indexes,
                                                       OpBuilder builder,
                                                       Location loc) {
  assert(idTy.getDimension() == indexes.size() &&
         "Expecting the size of indexes to be the id dimension");
  return builder.create<SYCLIDConstructorOp>(loc, MemRefType::get(1, idTy),
                                             indexes);
}

SYCLAccessorSubscriptOp
sycl::createSYCLAccessorSubscriptOp(AccessorPtrValue accessor,
                                    TypedValue<MemRefType> id,
                                    OpBuilder builder, Location loc) {
  const AccessorType accTy = accessor.getAccessorType();
  assert(accTy.getDimension() != 0 && "Dimensions cannot be zero");
  const auto MT = MemRefType::get(
      ShapedType::kDynamic, accTy.getType(), MemRefLayoutAttrInterface(),
      builder.getI64IntegerAttr(targetToAddressSpace(accTy.getTargetMode())));
  return builder.create<SYCLAccessorSubscriptOp>(loc, MT, accessor, id);
}

sycl::SYCLWorkGroupSizeOp
sycl::createWorkGroupSize(unsigned numDims, OpBuilder builder, Location loc) {
  const auto arrayType = builder.getType<sycl::ArrayType>(
      numDims, MemRefType::get(numDims, builder.getIndexType()));
  const auto rangeTy = builder.getType<sycl::RangeType>(numDims, arrayType);
  return builder.create<sycl::SYCLWorkGroupSizeOp>(loc, rangeTy);
}

void sycl::populateWorkGroupSize(SmallVectorImpl<Value> &wgSizes,
                                 unsigned numDims, OpBuilder builder,
                                 Location loc) {
  const auto arrayType = builder.getType<sycl::ArrayType>(
      numDims, MemRefType::get(numDims, builder.getIndexType()));
  const auto rangeTy = builder.getType<sycl::RangeType>(numDims, arrayType);
  auto wgSize = createWorkGroupSize(numDims, builder, loc);
  auto range =
      builder.create<memref::AllocaOp>(loc, MemRefType::get(1, rangeTy));
  const Value zeroIndex = builder.create<arith::ConstantIndexOp>(loc, 0);
  builder.create<memref::StoreOp>(loc, wgSize, range, zeroIndex);
  Type resTy = builder.getIndexType();
  for (unsigned dim = 0; dim < numDims; ++dim)
    wgSizes.push_back(
        sycl::createSYCLRangeGetOp(resTy, range, dim, builder, loc));
}

void sycl::populateLocalID(SmallVectorImpl<Value> &localIDs, unsigned numDims,
                           OpBuilder builder, Location loc) {
  const auto arrayType = builder.getType<sycl::ArrayType>(
      numDims, MemRefType::get(numDims, builder.getIndexType()));
  const auto idTy = builder.getType<sycl::IDType>(numDims, arrayType);
  auto localID = builder.create<sycl::SYCLLocalIDOp>(loc, idTy);
  auto id = builder.create<memref::AllocaOp>(loc, MemRefType::get(1, idTy));
  const Value zeroIndex = builder.create<arith::ConstantIndexOp>(loc, 0);
  builder.create<memref::StoreOp>(loc, localID, id, zeroIndex);
  Type resTy = builder.getIndexType();
  for (unsigned dim = 0; dim < numDims; ++dim)
    localIDs.push_back(sycl::createSYCLIDGetOp(resTy, id, dim, builder, loc));
}

LogicalResult sycl::checkEquivalent(Value lhs, Value rhs) {
  if (lhs == rhs) {
    // Identical values
    return success();
  } else {
    // Else check their define ops
    if (llvm::isa<mlir::BlockArgument>(lhs)) {
      // No define ops
      return failure();
    } else {
      // Have define ops
      Operation *def1 = lhs.getDefiningOp();
      Operation *def2 = rhs.getDefiningOp();
      if (def1->getName() != def2->getName() ||
          def1->getAttrs() != def2->getAttrs() ||
          def1->getPropertiesStorage() != def2->getPropertiesStorage()) {
        // Different op types
        return failure();
      }
      // Check equivalent of values
      unsigned int def1NumOperands = def1->getNumOperands();
      unsigned int def2NumOperands = def2->getNumOperands();
      if (def1NumOperands != def2NumOperands) {
        // Different operand number
        return failure();
      }
      for (int i = 0; i < def1NumOperands; i++) {
        if (failed(checkEquivalent(def1->getOperand(i), def2->getOperand(i)))) {
          return failure();
        }
      }
      return success();
    }
  }
}

Value sycl::getOffsetFromSubscriptOp(sycl::SYCLAccessorSubscriptOp op, StringAttr &tripleAttr) {
  Value memLoc1, memId1, memLoc2, memCast2, offset;

  if (tripleAttr.getValue() == "spir64-unknown-unknown-syclmlir") {
    memLoc1 = llvm::dyn_cast<memref::CastOp>(op.getIndex().getDefiningOp()).getSource();
    for (auto user: memLoc1.getUsers()) {
      if (auto storeUser = llvm::dyn_cast<affine::AffineStoreOp>(user)) {
        memId1 = storeUser.getValue();
        break;
      }
    }
    memLoc2 = llvm::dyn_cast<affine::AffineLoadOp>(memId1.getDefiningOp()).getMemref();
    for (auto user: memLoc2.getUsers()) {
      if (auto castUser = llvm::dyn_cast<memref::CastOp>(user)) {
        for (auto userInner: castUser.getDest().getUsers()) {
          if (auto castUserInner = llvm::dyn_cast<memref::MemorySpaceCastOp>(userInner)) {
            memCast2 = castUserInner.getDest();
            break;
          }
        }
        break;
      }
    }
    for (auto user: memCast2.getUsers()) {
      if (auto idUser = llvm::dyn_cast<sycl::SYCLConstructorOp>(user)) {
        auto idArgs = idUser.getArgs();
        offset = idArgs[0];
        break;
      }
    }
  } else if (tripleAttr.getValue() == "amdgcn-amd-amdhsa-syclmlir") {
    for (auto user: op.getIndex().getUsers()) {
      if (auto idUser = llvm::dyn_cast<sycl::SYCLConstructorOp>(user)) {
        auto idArgs = idUser.getArgs();
        offset = idArgs[0];
        break;
      }
    }
  }

  return offset;
}