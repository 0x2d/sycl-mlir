//===- SYCLToROCDL.h - SYCL to ROCDL Patterns -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides patterns to convert SYCL dialect to ROCDL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_SYCLTOROCDL_SYCLTOROCDL_H
#define MLIR_CONVERSION_SYCLTOROCDL_SYCLTOROCDL_H

#include <memory>

namespace mlir {
class Pass;
class RewritePatternSet;
class TypeConverter;

#define GEN_PASS_DECL_CONVERTSYCLTOROCDL
#include "mlir/Conversion/SYCLPasses.h.inc"
#undef GEN_PASS_DECL_CONVERTSYCLTOROCDL

/// Populates the given list with patterns that convert from SYCL to ROCDL.
void populateSYCLToROCDLConversionPatterns(TypeConverter &typeConverter,
                                           RewritePatternSet &patterns);

} // namespace mlir

#endif // MLIR_CONVERSION_SYCLTOROCDL_SYCLTOROCDL_H
