//===--- CodeGenTypes.h - Type translation for MLIR CodeGen -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the code that handles AST -> MLIR type lowering.
//
//===----------------------------------------------------------------------===//

#ifndef CGEIST_LIB_CODEGEN_CODEGENTYPES_H
#define CGEIST_LIB_CODEGEN_CODEGENTYPES_H

#include "Attributes.h"
#include "mlir/IR/OwningOpRef.h"
#include "clang/Basic/ABI.h"
#include "llvm/ADT/SmallVector.h"
#include <cassert>
#include <map>
#include <utility>

namespace clang {
class ASTContext;
class CodeGenOptions;
class FunctionDecl;
class GlobalDecl;
class RecordDecl;
class BuiltinType;
class QualType;
class RecordType;
class Type;

namespace CodeGen {
class ABIInfo;
class CGCXXABI;
class CGCalleeInfo;
class CGFunctionInfo;
class CodeGenModule;
} // namespace CodeGen
} // namespace clang

namespace mlir {
class FunctionType;
class ModuleOp;
class Type;

namespace LLVM {
class LLVMStructType;
class LLVMPointerType;
} // namespace LLVM
} // namespace mlir

namespace mlirclang {
namespace CodeGen {

/// Encapsulates information about the way function arguments from
/// CGFunctionInfo should be passed to actual LLVM IR function.
class ClangToLLVMArgMapping {
  static const unsigned InvalidIndex = ~0U;
  unsigned InallocaArgNo;
  unsigned SRetArgNo;
  unsigned TotalIRArgs;

  /// Arguments of LLVM IR function corresponding to single Clang argument.
  struct IRArgs {
    unsigned PaddingArgIndex;
    // Argument is expanded to IR arguments at positions
    // [FirstArgIndex, FirstArgIndex + NumberOfArgs).
    unsigned FirstArgIndex;
    unsigned NumberOfArgs;

    IRArgs()
        : PaddingArgIndex(InvalidIndex), FirstArgIndex(InvalidIndex),
          NumberOfArgs(0) {}
  };

  llvm::SmallVector<IRArgs, 8> ArgInfo;

public:
  ClangToLLVMArgMapping(const clang::ASTContext &Context,
                        const clang::CodeGen::CGFunctionInfo &FI,
                        bool OnlyRequiredArgs = false);

  bool hasInallocaArg() const { return InallocaArgNo != InvalidIndex; }
  unsigned getInallocaArgNo() const {
    assert(hasInallocaArg());
    return InallocaArgNo;
  }

  bool hasSRetArg() const { return SRetArgNo != InvalidIndex; }
  unsigned getSRetArgNo() const {
    assert(hasSRetArg());
    return SRetArgNo;
  }

  unsigned totalIRArgs() const { return TotalIRArgs; }

  bool hasPaddingArg(unsigned ArgNo) const {
    assert(ArgNo < ArgInfo.size());
    return ArgInfo[ArgNo].PaddingArgIndex != InvalidIndex;
  }
  unsigned getPaddingArgNo(unsigned ArgNo) const {
    assert(hasPaddingArg(ArgNo));
    return ArgInfo[ArgNo].PaddingArgIndex;
  }

  /// Returns index of first IR argument corresponding to ArgNo, and their
  /// quantity.
  std::pair<unsigned, unsigned> getIRArgs(unsigned ArgNo) const {
    assert(ArgNo < ArgInfo.size());
    return std::make_pair(ArgInfo[ArgNo].FirstArgIndex,
                          ArgInfo[ArgNo].NumberOfArgs);
  }

private:
  void construct(const clang::ASTContext &Context,
                 const clang::CodeGen::CGFunctionInfo &FI,
                 bool OnlyRequiredArgs);
};

/// This class organizes the cross-module state that is used while lowering
/// AST types to MLIR types.
class CodeGenTypes {
  clang::CodeGen::CodeGenModule &CGM;
  clang::ASTContext &Context;
  mlir::OwningOpRef<mlir::ModuleOp> &TheModule;
  clang::CodeGen::CGCXXABI &TheCXXABI;

  std::map<const clang::RecordType *, mlir::LLVM::LLVMStructType> TypeCache;
  std::map<const clang::Type *, mlir::Type> BuiltinTypeCache;

public:
  CodeGenTypes(clang::CodeGen::CodeGenModule &CGM,
               mlir::OwningOpRef<mlir::ModuleOp> &Module);

  clang::CodeGen::CodeGenModule &getCGM() const { return CGM; }
  clang::ASTContext &getContext() const { return Context; }
  mlir::OwningOpRef<mlir::ModuleOp> &getModule() const { return TheModule; }
  clang::CodeGen::CGCXXABI &getCXXABI() const { return TheCXXABI; }
  const clang::CodeGenOptions &getCodeGenOpts() const;

  /// Construct the MLIR function type.
  mlir::FunctionType getFunctionType(const clang::CodeGen::CGFunctionInfo &FI,
                                     const clang::FunctionDecl &FD);

  /// Construct the IR attribute list of a function type or function call.
  void constructAttributeList(llvm::StringRef Name,
                              const clang::CodeGen::CGFunctionInfo &FI,
                              clang::CodeGen::CGCalleeInfo CalleeInfo,
                              mlirclang::AttributeList &AttrList,
                              unsigned &CallingConv, bool AttrOnCallSite,
                              bool IsThunk);

  /// Convert type T into an mlir::Type.
  ///
  /// This differs from getMLIRType in that it is used to convert to the memory
  /// representation for a type.  For example, the scalar representation for
  /// _Bool is i1, but the memory representation is usually i8 or i32, depending
  /// on the target.
  mlir::Type getMLIRTypeForMem(clang::QualType QT, bool *ImplicitRef = nullptr,
                               bool AllowMerge = true);
  // TODO: Possibly create a SYCLTypeCache
  mlir::Type getMLIRType(clang::QualType QT, bool *ImplicitRef = nullptr,
                         bool AllowMerge = true);

  mlir::Type getPointerOrMemRefType(mlir::Type Ty, unsigned AddressSpace,
                                    bool IsAlloca = false) const;

  mlir::LLVM::LLVMPointerType getPointerType(mlir::Type ElementType,
                                             unsigned AddressSpace = 0) const;

  const clang::CodeGen::CGFunctionInfo &
  arrangeGlobalDeclaration(clang::GlobalDecl GD);

  static bool isLLVMStructABI(const clang::RecordDecl *RD,
                              llvm::StructType *ST);

  clang::QualType getPromotionType(clang::QualType Ty) const;

private:
  void getDefaultFunctionAttributes(llvm::StringRef Name, bool HasOptnone,
                                    bool AttrOnCallSite,
                                    mlirclang::AttrBuilder &FuncAttrs) const;

  bool getCPUAndFeaturesAttributes(clang::GlobalDecl GD,
                                   AttrBuilder &Attrs) const;

  mlir::Type getMLIRType(const clang::BuiltinType *BT) const;
};

} // namespace CodeGen
} // namespace mlirclang

#endif // CGEIST_LIB_CODEGEN_CODEGENTYPES_H
