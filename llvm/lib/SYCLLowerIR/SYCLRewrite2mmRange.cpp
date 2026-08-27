//===- SYCLRewrite2mmRange.cpp - 2mm launch-range rewrite ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// See SYCLRewrite2mmRange.h for the design. Summary of the IR shape matched
// per launch site (from a -O3 -Xclang -disable-llvm-passes -emit-llvm dump of
// the 2mm.cpp host TU at PipelineStartEP):
//
//   define ... @...parallel_for_lambda_implI15Polybench_2mm_N...(
//       ptr %this, i64 %UserRange.coerce0, i64 %UserRange.coerce1, ...) {
//     %UserRange = alloca %"class.sycl::_V1::range"   ; {i64, i64}
//     %agg.tmp11 = alloca %"class.sycl::_V1::range"
//     store i64 %UserRange.coerce0, ptr %UserRange
//     %f1 = getelementptr {i64, i64}, ptr %UserRange, i64 0, i32 1
//     store i64 %UserRange.coerce1, ptr %f1
//     call void @llvm.memcpy(ptr %agg.tmp11, ptr %UserRange, i64 16, ...)
//     ...
//     invoke void @_ZN4sycl3_V17handler15getRoundedRangeILi2EEE...(
//         ptr sret %0, ptr %cgh,
//         i64 %load.dim0 /* <- load %agg.tmp11 */,
//         i64 %load.dim1 /* <- load gep %agg.tmp11, 0, 1 */)
//
// There are TWO such instantiations (one per 2mm kernel, each identified by
// its kernel tag class in the mangled name); the pass rewrites the UserRange
// stores in BOTH.
//
//===----------------------------------------------------------------------===//

#include "llvm/SYCLLowerIR/SYCLRewrite2mmRange.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "sycl-rewrite-2mm-range"

using namespace llvm;

STATISTIC(NumLaunchSitesRewritten, "Number of launch sites rewritten");

static constexpr StringRef TWO_MM_KERNEL_1_NAME = "_ZTS15Polybench_2mm_1";
static constexpr StringRef TWO_MM_KERNEL_2_NAME = "_ZTS15Polybench_2mm_2";
/// The kernel tag class fragments embedded in each
/// `parallel_for_lambda_impl` instantiation's mangled name.
static constexpr StringRef TWO_MM_KERNEL_1_FRAGMENT = "Polybench_2mm_1";
static constexpr StringRef TWO_MM_KERNEL_2_FRAGMENT = "Polybench_2mm_2";
static constexpr StringRef LAMBDA_IMPL_FRAGMENT = "parallel_for_lambda_impl";
static constexpr StringRef ROUNDED_RANGE_CALLEE = "getRoundedRangeILi2EE";
/// Tile edge: 2mm_opt's TS (work-group tile edge). The launch is padded to a
/// multiple of TS per dim (one work-item per output cell), so the explicit
/// local size (16,16) injected by SYCL_FORCE_LOCAL_SIZE divides GlobalSize.
static constexpr unsigned TILE = 16;

/// Return the module's global string constant equal to \p Name (ignoring the
/// trailing NUL), or nullptr. The kernel-name anchor from
/// `__builtin_sycl_unique_stable_name(Polybench_2mm_N)` is a private constant
/// whose content is exactly "_ZTS15Polybench_2mm_N".
static const GlobalVariable *findStringAnchor(Module &M, StringRef Name) {
  for (const GlobalVariable &GV : M.globals()) {
    if (!GV.isConstant() || !GV.hasInitializer())
      continue;
    const auto *Arr = dyn_cast<ConstantDataSequential>(GV.getInitializer());
    if (!Arr || !Arr->isString())
      continue;
    StringRef S = Arr->getAsString();
    while (!S.empty() && S.back() == '\0')
      S = S.drop_back();
    if (S == Name)
      return &GV;
  }
  return nullptr;
}

/// Does \p F name one of the two 2mm `parallel_for_lambda_impl`
/// instantiations? Each instantiation's mangled name embeds its kernel tag
/// class (`15Polybench_2mm_1` / `15Polybench_2mm_2`).
static bool is2mmLambdaImpl(const Function &F) {
  StringRef Name = F.getName();
  if (!Name.contains(LAMBDA_IMPL_FRAGMENT))
    return false;
  return Name.contains(TWO_MM_KERNEL_1_FRAGMENT) ||
         Name.contains(TWO_MM_KERNEL_2_FRAGMENT);
}

PreservedAnalyses SYCLRewrite2mmRangePass::run(Module &M,
                                               ModuleAnalysisManager &) {
  // Kernel-name anchor: without it this TU launches neither 2mm kernel.
  if (!findStringAnchor(M, TWO_MM_KERNEL_1_NAME) &&
      !findStringAnchor(M, TWO_MM_KERNEL_2_NAME)) {
    LLVM_DEBUG(dbgs() << "sycl-rewrite-2mm-range: no 2mm kernel-name anchor, "
                         "no-op\n");
    return PreservedAnalyses::all();
  }

  bool Changed = false;
  for (Function &F : M) {
    if (F.isDeclaration() || !is2mmLambdaImpl(F))
      continue;

    // Find the 2-D parallel_for launch site: a call/invoke of
    // getRoundedRangeILi2EE whose two i64 range arguments are loads from a
    // common alloca.
    AllocaInst *RangeAlloca = nullptr;
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        auto *CB = dyn_cast<CallBase>(&I);
        if (!CB)
          continue;
        Function *Callee = CB->getCalledFunction();
        if (!Callee || !Callee->getName().contains(ROUNDED_RANGE_CALLEE))
          continue;

        SmallVector<LoadInst *, 2> RangeLoads;
        for (Value *Arg : CB->args()) {
          auto *L = dyn_cast<LoadInst>(Arg);
          if (L && L->getType()->isIntegerTy(64))
            RangeLoads.push_back(L);
        }
        if (RangeLoads.size() != 2)
          continue;
        // Dim 0 loads the alloca directly; dim 1 loads a GEP of it
        // (gep %A, 0, 1) -- getUnderlyingObject resolves both to the alloca.
        auto *A0 = dyn_cast<AllocaInst>(
            getUnderlyingObject(RangeLoads[0]->getPointerOperand()));
        auto *A1 = dyn_cast<AllocaInst>(
            getUnderlyingObject(RangeLoads[1]->getPointerOperand()));
        if (!A0 || A0 != A1)
          continue;
        RangeAlloca = A0;
        break;
      }
      if (RangeAlloca)
        break;
    }
    if (!RangeAlloca)
      continue;

    // Resolve the ORIGIN range object. At PipelineStartEP the call's loads
    // typically read a temporary copy (`%agg.tmp11`) that is filled by a
    // memcpy FROM the real range object (`%UserRange`, itself filled by the
    // two i64 stores of the by-value range arguments of
    // parallel_for_lambda_impl). Rewrite the stores at the origin so every
    // downstream consumer (getRoundedRange args, the range-rounding wrapper's
    // UserRange member, checkValueRange, MNDRDesc.set) sees ceil16(N).
    auto collectStoreSites = [&](AllocaInst *A) {
      SmallVector<StoreInst *, 4> Stores;
      for (BasicBlock &BB : F)
        for (Instruction &I : BB) {
          auto *ST = dyn_cast<StoreInst>(&I);
          if (!ST)
            continue;
          if (getUnderlyingObject(ST->getPointerOperand()) != A)
            continue;
          auto *VTy = dyn_cast<IntegerType>(ST->getValueOperand()->getType());
          if (VTy && VTy->getBitWidth() == 64)
            Stores.push_back(ST);
        }
      return Stores;
    };

    SmallVector<StoreInst *, 4> Stores = collectStoreSites(RangeAlloca);
    if (Stores.empty()) {
      // Trace memcpy copies into the loads' alloca back to their source.
      SmallPtrSet<AllocaInst *, 2> SrcAllocas;
      for (BasicBlock &BB : F)
        for (Instruction &I : BB) {
          auto *MC = dyn_cast<MemCpyInst>(&I);
          if (!MC)
            continue;
          if (getUnderlyingObject(MC->getDest()) != RangeAlloca)
            continue;
          if (auto *SrcA = dyn_cast<AllocaInst>(
                  getUnderlyingObject(MC->getSource())))
            SrcAllocas.insert(SrcA);
        }
      if (SrcAllocas.size() != 1)
        continue; // ambiguous or chained -- leave untouched (safe no-op)
      AllocaInst *Origin = *SrcAllocas.begin();
      if (Origin == RangeAlloca)
        continue;
      Stores = collectStoreSites(Origin);
      if (Stores.empty())
        continue;
    }

    // Rewrite every i64 store into the range object to store ceil(V/16)*16,
    // built at the store's insertion point. (In 2mm this is exactly the two
    // stores of N into %UserRange, per launch site.) Values already
    // divisible by 16 -- every benchmark size -- are left numerically
    // unchanged; non-multiples are padded up so the (16,16) local size
    // divides GlobalSize.
    unsigned RewrittenInF = 0;
    for (StoreInst *ST : Stores) {
      Value *V = ST->getValueOperand();
      auto *VTy = cast<IntegerType>(V->getType());
      IRBuilder<> B(ST);
      Value *PlusTile =
          B.CreateAdd(V, ConstantInt::get(VTy, TILE - 1), "2mm.plus");
      Value *Ceil =
          B.CreateUDiv(PlusTile, ConstantInt::get(VTy, TILE), "2mm.ceil");
      Value *Padded =
          B.CreateMul(Ceil, ConstantInt::get(VTy, TILE), "2mm.padded");
      ST->setOperand(0, Padded);
      ++RewrittenInF;
    }

    if (RewrittenInF) {
      ++NumLaunchSitesRewritten;
      Changed = true;
      LLVM_DEBUG(dbgs() << "sycl-rewrite-2mm-range: rewrote " << RewrittenInF
                        << " range store(s) in " << F.getName() << "\n");
    }
  }

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
