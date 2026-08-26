//===- Syr2kRegisterTile.cpp - syr2k 8x8 register-tile rewrite --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Device-side half of the launch-grid-shrinking SYR2K optimization (a
// per-benchmark sibling of SyrkRegisterTile; the two are distinguished purely
// by the kernel anchor -- Syr2k1 here vs Syr2k2 there -- so they never
// cross-fire). The source-level `polybench/orise/syr2k_opt.cpp` win is an 8x8
// register micro-kernel per work-item (arithmetic intensity ~0.5 -> ~8
// FMA/load) plus a host-side launch-grid shrink
// `range<2>(N,N)` -> `range<2>(ceil(N/8),ceil(N/8))`. This pass rewrites the
// device body to the register-tile form where each work-item computes the 8x8
// tile whose origin is its own item id:
//
//   bi = item.get_id(0) * 8 ;  bj = item.get_id(1) * 8
//
// The syr2k kernel is C = alpha*(A*B^T + B*A^T) + beta*C with THREE
// accessors: the captured struct is (i64 M_, C read_write, A read, B read) --
// member 2 is A and member 3 is B (verified from the cgeist IR dump: member 2
// is subscripted at (i,k) and (j,k) in the baseline body). Note that even a
// hypothetical A/B swap would be result-preserving here (the two k-terms
// swap), a safety property unique to syr2k.
//
// TWO rewrite sites are required, because the cgeist frontend emits the
// Syr2k1 body in TWO places with different consumers:
//
//   1. `func.func @_ZZZN..._clESB_` (top-level, signature
//      (memref<?x!llvm.struct<(i64, rw_acc, r_acc, r_acc)>>, !sycl_item_2_)):
//      the wrapper path. The `gpu.func @__pf_kernel_wrapperI6Syr2k1EE` kernel
//      calls this function once per enumerated user item (from the
//      RoundedRangeIDGenerator loop), and it is later inlined into
//      RoundedRangeKernel<...>::operator(). This is the ONLY path exercised
//      when range rounding fires (launch range not local-size divisible).
//      Rewriting the whole function body is safe: its two arguments are
//      exactly the (captured struct, item) pair the tile body needs.
//
//   2. `gpu.func @_ZTS6Syr2k1` (the direct kernel, body born INLINED from the
//      frontend -- it never calls the func.func): the path exercised when NO
//      rounding is needed, which is the common case (and the case the paired
//      host pass produces: ceil(N/8) is usually already local-size
//      divisible). Here the entry-block prologue (kernel-arg -> captured
//      struct reassembly + accessor __init) must be PRESERVED: the tile body
//      reuses the reassembled struct memref (found by type:
//      `polygeist.pointer2memref` result of
//      `memref<?x!llvm.struct<(i64, !sycl_accessor_2_f32_rw_dev,
//      !sycl_accessor_2_f32_r_dev, !sycl_accessor_2_f32_r_dev)>>`) and the
//      work-item (the zero-operand `sycl.call @getItem` result). Everything
//      after the getItem op is the inlined baseline body; it is erased and
//      replaced by the tile body.
//
// The launch-grid shrink itself is HOST-side and is performed by the paired
// LLVM pass `SYCLRewriteSyr2kRange` (llvm/lib/SYCLLowerIR/), enabled together
// with this pass via the sycl-bench `SYR2K_LAUNCH_TILE` CMake option
// (`-fsycl-rewrite-syr2k-range`). Each work-item computes its 8x8 `regC` tile
// exactly as `syr2k_opt` does: seed `regC[a][b] = C[{bi+a,bj+b}]*beta` (one
// global read per cell), `scf.for %k`: load the 16 per-k values
// `avA[a]=A[{bi+a,k}]`, `avB[a]=B[{bi+a,k}]`, `bvA[b]=A[{bj+b,k}]`,
// `bvB[b]=B[{bj+b,k}]`, then 2 FMAs per cell (`regC[a][b] +=
// alpha*avA[a]*bvB[b] + alpha*avB[a]*bvA[b]`), then one guarded store per
// cell. NO LDS, NO barrier, NO cooperative loading -- a faithful port of
// `syr2k_opt`'s algorithm. ALPHA and BETA are constexpr 1.0 in this
// benchmark; the emitted `* alpha` / `* beta` are folded away by the
// canonicalizer that follows this pass in the cgeist pipeline.
//
// Fail-safe guard: the tile computation is wrapped in
// `if (bi < M && bj < M)`. In the paired mode (host pass fired) the launch is
// ceil(N/8) x ceil(N/8) and the guard is always true (dead). If the host pass
// failed to match (launch still N x N), items with id >= ceil(N/8) exit
// immediately and every tile is still computed exactly once by the item whose
// id IS the tile index -- correct for any launch size, merely not faster.
//
// NO `reqd_work_group_size` is emitted: each work-item is now an independent
// tile, so any work-group shape is valid and LocalSize stays auto (256). A
// fixed (8,8) enforced size would break launches whose ceil(N/8) is not a
// multiple of 8 (the runtime enforces reqd_work_group_size as LocalSize,
// scheduler/commands.cpp:2344-2354).
//
// Edge handling for N % 8 != 0 (only the last tile row/col): loads use a
// clamped index (min(idx, N-1), safe, garbage into discarded cells), stores
// are predicated (`bi+a < N && bj+b < N`). Out-of-range `regC` cells are
// computed but never stored, so the result is correct for all N.
//
// Two gates (same as `SyrkRegisterTile`): (1) opt-in flag
// `--sycl-syr2k-register-tile` (off; wired in cgeist driver.cc), and (2) the
// module's `llvm.target_triple` must equal `amdgcn-amd-amdhsa-syclmlir`.
//
// Correctness: the k-summation order is preserved (0..M), each output cell is
// computed exactly once by exactly one work-item, and whatever offset
// behavior the baseline has is inherited unchanged: `bi`/`bj` derive from the
// SAME `item.get_id` the baseline body uses.
//
//===----------------------------------------------------------------------===//

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
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

#include "llvm/Support/Debug.h"

namespace mlir {
namespace sycl {
#define GEN_PASS_DEF_SYR2KREGISTERTILEPASS
#include "mlir/Dialect/SYCL/Transforms/Passes.h.inc"
} // namespace sycl
} // namespace mlir

#define DEBUG_TYPE "sycl-syr2k-register-tile"

using namespace mlir;
using namespace mlir::sycl;

namespace {

static constexpr llvm::StringRef AMDGCN_SYCLMLIR_TRIPLE =
    "amdgcn-amd-amdhsa-syclmlir";

/// The mangled kernel name fragment uniquely identifying the syr2k body
/// function. syr2k.cpp's kernel tag class is `Syr2k1` (distinct from syrk's
/// `Syr2k2`, so the two register-tile passes never cross-fire).
static constexpr llvm::StringRef SYR2K_KERNEL_FRAGMENT = "Syr2k1";

/// The direct (non-wrapper) kernel name: launched by the runtime whenever
/// range rounding is NOT needed -- the common case for the shrunk
/// ceil(N/8) launch produced by the paired host pass.
static constexpr llvm::StringRef SYR2K_DIRECT_KERNEL = "_ZTS6Syr2k1";

/// Register-tile dimensions (compile-time; match `syr2k_opt`'s BM=BN=8).
static constexpr int64_t BM = 8;
static constexpr int64_t BN = 8;

/// Number of members in the syr2k lambda's captured struct:
/// (i64 M_, C read_write, A read, B read).
static constexpr size_t SYR2K_STRUCT_MEMBERS = 4;

class Syr2kRegisterTilePass
    : public mlir::sycl::impl::Syr2kRegisterTilePassBase<Syr2kRegisterTilePass> {
public:
  void runOnOperation() final;

private:
  bool targetsAmdgcnSyclmlir(ModuleOp module) const {
    auto tripleAttr = module->getAttrOfType<StringAttr>(
        LLVM::LLVMDialect::getTargetTripleAttrName());
    return tripleAttr && tripleAttr.getValue() == AMDGCN_SYCLMLIR_TRIPLE;
  }

  bool isSyr2kBody(func::FuncOp func) const {
    auto kernelFuncObj = func->getAttrOfType<ArrayAttr>(
        sycl::SYCLDialect::getKernelFuncObjAttrName());
    if (!kernelFuncObj)
      return false;
    return llvm::any_of(kernelFuncObj.getAsRange<FlatSymbolRefAttr>(),
                        [](FlatSymbolRefAttr symbol) {
                          return symbol.getValue().contains(SYR2K_KERNEL_FRAGMENT);
                        });
  }

  /// Rewrite `body`'s region into the register-tile form. Baseline signature:
  /// arg0 = captured struct memref (M bound i64, C rw accessor, A r accessor,
  /// B r accessor), arg1 = !sycl_item_2_. Returns false (body untouched) if
  /// the signature does not match.
  bool rewriteBodyTile(func::FuncOp body);

  /// Rewrite the inlined baseline body of the DIRECT kernel
  /// `gpu.func @_ZTS6Syr2k1` to the register-tile form. Keeps the entry-block
  /// prologue (kernel-arg -> captured-struct reassembly, accessor __init),
  /// erases everything after the zero-operand `sycl.call @getItem`, and
  /// appends the tile body built from the prologue's struct memref and the
  /// getItem item. Returns false (kernel untouched) on any shape mismatch.
  bool rewriteDirectKernel(gpu::GPUFuncOp kernel);

  /// Shared tile-body emitter. \p structMemref must be a
  /// `memref<?x!llvm.struct<(i64, rw_acc, r_acc, r_acc)>>` and \p item an
  /// `!sycl_item_2_`. Emits at the builder's current insertion point; the
  /// caller adds the terminator. Returns false if \p structMemref's type does
  /// not match.
  bool emitTileBody(OpBuilder &b, Value structMemref, Value item);
};

/// Validate the captured-struct element type: must be an LLVM struct with
/// exactly the syr2k capture layout (i64 M_ first, then the 3 accessors).
static bool isSyr2kStruct(Type elemTy) {
  auto sTy = dyn_cast<LLVM::LLVMStructType>(elemTy);
  if (!sTy || sTy.getBody().size() != SYR2K_STRUCT_MEMBERS)
    return false;
  auto i0 = dyn_cast<IntegerType>(sTy.getBody()[0]);
  return i0 && i0.getWidth() == 64;
}

} // namespace

bool Syr2kRegisterTilePass::rewriteBodyTile(func::FuncOp body) {
  MLIRContext *ctx = body.getContext();

  if (body.getNumArguments() != 2)
    return false;

  Value arg0 = body.getArgument(0); // captured struct memref
  Value item = body.getArgument(1); // !sycl_item_2_ (tile origin source)

  auto arg0MT = dyn_cast<MemRefType>(arg0.getType());
  if (!arg0MT || !isSyr2kStruct(arg0MT.getElementType()))
    return false;

  // Wipe the entry-block body (reverse-erase; see SyrkRegisterAccumulator.cpp:
  // 194-218 for why reverse order + not clear()/walk).
  Region &region = body.getFunctionBody();
  Block &entry = region.front();
  SmallVector<Operation *> topOps;
  for (Operation &op : entry.getOperations())
    topOps.push_back(&op);
  for (Operation *op : llvm::reverse(topOps))
    op->erase();
  SmallVector<Block *> extraBlocks;
  for (Block &blk : region.getBlocks())
    if (&blk != &entry)
      extraBlocks.push_back(&blk);
  for (Block *blk : extraBlocks)
    blk->erase();

  OpBuilder b(ctx);
  b.setInsertionPointToStart(&entry);
  if (!emitTileBody(b, arg0, item))
    return false;
  b.create<func::ReturnOp>(body.getLoc());
  return true;
}

bool Syr2kRegisterTilePass::rewriteDirectKernel(gpu::GPUFuncOp kernel) {
  Region &region = kernel.getBody();
  // The direct kernel is emitted by the frontend as a single entry block
  // (loop nests live in nested regions). Bail on anything else.
  if (region.getBlocks().size() != 1)
    return false;
  Block &entry = region.front();

  // Locate, in program order:
  //   - the reassembled captured-struct memref: pointer2memref result typed
  //     memref<?x!llvm.struct<(i64, rw_acc, r_acc, r_acc)>> (exactly the
  //     syr2k capture layout, i64 first)
  //   - the work-item: the zero-operand `sycl.call @getItem` result
  // The struct must be defined at or before getItem (it is: the prologue
  // reassembles the struct and inits the accessors, then fetches the item).
  Value structMemref;
  Operation *structDef = nullptr;
  Value item;
  Operation *getItemOp = nullptr;
  for (Operation &op : entry.getOperations()) {
    if (auto p2m = dyn_cast<polygeist::Pointer2MemrefOp>(op)) {
      auto mt = dyn_cast<MemRefType>(p2m.getResult().getType());
      if (mt && !structDef) {
        if (isSyr2kStruct(mt.getElementType())) {
          structMemref = p2m.getResult();
          structDef = &op;
        }
      }
    }
    if (auto call = dyn_cast<sycl::SYCLCallOp>(op)) {
      if (call.getFunctionName() == "getItem" &&
          call.getArgs().empty() && call.getNumResults() == 1 && !getItemOp) {
        item = call.getResult();
        getItemOp = &op;
      }
    }
  }
  if (!getItemOp || !structDef || !structDef->isBeforeInBlock(getItemOp))
    return false;

  // Erase everything after the getItem op (the inlined baseline body,
  // including the gpu.return). The erased set is a contiguous tail, so no
  // surviving op can reference it (that would be use-before-def).
  SmallVector<Operation *> tail;
  bool after = false;
  for (Operation &op : entry.getOperations()) {
    if (after)
      tail.push_back(&op);
    if (&op == getItemOp)
      after = true;
  }
  for (Operation *op : llvm::reverse(tail))
    op->erase();

  OpBuilder b(kernel.getContext());
  b.setInsertionPointToEnd(&entry);
  if (!emitTileBody(b, structMemref, item))
    return false;
  b.create<gpu::ReturnOp>(kernel.getLoc());
  return true;
}

bool Syr2kRegisterTilePass::emitTileBody(OpBuilder &b, Value structMemref,
                                         Value item) {
  Location loc = structMemref.getLoc();
  MLIRContext *ctx = b.getContext();

  Type i32 = b.getI32Type();
  Type i64 = b.getIntegerType(64);
  Type indexTy = b.getIndexType();
  FloatType f32 = b.getF32Type();

  auto arg0MT = dyn_cast<MemRefType>(structMemref.getType());
  if (!arg0MT || !isSyr2kStruct(arg0MT.getElementType()))
    return false;
  ArrayRef<Type> members =
      cast<LLVM::LLVMStructType>(arg0MT.getElementType()).getBody();

  // 2-D id payload -- MUST be the array form (see SyrkRegisterAccumulator.cpp:
  // 171-185 for the provenance; the `(i64)` form fails phase-3 GEP lowering).
  Type idElemTy = parseType("!sycl.id<[2], (!sycl.array<[2], (memref<2xi64>)>)>",
                            ctx);
  if (!idElemTy)
    return false;
  MemRefType idMemrefTy = MemRefType::Builder({1}, idElemTy);
  MemRefType itemMemrefTy =
      MemRefType::Builder({1}, item.getType())
          .setMemorySpace(b.getI64IntegerAttr(5));
  MemRefType f32DynTy = MemRefType::Builder({ShapedType::kDynamic}, f32);
  // regC: per-work-item 8×8 register tile as a private memref alloca (AS 5) --
  // EXACTLY syr2k_opt's representation (syr2k_opt uses memref<8x8xf32,5>). A
  // loop-nest form lowers to a [64 x float] alloca with loop-variable GEP
  // indices; LLVM's AMDGPU promote-alloca rejects arrays > 16 elements
  // (AMDGPUPromoteAlloca.cpp tryPromoteAllocaToVector) and SROA cannot
  // partition variable-index accesses, so the accumulators would live in
  // scratch memory. With constant indices, SROA slices regC into 64 scalar
  // registers.
  MemRefType regCTy = MemRefType::Builder({BM, BN}, f32)
                          .setMemorySpace(b.getI64IntegerAttr(5));

  // --- constants ---
  Value c0i32 = b.create<arith::ConstantIntOp>(loc, 0, i32);
  Value c1i32 = b.create<arith::ConstantIntOp>(loc, 1, i32);
  Value c0idx = b.create<arith::ConstantIndexOp>(loc, 0);
  Value c1idx = b.create<arith::ConstantIndexOp>(loc, 1);
  Value c2idx = b.create<arith::ConstantIndexOp>(loc, 2);
  Value c3idx = b.create<arith::ConstantIndexOp>(loc, 3);
  Value c8idx = b.create<arith::ConstantIndexOp>(loc, 8);
  Value cM1idx = b.create<arith::ConstantIndexOp>(loc, -1); // for clamp (M-1)
  // ALPHA and BETA are constexpr 1 in syr2k.cpp; keep the multiplications so
  // the emitter stays a mechanical sibling of SyrkRegisterTile (the
  // canonicalizer that follows this pass in the cgeist pipeline folds them).
  Value beta = b.create<arith::ConstantFloatOp>(loc, llvm::APFloat(1.0f), f32);
  Value alpha = b.create<arith::ConstantFloatOp>(loc, llvm::APFloat(1.0f), f32);

  // --- captured struct members (M bound, C rw, A r, B r) ---
  Value mboundAcc = b.create<polygeist::SubIndexOp>(
      loc, MemRefType::get({ShapedType::kDynamic}, members[0]), structMemref, c0idx);
  Value cAcc = b.create<polygeist::SubIndexOp>(
      loc, MemRefType::get({ShapedType::kDynamic}, members[1]), structMemref, c1idx);
  Value aAcc = b.create<polygeist::SubIndexOp>(
      loc, MemRefType::get({ShapedType::kDynamic}, members[2]), structMemref, c2idx);
  Value bAcc = b.create<polygeist::SubIndexOp>(
      loc, MemRefType::get({ShapedType::kDynamic}, members[3]), structMemref, c3idx);
  Value Mraw = b.create<memref::LoadOp>(loc, i64, mboundAcc, ValueRange{c0idx});
  Value M = b.create<arith::IndexCastOp>(loc, indexTy, Mraw); // N == M == size
  Value Mm1 = b.create<arith::AddIOp>(loc, M, cM1idx); // M - 1 (clamp upper bound)

  // --- item -> tile origin: bi = item.get_id(0)*8, bj = item.get_id(1)*8 ---
  // Same item-memref + SYCLItemGetIDOp idiom as SyrkRegisterAccumulator.cpp:
  // 230-240 (GPU-verified on gfx906); item.get_id returns i64 -> cast to index.
  Value itemMem = b.create<memref::AllocaOp>(loc, itemMemrefTy);
  b.create<memref::StoreOp>(loc, item, itemMem, ValueRange{c0idx});
  Value ii64 = b.create<sycl::SYCLItemGetIDOp>(loc, TypeRange{i64},
                                              ValueRange{itemMem, c0i32});
  Value ji64 = b.create<sycl::SYCLItemGetIDOp>(loc, TypeRange{i64},
                                               ValueRange{itemMem, c1i32});
  Value iIdx = b.create<arith::IndexCastOp>(loc, indexTy, ii64);
  Value jIdx = b.create<arith::IndexCastOp>(loc, indexTy, ji64);
  Value bi = b.create<arith::MulIOp>(loc, iIdx, c8idx);
  Value bj = b.create<arith::MulIOp>(loc, jIdx, c8idx);

  // --- helpers (capture SSA values dominating the nested regions) ---
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
  auto loadC = [&](OpBuilder &bb, Value r, Value c) -> Value {
    return loadData(bb, cAcc, r, c);
  };
  auto storeC = [&](OpBuilder &bb, Value v, Value r, Value c) {
    Value idm = mkId(bb, r, c);
    Value sub = bb.create<sycl::SYCLAccessorSubscriptOp>(
        loc, TypeRange{f32DynTy}, ValueRange{cAcc, idm});
    bb.create<memref::StoreOp>(loc, v, sub, ValueRange{c0idx});
  };
  auto clampIdx = [&](OpBuilder &bb, Value x) -> Value {
    // min(x, M-1): safe index for out-of-range cells (garbage into discarded
    // regC cells; stores are separately predicated).
    return bb.create<arith::MinSIOp>(loc, x, Mm1);
  };

  // --- work-item: compute the 8×8 tile at (bi, bj) ---
  // regC is a memref<8x8xf32,5> alloca at FUNCTION SCOPE (NOT inside the
  // guard), and every regC access uses a CONSTANT index: the 8×8 seed/fma/
  // store bodies are FULLY UNROLLED, leaving only the k-loop as a loop.
  // This mirrors the machine code of the source-level syr2k_opt kernel (128
  // v_fma, ~80 global loads, ZERO private-memory accesses).
  //
  // Fail-safe guard `bi < M && bj < M`: dead in the paired mode (host pass
  // shrank the launch to ceil(N/8) x ceil(N/8), so bi/bj < M always); if the
  // host pass did NOT fire (launch still N x N), work-items whose id >=
  // ceil(N/8) exit here and every tile is still computed exactly once by the
  // item whose id IS the tile index -- correct for any launch size.
  SmallVector<Value, 8> tileIdx;
  for (int64_t t = 0; t < BM; ++t)
    tileIdx.push_back(b.create<arith::ConstantIndexOp>(loc, t));

  Value regC = b.create<memref::AllocaOp>(loc, regCTy);

  Value biIn = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, bi, M);
  Value bjIn = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, bj, M);
  Value inRange = b.create<arith::AndIOp>(loc, biIn, bjIn);

  b.create<scf::IfOp>(
      loc, inRange,
      [&](OpBuilder &ib, Location il) {
        // Seed regC[a][b] = C[{clamp(bi+a), clamp(bj+b)}] * beta, unrolled.
        for (int a = 0; a < BM; ++a) {
          Value r = ib.create<arith::AddIOp>(il, bi, tileIdx[a]);
          Value cr = clampIdx(ib, r);
          for (int c = 0; c < BN; ++c) {
            Value cc = ib.create<arith::AddIOp>(il, bj, tileIdx[c]);
            Value ccl = clampIdx(ib, cc);
            Value cv = loadC(ib, cr, ccl);
            Value sv = ib.create<arith::MulFOp>(il, cv, beta);
            ib.create<memref::StoreOp>(il, sv, regC,
                                       ValueRange{tileIdx[a], tileIdx[c]});
          }
        }

        // k loop (the only loop): for k = 0..M accumulate 2 FMAs per cell
        // into regC -- the A*B^T term (avA[a]*bvB[b]) and the B*A^T term
        // (avB[a]*bvA[b]) -- exactly syr2k_opt's loop structure: 16 loads
        // per iteration (rows bi+a of A and B, columns bj+b of A and B),
        // then the fully-unrolled 8x8 cell nest with constant regC indices.
        ib.create<scf::ForOp>(
            il, c0idx, M, c1idx, ValueRange{},
            [&](OpBuilder &kb, Location kl, Value k, ValueRange) {
              // Row-block loads: rows bi+a of A and B (the "i" operands).
              SmallVector<Value, 8> avA, avB;
              for (int a = 0; a < BM; ++a) {
                Value ar = kb.create<arith::AddIOp>(kl, bi, tileIdx[a]);
                Value acr = clampIdx(kb, ar);
                avA.push_back(loadData(kb, aAcc, acr, k));
                avB.push_back(loadData(kb, bAcc, acr, k));
              }
              // Column-block loads: columns bj+b of A and B (the "j"
              // operands). A and B are symmetric roles here -- both are
              // indexed by the column block for the transposed term.
              SmallVector<Value, 8> bvA, bvB;
              for (int c = 0; c < BN; ++c) {
                Value bc = kb.create<arith::AddIOp>(kl, bj, tileIdx[c]);
                Value bcc = clampIdx(kb, bc);
                bvA.push_back(loadData(kb, aAcc, bcc, k));
                bvB.push_back(loadData(kb, bAcc, bcc, k));
              }
              // alpha pre-scale (folds away at alpha == 1.0).
              SmallVector<Value, 8> avAs, avBs;
              for (int a = 0; a < BM; ++a) {
                avAs.push_back(kb.create<arith::MulFOp>(kl, avA[a], alpha));
                avBs.push_back(kb.create<arith::MulFOp>(kl, avB[a], alpha));
              }
              // 64 cells x 2 FMAs:
              //   regC[a][b] = fma(avB[a], bvA[b], fma(avA[a], bvB[b], regC[a][b]))
              for (int a = 0; a < BM; ++a) {
                for (int c = 0; c < BN; ++c) {
                  Value old = kb.create<memref::LoadOp>(
                      kl, f32, regC, ValueRange{tileIdx[a], tileIdx[c]});
                  Value t1 = kb.create<math::FmaOp>(kl, avAs[a], bvB[c], old);
                  Value nv = kb.create<math::FmaOp>(kl, avBs[a], bvA[c], t1);
                  kb.create<memref::StoreOp>(kl, nv, regC,
                                             ValueRange{tileIdx[a], tileIdx[c]});
                }
              }
              kb.create<scf::YieldOp>(kl);
            });

        // Store regC[a][b] -> C[{bi+a, bj+b}], predicated per cell, unrolled.
        for (int a = 0; a < BM; ++a) {
          Value r = ib.create<arith::AddIOp>(il, bi, tileIdx[a]);
          Value rin = ib.create<arith::CmpIOp>(il, arith::CmpIPredicate::ult,
                                               r, M);
          for (int c = 0; c < BN; ++c) {
            Value cc = ib.create<arith::AddIOp>(il, bj, tileIdx[c]);
            Value cin = ib.create<arith::CmpIOp>(il, arith::CmpIPredicate::ult,
                                                 cc, M);
            Value in = ib.create<arith::AndIOp>(il, rin, cin);
            Value rv = ib.create<memref::LoadOp>(il, f32, regC,
                                                  ValueRange{tileIdx[a], tileIdx[c]});
            ib.create<scf::IfOp>(
                il, in,
                [&](OpBuilder &sb, Location sl) {
                  storeC(sb, rv, r, cc);
                  sb.create<scf::YieldOp>(sl);
                });
          }
        }

        ib.create<scf::YieldOp>(il);
      });

  return true;
}

void Syr2kRegisterTilePass::runOnOperation() {
  ModuleOp module = getOperation();

  if (!targetsAmdgcnSyclmlir(module)) {
    LLVM_DEBUG(llvm::dbgs() << "Syr2kRegisterTile: target is not "
                            << AMDGCN_SYCLMLIR_TRIPLE << ", skipping.\n");
    return;
  }

  // Rewrite the Syr2k1 lambda body to the register-tile form (each work-item
  // owns the 8x8 tile at (item_id*8, item_id*8)). The paired host-side launch
  // shrink is performed by the LLVM pass SYCLRewriteSyr2kRange
  // (-fsycl-rewrite-syr2k-range). NO reqd_work_group_size: amdgcn HSA metadata
  // requires a 3-element array but the SYCL runtime's program_manager
  // dimension check (program_manager.cpp:2810) requires the element count to
  // match the 2-D launch -- an irreconcilable conflict, so any reqd value is
  // rejected at launch. LocalSize stays auto (256).
  //
  // Rewrite site 1 -- the top-level lambda body func.func (the WRAPPER path:
  // __pf_kernel_wrapper -> RoundedRangeIDGenerator loop -> this function; the
  // only path that runs when range rounding fires).
  SmallVector<func::FuncOp, 2> syr2kBodies;
  module.walk([&](func::FuncOp func) {
    if (isSyr2kBody(func))
      syr2kBodies.push_back(func);
  });

  for (func::FuncOp body : syr2kBodies) {
    ++NumDetected;
    LLVM_DEBUG(llvm::dbgs() << "Syr2kRegisterTile: detected body " << body.getName()
                   << " (" << body.getNumArguments() << " args)\n");
    if (rewriteBodyTile(body))
      ++NumRewritten;
  }

  // Rewrite site 2 -- the DIRECT kernel gpu.func @_ZTS6Syr2k1 (body born
  // inlined from the frontend; the path that runs when NO rounding is needed,
  // which is the common case for the shrunk ceil(N/8) launch). Without this,
  // the paired host pass shrinks the launch while the direct kernel still
  // runs the baseline single-cell body -- 1/64 of C computed, wrong results.
  SmallVector<gpu::GPUFuncOp, 2> directKernels;
  module.walk([&](gpu::GPUFuncOp kernel) {
    if (kernel.getName() == SYR2K_DIRECT_KERNEL)
      directKernels.push_back(kernel);
  });

  for (gpu::GPUFuncOp kernel : directKernels) {
    ++NumDetected;
    LLVM_DEBUG(llvm::dbgs() << "Syr2kRegisterTile: detected direct kernel "
                   << kernel.getName() << "\n");
    if (rewriteDirectKernel(kernel))
      ++NumRewritten;
  }
}

std::unique_ptr<Pass> mlir::sycl::createSyr2kRegisterTilePass() {
  return std::make_unique<Syr2kRegisterTilePass>();
}
