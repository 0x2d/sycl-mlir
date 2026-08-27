//===- GemmLocalTile.cpp - gemm 16x16 local-memory (LDS) tile rewrite ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Device-side half of the GEMM optimization, transcribing the LDS-tiled
// micro-kernel of the source-level `polybench/orise/gemm_opt.cpp` (a direct
// sibling of TwoMmLocalTile, which GPU-verified 4.61x over the baseline on
// gfx906 and BEAT 2mm_opt; the earlier register-tile form of that family
// measured 2.2x SLOWER and was replaced). This pass rewrites the ONE gemm
// kernel (`Gemm`: C = ALPHA*A*B + BETA*C, C read_write, A/B read); it shares
// the 2mm/3mm GEMM shape and captured-struct layout (i64 NK_, OUT accessor,
// IN1 read, IN2 read) and differs only in the seed (existing C scaled by
// BETA) and the ALPHA-scaled FMA.
//
// The tile shape is gemm_opt's (same as 2mm_opt's): TS = TK = 16. Each
// work-group is 16x16 = 256 work-items and cooperatively stages a 16x16
// tile of A (rows) and B (columns) in LDS per k-chunk; each work-item
// computes ONE output cell with a scalar accumulator:
//
//   irow = item.get_id(0) ; jcol = item.get_id(1)     (cell = item id)
//   ty   = irow % 16      ; tx   = jcol % 16          (intra-tile coords)
//   acc  = C[irow][jcol] * BETA
//   for (kt = 0; kt < M; kt += 16) {
//     aTile[ty][tx] = A[irow][kt+tx]   (clamped)      one load per item
//     bTile[ty][tx] = B[kt+ty][jcol]   (clamped)      one load per item
//     barrier
//     for (kk = 0; kk < 16; kk++)
//       acc += ALPHA * aTile[ty][kk] * bTile[kk][tx]
//     barrier
//   }
//   if (irow < M && jcol < M) C[irow][jcol] = acc
//
// (ALPHA is folded into the FMA as a multiply of the staged product; BETA
// scales the seed -- the baseline body is `C[item] *= BETA` followed by
// `C[item] += ALPHA * A[i,k] * B[k,j]`, so this is the exact floating-point
// sequence with the k-loop reassociated into the tile order. The k-tail
// zero-fill below is the one deliberate numerical difference: it makes
// N % 16 != 0 CORRECT where gemm_opt's clamp-only form double-counts the
// last k row/column -- the same latent bug 2mm's round-1 GPU verify exposed
// at N=1030 and fixed with the same select-zero.)
//
// The two LDS tiles are private module-scope `memref.global`s with the SYCL
// local address space (#sycl.access.address_space<local> = AS 3), created in
// the enclosing gpu.module (same construction as LoopInternalization's
// WGLocalMem). Phase-3 lowers them to internal addrspace(3) LLVM globals,
// which the AMDGPU backend allocates into group_segment_fixed_size -- static
// LDS, no kernel-signature or host-side plumbing (the local-accessor dynamic
// shared path is NOT used; this also sidesteps the 2-local-accessor
// miscompilation, since no local accessor exists at all).
//
// Barriers are `rocdl.barrier` ops: the ROCDL dialect is legal in the
// phase-3 conversion target (PolygeistToLLVM.cpp marks it legal for the
// SYCLToROCDL intrinsics) and its LLVMIR translation lowers to
// fence(workgroup) + llvm.amdgcn.s.barrier (registered in cgeist's driver).
// `gpu.barrier` is explicitly illegal in phase 3 and `spirv.ControlBarrier`
// has no ROCDL lowering -- rocdl.barrier is the only viable op here.
//
// The launch must provide 16x16 work-groups: the paired HOST LLVM pass
// `SYCLRewriteGemmRange` (-fsycl-rewrite-gemm-range) rewrites the range<2>
// launch from (N,N) to (ceil16(N), ceil16(N)) -- the FULL padded grid, one
// work-item per output CELL (unlike the register-tile design's shrunk grid)
// -- and the SYCL runtime env var SYCL_FORCE_LOCAL_SIZE=16,16 (this tree)
// makes the launch carry the explicit local size (16,16) -- neither auto
// local (UR's simpleGuessLocalWorkSize sizes only dim 0) nor
// reqd_work_group_size (UR HIP stubs COMPILE_WORK_GROUP_SIZE to {0,0,0},
// and the runtime rejects 3-element reqd against a 2-D launch) can supply
// it. With local (16,16) guaranteed, all of gemm_opt's index math reduces
// to arithmetic on the item id: ty = irow % 16, tx = jcol % 16 -- no
// local_id/group_id ops needed.
//
// TWO rewrite sites are required, because the cgeist frontend emits the
// body in TWO places with different consumers:
//
//   1. `func.func @_ZZZN..._clESB_` (top-level, signature
//      (memref<?x!llvm.struct<(i64, out_acc, r_acc, r_acc)>>, !sycl_item_2_)):
//      the wrapper path (`gpu.func @__pf_kernel_wrapperI4GemmE` calls it
//      once per enumerated user item). Only exercised when range rounding
//      fires -- which never happens for the rewritten values: the host pass
//      stores ceil16(N), and getRoundedRange only rounds dim-0 values that
//      are NOT multiples of its MinFactor 16. Rewritten anyway so both
//      copies of the body stay consistent.
//
//   2. `gpu.func @_ZTS4Gemm` (the direct kernel, body born INLINED from the
//      frontend -- the path that always runs here). The entry-block
//      prologue (kernel-arg -> captured struct reassembly + accessor
//      __init) is PRESERVED; everything after the zero-operand
//      `sycl.call @getItem` is erased and replaced by the tile body.
//
// Edge handling for N % 16 != 0 (two kinds of out-of-range):
//  - Padded WORK-ITEMS (irow >= M or jcol >= M, up to 15 per dim): loads
//    clamped (min(idx, M-1), safe -- the values feed a discarded
//    accumulator), final store predicated (irow < M && jcol < M).
//  - Tail K-LANES (kt+tx >= M or kt+ty >= M in the last chunk): clamping
//    alone would re-read the last row/column and DOUBLE-COUNT it (a latent
//    bug in gemm_opt itself, which is never run at N % 16 != 0). Instead the
//    out-of-range k lanes store 0.0 into the tile (branch-free select; the
//    clamped load still executes safely), contributing 0*0 to every FMA.
//
// No reqd_work_group_size is attached (see above -- it is unusable on this
// stack); the (16,16) work-group shape is supplied at launch time by
// SYCL_FORCE_LOCAL_SIZE.
//
// Two gates (same as the sibling passes): (1) opt-in flag
// `--sycl-gemm-local-tile` (off; wired in cgeist driver.cc), and (2) the
// module's `llvm.target_triple` must equal `amdgcn-amd-amdhsa-syclmlir`.
//
//===----------------------------------------------------------------------===//

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Polygeist/IR/PolygeistOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SYCL/IR/SYCLOps.h"
#include "mlir/Dialect/SYCL/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Visitors.h"

#include "llvm/Support/Debug.h"

namespace mlir {
namespace sycl {
#define GEN_PASS_DEF_GEMMLOCALTILEPASS
#include "mlir/Dialect/SYCL/Transforms/Passes.h.inc"
} // namespace sycl
} // namespace mlir

#define DEBUG_TYPE "sycl-gemm-local-tile"

using namespace mlir;
using namespace mlir::sycl;

namespace {

static constexpr llvm::StringRef AMDGCN_SYCLMLIR_TRIPLE =
    "amdgcn-amd-amdhsa-syclmlir";

/// The mangled kernel name fragment uniquely identifying the gemm lambda
/// body. gemm.cpp's kernel tag class is `Gemm` (C = ALPHA*A*B + BETA*C, C
/// read_write so the accumulator is seeded from the existing C scaled by
/// BETA). EXACT-match against `_ZTS4Gemm` -- bare "Gemm" is a substring of
/// the unrelated gemm_opt kernel `_ZTS16Polybench_Gemm1` and must never
/// match.
static constexpr llvm::StringRef GEMM_KERNEL_FRAGMENT = "_ZTS4Gemm";

/// The direct (non-wrapper) kernel name: launched by the runtime whenever
/// range rounding is NOT needed -- always the case for the ceil16(N) launch
/// produced by the paired host pass (ceil16(N) is a multiple of the rounding
/// MinFactor 16, so getRoundedRange never fires).
static constexpr llvm::StringRef GEMM_DIRECT_KERNEL = "_ZTS4Gemm";

/// Tile shape, matching polybench/orise/gemm_opt.cpp: TS is the work-group
/// tile edge in the output (16x16 work-items, one output cell each) and TK
/// the k-depth staged in LDS per chunk (16, so the 256 work-items fill each
/// tile with one load each).
static constexpr int64_t TS = 16;
static constexpr int64_t TK = 16;

/// The scalar constants of polybench/gemm.cpp (#define'd there, constexpr
/// in gemm_opt): the seed is C * BETA and each staged product is scaled by
/// ALPHA inside the FMA chain.
static constexpr float ALPHA = 32412.0f;
static constexpr float BETA = 2123.0f;

/// LDS tile symbol names (private globals; only one gemm kernel exists, but
/// the per-kernel naming convention of the 2mm/3mm siblings is kept).
static constexpr llvm::StringRef TILE_A_KERNEL = "_gemm_aTile";
static constexpr llvm::StringRef TILE_B_KERNEL = "_gemm_bTile";

/// Number of members in the gemm lambda's captured struct:
/// (i64 NK_, C read_write, A read, B read) -- verified from the cgeist IR
/// dump; identical layout to the 2mm/3mm siblings (i64 first, then OUT,
/// IN1, IN2).
static constexpr size_t GEMM_STRUCT_MEMBERS = 4;

class GemmLocalTilePass
    : public mlir::sycl::impl::GemmLocalTilePassBase<GemmLocalTilePass> {
public:
  void runOnOperation() final;

private:
  bool targetsAmdgcnSyclmlir(ModuleOp module) const {
    auto tripleAttr = module->getAttrOfType<StringAttr>(
        LLVM::LLVMDialect::getTargetTripleAttrName());
    return tripleAttr && tripleAttr.getValue() == AMDGCN_SYCLMLIR_TRIPLE;
  }

  /// Returns true if \p func is the gemm lambda body (matched EXACTLY on
  /// the `sycl.kernel_func_obj` anchor `_ZTS4Gemm`).
  bool isGemmBody(func::FuncOp func) const {
    auto kernelFuncObj = func->getAttrOfType<ArrayAttr>(
        sycl::SYCLDialect::getKernelFuncObjAttrName());
    if (!kernelFuncObj)
      return false;
    for (FlatSymbolRefAttr symbol :
         kernelFuncObj.getAsRange<FlatSymbolRefAttr>()) {
      // Exact match: "Gemm" is a substring of gemm_opt's
      // `_ZTS16Polybench_Gemm1` and must not match.
      if (symbol.getValue() == GEMM_KERNEL_FRAGMENT)
        return true;
    }
    return false;
  }

  /// Rewrite `body`'s region into the LDS-tile form. Baseline signature:
  /// arg0 = captured struct memref (i64 NK_ bound, C rw accessor, A r
  /// accessor, B r accessor), arg1 = !sycl_item_2_. Returns false (body
  /// untouched) if the signature does not match.
  bool rewriteBodyTile(func::FuncOp body);

  /// Rewrite the inlined baseline body of the DIRECT kernel
  /// `gpu.func @_ZTS4Gemm` to the LDS-tile form. Keeps the entry-block
  /// prologue (kernel-arg -> captured-struct reassembly, accessor __init),
  /// erases everything after the zero-operand `sycl.call @getItem`, and
  /// appends the tile body built from the prologue's struct memref and the
  /// getItem item. Returns false (kernel untouched) on any shape mismatch.
  bool rewriteDirectKernel(gpu::GPUFuncOp kernel);

  /// Shared tile-body emitter. \p structMemref must be a
  /// `memref<?x!llvm.struct<(i64, out_acc, r_acc, r_acc)>>`, \p item an
  /// `!sycl_item_2_`, and \p gpuModule the enclosing gpu.module (where the
  /// LDS tile globals live -- resolved by the caller BEFORE any destructive
  /// body rewrite, so a missing module leaves the function untouched).
  /// Emits at the builder's current insertion point; the caller adds the
  /// terminator. Returns false if \p structMemref's type does not match.
  bool emitTileBody(OpBuilder &b, gpu::GPUModuleOp gpuModule,
                    Value structMemref, Value item);
};

/// Validate the captured-struct element type: must be an LLVM struct with
/// exactly the gemm capture layout (i64 NK_ first, then the 3 accessors:
/// C rw, A r, B r).
static bool isGemmStruct(Type elemTy) {
  auto sTy = dyn_cast<LLVM::LLVMStructType>(elemTy);
  if (!sTy || sTy.getBody().size() != GEMM_STRUCT_MEMBERS)
    return false;
  auto i0 = dyn_cast<IntegerType>(sTy.getBody()[0]);
  return i0 && i0.getWidth() == 64;
}

/// The 16x16 f32 LDS tile type (SYCL local address space = AS 3).
static MemRefType localTileType(MLIRContext *ctx) {
  return MemRefType::get(
      {TS, TK}, FloatType::getF32(ctx), MemRefLayoutAttrInterface(),
      sycl::AccessAddrSpaceAttr::get(ctx, sycl::AccessAddrSpace::LocalAccess));
}

/// Return (creating on first use) a private module-scope LDS global named
/// \p name inside \p gpuModule. Same construction as LoopInternalization's
/// getWorkGroupSharedLocalMemory: phase-3 lowers it to an internal
/// addrspace(3) llvm.mlir.global, which the AMDGPU backend allocates into
/// static LDS (group_segment_fixed_size).
static memref::GlobalOp getOrCreateLocalTile(gpu::GPUModuleOp gpuModule,
                                             llvm::StringRef name) {
  if (auto existing =
          gpuModule.lookupSymbol<memref::GlobalOp>(name))
    return existing;
  OpBuilder globalBuilder(gpuModule->getRegion(0));
  return globalBuilder.create<memref::GlobalOp>(
      gpuModule->getLoc(), name,
      /*sym_visibility=*/globalBuilder.getStringAttr("private"),
      localTileType(globalBuilder.getContext()),
      /*initial_value=*/Attribute(), /*constant=*/false,
      /*alignment=*/IntegerAttr());
}

} // namespace

bool GemmLocalTilePass::rewriteBodyTile(func::FuncOp body) {
  MLIRContext *ctx = body.getContext();

  if (body.getNumArguments() != 2)
    return false;

  Value arg0 = body.getArgument(0); // captured struct memref
  Value item = body.getArgument(1); // !sycl_item_2_ (cell coordinate source)

  auto arg0MT = dyn_cast<MemRefType>(arg0.getType());
  if (!arg0MT || !isGemmStruct(arg0MT.getElementType()))
    return false;

  // The LDS tile globals must live at gpu.module scope; resolve the module
  // BEFORE the destructive body wipe so a missing module leaves the
  // function untouched.
  gpu::GPUModuleOp gpuModule = body->getParentOfType<gpu::GPUModuleOp>();
  if (!gpuModule)
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
  if (!emitTileBody(b, gpuModule, arg0, item))
    return false;
  b.create<func::ReturnOp>(body.getLoc());
  return true;
}

bool GemmLocalTilePass::rewriteDirectKernel(gpu::GPUFuncOp kernel) {
  Region &region = kernel.getBody();
  // The direct kernel is emitted by the frontend as a single entry block
  // (loop nests live in nested regions). Bail on anything else.
  if (region.getBlocks().size() != 1)
    return false;
  Block &entry = region.front();

  // Locate, in program order:
  //   - the reassembled captured-struct memref: pointer2memref result typed
  //     memref<?x!llvm.struct<(i64, c_rw_acc, a_r_acc, b_r_acc)>> (exactly
  //     the gemm capture layout, i64 first)
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
        if (isGemmStruct(mt.getElementType())) {
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

  // The LDS tile globals must live at gpu.module scope; resolve the module
  // BEFORE the destructive tail-erase so a missing module leaves the kernel
  // untouched.
  gpu::GPUModuleOp gpuModule = kernel->getParentOfType<gpu::GPUModuleOp>();
  if (!gpuModule)
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
  if (!emitTileBody(b, gpuModule, structMemref, item))
    return false;
  b.create<gpu::ReturnOp>(kernel.getLoc());
  return true;
}

bool GemmLocalTilePass::emitTileBody(OpBuilder &b, gpu::GPUModuleOp gpuModule,
                                     Value structMemref, Value item) {
  Location loc = structMemref.getLoc();
  MLIRContext *ctx = b.getContext();

  Type i32 = b.getI32Type();
  Type i64 = b.getIntegerType(64);
  Type indexTy = b.getIndexType();
  FloatType f32 = b.getF32Type();

  auto arg0MT = dyn_cast<MemRefType>(structMemref.getType());
  if (!arg0MT || !isGemmStruct(arg0MT.getElementType()))
    return false;
  ArrayRef<Type> members =
      cast<LLVM::LLVMStructType>(arg0MT.getElementType()).getBody();

  memref::GlobalOp aTileGlobal = getOrCreateLocalTile(gpuModule, TILE_A_KERNEL);
  memref::GlobalOp bTileGlobal = getOrCreateLocalTile(gpuModule, TILE_B_KERNEL);

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

  // --- constants ---
  Value c0i32 = b.create<arith::ConstantIntOp>(loc, 0, i32);
  Value c1i32 = b.create<arith::ConstantIntOp>(loc, 1, i32);
  Value c0idx = b.create<arith::ConstantIndexOp>(loc, 0);
  Value c1idx = b.create<arith::ConstantIndexOp>(loc, 1);
  Value c2idx = b.create<arith::ConstantIndexOp>(loc, 2);
  Value c3idx = b.create<arith::ConstantIndexOp>(loc, 3);
  Value c16idx = b.create<arith::ConstantIndexOp>(loc, TS);
  Value cM1idx = b.create<arith::ConstantIndexOp>(loc, -1); // for clamp (M-1)
  Value zero = b.create<arith::ConstantFloatOp>(loc, llvm::APFloat(0.0f), f32);
  Value alpha = b.create<arith::ConstantFloatOp>(loc, llvm::APFloat(ALPHA), f32);
  Value beta = b.create<arith::ConstantFloatOp>(loc, llvm::APFloat(BETA), f32);

  // --- captured struct members (NK_ bound, C read_write, A read, B read) ---
  // Verified from the cgeist IR dump of the baseline gemm lambda body:
  // member 2 (A) is subscripted at (i,k) and member 3 (B) at (k,j) -- the
  // aTile rows / bTile columns of the tiled form.
  Value mboundAcc = b.create<polygeist::SubIndexOp>(
      loc, MemRefType::get({ShapedType::kDynamic}, members[0]), structMemref, c0idx);
  Value outAcc = b.create<polygeist::SubIndexOp>(
      loc, MemRefType::get({ShapedType::kDynamic}, members[1]), structMemref, c1idx);
  Value aAcc = b.create<polygeist::SubIndexOp>(
      loc, MemRefType::get({ShapedType::kDynamic}, members[2]), structMemref, c2idx);
  Value bAcc = b.create<polygeist::SubIndexOp>(
      loc, MemRefType::get({ShapedType::kDynamic}, members[3]), structMemref, c3idx);
  Value Mraw = b.create<memref::LoadOp>(loc, i64, mboundAcc, ValueRange{c0idx});
  Value M = b.create<arith::IndexCastOp>(loc, indexTy, Mraw); // N == M == size
  Value Mm1 = b.create<arith::AddIOp>(loc, M, cM1idx); // M - 1 (clamp bound)

  // --- item -> cell coordinates (identical mapping to the baseline body) ---
  // irow/jcol ARE the output cell (one work-item per cell in the ceil16(N)
  // padded launch); ty/tx are the intra-tile coordinates (work-groups are
  // 16x16 via SYCL_FORCE_LOCAL_SIZE=16,16).
  Value itemMem = b.create<memref::AllocaOp>(loc, itemMemrefTy);
  b.create<memref::StoreOp>(loc, item, itemMem, ValueRange{c0idx});
  Value ii64 = b.create<sycl::SYCLItemGetIDOp>(loc, TypeRange{i64},
                                              ValueRange{itemMem, c0i32});
  Value ji64 = b.create<sycl::SYCLItemGetIDOp>(loc, TypeRange{i64},
                                               ValueRange{itemMem, c1i32});
  Value irow = b.create<arith::IndexCastOp>(loc, indexTy, ii64);
  Value jcol = b.create<arith::IndexCastOp>(loc, indexTy, ji64);
  Value ty = b.create<arith::RemUIOp>(loc, irow, c16idx);
  Value tx = b.create<arith::RemUIOp>(loc, jcol, c16idx);

  // --- LDS tile handles ---
  MemRefType tileTy = localTileType(ctx);
  Value aTile = b.create<memref::GetGlobalOp>(
      loc, tileTy, FlatSymbolRefAttr::get(ctx, aTileGlobal.getName()));
  Value bTile = b.create<memref::GetGlobalOp>(
      loc, tileTy, FlatSymbolRefAttr::get(ctx, bTileGlobal.getName()));

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
  auto storeOut = [&](OpBuilder &bb, Value v, Value r, Value c) {
    Value idm = mkId(bb, r, c);
    Value sub = bb.create<sycl::SYCLAccessorSubscriptOp>(
        loc, TypeRange{f32DynTy}, ValueRange{outAcc, idm});
    bb.create<memref::StoreOp>(loc, v, sub, ValueRange{c0idx});
  };
  auto clampIdx = [&](OpBuilder &bb, Value x) -> Value {
    // min(x, M-1): safe index for padded out-of-range work-items (their
    // loads feed discarded lanes; their stores are separately predicated).
    return bb.create<arith::MinSIOp>(loc, x, Mm1);
  };
  // Work-group barrier: rocdl.barrier lowers to fence(workgroup) +
  // llvm.amdgcn.s.barrier. Uniform execution is guaranteed: the loop bounds
  // (M, then constant 16) are uniform across the work-group.
  auto barrier = [&](OpBuilder &bb) {
    bb.create<ROCDL::BarrierOp>(loc);
  };

  // --- seed: the baseline is `C[item] *= BETA` then `C[item] += ...`, so
  // the accumulator starts at the existing C (irow, jcol) scaled by BETA
  // (the read_write C carries the init() values).
  Value seed = loadData(b, outAcc, clampIdx(b, irow), clampIdx(b, jcol));
  Value acc0 = b.create<arith::MulFOp>(loc, seed, beta);

  // --- k-chunk loop: for kt = 0..M step 16, acc carried as an SSA iter_arg
  // (a register; there is no accumulator memref in this design at all).
  Value clampedIrow = clampIdx(b, irow);
  Value clampedJcol = clampIdx(b, jcol);
  scf::ForOp ktLoop = b.create<scf::ForOp>(
      loc, c0idx, M, c16idx, ValueRange{acc0},
      [&](OpBuilder &kb, Location kl, Value kt, ValueRange accArgs) {
        // Cooperative loads: ONE element of each tile per work-item
        // (256 items fill each 16x16 tile); consecutive tx -> consecutive
        // global columns -> coalesced (gemm_opt's submitTiledGemm).
        //
        // Tail-chunk correctness (N % 16 != 0): lanes with kt+tx >= M (or
        // kt+ty >= M) would read the LAST row/column again under the clamp
        // and double-count it in every tile column (tx >= kt-tail). That is
        // a latent bug in gemm_opt itself (never run at N % 16 != 0).
        // Instead, out-of-range k lanes STORE 0.0 (branch-free select; the
        // clamped load still executes safely) so they contribute 0*0 to the
        // FMA.
        Value ktTx = kb.create<arith::AddIOp>(kl, kt, tx);
        Value ktxIn = kb.create<arith::CmpIOp>(kl, arith::CmpIPredicate::ult,
                                               ktTx, M);
        Value aLoaded = loadData(kb, aAcc, clampedIrow, clampIdx(kb, ktTx));
        Value aVal =
            kb.create<arith::SelectOp>(kl, ktxIn, aLoaded, zero);
        kb.create<memref::StoreOp>(kl, aVal, aTile, ValueRange{ty, tx});
        Value ktTy = kb.create<arith::AddIOp>(kl, kt, ty);
        Value ktyIn = kb.create<arith::CmpIOp>(kl, arith::CmpIPredicate::ult,
                                               ktTy, M);
        Value bLoaded = loadData(kb, bAcc, clampIdx(kb, ktTy), clampedJcol);
        Value bVal =
            kb.create<arith::SelectOp>(kl, ktyIn, bLoaded, zero);
        kb.create<memref::StoreOp>(kl, bVal, bTile, ValueRange{ty, tx});
        barrier(kb);

        // Micro-kernel: 16 ALPHA-scaled FMAs from the staged tiles
        // (acc += ALPHA * aTile[ty,kk] * bTile[kk,tx]; the product is
        // multiplied out and ALPHA rides the FMA addend, keeping one FMA
        // per k-lane exactly like the baseline's
        // `C += ALPHA * A[i,k] * B[k,j]`).
        Value kkAcc = accArgs[0];
        Value kkAccFinal = kb.create<scf::ForOp>(
            kl, c0idx, c16idx, c1idx, ValueRange{kkAcc},
            [&](OpBuilder &ib, Location il, Value kk, ValueRange iArgs) {
              Value av = ib.create<memref::LoadOp>(il, f32, aTile,
                                                   ValueRange{ty, kk});
              Value bv = ib.create<memref::LoadOp>(il, f32, bTile,
                                                   ValueRange{kk, tx});
              Value scaled =
                  ib.create<arith::MulFOp>(il, av, bv);
              Value nv = ib.create<math::FmaOp>(il, scaled, alpha, iArgs[0]);
              ib.create<scf::YieldOp>(il, ValueRange{nv});
            })->getResult(0);

        // Second barrier: the next iteration's cooperative loads must not
        // overwrite the tiles until every item finished the micro-kernel
        // (gemm_opt's second item.barrier).
        barrier(kb);
        kb.create<scf::YieldOp>(kl, ValueRange{kkAccFinal});
      });
  Value acc = ktLoop->getResult(0);

  // --- final predicated store (padded work-items exist for N % 16 != 0) ---
  Value iIn = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult,
                                      irow, M);
  Value jIn = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult,
                                      jcol, M);
  Value inRange = b.create<arith::AndIOp>(loc, iIn, jIn);
  b.create<scf::IfOp>(
      loc, inRange,
      [&](OpBuilder &ib, Location il) {
        storeOut(ib, acc, irow, jcol);
        ib.create<scf::YieldOp>(il);
      });

  return true;
}

void GemmLocalTilePass::runOnOperation() {
  ModuleOp module = getOperation();

  if (!targetsAmdgcnSyclmlir(module)) {
    LLVM_DEBUG(llvm::dbgs() << "GemmLocalTile: target is not "
                            << AMDGCN_SYCLMLIR_TRIPLE << ", skipping.\n");
    return;
  }

  // Needed to construct rocdl.barrier (idempotent; already loaded in cgeist
  // and in tools that register all dialects).
  getContext().loadDialect<ROCDL::ROCDLDialect>();

  // Rewrite the gemm lambda body to the LDS-tile form (each work-item
  // computes the single output cell at its item id, staged through the
  // 16x16 LDS tiles). The paired host-side launch rewrite (ceil16(N)
  // padding) is the LLVM pass SYCLRewriteGemmRange
  // (-fsycl-rewrite-gemm-range); the (16,16) work-group shape is supplied
  // at launch time by the SYCL runtime env var SYCL_FORCE_LOCAL_SIZE=16,16.
  //
  // Rewrite site 1 -- the top-level lambda body func.func (the WRAPPER
  // path: __pf_kernel_wrapper -> RoundedRangeIDGenerator loop -> this
  // function; only runs when range rounding fires, which ceil16 values
  // never trigger).
  SmallVector<func::FuncOp, 4> gemmBodies;
  module.walk([&](func::FuncOp func) {
    if (isGemmBody(func))
      gemmBodies.push_back(func);
  });

  for (func::FuncOp body : gemmBodies) {
    ++NumDetected;
    LLVM_DEBUG(llvm::dbgs() << "GemmLocalTile: detected body " << body.getName()
                            << " (" << body.getNumArguments() << " args)\n");
    if (rewriteBodyTile(body))
      ++NumRewritten;
  }

  // Rewrite site 2 -- the DIRECT kernel gpu.func @_ZTS4Gemm (body born
  // inlined from the frontend; the path that always runs). Without this,
  // the host pass pads the launch while the direct kernel still runs the
  // baseline single-cell body -- an entirely untiled kernel.
  SmallVector<gpu::GPUFuncOp, 4> directKernels;
  module.walk([&](gpu::GPUFuncOp kernel) {
    if (kernel.getName() == GEMM_DIRECT_KERNEL)
      directKernels.push_back(kernel);
  });

  for (gpu::GPUFuncOp kernel : directKernels) {
    ++NumDetected;
    LLVM_DEBUG(llvm::dbgs() << "GemmLocalTile: detected direct kernel "
                            << kernel.getName() << "\n");
    if (rewriteDirectKernel(kernel))
      ++NumRewritten;
  }
}

std::unique_ptr<Pass> mlir::sycl::createGemmLocalTilePass() {
  return std::make_unique<GemmLocalTilePass>();
}
