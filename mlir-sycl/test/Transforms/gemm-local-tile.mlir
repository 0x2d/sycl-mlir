// RUN: sycl-mlir-opt %s -split-input-file -sycl-gemm-local-tile \
// RUN:     -mlir-pass-statistics 2>&1 | FileCheck %s

// COM: GemmLocalTilePass lit coverage.
// COM:
// COM: The pass rewrites the ONE gemm kernel at TWO sites: (1) the
// COM: top-level lambda-body `func.func` (the WRAPPER path:
// COM: `gpu.func @__pf_kernel_wrapperI4GemmE` calls it per enumerated user
// COM: item from the RoundedRangeIDGenerator loop), and (2) the DIRECT
// COM: kernel `gpu.func @_ZTS4Gemm` whose body is born INLINED from the
// COM: frontend (the path that always runs for the ceil16(N) launch:
// COM: ceil16 values never trigger range rounding). The full rewrite
// COM: targets the real gemm lambda body, whose captured-struct argument
// COM: is typed
// COM:   `memref<?x!llvm.struct<(i64, !sycl_accessor_2_f32_rw_dev,
// COM:                            !sycl_accessor_2_f32_r_dev,
// COM:                            !sycl_accessor_2_f32_r_dev)>>`
// COM: (sycl.kernel_func_obj = [@_ZTS4Gemm, ...] anchor; FOUR members --
// COM: NK_ bound, C rw accessor, A r accessor, B r accessor).
// COM: sycl-mlir-opt's asm parser rejects `memref<?x!llvm.struct<...>>` as
// COM: a memref element type ("invalid memref element type") -- a parser
// COM: limitation, not a pass bug; the production cgeist/clang pipeline
// COM: consumes that exact type and the full -fsycl build links cleanly. The
// COM: full rewrite (two private memref.global 16x16 f32 LDS tiles in the
// COM: enclosing gpu.module, cooperative clamped loads with k-tail
// COM: zero-fill select, rocdl.barrier pair per k-chunk, a step-16 scf.for
// COM: carrying the scalar f32 accumulator as an iter_arg seeded from
// COM: C*BETA, a 16-FMA inner loop with ALPHA riding the FMA addend, and
// COM: the final predicated store) is verified out-of-tree via the cgeist
// COM: IR dump (`-Xcgeist --sycl-gemm-local-tile`); the paired host-side
// COM: launch rewrite is the LLVM pass SYCLRewriteGemmRange
// COM: (-fsycl-rewrite-gemm-range) and the (16,16) work-group shape comes
// COM: from the runtime env SYCL_FORCE_LOCAL_SIZE.
// COM:
// COM: What this lit test DOES lock in (all parseable): the EXACT-match
// COM: detection anchor (the gemm_opt kernel `_ZTS16Polybench_Gemm1`
// COM: contains "Gemm" and must NOT match), gate #2 (the
// COM: amdgcn-amd-amdhsa-syclmlir triple guard), plus the
// COM: signature-validation guard inside rewriteBodyTile (a detected body
// COM: whose arg0 is NOT the captured struct is left untouched --
// COM: num-rewritten stays 0 -- rather than half-erased). Here arg0 is a
// COM: plain `memref<?xi64>` stand-in so the module parses; detection
// COM: fires on the anchor, then the struct-shape check returns false and
// COM: the body is preserved verbatim.
// COM:
// COM: NOTE on check order: with -split-input-file, sycl-mlir-opt prints
// COM: the per-chunk pass statistics (stderr) BEFORE the per-chunk modules
// COM: (stdout), so the merged output is: stats(chunk1), stats(chunk2),
// COM: stats(chunk3), stats(chunk4), module(chunk1), module(chunk2),
// COM: module(chunk3), module(chunk4). The CHECKs below follow that order:
// COM: all four stats blocks first, then the four module bodies.

// COM: chunk 1 (gemm, amdgcn-amd-amdhsa-syclmlir): gate #2 passes; the
// COM: _ZTS4Gemm anchor is detected; the non-struct arg0 trips the
// COM: validation guard so no rewrite occurs.
!sycl_array_2_ = !sycl.array<[2], (memref<2xi64>)>
!sycl_id_2_ = !sycl.id<[2], (!sycl_array_2_)>
!sycl_range_2_ = !sycl.range<[2], (!sycl_array_2_)>
!sycl_accessor_impl_device_2_ = !sycl.accessor_impl_device<[2], (!sycl_id_2_, !sycl_range_2_, !sycl_range_2_)>
!sycl_item_base_2_ = !sycl.item_base<[2, true], (!sycl_range_2_, !sycl_id_2_, !sycl_id_2_)>
!sycl_item_2_ = !sycl.item<[2, true], (!sycl_item_base_2_)>

module attributes {llvm.target_triple = "amdgcn-amd-amdhsa-syclmlir"} {
  func.func private @gemm_body(%arg0: memref<?xi64>, %arg1: !sycl_item_2_) attributes {sycl.kernel_func_obj = [@_ZTS4Gemm]} {
    return
  }
}

// -----

// COM: chunk 2 (negative anchor, amdgcn-amd-amdhsa-syclmlir): the
// COM: gemm_opt kernel `_ZTS16Polybench_Gemm1` CONTAINS "Gemm" but is NOT
// COM: `_ZTS4Gemm` -- the exact-match anchor must reject it (a bare
// COM: contains("Gemm") would misfire and rewrite the unrelated source-opt
// COM: kernel when both TUs are linked).
!sycl_array_2_ = !sycl.array<[2], (memref<2xi64>)>
!sycl_id_2_ = !sycl.id<[2], (!sycl_array_2_)>
!sycl_range_2_ = !sycl.range<[2], (!sycl_array_2_)>
!sycl_accessor_impl_device_2_ = !sycl.accessor_impl_device<[2], (!sycl_id_2_, !sycl_range_2_, !sycl_range_2_)>
!sycl_item_base_2_ = !sycl.item_base<[2, true], (!sycl_range_2_, !sycl_id_2_, !sycl_id_2_)>
!sycl_item_2_ = !sycl.item<[2, true], (!sycl_item_base_2_)>

module attributes {llvm.target_triple = "amdgcn-amd-amdhsa-syclmlir"} {
  func.func private @gemm_opt_body(%arg0: memref<?xi64>, %arg1: !sycl_item_2_) attributes {sycl.kernel_func_obj = [@_ZTS16Polybench_Gemm1]} {
    return
  }
}

// -----

// COM: chunk 3 (negative, spir64 target): gate #2 fails, the pass is a
// COM: no-op -- nothing is detected and the body is preserved.
!sycl_array_2_ = !sycl.array<[2], (memref<2xi64>)>
!sycl_id_2_ = !sycl.id<[2], (!sycl_array_2_)>
!sycl_range_2_ = !sycl.range<[2], (!sycl_array_2_)>
!sycl_accessor_impl_device_2_ = !sycl.accessor_impl_device<[2], (!sycl_id_2_, !sycl_range_2_, !sycl_range_2_)>
!sycl_item_base_2_ = !sycl.item_base<[2, true], (!sycl_range_2_, !sycl_id_2_, !sycl_id_2_)>
!sycl_item_2_ = !sycl.item<[2, true], (!sycl_item_base_2_)>

module attributes {llvm.target_triple = "spir64-unknown-unknown-syclmlir"} {
  func.func private @gemm_body_neg(%arg0: memref<?xi64>, %arg1: !sycl_item_2_) attributes {sycl.kernel_func_obj = [@_ZTS4Gemm]} {
    return
  }
}

// CHECK: (S) 1 num-detected
// CHECK: (S) 0 num-rewritten
// CHECK: (S) 0 num-detected
// CHECK: (S) 0 num-rewritten
// CHECK: (S) 0 num-detected
// CHECK: (S) 0 num-rewritten

// CHECK-LABEL: func.func private @gemm_body
// COM: Body preserved verbatim (validation guard: arg0 is not the captured struct).
// CHECK: return
// CHECK-LABEL: func.func private @gemm_opt_body
// CHECK: return
// CHECK-LABEL: func.func private @gemm_body_neg
// CHECK: return
