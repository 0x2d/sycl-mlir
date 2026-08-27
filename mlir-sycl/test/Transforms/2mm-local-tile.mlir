// RUN: sycl-mlir-opt %s -split-input-file -sycl-2mm-local-tile \
// RUN:     -mlir-pass-statistics 2>&1 | FileCheck %s

// COM: TwoMmLocalTilePass lit coverage.
// COM:
// COM: The pass rewrites BOTH 2mm kernels, each at TWO sites: (1) the
// COM: top-level lambda-body `func.func` (the WRAPPER path:
// COM: `gpu.func @__pf_kernel_wrapperI15Polybench_2mm_NEE` calls it per
// COM: enumerated user item from the RoundedRangeIDGenerator loop), and (2)
// COM: the DIRECT kernel `gpu.func @_ZTS15Polybench_2mm_{1,2}` whose body is
// COM: born INLINED from the frontend (the path that always runs for the
// COM: ceil16(N) launch: ceil16 values never trigger range rounding). The
// COM: full rewrite targets the real 2mm lambda bodies, whose captured-struct
// COM: argument is typed
// COM:   `memref<?x!llvm.struct<(i64, !sycl_accessor_2_f32_rw_dev,
// COM:                            !sycl_accessor_2_f32_r_dev,
// COM:                            !sycl_accessor_2_f32_r_dev)>>`
// COM: (kernel 1; kernel 2's member 1 is !sycl_accessor_2_f32_dw_dev)
// COM: (sycl.kernel_func_obj = [@_ZTS15Polybench_2mm_1, ...] anchor; FOUR
// COM: members -- size_ bound, OUT accessor, IN1 read, IN2 read).
// COM: sycl-mlir-opt's asm parser rejects `memref<?x!llvm.struct<...>>` as a
// COM: memref element type ("invalid memref element type") -- a parser
// COM: limitation, not a pass bug; the production cgeist/clang pipeline
// COM: consumes that exact type and the full -fsycl build links cleanly. The
// COM: full rewrite (two private memref.global 16x16 f32 LDS tiles in the
// COM: enclosing gpu.module, cooperative clamped loads, rocdl.barrier pair
// COM: per k-chunk, a step-16 scf.for carrying the scalar f32 accumulator as
// COM: an iter_arg, a 16-FMA inner loop, and the final predicated store --
// COM: kernel 1 seeded from C, kernel 2 seeded from zero) is verified
// COM: out-of-tree via the cgeist IR dump (`-Xcgeist --sycl-2mm-local-tile`);
// COM: the paired host-side launch rewrite is the LLVM pass
// COM: SYCLRewrite2mmRange (-fsycl-rewrite-2mm-range) and the (16,16)
// COM: work-group shape comes from the runtime env SYCL_FORCE_LOCAL_SIZE.
// COM:
// COM: What this lit test DOES lock in (all parseable): the detection
// COM: anchors for BOTH kernels and gate #2 (the amdgcn-amd-amdhsa-syclmlir
// COM: triple guard), plus the signature-validation guard inside
// COM: rewriteBodyTile (a detected body whose arg0 is NOT the captured struct
// COM: is left untouched -- num-rewritten stays 0 -- rather than half-erased).
// COM: Here arg0 is a plain `memref<?xi64>` stand-in so the module parses;
// COM: detection fires on the anchor, then the struct-shape check returns
// COM: false and the body is preserved verbatim.
// COM:
// COM: NOTE on check order: with -split-input-file, sycl-mlir-opt prints the
// COM: per-chunk pass statistics (stderr) BEFORE the per-chunk modules
// COM: (stdout), so the merged output is: stats(chunk1), stats(chunk2),
// COM: stats(chunk3), module(chunk1), module(chunk2), module(chunk3). The
// COM: CHECKs below follow that order: all three stats blocks first, then
// COM: the three module bodies.

// COM: chunk 1 (kernel 1, amdgcn-amd-amdhsa-syclmlir): gate #2 passes; the
// COM: Polybench_2mm_1 anchor is detected; the non-struct arg0 trips the
// COM: validation guard so no rewrite occurs.
!sycl_array_2_ = !sycl.array<[2], (memref<2xi64>)>
!sycl_id_2_ = !sycl.id<[2], (!sycl_array_2_)>
!sycl_range_2_ = !sycl.range<[2], (!sycl_array_2_)>
!sycl_accessor_impl_device_2_ = !sycl.accessor_impl_device<[2], (!sycl_id_2_, !sycl_range_2_, !sycl_range_2_)>
!sycl_item_base_2_ = !sycl.item_base<[2, true], (!sycl_range_2_, !sycl_id_2_, !sycl_id_2_)>
!sycl_item_2_ = !sycl.item<[2, true], (!sycl_item_base_2_)>

module attributes {llvm.target_triple = "amdgcn-amd-amdhsa-syclmlir"} {
  func.func private @twomm_body_1(%arg0: memref<?xi64>, %arg1: !sycl_item_2_) attributes {sycl.kernel_func_obj = [@_ZTS15Polybench_2mm_1]} {
    return
  }
}

// -----

// COM: chunk 2 (kernel 2, amdgcn-amd-amdhsa-syclmlir): the Polybench_2mm_2
// COM: anchor is detected independently of kernel 1's.
!sycl_array_2_ = !sycl.array<[2], (memref<2xi64>)>
!sycl_id_2_ = !sycl.id<[2], (!sycl_array_2_)>
!sycl_range_2_ = !sycl.range<[2], (!sycl_array_2_)>
!sycl_accessor_impl_device_2_ = !sycl.accessor_impl_device<[2], (!sycl_id_2_, !sycl_range_2_, !sycl_range_2_)>
!sycl_item_base_2_ = !sycl.item_base<[2, true], (!sycl_range_2_, !sycl_id_2_, !sycl_id_2_)>
!sycl_item_2_ = !sycl.item<[2, true], (!sycl_item_base_2_)>

module attributes {llvm.target_triple = "amdgcn-amd-amdhsa-syclmlir"} {
  func.func private @twomm_body_2(%arg0: memref<?xi64>, %arg1: !sycl_item_2_) attributes {sycl.kernel_func_obj = [@_ZTS15Polybench_2mm_2]} {
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
  func.func private @twomm_body_neg(%arg0: memref<?xi64>, %arg1: !sycl_item_2_) attributes {sycl.kernel_func_obj = [@_ZTS15Polybench_2mm_1]} {
    return
  }
}

// CHECK: (S) 1 num-detected
// CHECK: (S) 0 num-rewritten
// CHECK: (S) 1 num-detected
// CHECK: (S) 0 num-rewritten
// CHECK: (S) 0 num-detected
// CHECK: (S) 0 num-rewritten

// CHECK-LABEL: func.func private @twomm_body_1
// COM: Body preserved verbatim (validation guard: arg0 is not the captured struct).
// CHECK: return
// CHECK-LABEL: func.func private @twomm_body_2
// CHECK: return
// CHECK-LABEL: func.func private @twomm_body_neg
// CHECK: return
