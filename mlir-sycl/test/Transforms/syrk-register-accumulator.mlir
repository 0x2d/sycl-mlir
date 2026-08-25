// RUN: sycl-mlir-opt %s -split-input-file -sycl-syrk-register-accumulator \
// RUN:     -mlir-pass-statistics 2>&1 | FileCheck %s

// COM: Stage-1 SyrkRegisterAccumulatorPass lit coverage.
// COM:
// COM: The pass's full rewrite targets the real Syr2k2 lambda-body `func.func`,
// COM: whose captured-struct argument is typed
// COM:   `memref<?x!llvm.struct<(i64, !sycl_accessor_2_f32_rw_dev,
// COM:                            !sycl_accessor_2_f32_r_dev)>>`
// COM: (see polybench/syrk.cpp post-cgeist IR, func at the
// COM: `sycl.kernel_func_obj = [@_ZTS6Syr2k2, ...]` anchor). sycl-mlir-opt's
// COM: asm parser rejects `memref<?x!llvm.struct<...>>` as a memref element
// COM: type ("invalid memref element type") -- a parser limitation, not a pass
// COM: bug: the production cgeist/clang pipeline consumes that exact type and
// COM: the full -fsycl build links cleanly. The full rewrite is therefore
// COM: verified out-of-tree via the cgeist IR dump (`-Xcgeist
// COM: --sycl-syrk-register-accumulator` => hoisted item.get_id, single
// COM: scf.for k carrying one f32 iter_arg, math.fma, single affine.store) and
// COM: the gfx906 GPU run (`Verify: PASS`).
// COM:
// COM: What this lit test DOES lock in (all parseable): the Syr2k2 detection
// COM: anchor, gate #2 (the amdgcn-amd-amdhsa-syclmlir triple guard), and the
// COM: signature-validation guard inside rewriteBodyAccumulate (a detected body
// COM: whose arg0 is NOT the captured struct is left untouched -- num-rewritten
// COM: stays 0 -- rather than half-erased). Here arg0 is a plain `memref<?xi64>`
// COM: stand-in so the module parses; detection fires on the anchor, then the
// COM: struct-shape check returns false and the body is preserved verbatim.

// COM: Positive chunk: amdgcn-amd-amdhsa-syclmlir => gate #2 passes. The
// COM: Syr2k2 anchor is detected (num-detected = 1); the non-struct arg0 trips
// COM: the validation guard so no rewrite occurs (num-rewritten = 0) and the
// COM: body is preserved.
!sycl_array_2_ = !sycl.array<[2], (memref<2xi64>)>
!sycl_id_2_ = !sycl.id<[2], (!sycl_array_2_)>
!sycl_range_2_ = !sycl.range<[2], (!sycl_array_2_)>
!sycl_accessor_impl_device_2_ = !sycl.accessor_impl_device<[2], (!sycl_id_2_, !sycl_range_2_, !sycl_range_2_)>
!sycl_item_base_2_ = !sycl.item_base<[2, true], (!sycl_range_2_, !sycl_id_2_, !sycl_id_2_)>
!sycl_item_2_ = !sycl.item<[2, true], (!sycl_item_base_2_)>

module attributes {llvm.target_triple = "amdgcn-amd-amdhsa-syclmlir"} {
  func.func private @syrk_body(%arg0: memref<?xi64>, %arg1: !sycl_item_2_) attributes {sycl.kernel_func_obj = [@_ZTS6Syr2k2]} {
    return
  }
}

// CHECK: (S) 1 num-detected
// CHECK: (S) 0 num-rewritten
// CHECK-LABEL: func.func private @syrk_body
// COM: Body preserved verbatim (validation guard: arg0 is not the captured struct).
// CHECK: return

// -----

// COM: Negative chunk: spir64 target => gate #2 fails, the pass is a no-op.
// COM: Nothing is detected; the body is preserved.
!sycl_array_2_ = !sycl.array<[2], (memref<2xi64>)>
!sycl_id_2_ = !sycl.id<[2], (!sycl_array_2_)>
!sycl_range_2_ = !sycl.range<[2], (!sycl_array_2_)>
!sycl_accessor_impl_device_2_ = !sycl.accessor_impl_device<[2], (!sycl_id_2_, !sycl_range_2_, !sycl_range_2_)>
!sycl_item_base_2_ = !sycl.item_base<[2, true], (!sycl_range_2_, !sycl_id_2_, !sycl_id_2_)>
!sycl_item_2_ = !sycl.item<[2, true], (!sycl_item_base_2_)>

module attributes {llvm.target_triple = "spir64-unknown-unknown-syclmlir"} {
  func.func private @syrk_body(%arg0: memref<?xi64>, %arg1: !sycl_item_2_) attributes {sycl.kernel_func_obj = [@_ZTS6Syr2k2]} {
    return
  }
}
// COM: On spir64 the pass early-returns before walking, so it reports no
// COM: detections and leaves the body untouched.
// CHECK-LABEL: func.func private @syrk_body
// CHECK: return
