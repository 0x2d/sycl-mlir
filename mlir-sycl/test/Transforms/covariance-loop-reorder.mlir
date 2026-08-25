// RUN: sycl-mlir-opt %s -split-input-file -sycl-covariance-loop-reorder \
// RUN:     -mlir-pass-statistics 2>&1 | FileCheck %s

// COM: Stage-1 CovarianceLoopReorderPass: rewrites the real 6-arg baseline
// COM: CovarianceCovar body (M, N, symmat dw, data r, symmat2 dw, item<1>) in
// COM: place into a register-blocked loop reorder (outer j2-block loop of width
// COM: tile-size=16 carrying 16 f32 accumulators, inner i loop with the
// COM: column-j1 load hoisted out of the j2 unroll). Both gates are exercised:
// COM: the amdgcn chunk acts; the spir64 chunk (same anchor) is gated to a
// COM: no-op. Type aliases are taken verbatim from the cgeist-lowered
// COM: `covariance.cpp` IR (known to verify); repeated at the top of each
// COM: chunk because `-split-input-file` parses each chunk independently.

// COM: Positive chunk: amdgcn-amd-amdhsa-syclmlir target => both gates pass.
// COM: The 6-arg baseline body => detected AND rewritten. The rewritten body
// COM: must carry the reorder shape: sycl.item.get_id (j1), outer scf.for
// COM: (j2-block), inner scf.for (i, 16 f32 iter_args), the hoisted
// COM: sycl.accessor.subscript data load, math.fma, arith.select (tail guard),
// COM: and the upper + mirror (symmat2) stores.
!sycl_array_1_ = !sycl.array<[1], (memref<1xi64>)>
!sycl_array_2_ = !sycl.array<[2], (memref<2xi64>)>
!sycl_id_1_ = !sycl.id<[1], (!sycl_array_1_)>
!sycl_id_2_ = !sycl.id<[2], (!sycl_array_2_)>
!sycl_range_1_ = !sycl.range<[1], (!sycl_array_1_)>
!sycl_range_2_ = !sycl.range<[2], (!sycl_array_2_)>
!sycl_accessor_impl_device_2_ = !sycl.accessor_impl_device<[2], (!sycl_id_2_, !sycl_range_2_, !sycl_range_2_)>
!sycl_item_base_1_ = !sycl.item_base<[1, true], (!sycl_range_1_, !sycl_id_1_, !sycl_id_1_)>
!sycl_accessor_2_f32_dw_dev = !sycl.accessor<[2, f32, discard_write, device], (!sycl_accessor_impl_device_2_, !llvm.struct<(memref<?xf32, 1>)>)>
!sycl_accessor_2_f32_r_dev = !sycl.accessor<[2, f32, read, device], (!sycl_accessor_impl_device_2_, !llvm.struct<(memref<?xf32, 1>)>)>
!sycl_item_1_ = !sycl.item<[1, true], (!sycl_item_base_1_)>

module attributes {llvm.target_triple = "amdgcn-amd-amdhsa-syclmlir"} {
  func.func private @covar_body(%arg0: memref<?xi64>, %arg1: memref<?xi64>, %arg2: memref<?x!sycl_accessor_2_f32_dw_dev>, %arg3: memref<?x!sycl_accessor_2_f32_r_dev>, %arg4: memref<?x!sycl_accessor_2_f32_dw_dev>, %arg5: !sycl_item_1_) attributes {sycl.kernel_func_obj = [@_ZTS15CovarianceCovar]} {
    return
  }
}

// COM: Statistics print once per split chunk. The positive chunk reports
// COM: 1 detected / 1 rewritten; the spir64 chunk reports 0 / 0 (gate #2).
// CHECK: (S) 1 num-detected
// CHECK: (S) 1 num-rewritten

// COM: Rewritten body: j1 from sycl.item.get_id; outer j2-block scf.for; inner
// COM: i scf.for carrying 16 f32 accumulators; hoisted data load; FMA; select;
// COM: upper + mirror stores.
// CHECK-LABEL: func.func private @covar_body
// CHECK: sycl.item.get_id
// COM: Outer j2-block loop (step 16).
// CHECK: scf.for
// COM: Inner i loop carrying the f32 accumulators (16 iter_args).
// CHECK: scf.for
// COM: Hoisted column-j1 data load (one sycl.accessor.subscript per i, outside
// COM: the jj unroll) and the per-jj data load.
// CHECK: sycl.accessor.subscript
// CHECK: sycl.accessor.subscript
// CHECK: math.fma
// CHECK: arith.select
// COM: Tail guard (j2 <= M) and the upper + transposed-mirror stores.
// CHECK: arith.cmpi
// CHECK: scf.if
// CHECK: sycl.accessor.subscript
// CHECK: sycl.accessor.subscript

// -----

// COM: Negative chunk: spir64 target => gate #2 fails, pass no-ops; the body is
// COM: NOT counted and is preserved verbatim.
!sycl_array_1_ = !sycl.array<[1], (memref<1xi64>)>
!sycl_array_2_ = !sycl.array<[2], (memref<2xi64>)>
!sycl_id_1_ = !sycl.id<[1], (!sycl_array_1_)>
!sycl_id_2_ = !sycl.id<[2], (!sycl_array_2_)>
!sycl_range_1_ = !sycl.range<[1], (!sycl_array_1_)>
!sycl_range_2_ = !sycl.range<[2], (!sycl_array_2_)>
!sycl_accessor_impl_device_2_ = !sycl.accessor_impl_device<[2], (!sycl_id_2_, !sycl_range_2_, !sycl_range_2_)>
!sycl_item_base_1_ = !sycl.item_base<[1, true], (!sycl_range_1_, !sycl_id_1_, !sycl_id_1_)>
!sycl_accessor_2_f32_dw_dev = !sycl.accessor<[2, f32, discard_write, device], (!sycl_accessor_impl_device_2_, !llvm.struct<(memref<?xf32, 1>)>)>
!sycl_accessor_2_f32_r_dev = !sycl.accessor<[2, f32, read, device], (!sycl_accessor_impl_device_2_, !llvm.struct<(memref<?xf32, 1>)>)>
!sycl_item_1_ = !sycl.item<[1, true], (!sycl_item_base_1_)>

module attributes {llvm.target_triple = "spir64-unknown-unknown-syclmlir"} {
  func.func private @covar_body(%arg0: memref<?xi64>, %arg1: memref<?xi64>, %arg2: memref<?x!sycl_accessor_2_f32_dw_dev>, %arg3: memref<?x!sycl_accessor_2_f32_r_dev>, %arg4: memref<?x!sycl_accessor_2_f32_dw_dev>, %arg5: !sycl_item_1_) attributes {sycl.kernel_func_obj = [@_ZTS15CovarianceCovar]} {
    return
  }
}
// CHECK-LABEL: func.func private @covar_body
// CHECK: return
