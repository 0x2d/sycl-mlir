// RUN: sycl-mlir-opt -convert-sycl-to-rocdl %s | FileCheck %s

!sycl_id_2_ = !sycl.id<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>

module attributes {gpu.container_module} {
  gpu.module @kernels {
    // HIP/AMDGCN has no kernel-launch global offset, so the result is filled
    // with zeros. No ROCDL intrinsic must be emitted.
    // CHECK-LABEL: func.func @test_global_offset_2
    // CHECK:         memref.alloca
    // CHECK:         %[[ZERO:.*]] = arith.constant 0 : i64
    // CHECK-NOT:     rocdl.
    // CHECK:         memref.store %[[ZERO]]
    // CHECK:         memref.store %[[ZERO]]
    func.func @test_global_offset_2() -> !sycl_id_2_ {
      %0 = sycl.global_offset : !sycl_id_2_
      return %0 : !sycl_id_2_
    }
  }
}
