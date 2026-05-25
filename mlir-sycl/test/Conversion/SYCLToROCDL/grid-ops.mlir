// RUN: sycl-mlir-opt -convert-sycl-to-rocdl %s | FileCheck %s

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_id_2_ = !sycl.id<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>
!sycl_id_3_ = !sycl.id<[3], (!sycl.array<[3], (memref<3xi64, 4>)>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_range_2_ = !sycl.range<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>
!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<3xi64, 4>)>)>

module attributes {gpu.container_module} {
  gpu.module @kernels {
    // CHECK-LABEL: func.func @test_local_id_1
    // CHECK:         rocdl.workitem.id.x
    // CHECK-NOT:     rocdl.workitem.id.y
    // CHECK-NOT:     rocdl.workitem.id.z
    func.func @test_local_id_1() -> !sycl_id_1_ {
      %0 = sycl.local_id : !sycl_id_1_
      return %0 : !sycl_id_1_
    }

    // SYCL dim 0 is slowest-varying; AMDGCN X is fastest-varying. The mirror
    // therefore writes dim 0 from rocdl.*.z and dim N-1 from rocdl.*.x.
    // CHECK-LABEL: func.func @test_local_id_3
    // CHECK:         rocdl.workitem.id.z
    // CHECK:         rocdl.workitem.id.y
    // CHECK:         rocdl.workitem.id.x
    func.func @test_local_id_3() -> !sycl_id_3_ {
      %0 = sycl.local_id : !sycl_id_3_
      return %0 : !sycl_id_3_
    }

    // CHECK-LABEL: func.func @test_work_group_id_2
    // CHECK:         rocdl.workgroup.id.y
    // CHECK:         rocdl.workgroup.id.x
    // CHECK-NOT:     rocdl.workgroup.id.z
    func.func @test_work_group_id_2() -> !sycl_id_2_ {
      %0 = sycl.work_group_id : !sycl_id_2_
      return %0 : !sycl_id_2_
    }

    // CHECK-LABEL: func.func @test_work_group_size_3
    // CHECK:         rocdl.workgroup.dim.z
    // CHECK:         rocdl.workgroup.dim.y
    // CHECK:         rocdl.workgroup.dim.x
    func.func @test_work_group_size_3() -> !sycl_range_3_ {
      %0 = sycl.work_group_size : !sycl_range_3_
      return %0 : !sycl_range_3_
    }

    // CHECK-LABEL: func.func @test_num_work_groups_2
    // CHECK:         rocdl.grid.dim.y
    // CHECK:         rocdl.grid.dim.x
    // CHECK-NOT:     rocdl.grid.dim.z
    func.func @test_num_work_groups_2() -> !sycl_range_2_ {
      %0 = sycl.num_work_groups : !sycl_range_2_
      return %0 : !sycl_range_2_
    }
  }
}
