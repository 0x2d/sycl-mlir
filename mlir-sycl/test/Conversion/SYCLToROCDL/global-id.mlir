// RUN: sycl-mlir-opt -convert-sycl-to-rocdl %s | FileCheck %s

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_id_3_ = !sycl.id<[3], (!sycl.array<[3], (memref<3xi64, 4>)>)>
!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<3xi64, 4>)>)>

module attributes {gpu.container_module} {
  gpu.module @kernels {
    // global_id_i = workgroup_id_i * workgroup_dim_i + workitem_id_i, per dim.
    // CHECK-LABEL: func.func @test_global_id_1
    // CHECK:         %[[BID:.*]] = rocdl.workgroup.id.x
    // CHECK:         %[[BDIM:.*]] = rocdl.workgroup.dim.x
    // CHECK:         %[[TID:.*]] = rocdl.workitem.id.x
    // CHECK:         %[[MUL:.*]] = arith.muli %[[BID]], %[[BDIM]]
    // CHECK:         arith.addi %[[MUL]], %[[TID]]
    func.func @test_global_id_1() -> !sycl_id_1_ {
      %0 = sycl.global_id : !sycl_id_1_
      return %0 : !sycl_id_1_
    }

    // SYCL dim 0 mirrors to AMDGCN Z, dim 2 mirrors to AMDGCN X.
    // CHECK-LABEL: func.func @test_global_id_3
    // CHECK:         rocdl.workgroup.id.z
    // CHECK:         rocdl.workgroup.dim.z
    // CHECK:         rocdl.workitem.id.z
    // CHECK:         rocdl.workgroup.id.y
    // CHECK:         rocdl.workgroup.dim.y
    // CHECK:         rocdl.workitem.id.y
    // CHECK:         rocdl.workgroup.id.x
    // CHECK:         rocdl.workgroup.dim.x
    // CHECK:         rocdl.workitem.id.x
    func.func @test_global_id_3() -> !sycl_id_3_ {
      %0 = sycl.global_id : !sycl_id_3_
      return %0 : !sycl_id_3_
    }

    // num_work_items_i = workgroup_dim_i * grid_dim_i, per dim.
    // CHECK-LABEL: func.func @test_num_work_items_3
    // CHECK:         rocdl.workgroup.dim.z
    // CHECK:         rocdl.grid.dim.z
    // CHECK:         arith.muli
    // CHECK:         rocdl.workgroup.dim.y
    // CHECK:         rocdl.grid.dim.y
    // CHECK:         arith.muli
    // CHECK:         rocdl.workgroup.dim.x
    // CHECK:         rocdl.grid.dim.x
    // CHECK:         arith.muli
    func.func @test_num_work_items_3() -> !sycl_range_3_ {
      %0 = sycl.num_work_items : !sycl_range_3_
      return %0 : !sycl_range_3_
    }
  }
}
