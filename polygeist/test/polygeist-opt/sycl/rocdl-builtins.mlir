// RUN: polygeist-opt --convert-polygeist-to-llvm="sycl-target=rocdl" %s | FileCheck %s

!sycl_id_1_ = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_id_2_ = !sycl.id<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>
!sycl_id_3_ = !sycl.id<[3], (!sycl.array<[3], (memref<3xi64, 4>)>)>
!sycl_range_2_ = !sycl.range<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>
!sycl_range_3_ = !sycl.range<[3], (!sycl.array<[3], (memref<3xi64, 4>)>)>

module attributes {gpu.container_module} {
  gpu.module @kernels {
    // CHECK-LABEL: llvm.func @test_local_id_1
    // CHECK:         rocdl.workitem.id.x
    func.func @test_local_id_1() -> !sycl_id_1_ {
      %0 = sycl.local_id : !sycl_id_1_
      return %0 : !sycl_id_1_
    }

    // CHECK-LABEL: llvm.func @test_local_id_3
    // CHECK-DAG:     rocdl.workitem.id.x
    // CHECK-DAG:     rocdl.workitem.id.y
    // CHECK-DAG:     rocdl.workitem.id.z
    func.func @test_local_id_3() -> !sycl_id_3_ {
      %0 = sycl.local_id : !sycl_id_3_
      return %0 : !sycl_id_3_
    }

    // CHECK-LABEL: llvm.func @test_work_group_id_2
    // CHECK-DAG:     rocdl.workgroup.id.x
    // CHECK-DAG:     rocdl.workgroup.id.y
    func.func @test_work_group_id_2() -> !sycl_id_2_ {
      %0 = sycl.work_group_id : !sycl_id_2_
      return %0 : !sycl_id_2_
    }

    // CHECK-LABEL: llvm.func @test_work_group_size_3
    // CHECK-DAG:     rocdl.workgroup.dim.x
    // CHECK-DAG:     rocdl.workgroup.dim.y
    // CHECK-DAG:     rocdl.workgroup.dim.z
    func.func @test_work_group_size_3() -> !sycl_range_3_ {
      %0 = sycl.work_group_size : !sycl_range_3_
      return %0 : !sycl_range_3_
    }

    // CHECK-LABEL: llvm.func @test_num_work_groups_2
    // CHECK-DAG:     rocdl.grid.dim.x
    // CHECK-DAG:     rocdl.grid.dim.y
    func.func @test_num_work_groups_2() -> !sycl_range_2_ {
      %0 = sycl.num_work_groups : !sycl_range_2_
      return %0 : !sycl_range_2_
    }

    // CHECK-LABEL: llvm.func @test_global_id_3
    // CHECK-DAG:     rocdl.workgroup.id.x
    // CHECK-DAG:     rocdl.workgroup.id.y
    // CHECK-DAG:     rocdl.workgroup.id.z
    // CHECK-DAG:     rocdl.workgroup.dim.x
    // CHECK-DAG:     rocdl.workgroup.dim.y
    // CHECK-DAG:     rocdl.workgroup.dim.z
    // CHECK-DAG:     rocdl.workitem.id.x
    // CHECK-DAG:     rocdl.workitem.id.y
    // CHECK-DAG:     rocdl.workitem.id.z
    func.func @test_global_id_3() -> !sycl_id_3_ {
      %0 = sycl.global_id : !sycl_id_3_
      return %0 : !sycl_id_3_
    }

    // CHECK-LABEL: llvm.func @test_global_offset_2
    // CHECK:         llvm.alloca {{.*}} !llvm.struct<"class.sycl::_V1::id.2"
    // CHECK-NOT:     rocdl.
    // CHECK:         llvm.return
    func.func @test_global_offset_2() -> !sycl_id_2_ {
      %0 = sycl.global_offset : !sycl_id_2_
      return %0 : !sycl_id_2_
    }

    // CHECK-LABEL: llvm.func @test_sub_group_size
    // CHECK:         llvm.call @__ockl_get_sub_group_size
    func.func @test_sub_group_size() -> i32 {
      %0 = sycl.sub_group_size : i32
      return %0 : i32
    }

    // CHECK-LABEL: llvm.func @test_sub_group_local_id
    // CHECK:         llvm.call @__ockl_get_sub_group_local_id
    func.func @test_sub_group_local_id() -> i32 {
      %0 = sycl.sub_group_local_id : i32
      return %0 : i32
    }
  }
}
