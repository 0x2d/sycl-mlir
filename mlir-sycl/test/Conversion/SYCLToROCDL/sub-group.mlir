// RUN: sycl-mlir-opt -convert-sycl-to-rocdl %s | FileCheck %s

module attributes {gpu.container_module} {
  gpu.module @kernels {
    // CHECK-DAG: llvm.func @__ockl_get_sub_group_size() -> i32
    // CHECK-DAG: llvm.func @__ockl_get_max_sub_group_size() -> i32
    // CHECK-DAG: llvm.func @__ockl_get_sub_group_id() -> i32
    // CHECK-DAG: llvm.func @__ockl_get_num_sub_groups() -> i32
    // CHECK-DAG: llvm.func @__ockl_get_sub_group_local_id() -> i32

    // CHECK-LABEL: func.func @test_sub_group_size
    // CHECK:         llvm.call @__ockl_get_sub_group_size
    func.func @test_sub_group_size() -> i32 {
      %0 = sycl.sub_group_size : i32
      return %0 : i32
    }

    // CHECK-LABEL: func.func @test_sub_group_max_size
    // CHECK:         llvm.call @__ockl_get_max_sub_group_size
    func.func @test_sub_group_max_size() -> i32 {
      %0 = sycl.sub_group_max_size : i32
      return %0 : i32
    }

    // CHECK-LABEL: func.func @test_sub_group_id
    // CHECK:         llvm.call @__ockl_get_sub_group_id
    func.func @test_sub_group_id() -> i32 {
      %0 = sycl.sub_group_id : i32
      return %0 : i32
    }

    // CHECK-LABEL: func.func @test_num_sub_groups
    // CHECK:         llvm.call @__ockl_get_num_sub_groups
    func.func @test_num_sub_groups() -> i32 {
      %0 = sycl.num_sub_groups : i32
      return %0 : i32
    }

    // CHECK-LABEL: func.func @test_sub_group_local_id
    // CHECK:         llvm.call @__ockl_get_sub_group_local_id
    func.func @test_sub_group_local_id() -> i32 {
      %0 = sycl.sub_group_local_id : i32
      return %0 : i32
    }
  }
}
