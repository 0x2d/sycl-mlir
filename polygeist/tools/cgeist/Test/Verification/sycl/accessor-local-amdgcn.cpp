// RUN: clang++ -fsycl -fsycl-device-only -fsycl-targets=amdgcn-amd-amdhsa-syclmlir -Xsycl-target-backend --offload-arch=gfx906 -nogpulib -O0 -w -emit-llvm -S -o - %s 2>&1 | FileCheck %s --implicit-check-not="{{Assertion|reused this field's tail padding}}"
// RUN: clang++ -fsycl -fsycl-device-only -fsycl-targets=amdgcn-amd-amdhsa-syclmlir -Xsycl-target-backend --offload-arch=gfx906 -nogpulib -O2 -w -emit-llvm -S -o - %s 2>&1 | FileCheck %s --implicit-check-not="{{Assertion|reused this field's tail padding}}"

// Reproduces the CGRecordLowering::clipTailPadding NoUniqueAddressAttr
// assertion that fired on AMDGCN. Root cause: cgeist constructed its
// llvm::Module without setting the target DataLayout, so getTypeAllocSize
// for `__local int *` (AS3, 32-bit on AMDGCN) returned the LLVM default of
// 8 bytes, which disagreed with the AST's 4-byte layout for MData and
// produced an apparent overlap with the next member. Fixed by initializing
// LLVMMod's DataLayout from the target before CodeGenModule reads it.

// CHECK: target datalayout = "{{.*}}p3:32:32{{.*}}"
// CHECK: define weak_odr amdgpu_kernel void @{{.*}}amdgcn_acc_local_kernel
// CHECK-SAME: ptr addrspace(3) noundef align 4

#include <vector>

#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
  auto q = sycl::queue{};
  std::vector<int> data(256, 0);
  {
    auto buf = buffer{data};
    q.submit([&](sycl::handler &cgh) {
      auto acc = buf.get_access<sycl::access::mode::read_write>(cgh);
      sycl::accessor<int, 1, sycl::access::mode::read_write, target::local>
          local{8, cgh};
      cgh.parallel_for<class amdgcn_acc_local_kernel>(
          nd_range<1>{{256}, {8}}, [=](nd_item<1> item) {
            auto global_id = item.get_global_id();
            auto local_id = item.get_local_id();
            local[local_id] = local_id;
            acc[global_id] = local[local_id];
          });
    }).wait();
  }

  return 0;
}
