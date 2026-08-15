// RUN: clang++ -fsycl -fsycl-device-only -fsycl-targets=amdgcn-amd-amdhsa-syclmlir -Xsycl-target-backend --offload-arch=gfx906 -nogpulib -O0 -w -emit-llvm -S -o - %s 2>&1 | FileCheck %s

// Verifies emitSYCLKernelArgMetadata (polygeist/tools/cgeist/driver.cc),
// which attaches the SYCL kernel-arg metadata that clang CodeGen emits on
// the SPIR path but the cgeist C->MLIR->LLVM path skips (it never runs clang
// CodeGen for device code). The decisive node is `!amdgcn.annotations`: it
// feeds TargetHelpers::populateKernels (llvm/lib/SYCLLowerIR/TargetHelpers.cpp)
// so LocalAccessorToSharedMemoryPass can find the kernels and rewrite
// `ptr addrspace(3)` local args to `i32` offsets -> HSA `by_value` -> the
// runtime spaces >=2 local accessors by host buffer size (distinct LDS)
// instead of the pointer-slot size (aliasing). Without it the >=2-local
// kernels mis-compile. See local-accessor-amdgcn-lever-amdgcn-annotations.md.
//
// The kernel uses TWO local accessors (the aliasing trigger): if the metadata
// regresses, the i1 vector / annotation node shape changes and this test
// fails.

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
          aTile{8, cgh};
      sycl::accessor<int, 1, sycl::access::mode::read_write, target::local>
          bTile{8, cgh};
      cgh.parallel_for<class two_local_metadata_kernel>(
          nd_range<1>{{256}, {8}}, [=](nd_item<1> item) {
            size_t lid = item.get_local_id(0);
            size_t gid = item.get_global_id(0);
            aTile[lid] = static_cast<int>(lid);
            bTile[lid] = static_cast<int>(lid);
            acc[gid] = bTile[lid];
          });
    }).wait();
  }

  return 0;
}

// The kernel signature carries two `ptr addrspace(3)` local args (aTile, bTile)
// and one `ptr addrspace(1)` global arg (acc) -> three accessor base pointers.
// !kernel_arg_runtime_aligned and !kernel_arg_exclusive_ptr share one i1 vector
// (![[EXC]]); !kernel_arg_buffer_location is ![[BL]]; !sycl_fixed_targets is
// ![[SFT]].
// CHECK: define {{.*}}amdgpu_kernel{{.*}}ptr addrspace(3){{.*}}ptr addrspace(3){{.*}}ptr addrspace(1){{.*}}!kernel_arg_buffer_location ![[BL:[0-9]+]] !kernel_arg_runtime_aligned ![[EXC:[0-9]+]] !kernel_arg_exclusive_ptr ![[EXC]] !sycl_fixed_targets ![[SFT:[0-9]+]]

// Named metadata (appear before the numbered node definitions). Check them in
// file order — FileCheck scans forward only.
// CHECK: !amdgcn.annotations = !{![[ANN:[0-9]+]]}
// CHECK: !opencl.spir.version = !{![[SPV:[0-9]+]]}
// CHECK: !opencl.ocl.version = !{![[OCV:[0-9]+]]}
// CHECK: !spirv.Source = !{![[SRC:[0-9]+]]}

// Numbered node definitions, in !N order (!0 is clang's Debug Info Version,
// not emitted by us; !1 onward is ours).
// Code object version v4: module flag, Module::Error behavior = 1.
// CHECK: !{i32 1, !"amdgpu_code_object_version", i32 400}
// !amdgcn.annotations entry: !{ptr @kernel, !"kernel", i32 1} per amdgpu_kernel
// (mirrors clang AMDGPUTargetCodeGenInfo::addAMDGCNMetadata).
// CHECK: ![[ANN]] = !{ptr @{{.*}}, !"kernel", i32 1}
// CHECK: ![[SPV]] = !{i32 1, i32 2}
// CHECK: ![[OCV]] = !{i32 2, i32 0}
// CHECK: ![[SRC]] = !{i32 4, i32 100000}

// !kernel_arg_buffer_location: vector of -1, one entry per arg (12 args).
// CHECK: ![[BL]] = !{i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1}

// !kernel_arg_exclusive_ptr / !kernel_arg_runtime_aligned share this i1 vector
// (same node ![[EXC]]): `true` at each accessor base pointer arg -> args 0, 4,
// 8 (aTile, bTile, acc); `false` at the range/id struct args.
// CHECK: ![[EXC]] = !{i1 true, i1 false, i1 false, i1 false, i1 true, i1 false, i1 false, i1 false, i1 true, i1 false, i1 false, i1 false}

// !sycl_fixed_targets: empty node (as on the SPIR path).
// CHECK: ![[SFT]] = !{}
