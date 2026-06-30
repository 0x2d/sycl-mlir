// RUN: clang++ -fsycl -fsycl-device-only -fsycl-targets=amdgcn-amd-amdhsa-syclmlir -Xsycl-target-backend --offload-arch=gfx906 -nogpulib -O0 -w -emit-llvm -S -o - %s 2>&1 | FileCheck %s --implicit-check-not="{{Assertion|Callsite argument mismatch}}"

// Exercises the callee-side struct-flatten prologue on AMDGCN.
//
// `h_item<1>` is not a SYCL dialect type (hierarchical parallelism is
// deferred), so cgeist emits its `!llvm.struct<(!sycl.item, !sycl.item,
// !sycl.item)>` representation. When the AMDGPU calling convention flattens
// such an aggregate into multiple IR args, the prologue in
// MLIRScanner::init reassembles them via per-field GEP+store (NOT
// llvm.insertvalue, whose operands must be primitive LLVM types and cannot
// be SYCL dialect types).
//
// This is a smoke test: it confirms cgeist emits the kernel without
// asserting and that the reassembly path produces GEP+store instructions
// in the function body. The exact IR shape varies with SYCL header
// revisions.

// CHECK: define {{.*}}amdgpu_kernel void @{{.*}}HItemFlattenKernel
// CHECK: getelementptr
// CHECK: store

#include <sycl/sycl.hpp>

using namespace sycl;

class HItemFlattenKernel;

int main() {
  queue q;
  constexpr size_t N = 16;
  buffer<int, 1> buf{range<1>{N}};

  q.submit([&](handler &cgh) {
    auto acc = buf.get_access<access::mode::write>(cgh);
    cgh.parallel_for_work_group<HItemFlattenKernel>(
        range<1>{N / 4}, range<1>{4}, [=](group<1> grp) {
          grp.parallel_for_work_item([&](h_item<1> idx) {
            size_t gid = idx.get_global_id(0);
            acc[gid] = static_cast<int>(gid);
          });
        });
  }).wait();

  return 0;
}
