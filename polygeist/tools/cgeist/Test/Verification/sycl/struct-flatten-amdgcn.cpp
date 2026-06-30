// RUN: clang++ -fsycl -fsycl-device-only -fsycl-targets=amdgcn-amd-amdhsa-syclmlir -Xsycl-target-backend --offload-arch=gfx906 -nogpulib -O0 -w -emit-llvm -S -o - %s 2>&1 | FileCheck %s --implicit-check-not="{{Assertion|struct should be flattened}}"

// Exercises two fixes that previously aborted cgeist on AMDGCN:
//
//   1. createAllocOp now routes allocas through getAllocaAddrSpace() so
//      parameter-staging slots land in the AMDGPU private address space (5)
//      and bridge back to the flat address space via addrspacecast. Without
//      this, the AMDGPU backend aborts in ISel on an i64 = FrameIndex SDNode
//      matched against an addrspace(0) ptr type.
//   2. The kernel functor captures `Point two` (a 16-byte aggregate) by
//      value. AMDGPU's non-kernel calling convention would coerce such a
//      small aggregate to flattened scalars (Direct + canBeFlattened); cgeist
//      now honors that flattening intent on both caller and callee, instead
//      of asserting with "too many arguments in calls" or "Callsite argument
//      mismatch".
//
// This is a smoke test: it verifies cgeist produces an AMDGPU kernel without
// asserting and emits the expected alloca/addrspacecast shape. It is NOT a
// bit-exact IR test — the surrounding accessors and capture layout vary with
// SYCL header revisions.

// CHECK-DAG: alloca {{.*}}, align {{[0-9]+}}, addrspace(5)
// CHECK-DAG: addrspacecast ptr addrspace(5) {{%.*}} to ptr

// CHECK: define {{.*}}amdgpu_kernel void @{{.*}}_ZTS8SobelMin

#include <sycl/sycl.hpp>

using namespace sycl;

class SobelMin;

struct Point {
  float x;
  float y;
  float z;
  float w;
};

int main() {
  queue q;
  constexpr size_t N = 16;
  float *in = malloc_shared<float>(N, q);
  float *out = malloc_shared<float>(N, q);
  for (size_t i = 0; i < N; ++i)
    in[i] = float(i);

  buffer<float4, 1> ibuf{range<1>{N}};
  buffer<float4, 1> obuf{range<1>{N}};

  Point two{1.0f, 2.0f, 3.0f, 4.0f};

  q.submit([&](handler &cgh) {
    auto iacc = ibuf.get_access<access::mode::read>(cgh);
    auto oacc = obuf.get_access<access::mode::write>(cgh);
    size_t size = N;
    float kernel[9] = {1.f, 2.f, 1.f, 0.f, 0.f, 0.f, -1.f, -2.f, -1.f};
    cgh.parallel_for<SobelMin>(range<1>{N}, [=](id<1> gid) {
      float4 acc{0.f, 0.f, 0.f, 0.f};
      for (int k = 0; k < 9; ++k)
        acc += iacc[gid] * kernel[k];
      // Touch `two` so it is captured by value into the closure, exercising
      // the aggregate-capture path that previously tripped the flatten
      // assertion.
      acc[0] += two.x;
      acc[1] += two.y;
      acc[2] += two.z;
      acc[3] += two.w;
      if (gid[0] < size)
        oacc[gid] = acc;
    });
  }).wait();

  sycl::free(in, q);
  sycl::free(out, q);

  return 0;
}
