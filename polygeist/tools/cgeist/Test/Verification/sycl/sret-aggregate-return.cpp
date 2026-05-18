// RUN: clang++ -fsycl -fsycl-targets=spir64-unknown-unknown-syclmlir -O0 -w -emit-llvm -fsycl-device-only %s -o - 2>&1 | FileCheck %s --implicit-check-not="function should return its value indirectly"

// Verify that cgeist no longer emits the "function should return its value
// indirectly (as an extra reference parameter). This is not yet handled by
// the MLIR codegen" warning when a SYCL kernel uses the hierarchical
// parallelism API. Several SYCL header methods (h_item::get_global,
// h_item::get_local, Builder::createItem, InitializedVal::get) return
// non-trivial structs that the Itanium ABI lowers as ABIArgInfo::Indirect
// (sret). cgeist must emit them with the proper sret signature instead of
// a direct struct return.

#include <sycl/sycl.hpp>

int main() {
  using namespace sycl;
  queue q;
  int data[1] = {0};
  buffer<int, 1> buf{data, range<1>{1}};
  q.submit([&](handler &cgh) {
    auto acc = buf.get_access<access::mode::read_write>(cgh);
    cgh.parallel_for_work_group<class sret_kernel>(
        range<1>{1}, range<1>{1}, [=](group<1> g) {
          g.parallel_for_work_item([&](h_item<1> hi) {
            acc[hi.get_local_id()] = 42;
          });
        });
  });
  return 0;
}

// CHECK-NOT: function should return its value indirectly
