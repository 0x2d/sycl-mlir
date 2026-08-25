; RUN: opt < %s -passes=sycl-rewrite-syrk-range -S | FileCheck %s

; Negative case: NO kernel-name anchor (_ZTS6Syr2k2 string constant absent).
; Even though the launch-site shape matches, the pass must be a no-op.

%class.sycl.handler = type { i32 }
%"class.sycl::_V1::range" = type { i64, i64 }

declare void @_ZN4sycl3_V17handler15getRoundedRangeILi2EEESt8optionalINS0_5rangeILi2EEES5_RNS0_6detail16RoundedRangeDataEERmS7_S7_(ptr sret(%"class.sycl::_V1::range"), ptr, i64, i64)

define void @someOtherKernelLaunch(ptr %cgh, i64 %N) {
entry:
; CHECK-LABEL: @someOtherKernelLaunch(
  %UserRange = alloca %"class.sycl::_V1::range", align 8
  %ret = alloca %"class.sycl::_V1::range", align 8

  ; CHECK-NOT: udiv
  ; CHECK: store i64 %N, ptr %UserRange
  store i64 %N, ptr %UserRange, align 8

  %dim1 = getelementptr inbounds { i64, i64 }, ptr %UserRange, i64 0, i32 1
  ; CHECK: store i64 %N, ptr %dim1
  store i64 %N, ptr %dim1, align 8

  %l0 = load i64, ptr %UserRange, align 8
  %l1 = load i64, ptr %dim1, align 8
  call void @_ZN4sycl3_V17handler15getRoundedRangeILi2EEESt8optionalINS0_5rangeILi2EEES5_RNS0_6detail16RoundedRangeDataEERmS7_S7_(ptr sret(%"class.sycl::_V1::range") %ret, ptr %cgh, i64 %l0, i64 %l1)

  ret void
}
