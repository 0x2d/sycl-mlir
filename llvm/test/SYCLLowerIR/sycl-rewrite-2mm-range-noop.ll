; RUN: opt < %s -passes=sycl-rewrite-2mm-range -S | FileCheck %s

; No-op case: no `_ZTS15Polybench_2mm_1` / `_ZTS15Polybench_2mm_2` string
; constant anywhere in the module -> this TU launches neither 2mm kernel and
; the pass must not touch anything, even though one function looks like a
; 2mm launch site by name and shape.

%class.sycl.handler = type { i32 }
%"class.sycl::_V1::range" = type { i64, i64 }

declare void @_ZN4sycl3_V17handler15getRoundedRangeILi2EEESt5tupleIJNS0_5rangeILi2EEEEbEES5_(ptr sret(%"class.sycl::_V1::range"), ptr, i64, i64)

define void @_ZN4sycl3_V17handler24parallel_for_lambda_implI15Polybench_2mm_1E(ptr %cgh, i64 %UserRange.coerce0, i64 %UserRange.coerce1) {
entry:
; CHECK-LABEL: @_ZN4sycl3_V17handler24parallel_for_lambda_implI15Polybench_2mm_1E(
  %UserRange = alloca %"class.sycl::_V1::range", align 8
  %agg.tmp11 = alloca %"class.sycl::_V1::range", align 8
  %ret = alloca %"class.sycl::_V1::range", align 8

  %f0 = getelementptr inbounds { i64, i64 }, ptr %UserRange, i32 0, i32 0
  ; CHECK: store i64 %UserRange.coerce0, ptr %f0
  store i64 %UserRange.coerce0, ptr %f0, align 8

  %f1 = getelementptr inbounds { i64, i64 }, ptr %UserRange, i32 0, i32 1
  ; CHECK: store i64 %UserRange.coerce1, ptr %f1
  store i64 %UserRange.coerce1, ptr %f1, align 8

  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %agg.tmp11, ptr align 8 %UserRange, i64 16, i1 false)
  %g0 = getelementptr inbounds { i64, i64 }, ptr %agg.tmp11, i32 0, i32 0
  %l0 = load i64, ptr %g0, align 8
  %g1 = getelementptr inbounds { i64, i64 }, ptr %agg.tmp11, i32 0, i32 1
  %l1 = load i64, ptr %g1, align 8
  call void @_ZN4sycl3_V17handler15getRoundedRangeILi2EEESt5tupleIJNS0_5rangeILi2EEEEbEES5_(ptr sret(%"class.sycl::_V1::range") %ret, ptr %cgh, i64 %l0, i64 %l1)

  ret void
}

; CHECK-NOT: udiv

declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)
