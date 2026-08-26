; RUN: opt < %s -passes=sycl-rewrite-syr2k-range -S | FileCheck %s

; Positive case: the real PipelineStartEP shape of the syr2k launch site. The
; by-value range arguments of parallel_for_lambda_impl are stored into the
; %UserRange alloca; the getRoundedRange call's two i64 args may be loads from
; a temporary copy %agg.tmp11 that is filled by a memcpy FROM %UserRange. The
; pass must trace the memcpy back to %UserRange and rewrite its two stores to
; ceil(V/8) = (V+7) udiv 8.

%class.sycl.handler = type { i32 }
%"class.sycl::_V1::range" = type { i64, i64 }

@.str.usn = private unnamed_addr constant [12 x i8] c"_ZTS6Syr2k1\00", align 1

declare void @_ZN4sycl3_V17handler15getRoundedRangeILi2EEESt8optionalINS0_5rangeILi2EEES5_RNS0_6detail16RoundedRangeDataEERmS7_S7_(ptr sret(%"class.sycl::_V1::range"), ptr, i64, i64)

define void @parallel_for_lambda_impl(ptr %cgh, i64 %UserRange.coerce0, i64 %UserRange.coerce1) {
entry:
; CHECK-LABEL: @parallel_for_lambda_impl(
  %UserRange = alloca %"class.sycl::_V1::range", align 8
  %agg.tmp11 = alloca %"class.sycl::_V1::range", align 8
  %ret = alloca %"class.sycl::_V1::range", align 8

  %f0 = getelementptr inbounds { i64, i64 }, ptr %UserRange, i32 0, i32 0
  ; CHECK: [[PLUS0:%.*]] = add i64 %UserRange.coerce0, 7
  ; CHECK-NEXT: [[CEIL0:%.*]] = udiv i64 [[PLUS0]], 8
  ; CHECK-NEXT: store i64 [[CEIL0]], ptr %f0
  store i64 %UserRange.coerce0, ptr %f0, align 8

  %f1 = getelementptr inbounds { i64, i64 }, ptr %UserRange, i32 0, i32 1
  ; CHECK: [[PLUS1:%.*]] = add i64 %UserRange.coerce1, 7
  ; CHECK-NEXT: [[CEIL1:%.*]] = udiv i64 [[PLUS1]], 8
  ; CHECK-NEXT: store i64 [[CEIL1]], ptr %f1
  store i64 %UserRange.coerce1, ptr %f1, align 8

  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %agg.tmp11, ptr align 8 %UserRange, i64 16, i1 false)
  %g0 = getelementptr inbounds { i64, i64 }, ptr %agg.tmp11, i32 0, i32 0
  %l0 = load i64, ptr %g0, align 8
  %g1 = getelementptr inbounds { i64, i64 }, ptr %agg.tmp11, i32 0, i32 1
  %l1 = load i64, ptr %g1, align 8
  call void @_ZN4sycl3_V17handler15getRoundedRangeILi2EEESt8optionalINS0_5rangeILi2EEES5_RNS0_6detail16RoundedRangeDataEERmS7_S7_(ptr sret(%"class.sycl::_V1::range") %ret, ptr %cgh, i64 %l0, i64 %l1)

  ret void
}

; Negative case: the call's range args are NOT loads from a common alloca ->
; left untouched.
define void @notALaunchSite(ptr %cgh, i64 %A, i64 %B) {
entry:
; CHECK-LABEL: @notALaunchSite(
  %ret = alloca %"class.sycl::_V1::range", align 8
  ; CHECK: store i64 %A, ptr %dead
  %dead = alloca i64, align 8
  store i64 %A, ptr %dead, align 8
  call void @_ZN4sycl3_V17handler15getRoundedRangeILi2EEESt8optionalINS0_5rangeILi2EEES5_RNS0_6detail16RoundedRangeDataEERmS7_S7_(ptr sret(%"class.sycl::_V1::range") %ret, ptr %cgh, i64 %A, i64 %B)
  ret void
}

declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)
