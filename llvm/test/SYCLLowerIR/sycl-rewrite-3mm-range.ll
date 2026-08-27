; RUN: opt < %s -passes=sycl-rewrite-3mm-range -S | FileCheck %s

; Positive cases: the real PipelineStartEP shape of the THREE 3mm launch
; sites. 3mm launches three GEMM kernels back-to-back; each
; `parallel_for_lambda_impl` instantiation's mangled name embeds its kernel
; tag class (15Polybench_3mm_1 / _2 / _3). The by-value range arguments are
; stored into the %UserRange alloca; the getRoundedRange call's two i64 args
; are loads from a temporary copy %agg.tmp11 that is filled by a memcpy FROM
; %UserRange. The pass must rewrite ALL THREE functions' stores to
; ceil(V/16)*16 = ((V+15) udiv 16) * 16 (the FULL padded grid, one work-item
; per output cell of the sycl-3mm-local-tile device pass; divisibility by 16
; satisfies the UR launch validator for the SYCL_FORCE_LOCAL_SIZE local size
; (16,16)).

%class.sycl.handler = type { i32 }
%"class.sycl::_V1::range" = type { i64, i64 }

@.str.87 = private unnamed_addr constant [22 x i8] c"_ZTS15Polybench_3mm_1\00", align 1
@.str.93 = private unnamed_addr constant [22 x i8] c"_ZTS15Polybench_3mm_2\00", align 1
@.str.99 = private unnamed_addr constant [22 x i8] c"_ZTS15Polybench_3mm_3\00", align 1

declare void @_ZN4sycl3_V17handler15getRoundedRangeILi2EEESt5tupleIJNS0_5rangeILi2EEEEbEES5_(ptr sret(%"class.sycl::_V1::range"), ptr, i64, i64)

define void @_ZN4sycl3_V17handler24parallel_for_lambda_implI15Polybench_3mm_1E(ptr %cgh, i64 %UserRange.coerce0, i64 %UserRange.coerce1) {
entry:
; CHECK-LABEL: @_ZN4sycl3_V17handler24parallel_for_lambda_implI15Polybench_3mm_1E(
  %UserRange = alloca %"class.sycl::_V1::range", align 8
  %agg.tmp11 = alloca %"class.sycl::_V1::range", align 8
  %ret = alloca %"class.sycl::_V1::range", align 8

  %f0 = getelementptr inbounds { i64, i64 }, ptr %UserRange, i32 0, i32 0
  ; CHECK: [[PLUS0:%.*]] = add i64 %UserRange.coerce0, 15
  ; CHECK-NEXT: [[CEIL0:%.*]] = udiv i64 [[PLUS0]], 16
  ; CHECK-NEXT: [[PAD0:%.*]] = mul i64 [[CEIL0]], 16
  ; CHECK-NEXT: store i64 [[PAD0]], ptr %f0
  store i64 %UserRange.coerce0, ptr %f0, align 8

  %f1 = getelementptr inbounds { i64, i64 }, ptr %UserRange, i32 0, i32 1
  ; CHECK: [[PLUS1:%.*]] = add i64 %UserRange.coerce1, 15
  ; CHECK-NEXT: [[CEIL1:%.*]] = udiv i64 [[PLUS1]], 16
  ; CHECK-NEXT: [[PAD1:%.*]] = mul i64 [[CEIL1]], 16
  ; CHECK-NEXT: store i64 [[PAD1]], ptr %f1
  store i64 %UserRange.coerce1, ptr %f1, align 8

  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %agg.tmp11, ptr align 8 %UserRange, i64 16, i1 false)
  %g0 = getelementptr inbounds { i64, i64 }, ptr %agg.tmp11, i32 0, i32 0
  %l0 = load i64, ptr %g0, align 8
  %g1 = getelementptr inbounds { i64, i64 }, ptr %agg.tmp11, i32 0, i32 1
  %l1 = load i64, ptr %g1, align 8
  call void @_ZN4sycl3_V17handler15getRoundedRangeILi2EEESt5tupleIJNS0_5rangeILi2EEEEbEES5_(ptr sret(%"class.sycl::_V1::range") %ret, ptr %cgh, i64 %l0, i64 %l1)

  ret void
}

define void @_ZN4sycl3_V17handler24parallel_for_lambda_implI15Polybench_3mm_2E(ptr %cgh, i64 %UserRange.coerce0, i64 %UserRange.coerce1) {
entry:
; CHECK-LABEL: @_ZN4sycl3_V17handler24parallel_for_lambda_implI15Polybench_3mm_2E(
  %UserRange = alloca %"class.sycl::_V1::range", align 8
  %agg.tmp11 = alloca %"class.sycl::_V1::range", align 8
  %ret = alloca %"class.sycl::_V1::range", align 8

  %f0 = getelementptr inbounds { i64, i64 }, ptr %UserRange, i32 0, i32 0
  ; CHECK: [[PLUS0:%.*]] = add i64 %UserRange.coerce0, 15
  ; CHECK-NEXT: [[CEIL0:%.*]] = udiv i64 [[PLUS0]], 16
  ; CHECK-NEXT: [[PAD0:%.*]] = mul i64 [[CEIL0]], 16
  ; CHECK-NEXT: store i64 [[PAD0]], ptr %f0
  store i64 %UserRange.coerce0, ptr %f0, align 8

  %f1 = getelementptr inbounds { i64, i64 }, ptr %UserRange, i32 0, i32 1
  ; CHECK: [[PLUS1:%.*]] = add i64 %UserRange.coerce1, 15
  ; CHECK-NEXT: [[CEIL1:%.*]] = udiv i64 [[PLUS1]], 16
  ; CHECK-NEXT: [[PAD1:%.*]] = mul i64 [[CEIL1]], 16
  ; CHECK-NEXT: store i64 [[PAD1]], ptr %f1
  store i64 %UserRange.coerce1, ptr %f1, align 8

  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %agg.tmp11, ptr align 8 %UserRange, i64 16, i1 false)
  %g0 = getelementptr inbounds { i64, i64 }, ptr %agg.tmp11, i32 0, i32 0
  %l0 = load i64, ptr %g0, align 8
  %g1 = getelementptr inbounds { i64, i64 }, ptr %agg.tmp11, i32 0, i32 1
  %l1 = load i64, ptr %g1, align 8
  call void @_ZN4sycl3_V17handler15getRoundedRangeILi2EEESt5tupleIJNS0_5rangeILi2EEEEbEES5_(ptr sret(%"class.sycl::_V1::range") %ret, ptr %cgh, i64 %l0, i64 %l1)

  ret void
}

define void @_ZN4sycl3_V17handler24parallel_for_lambda_implI15Polybench_3mm_3E(ptr %cgh, i64 %UserRange.coerce0, i64 %UserRange.coerce1) {
entry:
; CHECK-LABEL: @_ZN4sycl3_V17handler24parallel_for_lambda_implI15Polybench_3mm_3E(
  %UserRange = alloca %"class.sycl::_V1::range", align 8
  %agg.tmp11 = alloca %"class.sycl::_V1::range", align 8
  %ret = alloca %"class.sycl::_V1::range", align 8

  %f0 = getelementptr inbounds { i64, i64 }, ptr %UserRange, i32 0, i32 0
  ; CHECK: [[PLUS0:%.*]] = add i64 %UserRange.coerce0, 15
  ; CHECK-NEXT: [[CEIL0:%.*]] = udiv i64 [[PLUS0]], 16
  ; CHECK-NEXT: [[PAD0:%.*]] = mul i64 [[CEIL0]], 16
  ; CHECK-NEXT: store i64 [[PAD0]], ptr %f0
  store i64 %UserRange.coerce0, ptr %f0, align 8

  %f1 = getelementptr inbounds { i64, i64 }, ptr %UserRange, i32 0, i32 1
  ; CHECK: [[PLUS1:%.*]] = add i64 %UserRange.coerce1, 15
  ; CHECK-NEXT: [[CEIL1:%.*]] = udiv i64 [[PLUS1]], 16
  ; CHECK-NEXT: [[PAD1:%.*]] = mul i64 [[CEIL1]], 16
  ; CHECK-NEXT: store i64 [[PAD1]], ptr %f1
  store i64 %UserRange.coerce1, ptr %f1, align 8

  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %agg.tmp11, ptr align 8 %UserRange, i64 16, i1 false)
  %g0 = getelementptr inbounds { i64, i64 }, ptr %agg.tmp11, i32 0, i32 0
  %l0 = load i64, ptr %g0, align 8
  %g1 = getelementptr inbounds { i64, i64 }, ptr %agg.tmp11, i32 0, i32 1
  %l1 = load i64, ptr %g1, align 8
  call void @_ZN4sycl3_V17handler15getRoundedRangeILi2EEESt5tupleIJNS0_5rangeILi2EEEEbEES5_(ptr sret(%"class.sycl::_V1::range") %ret, ptr %cgh, i64 %l0, i64 %l1)

  ret void
}

; Negative case 1: a parallel_for_lambda_impl instantiation for a DIFFERENT
; kernel (not one of the three 3mm tag classes) -> per-function name filter
; leaves it untouched, even though the module carries the 3mm anchors.
define void @_ZN4sycl3_V17handler24parallel_for_lambda_implI6Syr2k2E(ptr %cgh, i64 %UserRange.coerce0, i64 %UserRange.coerce1) {
entry:
; CHECK-LABEL: @_ZN4sycl3_V17handler24parallel_for_lambda_implI6Syr2k2E(
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

; Negative case 2: the call's range args are NOT loads from a common alloca
; -> left untouched.
define void @notALaunchSite(ptr %cgh, i64 %A, i64 %B) {
entry:
; CHECK-LABEL: @notALaunchSite(
  %ret = alloca %"class.sycl::_V1::range", align 8
  ; CHECK: store i64 %A, ptr %dead
  %dead = alloca i64, align 8
  store i64 %A, ptr %dead, align 8
  call void @_ZN4sycl3_V17handler15getRoundedRangeILi2EEESt5tupleIJNS0_5rangeILi2EEEEbEES5_(ptr sret(%"class.sycl::_V1::range") %ret, ptr %cgh, i64 %A, i64 %B)
  ret void
}

declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)
