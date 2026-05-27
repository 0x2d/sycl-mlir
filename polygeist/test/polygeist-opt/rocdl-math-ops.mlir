// RUN: polygeist-opt %s --convert-polygeist-to-llvm='sycl-target=rocdl' --split-input-file | FileCheck %s --check-prefix=ROCDL
// RUN: polygeist-opt %s --convert-polygeist-to-llvm | FileCheck %s --check-prefix=SPIRV

// Verify that with sycl-target=rocdl, the transcendental math ops are
// rewritten to llvm.call @__ocml_*_{f32,f64} (since AMDGPU has no f64 ISel
// for the corresponding LLVM intrinsics), while the default (SPIR-V) target
// still routes them through the LLVM intrinsics via populateMathToLLVMConversionPatterns.

// ROCDL-LABEL: llvm.func @transcendentals_f64
// ROCDL: llvm.call @__ocml_sin_f64
// ROCDL: llvm.call @__ocml_cos_f64
// ROCDL: llvm.call @__ocml_tan_f64
// ROCDL: llvm.call @__ocml_exp_f64
// ROCDL: llvm.call @__ocml_log_f64
// ROCDL: llvm.call @__ocml_pow_f64
// ROCDL-NOT: llvm.intr.sin
// ROCDL-NOT: llvm.intr.cos

// SPIRV-LABEL: llvm.func @transcendentals_f64
// SPIRV: llvm.intr.sin
// SPIRV: llvm.intr.cos
// SPIRV-NOT: __ocml_sin_f64
// SPIRV-NOT: __ocml_cos_f64
func.func @transcendentals_f64(%x: f64, %y: f64) -> f64 {
  %0 = math.sin %x : f64
  %1 = math.cos %0 : f64
  %2 = math.tan %1 : f64
  %3 = math.exp %2 : f64
  %4 = math.log %3 : f64
  %5 = math.powf %4, %y : f64
  return %5 : f64
}

// -----

// ROCDL-LABEL: llvm.func @transcendentals_f32
// ROCDL: llvm.call @__ocml_sin_f32
// ROCDL: llvm.call @__ocml_cos_f32
// ROCDL: llvm.call @__ocml_exp_f32
func.func @transcendentals_f32(%x: f32) -> f32 {
  %0 = math.sin %x : f32
  %1 = math.cos %0 : f32
  %2 = math.exp %1 : f32
  return %2 : f32
}

// -----

// Regression guard: math.sqrt on f64 should NOT be redirected to ocml.
// AMDGPU has a native f64 ISel pattern for llvm.intr.sqrt, so we keep using
// the LLVM intrinsic for performance.

// ROCDL-LABEL: llvm.func @sqrt_f64
// ROCDL: llvm.intr.sqrt
// ROCDL-NOT: __ocml_sqrt_f64
func.func @sqrt_f64(%x: f64) -> f64 {
  %0 = math.sqrt %x : f64
  return %0 : f64
}
