add_mlir_library(MLIRSYCLTransforms
  RegisterPromotionPattern.cpp
  FusionPass.cpp
  ConstantPropagation.cpp
  Inliner.cpp
  CovarianceLoopReorder.cpp
  SyrkRegisterAccumulator.cpp
  SyrkRegisterTile.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Transforms

  DEPENDS
  MLIRSYCLTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRSYCLAnalysis
  MLIRTransforms
  MLIRTransformUtils
  MLIRSYCLDialect
  MLIRFuncDialect
  MLIRGPUDialect
  )
