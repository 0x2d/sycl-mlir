add_mlir_conversion_library(MLIRSYCLToROCDL
  SYCLToROCDL.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_SYCL_SOURCE_DIR}/mlir/Conversion/SYCLToROCDL

  DEPENDS
  MLIRSYCLConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRGPUDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRROCDLDialect
  MLIRSYCLDialect
  MLIRTransforms
  MLIRVectorDialect
  )
