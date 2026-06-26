add_mlir_library(MLIRTaskflowThroughputGuidedTaskOrchestration
    ResourceAssignmentState.cpp
    TaskProfiler.cpp
    ThroughputGuidedTaskOrchestration.cpp

    DEPENDS
    MLIRTaskflowTransformsIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRSupport
    MLIRAffineToStandard
    MLIRControlFlowToLLVM
    MLIRFuncDialect
    MLIRSCFToControlFlow
    MLIRTransforms
    MLIRNeura
    MLIRNeuraTransforms
    MLIRConversion
    MLIRTaskflow
    MLIRTaskflowOrchestrationUtil
    LLVMSupport
)
