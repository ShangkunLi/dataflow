// Profile Taskflow task mapping candidates into a JSON cache.

#include "NeuraDialect/NeuraDialect.h"
#include "TaskflowDialect/Orchestration/ThroughputGuidedTaskOrchestration/TaskProfiler.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "TaskflowDialect/TaskflowPasses.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::taskflow;

namespace {

struct ProfileTaskCandidatesPass
    : public PassWrapper<ProfileTaskCandidatesPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ProfileTaskCandidatesPass)

  ProfileTaskCandidatesPass() = default;
  ProfileTaskCandidatesPass(const ProfileTaskCandidatesPass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const override { return "profile-task-candidates"; }
  StringRef getDescription() const override {
    return "Profiles each Taskflow task across composed-CGRA candidates and "
           "writes a reusable JSON cache";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<affine::AffineDialect, arith::ArithDialect, LLVM::LLVMDialect,
                func::FuncDialect, memref::MemRefDialect, neura::NeuraDialect,
                scf::SCFDialect, taskflow::TaskflowDialect>();
  }

  Option<std::string> outputJson{
      *this, "output-json",
      llvm::cl::desc("Output JSON file for task candidate profiles."),
      llvm::cl::init("")};

  Option<int> maxComposedCgraCount{
      *this, "max-composed-cgra-count",
      llvm::cl::desc("Maximum number of adjacent CGRAs to profile for one "
                     "composed-CGRA candidate."),
      llvm::cl::init(4)};

  Option<int> symbolBoundTripCount{
      *this, "symbol-bound-trip-count",
      llvm::cl::desc("Sample trip count used for symbol-bound taskflow "
                     "counters during profiling."),
      llvm::cl::init(256)};

  void runOnOperation() override {
    if (outputJson.getValue().empty()) {
      getOperation().emitError()
          << "requires output-json for profile-task-candidates";
      signalPassFailure();
      return;
    }
    if (maxComposedCgraCount.getValue() <= 0) {
      getOperation().emitError() << "max-composed-cgra-count must be positive";
      signalPassFailure();
      return;
    }
    if (symbolBoundTripCount.getValue() <= 0) {
      getOperation().emitError() << "symbol-bound-trip-count must be positive";
      signalPassFailure();
      return;
    }

    TaskProfiler profiler(maxComposedCgraCount.getValue(),
                          symbolBoundTripCount.getValue());
    if (failed(profiler.profileFunctionToJson(getOperation(),
                                              outputJson.getValue()))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir {
namespace taskflow {

std::unique_ptr<Pass> createProfileTaskCandidatesPass() {
  return std::make_unique<ProfileTaskCandidatesPass>();
}

} // namespace taskflow
} // namespace mlir
