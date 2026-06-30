// Orchestrate Taskflow tasks onto a multi-CGRA grid.

#include "NeuraDialect/Architecture/Architecture.h"
#include "NeuraDialect/NeuraDialect.h"
#include "TaskflowDialect/Orchestration/RoutingCriticalPathOrchestration/RoutingCriticalPathOrchestration.h"
#include "TaskflowDialect/Orchestration/ThroughputGuidedTaskOrchestration/ThroughputGuidedTaskOrchestration.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "TaskflowDialect/TaskflowPasses.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringSwitch.h"

using namespace mlir;
using namespace mlir::taskflow;

namespace {

std::unique_ptr<Orchestration> createOrchestrationStrategy(
    StringRef strategy_name, int grid_rows, int grid_cols, SchedulingMode mode,
    StringRef task_profile_json, int max_contexts_per_cgra) {
  return llvm::StringSwitch<std::unique_ptr<Orchestration>>(strategy_name)
      .Case("routing-critical-path",
            std::make_unique<RoutingCriticalPathOrchestration>(
                grid_rows, grid_cols, mode, max_contexts_per_cgra))
      .Case("throughput-guided",
            std::make_unique<ThroughputGuidedTaskOrchestration>(
                grid_rows, grid_cols, mode, task_profile_json.str(),
                max_contexts_per_cgra))
      .Default(nullptr);
}

struct OrchestrateTaskOnCgraPass
    : public PassWrapper<OrchestrateTaskOnCgraPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OrchestrateTaskOnCgraPass)

  OrchestrateTaskOnCgraPass() = default;
  OrchestrateTaskOnCgraPass(const OrchestrateTaskOnCgraPass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const override { return "orchestrate-task-on-cgra"; }
  StringRef getDescription() const override {
    return "Orchestrates Taskflow tasks onto a 2D multi-CGRA grid (spatial or "
           "spatial-temporal)";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<affine::AffineDialect, arith::ArithDialect, LLVM::LLVMDialect,
                func::FuncDialect, memref::MemRefDialect, neura::NeuraDialect,
                scf::SCFDialect, taskflow::TaskflowDialect>();
  }

  Option<std::string> schedulingMode{
      *this, "scheduling-mode",
      llvm::cl::desc("Task scheduling mode: 'spatial' (one task per CGRA, "
                     "asserts if tasks exceed the grid size) or "
                     "'spatial-temporal' (default, time-multiplexes CGRAs so "
                     "task count is not bounded by grid size)."),
      llvm::cl::init("spatial-temporal")};

  Option<std::string> orchestrationStrategy{
      *this, "orchestration-strategy",
      llvm::cl::desc("Task orchestration strategy: 'routing-critical-path' "
                     "(default) or 'throughput-guided'."),
      llvm::cl::init("routing-critical-path")};

  Option<std::string> taskProfileJson{
      *this, "task-profile-json",
      llvm::cl::desc("Optional JSON profile cache for throughput-guided "
                     "orchestration. When provided, task profiles are read "
                     "from this file instead of invoking the profiler."),
      llvm::cl::init("")};

  void runOnOperation() override {
    SchedulingMode mode = (schedulingMode.getValue() == "spatial")
                              ? SchedulingMode::Spatial
                              : SchedulingMode::SpatialTemporal;
    const neura::Architecture &architecture = neura::getArchitecture();
    std::unique_ptr<Orchestration> strategy = createOrchestrationStrategy(
        orchestrationStrategy.getValue(), architecture.getMultiCgraRows(),
        architecture.getMultiCgraColumns(), mode, taskProfileJson.getValue(),
        architecture.getMaxContextMemItems());
    if (!strategy) {
      getOperation()->emitError() << "unknown task orchestration strategy: "
                                  << orchestrationStrategy.getValue();
      signalPassFailure();
      return;
    }

    if (!strategy->runTaskOrchestration(getOperation())) {
      getOperation()->emitError()
          << "failed to orchestrate taskflow tasks with strategy: "
          << orchestrationStrategy.getValue();
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir {
namespace taskflow {

std::unique_ptr<Pass> createOrchestrateTaskOnCgraPass() {
  return std::make_unique<OrchestrateTaskOnCgraPass>();
}

} // namespace taskflow
} // namespace mlir
