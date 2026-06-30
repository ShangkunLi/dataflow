// Throughput-guided CGRA task orchestration.

#ifndef TASKFLOW_THROUGHPUT_GUIDED_TASK_ORCHESTRATION_H
#define TASKFLOW_THROUGHPUT_GUIDED_TASK_ORCHESTRATION_H

#include "TaskflowDialect/Orchestration/Orchestration.h"
#include "TaskflowDialect/Orchestration/orchestration_utils.h"

#include <string>
#include <utility>

namespace mlir {
namespace taskflow {

// Concrete orchestration strategy for the throughput-guided resource search.
//
// This strategy is the compiler-side entry point for AMOEBA's holistic
// orchestration algorithm. It uses sample-input profiling to decide how many
// adjacent CGRAs should be composed into a larger tile array for each task, how
// many static copies of a data-parallel task should run, and when each task
// should execute.
class ThroughputGuidedTaskOrchestration : public Orchestration {
public:
  ThroughputGuidedTaskOrchestration(
      int grid_rows = kCgraGridRows, int grid_cols = kCgraGridCols,
      SchedulingMode mode = SchedulingMode::SpatialTemporal,
      std::string task_profile_json = "",
      int max_contexts_per_cgra = kDefaultMaxContextsPerCgra)
      : grid_rows_(grid_rows), grid_cols_(grid_cols), mode_(mode),
        task_profile_json_(std::move(task_profile_json)),
        max_contexts_per_cgra_(max_contexts_per_cgra) {}

  // Runs throughput-guided task orchestration and emits the selected static
  // task orchestration metadata.
  bool runTaskOrchestration(mlir::func::FuncOp func) override;

  std::string getName() const override { return "throughput-guided"; }

private:
  int grid_rows_;
  int grid_cols_;
  SchedulingMode mode_;
  std::string task_profile_json_;
  int max_contexts_per_cgra_ = kDefaultMaxContextsPerCgra;
};

} // namespace taskflow
} // namespace mlir

#endif // TASKFLOW_THROUGHPUT_GUIDED_TASK_ORCHESTRATION_H
