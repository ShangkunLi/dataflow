// Throughput-guided task profiler.

#ifndef TASKFLOW_THROUGHPUT_GUIDED_TASK_PROFILER_H
#define TASKFLOW_THROUGHPUT_GUIDED_TASK_PROFILER_H

#include "TaskflowDialect/Orchestration/orchestration_utils.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>
#include <string>

namespace mlir {
namespace taskflow {

// Mapper-produced performance estimate for one composed-CGRA option.
struct TaskProfile {
  int composed_cgra_count = 1;
  std::string composed_cgra_shape = "1x1";
  int compiled_ii = 1;
  int steps = 1;
  int sample_trip_count = 1;
  int materialized_operation_count = 1;
  int estimated_latency = 1;
  bool mapper_succeeded = false;
};

using TaskProfileMap =
    llvm::DenseMap<TaskflowTaskOp, llvm::SmallVector<TaskProfile>>;

// Produces per-task performance profiles used by throughput-guided
// orchestration.
class TaskProfiler {
public:
  explicit TaskProfiler(int max_composed_cgra_count = 4,
                        int symbol_bound_trip_count = 256)
      : max_composed_cgra_count_(max_composed_cgra_count),
        symbol_bound_trip_count_(symbol_bound_trip_count) {}

  // Profiles every task in the function and keeps resource profiles in memory.
  // These profiles are internal search inputs and are not emitted to IR.
  TaskProfileMap profileFunction(func::FuncOp func) const;

  // Profiles one task across rectangular composed-CGRA options by invoking
  // the Neura mapper on the task's kernel. Symbol-bound counters use the
  // profiler's configured sample trip count.
  llvm::SmallVector<TaskProfile> profileTask(TaskflowTaskOp task) const;

private:
  // Profiles one task on the given rectangular composed-CGRA.
  std::optional<TaskProfile>
  profileTaskOnComposedCgra(TaskflowTaskOp task, const CgraShape &shape,
                            int composed_cgra_count) const;

  // Invokes the Neura mapper for one task profile and extracts compiled II and
  // step count, and materialized operation count.
  bool runMapperForTaskProfile(TaskflowTaskOp task, const CgraShape &shape,
                               int &compiled_ii, int &steps,
                               int &materialized_operation_count,
                               bool &mapper_succeeded) const;

  int max_composed_cgra_count_;
  int symbol_bound_trip_count_;
};

} // namespace taskflow
} // namespace mlir

#endif // TASKFLOW_THROUGHPUT_GUIDED_TASK_PROFILER_H
