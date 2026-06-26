// Throughput-guided resource assignment state.

#ifndef TASKFLOW_THROUGHPUT_GUIDED_RESOURCE_ASSIGNMENT_STATE_H
#define TASKFLOW_THROUGHPUT_GUIDED_RESOURCE_ASSIGNMENT_STATE_H

#include "TaskflowDialect/Orchestration/ThroughputGuidedTaskOrchestration/TaskProfiler.h"
#include "TaskflowDialect/Orchestration/orchestration_utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>
#include <string>

namespace mlir {
namespace taskflow {

// Static resource choice currently assigned to one task.
//
// The throughput-guided search mutates these fields speculatively before
// re-running TaskScheduler. The initial state is intentionally conservative:
// one CGRA and one active replica per task.
struct TaskResourceAssignment {
  Operation *task = nullptr;

  // Number of adjacent CGRAs composed into one larger execution unit for this
  // task. A value of 1 means the task uses one standalone CGRA.
  int composed_cgra_count = 1;

  // Shape of the composed CGRA group. "1x1" means no CGRA stitching.
  std::string composed_cgra_shape = "1x1";

  int active_replicas = 1;
  int estimated_latency = 1;
  bool dlp_replicable = false;
  int selected_profile_index = 0;
  llvm::SmallVector<TaskProfile> task_profiles;
};

// Per-function static orchestration state used by throughput-guided search.
class ResourceAssignmentState {
public:
  // Creates one resource assignment for every task in `func` using profiler
  // results as the candidate search space.
  //
  // The baseline is intentionally conservative: one standalone CGRA and one
  // active replica per task.
  explicit ResourceAssignmentState(func::FuncOp func,
                                   const TaskProfileMap &profile_map);

  // Returns all per-task resource assignments in deterministic IR walk order.
  llvm::ArrayRef<TaskResourceAssignment> getResourceAssignments() const {
    return resource_assignments_;
  }

  // Returns the mutable assignment for a task by deterministic IR walk index.
  TaskResourceAssignment &getResourceAssignment(int task_index) {
    return resource_assignments_[task_index];
  }

  // Returns the number of tracked task resource assignments.
  int getResourceAssignmentCount() const {
    return static_cast<int>(resource_assignments_.size());
  }

  // Builds a task-priority map. Tasks on `priority_path` are scheduled first;
  // an empty path gives every task neutral priority.
  TaskPriorityMap
  buildTaskPriority(llvm::ArrayRef<TaskflowTaskOp> priority_path = {}) const;

  // Returns the deterministic assignment index for a task op.
  int getTaskIndex(TaskflowTaskOp task) const;

  // Returns the next profile that uses more adjacent CGRAs than the current
  // assignment. Among profiles with the same CGRA count, the fastest one is
  // selected.
  std::optional<int> getNextLargerComposedCgraProfile(int task_index) const;

  // Selects one profiled composed-CGRA option for the task.
  void selectComposedCgraProfile(int task_index, int profile_index);

  // Returns the next static replica count for a DLP-replicable task.
  std::optional<int> getNextReplicaCount(int task_index) const;

  // Selects the number of active static replicas for the task.
  void selectReplicaCount(int task_index, int replica_count);

  // Writes the current resource assignment to task attributes consumed by
  // TaskScheduler.
  void applyCurrentResourceAssignment() const;

private:
  llvm::SmallVector<TaskResourceAssignment> resource_assignments_;
  llvm::DenseMap<Operation *, int> task_to_assignment_index_;
};

} // namespace taskflow
} // namespace mlir

#endif // TASKFLOW_THROUGHPUT_GUIDED_RESOURCE_ASSIGNMENT_STATE_H
