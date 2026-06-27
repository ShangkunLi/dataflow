// Implements throughput-guided resource assignment state.

#include "TaskflowDialect/Orchestration/ThroughputGuidedTaskOrchestration/ResourceAssignmentState.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "mlir/IR/Attributes.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>
#include <optional>

namespace mlir {
namespace taskflow {

static bool getBoolAttr(Operation *op, StringRef attr_name) {
  if (auto attr = op->getAttrOfType<BoolAttr>(attr_name)) {
    return attr.getValue();
  }
  return false;
}

static int getOneCgraProfileIndex(llvm::ArrayRef<TaskProfile> profiles) {
  int one_cgra_profile_index = -1;
  for (auto [index, profile] : llvm::enumerate(profiles)) {
    if (profile.composed_cgra_count != 1) {
      continue;
    }
    assert(one_cgra_profile_index < 0 &&
           "A task should have exactly one 1-CGRA profile.\n");
    one_cgra_profile_index = static_cast<int>(index);
  }
  assert(one_cgra_profile_index >= 0 &&
         "Initial resource assignment requires a 1-CGRA profile.\n");
  return one_cgra_profile_index;
}

static TaskResourceAssignment
buildTaskResourceAssignment(TaskflowTaskOp task,
                            ArrayRef<TaskProfile> profiles) {
  Operation *op = task.getOperation();
  assert(!profiles.empty() &&
         "Resource assignment requires TaskProfiler profiles.\n");
  TaskResourceAssignment assignment;
  assignment.task = op;
  assignment.dlp_replicable = getBoolAttr(op, "dlp_replicable");
  assignment.task_profiles.assign(profiles.begin(), profiles.end());
  assignment.selected_profile_index =
      getOneCgraProfileIndex(assignment.task_profiles);
  const TaskProfile &profile =
      assignment.task_profiles[assignment.selected_profile_index];
  assignment.composed_cgra_count = profile.composed_cgra_count;
  assignment.composed_cgra_shape = profile.composed_cgra_shape;
  assignment.estimated_latency = profile.estimated_latency;
  return assignment;
}

ResourceAssignmentState::ResourceAssignmentState(
    func::FuncOp func, const TaskProfileMap &profile_map) {
  int task_index = 0;
  func.walk([&](TaskflowTaskOp task) {
    auto it = profile_map.find(task);
    ArrayRef<TaskProfile> profiles;
    if (it != profile_map.end()) {
      profiles = it->second;
    }
    resource_assignments_.push_back(
        buildTaskResourceAssignment(task, profiles));
    task_to_assignment_index_[task.getOperation()] = task_index++;
  });
}

TaskPriorityMap ResourceAssignmentState::buildTaskPriority(
    ArrayRef<TaskflowTaskOp> priority_path) const {
  TaskPriorityMap priority;
  // Initialization.
  for (const TaskResourceAssignment &assignment : resource_assignments_) {
    priority[assignment.task] = 0;
  }

  int path_priority = static_cast<int>(priority_path.size());
  for (TaskflowTaskOp task : priority_path) {
    priority[task.getOperation()] = path_priority--;
  }
  return priority;
}

int ResourceAssignmentState::getTaskIndex(TaskflowTaskOp task) const {
  auto it = task_to_assignment_index_.find(task.getOperation());
  if (it == task_to_assignment_index_.end()) {
    return -1;
  }
  return it->second;
}

std::optional<int> ResourceAssignmentState::getNextLargerComposedCgraProfile(
    int task_index) const {
  if (task_index < 0 ||
      task_index >= static_cast<int>(resource_assignments_.size())) {
    return std::nullopt;
  }

  const TaskResourceAssignment &assignment = resource_assignments_[task_index];
  int current_count = assignment.composed_cgra_count;
  int next_count = std::numeric_limits<int>::max();
  int next_index = -1;
  int next_latency = std::numeric_limits<int>::max();

  for (auto [index, profile] : llvm::enumerate(assignment.task_profiles)) {
    if (profile.composed_cgra_count <= current_count) {
      continue;
    }
    if (profile.composed_cgra_count > next_count) {
      continue;
    }

    bool better_count = profile.composed_cgra_count < next_count;
    bool better_latency = profile.composed_cgra_count == next_count &&
                          profile.estimated_latency < next_latency;
    if (better_count || better_latency) {
      next_count = profile.composed_cgra_count;
      next_latency = profile.estimated_latency;
      next_index = static_cast<int>(index);
    }
  }

  if (next_index < 0) {
    return std::nullopt;
  }
  return next_index;
}

void ResourceAssignmentState::selectComposedCgraProfile(int task_index,
                                                        int profile_index) {
  TaskResourceAssignment &assignment = resource_assignments_[task_index];
  assignment.selected_profile_index = profile_index;
  const TaskProfile &profile = assignment.task_profiles[profile_index];
  assignment.composed_cgra_count = profile.composed_cgra_count;
  assignment.composed_cgra_shape = profile.composed_cgra_shape;
  assignment.estimated_latency = profile.estimated_latency;
}

std::optional<int>
ResourceAssignmentState::getNextReplicaCount(int task_index) const {
  if (task_index < 0 ||
      task_index >= static_cast<int>(resource_assignments_.size())) {
    return std::nullopt;
  }

  const TaskResourceAssignment &assignment = resource_assignments_[task_index];
  if (!assignment.dlp_replicable) {
    return std::nullopt;
  }

  constexpr int kMaxStaticReplicas = 4;
  if (assignment.active_replicas >= kMaxStaticReplicas) {
    return std::nullopt;
  }
  return assignment.active_replicas + 1;
}

void ResourceAssignmentState::selectReplicaCount(int task_index,
                                                 int replica_count) {
  TaskResourceAssignment &assignment = resource_assignments_[task_index];
  assignment.active_replicas = std::max(1, replica_count);
}

void ResourceAssignmentState::applyCurrentResourceAssignment() const {
  for (const TaskResourceAssignment &assignment : resource_assignments_) {
    Operation *task = assignment.task;
    if (!task) {
      continue;
    }

    OpBuilder builder(task->getContext());
    task->removeAttr("task_candidate_profiles");
    task->removeAttr("compiled_ii");
    task->removeAttr("sample_trip_count");

    task->setAttr("cgra_count",
                  builder.getI32IntegerAttr(assignment.composed_cgra_count));
    task->setAttr("cgra_shape",
                  builder.getStringAttr(assignment.composed_cgra_shape));
    task->setAttr("active_replicas",
                  builder.getI32IntegerAttr(assignment.active_replicas));

    llvm::SmallVector<NamedAttribute> profile_attrs;
    const TaskProfile *profile = nullptr;
    if (!assignment.task_profiles.empty()) {
      profile = &assignment.task_profiles[assignment.selected_profile_index];
    }
    int compiled_ii = profile ? profile->compiled_ii : 1;
    int single_replica_duration =
        profile ? profile->estimated_latency : assignment.estimated_latency;
    int duration =
        std::max(1, (single_replica_duration + assignment.active_replicas - 1) /
                        assignment.active_replicas);
    int sample_trip_count = profile ? profile->sample_trip_count : 1;
    int steps = profile ? profile->steps : 1;
    int materialized_operation_count =
        profile ? profile->materialized_operation_count : 1;
    profile_attrs.push_back(
        NamedAttribute(StringAttr::get(task->getContext(), "compiled_ii"),
                       builder.getI32IntegerAttr(compiled_ii)));
    profile_attrs.push_back(
        NamedAttribute(StringAttr::get(task->getContext(), "duration"),
                       builder.getI32IntegerAttr(duration)));
    profile_attrs.push_back(
        NamedAttribute(StringAttr::get(task->getContext(), "sample_trip_count"),
                       builder.getI32IntegerAttr(sample_trip_count)));
    profile_attrs.push_back(
        NamedAttribute(StringAttr::get(task->getContext(), "steps"),
                       builder.getI32IntegerAttr(steps)));
    profile_attrs.push_back(NamedAttribute(
        StringAttr::get(task->getContext(), "materialized_operation_count"),
        builder.getI32IntegerAttr(materialized_operation_count)));
    task->setAttr("profile_info",
                  DictionaryAttr::get(task->getContext(), profile_attrs));
  }
}

} // namespace taskflow
} // namespace mlir
