// Implements the throughput-guided task orchestration strategy.

#include "TaskflowDialect/Orchestration/ThroughputGuidedTaskOrchestration/ThroughputGuidedTaskOrchestration.h"
#include "TaskflowDialect/Orchestration/ThroughputGuidedTaskOrchestration/ResourceAssignmentState.h"
#include "TaskflowDialect/Orchestration/ThroughputGuidedTaskOrchestration/TaskProfiler.h"
#include "TaskflowDialect/Orchestration/orchestration_utils.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/ArrayRef.h"

#include <optional>
#include <string>

namespace mlir {
namespace taskflow {

namespace {

struct OrchestrationMove {
  enum class Kind { ExpandComposedCgra, IncreaseReplica };

  Kind kind = Kind::ExpandComposedCgra;
  int task_index = -1;
  int profile_index = -1;
  int replica_count = 1;
};

std::optional<TaskPipelineIntervalResult> evaluateResourceAssignment(
    func::FuncOp func, ResourceAssignmentState &assignment_state, int grid_rows,
    int grid_cols, SchedulingMode mode, int max_contexts_per_cgra,
    llvm::StringRef evaluation_name,
    llvm::ArrayRef<TaskflowTaskOp> priority_path = {}) {
  assignment_state.applyCurrentResourceAssignment();

  TaskPriorityMap priority = assignment_state.buildTaskPriority(priority_path);
  TaskScheduler scheduler(grid_rows, grid_cols, mode, max_contexts_per_cgra);
  if (!scheduler.schedule(func, priority)) {
    func.emitError() << evaluation_name << " task schedule failed";
    return std::nullopt;
  }

  return TaskPipelineIntervalAnalyzer(scheduler.getScheduleResult()).analyze();
}

llvm::SmallVector<OrchestrationMove>
buildBottleneckMoves(const ResourceAssignmentState &assignment_state,
                     const TaskPipelineIntervalResult &interval_result) {
  llvm::SmallVector<OrchestrationMove> moves;
  if (!interval_result.bottleneck_task) {
    return moves;
  }

  int task_index =
      assignment_state.getTaskIndex(interval_result.bottleneck_task);
  if (task_index < 0) {
    return moves;
  }

  // std::optional<int> profile_index =
  //     assignment_state.getNextLargerComposedCgraProfile(task_index);
  // if (profile_index) {
  //   OrchestrationMove move;
  //   move.kind = OrchestrationMove::Kind::ExpandComposedCgra;
  //   move.task_index = task_index;
  //   move.profile_index = *profile_index;
  //   moves.push_back(move);
  // }

  // std::optional<int> replica_count =
  //     assignment_state.getNextReplicaCount(task_index);
  // if (replica_count) {
  //   OrchestrationMove move;
  //   move.kind = OrchestrationMove::Kind::IncreaseReplica;
  //   move.task_index = task_index;
  //   move.replica_count = *replica_count;
  //   moves.push_back(move);
  // }

  return moves;
}

void applyMove(ResourceAssignmentState &assignment_state,
               const OrchestrationMove &move) {
  if (move.kind == OrchestrationMove::Kind::ExpandComposedCgra) {
    assignment_state.selectComposedCgraProfile(move.task_index,
                                               move.profile_index);
    return;
  }

  assignment_state.selectReplicaCount(move.task_index, move.replica_count);
}

std::string getTaskName(TaskflowTaskOp task) {
  if (!task) {
    return "";
  }
  if (auto task_name = task->getAttrOfType<StringAttr>("task_name")) {
    return task_name.getValue().str();
  }
  return task->getName().getStringRef().str();
}

void emitTaskOrchestrationSummary(
    func::FuncOp func, const TaskPipelineIntervalResult &interval_result) {
  OpBuilder builder(func.getContext());

  llvm::SmallVector<Attribute> critical_path;
  for (TaskflowTaskOp task : interval_result.critical_path) {
    critical_path.push_back(builder.getStringAttr(getTaskName(task)));
  }

  llvm::SmallVector<NamedAttribute> summary_attrs;
  summary_attrs.push_back(NamedAttribute(
      builder.getStringAttr("bottleneck_task"),
      builder.getStringAttr(getTaskName(interval_result.bottleneck_task))));
  summary_attrs.push_back(NamedAttribute(builder.getStringAttr("critical_path"),
                                         builder.getArrayAttr(critical_path)));
  summary_attrs.push_back(NamedAttribute(
      builder.getStringAttr("pipeline_interval"),
      builder.getI64IntegerAttr(interval_result.pipeline_interval)));
  summary_attrs.push_back(
      NamedAttribute(builder.getStringAttr("strategy"),
                     builder.getStringAttr("throughput-guided")));

  func->setAttr("task_orchestration_summary",
                DictionaryAttr::get(func.getContext(), summary_attrs));
}

} // namespace

bool ThroughputGuidedTaskOrchestration::runTaskOrchestration(
    func::FuncOp func) {
  TaskProfileMap profile_map;
  if (task_profile_json_.empty()) {
    TaskProfiler profiler;
    profile_map = profiler.profileFunction(func);
  } else {
    FailureOr<TaskProfileMap> cached_profile_map =
        TaskProfiler::readTaskProfileMapFromJson(func, task_profile_json_);
    if (failed(cached_profile_map)) {
      return false;
    }
    profile_map = std::move(*cached_profile_map);
  }

  ResourceAssignmentState assignment_state(func, profile_map);
  std::optional<TaskPipelineIntervalResult> current =
      evaluateResourceAssignment(func, assignment_state, grid_rows_, grid_cols_,
                                 mode_, max_contexts_per_cgra_,
                                 "initial throughput-guided");
  if (!current) {
    return false;
  }

  constexpr int kMaxMoveIterations = 16;
  for (int iter = 0; iter < kMaxMoveIterations; ++iter) {
    llvm::SmallVector<OrchestrationMove> moves =
        buildBottleneckMoves(assignment_state, *current);
    if (moves.empty()) {
      break;
    }

    std::optional<OrchestrationMove> best_move;
    std::optional<TaskPipelineIntervalResult> best_result;
    int best_interval = current->pipeline_interval;

    for (const OrchestrationMove &move : moves) {
      ResourceAssignmentState speculative_state = assignment_state;
      applyMove(speculative_state, move);
      std::optional<TaskPipelineIntervalResult> speculative =
          evaluateResourceAssignment(func, speculative_state, grid_rows_,
                                     grid_cols_, mode_, max_contexts_per_cgra_,
                                     "speculative throughput-guided",
                                     current->critical_path);
      if (!speculative) {
        continue;
      }
      if (speculative->pipeline_interval < best_interval) {
        best_interval = speculative->pipeline_interval;
        best_move = move;
        best_result = std::move(speculative);
      }
    }

    if (!best_move) {
      break;
    }

    applyMove(assignment_state, *best_move);
    current = std::move(best_result);
  }

  std::optional<TaskPipelineIntervalResult> final_result =
      evaluateResourceAssignment(func, assignment_state, grid_rows_, grid_cols_,
                                 mode_, max_contexts_per_cgra_,
                                 "final throughput-guided",
                                 current->critical_path);
  if (!final_result) {
    return false;
  }

  emitTaskOrchestrationSummary(func, *final_result);
  return true;
}

} // namespace taskflow
} // namespace mlir
