// Visualize scheduled Taskflow orchestration.

#include "TaskflowDialect/Orchestration/orchestration_utils.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "TaskflowDialect/TaskflowPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>

using namespace mlir;
using namespace mlir::taskflow;

namespace {

struct CgraUse {
  int row = 0;
  int col = 0;
  int context_id = 0;
  int replica_id = 0;
};

struct VisualTask {
  TaskflowTaskOp task;
  std::string alias;
  std::string name;
  int duration = 1;
  int start_time = 0;
  int end_time = 1;
  double draw_start = 0.0;
  double draw_width = 1.0;
  int active_replicas = 1;
  SmallVector<CgraUse> cgra_uses;
  SmallVector<int> predecessors;
  SmallVector<int> successors;
};

struct Lane {
  int row = 0;
  int col = 0;
  std::string label;
};

struct TaskBlock {
  int replica_id = 0;
  int top_lane = 0;
  int bottom_lane = 0;
  int y = 0;
  int height = 0;
};

static std::string getTaskName(TaskflowTaskOp task) {
  if (auto task_name = task->getAttrOfType<StringAttr>("task_name")) {
    return task_name.getValue().str();
  }
  return task.getTaskName().str();
}

static int getIntegerAttr(Operation *op, StringRef attr_name,
                          int default_value) {
  if (auto attr = op->getAttrOfType<IntegerAttr>(attr_name)) {
    return static_cast<int>(attr.getInt());
  }
  return default_value;
}

static int getDictionaryIntegerAttr(DictionaryAttr dict, StringRef attr_name,
                                    int default_value) {
  if (!dict) {
    return default_value;
  }
  auto attr = dyn_cast_or_null<IntegerAttr>(dict.get(attr_name));
  if (!attr) {
    return default_value;
  }
  return static_cast<int>(attr.getInt());
}

static std::string escapeXml(StringRef text) {
  std::string escaped;
  llvm::raw_string_ostream os(escaped);
  for (char c : text) {
    switch (c) {
    case '&':
      os << "&amp;";
      break;
    case '<':
      os << "&lt;";
      break;
    case '>':
      os << "&gt;";
      break;
    case '"':
      os << "&quot;";
      break;
    default:
      os << c;
      break;
    }
  }
  return os.str();
}

static int64_t encodeCgraLocation(int row, int col) {
  return (static_cast<int64_t>(row) << 32) | static_cast<uint32_t>(col);
}

static SmallVector<CgraUse> parseCgraUses(TaskflowTaskOp task) {
  SmallVector<CgraUse> cgra_uses;
  auto orchestration_info =
      task->getAttrOfType<DictionaryAttr>("task_orchestration_info");
  if (!orchestration_info) {
    return cgra_uses;
  }

  auto cgra_positions =
      dyn_cast_or_null<ArrayAttr>(orchestration_info.get("cgra_positions"));
  if (!cgra_positions) {
    return cgra_uses;
  }

  for (Attribute attr : cgra_positions) {
    auto coord = dyn_cast<DictionaryAttr>(attr);
    if (!coord) {
      continue;
    }

    CgraUse cgra_use;
    cgra_use.row = getDictionaryIntegerAttr(coord, "row", 0);
    cgra_use.col = getDictionaryIntegerAttr(coord, "col", 0);
    cgra_use.context_id = getDictionaryIntegerAttr(coord, "context_id", 0);
    cgra_use.replica_id = getDictionaryIntegerAttr(coord, "replica_id", 0);
    cgra_uses.push_back(cgra_use);
  }
  return cgra_uses;
}

static VisualTask buildVisualTask(TaskflowTaskOp task, int task_index) {
  VisualTask visual_task;
  visual_task.task = task;
  visual_task.alias = "T" + std::to_string(task_index);
  visual_task.name = getTaskName(task);
  visual_task.active_replicas =
      getIntegerAttr(task.getOperation(), "active_replicas", 1);

  auto profile_info = task->getAttrOfType<DictionaryAttr>("profile_info");
  visual_task.duration = std::max(
      1, getDictionaryIntegerAttr(profile_info, "duration", 1));
  visual_task.end_time = visual_task.duration;
  visual_task.cgra_uses = parseCgraUses(task);
  return visual_task;
}

static SmallVector<VisualTask, 8> collectVisualTasks(func::FuncOp func) {
  SmallVector<VisualTask, 8> tasks;
  func.walk([&](TaskflowTaskOp task) {
    tasks.push_back(buildVisualTask(task, static_cast<int>(tasks.size())));
  });
  return tasks;
}

static void addScheduleEdge(int task, int next_task,
                            SmallVectorImpl<VisualTask> &tasks,
                            DenseSet<int64_t> &edge_keys) {
  if (task < 0 || next_task < 0 || task == next_task) {
    return;
  }

  int64_t edge_key =
      (static_cast<int64_t>(task) << 32) | static_cast<uint32_t>(next_task);
  if (!edge_keys.insert(edge_key).second) {
    return;
  }

  tasks[task].successors.push_back(next_task);
  tasks[next_task].predecessors.push_back(task);
}

static void buildDataDependenceEdges(SmallVectorImpl<VisualTask> &tasks) {
  DenseMap<Operation *, int> task_to_index;
  for (auto [index, task] : llvm::enumerate(tasks)) {
    task_to_index[task.task.getOperation()] = static_cast<int>(index);
  }

  DenseSet<int64_t> edge_keys;
  for (auto [task_idx, task] : llvm::enumerate(tasks)) {
    for (Value operand : task.task->getOperands()) {
      auto producer = operand.getDefiningOp<TaskflowTaskOp>();
      if (!producer) {
        continue;
      }
      auto it = task_to_index.find(producer.getOperation());
      if (it == task_to_index.end()) {
        continue;
      }
      addScheduleEdge(it->second, static_cast<int>(task_idx), tasks,
                      edge_keys);
    }
  }
}

static int getTaskContextOnCgra(const VisualTask &task, int row, int col) {
  for (const CgraUse &cgra_use : task.cgra_uses) {
    if (cgra_use.row == row && cgra_use.col == col) {
      return cgra_use.context_id;
    }
  }
  return 0;
}

static void buildContextOrderEdges(SmallVectorImpl<VisualTask> &tasks) {
  DenseSet<int64_t> edge_keys;
  for (auto [task_idx, task] : llvm::enumerate(tasks)) {
    for (int successor : task.successors) {
      edge_keys.insert((static_cast<int64_t>(task_idx) << 32) |
                       static_cast<uint32_t>(successor));
    }
  }

  DenseMap<int64_t, SmallVector<int>> cgra_to_tasks;
  for (auto [task_idx, task] : llvm::enumerate(tasks)) {
    for (const CgraUse &cgra_use : task.cgra_uses) {
      cgra_to_tasks[encodeCgraLocation(cgra_use.row, cgra_use.col)].push_back(
          static_cast<int>(task_idx));
    }
  }

  for (auto &entry : cgra_to_tasks) {
    SmallVector<int> &tasks_on_cgra = entry.second;
    int row = static_cast<int>(entry.first >> 32);
    int col = static_cast<int>(static_cast<uint32_t>(entry.first));
    llvm::sort(tasks_on_cgra, [&](int lhs, int rhs) {
      int lhs_context = getTaskContextOnCgra(tasks[lhs], row, col);
      int rhs_context = getTaskContextOnCgra(tasks[rhs], row, col);
      if (lhs_context != rhs_context) {
        return lhs_context < rhs_context;
      }
      return lhs < rhs;
    });

    for (size_t i = 1; i < tasks_on_cgra.size(); ++i) {
      addScheduleEdge(tasks_on_cgra[i - 1], tasks_on_cgra[i], tasks,
                      edge_keys);
    }
  }
}

static LogicalResult computeTaskTimes(func::FuncOp func,
                                      SmallVectorImpl<VisualTask> &tasks) {
  SmallVector<int> predecessor_count(tasks.size(), 0);
  for (auto [task_idx, task] : llvm::enumerate(tasks)) {
    predecessor_count[task_idx] = task.predecessors.size();
  }

  SmallVector<int> ready_tasks;
  for (auto [task_idx, count] : llvm::enumerate(predecessor_count)) {
    if (count == 0) {
      ready_tasks.push_back(static_cast<int>(task_idx));
    }
  }

  size_t processed_tasks = 0;
  while (!ready_tasks.empty()) {
    int task_idx = ready_tasks.pop_back_val();
    ++processed_tasks;

    VisualTask &task = tasks[task_idx];
    task.end_time = task.start_time + task.duration;
    for (int successor : task.successors) {
      tasks[successor].start_time =
          std::max(tasks[successor].start_time, task.end_time);
      --predecessor_count[successor];
      if (predecessor_count[successor] == 0) {
        ready_tasks.push_back(successor);
      }
    }
  }

  if (processed_tasks != tasks.size()) {
    return func.emitError()
           << "cannot visualize task orchestration because task dependences "
              "and CGRA context order form a cycle";
  }

  return success();
}

static LogicalResult prepareSchedule(func::FuncOp func,
                                     SmallVectorImpl<VisualTask> &tasks) {
  buildDataDependenceEdges(tasks);
  buildContextOrderEdges(tasks);
  return computeTaskTimes(func, tasks);
}

static SmallVector<Lane> collectUsedLanes(ArrayRef<VisualTask> tasks) {
  DenseSet<int64_t> seen_lanes;
  SmallVector<Lane> lanes;
  for (const VisualTask &task : tasks) {
    for (const CgraUse &cgra_use : task.cgra_uses) {
      int64_t key = encodeCgraLocation(cgra_use.row, cgra_use.col);
      if (!seen_lanes.insert(key).second) {
        continue;
      }
      Lane lane;
      lane.row = cgra_use.row;
      lane.col = cgra_use.col;
      lane.label = "CGRA(" + std::to_string(cgra_use.row) + "," +
                   std::to_string(cgra_use.col) + ")";
      lanes.push_back(std::move(lane));
    }
  }

  llvm::sort(lanes, [](const Lane &lhs, const Lane &rhs) {
    if (lhs.row != rhs.row) {
      return lhs.row < rhs.row;
    }
    return lhs.col < rhs.col;
  });
  return lanes;
}

static int findLaneIndex(ArrayRef<Lane> lanes, int row, int col) {
  for (auto [index, lane] : llvm::enumerate(lanes)) {
    if (lane.row == row && lane.col == col) {
      return static_cast<int>(index);
    }
  }
  return -1;
}

static int getTaskTopLane(const VisualTask &task, ArrayRef<Lane> lanes) {
  int top_lane = std::numeric_limits<int>::max();
  for (const CgraUse &cgra_use : task.cgra_uses) {
    int lane = findLaneIndex(lanes, cgra_use.row, cgra_use.col);
    if (lane >= 0) {
      top_lane = std::min(top_lane, lane);
    }
  }
  return top_lane == std::numeric_limits<int>::max() ? 0 : top_lane;
}

static int getTaskBottomLane(const VisualTask &task, ArrayRef<Lane> lanes) {
  int bottom_lane = 0;
  for (const CgraUse &cgra_use : task.cgra_uses) {
    int lane = findLaneIndex(lanes, cgra_use.row, cgra_use.col);
    if (lane >= 0) {
      bottom_lane = std::max(bottom_lane, lane);
    }
  }
  return bottom_lane;
}

static SmallVector<int> collectTaskReplicaIds(const VisualTask &task) {
  SmallVector<int> replica_ids;
  DenseSet<int> seen_replica_ids;
  if (task.cgra_uses.empty()) {
    replica_ids.push_back(0);
    return replica_ids;
  }

  for (const CgraUse &cgra_use : task.cgra_uses) {
    if (seen_replica_ids.insert(cgra_use.replica_id).second) {
      replica_ids.push_back(cgra_use.replica_id);
    }
  }
  llvm::sort(replica_ids);
  return replica_ids;
}

static std::pair<int, int>
getTaskLaneRangeForReplica(const VisualTask &task, ArrayRef<Lane> lanes,
                           int replica_id) {
  int top_lane = std::numeric_limits<int>::max();
  int bottom_lane = 0;
  for (const CgraUse &cgra_use : task.cgra_uses) {
    if (cgra_use.replica_id != replica_id) {
      continue;
    }
    int lane = findLaneIndex(lanes, cgra_use.row, cgra_use.col);
    if (lane < 0) {
      continue;
    }
    top_lane = std::min(top_lane, lane);
    bottom_lane = std::max(bottom_lane, lane);
  }

  if (top_lane == std::numeric_limits<int>::max()) {
    return {0, 0};
  }
  return {top_lane, bottom_lane};
}

static TaskBlock buildTaskBlock(const VisualTask &task, ArrayRef<Lane> lanes,
                                int replica_id, int top_margin,
                                int lane_height, int task_height) {
  auto [top_lane, bottom_lane] =
      getTaskLaneRangeForReplica(task, lanes, replica_id);
  TaskBlock block;
  block.replica_id = replica_id;
  block.top_lane = top_lane;
  block.bottom_lane = bottom_lane;
  block.y = top_margin + top_lane * lane_height + 12;
  block.height = task_height + (bottom_lane - top_lane) * lane_height;
  return block;
}

static TaskBlock buildTaskBlockFromLaneRange(int replica_id, int top_lane,
                                             int bottom_lane, int top_margin,
                                             int lane_height,
                                             int task_height) {
  TaskBlock block;
  block.replica_id = replica_id;
  block.top_lane = top_lane;
  block.bottom_lane = bottom_lane;
  block.y = top_margin + top_lane * lane_height + 12;
  block.height = task_height + (bottom_lane - top_lane) * lane_height;
  return block;
}

static SmallVector<int> collectTaskLanesForReplica(const VisualTask &task,
                                                   ArrayRef<Lane> lanes,
                                                   int replica_id) {
  SmallVector<int> task_lanes;
  DenseSet<int> seen_lanes;
  for (const CgraUse &cgra_use : task.cgra_uses) {
    if (cgra_use.replica_id != replica_id) {
      continue;
    }
    int lane = findLaneIndex(lanes, cgra_use.row, cgra_use.col);
    if (lane >= 0 && seen_lanes.insert(lane).second) {
      task_lanes.push_back(lane);
    }
  }
  llvm::sort(task_lanes);
  return task_lanes;
}

static SmallVector<TaskBlock> buildTaskBlocks(const VisualTask &task,
                                              ArrayRef<Lane> lanes,
                                              int top_margin,
                                              int lane_height,
                                              int task_height) {
  SmallVector<TaskBlock> blocks;
  for (int replica_id : collectTaskReplicaIds(task)) {
    SmallVector<int> task_lanes =
        collectTaskLanesForReplica(task, lanes, replica_id);
    if (task_lanes.empty()) {
      blocks.push_back(buildTaskBlock(task, lanes, replica_id, top_margin,
                                      lane_height, task_height));
      continue;
    }

    int run_start = task_lanes.front();
    int run_end = run_start;
    for (size_t i = 1; i < task_lanes.size(); ++i) {
      int lane = task_lanes[i];
      if (lane == run_end + 1) {
        run_end = lane;
        continue;
      }

      blocks.push_back(buildTaskBlockFromLaneRange(
          replica_id, run_start, run_end, top_margin, lane_height,
          task_height));
      run_start = lane;
      run_end = lane;
    }
    blocks.push_back(buildTaskBlockFromLaneRange(
        replica_id, run_start, run_end, top_margin, lane_height, task_height));
  }
  return blocks;
}

static TaskBlock buildTaskOverallBlock(const VisualTask &task,
                                       ArrayRef<Lane> lanes, int top_margin,
                                       int lane_height, int task_height) {
  int top_lane = getTaskTopLane(task, lanes);
  int bottom_lane = getTaskBottomLane(task, lanes);
  TaskBlock block;
  block.replica_id = 0;
  block.top_lane = top_lane;
  block.bottom_lane = bottom_lane;
  block.y = top_margin + top_lane * lane_height + 12;
  block.height = task_height + (bottom_lane - top_lane) * lane_height;
  return block;
}

static TaskBlock getIncomingAnchorBlock(const VisualTask &task,
                                        ArrayRef<Lane> lanes, int top_margin,
                                        int lane_height, int task_height) {
  SmallVector<TaskBlock> blocks =
      buildTaskBlocks(task, lanes, top_margin, lane_height, task_height);
  if (!blocks.empty()) {
    return blocks.front();
  }
  return buildTaskOverallBlock(task, lanes, top_margin, lane_height,
                               task_height);
}

static TaskBlock getOutgoingAnchorBlock(const VisualTask &task,
                                        ArrayRef<Lane> lanes, int top_margin,
                                        int lane_height, int task_height) {
  SmallVector<TaskBlock> blocks =
      buildTaskBlocks(task, lanes, top_margin, lane_height, task_height);
  if (!blocks.empty()) {
    return blocks.back();
  }
  return buildTaskOverallBlock(task, lanes, top_margin, lane_height,
                               task_height);
}

static const char *getTaskColor(int task_index) {
  static constexpr const char *kColors[] = {
      "#fff4a8", "#ffd7a8", "#bfeaf6", "#f1a6a6", "#c8e8b8",
      "#d9b7f0", "#f6c1d1", "#b8d6ff", "#d9d9d9", "#c7f0df"};
  return kColors[task_index % (sizeof(kColors) / sizeof(kColors[0]))];
}

static int getMaxEndTime(ArrayRef<VisualTask> tasks) {
  int max_end_time = 1;
  for (const VisualTask &task : tasks) {
    max_end_time = std::max(max_end_time, task.end_time);
  }
  return max_end_time;
}

static void computeProportionalTimelineLayout(
    SmallVectorImpl<VisualTask> &tasks, double time_scale) {
  // Keep the SVG timeline faithful to the scheduled latency model. Task box
  // width is exactly duration * time_scale, so a task with 256x larger
  // duration is drawn 256x wider. Labels may be moved outside tiny boxes, but
  // the boxes themselves are never widened for readability.
  for (VisualTask &task : tasks) {
    task.draw_start = static_cast<double>(task.start_time) * time_scale;
    task.draw_width = static_cast<double>(task.duration) * time_scale;
  }
}

static double getMaxDrawEnd(ArrayRef<VisualTask> tasks) {
  double max_draw_end = 1.0;
  for (const VisualTask &task : tasks) {
    max_draw_end = std::max(max_draw_end, task.draw_start + task.draw_width);
  }
  return max_draw_end;
}

static void emitSvg(ArrayRef<VisualTask> tasks, func::FuncOp func,
                    raw_ostream &os) {
  SmallVector<Lane> lanes = collectUsedLanes(tasks);
  if (lanes.empty()) {
    lanes.push_back({0, 0, "CGRA(0,0)"});
  }

  constexpr int kLeftMargin = 135;
  constexpr int kRightMargin = 50;
  constexpr int kTopMargin = 90;
  constexpr int kLaneHeight = 78;
  constexpr int kBottomMargin = 70;
  constexpr int kBaseTimelineWidth = 900;
  constexpr int kTaskHeight = 46;
  constexpr int kConfigWidth = 76;

  int max_end_time = getMaxEndTime(tasks);
  double time_scale = static_cast<double>(kBaseTimelineWidth - kConfigWidth - 40) /
                      static_cast<double>(std::max(1, max_end_time));
  auto mutable_tasks = SmallVector<VisualTask, 8>(tasks.begin(), tasks.end());
  computeProportionalTimelineLayout(mutable_tasks, time_scale);
  tasks = mutable_tasks;

  int timeline_width = std::max(
      kBaseTimelineWidth,
      static_cast<int>(std::ceil(kConfigWidth + 120 + getMaxDrawEnd(tasks))));
  int width = kLeftMargin + timeline_width + kRightMargin;
  int height = kTopMargin + static_cast<int>(lanes.size()) * kLaneHeight +
               kBottomMargin;

  os << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << width
     << "\" height=\"" << height << "\" viewBox=\"0 0 " << width << " "
     << height << "\">\n";
  os << "  <defs>\n";
  os << "    <marker id=\"arrow\" markerWidth=\"10\" markerHeight=\"10\" "
        "refX=\"8\" refY=\"3\" orient=\"auto\" markerUnits=\"strokeWidth\">\n";
  os << "      <path d=\"M0,0 L0,6 L9,3 z\" fill=\"#d11\"/>\n";
  os << "    </marker>\n";
  os << "    <pattern id=\"configHatch\" patternUnits=\"userSpaceOnUse\" "
        "width=\"8\" height=\"8\" patternTransform=\"rotate(45)\">\n";
  os << "      <rect width=\"8\" height=\"8\" fill=\"#f7f7f7\"/>\n";
  os << "      <line x1=\"0\" y1=\"0\" x2=\"0\" y2=\"8\" "
        "stroke=\"#9e9e9e\" stroke-width=\"3\"/>\n";
  os << "    </pattern>\n";
  os << "  </defs>\n";
  os << "  <rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n";
  os << "  <text x=\"" << kLeftMargin
     << "\" y=\"32\" font-family=\"sans-serif\" font-size=\"22\" "
        "font-weight=\"700\">"
     << escapeXml(func.getSymName()) << "</text>\n";

  for (auto [lane_idx, lane] : llvm::enumerate(lanes)) {
    int y = kTopMargin + static_cast<int>(lane_idx) * kLaneHeight;
    os << "  <text x=\"24\" y=\"" << (y + 43)
       << "\" font-family=\"sans-serif\" font-size=\"20\" "
          "font-weight=\"700\">"
       << escapeXml(lane.label) << "</text>\n";
    os << "  <line x1=\"" << kLeftMargin << "\" y1=\"" << y
       << "\" x2=\"" << (kLeftMargin + timeline_width) << "\" y2=\"" << y
       << "\" stroke=\"#b5b5b5\" stroke-dasharray=\"8 8\"/>\n";
  }

  int axis_y = kTopMargin + static_cast<int>(lanes.size()) * kLaneHeight + 10;
  os << "  <line x1=\"" << kLeftMargin << "\" y1=\"" << (kTopMargin - 28)
     << "\" x2=\"" << kLeftMargin << "\" y2=\"" << axis_y
     << "\" stroke=\"black\" stroke-width=\"3\"/>\n";
  os << "  <line x1=\"" << kLeftMargin << "\" y1=\"" << axis_y
     << "\" x2=\"" << (kLeftMargin + timeline_width) << "\" y2=\"" << axis_y
     << "\" stroke=\"black\" stroke-width=\"3\"/>\n";
  os << "  <path d=\"M" << kLeftMargin << " " << (kTopMargin - 40)
     << " L" << (kLeftMargin - 10) << " " << (kTopMargin - 18) << " L"
     << (kLeftMargin + 10) << " " << (kTopMargin - 18)
     << " Z\" fill=\"black\"/>\n";
  os << "  <path d=\"M" << (kLeftMargin + timeline_width + 18) << " "
     << axis_y << " L" << (kLeftMargin + timeline_width - 8) << " "
     << (axis_y - 10) << " L" << (kLeftMargin + timeline_width - 8) << " "
     << (axis_y + 10) << " Z\" fill=\"black\"/>\n";
  os << "  <text x=\"" << (kLeftMargin + timeline_width - 58) << "\" y=\""
     << (axis_y + 34)
     << "\" font-family=\"sans-serif\" font-size=\"24\" "
        "font-weight=\"700\">Time</text>\n";

  int config_x = kLeftMargin + 12;
  int config_y = kTopMargin - 18;
  int config_h = static_cast<int>(lanes.size()) * kLaneHeight - 12;
  os << "  <rect x=\"" << config_x << "\" y=\"" << config_y
     << "\" width=\"" << kConfigWidth << "\" height=\"" << config_h
     << "\" rx=\"10\" fill=\"url(#configHatch)\" stroke=\"black\" "
        "stroke-width=\"2\"/>\n";
  os << "  <text x=\"" << (config_x + 44) << "\" y=\""
     << (config_y + config_h / 2)
     << "\" transform=\"rotate(-90 " << (config_x + 44) << " "
     << (config_y + config_h / 2)
     << ")\" font-family=\"sans-serif\" font-size=\"16\" "
        "font-weight=\"700\">Config</text>\n";

  auto task_x = [&](const VisualTask &task) {
    return kLeftMargin + kConfigWidth + 34 + task.draw_start;
  };
  auto task_w = [&](const VisualTask &task) {
    return task.draw_width;
  };

  for (auto [task_idx, task] : llvm::enumerate(tasks)) {
    double x = task_x(task);
    double w = task_w(task);
    SmallVector<TaskBlock> blocks =
        buildTaskBlocks(task, lanes, kTopMargin, kLaneHeight, kTaskHeight);
    for (const TaskBlock &block : blocks) {
      os << "  <rect x=\"" << x << "\" y=\"" << block.y << "\" width=\""
         << w << "\" height=\"" << block.height
         << "\" rx=\"8\" fill=\"" << getTaskColor(task_idx)
         << "\" stroke=\"black\" stroke-width=\"2\"/>\n";
      std::string label = task.alias;
      if (task.active_replicas > 1) {
        label += ".r" + std::to_string(block.replica_id);
      }
      if (w >= 34.0) {
        os << "  <text x=\"" << (x + w / 2) << "\" y=\""
           << (block.y + block.height / 2 + 6)
           << "\" text-anchor=\"middle\" font-family=\"sans-serif\" "
              "font-size=\"16\" font-weight=\"700\">"
           << escapeXml(label) << "</text>\n";
      } else {
        os << "  <text x=\"" << (x + w + 4) << "\" y=\""
           << (block.y + 16)
           << "\" font-family=\"sans-serif\" font-size=\"13\" "
              "font-weight=\"700\">"
           << escapeXml(label) << "</text>\n";
      }
      os << "  <title>" << escapeXml(task.name)
         << " replica=" << block.replica_id << " start=" << task.start_time
         << " end=" << task.end_time << " duration=" << task.duration
         << " replicas=" << task.active_replicas << "</title>\n";
    }
  }

  for (auto [task_idx, task] : llvm::enumerate(tasks)) {
    double x1 = task_x(task) + task_w(task);
    TaskBlock task_block =
        getOutgoingAnchorBlock(task, lanes, kTopMargin, kLaneHeight,
                               kTaskHeight);
    int y1 = task_block.y + task_block.height / 2;
    for (int successor : task.successors) {
      const VisualTask &next_task = tasks[successor];
      double x2 = task_x(next_task);
      TaskBlock next_task_block =
          getIncomingAnchorBlock(next_task, lanes, kTopMargin, kLaneHeight,
                                 kTaskHeight);
      int y2 = next_task_block.y + next_task_block.height / 2;
      double mid_x = (x1 + x2) / 2.0;
      os << "  <path d=\"M" << x1 << " " << y1 << " C" << mid_x << " "
         << y1 << ", " << mid_x << " " << y2 << ", " << x2 << " " << y2
         << "\" fill=\"none\" stroke=\"#d11\" stroke-width=\"2\" "
            "marker-end=\"url(#arrow)\"/>\n";
    }
  }

  os << "</svg>\n";
}

struct VisualizeTaskOrchestrationPass
    : public PassWrapper<VisualizeTaskOrchestrationPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VisualizeTaskOrchestrationPass)

  VisualizeTaskOrchestrationPass() = default;
  VisualizeTaskOrchestrationPass(const VisualizeTaskOrchestrationPass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const override {
    return "visualize-task-orchestration";
  }

  StringRef getDescription() const override {
    return "Emits an SVG visualization of scheduled Taskflow orchestration";
  }

  Option<std::string> outputFile{
      *this, "output-file",
      llvm::cl::desc("Output SVG file. If empty, print SVG to stdout."),
      llvm::cl::init("")};

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    SmallVector<VisualTask, 8> tasks = collectVisualTasks(func);
    if (failed(prepareSchedule(func, tasks))) {
      signalPassFailure();
      return;
    }

    if (outputFile.empty()) {
      emitSvg(tasks, func, llvm::outs());
      return;
    }

    std::error_code error;
    llvm::raw_fd_ostream file(outputFile, error, llvm::sys::fs::OF_Text);
    if (error) {
      func.emitError() << "failed to open orchestration SVG output file '"
                       << outputFile << "': " << error.message();
      signalPassFailure();
      return;
    }
    emitSvg(tasks, func, file);
  }
};

} // namespace

namespace mlir {
namespace taskflow {

std::unique_ptr<Pass> createVisualizeTaskOrchestrationPass() {
  return std::make_unique<VisualizeTaskOrchestrationPass>();
}

} // namespace taskflow
} // namespace mlir
