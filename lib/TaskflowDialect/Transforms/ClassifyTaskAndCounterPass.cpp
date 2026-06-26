//===- ClassifyTaskAndCounterPass.cpp - Classify tasks and counters -------===//
//
// Single pass that annotates every taskflow.task and its contained
// taskflow.counter ops with AMOEBA execution-model attributes:
//
// Per taskflow.counter:
//   counter_hierarchy  – structural role in the loop nest:
//     "root"   : outermost loop (no parent, has children)
//     "relay"  : middle loop   (has parent and children)
//     "leaf"   : innermost loop (no children; also single-loop tasks)
//
//   counter_dynamism   – AMOEBA bound class:
//     "constant_bound" : all bounds are compile-time constants
//     "symbol_bound"   : bounds are known before the task launches
//     "dynamic_bound"  : bounds are produced during task execution
//
//   counter_id         – unique integer index within the task (0-based)
//
// Per taskflow.task:
//   dlp_replicable = true when the task has at least one counter-driven
//   hyperblock dimension that can be partitioned for data-level parallelism.
//
// Classification rules for a bound value inside the task body:
//   arith.constant                             → constant_bound
//   affine.apply                               → recurse through operands
//   block arg mapping to an outer value:
//     func.func argument                       → symbol_bound
//     memref.dim / function-scope arithmetic   → symbol_bound
//     arith.constant at call site              → constant_bound
//   anything else                              → dynamic_bound
//
//===----------------------------------------------------------------------===//

#include "TaskflowDialect/TaskflowDialect.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "TaskflowDialect/TaskflowPasses.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <memory>
#include <optional>

using namespace mlir;
using namespace mlir::taskflow;

namespace {

//===----------------------------------------------------------------------===//
// Bound kind
//===----------------------------------------------------------------------===//

enum class BoundKind { ConstantBound = 0, SymbolBound = 1, DynamicBound = 2 };

static BoundKind worstCase(BoundKind a, BoundKind b) {
  return static_cast<BoundKind>(
      std::max(static_cast<int>(a), static_cast<int>(b)));
}

static StringRef boundKindToStr(BoundKind k) {
  switch (k) {
  case BoundKind::ConstantBound:
    return "constant_bound";
  case BoundKind::SymbolBound:
    return "symbol_bound";
  case BoundKind::DynamicBound:
    return "dynamic_bound";
  }
  llvm_unreachable("unknown BoundKind");
}

//===----------------------------------------------------------------------===//
// Bound classification helpers
//===----------------------------------------------------------------------===//

// Classifies a value that lives OUTSIDE the task body (a value_input operand
// at the call site) by tracing it to its origin.
static BoundKind classifyOuterValue(Value v) {
  if (auto block_arg = dyn_cast<BlockArgument>(v)) {
    auto *parent_region = block_arg.getParentRegion();
    if (parent_region && isa<func::FuncOp>(parent_region->getParentOp())) {
      return BoundKind::SymbolBound;
    }
    return BoundKind::DynamicBound;
  }
  Operation *def = v.getDefiningOp();
  if (!def) {
    assert(false && "unexpected value with no defining op and not a block arg");
  }
  if (isa<arith::ConstantOp, arith::ConstantIndexOp>(def)) {
    return BoundKind::ConstantBound;
  }
  // Any op at the function body scope (memref.dim, preceding taskflow.task
  // results, arith on func args, etc.) is fully determined before this task
  // launches.
  if (def->getParentRegion() &&
      isa<func::FuncOp>(def->getParentRegion()->getParentOp())) {
    return BoundKind::SymbolBound;
  }
  return BoundKind::DynamicBound;
}

// Classifies one lower/upper/step value used by a taskflow.counter.
// task_op is used to map task block arguments back to value_input operands.
static BoundKind classifyCounterBoundValue(Value v, TaskflowTaskOp task_op) {
  if (Operation *def = v.getDefiningOp()) {
    if (isa<arith::ConstantOp, arith::ConstantIndexOp>(def)) {
      return BoundKind::ConstantBound;
    }
    // affine.apply is a pure affine computation; its dynamism is determined
    // entirely by its operands (e.g. affine_map<()[s0]->(s0-2)>()[%n]).
    if (isa<affine::AffineApplyOp>(def)) {
      BoundKind k = BoundKind::ConstantBound;
      for (Value operand : def->getOperands()) {
        k = worstCase(k, classifyCounterBoundValue(operand, task_op));
      }
      return k;
    }
    return BoundKind::DynamicBound;
  }

  auto block_arg = dyn_cast<BlockArgument>(v);
  if (!block_arg) {
    return BoundKind::DynamicBound;
  }

  // Block args layout: [dep_read_in…] [dep_write_in…] [value_inputs…]
  unsigned num_dep_read = task_op.getDependencyReadIn().size();
  unsigned num_dep_write = task_op.getDependencyWriteIn().size();
  unsigned value_input_start = num_dep_read + num_dep_write;
  unsigned arg_idx = block_arg.getArgNumber();

  assert(arg_idx >= value_input_start &&
         "counter bound is a memref dependency block arg — IR is malformed");

  unsigned vi_idx = arg_idx - value_input_start;
  auto value_inputs = task_op.getValueInputs();
  assert(vi_idx < value_inputs.size() &&
         "counter bound block arg index out of range of value inputs");

  return classifyOuterValue(value_inputs[vi_idx]);
}

// Returns the drive-pattern class implied by a counter's lb/ub/step operands.
static BoundKind classifyCounterDynamism(TaskflowCounterOp counter_op,
                                         TaskflowTaskOp task_op) {
  BoundKind k = BoundKind::ConstantBound;
  for (Value bound : {counter_op.getLowerBound(), counter_op.getUpperBound(),
                      counter_op.getStep()}) {
    k = worstCase(k, classifyCounterBoundValue(bound, task_op));
  }
  return k;
}

//===----------------------------------------------------------------------===//
// Hyperblock patterns for DLP suitability.
//===----------------------------------------------------------------------===//

static unsigned
getHyperblockPartitionableCounterCount(TaskflowHyperblockOp hb) {
  unsigned trigger_count = hb.getIndices().size();
  if (trigger_count == 0) {
    return 0;
  }

  // In the counter-chain lowering used before kernel generation, iter_args
  // are produced by the deepest loop in the perfect band. Those deepest loop
  // counters carry reduction state and are not valid DLP partition dimensions.
  // Outer counter triggers still describe independent output/input tiles and
  // remain partitionable. For example, a matmul-like (i, j, k) nest with an
  // accumulator on k has three trigger counters and one iter_arg; i and j are
  // partitionable while k is not.
  unsigned loop_carried_trigger_count = hb.getIterArgs().empty() ? 0 : 1;
  return trigger_count - loop_carried_trigger_count;
}

static DenseMap<Value, TaskflowCounterOp>
buildCounterIndexMap(TaskflowTaskOp task_op) {
  DenseMap<Value, TaskflowCounterOp> counter_by_index;
  task_op.walk([&](TaskflowCounterOp counter_op) {
    counter_by_index[counter_op.getCounterIndex()] = counter_op;
  });
  return counter_by_index;
}

static SmallVector<int32_t>
collectPartitionableCounterIds(TaskflowTaskOp task_op) {
  DenseMap<Value, TaskflowCounterOp> counter_by_index =
      buildCounterIndexMap(task_op);
  DenseSet<int32_t> seen_counter_ids;
  SmallVector<int32_t> partitionable_counter_ids;

  task_op.walk([&](TaskflowHyperblockOp hb) {
    unsigned partitionable_count = getHyperblockPartitionableCounterCount(hb);
    for (auto [trigger_idx, counter_index] : llvm::enumerate(hb.getIndices())) {
      if (trigger_idx >= partitionable_count) {
        break;
      }

      TaskflowCounterOp counter_op = counter_by_index.lookup(counter_index);
      if (!counter_op) {
        continue;
      }

      std::optional<uint32_t> counter_id = counter_op.getCounterId();
      assert(counter_id &&
             "partitionable taskflow.counter must be classified first");

      int32_t signed_counter_id = static_cast<int32_t>(*counter_id);
      if (seen_counter_ids.insert(signed_counter_id).second) {
        partitionable_counter_ids.push_back(signed_counter_id);
      }
    }
  });
  return partitionable_counter_ids;
}

//===----------------------------------------------------------------------===//
// Counter classification
//===----------------------------------------------------------------------===//

static LogicalResult classifyCounters(TaskflowTaskOp task_op) {
  // Pre-flight check: construct-hyperblock-from-task must run first.
  bool has_affine_for = false;
  task_op.walk([&](affine::AffineForOp) -> WalkResult {
    has_affine_for = true;
    return WalkResult::interrupt();
  });
  bool has_counter = false;
  task_op.walk([&](TaskflowCounterOp) -> WalkResult {
    has_counter = true;
    return WalkResult::interrupt();
  });
  if (has_affine_for && !has_counter) {
    return task_op.emitError()
           << "[ClassifyTaskAndCounter]: task '" << task_op.getTaskName()
           << "' contains affine.for loops but no taskflow.counter ops — "
              "run 'construct-hyperblock-from-task' before "
              "'classify-task-and-counter'";
  }

  SmallVector<TaskflowCounterOp> counters;
  task_op.walk(
      [&](TaskflowCounterOp counter_op) { counters.push_back(counter_op); });

  OpBuilder builder(task_op.getContext());

  if (counters.empty()) {
    return success();
  }

  // Build parent-child relationships among counters.
  DenseMap<Value, TaskflowCounterOp> value_to_counter;
  for (TaskflowCounterOp counter_op : counters) {
    value_to_counter[counter_op.getCounterIndex()] = counter_op;
  }

  DenseSet<TaskflowCounterOp> counters_with_children;
  for (TaskflowCounterOp counter_op : counters) {
    if (auto parent_idx = counter_op.getParentIndex()) {
      if (auto parent_counter = value_to_counter.lookup(parent_idx)) {
        counters_with_children.insert(parent_counter);
      }
    }
  }

  int counter_id = 0;

  for (TaskflowCounterOp counter_op : counters) {
    // --- counter_dynamism ---
    BoundKind bound_kind = classifyCounterDynamism(counter_op, task_op);

    // --- counter_hierarchy ---
    bool has_parent = (counter_op.getParentIndex() != nullptr);
    bool has_child = counters_with_children.contains(counter_op);

    StringRef structural_type;
    if (!has_parent && has_child) {
      structural_type = "root";
    } else if (has_parent && has_child) {
      structural_type = "relay";
    } else {
      // No children (or single loop with no parent/child): leaf.
      structural_type = "leaf";
    }

    counter_op.setCounterHierarchyAttr(builder.getStringAttr(structural_type));
    counter_op.setCounterDynamismAttr(
        builder.getStringAttr(boundKindToStr(bound_kind)));
    counter_op.setCounterIdAttr(builder.getI32IntegerAttr(counter_id++));
  }

  return success();
}

//===----------------------------------------------------------------------===//
// DLP-capability task tagging
//===----------------------------------------------------------------------===//

static void identifyDlpCapability(TaskflowTaskOp task_op) {
  // Re-running this pass should not preserve task-level tags from an older
  // classification result.
  task_op->removeAttr("dlp_replicable");
  task_op->removeAttr("dlp_partitionable_counter_ids");
  task_op->removeAttr("task_type");
  task_op->removeAttr("runtime_managable");

  SmallVector<int32_t> partitionable_counter_ids =
      collectPartitionableCounterIds(task_op);
  if (!partitionable_counter_ids.empty()) {
    Builder builder(task_op.getContext());
    task_op->setAttr("dlp_replicable", builder.getBoolAttr(true));
    task_op->setAttr("dlp_partitionable_counter_ids",
                     builder.getDenseI32ArrayAttr(partitionable_counter_ids));
  }
}

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

struct ClassifyTaskAndCounterPass
    : public PassWrapper<ClassifyTaskAndCounterPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ClassifyTaskAndCounterPass)

  StringRef getArgument() const override { return "classify-task-and-counter"; }
  StringRef getDescription() const override {
    return "Classify taskflow counters and DLP-replicable tasks.";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    WalkResult counter_result =
        module.walk([&](TaskflowTaskOp task_op) -> WalkResult {
          if (failed(classifyCounters(task_op))) {
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        });
    if (counter_result.wasInterrupted()) {
      signalPassFailure();
      return;
    }

    module.walk(
        [&](TaskflowTaskOp task_op) { identifyDlpCapability(task_op); });
  }
};

} // namespace

std::unique_ptr<Pass> mlir::taskflow::createClassifyTaskAndCounterPass() {
  return std::make_unique<ClassifyTaskAndCounterPass>();
}
