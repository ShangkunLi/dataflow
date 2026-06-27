// Implements throughput-guided task profiling.

#include "TaskflowDialect/Orchestration/ThroughputGuidedTaskOrchestration/TaskProfiler.h"

#include "Conversion/ConversionPasses.h"
#include "NeuraDialect/Architecture/Architecture.h"
#include "NeuraDialect/Mapping/mapping_util.h"
#include "NeuraDialect/NeuraAttributes.h"
#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraPasses.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <limits>
#include <optional>

namespace mlir {
namespace taskflow {

static std::optional<int64_t> getConstantIndex(Value value) {
  if (auto cst = value.getDefiningOp<arith::ConstantIndexOp>()) {
    return cst.value();
  }
  return std::nullopt;
}

static std::optional<int>
computeTaskflowCounterTripCount(TaskflowTaskOp task,
                                int symbol_bound_trip_count) {
  llvm::SmallVector<TaskflowCounterOp> counters;
  for (Operation &op : task.getBody().front()) {
    if (auto counter = dyn_cast<TaskflowCounterOp>(&op)) {
      counters.push_back(counter);
    }
  }
  if (counters.empty()) {
    return std::nullopt;
  }

  llvm::SmallVector<TaskflowCounterOp> roots;
  llvm::DenseMap<Value, llvm::SmallVector<TaskflowCounterOp>>
      parent_to_children;
  for (TaskflowCounterOp counter : counters) {
    if (Value parent = counter.getParentIndex()) {
      parent_to_children[parent].push_back(counter);
    } else {
      roots.push_back(counter);
    }
  }

  auto getCounterTripCount =
      [symbol_bound_trip_count](
          TaskflowCounterOp counter) -> std::optional<int64_t> {
    std::optional<int64_t> lb = getConstantIndex(counter.getLowerBound());
    std::optional<int64_t> ub = getConstantIndex(counter.getUpperBound());
    std::optional<int64_t> step = getConstantIndex(counter.getStep());
    assert(step && *step > 0 && "Counter step must be a positive constant.\n");
    if (!step || *step <= 0) {
      return std::nullopt;
    }

    if (!lb || !ub) {
      assert(symbol_bound_trip_count > 0 &&
             "Symbol-bound trip count must be positive.\n");
      return symbol_bound_trip_count;
    }

    int64_t range = *ub - *lb;
    assert(range > 0 && "Counter range must be positive.\n");
    if (range <= 0) {
      return std::nullopt;
    }
    return llvm::divideCeil(range, *step);
  };

  int64_t max_chain_trip_count = 1;
  for (TaskflowCounterOp root : roots) {
    int64_t chain_trip_count = 1;
    llvm::SmallVector<TaskflowCounterOp> worklist;
    worklist.push_back(root);
    while (!worklist.empty()) {
      TaskflowCounterOp current = worklist.pop_back_val();
      std::optional<int64_t> counter_trip_count = getCounterTripCount(current);
      if (!counter_trip_count) {
        return std::nullopt;
      }
      chain_trip_count *= *counter_trip_count;

      auto children = parent_to_children.find(current.getCounterIndex());
      if (children != parent_to_children.end()) {
        for (TaskflowCounterOp child : children->second) {
          worklist.push_back(child);
        }
      }
    }
    max_chain_trip_count = std::max(max_chain_trip_count, chain_trip_count);
  }

  assert(max_chain_trip_count > 0 && "Counter trip count must be positive.\n");
  assert(max_chain_trip_count <= std::numeric_limits<int>::max() &&
         "Counter trip count exceeds int range.\n");
  return static_cast<int>(max_chain_trip_count);
}

static int getProfileTripCount(TaskflowTaskOp task,
                               int symbol_bound_trip_count) {
  std::optional<int> trip_count =
      computeTaskflowCounterTripCount(task, symbol_bound_trip_count);
  assert(trip_count && "Task profiling requires taskflow counters.\n");
  if (!trip_count) {
    llvm_unreachable("Task profiling requires taskflow counters.");
  }
  return *trip_count;
}

static bool taskContainsNeuraKernel(TaskflowTaskOp task) {
  bool found_kernel = false;
  task.walk([&](neura::KernelOp kernel) { found_kernel = true; });
  return found_kernel;
}

static std::string buildValidTileList(const CgraShape &shape) {
  if (shape.is_rectangular) {
    return "";
  }

  int per_cgra_cols = neura::getArchitecture().getPerCgraColumns();
  int per_cgra_rows = neura::getArchitecture().getPerCgraRows();
  std::string valid_tiles;
  llvm::raw_string_ostream os(valid_tiles);
  bool first_tile = true;
  for (auto [cgra_col, cgra_row] : shape.cgra_positions) {
    for (int tile_row = 0; tile_row < per_cgra_rows; ++tile_row) {
      for (int tile_col = 0; tile_col < per_cgra_cols; ++tile_col) {
        if (!first_tile) {
          os << ",";
        }
        first_tile = false;
        os << (cgra_col * per_cgra_cols + tile_col) << "_"
           << (cgra_row * per_cgra_rows + tile_row);
      }
    }
  }
  return os.str();
}

static void addMapperLoweringPipeline(PassManager &pm) {
  pm.addPass(createCSEPass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(createConvertSCFToCFPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(neura::createAssignAcceleratorPass());
  pm.addPass(createLowerMemRefToNeuraPass());
  pm.addPass(createLowerArithToNeuraPass());
  pm.addPass(createLowerBuiltinToNeuraPass());
  pm.addPass(createLowerLlvmToNeuraPass());
  pm.addPass(neura::createPromoteInputArgToConstPass());
  pm.addPass(neura::createFoldConstantPass());
  pm.addPass(neura::createCanonicalizeReturnPass());
  pm.addPass(neura::createCanonicalizeLiveInPass());
  pm.addPass(neura::createLeveragePredicatedValuePass());
  pm.addPass(neura::createTransformCtrlToDataFlowPass());
  pm.addPass(neura::createFoldConstantPass());
  pm.addPass(neura::createInsertDataMovPass());
}

static std::optional<std::string>
buildMapperWrapperModuleText(neura::KernelOp kernel) {
  MLIRContext *ctx = kernel.getContext();
  Location loc = kernel.getLoc();
  ModuleOp wrapper_module = ModuleOp::create(loc);
  OpBuilder builder(ctx);
  builder.setInsertionPointToStart(wrapper_module.getBody());

  Region &kernel_body = kernel.getBody();
  if (kernel_body.empty()) {
    wrapper_module.erase();
    return std::nullopt;
  }

  llvm::SmallVector<Type> arg_types;
  for (BlockArgument arg : kernel_body.front().getArguments()) {
    arg_types.push_back(arg.getType());
  }
  llvm::SmallVector<Type> result_types;
  bool wrapper_uses_dummy_trigger = true;
  kernel.walk([&](neura::YieldOp yield) {
    if (!yield.getResults().empty()) {
      result_types.assign(yield.getResults().getTypes().begin(),
                          yield.getResults().getTypes().end());
      wrapper_uses_dummy_trigger = false;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (result_types.empty()) {
    result_types.push_back(builder.getI1Type());
  }

  auto func_type = builder.getFunctionType(arg_types, result_types);
  auto wrapper_func =
      builder.create<func::FuncOp>(loc, "__task_profile__", func_type);
  wrapper_func->setAttr("accelerator", builder.getStringAttr("neura"));

  IRMapping mapping;
  kernel_body.cloneInto(&wrapper_func.getBody(), mapping);
  for (Block &block : wrapper_func.getBody()) {
    if (auto yield = dyn_cast<neura::YieldOp>(block.getTerminator())) {
      builder.setInsertionPoint(yield);
      if (wrapper_uses_dummy_trigger) {
        Value trigger =
            builder.create<arith::ConstantIntOp>(loc, 1, builder.getI1Type());
        builder.create<func::ReturnOp>(loc, trigger);
      } else {
        builder.create<func::ReturnOp>(loc, yield.getResults());
      }
      yield.erase();
    }
  }

  std::string module_text;
  llvm::raw_string_ostream os(module_text);
  wrapper_module.print(os);
  wrapper_module.erase();
  return os.str();
}

static void loadMapperDialects(MLIRContext &ctx) {
  ctx.loadDialect<affine::AffineDialect, arith::ArithDialect,
                  cf::ControlFlowDialect, LLVM::LLVMDialect, func::FuncDialect,
                  memref::MemRefDialect, neura::NeuraDialect,
                  scf::SCFDialect>();
}

static LogicalResult extractMappingInfo(ModuleOp module, int &compiled_ii,
                                        int &steps,
                                        int &materialized_operation_count) {
  bool found_mapping_info = false;
  int max_time_step = -1;
  int mapped_op_count = 0;
  module.walk([&](func::FuncOp fn) {
    if (!fn->hasAttr("accelerator")) {
      return;
    }

    auto mapping_info =
        fn->getAttrOfType<DictionaryAttr>(neura::attr::kMappingInfo);
    if (mapping_info) {
      if (auto ii_attr =
              mapping_info.getAs<IntegerAttr>(neura::attr::kCompiledII)) {
        compiled_ii = static_cast<int>(ii_attr.getInt());
        assert(compiled_ii > 0 && "Mapper compiled II must be positive.\n");
        found_mapping_info = true;
      }
    }

    fn.walk([&](Operation *op) {
      auto mapping_locs = op->getAttrOfType<ArrayAttr>("mapping_locs");
      if (!mapping_locs) {
        return;
      }
      if (!mapping_locs.empty() && !neura::is_non_materialized(op)) {
        ++mapped_op_count;
      }
      for (Attribute loc_attr : mapping_locs) {
        auto loc_dict = dyn_cast<DictionaryAttr>(loc_attr);
        if (!loc_dict) {
          continue;
        }
        auto time_step =
            dyn_cast_or_null<IntegerAttr>(loc_dict.get("time_step"));
        if (!time_step) {
          continue;
        }
        max_time_step =
            std::max(max_time_step, static_cast<int>(time_step.getInt()));
      }
    });
  });

  if (!found_mapping_info || max_time_step < 0 || mapped_op_count <= 0) {
    return failure();
  }
  steps = max_time_step + 1;
  materialized_operation_count = mapped_op_count;
  assert(steps > 0 && "Mapper step count must be positive.\n");
  assert(materialized_operation_count > 0 &&
         "Mapper materialized operation count must be positive.\n");
  return success();
}

static LogicalResult runMapperOnKernel(neura::KernelOp kernel, int x_tiles,
                                       int y_tiles,
                                       const std::string &valid_tiles,
                                       int &compiled_ii, int &steps,
                                       int &materialized_operation_count,
                                       bool &mapper_succeeded) {
  std::optional<std::string> module_text = buildMapperWrapperModuleText(kernel);
  assert(module_text && "Task profiling must build a mapper wrapper module.\n");
  if (!module_text) {
    return failure();
  }

  MLIRContext mapper_context;
  mapper_context.disableMultithreading();
  loadMapperDialects(mapper_context);

  ParserConfig parser_config(&mapper_context);
  OwningOpRef<ModuleOp> module_ref = parseSourceString<ModuleOp>(
      *module_text, parser_config, "task-profile.mlir");
  assert(module_ref && "Task profiling wrapper module must parse.\n");
  if (!module_ref) {
    return failure();
  }
  ModuleOp module = *module_ref;

  PassManager lowering_pm(&mapper_context);
  lowering_pm.enableVerifier(false);
  addMapperLoweringPipeline(lowering_pm);
  LogicalResult lowering_result = lowering_pm.run(module);
  assert(succeeded(lowering_result) &&
         "Task profiling lowering pipeline must succeed before mapping.\n");
  if (failed(lowering_result)) {
    return failure();
  }

  PassManager mapper_pm(&mapper_context);
  mapper_pm.enableVerifier(false);
  neura::MapToAcceleratorOptions map_options;
  map_options.x_tiles = x_tiles;
  map_options.y_tiles = y_tiles;
  map_options.valid_tiles = valid_tiles;
  mapper_pm.addPass(neura::createMapToAcceleratorPass(map_options));
  LogicalResult mapper_result = mapper_pm.run(module);
  if (failed(mapper_result)) {
    return failure();
  }

  LogicalResult extract_result = extractMappingInfo(
      module, compiled_ii, steps, materialized_operation_count);
  assert(succeeded(extract_result) &&
         "Task profiling mapper result must contain mapping info.\n");
  if (failed(extract_result)) {
    return failure();
  }
  mapper_succeeded = true;
  return success();
}

TaskProfileMap TaskProfiler::profileFunction(func::FuncOp func) const {
  TaskProfileMap profile_map;
  func.walk(
      [&](TaskflowTaskOp task) { profile_map[task] = profileTask(task); });
  return profile_map;
}

llvm::SmallVector<TaskProfile>
TaskProfiler::profileTask(TaskflowTaskOp task) const {
  llvm::SmallVector<TaskProfile> profiles;
  assert(taskContainsNeuraKernel(task) && "Each task must contain a "
                                          "neura.kernel for profiling.\n");

  for (int cgra_count = 1; cgra_count <= max_composed_cgra_count_;
       ++cgra_count) {
    for (const CgraShape &shape : getAllPlacementShapes(cgra_count)) {
      std::optional<TaskProfile> profile =
          profileTaskOnComposedCgra(task, shape, cgra_count);
      if (profile) {
        profiles.push_back(std::move(*profile));
      }
    }
  }

  assert(!profiles.empty() &&
         "Task profiling requires at least one mapper-valid composed-CGRA "
         "profile.\n");
  return profiles;
}

std::optional<TaskProfile>
TaskProfiler::profileTaskOnComposedCgra(TaskflowTaskOp task,
                                        const CgraShape &shape,
                                        int composed_cgra_count) const {
  TaskProfile profile;
  profile.composed_cgra_count = composed_cgra_count;
  profile.composed_cgra_shape = shape.irAttr();

  profile.sample_trip_count =
      getProfileTripCount(task, symbol_bound_trip_count_);

  int compiled_ii = -1;
  int steps = -1;
  int materialized_operation_count = -1;
  bool mapper_succeeded = false;
  bool profiling_succeeded = runMapperForTaskProfile(
      task, shape, compiled_ii, steps, materialized_operation_count,
      mapper_succeeded);
  if (!profiling_succeeded || !mapper_succeeded) {
    return std::nullopt;
  }

  assert(compiled_ii > 0 && "Mapper compiled II must be positive.\n");
  assert(steps > 0 && "Mapper step count must be positive.\n");
  assert(materialized_operation_count > 0 &&
         "Mapper materialized operation count must be positive.\n");
  assert(profile.sample_trip_count > 0 &&
         "Task profile trip count must be positive.\n");
  profile.compiled_ii = compiled_ii;
  profile.steps = steps;
  profile.materialized_operation_count = materialized_operation_count;
  profile.mapper_succeeded = mapper_succeeded;
  int64_t estimated_latency = static_cast<int64_t>(profile.compiled_ii) *
                                  (profile.sample_trip_count - 1) +
                              profile.steps;
  assert(estimated_latency > 0 && "Estimated latency must be positive.\n");
  assert(estimated_latency <= std::numeric_limits<int>::max() &&
         "Estimated latency exceeds int range.\n");
  profile.estimated_latency = static_cast<int>(estimated_latency);
  return profile;
}

bool TaskProfiler::runMapperForTaskProfile(TaskflowTaskOp task,
                                           const CgraShape &shape,
                                           int &compiled_ii, int &steps,
                                           int &materialized_operation_count,
                                           bool &mapper_succeeded) const {
  neura::KernelOp source_kernel;
  task.walk([&](neura::KernelOp kernel) {
    if (!source_kernel) {
      source_kernel = kernel;
    }
  });
  if (!source_kernel) {
    return false;
  }

  int per_cgra_cols = neura::getArchitecture().getPerCgraColumns();
  int per_cgra_rows = neura::getArchitecture().getPerCgraRows();
  int x_tiles = shape.cols * per_cgra_cols;
  int y_tiles = shape.rows * per_cgra_rows;
  assert(x_tiles > 0 && y_tiles > 0 &&
         "Composed-CGRA profile shape must contain at least one tile.\n");
  std::string valid_tiles = buildValidTileList(shape);

  LogicalResult result =
      runMapperOnKernel(source_kernel, x_tiles, y_tiles, valid_tiles,
                        compiled_ii, steps, materialized_operation_count,
                        mapper_succeeded);
  return succeeded(result);
}

} // namespace taskflow
} // namespace mlir
