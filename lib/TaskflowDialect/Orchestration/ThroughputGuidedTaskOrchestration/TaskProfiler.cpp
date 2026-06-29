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
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <limits>
#include <optional>

namespace mlir {
namespace taskflow {

namespace {

struct ProfileJsonProgress {
  int expected_candidate_count = -1;
  int completed_candidate_count = -1;
  std::string last_task;
  int last_candidate_index = -1;
  int last_composed_cgra_count = -1;
  std::string last_shape;
  bool last_mapper_succeeded = false;
};

void writeJsonString(raw_ostream &os, StringRef value) {
  os << "\"";
  for (char c : value) {
    switch (c) {
    case '"':
      os << "\\\"";
      break;
    case '\\':
      os << "\\\\";
      break;
    case '\b':
      os << "\\b";
      break;
    case '\f':
      os << "\\f";
      break;
    case '\n':
      os << "\\n";
      break;
    case '\r':
      os << "\\r";
      break;
    case '\t':
      os << "\\t";
      break;
    default:
      os << c;
      break;
    }
  }
  os << "\"";
}

std::string getTaskProfileName(TaskflowTaskOp task) {
  return task.getTaskName().str();
}

void writeTaskProfile(raw_ostream &os, const TaskProfile &profile,
                      StringRef indent) {
  os << indent << "{\n";
  os << indent << "  \"composed_cgra_count\": "
     << profile.composed_cgra_count << ",\n";
  os << indent << "  \"composed_cgra_shape\": ";
  writeJsonString(os, profile.composed_cgra_shape);
  os << ",\n";
  os << indent << "  \"compiled_ii\": " << profile.compiled_ii << ",\n";
  os << indent << "  \"steps\": " << profile.steps << ",\n";
  os << indent << "  \"sample_trip_count\": " << profile.sample_trip_count
     << ",\n";
  os << indent << "  \"materialized_operation_count\": "
     << profile.materialized_operation_count << ",\n";
  os << indent << "  \"estimated_latency\": " << profile.estimated_latency
     << ",\n";
  os << indent << "  \"mapper_succeeded\": "
     << (profile.mapper_succeeded ? "true" : "false") << "\n";
  os << indent << "}";
}

LogicalResult writeTaskProfileMapToJsonImpl(
    func::FuncOp func, const TaskProfileMap &profile_map,
    StringRef output_file, const ProfileJsonProgress *progress) {
  std::error_code error;
  llvm::raw_fd_ostream os(output_file, error);
  if (error) {
    func.emitError() << "failed to open task profile JSON file '"
                     << output_file << "': " << error.message();
    return failure();
  }

  SmallVector<TaskflowTaskOp> tasks;
  func.walk([&](TaskflowTaskOp task) { tasks.push_back(task); });

  os << "{\n";
  os << "  \"format\": \"amoeba-task-profile-v1\",\n";
  os << "  \"function\": ";
  writeJsonString(os, func.getSymName());
  os << ",\n";
  os << "  \"task_count\": " << tasks.size() << ",\n";
  if (progress) {
    os << "  \"expected_candidate_count\": "
       << progress->expected_candidate_count << ",\n";
    os << "  \"completed_candidate_count\": "
       << progress->completed_candidate_count << ",\n";
    os << "  \"last_completed_candidate\": ";
    if (progress->last_task.empty()) {
      os << "null";
    } else {
      os << "{\n";
      os << "    \"task\": ";
      writeJsonString(os, progress->last_task);
      os << ",\n";
      os << "    \"candidate_index_in_task\": "
         << progress->last_candidate_index << ",\n";
      os << "    \"composed_cgra_count\": "
         << progress->last_composed_cgra_count << ",\n";
      os << "    \"shape\": ";
      writeJsonString(os, progress->last_shape);
      os << ",\n";
      os << "    \"mapper_succeeded\": "
         << (progress->last_mapper_succeeded ? "true" : "false") << "\n";
      os << "  }";
    }
    os << ",\n";
  }

  os << "  \"tasks\": [\n";
  for (auto [task_index, task] : llvm::enumerate(tasks)) {
    os << "    {\n";
    os << "      \"task\": ";
    writeJsonString(os, getTaskProfileName(task));
    os << ",\n";
    os << "      \"profiles\": [\n";

    auto it = profile_map.find(task);
    ArrayRef<TaskProfile> profiles;
    if (it != profile_map.end()) {
      profiles = it->second;
    }
    for (auto [profile_index, profile] : llvm::enumerate(profiles)) {
      writeTaskProfile(os, profile, "        ");
      if (profile_index + 1 != profiles.size()) {
        os << ",";
      }
      os << "\n";
    }

    os << "      ]\n";
    os << "    }";
    if (task_index + 1 != tasks.size()) {
      os << ",";
    }
    os << "\n";
  }
  os << "  ]\n";
  os << "}\n";
  return success();
}

std::optional<int> readRequiredJsonInt(TaskflowTaskOp task,
                                       const llvm::json::Object &object,
                                       StringRef key) {
  std::optional<int64_t> value = object.getInteger(key);
  if (!value) {
    task.emitError() << "task profile JSON entry requires integer key '" << key
                     << "'";
    return std::nullopt;
  }
  if (*value < std::numeric_limits<int>::min() ||
      *value > std::numeric_limits<int>::max()) {
    task.emitError() << "task profile JSON integer key '" << key
                     << "' exceeds int range";
    return std::nullopt;
  }
  return static_cast<int>(*value);
}

std::optional<TaskProfile>
readTaskProfileFromJson(TaskflowTaskOp task, const llvm::json::Object &object) {
  TaskProfile profile;
  std::optional<int> composed_cgra_count =
      readRequiredJsonInt(task, object, "composed_cgra_count");
  std::optional<StringRef> composed_cgra_shape =
      object.getString("composed_cgra_shape");
  std::optional<int> compiled_ii =
      readRequiredJsonInt(task, object, "compiled_ii");
  std::optional<int> steps = readRequiredJsonInt(task, object, "steps");
  std::optional<int> sample_trip_count =
      readRequiredJsonInt(task, object, "sample_trip_count");
  std::optional<int> materialized_operation_count =
      readRequiredJsonInt(task, object, "materialized_operation_count");
  std::optional<int> estimated_latency =
      readRequiredJsonInt(task, object, "estimated_latency");
  std::optional<bool> mapper_succeeded = object.getBoolean("mapper_succeeded");

  if (!composed_cgra_count || !composed_cgra_shape || !compiled_ii || !steps ||
      !sample_trip_count || !materialized_operation_count ||
      !estimated_latency || !mapper_succeeded) {
    if (!composed_cgra_shape) {
      task.emitError()
          << "task profile JSON entry requires string key "
             "'composed_cgra_shape'";
    }
    if (!mapper_succeeded) {
      task.emitError()
          << "task profile JSON entry requires boolean key "
             "'mapper_succeeded'";
    }
    return std::nullopt;
  }

  profile.composed_cgra_count = *composed_cgra_count;
  profile.composed_cgra_shape = composed_cgra_shape->str();
  profile.compiled_ii = *compiled_ii;
  profile.steps = *steps;
  profile.sample_trip_count = *sample_trip_count;
  profile.materialized_operation_count = *materialized_operation_count;
  profile.estimated_latency = *estimated_latency;
  profile.mapper_succeeded = *mapper_succeeded;

  if (profile.composed_cgra_count <= 0 || profile.compiled_ii <= 0 ||
      profile.steps <= 0 || profile.sample_trip_count <= 0 ||
      profile.materialized_operation_count <= 0 ||
      profile.estimated_latency <= 0 || !profile.mapper_succeeded) {
    task.emitError() << "task profile JSON entry contains invalid profile "
                        "values";
    return std::nullopt;
  }
  return profile;
}

} // namespace

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

static LogicalResult
runMapperOnKernel(neura::KernelOp kernel, int x_tiles, int y_tiles,
                  const std::string &valid_tiles, int &compiled_ii, int &steps,
                  int &materialized_operation_count, bool &mapper_succeeded) {
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

LogicalResult TaskProfiler::profileFunctionToJson(
    func::FuncOp func, llvm::StringRef output_file) const {
  TaskProfileMap profile_map;

  SmallVector<TaskflowTaskOp> tasks;
  func.walk([&](TaskflowTaskOp task) { tasks.push_back(task); });

  int candidates_per_task = 0;
  for (int cgra_count = 1; cgra_count <= max_composed_cgra_count_;
       ++cgra_count) {
    for (const CgraShape &shape : getAllPlacementShapes(cgra_count)) {
      if (shape.is_rectangular) {
        ++candidates_per_task;
      }
    }
  }

  ProfileJsonProgress progress;
  progress.expected_candidate_count =
      candidates_per_task * static_cast<int>(tasks.size());
  progress.completed_candidate_count = 0;
  if (failed(
          writeTaskProfileMapToJsonImpl(func, profile_map, output_file,
                                        &progress))) {
    return failure();
  }

  for (TaskflowTaskOp task : tasks) {
    auto &profiles = profile_map[task];
    bool write_failed = false;
    llvm::SmallVector<TaskProfile> task_profiles =
        profileTaskWithCandidateCallback(
            task,
            [&](int candidate_index, const CgraShape &shape,
                int composed_cgra_count,
                const std::optional<TaskProfile> &profile) {
              ++progress.completed_candidate_count;
              progress.last_task = getTaskProfileName(task);
              progress.last_candidate_index = candidate_index;
              progress.last_composed_cgra_count = composed_cgra_count;
              progress.last_shape = shape.irAttr();
              progress.last_mapper_succeeded =
                  profile && profile->mapper_succeeded;
              if (profile) {
                profiles.push_back(*profile);
              }
              if (failed(writeTaskProfileMapToJsonImpl(
                      func, profile_map, output_file, &progress))) {
                write_failed = true;
              }
            });
    if (write_failed) {
      return failure();
    }
    assert(task_profiles.size() == profiles.size() &&
           "Incremental JSON profile cache should match profiler results.\n");
  }
  return writeTaskProfileMapToJsonImpl(func, profile_map, output_file,
                                       &progress);
}

LogicalResult TaskProfiler::writeTaskProfileMapToJson(
    func::FuncOp func, const TaskProfileMap &profile_map,
    llvm::StringRef output_file) {
  return writeTaskProfileMapToJsonImpl(func, profile_map, output_file,
                                       /*progress=*/nullptr);
}

FailureOr<TaskProfileMap>
TaskProfiler::readTaskProfileMapFromJson(func::FuncOp func,
                                         llvm::StringRef input_file) {
  auto buffer = llvm::MemoryBuffer::getFile(input_file);
  if (!buffer) {
    func.emitError() << "failed to open task profile JSON file '" << input_file
                     << "': " << buffer.getError().message();
    return failure();
  }

  llvm::Expected<llvm::json::Value> parsed =
      llvm::json::parse((*buffer)->getBuffer());
  if (!parsed) {
    std::string error_message;
    llvm::raw_string_ostream os(error_message);
    llvm::logAllUnhandledErrors(parsed.takeError(), os);
    func.emitError() << "failed to parse task profile JSON file '"
                     << input_file << "': " << os.str();
    return failure();
  }

  llvm::json::Object *root = parsed->getAsObject();
  if (!root) {
    func.emitError() << "task profile JSON root must be an object";
    return failure();
  }

  std::optional<StringRef> format = root->getString("format");
  if (!format || *format != "amoeba-task-profile-v1") {
    func.emitError() << "task profile JSON requires format "
                        "'amoeba-task-profile-v1'";
    return failure();
  }

  std::optional<StringRef> function_name = root->getString("function");
  if (function_name && *function_name != func.getSymName()) {
    func.emitError() << "task profile JSON was generated for function '"
                     << *function_name << "', not '" << func.getSymName()
                     << "'";
    return failure();
  }

  llvm::StringMap<TaskflowTaskOp> task_by_name;
  SmallVector<TaskflowTaskOp> tasks;
  func.walk([&](TaskflowTaskOp task) {
    std::string task_name = getTaskProfileName(task);
    task_by_name[task_name] = task;
    tasks.push_back(task);
  });

  llvm::json::Array *task_entries = root->getArray("tasks");
  if (!task_entries) {
    func.emitError() << "task profile JSON requires a tasks array";
    return failure();
  }

  TaskProfileMap profile_map;
  llvm::DenseSet<Operation *> seen_tasks;
  for (llvm::json::Value &task_value : *task_entries) {
    llvm::json::Object *task_object = task_value.getAsObject();
    if (!task_object) {
      func.emitError() << "task profile JSON tasks entries must be objects";
      return failure();
    }

    std::optional<StringRef> task_name = task_object->getString("task");
    if (!task_name) {
      func.emitError() << "task profile JSON task entry requires task name";
      return failure();
    }

    auto task_it = task_by_name.find(*task_name);
    if (task_it == task_by_name.end()) {
      func.emitError() << "task profile JSON contains unknown task '"
                       << *task_name << "'";
      return failure();
    }
    TaskflowTaskOp task = task_it->second;

    llvm::json::Array *profile_entries = task_object->getArray("profiles");
    if (!profile_entries) {
      task.emitError() << "task profile JSON task entry requires profiles "
                          "array";
      return failure();
    }

    auto &profiles = profile_map[task];
    for (llvm::json::Value &profile_value : *profile_entries) {
      llvm::json::Object *profile_object = profile_value.getAsObject();
      if (!profile_object) {
        task.emitError()
            << "task profile JSON profile entries must be objects";
        return failure();
      }
      std::optional<TaskProfile> profile =
          readTaskProfileFromJson(task, *profile_object);
      if (!profile) {
        return failure();
      }
      profiles.push_back(std::move(*profile));
    }

    if (profiles.empty()) {
      task.emitError() << "task profile JSON contains no valid profiles for "
                          "this task";
      return failure();
    }
    seen_tasks.insert(task.getOperation());
  }

  for (TaskflowTaskOp task : tasks) {
    if (!seen_tasks.contains(task.getOperation())) {
      task.emitError() << "task profile JSON is missing this task";
      return failure();
    }
  }
  return profile_map;
}

llvm::SmallVector<TaskProfile>
TaskProfiler::profileTask(TaskflowTaskOp task) const {
  return profileTaskWithCandidateCallback(
      task, [](int, const CgraShape &, int,
               const std::optional<TaskProfile> &) {});
}

llvm::SmallVector<TaskProfile>
TaskProfiler::profileTaskWithCandidateCallback(
    TaskflowTaskOp task,
    llvm::function_ref<void(int, const CgraShape &, int,
                            const std::optional<TaskProfile> &)>
        candidate_callback) const {
  llvm::SmallVector<TaskProfile> profiles;
  assert(taskContainsNeuraKernel(task) && "Each task must contain a "
                                          "neura.kernel for profiling.\n");

  int candidate_index = 0;
  for (int cgra_count = 1; cgra_count <= max_composed_cgra_count_;
       ++cgra_count) {
    for (const CgraShape &shape : getAllPlacementShapes(cgra_count)) {
      // Throughput-guided profiling intentionally keeps the tile-array
      // candidate frontier rectangular. Irregular L/T shapes are not profiled
      // or selected by this strategy.
      if (!shape.is_rectangular) {
        continue;
      }
      ++candidate_index;
      std::optional<TaskProfile> profile =
          profileTaskOnComposedCgra(task, shape, cgra_count);
      candidate_callback(candidate_index, shape, cgra_count, profile);
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
  assert(shape.is_rectangular &&
         "Throughput-guided profiling only accepts rectangular "
         "composed-CGRA shapes.\n");
  if (!shape.is_rectangular) {
    return std::nullopt;
  }

  TaskProfile profile;
  profile.composed_cgra_count = composed_cgra_count;
  profile.composed_cgra_shape = shape.irAttr();

  profile.sample_trip_count =
      getProfileTripCount(task, symbol_bound_trip_count_);

  int compiled_ii = -1;
  int steps = -1;
  int materialized_operation_count = -1;
  bool mapper_succeeded = false;
  bool profiling_succeeded =
      runMapperForTaskProfile(task, shape, compiled_ii, steps,
                              materialized_operation_count, mapper_succeeded);
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
  assert(shape.is_rectangular &&
         "Task profiler should only map rectangular composed-CGRA shapes.\n");
  std::string valid_tiles;

  LogicalResult result = runMapperOnKernel(
      source_kernel, x_tiles, y_tiles, valid_tiles, compiled_ii, steps,
      materialized_operation_count, mapper_succeeded);
  return succeeded(result);
}

} // namespace taskflow
} // namespace mlir
