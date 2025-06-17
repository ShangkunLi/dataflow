#include "Common/AcceleratorAttrs.h"
#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraTypes.h"
#include "NeuraDialect/NeuraPasses.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlir;

#define GEN_PASS_DEF_TransformCtrlToDataFlow
#include "NeuraDialect/NeuraPasses.h.inc"

// Returns blocks in post-order traversal order.
void getBlocksInPostOrder(Block *startBlock, SmallVectorImpl<Block *> &postOrder,
                         DenseSet<Block *> &visited) {
  if (!visited.insert(startBlock).second)
    return;

  // Visits successors first.
  for (Block *succ : startBlock->getSuccessors())
    getBlocksInPostOrder(succ, postOrder, visited);

  // Adds current block to post-order sequence.
  postOrder.push_back(startBlock);
}

// Creates phi nodes for all live-in values in the given block.
void createPhiNodesForBlock(Block *block, OpBuilder &builder,
                            DenseMap<Value, Value> &value_map,
                            SmallVectorImpl<std::tuple<Value, Value, Block *>> &deferred_ctrl_movs) {
  if (block->hasNoPredecessors()) {
    // Skips phi insertion for entry block.
    return;
  }

  // Collects all live-in values.
  std::vector<Value> live_ins;
  for (Operation &op : *block) {
    for (Value operand : op.getOperands()) {
      // Identifies operands defined in other blocks.
      if (operand.getDefiningOp() &&
          operand.getDefiningOp()->getBlock() != block) {
        live_ins.push_back(operand);
        continue;
      }
      // Collects all block arguments.
      if (auto blockArg = llvm::dyn_cast<BlockArgument>(operand)) {
        live_ins.push_back(operand);
      }
    }
  }

  // Creates a phi node for each live-in value.
  builder.setInsertionPointToStart(block);
  for (Value live_in : live_ins) {
    // Creates predicated type for phi node.
    Type live_in_type = live_in.getType();
    Type predicated_type = isa<neura::PredicatedValue>(live_in_type)
        ? live_in_type
        : neura::PredicatedValue::get(builder.getContext(), live_in_type, builder.getI1Type());

    // Uses the location from the first operation in the block or block's parent operation.
    Location loc = block->empty() ?
                  block->getParent()->getLoc() :
                  block->front().getLoc();

    SmallVector<Value> phi_operands;
    BlockArgument arg = dyn_cast<BlockArgument>(live_in);
    // Handles the case where live_in is not a block argument.
    if (!arg) {
      phi_operands.push_back(live_in);
    } else {
      // Finds index of live_in in block arguments.
      unsigned arg_index = arg.getArgNumber();
      for (Block *pred : block->getPredecessors()) {
        Value incoming;
        Operation *term = pred->getTerminator();

        // If it's a branch or cond_br, get the value passed into this block argument
        if (auto br = dyn_cast<neura::Br>(term)) {
          auto args = br.getArgs();
          assert(arg_index < args.size());
          incoming = args[arg_index];
        } else if (auto condBr = dyn_cast<neura::CondBr>(term)) {
          if (condBr.getTrueDest() == block) {
            auto trueArgs = condBr.getTrueArgs();
            assert(arg_index < trueArgs.size());
            incoming = trueArgs[arg_index];
          } else if (condBr.getFalseDest() == block) {
            auto falseArgs = condBr.getFalseArgs();
            assert(arg_index < falseArgs.size());
            incoming = falseArgs[arg_index];
          } else {
            llvm::errs() << "cond_br does not target block:\n" << *block << "\n";
            continue;
          }
        } else {
          llvm::errs() << "Unknown branch terminator in block: " << *pred << "\n";
          continue;
        }

        // If the incoming value is defined in the same block, inserts a `neura.reserve`
        // and defer a backward ctrl move.
        if (incoming.getDefiningOp() && incoming.getDefiningOp()->getBlock() == block) {
          builder.setInsertionPointToStart(block);
          auto placeholder = builder.create<neura::ReserveOp>(loc, incoming.getType());
          phi_operands.push_back(placeholder.getResult());
          // Defers the backward ctrl move operation to be inserted after all phi operands
          // are defined. Inserted: (real_defined_value, just_created_reserve, current_block).
          deferred_ctrl_movs.emplace_back(incoming, placeholder.getResult(), block);
        } else {
          phi_operands.push_back(incoming);
        }
      }
    }

    assert(!phi_operands.empty());

    // Puts all operands into a set to ensure uniqueness. Specifically, following
    // case is handled:
    // ---------------------------------------------------------
    // ^bb1:
    //   "neura.br"(%a)[^bb3] : (!neura.data<f32, i1>) -> ()
    //
    // ^bb2:
    //   "neura.br"(%a)[^bb3] : (!neura.data<f32, i1>) -> ()
    //
    // ^bb3(%x: !neura.data<f32, i1>):
    //   ...
    // ---------------------------------------------------------
    // In above case, %a is used in both branches of the control flow, so we
    // don't need a phi node, but we still need to replace its uses with the
    // result of the phi node.
    // This ensures that we only create a phi node if there are multiple unique
    // operands.
    llvm::SmallDenseSet<Value, 4> unique_operands(phi_operands.begin(), phi_operands.end());

    if (unique_operands.size() == 1) {
      // No phi needed, but still replace
      Value single = *unique_operands.begin();
      SmallVector<OpOperand *, 4> uses;
      for (OpOperand &use : live_in.getUses()) {
        uses.push_back(&use);
      }
      for (OpOperand *use : uses) {
        use->set(single);
      }
      value_map[live_in] = single;
      continue;
    }

    // Creates the phi node with dynamic number of operands.
    auto phi_op = builder.create<neura::PhiOp>(loc, predicated_type, phi_operands);

    // Saves users to be replaced *after* phi is constructed.
    SmallVector<OpOperand *> uses_to_be_replaced;
    for (OpOperand &use : live_in.getUses()) {
      if (use.getOwner() != phi_op) {
        uses_to_be_replaced.push_back(&use);
      }
    }
    // Replaces live-in uses with the phi result.
    for (OpOperand *use : uses_to_be_replaced) {
      use->set(phi_op.getResult());
    }
    value_map[live_in] = phi_op.getResult();
  }
}

namespace {
struct TransformCtrlToDataFlowPass 
    : public PassWrapper<TransformCtrlToDataFlowPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TransformCtrlToDataFlowPass)

  StringRef getArgument() const override { return "transform-ctrl-to-data-flow"; }
  StringRef getDescription() const override {
    return "Transforms control flow into data flow using predicated execution";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::neura::NeuraDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Declares a vector to hold deferred backward ctrl move operations.
    // This is useful when a live-in value is defined within the same block.
    // The tuple contains:
    // - real value (the one that is defined in the same block, after the placeholder)
    // - placeholder value (the one that will be used in the phi node)
    // - block where the backward ctrl move should be inserted
    SmallVector<std::tuple<Value, Value, Block *>, 4> deferred_ctrl_movs;
    module.walk([&](func::FuncOp func) {
      // Get blocks in post-order
      SmallVector<Block *> postOrder;
      DenseSet<Block *> visited;
      getBlocksInPostOrder(&func.getBody().front(), postOrder, visited);

      // Value mapping for phi node creation.
      DenseMap<Value, Value> value_map;
      OpBuilder builder(func.getContext());

      // Process blocks bottom-up
      for (Block *block : postOrder) {
        // Creates phi nodes for live-ins.
        createPhiNodesForBlock(block, builder, value_map, deferred_ctrl_movs);
      }

      // Flattens blocks into the entry block.
      Block *entryBlock = &func.getBody().front();
      SmallVector<Block *> blocks_to_flatten;
      for (Block &block : func.getBody()) {
        if (&block != entryBlock)
          blocks_to_flatten.push_back(&block);
      }

      // Erases terminators before moving ops into entry block.
      for (Block *block : blocks_to_flatten) {
        for (Operation &op : llvm::make_early_inc_range(*block)) {
          if (isa<neura::Br>(op) || isa<neura::CondBr>(op)) {
            op.erase();
          }
        }
      }

      // Moves all operations from blocks to the entry block before the terminator.
      for (Block *block : blocks_to_flatten) {
        auto &ops = block->getOperations();
        while (!ops.empty()) {
          Operation &op = ops.front();
          op.moveBefore(&entryBlock->back());
        }
      }

      // Erases any remaining br/cond_br that were moved into the entry block.
      for (Operation &op : llvm::make_early_inc_range(*entryBlock)) {
        if (isa<neura::Br>(op) || isa<neura::CondBr>(op)) {
          op.erase();
        }
      }

      for (Block *block : blocks_to_flatten) {
        block->erase();
      }
    });

    // Inserts the deferred backward ctrl move operations after phi operands
    // are defined.
    for (auto &[realVal, placeholder, block] : deferred_ctrl_movs) {
      Operation *defOp = realVal.getDefiningOp();
      assert(defOp && "Backward ctrl move's source must be an op result");
      OpBuilder movBuilder(defOp->getBlock(), ++Block::iterator(defOp));
      movBuilder.create<neura::CtrlMovOp>(defOp->getLoc(), realVal, placeholder);
    }
  }
};
} // namespace

namespace mlir::neura {
std::unique_ptr<mlir::Pass> createTransformCtrlToDataFlowPass() {
  return std::make_unique<TransformCtrlToDataFlowPass>();
}
} // namespace mlir::neura