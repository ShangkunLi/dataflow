#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraOps.h"
#include "NeuraDialect/NeuraPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

#define GEN_PASS_DEF_InsertDataMov
#include "NeuraDialect/NeuraPasses.h.inc"

namespace {
struct InsertDataMovForNeuraOps : public RewritePattern {
  InsertDataMovForNeuraOps(MLIRContext *context)
      : RewritePattern(/*matchAnyOpTypeTag=*/MatchAnyOpTypeTag(), /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (op->getDialect()->getNamespace() != "neura" ||
        isa<neura::DataMovOp>(op)) {
      return failure();
    }

    // Skips ops that already being inserted mov on the operands.
    bool allInputsAreMov = llvm::all_of(op->getOperands(), [](Value v) {
      return isa_and_nonnull<neura::DataMovOp>(v.getDefiningOp());
    });
    if (allInputsAreMov) {
      return failure();
    }

    // Makes sure none of the operand has being processed.
    bool hasAnyMovInput = llvm::any_of(op->getOperands(), [](Value v) {
      return isa_and_nonnull<neura::DataMovOp>(v.getDefiningOp());
    });
    assert(!hasAnyMovInput && "Unexpected: operand already wrapped in neura.mov");

    Location loc = op->getLoc();

    // Wraps operands in mov.
    SmallVector<Value> newOperands;
    for (Value operand : op->getOperands()) {
      auto mov = rewriter.create<neura::DataMovOp>(loc, operand.getType(), operand);
      newOperands.push_back(mov);
    }

    // Clones op with new operands.
    OperationState state(loc, op->getName());
    state.addOperands(newOperands);
    state.addTypes(op->getResultTypes());
    state.addAttributes(op->getAttrs());

    // Copies successors for terminator operations.
    if (op->hasTrait<OpTrait::IsTerminator>()) {
      for (Block *successor : op->getSuccessors()) {
        state.addSuccessors(successor);
      }
    }

    Operation *newOp = rewriter.create(state);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

struct InsertDataMovPass
    : public PassWrapper<InsertDataMovPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InsertDataMovPass)

  StringRef getArgument() const override { return "insert-data-mov"; }
  StringRef getDescription() const override {
    return "Insert neura.data_mov before all neura dialect operations.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::neura::NeuraDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<InsertDataMovForNeuraOps>(&getContext());
    FrozenRewritePatternSet frozen(std::move(patterns));

    ModuleOp module_op = getOperation();

    // Applies to every region inside the module (regardless of func type,
    // e.g., mlir func or llvm func).
    module_op.walk([&](Operation *op) {
      if (!op->getRegions().empty()) {
        for (Region &region : op->getRegions()) {
          if (failed(applyPatternsGreedily(region, frozen))) {
            signalPassFailure();
          }
        }
      }
    });
  }
};
} // namespace

namespace mlir {
namespace neura {

std::unique_ptr<Pass> createInsertDataMovPass() {
  return std::make_unique<InsertDataMovPass>();
}

} // namespace neura
} // namespace mlir
