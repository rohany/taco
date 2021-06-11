#include <taco/lower/mode_format_compressed.h>
#include "taco/lower/lowerer_impl.h"

#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/index_notation_nodes.h"
#include "taco/index_notation/index_notation_visitor.h"
#include "taco/index_notation/index_notation_rewriter.h"
#include "taco/index_notation/provenance_graph.h"
#include "taco/ir/ir.h"
#include "ir/ir_generators.h"
#include "taco/ir/ir_visitor.h"
#include "taco/ir/simplify.h"
#include "taco/lower/iterator.h"
#include "taco/lower/merge_lattice.h"
#include "mode_access.h"
#include "taco/util/collections.h"
#include "taco/ir/ir_rewriter.h"
#include "taco/tensor.h"

using namespace std;
using namespace taco::ir;
using taco::util::combine;

namespace taco {

class LowererImpl::Visitor : public IndexNotationVisitorStrict {
public:
  Visitor(LowererImpl* impl) : impl(impl) {}
  Stmt lower(IndexStmt stmt) {
    this->stmt = Stmt();
    impl->accessibleIterators.scope();
    IndexStmtVisitorStrict::visit(stmt);
    impl->accessibleIterators.unscope();
    return this->stmt;
  }
  Expr lower(IndexExpr expr) {
    this->expr = Expr();
    IndexExprVisitorStrict::visit(expr);
    return this->expr;
  }
private:
  LowererImpl* impl;
  Expr expr;
  Stmt stmt;
  using IndexNotationVisitorStrict::visit;
  void visit(const AssignmentNode* node)    { stmt = impl->lowerAssignment(node); }
  void visit(const YieldNode* node)         { stmt = impl->lowerYield(node); }
  void visit(const ForallNode* node)        { stmt = impl->lowerForall(node); }
  void visit(const WhereNode* node)         { stmt = impl->lowerWhere(node); }
  void visit(const MultiNode* node)         { stmt = impl->lowerMulti(node); }
  void visit(const SuchThatNode* node)      { stmt = impl->lowerSuchThat(node); }
  void visit(const SequenceNode* node)      { stmt = impl->lowerSequence(node); }
  void visit(const AssembleNode* node)      { stmt = impl->lowerAssemble(node); }
  void visit(const AccessNode* node)        { expr = impl->lowerAccess(node); }
  void visit(const LiteralNode* node)       { expr = impl->lowerLiteral(node); }
  void visit(const NegNode* node)           { expr = impl->lowerNeg(node); }
  void visit(const AddNode* node)           { expr = impl->lowerAdd(node); }
  void visit(const SubNode* node)           { expr = impl->lowerSub(node); }
  void visit(const MulNode* node)           { expr = impl->lowerMul(node); }
  void visit(const DivNode* node)           { expr = impl->lowerDiv(node); }
  void visit(const SqrtNode* node)          { expr = impl->lowerSqrt(node); }
  void visit(const CastNode* node)          { expr = impl->lowerCast(node); }
  void visit(const CallIntrinsicNode* node) { expr = impl->lowerCallIntrinsic(node); }
  void visit(const ReductionNode* node)  {
    taco_ierror << "Reduction nodes not supported in concrete index notation";
  }
  void visit(const PlaceNode* node) { expr = impl->lower(node->expr); }
};

LowererImpl::LowererImpl() : visitor(new Visitor(this)) {
}


static void createCapacityVars(const map<TensorVar, Expr>& tensorVars,
                               map<Expr, Expr>* capacityVars) {
  for (auto& tensorVar : tensorVars) {
    Expr tensor = tensorVar.second;
    Expr capacityVar = Var::make(util::toString(tensor) + "_capacity", Int());
    capacityVars->insert({tensor, capacityVar});
  }
}

static void createReducedValueVars(const vector<Access>& inputAccesses,
                                   map<Access, Expr>* reducedValueVars) {
  for (const auto& access : inputAccesses) {
    const TensorVar inputTensor = access.getTensorVar();
    const std::string name = inputTensor.getName() + "_val";
    const Datatype type = inputTensor.getType().getDataType();
    reducedValueVars->insert({access, Var::make(name, type)});
  }
}

static void getDependentTensors(IndexStmt stmt, std::set<TensorVar>& tensors) {
  std::set<TensorVar> prev;
  do {
    prev = tensors;
    match(stmt,
      function<void(const AssignmentNode*, Matcher*)>([&](
          const AssignmentNode* n, Matcher* m) {
        if (util::contains(tensors, n->lhs.getTensorVar())) {
          const auto arguments = getArguments(Assignment(n));
          tensors.insert(arguments.begin(), arguments.end());
        }
      })
    );
  } while (prev != tensors);
}

static bool needComputeValues(IndexStmt stmt, TensorVar tensor) {
  if (tensor.getType().getDataType() != Bool) {
    return true;
  }

  struct ReturnsTrue : public IndexExprRewriterStrict {
    void visit(const AccessNode* op) {
      if (op->isAccessingStructure) {
        expr = op;
      }
    }

    void visit(const LiteralNode* op) {
      if (op->getDataType() == Bool && op->getVal<bool>()) {
        expr = op;
      }
    }

    void visit(const NegNode* op) {
      expr = rewrite(op->a);
    }

    void visit(const AddNode* op) {
      if (rewrite(op->a).defined() || rewrite(op->b).defined()) {
        expr = op;
      }
    }

    void visit(const MulNode* op) {
      if (rewrite(op->a).defined() && rewrite(op->b).defined()) {
        expr = op;
      }
    }

    void visit(const CastNode* op) {
      expr = rewrite(op->a);
    }

    void visit(const SqrtNode* op) {}
    void visit(const SubNode* op) {}
    void visit(const DivNode* op) {}
    void visit(const CallIntrinsicNode* op) {}
    void visit(const ReductionNode* op) {}
  };

  bool needComputeValue = false;
  match(stmt,
    function<void(const AssignmentNode*, Matcher*)>([&](
        const AssignmentNode* n, Matcher* m) {
      if (n->lhs.getTensorVar() == tensor &&
          !ReturnsTrue().rewrite(n->rhs).defined()) {
        needComputeValue = true;
      }
    })
  );

  return needComputeValue;
}

/// Returns true iff a result mode is assembled by inserting a sparse set of
/// result coordinates (e.g., compressed to dense).
static
bool hasSparseInserts(const std::vector<Iterator>& resultIterators,
                      const std::multimap<IndexVar, Iterator>& inputIterators) {
  for (const auto& resultIterator : resultIterators) {
    if (resultIterator.hasInsert()) {
      const auto indexVar = resultIterator.getIndexVar();
      const auto accessedInputs = inputIterators.equal_range(indexVar);
      for (auto inputIterator = accessedInputs.first;
           inputIterator != accessedInputs.second; ++inputIterator) {
        if (!inputIterator->second.isFull()) {
          return true;
        }
      }
    }
  }
  return false;
}

Stmt
LowererImpl::lower(IndexStmt stmt, string name,
                   bool assemble, bool compute, bool pack, bool unpack)
{
  this->assemble = assemble;
  this->compute = compute;
  this->legion = name.find("Legion") != std::string::npos;
  definedIndexVarsOrdered = {};
  definedIndexVars = {};

  // Create result and parameter variables
  vector<TensorVar> results = getResults(stmt);
  vector<TensorVar> arguments = getArguments(stmt);
  vector<TensorVar> temporaries = getTemporaries(stmt);

  this->resultTensors.insert(results.begin(), results.end());

  needCompute = {};
  if (generateAssembleCode()) {
    const auto attrQueryResults = getAttrQueryResults(stmt);
    needCompute.insert(attrQueryResults.begin(), attrQueryResults.end());
  }
  if (generateComputeCode()) {
    needCompute.insert(results.begin(), results.end());
  }
  getDependentTensors(stmt, needCompute);

  assembledByUngroupedInsert = util::toSet(
      getAssembledByUngroupedInsertion(stmt));

  // Create datastructure needed for temporary workspace hoisting/reuse
  temporaryInitialization = getTemporaryLocations(stmt);

  // Convert tensor results and arguments IR variables
  map<TensorVar, Expr> resultVars;
  vector<Expr> resultsIR = createVars(results, &resultVars, unpack);
  tensorVars.insert(resultVars.begin(), resultVars.end());
  vector<Expr> argumentsIR = createVars(arguments, &tensorVars, pack);

  // Create variables for index sets on result tensors.
  vector<Expr> indexSetArgs;
  for (auto& access : getResultAccesses(stmt).first) {
    // Any accesses that have index sets will be added.
    if (access.hasIndexSetModes()) {
      for (size_t i = 0; i < access.getIndexVars().size(); i++) {
        if (access.isModeIndexSet(i)) {
          auto t = access.getModeIndexSetTensor(i);
          if (tensorVars.count(t) == 0) {
            ir::Expr irVar = ir::Var::make(t.getName(), t.getType().getDataType(), true, true, pack);
            tensorVars.insert({t, irVar});
            indexSetArgs.push_back(irVar);
          }
        }
      }
    }
  }
  argumentsIR.insert(argumentsIR.begin(), indexSetArgs.begin(), indexSetArgs.end());

  // Create variables for temporaries
  // TODO Remove this
  for (auto& temp : temporaries) {
    ir::Expr irVar = ir::Var::make(temp.getName(), temp.getType().getDataType(),
                                   true, true);
    tensorVars.insert({temp, irVar});
  }

  // Create variables for keeping track of result values array capacity
  createCapacityVars(resultVars, &capacityVars);

  // Create iterators
  iterators = Iterators(stmt, tensorVars);

  provGraph = ProvenanceGraph(stmt);

  for (const IndexVar& indexVar : provGraph.getAllIndexVars()) {
    if (iterators.modeIterators().count(indexVar)) {
      indexVarToExprMap.insert({indexVar, iterators.modeIterators()[indexVar].getIteratorVar()});
    }
    else {
      indexVarToExprMap.insert({indexVar, Var::make(indexVar.getName(), Int())});
    }
  }

  vector<Access> inputAccesses, resultAccesses;
  set<Access> reducedAccesses;
  inputAccesses = getArgumentAccesses(stmt);
  std::tie(resultAccesses, reducedAccesses) = getResultAccesses(stmt);

  // Create variables that represent the reduced values of duplicated tensor
  // components
  createReducedValueVars(inputAccesses, &reducedValueVars);

  map<TensorVar, Expr> scalars;

  // Define and initialize dimension variables
  set<TensorVar> temporariesSet(temporaries.begin(), temporaries.end());
  vector<IndexVar> indexVars = getIndexVars(stmt);
  for (auto& indexVar : indexVars) {
    Expr dimension;
    // getDimension extracts an Expr that holds the dimension
    // of a particular tensor mode. This Expr should be used as a loop bound
    // when iterating over the dimension of the target tensor.
    auto getDimension = [&](const TensorVar& tv, const Access& a, int mode) {
      // If the tensor mode is windowed, then the dimension for iteration is the bounds
      // of the window. Otherwise, it is the actual dimension of the mode.
      if (a.isModeWindowed(mode)) {
        // The mode value used to access .levelIterator is 1-indexed, while
        // the mode input to getDimension is 0-indexed. So, we shift it up by 1.
        auto iter = iterators.levelIterator(ModeAccess(a, mode+1));
        return ir::Div::make(ir::Sub::make(iter.getWindowUpperBound(), iter.getWindowLowerBound()), iter.getStride());
      } else if (a.isModeIndexSet(mode)) {
        // If the mode has an index set, then the dimension is the size of
        // the index set.
        return ir::Literal::make(a.getIndexSet(mode).size());
      } else {
        return GetProperty::make(tensorVars.at(tv), TensorProperty::Dimension, mode);
      }
    };
    match(stmt,
      function<void(const AssignmentNode*, Matcher*)>([&](
          const AssignmentNode* n, Matcher* m) {
        m->match(n->rhs);
        if (!dimension.defined()) {
          auto ivars = n->lhs.getIndexVars();
          auto tv = n->lhs.getTensorVar();
          int loc = (int)distance(ivars.begin(),
                                  find(ivars.begin(),ivars.end(), indexVar));
          if(!util::contains(temporariesSet, tv)) {
            dimension = getDimension(tv, n->lhs, loc);
          }
        }
      }),
      function<void(const AccessNode*)>([&](const AccessNode* n) {
        auto indexVars = n->indexVars;
        if (util::contains(indexVars, indexVar)) {
          int loc = (int)distance(indexVars.begin(),
                                  find(indexVars.begin(),indexVars.end(),
                                       indexVar));
          if(!util::contains(temporariesSet, n->tensorVar)) {
            dimension = getDimension(n->tensorVar, Access(n), loc);
          }
        }
      })
    );

    // TODO (rohany): Big Hack: If an index var is unbounded (can happen if we're generating
    //  data placement code over a processor grid that is higher dimensional than the tensor
    //  itself), then just substitute a dummy value for the dimension.
    if (!dimension.defined()) {
      dimension = ir::GetProperty::make(this->tensorVars.begin()->second, TensorProperty::Dimension, 0);
    }

    dimensions.insert({indexVar, dimension});
    underivedBounds.insert({indexVar, {ir::Literal::make(0), dimension}});
  }

  // Define and initialize scalar results and arguments
  if (generateComputeCode()) {
    for (auto& result : results) {
      if (isScalar(result.getType())) {
        taco_iassert(!util::contains(scalars, result));
        taco_iassert(util::contains(tensorVars, result));
        scalars.insert({result, tensorVars.at(result)});
        header.push_back(defineScalarVariable(result, true));
      }
    }
    for (auto& argument : arguments) {
      if (isScalar(argument.getType())) {
        taco_iassert(!util::contains(scalars, argument));
        taco_iassert(util::contains(tensorVars, argument));
        scalars.insert({argument, tensorVars.at(argument)});
        header.push_back(defineScalarVariable(argument, false));
      }
    }
  }

  // Allocate memory for scalar results
  if (generateAssembleCode()) {
    for (auto& result : results) {
      if (result.getOrder() == 0) {
        Expr resultIR = resultVars.at(result);
        Expr vals = GetProperty::make(resultIR, TensorProperty::Values);
        header.push_back(Allocate::make(vals, 1));
      }
    }
  }

  // If we're computing on a partition, then make a variable for the partition, and add
  // it to the function inputs.
  TensorVar computingOn;
  match(stmt, function<void(const ForallNode*)>([&](const ForallNode* node) {
    if (node->computingOn.defined()) {
      computingOn = node->computingOn;
    }
  }));
  if (computingOn.defined()) {
    this->computingOnPartition = ir::Var::make(computingOn.getName() + "Partition", LogicalPartition);
    argumentsIR.push_back(this->computingOnPartition);
  }

  // If there are distributed loops, and no transfers present for an access, then that
  // transfer is occurring at the top level, so add it here.
  Stmt topLevelTransfers;
  bool foundDistributed = false;
  match(stmt, function<void(const ForallNode*)>([&](const ForallNode* node) {
    foundDistributed |= distributedParallelUnit(node->parallel_unit);
  }));
  if (foundDistributed) {
    // Collect all transfers in the index stmt.
//    Transfers transfers;
    std::vector<TensorVar> transfers;
    match(stmt, function<void(const ForallNode*)>([&](const ForallNode* node) {
      for (auto& t : node->transfers) {
        transfers.push_back(t.getAccess().getTensorVar());
      }
      if (node->computingOn.defined()) {
        transfers.push_back(node->computingOn);
      }
    }));

    auto hasTransfer = [&](Access a) {
      for (auto& t : transfers) {
        if (t == a.getTensorVar()) { return true; }
      }
      return false;
    };

    // For all accesses, see if they have transfers.
    std::vector<Access> accessesWithoutTransfers;
    match(stmt, function<void(const AccessNode*)>([&](const AccessNode* node) {
      Access a(node);
      if (!hasTransfer(a)) { accessesWithoutTransfers.push_back(a); }
    }), function<void(const AssignmentNode*)>([&](const AssignmentNode* node) {
      if (!hasTransfer(node->lhs)) { accessesWithoutTransfers.push_back(node->lhs); }
    }));

    std::vector<Stmt> stmts;
    for (auto t : accessesWithoutTransfers) {
      auto v = ir::Var::make("tx", Datatype::Int32);
      auto tv = ir::Var::make(t.getTensorVar().getName(), Datatype::Int32);
      auto fcall = ir::Call::make("top_level_transfer", {tv}, Datatype::Int32);
      stmts.push_back(ir::Assign::make(v, fcall));
    }
    topLevelTransfers = ir::Block::make(stmts);
  }

  // Allocate and initialize append and insert mode indices
  // TODO (rohany): I don't think that I want this. Or at least, it needs to be changed
  //  to not write out of partitions.
  Stmt initializeResults = initResultArrays(resultAccesses, inputAccesses,
                                            reducedAccesses);
  if (this->legion) {
    initializeResults = ir::Block::make();
  }



  // Begin hacking on bounds inference.

  // BoundsInferenceExprRewriter rewrites ...
  // TODO (rohany): I don't have a solid understanding yet of what this really does.
  struct BoundsInferenceExprRewriter : public IRRewriter {
    BoundsInferenceExprRewriter(ProvenanceGraph &pg, Iterators &iterators,
                                std::map<IndexVar, std::vector<ir::Expr>> &underivedBounds,
                                std::map<IndexVar, ir::Expr> &indexVarToExprMap,
                                std::set<IndexVar> &inScopeVars,
                                std::map<ir::Expr, IndexVar>& exprToIndexVarMap,
                                std::vector<IndexVar>& definedIndexVars,
                                bool lower,
                                std::set<IndexVar> presentIvars)
        : pg(pg), iterators(iterators), underivedBounds(underivedBounds), indexVarToExprMap(indexVarToExprMap),
          definedIndexVars(definedIndexVars), exprToIndexVarMap(exprToIndexVarMap), inScopeVars(inScopeVars), lower(lower),
          presentIvars(presentIvars) {}

    void visit(const Var* var) {
      // If there is a var that isn't an index variable (like a partition bounds var),
      // then just return.
      if (this->exprToIndexVarMap.count(var) == 0) {
        expr = var;
        return;
      }
      auto ivar = this->exprToIndexVarMap.at(var);
      if (util::contains(this->inScopeVars, ivar)) {
        // If this ivar is in scope of the request, then access along it is fixed.
        expr = var;
      } else {

        // If a variable being derived is not even going to be present in the loop
        // (i.e. a variable that we split again), then we might want to expand it
        // into the variables that derive it. However, if neither of those variables
        // are in scope, then the bounds the provenance graph provides us for the
        // suspect variable are the ones we should take.
        if (!util::contains(this->presentIvars, ivar)) {
          struct InscopeVarVisitor : public IRVisitor {
            InscopeVarVisitor(ProvenanceGraph& pg) : pg(pg) {}
            void visit(const Var* var) {
              auto ivar = this->exprToIndexVarMap[var];
              if (util::contains(this->inScopeVars, ivar)) {
                this->anyInScope = true;
                return;
              }
              // There's a special case here for staggered variables. These variables
              // aren't really the subject of a parent-child relationship, so we flatten
              // that relationship here when looking at bounds.
              auto res = this->pg.getStaggeredVar(ivar);
              if (res.first) {
                if (util::contains(this->inScopeVars, res.second)) {
                  this->anyInScope = true;
                  return;
                }
              }
            }
            std::set<IndexVar> inScopeVars;
            std::map<Expr, IndexVar> exprToIndexVarMap;
            bool anyInScope = false;
            ProvenanceGraph& pg;
          };
          InscopeVarVisitor isv(this->pg); isv.inScopeVars = this->inScopeVars; isv.exprToIndexVarMap = this->exprToIndexVarMap;
          auto recovered = this->pg.recoverVariable(ivar, this->definedIndexVars, this->underivedBounds, this->indexVarToExprMap, this->iterators);
          recovered.accept(&isv);
          // If there are some variables in scope, use this as the rewritten expression.
          // A future call to the rewriter will expand the resulting variables.
          if (isv.anyInScope) {
            this->expr = recovered;
            this->changed = true;
            return;
          }
        }

        // Otherwise, the full bounds of this ivar will be accessed. So, derive the
        // bounds. Depending on whether we are deriving a lower or upper bound, use the
        // appropriate one.
        auto bounds = this->pg.deriveIterBounds(ivar, this->definedIndexVars, this->underivedBounds, this->indexVarToExprMap, this->iterators);
        auto idx = lower ? 0 : 1;
        this->changed = true;
        // If we are deriving an upper bound, we substitute an inclusive
        // bound here. This ensures that we calculate indices for only the
        // exact locations we access, and will map cleanly to Legion partitioning.
        expr = ir::Sub::make(bounds[idx], ir::Literal::make(idx));
      }
    }

    void visit(const GetProperty* gp) {
      // TODO (rohany): For some reason, I need to have this empty visit method
      //  for GetProperty here.
      expr = gp;
    }

    ProvenanceGraph& pg;
    Iterators& iterators;
    std::map<IndexVar, std::vector<ir::Expr>>& underivedBounds;
    std::map<IndexVar, ir::Expr>& indexVarToExprMap;
    std::vector<IndexVar>& definedIndexVars;

    std::map<ir::Expr, IndexVar>& exprToIndexVarMap;

    std::set<IndexVar>& inScopeVars;

    bool lower;

    std::set<IndexVar> presentIvars;

    bool changed = false;
  };

  // BoundsInferenceVisitor infers the exact bounds (inclusive) that each tensor is accessed on.
  // In an actual implementation, this would happen as part of the lowering process, not a separate step.
  struct BoundsInferenceVisitor : public IndexNotationVisitor {

    BoundsInferenceVisitor(std::map<TensorVar, Expr> &tvs, ProvenanceGraph &pg, Iterators &iterators,
                           std::map<IndexVar, std::vector<ir::Expr>>& underivedBounds, std::map<IndexVar, ir::Expr>& indexVarToExprMap,
                           std::set<IndexVar> presentIvars)
        : pg(pg), iterators(iterators), underivedBounds(underivedBounds), indexVarToExprMap(indexVarToExprMap),
        presentIvars(presentIvars) {
      for (auto &it : tvs) {
        this->inScopeVars[it.first] = {};
      }
      for (auto& it : indexVarToExprMap) {
        exprToIndexVarMap[it.second] = it.first;
      }
    }

    void inferBounds(IndexStmt stmt) {
      IndexStmtVisitorStrict::visit(stmt);
    }
    void inferBounds(IndexExpr expr) {
      IndexExprVisitorStrict::visit(expr);
    }

    void visit(const ForallNode* node) {
      if (node == this->trackingForall) {
        this->tracking = true;
      }

      // Add the forall variable to the scope for each tensorVar that hasn't
      // been requested yet.
      for (auto& it : this->inScopeVars) {
        if (!util::contains(this->requestedTensorVars, it.first)) {
          it.second.insert(node->indexVar);
          auto fused = this->pg.getMultiFusedParents(node->indexVar);
          it.second.insert(fused.begin(), fused.end());
        }
      }

      if (this->tracking || (this->trackingForall == nullptr)) {
        for (auto& t : node->transfers) {
          this->requestedTensorVars.insert(t.getAccess().getTensorVar());
        }
        if (node->computingOn.defined()) {
          this->requestedTensorVars.insert(node->computingOn);
        }
      }

      // Recurse down the index statement.
      this->definedIndexVars.push_back(node->indexVar);
      this->forallDepth++;
      this->inferBounds(node->stmt);
    }

    void visit(const AssignmentNode* node) {
      this->inferBounds(node->lhs);
      this->inferBounds(node->rhs);
    }

    void visit(const AccessNode* node) {
      // For each variable of the access, find its bounds.
      for (auto& var : node->indexVars) {
        auto children = this->pg.getChildren(var);
        // If the index variable has no children, then it is a raw access.
        if (children.size() == 0) {
          // If the index variable is in scope for the request, then we will need to
          // just access that point of the index variable. Otherwise, we will access
          // the full bounds of that variable.
          if (util::contains(this->inScopeVars[node->tensorVar], var)) {
            auto expr = this->indexVarToExprMap[var];
            this->derivedBounds[node->tensorVar].push_back({expr, expr});
          } else {
            this->derivedBounds[node->tensorVar].push_back(this->pg.deriveIterBounds(var, this->definedIndexVars, this->underivedBounds, this->indexVarToExprMap, this->iterators));
          }
        } else {
          // If the index variable has children, then we need to recover how it accesses
          // the tensors in the expression based on how those children are made. We first
          // calculate how to recover the index variable.
          auto accessExpr = this->pg.recoverVariable(var, this->definedIndexVars, this->underivedBounds, this->indexVarToExprMap, this->iterators);

          // Next, we repeatedly replace variables the recovered expression until it
          // no longer changes. Exactly how the rewriting is done is detailed in the
          // BoundsInferenceExprRewriter.
          auto rwFn = [&](bool lower, ir::Expr bound) {
            BoundsInferenceExprRewriter rw(this->pg, this->iterators, this->underivedBounds, this->indexVarToExprMap,
                                           this->inScopeVars[node->tensorVar], this->exprToIndexVarMap,
                                           this->definedIndexVars, lower, this->presentIvars);
            do {
              rw.changed = false;
              bound = rw.rewrite(bound);
            } while(rw.changed);
            return bound;
          };
          auto lo = ir::simplify(rwFn(true, accessExpr));
          auto hi = ir::simplify(rwFn(false, accessExpr));
          this->derivedBounds[node->tensorVar].push_back({lo, hi});
        }
      }
    }

    ProvenanceGraph& pg;
    Iterators& iterators;
    std::map<IndexVar, std::vector<ir::Expr>>& underivedBounds;
    std::map<IndexVar, ir::Expr>& indexVarToExprMap;

    std::map<ir::Expr, IndexVar> exprToIndexVarMap;

    std::vector<IndexVar> definedIndexVars;
    std::map<TensorVar, std::set<IndexVar>> inScopeVars;
    std::set<TensorVar> requestedTensorVars;

    std::set<IndexVar> presentIvars;

    std::map<TensorVar, std::vector<std::vector<ir::Expr>>> derivedBounds;

    const ForallNode* trackingForall = nullptr;
    bool tracking = false;

    int forallDepth = 0;
  };

  std::set<IndexVar> presentIvars;
  match(stmt, function<void(const ForallNode*)>([&](const ForallNode* f) {
    auto fused = this->provGraph.getMultiFusedParents(f->indexVar);
    if (fused.size() > 0) {
      presentIvars.insert(fused.begin(), fused.end());
    } else {
      presentIvars.insert(f->indexVar);
    }
  }));

  for (auto& it : this->tensorVars) {
    auto pointT = Point(it.first.getType().getOrder());
    auto accessor = ir::Var::make(it.first.getName() + "_access_point", pointT);
    this->pointAccessVars[it.first] = accessor;
  }

  match(stmt, function<void(const ForallNode*)>([&](const ForallNode* node) {
    // Want to derive bounds for each distributed forall. Can worry about how to
    // connect this all together later.
    auto f = Forall(node);
    if (f.isDistributed()) {
      // Get bounds for this forall.
      BoundsInferenceVisitor bi(this->tensorVars, this->provGraph, this->iterators, this->underivedBounds, this->indexVarToExprMap, presentIvars);
      bi.trackingForall = node;
      bi.inferBounds(stmt);
      // std::cout << "Bounds for index var: " << f.getIndexVar() << " at forall: " << f << std::endl;
      // for (auto it : bi.derivedBounds) {
      //   cout << "Bounds for: " << it.first.getName() << endl;
      //   for (auto& bounds : it.second) {
      //     cout << util::join(bounds) << endl;
      //   }
      // }
      this->derivedBounds[f.getIndexVar()] = bi.derivedBounds;
    }
  }));

  if (this->legion) {
    auto lookupTV = [&](ir::Expr e) {
      for (auto it : this->tensorVars) {
        if (it.second == e) {
          return it.first;
        }
      }
      taco_ierror << "couldn't reverse lookup tensor: " << e << "in: " << util::join(this->tensorVars) << std::endl;
      return TensorVar();
    };

    for (auto ir : resultsIR) {
      this->tensorVarOrdering.push_back(lookupTV(ir));
    }
    for (auto ir : argumentsIR) {
      if (ir.as<Var>() && ir.as<Var>()->is_tensor) {
        this->tensorVarOrdering.push_back(lookupTV(ir));
      }
    }
  }

  match(stmt, function<void(const PlaceNode*)>([&](const PlaceNode* node) {
    this->isPlacementCode = true;
    this->placements = node->placements;
  }));

  if (this->isPlacementCode) {
    // Set up Face() index launch bounds restrictions for any placement operations
    // that use Face().
    struct IndexVarFaceCollector : public IndexNotationVisitor {
      IndexVarFaceCollector(std::map<IndexVar, int>& indexVarFaces,
                            std::vector<std::pair<Grid, GridPlacement>>& placements,
                            ProvenanceGraph& pg)
        : indexVarFaces(indexVarFaces), placements(placements), pg(pg) {}

      void visit (const ForallNode* node) {
        if (distributedParallelUnit(node->parallel_unit)) {
          auto fused = this->pg.getMultiFusedParents(node->indexVar);
          taco_iassert(fused.size()  > 0);
          auto placement = this->placements[distIndex].second;
          taco_iassert(fused.size() == placement.axes.size());
          // For all positions that are restricted to a Face of the processor grid,
          // override the iteration bounds of that variable to just that face of the
          // grid.
          for (size_t i = 0; i < placement.axes.size(); i++) {
            auto axis = placement.axes[i];
            if (axis.kind == GridPlacement::AxisMatch::Face) {
              this->indexVarFaces[fused[i]] = axis.face;
            }
          }
          distIndex++;
        }
        node->stmt.accept(this);
      }

      int distIndex = 0;
      std::map<IndexVar, int>& indexVarFaces;
      std::vector<std::pair<Grid, GridPlacement>>& placements;
      ProvenanceGraph& pg;
    };
    IndexVarFaceCollector fc(this->indexVarFaces, this->placements, this->provGraph);
    stmt.accept(&fc);
  }

  // Lower the index statement to compute and/or assemble
  Stmt body = lower(stmt);

  // Post-process result modes and allocate memory for values if necessary
  Stmt finalizeResults = finalizeResultArrays(resultAccesses);

  // Collect an add any parameter variables to the function's inputs.
  struct ParameterFinder : public IRVisitor {
    void visit(const Var* node) {
      if (node->is_parameter) {
        if (!util::contains(this->collectedVars, node)) {
          vars.push_back(node);
          collectedVars.insert(node);
        }
      }
    }
    std::vector<ir::Expr> vars;
    std::set<ir::Expr> collectedVars;
  } pfinder; body.accept(&pfinder);

  // Store scalar stack variables back to results
  if (generateComputeCode()) {
    for (auto& result : results) {
      if (isScalar(result.getType())) {
        taco_iassert(util::contains(scalars, result));
        taco_iassert(util::contains(tensorVars, result));
        Expr resultIR = scalars.at(result);
        Expr varValueIR = tensorVars.at(result);
        Expr valuesArrIR = GetProperty::make(resultIR, TensorProperty::Values);
        footer.push_back(Store::make(valuesArrIR, 0, varValueIR, markAssignsAtomicDepth > 0, atomicParallelUnit));
      }
    }
  }

  // Create function
  return Function::make(name, resultsIR, util::combine(argumentsIR, pfinder.vars),
                        Block::blanks(Block::make(header),
                                      initializeResults,
                                      topLevelTransfers,
                                      body,
                                      finalizeResults,
                                      Block::make(footer)));
}


Stmt LowererImpl::lowerAssignment(Assignment assignment)
{
  taco_iassert(generateAssembleCode() || generateComputeCode());

  Stmt computeStmt;
  TensorVar result = assignment.getLhs().getTensorVar();
  Expr var = getTensorVar(result);

  const bool needComputeAssign = util::contains(needCompute, result);

  Expr rhs;
  if (needComputeAssign) {
    rhs = lower(assignment.getRhs());
  }

  // Assignment to scalar variables.
  if (isScalar(result.getType())) {
    if (needComputeAssign) {
      if (!assignment.getOperator().defined()) {
        computeStmt = Assign::make(var, rhs);
      }
      else {
        taco_iassert(isa<taco::Add>(assignment.getOperator()));
        bool useAtomics = markAssignsAtomicDepth > 0 &&
                          !util::contains(whereTemps, result);
        // TODO (rohany): Might have to do a reduction assignment here?
        computeStmt = compoundAssign(var, rhs, useAtomics, atomicParallelUnit);
      }
    }
  }
  // Assignments to tensor variables (non-scalar).
  else {
    Expr values = getValuesArray(result);
    Expr loc = generateValueLocExpr(assignment.getLhs());

    std::vector<Stmt> accessStmts;

    if (isAssembledByUngroupedInsertion(result)) {
      std::vector<Expr> coords;
      Expr prevPos = 0;
      size_t i = 0;
      const auto resultIterators = getIterators(assignment.getLhs());
      for (const auto& it : resultIterators) {
        // TODO: Should only assemble levels that can be assembled together
        //if (it == this->nextTopResultIterator) {
        //  break;
        //}

        coords.push_back(getCoordinateVar(it));

        const auto yieldPos = it.getYieldPos(prevPos, coords);
        accessStmts.push_back(yieldPos.compute());
        Expr pos = it.getPosVar();
        accessStmts.push_back(VarDecl::make(pos, yieldPos[0]));

        if (generateAssembleCode()) {
          accessStmts.push_back(it.getInsertCoord(prevPos, pos, coords));
        }

        prevPos = pos;
        ++i;
      }
    }

    if (needComputeAssign && values.defined()) {
      if (!assignment.getOperator().defined()) {
        computeStmt = Store::make(values, loc, rhs);
      }
      else {
        if (this->legion && this->performingLegionReduction) {
          computeStmt = Store::make(
              values, loc, rhs, false, ParallelUnit::LegionReduction
          );
        } else {
          computeStmt = compoundStore(values, loc, rhs,
                                      markAssignsAtomicDepth > 0,
                                      atomicParallelUnit);
        }
      }
      taco_iassert(computeStmt.defined());
    }

    if (!accessStmts.empty()) {
      accessStmts.push_back(computeStmt);
      computeStmt = Block::make(accessStmts);
    }
  }

  if (util::contains(guardedTemps, result) && result.getOrder() == 0) {
    Expr guard = tempToBitGuard[result];
    Stmt setGuard = Assign::make(guard, true, markAssignsAtomicDepth > 0,
                                 atomicParallelUnit);
    computeStmt = Block::make(computeStmt, setGuard);
  }

  Expr assembleGuard = generateAssembleGuard(assignment.getRhs());
  const bool assembleGuardTrivial = isa<ir::Literal>(assembleGuard);

  // TODO: If only assembling so defer allocating value memory to the end when
  //       we'll know exactly how much we need.
  bool temporaryWithSparseAcceleration = util::contains(tempToIndexList, result);
  if (generateComputeCode() && !temporaryWithSparseAcceleration) {
    taco_iassert(computeStmt.defined());
    return assembleGuardTrivial ? computeStmt : IfThenElse::make(assembleGuard,
                                                                 computeStmt);
  }

  if (temporaryWithSparseAcceleration) {
    taco_iassert(markAssignsAtomicDepth == 0)
      << "Parallel assembly of sparse accelerator not supported";

    Expr values = getValuesArray(result);
    Expr loc = generateValueLocExpr(assignment.getLhs());

    Expr bitGuardArr = tempToBitGuard.at(result);
    Expr indexList = tempToIndexList.at(result);
    Expr indexListSize = tempToIndexListSize.at(result);

    Stmt markBitGuardAsTrue = Store::make(bitGuardArr, loc, true);
    Stmt trackIndex = Store::make(indexList, indexListSize, loc);
    Expr incrementSize = ir::Add::make(indexListSize, 1);
    Stmt incrementStmt = Assign::make(indexListSize, incrementSize);

    Stmt firstWriteAtIndex = Block::make(trackIndex, markBitGuardAsTrue, incrementStmt);
    if (needComputeAssign && values.defined()) {
      Stmt initialStorage = computeStmt;
      if (assignment.getOperator().defined()) {
        // computeStmt is a compund stmt so we need to emit an initial store
        // into the temporary
        initialStorage =  Store::make(values, loc, rhs);
      }
      firstWriteAtIndex = Block::make(initialStorage, firstWriteAtIndex);
    }

    Expr readBitGuard = Load::make(bitGuardArr, loc);
    computeStmt = IfThenElse::make(ir::Neg::make(readBitGuard),
                                   firstWriteAtIndex, computeStmt);
  }

  return assembleGuardTrivial ? computeStmt : IfThenElse::make(assembleGuard,
                                                               computeStmt);
}


  Stmt LowererImpl::lowerYield(Yield yield) {
  std::vector<Expr> coords;
  for (auto& indexVar : yield.getIndexVars()) {
    coords.push_back(getCoordinateVar(indexVar));
  }
  Expr val = lower(yield.getExpr());
  return ir::Yield::make(coords, val);
}


pair<vector<Iterator>, vector<Iterator>>
LowererImpl::splitAppenderAndInserters(const vector<Iterator>& results) {
  vector<Iterator> appenders;
  vector<Iterator> inserters;

  // TODO: Choose insert when the current forall is nested inside a reduction
  for (auto& result : results) {
    if (isAssembledByUngroupedInsertion(result.getTensor())) {
      continue;
    }

    taco_iassert(result.hasAppend() || result.hasInsert())
        << "Results must support append or insert";

    if (result.hasAppend()) {
      appenders.push_back(result);
    }
    else {
      taco_iassert(result.hasInsert());
      inserters.push_back(result);
    }
  }

  return {appenders, inserters};
}


Stmt LowererImpl::lowerForall(Forall forall)
{
  bool hasExactBound = provGraph.hasExactBound(forall.getIndexVar());
  bool forallNeedsUnderivedGuards = !hasExactBound && emitUnderivedGuards;
  if (!ignoreVectorize && forallNeedsUnderivedGuards &&
      (forall.getParallelUnit() == ParallelUnit::CPUVector ||
       forall.getUnrollFactor() > 0)) {
    return lowerForallCloned(forall);
  }

  if (forall.getParallelUnit() != ParallelUnit::NotParallel) {
    inParallelLoopDepth++;
  }

  // Recover any available parents that were not recoverable previously
  vector<Stmt> recoverySteps;
  for (const IndexVar& varToRecover : provGraph.newlyRecoverableParents(forall.getIndexVar(), definedIndexVars)) {
    // place pos guard
    if (forallNeedsUnderivedGuards && provGraph.isCoordVariable(varToRecover) &&
        provGraph.getChildren(varToRecover).size() == 1 &&
        provGraph.isPosVariable(provGraph.getChildren(varToRecover)[0])) {
      IndexVar posVar = provGraph.getChildren(varToRecover)[0];
      std::vector<ir::Expr> iterBounds = provGraph.deriveIterBounds(posVar, definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, iterators);

      Expr minGuard = Lt::make(indexVarToExprMap[posVar], iterBounds[0]);
      Expr maxGuard = Gte::make(indexVarToExprMap[posVar], iterBounds[1]);
      Expr guardCondition = Or::make(minGuard, maxGuard);
      if (isa<ir::Literal>(ir::simplify(iterBounds[0])) && ir::simplify(iterBounds[0]).as<ir::Literal>()->equalsScalar(0)) {
        guardCondition = maxGuard;
      }
      ir::Stmt guard = ir::IfThenElse::make(guardCondition, ir::Continue::make());
      recoverySteps.push_back(guard);
    }

    Expr recoveredValue = provGraph.recoverVariable(varToRecover, definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, iterators);
    taco_iassert(indexVarToExprMap.count(varToRecover));
    recoverySteps.push_back(VarDecl::make(indexVarToExprMap[varToRecover], recoveredValue));

    // After we've recovered this index variable, some iterators are now
    // accessible for use when declaring locator access variables. So, generate
    // the accessors for those locator variables as part of the recovery process.
    // This is necessary after a fuse transformation, for example: If we fuse
    // two index variables (i, j) into f, then after we've generated the loop for
    // f, all locate accessors for i and j are now available for use.
    std::vector<Iterator> itersForVar;
    for (auto& iters : iterators.levelIterators()) {
      // Collect all level iterators that have locate and iterate over
      // the recovered index variable.
      if (iters.second.getIndexVar() == varToRecover && iters.second.hasLocate()) {
        itersForVar.push_back(iters.second);
      }
    }
    // Finally, declare all of the collected iterators' position access variables.
    recoverySteps.push_back(this->declLocatePosVars(itersForVar));

    // place underived guard
    std::vector<ir::Expr> iterBounds = provGraph.deriveIterBounds(varToRecover, definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, iterators);
    if (forallNeedsUnderivedGuards && underivedBounds.count(varToRecover) &&
        !provGraph.hasPosDescendant(varToRecover)) {

      // FIXME: [Olivia] Check this with someone
      // Removed underived guard if indexVar is bounded is divisible by its split child indexVar
      vector<IndexVar> children = provGraph.getChildren(varToRecover);
      bool hasDirectDivBound = false;
      std::vector<ir::Expr> iterBoundsInner = provGraph.deriveIterBounds(forall.getIndexVar(), definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, iterators);

        for (auto& c: children) {
          if (provGraph.hasExactBound(c) && provGraph.derivationPath(varToRecover, c).size() == 2) {
              std::vector<ir::Expr> iterBoundsUnderivedChild = provGraph.deriveIterBounds(c, definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, iterators);
              if (iterBoundsUnderivedChild[1].as<ir::Literal>()->getValue<int>() % iterBoundsInner[1].as<ir::Literal>()->getValue<int>() == 0)
              hasDirectDivBound = true;
              break;
          }
      }
      if (!hasDirectDivBound) {
          Stmt guard = IfThenElse::make(Gte::make(indexVarToExprMap[varToRecover], underivedBounds[varToRecover][1]),
                                        Continue::make());
          recoverySteps.push_back(guard);
      }
    }

    // TODO (rohany): Is there a way to pull this check into the loop guard?
    // If this index variable was divided into multiple equal chunks, then we
    // must add an extra guard to make sure that further scheduling operations
    // on descendent index variables exceed the bounds of each equal portion of
    // the loop. For a concrete example, consider a loop of size 10 that is divided
    // into two equal components -- 5 and 5. If the loop is then transformed
    // with .split(..., 3), each inner chunk of 5 will be split into chunks of
    // 3. Without an extra guard, the second chunk of 3 in the first group of 5
    // may attempt to perform an iteration for the second group of 5, which is
    // incorrect.
    // TODO (rohany): Also add a case here for when it's a DivideOntoPartition.
    if (this->provGraph.isDivided(varToRecover)) {
      // Collect the children iteration variables.
      auto children = this->provGraph.getChildren(varToRecover);
      auto outer = children[0];
      auto inner = children[1];
      // Find the iteration bounds of the inner variable -- that is the size
      // that the outer loop was broken into.
      auto bounds = this->provGraph.deriveIterBounds(inner, definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, iterators);
      // Use the difference between the bounds to find the size of the loop.
      auto dimLen = ir::Sub::make(bounds[1], bounds[0]);
      // For a variable f divided into into f1 and f2, the guard ensures that
      // for iteration f, f should be within f1 * dimLen and (f1 + 1) * dimLen.
      auto guard = ir::Gte::make(this->indexVarToExprMap[varToRecover], ir::Mul::make(ir::Add::make(this->indexVarToExprMap[outer], 1), dimLen));
      recoverySteps.push_back(IfThenElse::make(guard, ir::Continue::make()));
    }
  }
  Stmt recoveryStmt = Block::make(recoverySteps);

  taco_iassert(!definedIndexVars.count(forall.getIndexVar()));
  definedIndexVars.insert(forall.getIndexVar());
  definedIndexVarsOrdered.push_back(forall.getIndexVar());

  if (forall.getParallelUnit() != ParallelUnit::NotParallel && !distributedParallelUnit(forall.getParallelUnit())) {
    taco_iassert(!parallelUnitSizes.count(forall.getParallelUnit()));
    taco_iassert(!parallelUnitIndexVars.count(forall.getParallelUnit()));
    parallelUnitIndexVars[forall.getParallelUnit()] = forall.getIndexVar();
    vector<Expr> bounds = provGraph.deriveIterBounds(forall.getIndexVar(), definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, iterators);
    parallelUnitSizes[forall.getParallelUnit()] = ir::Sub::make(bounds[1], bounds[0]);
  }

  MergeLattice lattice = MergeLattice::make(forall, iterators, provGraph, definedIndexVars, whereTempsToResult);
  vector<Access> resultAccesses;
  set<Access> reducedAccesses;
  std::tie(resultAccesses, reducedAccesses) = getResultAccesses(forall);

  // Pre-allocate/initialize memory of value arrays that are full below this
  // loops index variable
  Stmt preInitValues = initResultArrays(forall.getIndexVar(), resultAccesses,
                                        getArgumentAccesses(forall),
                                        reducedAccesses);

  // Emit temporary initialization if forall is sequential and leads to a where statement
  vector<Stmt> temporaryValuesInitFree = {Stmt(), Stmt()};
  auto temp = temporaryInitialization.find(forall);
  if (temp != temporaryInitialization.end() && forall.getParallelUnit() == ParallelUnit::NotParallel && !isScalar(temp->second.getTemporary().getType()))
    temporaryValuesInitFree = codeToInitializeTemporary(temp->second);

  Stmt loops;
  // Emit a loop that iterates over over a single iterator (optimization)
  if (lattice.iterators().size() == 1 && lattice.iterators()[0].isUnique()) {
    taco_iassert(lattice.points().size() == 1);

    MergePoint point = lattice.points()[0];
    Iterator iterator = lattice.iterators()[0];

    vector<Iterator> locators = point.locators();
    vector<Iterator> appenders;
    vector<Iterator> inserters;
    tie(appenders, inserters) = splitAppenderAndInserters(point.results());

    std::vector<IndexVar> underivedAncestors = provGraph.getUnderivedAncestors(iterator.getIndexVar());
    IndexVar posDescendant;
    bool hasPosDescendant = false;
    if (!underivedAncestors.empty()) {
      hasPosDescendant = provGraph.getPosIteratorFullyDerivedDescendant(underivedAncestors[0], &posDescendant);
    }

    bool isWhereProducer = false;
    vector<Iterator> results = point.results();
    for (Iterator result : results) {
      for (auto it = tensorVars.begin(); it != tensorVars.end(); it++) {
        if (it->second == result.getTensor()) {
          if (whereTempsToResult.count(it->first)) {
            isWhereProducer = true;
            break;
          }
        }
      }
    }

    // For now, this only works when consuming a single workspace.
    //bool canAccelWithSparseIteration = inParallelLoopDepth == 0 && provGraph.isFullyDerived(iterator.getIndexVar()) &&
    //                                   iterator.isDimensionIterator() && locators.size() == 1;
    bool canAccelWithSparseIteration =
        provGraph.isFullyDerived(iterator.getIndexVar()) &&
        iterator.isDimensionIterator() && locators.size() == 1;
    if (canAccelWithSparseIteration) {
      bool indexListsExist = false;
      // We are iterating over a dimension and locating into a temporary with a tracker to keep indices. Instead, we
      // can just iterate over the indices and locate into the dense workspace.
      for (auto it = tensorVars.begin(); it != tensorVars.end(); ++it) {
        if (it->second == locators[0].getTensor() && util::contains(tempToIndexList, it->first)) {
          indexListsExist = true;
          break;
        }
      }
      canAccelWithSparseIteration &= indexListsExist;
    }

    if (!isWhereProducer && hasPosDescendant && underivedAncestors.size() > 1 && provGraph.isPosVariable(iterator.getIndexVar()) && posDescendant == forall.getIndexVar()) {
      loops = lowerForallFusedPosition(forall, iterator, locators,
                                         inserters, appenders, reducedAccesses, recoveryStmt);
    }
    else if (canAccelWithSparseIteration) {
      loops = lowerForallDenseAcceleration(forall, locators, inserters, appenders, reducedAccesses, recoveryStmt);
    }
    // Emit dimension coordinate iteration loop
    else if (iterator.isDimensionIterator()) {
      loops = lowerForallDimension(forall, point.locators(),
                                   inserters, appenders, reducedAccesses, recoveryStmt);
    }
    // Emit position iteration loop
    else if (iterator.hasPosIter()) {
      loops = lowerForallPosition(forall, iterator, locators,
                                    inserters, appenders, reducedAccesses, recoveryStmt);
    }
    // Emit coordinate iteration loop
    else {
      taco_iassert(iterator.hasCoordIter());
//      taco_not_supported_yet
      loops = Stmt();
    }
  }
  // Emit general loops to merge multiple iterators
  else {
    std::vector<IndexVar> underivedAncestors = provGraph.getUnderivedAncestors(forall.getIndexVar());
    taco_iassert(underivedAncestors.size() == 1); // TODO: add support for fused coordinate of pos loop
    loops = lowerMergeLattice(lattice, underivedAncestors[0],
                              forall.getStmt(), reducedAccesses);
  }
//  taco_iassert(loops.defined());

  if (!generateComputeCode() && !hasStores(loops)) {
    // If assembly loop does not modify output arrays, then it can be safely
    // omitted.
    loops = Stmt();
  }
  definedIndexVars.erase(forall.getIndexVar());
  definedIndexVarsOrdered.pop_back();
  if (forall.getParallelUnit() != ParallelUnit::NotParallel && !distributedParallelUnit(forall.getParallelUnit())) {
    inParallelLoopDepth--;
    taco_iassert(parallelUnitSizes.count(forall.getParallelUnit()));
    taco_iassert(parallelUnitIndexVars.count(forall.getParallelUnit()));
    parallelUnitIndexVars.erase(forall.getParallelUnit());
    parallelUnitSizes.erase(forall.getParallelUnit());
  }
  return Block::blanks(preInitValues,
                       temporaryValuesInitFree[0],
                       loops,
                       temporaryValuesInitFree[1]);
}

Stmt LowererImpl::lowerForallCloned(Forall forall) {
  // want to emit guards outside of loop to prevent unstructured loop exits

  // construct guard
  // underived or pos variables that have a descendant that has not been defined yet
  vector<IndexVar> varsWithGuard;
  for (auto var : provGraph.getAllIndexVars()) {
    if (provGraph.isRecoverable(var, definedIndexVars)) {
      continue; // already recovered
    }
    if (provGraph.isUnderived(var) && !provGraph.hasPosDescendant(var)) { // if there is pos descendant then will be guarded already
      varsWithGuard.push_back(var);
    }
    else if (provGraph.isPosVariable(var)) {
      // if parent is coord then this is variable that will be guarded when indexing into coord array
      if(provGraph.getParents(var).size() == 1 && provGraph.isCoordVariable(provGraph.getParents(var)[0])) {
        varsWithGuard.push_back(var);
      }
    }
  }

  // determine min and max values for vars given already defined variables.
  // we do a recovery where we fill in undefined variables with either 0's or the max of their iteration
  std::map<IndexVar, Expr> minVarValues;
  std::map<IndexVar, Expr> maxVarValues;
  set<IndexVar> definedForGuard = definedIndexVars;
  vector<Stmt> guardRecoverySteps;
  Expr maxOffset = 0;
  bool setMaxOffset = false;

  for (auto var : varsWithGuard) {
    std::vector<IndexVar> currentDefinedVarOrder = definedIndexVarsOrdered; // TODO: get defined vars at time of this recovery

    std::map<IndexVar, Expr> minChildValues = indexVarToExprMap;
    std::map<IndexVar, Expr> maxChildValues = indexVarToExprMap;

    for (auto child : provGraph.getFullyDerivedDescendants(var)) {
      if (!definedIndexVars.count(child)) {
        std::vector<ir::Expr> childBounds = provGraph.deriveIterBounds(child, currentDefinedVarOrder, underivedBounds, indexVarToExprMap, iterators);

        minChildValues[child] = childBounds[0];
        maxChildValues[child] = childBounds[1];

        // recover new parents
        for (const IndexVar& varToRecover : provGraph.newlyRecoverableParents(child, definedForGuard)) {
          Expr recoveredValue = provGraph.recoverVariable(varToRecover, definedIndexVarsOrdered, underivedBounds,
                                                          minChildValues, iterators);
          Expr maxRecoveredValue = provGraph.recoverVariable(varToRecover, definedIndexVarsOrdered, underivedBounds,
                                                             maxChildValues, iterators);
          if (!setMaxOffset) { // TODO: work on simplifying this
            maxOffset = ir::Add::make(maxOffset, ir::Sub::make(maxRecoveredValue, recoveredValue));
            setMaxOffset = true;
          }
          taco_iassert(indexVarToExprMap.count(varToRecover));

          guardRecoverySteps.push_back(VarDecl::make(indexVarToExprMap[varToRecover], recoveredValue));
          definedForGuard.insert(varToRecover);
        }
        definedForGuard.insert(child);
      }
    }

    minVarValues[var] = provGraph.recoverVariable(var, currentDefinedVarOrder, underivedBounds, minChildValues, iterators);
    maxVarValues[var] = provGraph.recoverVariable(var, currentDefinedVarOrder, underivedBounds, maxChildValues, iterators);
  }

  // Build guards
  Expr guardCondition;
  for (auto var : varsWithGuard) {
    std::vector<ir::Expr> iterBounds = provGraph.deriveIterBounds(var, definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, iterators);

    Expr minGuard = Lt::make(minVarValues[var], iterBounds[0]);
    Expr maxGuard = Gte::make(ir::Add::make(maxVarValues[var], ir::simplify(maxOffset)), iterBounds[1]);
    Expr guardConditionCurrent = Or::make(minGuard, maxGuard);

    if (isa<ir::Literal>(ir::simplify(iterBounds[0])) && ir::simplify(iterBounds[0]).as<ir::Literal>()->equalsScalar(0)) {
      guardConditionCurrent = maxGuard;
    }

    if (guardCondition.defined()) {
      guardCondition = Or::make(guardConditionCurrent, guardCondition);
    }
    else {
      guardCondition = guardConditionCurrent;
    }
  }

  Stmt unvectorizedLoop;

  taco_uassert(guardCondition.defined())
    << "Unable to vectorize or unroll loop over unbound variable " << forall.getIndexVar();

  // build loop with guards (not vectorized)
  if (!varsWithGuard.empty()) {
    ignoreVectorize = true;
    unvectorizedLoop = lowerForall(forall);
    ignoreVectorize = false;
  }

  // build loop without guards
  emitUnderivedGuards = false;
  Stmt vectorizedLoop = lowerForall(forall);
  emitUnderivedGuards = true;

  // return guarded loops
  return Block::make(Block::make(guardRecoverySteps), IfThenElse::make(guardCondition, unvectorizedLoop, vectorizedLoop));
}

Stmt LowererImpl::searchForFusedPositionStart(Forall forall, Iterator posIterator) {
  vector<Stmt> searchForUnderivedStart;
  vector<IndexVar> underivedAncestors = provGraph.getUnderivedAncestors(forall.getIndexVar());
  ir::Expr last_block_start_temporary;
  for (int i = (int) underivedAncestors.size() - 2; i >= 0; i--) {
    Iterator posIteratorLevel = posIterator;
    for (int j = (int) underivedAncestors.size() - 2; j > i; j--) { // take parent of iterator enough times to get correct level
      posIteratorLevel = posIteratorLevel.getParent();
    }

    // want to get size of pos array not of crd_array
    ir::Expr parentSize = 1; // to find size of segment walk down sizes of iterator chain
    Iterator rootIterator = posIterator;
    while (!rootIterator.isRoot()) {
      rootIterator = rootIterator.getParent();
    }
    while (rootIterator.getChild() != posIteratorLevel) {
      rootIterator = rootIterator.getChild();
      if (rootIterator.hasAppend()) {
        parentSize = rootIterator.getSize(parentSize);
      } else if (rootIterator.hasInsert()) {
        parentSize = ir::Mul::make(parentSize, rootIterator.getWidth());
      }
    }

    // emit bounds search on cpu just bounds, on gpu search in blocks
    if (parallelUnitIndexVars.count(ParallelUnit::GPUBlock)) {
      Expr values_per_block;
      {
        // we do a recovery where we fill in undefined variables with 0's to get start target (just like for vector guards)
        std::map<IndexVar, Expr> zeroedChildValues = indexVarToExprMap;
        zeroedChildValues[parallelUnitIndexVars[ParallelUnit::GPUBlock]] = 1;
        set<IndexVar> zeroDefinedIndexVars = {parallelUnitIndexVars[ParallelUnit::GPUBlock]};
        for (IndexVar child : provGraph.getFullyDerivedDescendants(posIterator.getIndexVar())) {
          if (child != parallelUnitIndexVars[ParallelUnit::GPUBlock]) {
            zeroedChildValues[child] = 0;

            // recover new parents
            for (const IndexVar &varToRecover : provGraph.newlyRecoverableParents(child, zeroDefinedIndexVars)) {
              Expr recoveredValue = provGraph.recoverVariable(varToRecover, definedIndexVarsOrdered, underivedBounds,
                                                              zeroedChildValues, iterators);
              taco_iassert(indexVarToExprMap.count(varToRecover));
              zeroedChildValues[varToRecover] = recoveredValue;
              zeroDefinedIndexVars.insert(varToRecover);
              if (varToRecover == posIterator.getIndexVar()) {
                break;
              }
            }
            zeroDefinedIndexVars.insert(child);
          }
        }
        values_per_block = zeroedChildValues[posIterator.getIndexVar()];
      }

      IndexVar underived = underivedAncestors[i];
      ir::Expr blockStarts_temporary = ir::Var::make(underived.getName() + "_blockStarts",
                                                     getCoordinateVar(underived).type(), true, false);
      header.push_back(ir::VarDecl::make(blockStarts_temporary, 0));
      header.push_back(
              Allocate::make(blockStarts_temporary, ir::Add::make(parallelUnitSizes[ParallelUnit::GPUBlock], 1)));
      footer.push_back(Free::make(blockStarts_temporary));


      Expr blockSize;
      if (parallelUnitSizes.count(ParallelUnit::GPUThread)) {
        blockSize = parallelUnitSizes[ParallelUnit::GPUThread];
        if (parallelUnitSizes.count(ParallelUnit::GPUWarp)) {
          blockSize = ir::Mul::make(blockSize, parallelUnitSizes[ParallelUnit::GPUWarp]);
        }
      } else {
        std::vector<IndexVar> definedIndexVarsMatched = definedIndexVarsOrdered;
        // find sub forall that tells us block size
        match(forall.getStmt(),
              function<void(const ForallNode *, Matcher *)>([&](
                      const ForallNode *n, Matcher *m) {
                if (n->parallel_unit == ParallelUnit::GPUThread) {
                  vector<Expr> bounds = provGraph.deriveIterBounds(forall.getIndexVar(), definedIndexVarsMatched,
                                                                   underivedBounds, indexVarToExprMap, iterators);
                  blockSize = ir::Sub::make(bounds[1], bounds[0]);
                }
                definedIndexVarsMatched.push_back(n->indexVar);
              })
        );
      }
      taco_iassert(blockSize.defined());

      if (i == (int) underivedAncestors.size() - 2) {
        std::vector<Expr> args = {
                posIteratorLevel.getMode().getModePack().getArray(0), // array
                blockStarts_temporary, // results
                ir::Literal::zero(posIteratorLevel.getBeginVar().type()), // arrayStart
                parentSize, // arrayEnd
                values_per_block, // values_per_block
                blockSize, // block_size
                parallelUnitSizes[ParallelUnit::GPUBlock] // num_blocks
        };
        header.push_back(ir::Assign::make(blockStarts_temporary,
                                          ir::Call::make("taco_binarySearchBeforeBlockLaunch", args,
                                                         getCoordinateVar(underived).type())));
      }
      else {
        std::vector<Expr> args = {
                posIteratorLevel.getMode().getModePack().getArray(0), // array
                blockStarts_temporary, // results
                ir::Literal::zero(posIteratorLevel.getBeginVar().type()), // arrayStart
                parentSize, // arrayEnd
                last_block_start_temporary, // targets
                blockSize, // block_size
                parallelUnitSizes[ParallelUnit::GPUBlock] // num_blocks
        };
        header.push_back(ir::Assign::make(blockStarts_temporary,
                                          ir::Call::make("taco_binarySearchIndirectBeforeBlockLaunch", args,
                                                         getCoordinateVar(underived).type())));
      }
      searchForUnderivedStart.push_back(VarDecl::make(posIteratorLevel.getBeginVar(),
                                                      ir::Load::make(blockStarts_temporary,
                                                                     indexVarToExprMap[parallelUnitIndexVars[ParallelUnit::GPUBlock]])));
      searchForUnderivedStart.push_back(VarDecl::make(posIteratorLevel.getEndVar(),
                                                      ir::Load::make(blockStarts_temporary, ir::Add::make(
                                                              indexVarToExprMap[parallelUnitIndexVars[ParallelUnit::GPUBlock]],
                                                              1))));
      last_block_start_temporary = blockStarts_temporary;
    } else {
      header.push_back(VarDecl::make(posIteratorLevel.getBeginVar(), ir::Literal::zero(posIteratorLevel.getBeginVar().type())));
      header.push_back(VarDecl::make(posIteratorLevel.getEndVar(), parentSize));
    }

    // we do a recovery where we fill in undefined variables with 0's to get start target (just like for vector guards)
    Expr underivedStartTarget;
    if (i == (int) underivedAncestors.size() - 2) {
      std::map<IndexVar, Expr> minChildValues = indexVarToExprMap;
      set<IndexVar> minDefinedIndexVars = definedIndexVars;
      minDefinedIndexVars.erase(forall.getIndexVar());

      for (IndexVar child : provGraph.getFullyDerivedDescendants(posIterator.getIndexVar())) {
        if (!minDefinedIndexVars.count(child)) {
          std::vector<ir::Expr> childBounds = provGraph.deriveIterBounds(child, definedIndexVarsOrdered,
                                                                         underivedBounds,
                                                                         indexVarToExprMap, iterators);
          minChildValues[child] = childBounds[0];

          // recover new parents
          for (const IndexVar &varToRecover : provGraph.newlyRecoverableParents(child, minDefinedIndexVars)) {
            Expr recoveredValue = provGraph.recoverVariable(varToRecover, definedIndexVarsOrdered, underivedBounds,
                                                            minChildValues, iterators);
            taco_iassert(indexVarToExprMap.count(varToRecover));
            searchForUnderivedStart.push_back(VarDecl::make(indexVarToExprMap[varToRecover], recoveredValue));
            minDefinedIndexVars.insert(varToRecover);
            if (varToRecover == posIterator.getIndexVar()) {
              break;
            }
          }
          minDefinedIndexVars.insert(child);
        }
      }
      underivedStartTarget = indexVarToExprMap[posIterator.getIndexVar()];
    }
    else {
      underivedStartTarget = this->iterators.modeIterator(underivedAncestors[i+1]).getPosVar();
    }

    vector<Expr> binarySearchArgs = {
            posIteratorLevel.getMode().getModePack().getArray(0), // array
            posIteratorLevel.getBeginVar(), // arrayStart
            posIteratorLevel.getEndVar(), // arrayEnd
            underivedStartTarget // target
    };
    Expr posVarUnknown = this->iterators.modeIterator(underivedAncestors[i]).getPosVar();
    searchForUnderivedStart.push_back(ir::VarDecl::make(posVarUnknown,
                                                        ir::Call::make("taco_binarySearchBefore", binarySearchArgs,
                                                                       getCoordinateVar(underivedAncestors[i]).type())));
    Stmt locateCoordVar;
    if (posIteratorLevel.getParent().hasPosIter()) {
      locateCoordVar = ir::VarDecl::make(indexVarToExprMap[underivedAncestors[i]], ir::Load::make(posIteratorLevel.getParent().getMode().getModePack().getArray(1), posVarUnknown));
    }
    else {
      locateCoordVar = ir::VarDecl::make(indexVarToExprMap[underivedAncestors[i]], posVarUnknown);
    }
    searchForUnderivedStart.push_back(locateCoordVar);
  }
  return ir::Block::make(searchForUnderivedStart);
}

// TODO (rohany): Replace this static incrementing ID with a pass during code
//  generation that collects all sharding functors and uniquely numbers them.
static int shardingFunctorID = 0;
Stmt LowererImpl::lowerForallDimension(Forall forall,
                                       vector<Iterator> locators,
                                       vector<Iterator> inserters,
                                       vector<Iterator> appenders,
                                       set<Access> reducedAccesses,
                                       ir::Stmt recoveryStmt)
{
  Expr coordinate = getCoordinateVar(forall.getIndexVar());

  if (forall.getParallelUnit() != ParallelUnit::NotParallel && forall.getOutputRaceStrategy() == OutputRaceStrategy::Atomics) {
    markAssignsAtomicDepth++;
    atomicParallelUnit = forall.getParallelUnit();
  }

  // TODO (rohany): Need some sort of stack mechanism to pop off the computing on
  //  var once (if) we support nested distributions.
  if (forall.getComputingOn().defined()) {
    this->computingOnTensorVar = forall.getComputingOn();
  }
  if (forall.getOutputRaceStrategy() == OutputRaceStrategy::ParallelReduction && forall.isDistributed()) {
    this->performingLegionReduction = true;
  }

  auto prevDistVar = this->curDistVar;

  if (forall.isDistributed()) {
    this->curDistVar = forall.getIndexVar();
    this->distLoopDepth++;
  }


  Stmt body = lowerForallBody(coordinate, forall.getStmt(),
                              locators, inserters, appenders, reducedAccesses);

  if (forall.isDistributed()) {
    this->curDistVar = forall.getIndexVar();
    this->distLoopDepth--;
  }

  // As a simple hack, don't emit code that actually performs the iteration within a placement node.
  // We just care about emitting the actual distributed loop to do the data placement, not waste
  // time iterating over the data within it. Placement can be nested though, so only exclude the
  // inner body for the deepest placement level.
  if (forall.isDistributed() && this->isPlacementCode && size_t(this->distLoopDepth + 1) == this->placements.size()) {
    body = ir::Block::make({});
  }

  if (forall.getParallelUnit() != ParallelUnit::NotParallel && forall.getOutputRaceStrategy() == OutputRaceStrategy::Atomics) {
    markAssignsAtomicDepth--;
  }

  // Emit loop with preamble and postamble
  std::vector<ir::Expr> bounds = provGraph.deriveIterBounds(forall.getIndexVar(), definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, iterators);

  Stmt declarePartitionBounds;
  auto isTask = forall.isDistributed() || (forall.getTransfers().size() > 0);
  auto taskID = -1;
  std::vector<ir::Stmt> transfers;
  if (isTask) {
    taskID = this->taskCounter;
    this->taskCounter++;
    // TODO (rohany): For now, we have only single dimension domains. We will get
    //  this from the access. Probably have to define each of these for each transfer,
    //  Since different transfers could have different dimensions.
    std::vector<IndexVar> distIvars = {forall.getIndexVar()};
    auto dim = 1;

    // TODO (rohany): Comment.
    {
      auto fusedVars = this->provGraph.getMultiFusedParents(forall.getIndexVar());
      if (fusedVars.size() > 0) {
        dim = fusedVars.size();
        distIvars = fusedVars;
      }
    }

    auto dimT = Domain(dim);
    auto pointInDimT = PointInDomainIterator(dim);
    auto pointT = Point(dim);
    auto rectT = Rect(dim);
    auto indexSpaceT = IndexSpaceT(dim);
    auto disjointPart = ir::Symbol::make("LEGION_DISJOINT_KIND");
    auto aliasedPart = ir::Symbol::make("LEGION_COMPUTE_KIND");
    auto readOnly = ir::Symbol::make("READ_ONLY");
    auto readWrite = ir::Symbol::make("READ_WRITE");
    // TODO (rohany): Assuming that all tensors have the same type right now.
    auto reduce = ir::Symbol::make(LegionRedopString(this->tensorVars.begin()->first.getType().getDataType()));
    auto exclusive = ir::Symbol::make("EXCLUSIVE");
    auto simultaneous = ir::Symbol::make("LEGION_SIMULTANEOUS");
    auto fidVal = ir::Symbol::make("FID_VAL");
    auto ctx = ir::Symbol::make("ctx");
    auto runtime = ir::Symbol::make("runtime");
    auto virtualMap = ir::Symbol::make("Mapping::DefaultMapper::VIRTUAL_MAP");
    auto placementMap = ir::Symbol::make("TACOMapper::PLACEMENT");
    auto placementShard = ir::Symbol::make("TACOMapper::PLACEMENT_SHARD");
    auto sameAddressSpace = ir::Symbol::make("Mapping::DefaultMapper::SAME_ADDRESS_SPACE");

    // We need to emit accessing the partition for any child task that uses the partition.
    // TODO (rohany): A hack that doesn't scale to nested distributions.
    if (forall.getComputingOn().defined()) {
      // Add a declaration of all the needed partition bounds variables.
      auto tensorIspace = ir::GetProperty::make(this->tensorVars[this->computingOnTensorVar], TensorProperty::IndexSpace);
      auto bounds = ir::Call::make("runtime->get_index_space_domain", {ctx, tensorIspace}, Auto);
      auto boundsVar = ir::Var::make(forall.getComputingOn().getName() + "PartitionBounds", Auto);
      std::vector<ir::Stmt> declareBlock;
      declareBlock.push_back(ir::VarDecl::make(boundsVar, bounds));
      for (auto tvItr : this->provGraph.getPartitionBounds()) {
        for (auto idxItr : tvItr.second) {
          auto lo = ir::Load::make(ir::MethodCall::make(boundsVar, "lo", {}, false, Int64), idxItr.first);
          auto hi = ir::Load::make(ir::MethodCall::make(boundsVar, "hi", {}, false, Int64), idxItr.first);
          declareBlock.push_back(ir::VarDecl::make(idxItr.second.first, lo));
          declareBlock.push_back(ir::VarDecl::make(idxItr.second.second, hi));
        }
      }
      declarePartitionBounds = ir::Block::make(declareBlock);
    }

    auto domain = ir::Var::make("domain", dimT);
    if (forall.getComputingOn().defined()) {
      // If we're computing on a tensor, then use the domain of the partition as the
      // launch domain for the task launch.
      // TODO (rohany): Might need a wrapper method call on computingOnVar.
      auto getDomain = ir::Call::make("runtime->get_index_partition_color_space", {ctx, ir::Call::make("get_index_partition", {this->computingOnPartition}, Auto)}, Auto);
      transfers.push_back(ir::VarDecl::make(domain, getDomain));
    } else {
      auto varIspace = ir::Var::make(forall.getIndexVar().getName() + "IndexSpace", Auto);
      auto lowerBound = ir::Var::make("lowerBound", pointT);
      auto upperBound = ir::Var::make("upperBound", pointT);
      std::vector<ir::Expr> lowerBoundExprs;
      std::vector<ir::Expr> upperBoundExprs;
      for (auto it : distIvars) {
        // If the bounds of an index variable have been overridden for placement code
        // use those bounds instead of the ones derived from the Provenance Graph.
        if (this->isPlacementCode && util::contains(this->indexVarFaces, it)) {
          auto face = this->indexVarFaces[it];
          lowerBoundExprs.push_back(face);
          upperBoundExprs.push_back(face);
        } else {
          auto bounds = provGraph.deriveIterBounds(it, definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, iterators);
          lowerBoundExprs.push_back(bounds[0]);
          upperBoundExprs.push_back(ir::Sub::make(bounds[1], 1));
        }
      }
      transfers.push_back(ir::VarDecl::make(lowerBound, makeConstructor(pointT, lowerBoundExprs)));
      transfers.push_back(ir::VarDecl::make(upperBound, makeConstructor(pointT, upperBoundExprs)));

      auto makeIspace = ir::Call::make(
          "runtime->create_index_space",
          {ctx, makeConstructor(rectT, {lowerBound, upperBound})},
          Auto
      );
      transfers.push_back(ir::VarDecl::make(varIspace, makeIspace));
      auto makeDomain = ir::Call::make(
          "runtime->get_index_space_domain",
          {ctx, makeConstructor(indexSpaceT, {varIspace})},
          dimT
      );
      transfers.push_back(ir::VarDecl::make(domain, makeDomain));
    }

    // Make a coloring for each transfer.
    std::vector<Expr> colorings;
    for (auto& t : forall.getTransfers()) {
      auto c = ir::Var::make(t.getAccess().getTensorVar().getName() + "Coloring", DomainPointColoring);
      transfers.push_back(ir::VarDecl::make(c, ir::Call::make(DomainPointColoring.getName(), {}, DomainPointColoring)));
      colorings.push_back(c);
    }

    auto domainIter = ir::Var::make("itr", pointInDimT);

    std::vector<Stmt> partStmts;
    for (size_t i = 0; i < distIvars.size(); i++) {
      auto ivar = distIvars[i];
      auto ivarExpr = this->indexVarToExprMap[ivar];
      partStmts.push_back(ir::VarDecl::make(ivarExpr, ir::Load::make(ir::Deref::make(domainIter, pointT), int32_t(i))));
    }

    // If operating on a partition, we need to get the bounds of the partition at each index point.
    if (forall.getComputingOn().defined()) {
      auto point = ir::Var::make("domPoint", Datatype("DomainPoint"));
      partStmts.push_back(ir::VarDecl::make(point, ir::Deref::make(domainIter, Auto)));
      auto partVar = ir::Var::make(forall.getComputingOn().getName() + "PartitionBounds", Auto);
      auto subreg = ir::Call::make("runtime->get_logical_subregion_by_color", {ctx, this->computingOnPartition, point}, Auto);
      auto subregispace = ir::MethodCall::make(subreg, "get_index_space", {}, false, Auto);
      auto bounds = ir::Call::make("runtime->get_index_space_domain", {subregispace}, Auto);
      partStmts.push_back(ir::VarDecl::make(partVar, bounds));
      // Declare all of the bounds variables here.
      for (auto tvItr : this->provGraph.getPartitionBounds()) {
        for (auto idxItr : tvItr.second) {
          auto lo = ir::Load::make(ir::MethodCall::make(partVar, "lo", {}, false, Int64), idxItr.first);
          auto hi = ir::Load::make(ir::MethodCall::make(partVar, "hi", {}, false, Int64), idxItr.first);
          partStmts.push_back(ir::VarDecl::make(idxItr.second.first, lo));
          partStmts.push_back(ir::VarDecl::make(idxItr.second.second, hi));
        }
      }
    }

    // Add a dummy partition object for each transfer.
    for (size_t idx = 0; idx < forall.getTransfers().size(); idx++) {
      auto& t = forall.getTransfers()[idx];
      auto n = t.getAccess().getTensorVar().getName();

      auto tensorDim = t.getAccess().getIndexVars().size();
      auto txPoint = Point(tensorDim);
      auto txRect = Rect(tensorDim);

      auto bounds = this->derivedBounds[this->curDistVar][t.getAccess().getTensorVar()];
      std::vector<Expr> los, his;
      for (size_t dimIdx = 0; dimIdx < tensorDim; dimIdx++) {
        los.push_back(bounds[dimIdx][0]);
        auto dimBound = ir::GetProperty::make(this->tensorVars[t.getAccess().getTensorVar()], TensorProperty::Dimension, dimIdx);
        auto upper = ir::Min::make(bounds[dimIdx][1], ir::Sub::make(dimBound, 1));
        his.push_back(upper);
      }
      auto start = ir::Var::make(n + "Start", txPoint);
      auto end = ir::Var::make(n + "End", txPoint);
      partStmts.push_back(ir::VarDecl::make(start, makeConstructor(txPoint, los)));
      partStmts.push_back(ir::VarDecl::make(end, makeConstructor(txPoint, his)));
      auto rect = ir::Var::make(n + "Rect", txRect);
      partStmts.push_back(ir::VarDecl::make(rect, makeConstructor(txRect, {start, end})));

      // It's possible that this partitioning makes a rectangle that goes out of bounds
      // of the tensor's index space. If so, replace the rectangle with an empty Rect.
      auto domain = ir::Var::make(n + "Domain", Auto);
      auto ispace = ir::GetProperty::make(this->tensorVars[t.getAccess().getTensorVar()], TensorProperty::IndexSpace);
      partStmts.push_back(ir::VarDecl::make(domain, ir::Call::make("runtime->get_index_space_domain", {ctx, ispace}, Auto)));
      auto lb = ir::MethodCall::make(domain, "contains", {ir::FieldAccess::make(rect, "lo", false, Auto)}, false, Bool);
      auto hb = ir::MethodCall::make(domain, "contains", {ir::FieldAccess::make(rect, "hi", false, Auto)}, false, Bool);
      auto guard = ir::Or::make(ir::Neg::make(lb), ir::Neg::make(hb));
      partStmts.push_back(ir::IfThenElse::make(guard, ir::Assign::make(rect, ir::MethodCall::make(rect, "make_empty", {}, false, Auto))));

      auto coloring = colorings[idx];
      partStmts.push_back(ir::Assign::make(ir::Load::make(coloring, ir::Deref::make(domainIter, Auto)), rect));
    }

    auto l = ir::For::make(
          domainIter,
          ir::Call::make(pointInDimT.getName(), {domain}, pointInDimT),
          ir::MethodCall::make(domainIter, "valid", {}, false /* deref */, Datatype::Bool),
          1 /* increment -- hack to get ++ */,
          ir::Block::make(partStmts)
    );
    transfers.push_back(l);

    // If we're doing a reduction, we're most likely not operating on a disjoint
    // partition. So, fall back to an aliased partition.
    auto partKind = disjointPart;
    // TODO (rohany): This is definitely not as accurate as we can be.
    if (this->performingLegionReduction) {
      partKind = aliasedPart;
    }

    std::map<TensorVar, Expr> partitionings;
    for (size_t idx = 0; idx < forall.getTransfers().size(); idx++) {
      auto& t = forall.getTransfers()[idx];
      auto& tv = t.getAccess().getTensorVar();
      auto coloring = colorings[idx];
      auto part = ir::Var::make(tv.getName() + "Partition", Auto);
      partitionings[tv] = part;
      auto partcall = ir::Call::make(
          "runtime->create_index_partition",
          {ctx, ir::GetProperty::make(this->tensorVars[tv], TensorProperty::IndexSpace), domain, coloring, partKind},
          Auto
      );
      transfers.push_back(ir::VarDecl::make(part, partcall));
    }

    auto getPriv = [&](const TensorVar& tv) {
      if (util::contains(this->resultTensors, tv)) {
        // If we're already reducing, we can't go up the lattice to read_write
        // so stay at reduction.
        if (forall.getOutputRaceStrategy() == OutputRaceStrategy::ParallelReduction || this->performingLegionReduction) {
          return std::make_pair(reduce, simultaneous);
        }
        return std::make_pair(readWrite, exclusive);
      }
      return std::make_pair(readOnly, exclusive);
    };

    auto getLogicalRegion = [](Expr e) {
      return ir::Call::make("get_logical_region", {e}, Auto);
    };

    // AccessFinder finds is the task being lowered accesses the target tensor.
    struct AccessFinder : public IRVisitor {
      void visit(const GetProperty* prop) {
        if (prop->tensor == this->targetVar) {
          switch (prop->property) {
            case ir::TensorProperty::ValuesReductionAccessor:
            case ir::TensorProperty::ValuesWriteAccessor:
            case ir::TensorProperty::ValuesReadAccessor:
              this->readsVar = true;
              break;
            default:
              return;
          }
        }
      }
      void visit(const For* node) {
        if (node->isTask) { return; }
        node->contents.accept(this);
        // TODO (rohany): When considering sparse tensors, we will need to recurse into
        //  the other parts of the for loop.
      }

      ir::Expr targetVar;
      bool readsVar = false;
    };

    if (forall.isDistributed()) {
      // In a distributed for-all, we have to make an index launch.
      std::vector<Stmt> itlStmts;
      std::vector<Expr> regionReqs;
      std::vector<Expr> regionReqArgs;
      for (auto& it : this->tensorVarOrdering) {
        auto tv = it;
        auto tvIR = this->tensorVars[tv];
        auto priv = getPriv(tv);
        // If the tensor is being transferred at this level, then use the
        // corresponding partition. Otherwise, use the tensorvar itself.
        if (util::contains(partitionings, tv)) {
          auto part = ir::Var::make(tv.getName() + "LogicalPartition", LogicalPartition);
          auto call = ir::Call::make("runtime->get_logical_partition", {ctx, getLogicalRegion(tvIR), partitionings.at(tv)}, LogicalPartition);
          itlStmts.push_back(ir::VarDecl::make(part, call));
          regionReqArgs = {
              part,
              0,
              priv.first,
              priv.second,
              getLogicalRegion(tvIR),
          };
        } else if (forall.getComputingOn().defined() && forall.getComputingOn() == tv) {
          regionReqArgs = {
              this->computingOnPartition,
              0,
              priv.first,
              priv.second,
              getLogicalRegion(tvIR),
          };
        } else {
          regionReqArgs = {
              getLogicalRegion(tvIR),
              priv.first,
              priv.second,
              getLogicalRegion(tvIR),
          };
        }
        auto regReq = ir::Var::make(tv.getName() + "Req", RegionRequirement);
        auto makeReq = ir::Call::make(
            RegionRequirement.getName(),
            regionReqArgs,
            RegionRequirement
        );
        itlStmts.push_back(ir::VarDecl::make(regReq, makeReq));
        itlStmts.push_back(ir::SideEffect::make(ir::MethodCall::make(regReq, "add_field", {fidVal}, false, Auto)));

        // If the task being launched doesn't access the target region, then we can
        // virtually map the region. Or, for placement code, we don't want to virtually
        // map a region that the leaf placement tasks use.
        AccessFinder finder; finder.targetVar = tvIR;
        body.accept(&finder);
        if (!finder.readsVar && !(this->isPlacementCode && size_t(this->distLoopDepth + 1) == this->placements.size())) {
          itlStmts.push_back(ir::Assign::make(ir::FieldAccess::make(regReq, "tag", false, Auto), virtualMap));
        }

        regionReqs.push_back(regReq);
      }

      // These args have to be for each of the subtasks.
      auto args = ir::Var::make("taskArgs", Auto);
      bool unpackFaceArgs = false;
      // We only generate code for control replicated placement if the distribution
      // is done at the top level.
      auto useCtrlRep = this->distLoopDepth == 0;
      if (this->isPlacementCode) {
        auto placementGrid = this->placements[this->distLoopDepth].first;
        auto placement = this->placements[this->distLoopDepth].second;

        // Count the number of Face() axes placements.
        int count = 0;
        for (auto axis : placement.axes) {
          if (axis.kind == GridPlacement::AxisMatch::Face) {
            count++;
          }
        }
        if (count > 0) {
          std::vector<Expr> prefixVars, prefixExprs;
          if (useCtrlRep) {
            // If we are using control replication, we'll need to do some extra
            // work to set up a sharding functor so that index tasks are sharded to
            // the right positions. To do so, we'll need to add a sharding functor ID
            // to the argument pack. Next, we need to register the sharding functor
            // to the runtime system, rather than letting the mapper handle it.
            int sfID = shardingFunctorID++;
            prefixVars.push_back(ir::Var::make("sfID", Int32));
            prefixExprs.push_back(ir::Call::make("shardingID", {sfID}, Int32));

            // Create the vector of dimensions.
            auto vecty = Datatype("std::vector<int>");
            auto dimVec = ir::Var::make("dims", vecty);
            itlStmts.push_back(ir::VarDecl::make(dimVec, ir::makeConstructor(vecty, {})));
            for (int i = 0; i < placementGrid.getDim(); i++) {
              itlStmts.push_back(ir::SideEffect::make(
                  ir::MethodCall::make(dimVec, "push_back", {placementGrid.getDimSize(i)}, false /* deref */, Auto)));
            }
            itlStmts.push_back(
              ir::SideEffect::make(
                ir::Call::make(
                  "registerPlacementShardingFunctor",
                  {ctx, runtime, ir::Call::make("shardingID", {sfID}, Int32), dimVec},
                  Auto
                )
              )
            );
          } else {
            // If we are directed to place a tensor onto a Face of the placement
            // grid, then we need to package up the full dimensions of the placement
            // grid into the task's arguments so that the mapper can extract it.
            for (int i = 0; i < placementGrid.getDim(); i++) {
              std::stringstream varname;
              varname << "dim" << i;
              auto var = ir::Var::make(varname.str(), Int32);
              prefixVars.push_back(var); prefixExprs.push_back(placementGrid.getDimSize(i));
            }
          }
          itlStmts.push_back(ir::PackTaskArgs::make(args, taskID, prefixVars, prefixExprs));
          unpackFaceArgs = true;
        } else {
          itlStmts.push_back(ir::PackTaskArgs::make(args, taskID, {}, {}));
        }
      } else {
        itlStmts.push_back(ir::PackTaskArgs::make(args, taskID, {}, {}));
      }

      auto launcher = ir::Var::make("launcher", IndexLauncher);
      auto launcherMake = ir::Call::make(
          IndexLauncher.getName(),
          {
              ir::Call::make("taskID", {taskID}, Datatype::Int32),
              domain,
              args,
              ir::Call::make(ArgumentMap.getName(), {}, ArgumentMap),
          },
          IndexLauncher
      );
      itlStmts.push_back(ir::VarDecl::make(launcher, launcherMake));
      for (auto& req : regionReqs) {
        auto mcall = ir::MethodCall::make(launcher, "add_region_requirement", {req}, false /* deref */, Auto);
        itlStmts.push_back(ir::SideEffect::make(mcall));
      }
      if (unpackFaceArgs) {
        auto tag = placementMap;
        if (useCtrlRep) {
          tag = placementShard;
        }
        auto addTag = ir::Assign::make(ir::FieldAccess::make(launcher, "tag", false, Auto), tag);
        itlStmts.push_back(addTag);
      }
      // If this is a nested distribution, keep it on the same node.
      if (this->distLoopDepth > 0) {
        auto tag = ir::FieldAccess::make(launcher, "tag", false, Auto);
        auto addTag = ir::Assign::make(tag, ir::BitOr::make(tag, sameAddressSpace));
        itlStmts.push_back(addTag);
      }

      auto fm = ir::Var::make("fm", Auto);
      auto fmCall = ir::Call::make(
          "runtime->execute_index_space",
          {ctx, launcher},
          Auto
      );
      if (this->distLoopDepth == 0) {
        itlStmts.push_back(ir::VarDecl::make(fm, fmCall));
        itlStmts.push_back(ir::SideEffect::make(ir::MethodCall::make(fm, "wait_all_results", {}, false, Auto)));
      } else {
        itlStmts.push_back(ir::SideEffect::make(fmCall));
      }

      // Placement code should return the LogicalPartition for the top level partition.
      if (this->isPlacementCode && this->distLoopDepth == 0) {
        auto tv = this->tensorVars.begin()->first;
        auto tvIR = this->tensorVars.begin()->second;
        auto call = ir::Call::make("runtime->get_logical_partition", {ctx, getLogicalRegion(tvIR), partitionings.at(tv)}, LogicalPartition);
        itlStmts.push_back(ir::Return::make(call));
      }

      transfers.push_back(ir::Block::make(itlStmts));
    } else {
      // TODO (rohany): This code assumes that we always distributed multi
      //  dimensional task launches via index launches.
      auto point = this->indexVarToExprMap[forall.getIndexVar()];

      // Otherwise, we make a loop that launches the task.
      std::vector<Stmt> taskCallStmts;
      taskCallStmts.push_back(ir::VarDecl::make(point, ir::Deref::make(domainIter, pointT)));
      std::vector<Expr> regionReqs;
      std::vector<Expr> regionReqArgs;
      for (auto& it : this->tensorVarOrdering) {
        auto tv = it;
        auto tvIR = this->tensorVars[tv];
        // If the tensor is being transferred at this level, then use the
        // corresponding partition. Otherwise, use the tensorvar itself.
        auto priv = getPriv(tv);
        if (util::contains(partitionings, tv)) {
          auto call = ir::Call::make(
              "runtime->get_logical_subregion_by_color",
              {
                  ctx,
                  ir::Call::make(
                      "runtime->get_logical_partition",
                      {ctx, getLogicalRegion(tvIR), partitionings.at(tv)},
                      Auto
                  ),
                  point
              },
              Auto
          );
          auto subreg = ir::Var::make(tv.getName() + "subReg", Auto);
          taskCallStmts.push_back(ir::VarDecl::make(subreg, call));
          regionReqArgs = {
              subreg,
              priv.first,
              priv.second,
              getLogicalRegion(tvIR),
          };
        } else {
          regionReqArgs = {
              getLogicalRegion(tvIR),
              priv.first,
              priv.second,
              getLogicalRegion(tvIR)
          };
        }

        auto regReq = ir::Var::make(tv.getName() + "Req", RegionRequirement);
        auto makeReq = ir::Call::make(
            RegionRequirement.getName(),
            regionReqArgs,
            RegionRequirement
        );
        taskCallStmts.push_back(ir::VarDecl::make(regReq, makeReq));
        taskCallStmts.push_back(ir::SideEffect::make(ir::MethodCall::make(regReq, "add_field", {fidVal}, false, Auto)));

        // If the task being launched doesn't access the target region, then we can
        // virtually map the region.
        AccessFinder finder; finder.targetVar = tvIR;
        body.accept(&finder);
        if (!finder.readsVar) {
          taskCallStmts.push_back(ir::Assign::make(ir::FieldAccess::make(regReq, "tag", false, Auto), virtualMap));
        }

        regionReqs.push_back(regReq);
      }

      auto args = ir::Var::make("taskArgs", Auto);
      taskCallStmts.push_back(ir::PackTaskArgs::make(args, taskID, {}, {}));

      auto launcher = ir::Var::make("launcher", TaskLauncher);
      auto launcherMake = ir::Call::make(
        TaskLauncher.getName(),
        {
          ir::Call::make("taskID", {taskID}, Datatype::Int32),
          args,
        },
        TaskLauncher
      );
      taskCallStmts.push_back(ir::VarDecl::make(launcher, launcherMake));
      for (auto& req : regionReqs) {
        auto mcall = ir::MethodCall::make(launcher, "add_region_requirement", {req}, false /* deref */, Auto);
        taskCallStmts.push_back(ir::SideEffect::make(mcall));
      }
      // The actual task call.
      auto tcall = ir::Call::make("runtime->execute_task", {ctx, launcher}, Auto);
      taskCallStmts.push_back(ir::SideEffect::make(tcall));

      auto tcallLoop = ir::For::make(
          domainIter,
          ir::Call::make(pointInDimT.getName(), {domain}, pointInDimT),
          ir::MethodCall::make(domainIter, "valid", {}, false /* deref */, Datatype::Bool),
          1 /* increment -- hack to get ++ */,
          ir::Block::make(taskCallStmts)
      );
      transfers.push_back(tcallLoop);
    }
  }

  // If this forall is supposed to be replaced with a call to a leaf kernel,
  // do so and don't emit the surrounding loop and recovery statements.
  if (util::contains(this->calls, forall.getIndexVar())) {
    return Block::make({declarePartitionBounds, this->calls[forall.getIndexVar()]->replaceValidStmt(
        forall,
        this->provGraph,
        this->tensorVars,
        this->performingLegionReduction,
        this->definedIndexVarsOrdered,
        this->underivedBounds,
        this->indexVarToExprMap,
        this->iterators
    )});
  }

  body = Block::make({recoveryStmt, declarePartitionBounds, body});

  Stmt posAppend = generateAppendPositions(appenders);

  LoopKind kind = LoopKind::Serial;
  if (forall.isDistributed()) {
    kind = LoopKind::Distributed;
  } else if (forall.getParallelUnit() == ParallelUnit::CPUVector && !ignoreVectorize) {
    kind = LoopKind::Vectorized;
  } else if (forall.getParallelUnit() != ParallelUnit::NotParallel
            && forall.getOutputRaceStrategy() != OutputRaceStrategy::ParallelReduction && !ignoreVectorize) {
    kind = LoopKind::Runtime;
  }

  if (forall.isDistributed()) {
    this->curDistVar = prevDistVar;
  }

  return Block::blanks(ir::Block::make(transfers),
                       For::make(coordinate, bounds[0], bounds[1], 1, body,
                                 kind,
                                 ignoreVectorize ? ParallelUnit::NotParallel : forall.getParallelUnit(),
                                 ignoreVectorize ? 0 : forall.getUnrollFactor(),
                                 // TODO (rohany): What do we do for vector width here?
                                 0,
                                 isTask, taskID),
                       posAppend);
}

  Stmt LowererImpl::lowerForallDenseAcceleration(Forall forall,
                                                 vector<Iterator> locators,
                                                 vector<Iterator> inserters,
                                                 vector<Iterator> appenders,
                                                 set<Access> reducedAccesses,
                                                 ir::Stmt recoveryStmt)
  {
    taco_iassert(locators.size() == 1) << "Optimizing a dense workspace is only supported when the consumer is the only RHS tensor";
    taco_iassert(provGraph.isFullyDerived(forall.getIndexVar())) << "Sparsely accelerating a dense workspace only works with fully derived index vars";
    taco_iassert(forall.getParallelUnit() == ParallelUnit::NotParallel) << "Sparsely accelerating a dense workspace only works within serial loops";


    TensorVar var;
    for (auto it = tensorVars.begin(); it != tensorVars.end(); ++it) {
      if (it->second == locators[0].getTensor() && util::contains(tempToIndexList, it->first)) {
        var = it->first;
        break;
      }
    }

    Expr indexList = tempToIndexList.at(var);
    Expr indexListSize = tempToIndexListSize.at(var);
    Expr bitGuard = tempToBitGuard.at(var);
    Expr loopVar = ir::Var::make(var.getName() + "_index_locator", taco::Int32, false, false);
    Expr coordinate = getCoordinateVar(forall.getIndexVar());

    if (forall.getParallelUnit() != ParallelUnit::NotParallel && forall.getOutputRaceStrategy() == OutputRaceStrategy::Atomics) {
      markAssignsAtomicDepth++;
      atomicParallelUnit = forall.getParallelUnit();
    }

    Stmt declareVar = VarDecl::make(coordinate, Load::make(indexList, loopVar));
    Stmt body = lowerForallBody(coordinate, forall.getStmt(), locators, inserters, appenders, reducedAccesses);
    Stmt resetGuard = ir::Store::make(bitGuard, coordinate, ir::Literal::make(false), markAssignsAtomicDepth > 0, atomicParallelUnit);

    if (forall.getParallelUnit() != ParallelUnit::NotParallel && forall.getOutputRaceStrategy() == OutputRaceStrategy::Atomics) {
      markAssignsAtomicDepth--;
    }

    body = Block::make(declareVar, recoveryStmt, body, resetGuard);

    Stmt posAppend = generateAppendPositions(appenders);

    LoopKind kind = LoopKind::Serial;
    if (forall.getParallelUnit() == ParallelUnit::CPUVector && !ignoreVectorize) {
      kind = LoopKind::Vectorized;
    }
    else if (forall.getParallelUnit() != ParallelUnit::NotParallel
             && forall.getOutputRaceStrategy() != OutputRaceStrategy::ParallelReduction && !ignoreVectorize) {
      kind = LoopKind::Runtime;
    }

    return Block::blanks(For::make(loopVar, 0, indexListSize, 1, body, kind,
                                         ignoreVectorize ? ParallelUnit::NotParallel : forall.getParallelUnit(),
                                         ignoreVectorize ? 0 : forall.getUnrollFactor()),
                                         posAppend);
  }

Stmt LowererImpl::lowerForallCoordinate(Forall forall, Iterator iterator,
                                        vector<Iterator> locators,
                                        vector<Iterator> inserters,
                                        vector<Iterator> appenders,
                                        set<Access> reducedAccesses,
                                        ir::Stmt recoveryStmt) {
  taco_not_supported_yet;
  return Stmt();
}

Stmt LowererImpl::lowerForallPosition(Forall forall, Iterator iterator,
                                      vector<Iterator> locators,
                                      vector<Iterator> inserters,
                                      vector<Iterator> appenders,
                                      set<Access> reducedAccesses,
                                      ir::Stmt recoveryStmt)
{
  Expr coordinate = getCoordinateVar(forall.getIndexVar());
  Stmt declareCoordinate = Stmt();
  Stmt strideGuard = Stmt();
  Stmt boundsGuard = Stmt();
  if (provGraph.isCoordVariable(forall.getIndexVar())) {
    Expr coordinateArray = iterator.posAccess(iterator.getPosVar(),
                                              coordinates(iterator)).getResults()[0];
    // If the iterator is windowed, we must recover the coordinate index
    // variable from the windowed space.
    if (iterator.isWindowed()) {
      if (iterator.isStrided()) {
        // In this case, we're iterating over a compressed level with a for
        // loop. Since the iterator variable will get incremented by the for
        // loop, the guard introduced for stride checking doesn't need to
        // increment the iterator variable.
        strideGuard = this->strideBoundsGuard(iterator, coordinateArray, false /* incrementPosVar */);
      }
      coordinateArray = this->projectWindowedPositionToCanonicalSpace(iterator, coordinateArray);
      // If this forall is being parallelized via CPU threads (OpenMP), then we can't
      // emit a `break` statement, since OpenMP doesn't support breaking out of a
      // parallel loop. Instead, we'll bound the top of the loop and omit the check.
      if (forall.getParallelUnit() != ParallelUnit::CPUThread) {
        boundsGuard = this->upperBoundGuardForWindowPosition(iterator, coordinate);
      }
    }
    declareCoordinate = VarDecl::make(coordinate, coordinateArray);
  }
  if (forall.getParallelUnit() != ParallelUnit::NotParallel && forall.getOutputRaceStrategy() == OutputRaceStrategy::Atomics) {
    markAssignsAtomicDepth++;
  }

  Stmt body = lowerForallBody(coordinate, forall.getStmt(),
                              locators, inserters, appenders, reducedAccesses);

  if (forall.getParallelUnit() != ParallelUnit::NotParallel && forall.getOutputRaceStrategy() == OutputRaceStrategy::Atomics) {
    markAssignsAtomicDepth--;
  }

  body = Block::make(recoveryStmt, body);

  // Code to append positions
  Stmt posAppend = generateAppendPositions(appenders);

  // Code to compute iteration bounds
  Stmt boundsCompute;
  Expr startBound, endBound;
  Expr parentPos = iterator.getParent().getPosVar();
  if (!provGraph.isUnderived(iterator.getIndexVar())) {
    vector<Expr> bounds = provGraph.deriveIterBounds(iterator.getIndexVar(), definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, iterators);
    startBound = bounds[0];
    endBound = bounds[1];
  }
  else if (iterator.getParent().isRoot() || iterator.getParent().isUnique()) {
    // E.g. a compressed mode without duplicates
    ModeFunction bounds = iterator.posBounds(parentPos);
    boundsCompute = bounds.compute();
    startBound = bounds[0];
    endBound = bounds[1];
    // If we have a window on this iterator, then search for the start of
    // the window rather than starting at the beginning of the level.
    if (iterator.isWindowed()) {
      auto startBoundCopy = startBound;
      startBound = this->searchForStartOfWindowPosition(iterator, startBound, endBound);
      // As discussed above, if this position loop is parallelized over CPU
      // threads (OpenMP), then we need to have an explicit upper bound to
      // the for loop, instead of breaking out of the loop in the middle.
      if (forall.getParallelUnit() == ParallelUnit::CPUThread) {
        endBound = this->searchForEndOfWindowPosition(iterator, startBoundCopy, endBound);
      }
    }
  } else {
    taco_iassert(iterator.isOrdered() && iterator.getParent().isOrdered());
    taco_iassert(iterator.isCompact() && iterator.getParent().isCompact());

    // E.g. a compressed mode with duplicates. Apply iterator chaining
    Expr parentSegend = iterator.getParent().getSegendVar();
    ModeFunction startBounds = iterator.posBounds(parentPos);
    ModeFunction endBounds = iterator.posBounds(ir::Sub::make(parentSegend, 1));
    boundsCompute = Block::make(startBounds.compute(), endBounds.compute());
    startBound = startBounds[0];
    endBound = endBounds[1];
  }

  LoopKind kind = LoopKind::Serial;
  // TODO (rohany): This isn't needed right now.
//  if (forall.isDistributed()) {
//    std::cout << "marking forall as distributed position" << std::endl;
//    kind = LoopKind::Distributed;
//  } else
  if (forall.getParallelUnit() == ParallelUnit::CPUVector && !ignoreVectorize) {
    kind = LoopKind::Vectorized;
  } else if (forall.getParallelUnit() != ParallelUnit::NotParallel
           && forall.getOutputRaceStrategy() != OutputRaceStrategy::ParallelReduction && !ignoreVectorize) {
    kind = LoopKind::Runtime;
  }

// Loop with preamble and postamble
  return Block::blanks(
                       boundsCompute,
                       For::make(iterator.getPosVar(), startBound, endBound, 1,
                                 Block::make(strideGuard, declareCoordinate, boundsGuard, body),
                                 kind,
                                 ignoreVectorize ? ParallelUnit::NotParallel : forall.getParallelUnit(), ignoreVectorize ? 0 : forall.getUnrollFactor()),
                       posAppend);

}

Stmt LowererImpl::lowerForallFusedPosition(Forall forall, Iterator iterator,
                                      vector<Iterator> locators,
                                      vector<Iterator> inserters,
                                      vector<Iterator> appenders,
                                      set<Access> reducedAccesses,
                                      ir::Stmt recoveryStmt)
{
  Expr coordinate = getCoordinateVar(forall.getIndexVar());
  Stmt declareCoordinate = Stmt();
  if (provGraph.isCoordVariable(forall.getIndexVar())) {
    Expr coordinateArray = iterator.posAccess(iterator.getPosVar(),
                                              coordinates(iterator)).getResults()[0];
    declareCoordinate = VarDecl::make(coordinate, coordinateArray);
  }

  // declare upper-level underived ancestors that will be tracked with while loops
  Expr writeResultCond;
  vector<Stmt> loopsToTrackUnderived;
  vector<Stmt> searchForUnderivedStart;
  std::map<IndexVar, vector<Expr>> coordinateBounds = provGraph.deriveCoordBounds(definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, iterators);
  vector<IndexVar> underivedAncestors = provGraph.getUnderivedAncestors(forall.getIndexVar());
  if (underivedAncestors.size() > 1) {
    // each underived ancestor is initialized to min coordinate bound
    IndexVar posIteratorVar;
#if TACO_ASSERTS
    bool hasIteratorAncestor = provGraph.getPosIteratorAncestor(
        iterator.getIndexVar(), &posIteratorVar);
    taco_iassert(hasIteratorAncestor);
#else /* !TACO_ASSERTS */
    provGraph.getPosIteratorAncestor(
        iterator.getIndexVar(), &posIteratorVar);
#endif /* TACO_ASSERTS */
    // get pos variable then search for leveliterators to find the corresponding iterator

    Iterator posIterator;
    auto iteratorMap = iterators.levelIterators();
    int modePos = -1; // select lowest level possible
    for (auto it = iteratorMap.begin(); it != iteratorMap.end(); it++) {
      if (it->second.getIndexVar() == posIteratorVar && (int) it->first.getModePos() > modePos) {
        posIterator = it->second;
        modePos = (int) it->first.getModePos();
      }
    }
    taco_iassert(posIterator.hasPosIter());

    if (inParallelLoopDepth == 0) {
      for (int i = 0; i < (int) underivedAncestors.size() - 1; i ++) {
        // TODO: only if level is sparse emit underived_pos
        header.push_back(VarDecl::make(this->iterators.modeIterator(underivedAncestors[i]).getPosVar(), 0)); // TODO: set to start position bound
        header.push_back(VarDecl::make(getCoordinateVar(underivedAncestors[i]), coordinateBounds[underivedAncestors[i]][0]));
      }
    } else {
      searchForUnderivedStart.push_back(searchForFusedPositionStart(forall, posIterator));
    }

    Expr parentPos = this->iterators.modeIterator(underivedAncestors[underivedAncestors.size() - 2]).getPosVar();
    ModeFunction posBounds = posIterator.posBounds(parentPos);
    writeResultCond = ir::Eq::make(ir::Add::make(indexVarToExprMap[posIterator.getIndexVar()], 1), posBounds[1]);

    Stmt loopToTrackUnderiveds; // to track next ancestor
    for (int i = 0; i < (int) underivedAncestors.size() - 1; i++) {
      Expr coordVarUnknown = getCoordinateVar(underivedAncestors[i]);
      Expr posVarKnown = this->iterators.modeIterator(underivedAncestors[i+1]).getPosVar();
      if (i == (int) underivedAncestors.size() - 2) {
        posVarKnown = indexVarToExprMap[posIterator.getIndexVar()];
      }
      Expr posVarUnknown = this->iterators.modeIterator(underivedAncestors[i]).getPosVar();

      Iterator posIteratorLevel = posIterator;
      for (int j = (int) underivedAncestors.size() - 2; j > i; j--) { // take parent of iterator enough times to get correct level
        posIteratorLevel = posIteratorLevel.getParent();
      }

      ModeFunction posBoundsLevel = posIteratorLevel.posBounds(posVarUnknown);
      Expr loopcond = ir::Eq::make(posVarKnown, posBoundsLevel[1]);
      Stmt locateCoordVar;
      if (posIteratorLevel.getParent().hasPosIter()) {
        locateCoordVar = ir::Assign::make(coordVarUnknown, ir::Load::make(posIteratorLevel.getParent().getMode().getModePack().getArray(1), posVarUnknown));
      }
      else {
        locateCoordVar = ir::Assign::make(coordVarUnknown, posVarUnknown);
      }
      Stmt loopBody = ir::Block::make(compoundAssign(posVarUnknown, 1), locateCoordVar, loopToTrackUnderiveds);
      if (posIteratorLevel.getParent().hasPosIter()) { // TODO: if level is unique or not
        loopToTrackUnderiveds = IfThenElse::make(loopcond, loopBody);
      }
      else {
        loopToTrackUnderiveds = While::make(loopcond, loopBody);
      }
    }
    loopsToTrackUnderived.push_back(loopToTrackUnderiveds);
  }

  if (forall.getParallelUnit() != ParallelUnit::NotParallel && forall.getOutputRaceStrategy() == OutputRaceStrategy::Atomics) {
    markAssignsAtomicDepth++;
  }

  Stmt body = lowerForallBody(coordinate, forall.getStmt(),
                              locators, inserters, appenders, reducedAccesses);

  if (forall.getParallelUnit() != ParallelUnit::NotParallel && forall.getOutputRaceStrategy() == OutputRaceStrategy::Atomics) {
    markAssignsAtomicDepth--;
  }

  body = Block::make(recoveryStmt, Block::make(loopsToTrackUnderived), body);

  // Code to write results if using temporary and reset temporary
  if (!whereConsumers.empty() && whereConsumers.back().defined()) {
    Expr temp = tensorVars.find(whereTemps.back())->second;
    Stmt writeResults = Block::make(whereConsumers.back(), ir::Assign::make(temp, ir::Literal::zero(temp.type())));
    body = Block::make(body, IfThenElse::make(writeResultCond, writeResults));
  }

  // Code to append positions
  Stmt posAppend = generateAppendPositions(appenders);

  // Code to compute iteration bounds
  Stmt boundsCompute;
  Expr startBound, endBound;
  if (!provGraph.isUnderived(iterator.getIndexVar())) {
    vector<Expr> bounds = provGraph.deriveIterBounds(iterator.getIndexVar(), definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, iterators);
    startBound = bounds[0];
    endBound = bounds[1];
  }
  else if (iterator.getParent().isRoot() || iterator.getParent().isUnique()) {
    // E.g. a compressed mode without duplicates
    Expr parentPos = iterator.getParent().getPosVar();
    ModeFunction bounds = iterator.posBounds(parentPos);
    boundsCompute = bounds.compute();
    startBound = bounds[0];
    endBound = bounds[1];
  } else {
    taco_iassert(iterator.isOrdered() && iterator.getParent().isOrdered());
    taco_iassert(iterator.isCompact() && iterator.getParent().isCompact());

    // E.g. a compressed mode with duplicates. Apply iterator chaining
    Expr parentPos = iterator.getParent().getPosVar();
    Expr parentSegend = iterator.getParent().getSegendVar();
    ModeFunction startBounds = iterator.posBounds(parentPos);
    ModeFunction endBounds = iterator.posBounds(ir::Sub::make(parentSegend, 1));
    boundsCompute = Block::make(startBounds.compute(), endBounds.compute());
    startBound = startBounds[0];
    endBound = endBounds[1];
  }

  LoopKind kind = LoopKind::Serial;
  if (forall.getParallelUnit() == ParallelUnit::CPUVector && !ignoreVectorize) {
    kind = LoopKind::Vectorized;
  }
  else if (forall.getParallelUnit() != ParallelUnit::NotParallel
           && forall.getOutputRaceStrategy() != OutputRaceStrategy::ParallelReduction && !ignoreVectorize) {
    kind = LoopKind::Runtime;
  }
  // Loop with preamble and postamble
  return Block::blanks(boundsCompute,
                       Block::make(Block::make(searchForUnderivedStart),
                       For::make(indexVarToExprMap[iterator.getIndexVar()], startBound, endBound, 1,
                                 Block::make(declareCoordinate, body),
                                 kind,
                                 ignoreVectorize ? ParallelUnit::NotParallel : forall.getParallelUnit(), ignoreVectorize ? 0 : forall.getUnrollFactor())),
                       posAppend);

}

Stmt LowererImpl::lowerMergeLattice(MergeLattice lattice, IndexVar coordinateVar,
                                    IndexStmt statement,
                                    const std::set<Access>& reducedAccesses)
{
  Expr coordinate = getCoordinateVar(coordinateVar);
  vector<Iterator> appenders = filter(lattice.results(),
                                      [](Iterator it){return it.hasAppend();});

  vector<Iterator> mergers = lattice.points()[0].mergers();
  Stmt iteratorVarInits = codeToInitializeIteratorVars(lattice.iterators(), lattice.points()[0].rangers(), mergers, coordinate, coordinateVar);

  // if modeiteratornonmerger then will be declared in codeToInitializeIteratorVars
  auto modeIteratorsNonMergers =
          filter(lattice.points()[0].iterators(), [mergers](Iterator it){
            bool isMerger = find(mergers.begin(), mergers.end(), it) != mergers.end();
            return it.isDimensionIterator() && !isMerger;
          });
  bool resolvedCoordDeclared = !modeIteratorsNonMergers.empty();

  vector<Stmt> mergeLoopsVec;
  for (MergePoint point : lattice.points()) {
    // Each iteration of this loop generates a while loop for one of the merge
    // points in the merge lattice.
    IndexStmt zeroedStmt = zero(statement, getExhaustedAccesses(point,lattice));
    MergeLattice sublattice = lattice.subLattice(point);
    Stmt mergeLoop = lowerMergePoint(sublattice, coordinate, coordinateVar, zeroedStmt, reducedAccesses, resolvedCoordDeclared);
    mergeLoopsVec.push_back(mergeLoop);
  }
  Stmt mergeLoops = Block::make(mergeLoopsVec);

  // Append position to the pos array
  Stmt appendPositions = generateAppendPositions(appenders);

  return Block::blanks(iteratorVarInits,
                       mergeLoops,
                       appendPositions);
}

Stmt LowererImpl::lowerMergePoint(MergeLattice pointLattice,
                                  ir::Expr coordinate, IndexVar coordinateVar, IndexStmt statement,
                                  const std::set<Access>& reducedAccesses, bool resolvedCoordDeclared)
{
  MergePoint point = pointLattice.points().front();

  vector<Iterator> iterators = point.iterators();
  vector<Iterator> mergers = point.mergers();
  vector<Iterator> rangers = point.rangers();
  vector<Iterator> locators = point.locators();

  taco_iassert(iterators.size() > 0);
  taco_iassert(mergers.size() > 0);
  taco_iassert(rangers.size() > 0);

  // Load coordinates from position iterators
  Stmt loadPosIterCoordinates = codeToLoadCoordinatesFromPosIterators(iterators, !resolvedCoordDeclared);

  // Any iterators with an index set have extra work to do at the header
  // of the merge point.
  std::vector<ir::Stmt> indexSetStmts;
  for (auto& iter : filter(iterators, [](Iterator it) { return it.hasIndexSet(); })) {
    // For each iterator A with an index set B, emit the following code:
    //   setMatch = min(A, B); // Check whether A matches its index set at this point.
    //   if (A == setMatch && B == setMatch) {
    //     // If there was a match, project down the values of the iterators
    //     // to be the position variable of the index set iterator. This has the
    //     // effect of remapping the index of A to be the i'th position of the set.
    //     A_coord = B_pos;
    //     B_coord = B_pos;
    //   } else {
    //     // Advance the iterator and it's index set iterator accordingly if
    //     // there wasn't a match.
    //     A_pos += (A == setMatch);
    //     B_pos += (B == setMatch);
    //     // We must continue so that we only proceed to the rest of the cases in
    //     // the merge if there actually is a point present for A.
    //     continue;
    //   }
    auto setMatch = ir::Var::make("setMatch", Int());
    auto indexSetIter = iter.getIndexSetIterator();
    indexSetStmts.push_back(ir::VarDecl::make(setMatch, ir::Min::make(this->coordinates({iter, indexSetIter}))));
    // Equality checks for each iterator.
    auto iterEq = ir::Eq::make(iter.getCoordVar(), setMatch);
    auto setEq = ir::Eq::make(indexSetIter.getCoordVar(), setMatch);
    // Code to shift down each iterator to the position space of the index set.
    auto shiftDown = ir::Block::make(
      ir::Assign::make(iter.getCoordVar(), indexSetIter.getPosVar()),
      ir::Assign::make(indexSetIter.getCoordVar(), indexSetIter.getPosVar())
    );
    // Code to increment both iterator variables.
    auto incr = ir::Block::make(
      compoundAssign(iter.getIteratorVar(), ir::Cast::make(Eq::make(iter.getCoordVar(), setMatch), iter.getIteratorVar().type())),
      compoundAssign(indexSetIter.getIteratorVar(), ir::Cast::make(Eq::make(indexSetIter.getCoordVar(), setMatch), indexSetIter.getIteratorVar().type())),
      ir::Continue::make()
    );
    // Code that uses the defined parts together in the if-then-else.
    indexSetStmts.push_back(ir::IfThenElse::make(ir::And::make(iterEq, setEq), shiftDown, incr));
  }

  // Merge iterator coordinate variables
  Stmt resolvedCoordinate = resolveCoordinate(mergers, coordinate, !resolvedCoordDeclared);

  // Locate positions
  Stmt loadLocatorPosVars = declLocatePosVars(locators);

  // Deduplication loops
  auto dupIters = filter(iterators, [](Iterator it){return !it.isUnique() &&
                                                           it.hasPosIter();});
  bool alwaysReduce = (mergers.size() == 1 && mergers[0].hasPosIter());
  Stmt deduplicationLoops = reduceDuplicateCoordinates(coordinate, dupIters,
                                                       alwaysReduce);

  // One case for each child lattice point lp
  Stmt caseStmts = lowerMergeCases(coordinate, coordinateVar, statement, pointLattice,
                                   reducedAccesses);

  // Increment iterator position variables
  Stmt incIteratorVarStmts = codeToIncIteratorVars(coordinate, coordinateVar, iterators, mergers);

  /// While loop over rangers
  return While::make(checkThatNoneAreExhausted(rangers),
                     Block::make(loadPosIterCoordinates,
                                 ir::Block::make(indexSetStmts),
                                 resolvedCoordinate,
                                 loadLocatorPosVars,
                                 deduplicationLoops,
                                 caseStmts,
                                 incIteratorVarStmts));
}

Stmt LowererImpl::resolveCoordinate(std::vector<Iterator> mergers, ir::Expr coordinate, bool emitVarDecl) {
  if (mergers.size() == 1) {
    Iterator merger = mergers[0];
    if (merger.hasPosIter()) {
      // Just one position iterator so it is the resolved coordinate
      ModeFunction posAccess = merger.posAccess(merger.getPosVar(),
                                                coordinates(merger));
      auto access = posAccess[0];
      auto windowVarDecl = Stmt();
      auto stride = Stmt();
      auto guard = Stmt();
      // If the iterator is windowed, we must recover the coordinate index
      // variable from the windowed space.
      if (merger.isWindowed()) {

        // If the iterator is strided, then we have to skip over coordinates
        // that don't match the stride. To do that, we insert a guard on the
        // access. We first extract the access into a temp to avoid emitting
        // a duplicate load on the _crd array.
        if (merger.isStrided()) {
          windowVarDecl = VarDecl::make(merger.getWindowVar(), access);
          access = merger.getWindowVar();
          // Since we're merging values from a compressed array (not iterating over it),
          // we need to advance the outer loop if the current coordinate is not
          // along the desired stride. So, we pass true to the incrementPosVar
          // argument of strideBoundsGuard.
          stride = this->strideBoundsGuard(merger, access, true /* incrementPosVar */);
        }

        access = this->projectWindowedPositionToCanonicalSpace(merger, access);
        guard = this->upperBoundGuardForWindowPosition(merger, coordinate);
      }
      Stmt resolution = emitVarDecl ? VarDecl::make(coordinate, access) : Assign::make(coordinate, access);
      return Block::make(posAccess.compute(),
                         windowVarDecl,
                         stride,
                         resolution,
                         guard);
    }
    else if (merger.hasCoordIter()) {
      taco_not_supported_yet;
      return Stmt();
    }
    else if (merger.isDimensionIterator()) {
      // Just one dimension iterator so resolved coordinate already exist and we
      // do nothing
      return Stmt();
    }
    else {
      taco_ierror << "Unexpected type of single iterator " << merger;
      return Stmt();
    }
  }
  else {
    // Multiple position iterators so the smallest is the resolved coordinate
    if (emitVarDecl) {
      return VarDecl::make(coordinate, Min::make(coordinates(mergers)));
    }
    else {
      return Assign::make(coordinate, Min::make(coordinates(mergers)));
    }
  }
}

Stmt LowererImpl::lowerMergeCases(ir::Expr coordinate, IndexVar coordinateVar, IndexStmt stmt,
                                  MergeLattice lattice,
                                  const std::set<Access>& reducedAccesses)
{
  vector<Stmt> result;

  vector<Iterator> appenders;
  vector<Iterator> inserters;
  tie(appenders, inserters) = splitAppenderAndInserters(lattice.results());

  // Just one iterator so no conditionals
  if (lattice.iterators().size() == 1) {
    Stmt body = lowerForallBody(coordinate, stmt, {}, inserters,
                                appenders, reducedAccesses);
    result.push_back(body);
  }
  else {
    vector<pair<Expr,Stmt>> cases;
    for (MergePoint point : lattice.points()) {

      // Construct case expression
      vector<Expr> coordComparisons;
      for (Iterator iterator : point.rangers()) {
        if (!(provGraph.isCoordVariable(iterator.getIndexVar()) && provGraph.isDerivedFrom(iterator.getIndexVar(), coordinateVar))) {
          coordComparisons.push_back(Eq::make(iterator.getCoordVar(), coordinate));
        }
      }

      // Construct case body
      IndexStmt zeroedStmt = zero(stmt, getExhaustedAccesses(point, lattice));
      Stmt body = lowerForallBody(coordinate, zeroedStmt, {},
                                  inserters, appenders, reducedAccesses);
      if (coordComparisons.empty()) {
        Stmt body = lowerForallBody(coordinate, stmt, {}, inserters,
                                    appenders, reducedAccesses);
        result.push_back(body);
        break;
      }
      cases.push_back({taco::ir::conjunction(coordComparisons), body});
    }
    result.push_back(Case::make(cases, lattice.exact()));
  }

  return Block::make(result);
}


Stmt LowererImpl::lowerForallBody(Expr coordinate, IndexStmt stmt,
                                  vector<Iterator> locators,
                                  vector<Iterator> inserters,
                                  vector<Iterator> appenders,
                                  const set<Access>& reducedAccesses) {
  Stmt initVals = resizeAndInitValues(appenders, reducedAccesses);

  // Inserter positions
  Stmt declInserterPosVars = declLocatePosVars(inserters);

  // Locate positions
  Stmt declLocatorPosVars = declLocatePosVars(locators);

  if (captureNextLocatePos) {
    capturedLocatePos = Block::make(declInserterPosVars, declLocatorPosVars);
    captureNextLocatePos = false;
  }

  // Code of loop body statement
  Stmt body = lower(stmt);

  // Code to append coordinates
  Stmt appendCoords = appendCoordinate(appenders, coordinate);

  // TODO: Emit code to insert coordinates

  return Block::make(initVals,
                     declInserterPosVars,
                     declLocatorPosVars,
                     body,
                     appendCoords);
}

Expr LowererImpl::getTemporarySize(Where where) {
  TensorVar temporary = where.getTemporary();
  Dimension temporarySize = temporary.getType().getShape().getDimension(0);
  Access temporaryAccess = getResultAccesses(where.getProducer()).first[0];
  std::vector<IndexVar> indexVars = temporaryAccess.getIndexVars();

  if(util::all(indexVars, [&](const IndexVar& var) { return provGraph.isUnderived(var);})) {
    // All index vars underived then use tensor properties to get tensor size
    taco_iassert(util::contains(dimensions, indexVars[0])) << "Missing " << indexVars[0];
    ir::Expr size = dimensions.at(indexVars[0]);
    for(size_t i = 1; i < indexVars.size(); ++i) {
      taco_iassert(util::contains(dimensions, indexVars[i])) << "Missing " << indexVars[i];
      size = ir::Mul::make(size, dimensions.at(indexVars[i]));
    }
    return size;
  }

  if (temporarySize.isFixed()) {
    return ir::Literal::make(temporarySize.getSize());
  }

  if (temporarySize.isIndexVarSized()) {
    IndexVar var = temporarySize.getIndexVarSize();
    vector<Expr> bounds = provGraph.deriveIterBounds(var, definedIndexVarsOrdered, underivedBounds,
                                                     indexVarToExprMap, iterators);
    return ir::Sub::make(bounds[1], bounds[0]);
  }

  taco_ierror; // TODO
  return Expr();
}

vector<Stmt> LowererImpl::codeToInitializeDenseAcceleratorArrays(Where where) {
  TensorVar temporary = where.getTemporary();

  // TODO: emit as uint64 and manually emit bit pack code
  const Datatype bitGuardType = taco::Bool;
  const std::string bitGuardName = temporary.getName() + "_already_set";
  const Expr bitGuardSize = getTemporarySize(where);
  const Expr alreadySetArr = ir::Var::make(bitGuardName,
                                           bitGuardType,
                                           true, false);

  // TODO: TACO should probably keep state on if it can use int32 or if it should switch to
  //       using int64 for indices. This assumption is made in other places of taco.
  const Datatype indexListType = taco::Int32;
  const std::string indexListName = temporary.getName() + "_index_list";
  const Expr indexListArr = ir::Var::make(indexListName,
                                          indexListType,
                                          true, false);

  // no decl for shared memory
  Stmt alreadySetDecl = Stmt();
  Stmt indexListDecl = Stmt();
  const Expr indexListSizeExpr = ir::Var::make(indexListName + "_size", taco::Int32, false, false);
  Stmt freeTemps = Block::make(Free::make(indexListArr), Free::make(alreadySetArr));
  if ((isa<Forall>(where.getProducer()) && inParallelLoopDepth == 0) || !should_use_CUDA_codegen()) {
    alreadySetDecl = VarDecl::make(alreadySetArr, ir::Literal::make(0));
    indexListDecl = VarDecl::make(indexListArr, ir::Literal::make(0));
  }

  tempToIndexList[temporary] = indexListArr;
  tempToIndexListSize[temporary] = indexListSizeExpr;
  tempToBitGuard[temporary] = alreadySetArr;

  Stmt allocateIndexList = Allocate::make(indexListArr, bitGuardSize);
  if(should_use_CUDA_codegen()) {
    Stmt allocateAlreadySet = Allocate::make(alreadySetArr, bitGuardSize);
    Expr p = Var::make("p" + temporary.getName(), Int());
    Stmt guardZeroInit = Store::make(alreadySetArr, p, ir::Literal::zero(bitGuardType));

    Stmt zeroInitLoop = For::make(p, 0, bitGuardSize, 1, guardZeroInit, LoopKind::Serial);
    Stmt inits = Block::make(alreadySetDecl, indexListDecl, allocateAlreadySet, allocateIndexList, zeroInitLoop);
    return {inits, freeTemps};
  } else {
    Expr sizeOfElt = Sizeof::make(bitGuardType);
    Expr callocAlreadySet = ir::Call::make("calloc", {bitGuardSize, sizeOfElt}, Int());
    Stmt allocateAlreadySet = VarDecl::make(alreadySetArr, callocAlreadySet);
    Stmt inits = Block::make(indexListDecl, allocateIndexList, allocateAlreadySet);
    return {inits, freeTemps};
  }

}

// Returns true if the following conditions are met:
// 1) The temporary is a dense vector
// 2) There is only one value on the right hand side of the consumer
//    -- We would need to handle sparse acceleration in the merge lattices for 
//       multiple operands on the RHS
// 3) The left hand side of the where consumer is sparse, if the consumer is an 
//    assignment
// 4) CPU Code is being generated (TEMPORARY - This should be removed)
//    -- The sorting calls and calloc call in lower where are CPU specific. We 
//       could map calloc to a cudaMalloc and use a library like CUB to emit 
//       the sort. CUB support is built into CUDA 11 but not prior versions of 
//       CUDA so in that case, we'd probably need to include the CUB headers in 
//       the generated code.
std::pair<bool,bool> LowererImpl::canAccelerateDenseTemp(Where where) {
  // TODO: TEMPORARY -- Needs to be removed
  if(should_use_CUDA_codegen()) {
    return std::make_pair(false, false);
  }

  TensorVar temporary = where.getTemporary();
  // (1) Temporary is dense vector
  if(!isDense(temporary.getFormat()) || temporary.getOrder() != 1) {
    return std::make_pair(false, false);
  }

  // (2) Multiple operands in inputs (need lattice to reason about iteration)
  const auto inputAccesses = getArgumentAccesses(where.getConsumer());
  if(inputAccesses.size() > 1 || inputAccesses.empty()) {
    return std::make_pair(false, false);
  }

  // No or multiple results?
  const auto resultAccesses = getResultAccesses(where.getConsumer()).first;
  if(resultAccesses.size() > 1 || resultAccesses.empty()) {
    return std::make_pair(false, false);
  }

  // No check for size of tempVar since we enforced the temporary is a vector 
  // and if there is only one RHS value, it must (should?) be the temporary
  std::vector<IndexVar> tempVar = inputAccesses[0].getIndexVars();

  // Get index vars in result.
  std::vector<IndexVar> resultVars = resultAccesses[0].getIndexVars();
  auto it = std::find_if(resultVars.begin(), resultVars.end(),
      [&](const auto& resultVar) {
          return resultVar == tempVar[0] ||
                 provGraph.isDerivedFrom(tempVar[0], resultVar);
  });

  if (it == resultVars.end()) {
    return std::make_pair(true, false);
  }

  int index = (int)(it - resultVars.begin());
  TensorVar resultTensor = resultAccesses[0].getTensorVar();
  int modeIndex = resultTensor.getFormat().getModeOrdering()[index];
  ModeFormat varFmt = resultTensor.getFormat().getModeFormats()[modeIndex];
  // (3) Level of result is sparse
  if(varFmt.isFull()) {
    return std::make_pair(false, false);
  }

  // Only need to sort the workspace if the result needs to be ordered
  return std::make_pair(true, varFmt.isOrdered());
}

vector<Stmt> LowererImpl::codeToInitializeTemporary(Where where) {
  TensorVar temporary = where.getTemporary();

  const bool accelerateDense = canAccelerateDenseTemp(where).first;

  Stmt freeTemporary = Stmt();
  Stmt initializeTemporary = Stmt();
  if (isScalar(temporary.getType())) {
    initializeTemporary = defineScalarVariable(temporary, true);
    Expr tempSet = ir::Var::make(temporary.getName() + "_set", Datatype::Bool);
    Stmt initTempSet = VarDecl::make(tempSet, false);
    initializeTemporary = Block::make(initializeTemporary, initTempSet);
    tempToBitGuard[temporary] = tempSet;
  } else {
    // TODO: Need to support keeping track of initialized elements for
    //       temporaries that don't have sparse accelerator
    taco_iassert(!util::contains(guardedTemps, temporary) || accelerateDense);

    // When emitting code to accelerate dense workspaces with sparse iteration, we need the following arrays
    // to construct the result indices
    if(accelerateDense) {
      vector<Stmt> initAndFree = codeToInitializeDenseAcceleratorArrays(where);
      initializeTemporary = initAndFree[0];
      freeTemporary = initAndFree[1];
    }

    Expr values;
    if (util::contains(needCompute, temporary) &&
        needComputeValues(where, temporary)) {
      values = ir::Var::make(temporary.getName(),
                             temporary.getType().getDataType(), true, false);
      taco_iassert(temporary.getType().getOrder() == 1)
          << " Temporary order was " << temporary.getType().getOrder();  // TODO
      Expr size = getTemporarySize(where);

      // no decl needed for shared memory
      Stmt decl = Stmt();
      if ((isa<Forall>(where.getProducer()) && inParallelLoopDepth == 0) || !should_use_CUDA_codegen()) {
        decl = VarDecl::make(values, ir::Literal::make(0));
      }
      Stmt allocate = Allocate::make(values, size);

      freeTemporary = Block::make(freeTemporary, Free::make(values));
      initializeTemporary = Block::make(decl, initializeTemporary, allocate);
    }

    /// Make a struct object that lowerAssignment and lowerAccess can read
    /// temporary value arrays from.
    TemporaryArrays arrays;
    arrays.values = values;
    this->temporaryArrays.insert({temporary, arrays});
  }
  return {initializeTemporary, freeTemporary};
}

Stmt LowererImpl::lowerWhere(Where where) {
  TensorVar temporary = where.getTemporary();
  bool accelerateDenseWorkSpace, sortAccelerator;
  std::tie(accelerateDenseWorkSpace, sortAccelerator) =
      canAccelerateDenseTemp(where);

  // Declare and initialize the where statement's temporary
  vector<Stmt> temporaryValuesInitFree = {Stmt(), Stmt()};
  bool temporaryHoisted = false;
  for (auto it = temporaryInitialization.begin(); it != temporaryInitialization.end(); ++it) {
    if (it->second == where && it->first.getParallelUnit() == ParallelUnit::NotParallel && !isScalar(temporary.getType())) {
      temporaryHoisted = true;
    }
  }

  if (!temporaryHoisted) {
    temporaryValuesInitFree = codeToInitializeTemporary(where);
  }

  Stmt initializeTemporary = temporaryValuesInitFree[0];
  Stmt freeTemporary = temporaryValuesInitFree[1];

  match(where.getConsumer(),
        std::function<void(const AssignmentNode*)>([&](const AssignmentNode* op) {
            if (op->lhs.getTensorVar().getOrder() > 0) {
              whereTempsToResult[where.getTemporary()] = (const AccessNode *) op->lhs.ptr;
            }
        })
  );

  Stmt consumer = lower(where.getConsumer());
  if (accelerateDenseWorkSpace && sortAccelerator) {
    // We need to sort the indices array
    Expr listOfIndices = tempToIndexList.at(temporary);
    Expr listOfIndicesSize = tempToIndexListSize.at(temporary);
    Expr sizeOfElt = ir::Sizeof::make(listOfIndices.type());
    Stmt sortCall = ir::Sort::make({listOfIndices, listOfIndicesSize, sizeOfElt});
    consumer = Block::make(sortCall, consumer);
  }

  // Now that temporary allocations are hoisted, we always need to emit an initialization loop before entering the
  // producer but only if there is no dense acceleration
  if (util::contains(needCompute, temporary) && !isScalar(temporary.getType()) && !accelerateDenseWorkSpace) {
    // TODO: We only actually need to do this if:
    //      1) We use the temporary multiple times
    //      2) The PRODUCER RHS is sparse(not full). (Guarantees that old values are overwritten before consuming)

    Expr p = Var::make("p" + temporary.getName(), Int());
    Expr values = ir::Var::make(temporary.getName(),
                                temporary.getType().getDataType(),
                                true, false);
    Expr size = getTemporarySize(where);
    Stmt zeroInit = Store::make(values, p, ir::Literal::zero(temporary.getType().getDataType()));
    Stmt loopInit = For::make(p, 0, size, 1, zeroInit, LoopKind::Serial);
    initializeTemporary = Block::make(initializeTemporary, loopInit);
  }

  whereConsumers.push_back(consumer);
  whereTemps.push_back(where.getTemporary());
  captureNextLocatePos = true;

  // don't apply atomics to producer TODO: mark specific assignments as atomic
  bool restoreAtomicDepth = false;
  if (markAssignsAtomicDepth > 0) {
    markAssignsAtomicDepth--;
    restoreAtomicDepth = true;
  }

  Stmt producer = lower(where.getProducer());
  if (accelerateDenseWorkSpace) {
    const Expr indexListSizeExpr = tempToIndexListSize.at(temporary);
    const Stmt indexListSizeDecl = VarDecl::make(indexListSizeExpr, ir::Literal::make(0));
    initializeTemporary = Block::make(indexListSizeDecl, initializeTemporary);
  }

  if (restoreAtomicDepth) {
    markAssignsAtomicDepth++;
  }

  whereConsumers.pop_back();
  whereTemps.pop_back();
  whereTempsToResult.erase(where.getTemporary());
  return Block::make(initializeTemporary, producer, markAssignsAtomicDepth > 0 ? capturedLocatePos : ir::Stmt(), consumer,  freeTemporary);
}


Stmt LowererImpl::lowerSequence(Sequence sequence) {
  Stmt definition = lower(sequence.getDefinition());
  Stmt mutation = lower(sequence.getMutation());
  return Block::make(definition, mutation);
}


Stmt LowererImpl::lowerAssemble(Assemble assemble) {
  Stmt queries, freeQueryResults;
  if (generateAssembleCode() && assemble.getQueries().defined()) {
    std::vector<Stmt> allocStmts, freeStmts;
    const auto queryAccesses = getResultAccesses(assemble.getQueries()).first;
    for (const auto& queryAccess : queryAccesses) {
      const auto queryResult = queryAccess.getTensorVar();
      Expr values = ir::Var::make(queryResult.getName(),
                                  queryResult.getType().getDataType(),
                                  true, false);

      TemporaryArrays arrays;
      arrays.values = values;
      this->temporaryArrays.insert({queryResult, arrays});

      // Compute size of query result
      const auto indexVars = queryAccess.getIndexVars();
      taco_iassert(util::all(indexVars,
          [&](const auto& var) { return provGraph.isUnderived(var); }));
      Expr size = 1;
      for (const auto& indexVar : indexVars) {
        size = ir::Mul::make(size, getDimension(indexVar));
      }

      multimap<IndexVar, Iterator> readIterators;
      for (auto& read : getArgumentAccesses(assemble.getQueries())) {
        for (auto& readIterator : getIterators(read)) {
          for (auto& underivedAncestor :
              provGraph.getUnderivedAncestors(readIterator.getIndexVar())) {
            readIterators.insert({underivedAncestor, readIterator});
          }
        }
      }
      const auto writeIterators = getIterators(queryAccess);
      const bool zeroInit = hasSparseInserts(writeIterators, readIterators);
      if (zeroInit) {
        Expr sizeOfElt = Sizeof::make(queryResult.getType().getDataType());
        Expr callocValues = ir::Call::make("calloc", {size, sizeOfElt},
                                           queryResult.getType().getDataType());
        Stmt allocResult = VarDecl::make(values, callocValues);
        allocStmts.push_back(allocResult);
      }
      else {
        Stmt declResult = VarDecl::make(values, 0);
        allocStmts.push_back(declResult);

        Stmt allocResult = Allocate::make(values, size);
        allocStmts.push_back(allocResult);
      }

      Stmt freeResult = Free::make(values);
      freeStmts.push_back(freeResult);
    }
    Stmt allocResults = Block::make(allocStmts);
    freeQueryResults = Block::make(freeStmts);

    queries = lower(assemble.getQueries());
    queries = Block::blanks(allocResults, queries);
  }

  const auto& queryResults = assemble.getAttrQueryResults();
  const auto resultAccesses = getResultAccesses(assemble.getCompute()).first;

  std::vector<Stmt> initAssembleStmts;
  for (const auto& resultAccess : resultAccesses) {
    Expr prevSize = 1;
    std::vector<Expr> coords;
    const auto resultIterators = getIterators(resultAccess);
    const auto resultTensor = resultAccess.getTensorVar();
    const auto resultTensorVar = getTensorVar(resultTensor);
    const auto resultModeOrdering = resultTensor.getFormat().getModeOrdering();
    for (const auto& resultIterator : resultIterators) {
      if (generateAssembleCode()) {
        const size_t resultLevel = resultIterator.getMode().getLevel() - 1;
        const auto queryResultVars = queryResults.at(resultTensor)[resultLevel];
        std::vector<AttrQueryResult> queryResults;
        for (const auto& queryResultVar : queryResultVars) {
          queryResults.emplace_back(getTensorVar(queryResultVar),
                                    getValuesArray(queryResultVar));
        }

        if (resultIterator.hasSeqInsertEdge()) {
          Stmt initEdges = resultIterator.getSeqInitEdges(prevSize,
                                                          queryResults);
          initAssembleStmts.push_back(initEdges);

          Stmt insertEdgeLoop = resultIterator.getSeqInsertEdge(
              resultIterator.getParent().getPosVar(), coords, queryResults);
          auto locateCoords = coords;
          for (auto iter = resultIterator.getParent(); !iter.isRoot();
               iter = iter.getParent()) {
            if (iter.hasLocate()) {
              Expr dim = GetProperty::make(resultTensorVar,
                  TensorProperty::Dimension,
                  resultModeOrdering[iter.getMode().getLevel() - 1]);
              Expr pos = iter.getPosVar();
              Stmt initPos = VarDecl::make(pos, iter.locate(locateCoords)[0]);
              insertEdgeLoop = For::make(coords.back(), 0, dim, 1,
                                         Block::make(initPos, insertEdgeLoop));
            } else {
              taco_not_supported_yet;
            }
            locateCoords.pop_back();
          }
          initAssembleStmts.push_back(insertEdgeLoop);
        }

        Stmt initCoords = resultIterator.getInitCoords(prevSize, queryResults);
        initAssembleStmts.push_back(initCoords);
      }

      Stmt initYieldPos = resultIterator.getInitYieldPos(prevSize);
      initAssembleStmts.push_back(initYieldPos);

      prevSize = resultIterator.getAssembledSize(prevSize);
      coords.push_back(getCoordinateVar(resultIterator));
    }

    if (generateAssembleCode()) {
      // TODO: call calloc if not compact or not unpadded
      Expr valuesArr = getValuesArray(resultTensor);
      Stmt initValues = Allocate::make(valuesArr, prevSize);
      initAssembleStmts.push_back(initValues);
    }
  }
  Stmt initAssemble = Block::make(initAssembleStmts);

  guardedTemps = util::toSet(getTemporaries(assemble.getCompute()));
  Stmt compute = lower(assemble.getCompute());

  std::vector<Stmt> finalizeAssembleStmts;
  for (const auto& resultAccess : resultAccesses) {
    Expr prevSize = 1;
    const auto resultIterators = getIterators(resultAccess);
    for (const auto& resultIterator : resultIterators) {
      Stmt finalizeYieldPos = resultIterator.getFinalizeYieldPos(prevSize);
      finalizeAssembleStmts.push_back(finalizeYieldPos);

      prevSize = resultIterator.getAssembledSize(prevSize);
    }
  }
  Stmt finalizeAssemble = Block::make(finalizeAssembleStmts);

  return Block::blanks(queries,
                       initAssemble,
                       compute,
                       finalizeAssemble,
                       freeQueryResults);
}


Stmt LowererImpl::lowerMulti(Multi multi) {
  Stmt stmt1 = lower(multi.getStmt1());
  Stmt stmt2 = lower(multi.getStmt2());
  return Block::make(stmt1, stmt2);
}

Stmt LowererImpl::lowerSuchThat(SuchThat suchThat) {
  auto scalls = suchThat.getCalls();
  this->calls.insert(scalls.begin(), scalls.end());
  Stmt stmt = lower(suchThat.getStmt());
  return Block::make(stmt);
}


Expr LowererImpl::lowerAccess(Access access) {
  if (access.isAccessingStructure()) {
    return true;
  }

  TensorVar var = access.getTensorVar();

  if (isScalar(var.getType())) {
    return getTensorVar(var);
  }

  if (!getIterators(access).back().isUnique()) {
    return getReducedValueVar(access);
  }

  if (var.getType().getDataType() == Bool &&
      getIterators(access).back().isZeroless())  {
    return true;
  } 

  const auto vals = getValuesArray(var);
  if (!vals.defined()) {
    return true;
  }

  return Load::make(vals, generateValueLocExpr(access));
}


Expr LowererImpl::lowerLiteral(Literal literal) {
  switch (literal.getDataType().getKind()) {
    case Datatype::Bool:
      return ir::Literal::make(literal.getVal<bool>());
    case Datatype::UInt8:
      return ir::Literal::make((unsigned long long)literal.getVal<uint8_t>());
    case Datatype::UInt16:
      return ir::Literal::make((unsigned long long)literal.getVal<uint16_t>());
    case Datatype::UInt32:
      return ir::Literal::make((unsigned long long)literal.getVal<uint32_t>());
    case Datatype::UInt64:
      return ir::Literal::make((unsigned long long)literal.getVal<uint64_t>());
    case Datatype::UInt128:
      taco_not_supported_yet;
      break;
    case Datatype::Int8:
      return ir::Literal::make((int)literal.getVal<int8_t>());
    case Datatype::Int16:
      return ir::Literal::make((int)literal.getVal<int16_t>());
    case Datatype::Int32:
      return ir::Literal::make((int)literal.getVal<int32_t>());
    case Datatype::Int64:
      return ir::Literal::make((long long)literal.getVal<int64_t>());
    case Datatype::Int128:
      taco_not_supported_yet;
      break;
    case Datatype::Float32:
      return ir::Literal::make(literal.getVal<float>());
    case Datatype::Float64:
      return ir::Literal::make(literal.getVal<double>());
    case Datatype::Complex64:
      return ir::Literal::make(literal.getVal<std::complex<float>>());
    case Datatype::Complex128:
      return ir::Literal::make(literal.getVal<std::complex<double>>());
    case Datatype::CppType:
      taco_unreachable;
      break;
    case Datatype::Undefined:
      taco_unreachable;
      break;
  }
  return ir::Expr();
}


Expr LowererImpl::lowerNeg(Neg neg) {
  return ir::Neg::make(lower(neg.getA()));
}


Expr LowererImpl::lowerAdd(Add add) {
  Expr a = lower(add.getA());
  Expr b = lower(add.getB());
  return (add.getDataType().getKind() == Datatype::Bool)
         ? ir::Or::make(a, b) : ir::Add::make(a, b);
}


Expr LowererImpl::lowerSub(Sub sub) {
  return ir::Sub::make(lower(sub.getA()), lower(sub.getB()));
}


Expr LowererImpl::lowerMul(Mul mul) {
  Expr a = lower(mul.getA());
  Expr b = lower(mul.getB());
  return (mul.getDataType().getKind() == Datatype::Bool)
         ? ir::And::make(a, b) : ir::Mul::make(a, b);
}


Expr LowererImpl::lowerDiv(Div div) {
  return ir::Div::make(lower(div.getA()), lower(div.getB()));
}


Expr LowererImpl::lowerSqrt(Sqrt sqrt) {
  return ir::Sqrt::make(lower(sqrt.getA()));
}


Expr LowererImpl::lowerCast(Cast cast) {
  return ir::Cast::make(lower(cast.getA()), cast.getDataType());
}


Expr LowererImpl::lowerCallIntrinsic(CallIntrinsic call) {
  std::vector<Expr> args;
  for (auto& arg : call.getArgs()) {
    args.push_back(lower(arg));
  }
  return call.getFunc().lower(args);
}


Stmt LowererImpl::lower(IndexStmt stmt) {
  return visitor->lower(stmt);
}


Expr LowererImpl::lower(IndexExpr expr) {
  return visitor->lower(expr);
}


bool LowererImpl::generateAssembleCode() const {
  return this->assemble;
}


bool LowererImpl::generateComputeCode() const {
  return this->compute;
}


Expr LowererImpl::getTensorVar(TensorVar tensorVar) const {
  taco_iassert(util::contains(this->tensorVars, tensorVar)) << tensorVar;
  return this->tensorVars.at(tensorVar);
}


Expr LowererImpl::getCapacityVar(Expr tensor) const {
  taco_iassert(util::contains(this->capacityVars, tensor)) << tensor;
  return this->capacityVars.at(tensor);
}


ir::Expr LowererImpl::getValuesArray(TensorVar var) const
{
  if (this->legion) {
    // TODO (rohany): Handle temporary arrays at some point.
    // TODO (rohany): Handle reduction accessors at some point.
    // TODO (rohany): Hackingly including the size as the mode here.
    if (util::contains(this->resultTensors, var)) {
      if (this->performingLegionReduction) {
        return GetProperty::make(getTensorVar(var), TensorProperty::ValuesReductionAccessor, var.getOrder());
      }
      return GetProperty::make(getTensorVar(var), TensorProperty::ValuesWriteAccessor, var.getOrder());
    } else {
      return GetProperty::make(getTensorVar(var), TensorProperty::ValuesReadAccessor, var.getOrder());
    }
  } else {
    return (util::contains(temporaryArrays, var))
           ? temporaryArrays.at(var).values
           : GetProperty::make(getTensorVar(var), TensorProperty::Values);
  }
}


Expr LowererImpl::getDimension(IndexVar indexVar) const {
  taco_iassert(util::contains(this->dimensions, indexVar)) << indexVar;
  return this->dimensions.at(indexVar);
}


std::vector<Iterator> LowererImpl::getIterators(Access access) const {
  vector<Iterator> result;
  TensorVar tensor = access.getTensorVar();
  for (int i = 0; i < tensor.getOrder(); i++) {
    int mode = tensor.getFormat().getModeOrdering()[i];
    result.push_back(iterators.levelIterator(ModeAccess(access, mode+1)));
  }
  return result;
}


set<Access> LowererImpl::getExhaustedAccesses(MergePoint point,
                                              MergeLattice lattice) const
{
  set<Access> exhaustedAccesses;
  for (auto& iterator : lattice.exhausted(point)) {
    exhaustedAccesses.insert(iterators.modeAccess(iterator).getAccess());
  }
  return exhaustedAccesses;
}


Expr LowererImpl::getReducedValueVar(Access access) const {
  return this->reducedValueVars.at(access);
}


Expr LowererImpl::getCoordinateVar(IndexVar indexVar) const {
  return this->iterators.modeIterator(indexVar).getCoordVar();
}


Expr LowererImpl::getCoordinateVar(Iterator iterator) const {
  if (iterator.isDimensionIterator()) {
    return iterator.getCoordVar();
  }
  return this->getCoordinateVar(iterator.getIndexVar());
}


vector<Expr> LowererImpl::coordinates(Iterator iterator) const {
  taco_iassert(iterator.defined());

  vector<Expr> coords;
  do {
    coords.push_back(getCoordinateVar(iterator));
    iterator = iterator.getParent();
  } while (!iterator.isRoot());
  auto reverse = util::reverse(coords);
  return vector<Expr>(reverse.begin(), reverse.end());
}

vector<Expr> LowererImpl::coordinates(vector<Iterator> iterators)
{
  taco_iassert(all(iterators, [](Iterator iter){ return iter.defined(); }));
  vector<Expr> result;
  for (auto& iterator : iterators) {
    result.push_back(iterator.getCoordVar());
  }
  return result;
}


Stmt LowererImpl::initResultArrays(vector<Access> writes,
                                   vector<Access> reads,
                                   set<Access> reducedAccesses) {
  multimap<IndexVar, Iterator> readIterators;
  for (auto& read : reads) {
    for (auto& readIterator : getIterators(read)) {
      for (auto& underivedAncestor : provGraph.getUnderivedAncestors(readIterator.getIndexVar())) {
        readIterators.insert({underivedAncestor, readIterator});
      }
    }
  }

  std::vector<Stmt> result;
  for (auto& write : writes) {
    if (write.getTensorVar().getOrder() == 0 ||
        isAssembledByUngroupedInsertion(write.getTensorVar())) {
      continue;
    }

    std::vector<Stmt> initArrays;

    const auto iterators = getIterators(write);
    taco_iassert(!iterators.empty());

    Expr tensor = getTensorVar(write.getTensorVar());
    Expr valuesArr = GetProperty::make(tensor, TensorProperty::Values);
    bool clearValuesAllocation = false;

    Expr parentSize = 1;
    if (generateAssembleCode()) {
      for (const auto& iterator : iterators) {
        Expr size;
        Stmt init;
        // Initialize data structures for storing levels
        if (iterator.hasAppend()) {
          size = 0;
          init = iterator.getAppendInitLevel(parentSize, size);
        } else if (iterator.hasInsert()) {
          size = simplify(ir::Mul::make(parentSize, iterator.getWidth()));
          init = iterator.getInsertInitLevel(parentSize, size);
        } else {
          taco_ierror << "Write iterator supports neither append nor insert";
        }
        initArrays.push_back(init);

        // Declare position variable of append modes that are not above a
        // branchless mode (if mode below is branchless, then can share same
        // position variable)
        if (iterator.hasAppend() && (iterator.isLeaf() ||
            !iterator.getChild().isBranchless())) {
          initArrays.push_back(VarDecl::make(iterator.getPosVar(), 0));
        }

        parentSize = size;
        // Writes into a windowed iterator require the allocation to be cleared.
        clearValuesAllocation |= (iterator.isWindowed() || iterator.hasIndexSet());
      }

      // Pre-allocate memory for the value array if computing while assembling
      if (generateComputeCode()) {
        taco_iassert(!iterators.empty());

        Expr capacityVar = getCapacityVar(tensor);
        Expr allocSize = isValue(parentSize, 0)
                         ? DEFAULT_ALLOC_SIZE : parentSize;
        initArrays.push_back(VarDecl::make(capacityVar, allocSize));
        initArrays.push_back(Allocate::make(valuesArr, capacityVar, false /* is_realloc */, Expr() /* old_elements */,
                                            clearValuesAllocation));
      }

      taco_iassert(!initArrays.empty());
      result.push_back(Block::make(initArrays));
    }
    else if (generateComputeCode()) {
      Iterator lastAppendIterator;
      // Compute size of values array
      for (auto& iterator : iterators) {
        if (iterator.hasAppend()) {
          lastAppendIterator = iterator;
          parentSize = iterator.getSize(parentSize);
        } else if (iterator.hasInsert()) {
          parentSize = ir::Mul::make(parentSize, iterator.getWidth());
        } else {
          taco_ierror << "Write iterator supports neither append nor insert";
        }
        parentSize = simplify(parentSize);
      }

      // Declare position variable for the last append level
      if (lastAppendIterator.defined()) {
        result.push_back(VarDecl::make(lastAppendIterator.getPosVar(), 0));
      }
    }

    if (generateComputeCode() && iterators.back().hasInsert() &&
        !isValue(parentSize, 0) &&
        (hasSparseInserts(iterators, readIterators) ||
         util::contains(reducedAccesses, write))) {
      // Zero-initialize values array if size statically known and might not
      // assign to every element in values array during compute
      // TODO: Right now for scheduled code we check if any iterator is not full and then emit
      // a zero-initialization loop. We only actually need a zero-initialization loop if the combined
      // iteration of all the iterators is not full. We can check this by seeing if we can recover a
      // full iterator from our set of iterators.
      Expr size = generateAssembleCode() ? getCapacityVar(tensor) : parentSize;
      result.push_back(zeroInitValues(tensor, 0, size));
    }
  }
  return result.empty() ? Stmt() : Block::blanks(result);
}


ir::Stmt LowererImpl::finalizeResultArrays(std::vector<Access> writes) {
  if (!generateAssembleCode()) {
    return Stmt();
  }

  bool clearValuesAllocation = false;
  std::vector<Stmt> result;
  for (auto& write : writes) {
    if (write.getTensorVar().getOrder() == 0 ||
        isAssembledByUngroupedInsertion(write.getTensorVar())) {
      continue;
    }

    const auto iterators = getIterators(write);
    taco_iassert(!iterators.empty());

    Expr parentSize = 1;
    for (const auto& iterator : iterators) {
      Expr size;
      Stmt finalize;
      // Post-process data structures for storing levels
      if (iterator.hasAppend()) {
        size = iterator.getPosVar();
        finalize = iterator.getAppendFinalizeLevel(parentSize, size);
      } else if (iterator.hasInsert()) {
        size = simplify(ir::Mul::make(parentSize, iterator.getWidth()));
        finalize = iterator.getInsertFinalizeLevel(parentSize, size);
      } else {
        taco_ierror << "Write iterator supports neither append nor insert";
      }
      result.push_back(finalize);
      parentSize = size;
      // Writes into a windowed iterator require the allocation to be cleared.
      clearValuesAllocation |= (iterator.isWindowed() || iterator.hasIndexSet());
    }

    if (!generateComputeCode()) {
      // Allocate memory for values array after assembly if not also computing
      Expr tensor = getTensorVar(write.getTensorVar());
      Expr valuesArr = GetProperty::make(tensor, TensorProperty::Values);
      result.push_back(Allocate::make(valuesArr, parentSize, false /* is_realloc */, Expr() /* old_elements */,
                                      clearValuesAllocation));
    }
  }
  return result.empty() ? Stmt() : Block::blanks(result);
}

Stmt LowererImpl::defineScalarVariable(TensorVar var, bool zero) {
  Datatype type = var.getType().getDataType();
  Expr varValueIR = Var::make(var.getName() + "_val", type, false, false);
  Expr init = (zero) ? ir::Literal::zero(type)
                     : Load::make(GetProperty::make(tensorVars.at(var),
                                                    TensorProperty::Values));
  tensorVars.find(var)->second = varValueIR;
  return VarDecl::make(varValueIR, init);
}

static
vector<Iterator> getIteratorsFrom(IndexVar var,
                                  const vector<Iterator>& iterators) {
  vector<Iterator> result;
  bool found = false;
  for (Iterator iterator : iterators) {
    if (var == iterator.getIndexVar()) found = true;
    if (found) {
      result.push_back(iterator);
    }
  }
  return result;
}


Stmt LowererImpl::initResultArrays(IndexVar var, vector<Access> writes,
                                   vector<Access> reads,
                                   set<Access> reducedAccesses) {
  if (!generateAssembleCode()) {
    return Stmt();
  }

  multimap<IndexVar, Iterator> readIterators;
  for (auto& read : reads) {
    for (auto& readIterator : getIteratorsFrom(var, getIterators(read))) {
      for (auto& underivedAncestor : provGraph.getUnderivedAncestors(readIterator.getIndexVar())) {
        readIterators.insert({underivedAncestor, readIterator});
      }
    }
  }

  vector<Stmt> result;
  for (auto& write : writes) {
    Expr tensor = getTensorVar(write.getTensorVar());
    Expr values = GetProperty::make(tensor, TensorProperty::Values);

    vector<Iterator> iterators = getIteratorsFrom(var, getIterators(write));

    if (iterators.empty()) {
      continue;
    }

    Iterator resultIterator = iterators.front();

    // Initialize begin var
    if (resultIterator.hasAppend() && !resultIterator.isBranchless()) {
      Expr begin = resultIterator.getBeginVar();
      result.push_back(VarDecl::make(begin, resultIterator.getPosVar()));
    }

    const bool isTopLevel = (iterators.size() == write.getIndexVars().size());
    if (resultIterator.getParent().hasAppend() || isTopLevel) {
      Expr resultParentPos = resultIterator.getParent().getPosVar();
      Expr resultParentPosNext = simplify(ir::Add::make(resultParentPos, 1));
      Expr initBegin = resultParentPos;
      Expr initEnd = resultParentPosNext;
      Expr stride = 1;

      Iterator initIterator;
      for (Iterator iterator : iterators) {
        if (!iterator.hasInsert()) {
          initIterator = iterator;
          break;
        }

        stride = simplify(ir::Mul::make(stride, iterator.getWidth()));
        initBegin = simplify(ir::Mul::make(resultParentPos, stride));
        initEnd = simplify(ir::Mul::make(resultParentPosNext, stride));

        // Initialize data structures for storing insert mode
        result.push_back(iterator.getInsertInitCoords(initBegin, initEnd));
      }

      if (initIterator.defined()) {
        // Initialize data structures for storing edges of next append mode
        taco_iassert(initIterator.hasAppend());
        result.push_back(initIterator.getAppendInitEdges(initBegin, initEnd));
      } else if (generateComputeCode() && !isTopLevel) {
        if (isa<ir::Mul>(stride)) {
          Expr strideVar = Var::make(util::toString(tensor) + "_stride", Int());
          result.push_back(VarDecl::make(strideVar, stride));
          stride = strideVar;
        }

        // Resize values array if not large enough
        Expr capacityVar = getCapacityVar(tensor);
        Expr size = simplify(ir::Mul::make(resultParentPosNext, stride));
        result.push_back(atLeastDoubleSizeIfFull(values, capacityVar, size));

        if (hasSparseInserts(iterators, readIterators) ||
            util::contains(reducedAccesses, write)) {
          // Zero-initialize values array if might not assign to every element
          // in values array during compute
          result.push_back(zeroInitValues(tensor, resultParentPos, stride));
        }
      }
    }
  }
  return result.empty() ? Stmt() : Block::make(result);
}


Stmt LowererImpl::resizeAndInitValues(const std::vector<Iterator>& appenders,
                                      const std::set<Access>& reducedAccesses) {
  if (!generateComputeCode()) {
    return Stmt();
  }

  std::function<Expr(Access)> getTensor = [&](Access access) {
    return getTensorVar(access.getTensorVar());
  };
  const auto reducedTensors = util::map(reducedAccesses, getTensor);

  std::vector<Stmt> result;

  for (auto& appender : appenders) {
    if (!appender.isLeaf()) {
      continue;
    }

    Expr tensor = appender.getTensor();
    Expr values = GetProperty::make(tensor, TensorProperty::Values);
    Expr capacity = getCapacityVar(appender.getTensor());
    Expr pos = appender.getIteratorVar();

    if (generateAssembleCode()) {
      result.push_back(doubleSizeIfFull(values, capacity, pos));
    }

    if (util::contains(reducedTensors, tensor)) {
      Expr zero = ir::Literal::zero(tensor.type());
      result.push_back(Store::make(values, pos, zero));
    }
  }

  return result.empty() ? Stmt() : Block::make(result);
}


Stmt LowererImpl::zeroInitValues(Expr tensor, Expr begin, Expr size) {
  Expr lower = simplify(ir::Mul::make(begin, size));
  Expr upper = simplify(ir::Mul::make(ir::Add::make(begin, 1), size));
  Expr p = Var::make("p" + util::toString(tensor), Int());
  Expr values = GetProperty::make(tensor, TensorProperty::Values);
  Stmt zeroInit = Store::make(values, p, ir::Literal::zero(tensor.type()));
  LoopKind parallel = (isa<ir::Literal>(size) &&
                       to<ir::Literal>(size)->getIntValue() < (1 << 10))
                      ? LoopKind::Serial : LoopKind::Static_Chunked;
  if (should_use_CUDA_codegen() && util::contains(parallelUnitSizes, ParallelUnit::GPUBlock)) {
    return ir::VarDecl::make(ir::Var::make("status", Int()),
                                    ir::Call::make("cudaMemset", {values, ir::Literal::make(0, Int()), ir::Mul::make(ir::Sub::make(upper, lower), ir::Literal::make(values.type().getNumBytes()))}, Int()));
  }
  return For::make(p, lower, upper, 1, zeroInit, parallel);
}

std::vector<IndexVar> getIndexVarFamily(const Iterator& it) {
  if (it.isRoot() || it.getMode().getLevel() == 1) {
    return {it.getIndexVar()};
  }
//  std::vector<IndexVar> result;
  auto rcall = getIndexVarFamily(it.getParent());
  rcall.push_back(it.getIndexVar());
  return rcall;
}

Stmt LowererImpl::declLocatePosVars(vector<Iterator> locators) {
  vector<Stmt> result;
  // We first collect pointAccesses in a buffer to ensure that we always emit them in
  // the same order.
  vector<std::pair<TensorVar, Stmt>> pointAccesses;
  for (Iterator& locator : locators) {
    accessibleIterators.insert(locator);

    bool doLocate = true;
    for (Iterator ancestorIterator = locator.getParent();
         !ancestorIterator.isRoot() && ancestorIterator.hasLocate();
         ancestorIterator = ancestorIterator.getParent()) {
      if (!accessibleIterators.contains(ancestorIterator)) {
        doLocate = false;
      }
    }

    if (doLocate) {
      Iterator locateIterator = locator;
      if (locateIterator.hasPosIter()) {
        taco_iassert(!provGraph.isUnderived(locateIterator.getIndexVar()));
        continue; // these will be recovered with separate procedure
      }
      do {
        auto coords = coordinates(locateIterator);
        // If this dimension iterator operates over a window, then it needs
        // to be projected up to the window's iteration space.
        if (locateIterator.isWindowed()) {
          auto expr = coords[coords.size() - 1];
          coords[coords.size() - 1] = this->projectCanonicalSpaceToWindowedPosition(locateIterator, expr);
        } else if (locateIterator.hasIndexSet()) {
          // If this dimension iterator operates over an index set, follow the
          // indirection by using the locator access the index set's crd array.
          // The resulting value is where we should locate into the actual tensor.
          auto expr = coords[coords.size() - 1];
          auto indexSetIterator = locateIterator.getIndexSetIterator();
          auto coordArray = indexSetIterator.posAccess(expr, coordinates(indexSetIterator)).getResults()[0];
          coords[coords.size() - 1] = coordArray;
        }
        ModeFunction locate = locateIterator.locate(coords);
        taco_iassert(isValue(locate.getResults()[1], true));
        Stmt declarePosVar = VarDecl::make(locateIterator.getPosVar(),
                                           locate.getResults()[0]);
        result.push_back(declarePosVar);

        if (locateIterator.isLeaf()) {
          if (this->legion) {
            // Emit the point accessor for it here.
            // TODO (rohany): Use a reverse map.
            TensorVar tv;
            for (auto& it : this->tensorVars) {
              if (it.second == locateIterator.getTensor()) {
                tv = it.first;
              }
            }
            if (!util::contains(this->emittedPointAccessors, tv)) {
              auto ivars = getIndexVarFamily(locateIterator);
              auto pointT = Point(ivars.size());
              auto point = this->pointAccessVars[tv];
              std::function<Expr(IndexVar)> getExpr = [&](IndexVar i) {
                return this->indexVarToExprMap.at(i);
              };
              auto makePoint = makeConstructor(pointT, util::map(ivars, getExpr));
              pointAccesses.push_back(std::make_pair(tv, ir::VarDecl::make(point, makePoint)));
              this->emittedPointAccessors.insert(tv);
            }
          }
          break;
        }
        locateIterator = locateIterator.getChild();
      } while (accessibleIterators.contains(locateIterator));
    }
  }
  // Sort the emitted point accesses at this level.
  struct TensorVarSorter {
    typedef std::pair<TensorVar, Stmt> elem;
    bool operator() (elem e1, elem e2) {
      return e1.first.getName() < e2.first.getName();
    }
  } tvs;
  std::sort(pointAccesses.begin(), pointAccesses.end(), tvs);
  for (auto it : pointAccesses) {
    result.push_back(it.second);
  }
  return result.empty() ? Stmt() : Block::make(result);
}


Stmt LowererImpl::reduceDuplicateCoordinates(Expr coordinate,
                                             vector<Iterator> iterators,
                                             bool alwaysReduce) {
  vector<Stmt> result;
  for (Iterator& iterator : iterators) {
    taco_iassert(!iterator.isUnique() && iterator.hasPosIter());

    Access access = this->iterators.modeAccess(iterator).getAccess();
    Expr iterVar = iterator.getIteratorVar();
    Expr segendVar = iterator.getSegendVar();
    Expr reducedVal = iterator.isLeaf() ? getReducedValueVar(access) : Expr();
    Expr tensorVar = getTensorVar(access.getTensorVar());
    Expr tensorVals = GetProperty::make(tensorVar, TensorProperty::Values);

    // Initialize variable storing reduced component value.
    if (reducedVal.defined()) {
      Expr reducedValInit = alwaysReduce
                          ? Load::make(tensorVals, iterVar)
                          : ir::Literal::zero(reducedVal.type());
      result.push_back(VarDecl::make(reducedVal, reducedValInit));
    }

    if (iterator.isLeaf()) {
      // If iterator is over bottommost coordinate hierarchy level and will
      // always advance (i.e., not merging with another iterator), then we don't
      // need a separate segend variable.
      segendVar = iterVar;
      if (alwaysReduce) {
        result.push_back(compoundAssign(segendVar, 1));
      }
    } else {
      Expr segendInit = alwaysReduce ? ir::Add::make(iterVar, 1) : iterVar;
      result.push_back(VarDecl::make(segendVar, segendInit));
    }

    vector<Stmt> dedupStmts;
    if (reducedVal.defined()) {
      Expr partialVal = Load::make(tensorVals, segendVar);
      dedupStmts.push_back(compoundAssign(reducedVal, partialVal));
    }
    dedupStmts.push_back(compoundAssign(segendVar, 1));
    Stmt dedupBody = Block::make(dedupStmts);

    ModeFunction posAccess = iterator.posAccess(segendVar,
                                                coordinates(iterator));
    // TODO: Support access functions that perform additional computations
    //       and/or might access invalid positions.
    taco_iassert(!posAccess.compute().defined());
    taco_iassert(to<ir::Literal>(posAccess.getResults()[1])->getBoolValue());
    Expr nextCoord = posAccess.getResults()[0];
    Expr withinBounds = Lt::make(segendVar, iterator.getEndVar());
    Expr isDuplicate = Eq::make(posAccess.getResults()[0], coordinate);
    result.push_back(While::make(And::make(withinBounds, isDuplicate),
                                 Block::make(dedupStmts)));
  }
  return result.empty() ? Stmt() : Block::make(result);
}

Stmt LowererImpl::codeToInitializeIteratorVar(Iterator iterator, vector<Iterator> iterators, vector<Iterator> rangers, vector<Iterator> mergers, Expr coordinate, IndexVar coordinateVar) {
  vector<Stmt> result;
  taco_iassert(iterator.hasPosIter() || iterator.hasCoordIter() ||
               iterator.isDimensionIterator());

  Expr iterVar = iterator.getIteratorVar();
  Expr endVar = iterator.getEndVar();
  if (iterator.hasPosIter()) {
    Expr parentPos = iterator.getParent().getPosVar();
    if (iterator.getParent().isRoot() || iterator.getParent().isUnique()) {
      // E.g. a compressed mode without duplicates
      ModeFunction bounds = iterator.posBounds(parentPos);
      result.push_back(bounds.compute());
      // if has a coordinate ranger then need to binary search
      if (any(rangers,
              [](Iterator it){ return it.isDimensionIterator(); })) {

        Expr binarySearchTarget = provGraph.deriveCoordBounds(definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, this->iterators)[coordinateVar][0];
        if (binarySearchTarget != underivedBounds[coordinateVar][0]) {
          // If we have a window, then we need to project up the binary search target
          // into the window rather than the beginning of the level.
          if (iterator.isWindowed()) {
            binarySearchTarget = this->projectCanonicalSpaceToWindowedPosition(iterator, binarySearchTarget);
          }
          result.push_back(VarDecl::make(iterator.getBeginVar(), binarySearchTarget));

          vector<Expr> binarySearchArgs = {
                  iterator.getMode().getModePack().getArray(1), // array
                  bounds[0], // arrayStart
                  bounds[1], // arrayEnd
                  iterator.getBeginVar() // target
          };
          result.push_back(
                  VarDecl::make(iterVar, Call::make("taco_binarySearchAfter", binarySearchArgs, iterVar.type())));
        }
        else {
          result.push_back(VarDecl::make(iterVar, bounds[0]));
        }
      }
      else {
        auto bound = bounds[0];
        // If we have a window on this iterator, then search for the start of
        // the window rather than starting at the beginning of the level.
        if (iterator.isWindowed()) {
            bound = this->searchForStartOfWindowPosition(iterator, bounds[0], bounds[1]);
        }
        result.push_back(VarDecl::make(iterVar, bound));
      }

      result.push_back(VarDecl::make(endVar, bounds[1]));
    } else {
      taco_iassert(iterator.isOrdered() && iterator.getParent().isOrdered());
      taco_iassert(iterator.isCompact() && iterator.getParent().isCompact());

      // E.g. a compressed mode with duplicates. Apply iterator chaining
      Expr parentSegend = iterator.getParent().getSegendVar();
      ModeFunction startBounds = iterator.posBounds(parentPos);
      ModeFunction endBounds = iterator.posBounds(ir::Sub::make(parentSegend, 1));
      result.push_back(startBounds.compute());
      result.push_back(VarDecl::make(iterVar, startBounds[0]));
      result.push_back(endBounds.compute());
      result.push_back(VarDecl::make(endVar, endBounds[1]));
    }
  }
  else if (iterator.hasCoordIter()) {
    // E.g. a hasmap mode
    vector<Expr> coords = coordinates(iterator);
    coords.erase(coords.begin());
    ModeFunction bounds = iterator.coordBounds(coords);
    result.push_back(bounds.compute());
    result.push_back(VarDecl::make(iterVar, bounds[0]));
    result.push_back(VarDecl::make(endVar, bounds[1]));
  }
  else if (iterator.isDimensionIterator()) {
    // A dimension
    // If a merger then initialize to 0
    // If not then get first coord value like doing normal merge

    // If derived then need to recoverchild from this coord value
    bool isMerger = find(mergers.begin(), mergers.end(), iterator) != mergers.end();
    if (isMerger) {
      Expr coord = coordinates(vector<Iterator>({iterator}))[0];
      result.push_back(VarDecl::make(coord, 0));
    }
    else {
      result.push_back(codeToLoadCoordinatesFromPosIterators(iterators, true));

      Stmt stmt = resolveCoordinate(mergers, coordinate, true);
      taco_iassert(stmt != Stmt());
      result.push_back(stmt);
      result.push_back(codeToRecoverDerivedIndexVar(coordinateVar, iterator.getIndexVar(), true));

      // emit bound for ranger too
      vector<Expr> startBounds;
      vector<Expr> endBounds;
      for (Iterator merger : mergers) {
        ModeFunction coordBounds = merger.coordBounds(merger.getParent().getPosVar());
        startBounds.push_back(coordBounds[0]);
        endBounds.push_back(coordBounds[1]);
      }
      //TODO: maybe needed after split reorder? underivedBounds[coordinateVar] = {ir::Max::make(startBounds), ir::Min::make(endBounds)};
      Stmt end_decl = VarDecl::make(iterator.getEndVar(), provGraph.deriveIterBounds(iterator.getIndexVar(), definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, this->iterators)[1]);
      result.push_back(end_decl);
    }
  }
  return result.empty() ? Stmt() : Block::make(result);
}

Stmt LowererImpl::codeToInitializeIteratorVars(vector<Iterator> iterators, vector<Iterator> rangers, vector<Iterator> mergers, Expr coordinate, IndexVar coordinateVar) {
  vector<Stmt> results;
  // initialize mergers first (can't depend on initializing rangers)
  for (Iterator iterator : mergers) {
    results.push_back(codeToInitializeIteratorVar(iterator, iterators, rangers, mergers, coordinate, coordinateVar));
  }

  for (Iterator iterator : rangers) {
      if (find(mergers.begin(), mergers.end(), iterator) == mergers.end()) {
        results.push_back(codeToInitializeIteratorVar(iterator, iterators, rangers, mergers, coordinate, coordinateVar));
      }
  }
  return results.empty() ? Stmt() : Block::make(results);
}

Stmt LowererImpl::codeToRecoverDerivedIndexVar(IndexVar underived, IndexVar indexVar, bool emitVarDecl) {
  if(underived != indexVar) {
    // iterator indexVar must be derived from coordinateVar
    std::vector<IndexVar> underivedAncestors = provGraph.getUnderivedAncestors(indexVar);
    taco_iassert(find(underivedAncestors.begin(), underivedAncestors.end(), underived) != underivedAncestors.end());

    vector<Stmt> recoverySteps;
    for (const IndexVar& varToRecover : provGraph.derivationPath(underived, indexVar)) {
      if(varToRecover == underived) continue;
      recoverySteps.push_back(provGraph.recoverChild(varToRecover, indexVarToExprMap, emitVarDecl, iterators));
    }
    return Block::make(recoverySteps);
  }
  return Stmt();
}

Stmt LowererImpl::codeToIncIteratorVars(Expr coordinate, IndexVar coordinateVar, vector<Iterator> iterators, vector<Iterator> mergers) {
  if (iterators.size() == 1) {
    Expr ivar = iterators[0].getIteratorVar();

    if (iterators[0].isUnique()) {
      return compoundAssign(ivar, 1);
    }

    // If iterator is over bottommost coordinate hierarchy level with
    // duplicates and iterator will always advance (i.e., not merging with
    // another iterator), then deduplication loop will take care of
    // incrementing iterator variable.
    return iterators[0].isLeaf()
           ? Stmt()
           : Assign::make(ivar, iterators[0].getSegendVar());
  }

  vector<Stmt> result;

  // We emit the level iterators before the mode iterator because the coordinate
  // of the mode iterator is used to conditionally advance the level iterators.

  auto levelIterators =
      filter(iterators, [](Iterator it){return !it.isDimensionIterator();});
  for (auto& iterator : levelIterators) {
    Expr ivar = iterator.getIteratorVar();
    if (iterator.isUnique()) {
      Expr increment = iterator.isFull()
                     ? 1
                     : ir::Cast::make(Eq::make(iterator.getCoordVar(),
                                               coordinate),
                                      ivar.type());
      result.push_back(compoundAssign(ivar, increment));
    } else if (!iterator.isLeaf()) {
      result.push_back(Assign::make(ivar, iterator.getSegendVar()));
    }
  }

  auto modeIterators =
      filter(iterators, [](Iterator it){return it.isDimensionIterator();});
  for (auto& iterator : modeIterators) {
    bool isMerger = find(mergers.begin(), mergers.end(), iterator) != mergers.end();
    if (isMerger) {
      Expr ivar = iterator.getIteratorVar();
      result.push_back(compoundAssign(ivar, 1));
    }
    else {
      result.push_back(codeToLoadCoordinatesFromPosIterators(iterators, false));
      Stmt stmt = resolveCoordinate(mergers, coordinate, false);
      taco_iassert(stmt != Stmt());
      result.push_back(stmt);
      result.push_back(codeToRecoverDerivedIndexVar(coordinateVar, iterator.getIndexVar(), false));
    }
  }

  return Block::make(result);
}

Stmt LowererImpl::codeToLoadCoordinatesFromPosIterators(vector<Iterator> iterators, bool declVars) {
  // Load coordinates from position iterators
  Stmt loadPosIterCoordinates;
  if (iterators.size() > 1) {
    vector<Stmt> loadPosIterCoordinateStmts;
    auto posIters = filter(iterators, [](Iterator it){return it.hasPosIter();});
    for (auto& posIter : posIters) {
      taco_tassert(posIter.hasPosIter());
      ModeFunction posAccess = posIter.posAccess(posIter.getPosVar(),
                                                 coordinates(posIter));
      loadPosIterCoordinateStmts.push_back(posAccess.compute());
      auto access = posAccess[0];
      // If this iterator is windowed, then it needs to be projected down to
      // recover the coordinate variable.
      // TODO (rohany): Would be cleaner to have this logic be moved into the
      //  ModeFunction, rather than having to check in some places?
      if (posIter.isWindowed()) {

        // If the iterator is strided, then we have to skip over coordinates
        // that don't match the stride. To do that, we insert a guard on the
        // access. We first extract the access into a temp to avoid emitting
        // a duplicate load on the _crd array.
        if (posIter.isStrided()) {
          loadPosIterCoordinateStmts.push_back(VarDecl::make(posIter.getWindowVar(), access));
          access = posIter.getWindowVar();
          // Since we're locating into a compressed array (not iterating over it),
          // we need to advance the outer loop if the current coordinate is not
          // along the desired stride. So, we pass true to the incrementPosVar
          // argument of strideBoundsGuard.
          loadPosIterCoordinateStmts.push_back(this->strideBoundsGuard(posIter, access, true /* incrementPosVar */));
        }

        access = this->projectWindowedPositionToCanonicalSpace(posIter, access);
      }
      if (declVars) {
        loadPosIterCoordinateStmts.push_back(VarDecl::make(posIter.getCoordVar(), access));
      }
      else {
        loadPosIterCoordinateStmts.push_back(Assign::make(posIter.getCoordVar(), access));
      }
      if (posIter.isWindowed()) {
        loadPosIterCoordinateStmts.push_back(this->upperBoundGuardForWindowPosition(posIter, posIter.getCoordVar()));
      }
    }
    loadPosIterCoordinates = Block::make(loadPosIterCoordinateStmts);
  }
  return loadPosIterCoordinates;
}


static
bool isLastAppender(Iterator iter) {
  taco_iassert(iter.hasAppend());
  while (!iter.isLeaf()) {
    iter = iter.getChild();
    if (iter.hasAppend()) {
      return false;
    }
  }
  return true;
}


Stmt LowererImpl::appendCoordinate(vector<Iterator> appenders, Expr coord) {
  vector<Stmt> result;
  for (auto& appender : appenders) {
    Expr pos = appender.getPosVar();
    Iterator appenderChild = appender.getChild();

    if (appenderChild.defined() && appenderChild.isBranchless()) {
      // Already emitted assembly code for current level when handling
      // branchless child level, so don't emit code again.
      continue;
    }

    vector<Stmt> appendStmts;

    if (generateAssembleCode()) {
      appendStmts.push_back(appender.getAppendCoord(pos, coord));
      while (!appender.isRoot() && appender.isBranchless()) {
        // Need to append result coordinate to parent level as well if child
        // level is branchless (so child coordinates will have unique parents).
        appender = appender.getParent();
        if (!appender.isRoot()) {
          taco_iassert(appender.hasAppend()) << "Parent level of branchless, "
              << "append-capable level must also be append-capable";
          taco_iassert(!appender.isUnique()) << "Need to be able to insert "
              << "duplicate coordinates to level, but level is declared unique";

          Expr coord = getCoordinateVar(appender);
          appendStmts.push_back(appender.getAppendCoord(pos, coord));
        }
      }
    }

    if (generateAssembleCode() || isLastAppender(appender)) {
      appendStmts.push_back(compoundAssign(pos, 1));

      Stmt appendCode = Block::make(appendStmts);
      if (appenderChild.defined() && appenderChild.hasAppend()) {
        // Emit guard to avoid appending empty slices to result.
        // TODO: Users should be able to configure whether to append zeroes.
        Expr shouldAppend = Lt::make(appenderChild.getBeginVar(),
                                     appenderChild.getPosVar());
        appendCode = IfThenElse::make(shouldAppend, appendCode);
      }
      result.push_back(appendCode);
    }
  }
  return result.empty() ? Stmt() : Block::make(result);
}


Stmt LowererImpl::generateAppendPositions(vector<Iterator> appenders) {
  vector<Stmt> result;
  if (generateAssembleCode()) {
    for (Iterator appender : appenders) {
      if (appender.isBranchless() || 
          isAssembledByUngroupedInsertion(appender.getTensor())) {
        continue;
      }

      Expr pos = [](Iterator appender) {
        // Get the position variable associated with the appender. If a mode
        // is above a branchless mode, then the two modes can share the same
        // position variable.
        while (!appender.isLeaf() && appender.getChild().isBranchless()) {
          appender = appender.getChild();
        }
        return appender.getPosVar();
      }(appender);
      Expr beginPos = appender.getBeginVar();
      Expr parentPos = appender.getParent().getPosVar();
      result.push_back(appender.getAppendEdges(parentPos, beginPos, pos));
    }
  }
  return result.empty() ? Stmt() : Block::make(result);
}


Expr LowererImpl::generateValueLocExpr(Access access) const {
  if (isScalar(access.getTensorVar().getType())) {
    return ir::Literal::make(0);
  }
  // If using legion, return the PointT<...> accessor.
  if (this->legion) {
    return this->pointAccessVars.at(access.getTensorVar());
  }

  Iterator it = getIterators(access).back();

  // to make indexing temporary arrays with index var work correctly
  if (!provGraph.isUnderived(it.getIndexVar()) && !access.getIndexVars().empty() &&
      util::contains(indexVarToExprMap, access.getIndexVars().front()) &&
      !it.hasPosIter() && access.getIndexVars().front() == it.getIndexVar()) {
    return indexVarToExprMap.at(access.getIndexVars().front());
  }

  return it.getPosVar();
}


Expr LowererImpl::checkThatNoneAreExhausted(std::vector<Iterator> iterators)
{
  taco_iassert(!iterators.empty());
  if (iterators.size() == 1 && iterators[0].isFull()) {
    std::vector<ir::Expr> bounds = provGraph.deriveIterBounds(iterators[0].getIndexVar(), definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, this->iterators);
    Expr guards = Lt::make(iterators[0].getIteratorVar(), bounds[1]);
    if (bounds[0] != ir::Literal::make(0)) {
      guards = And::make(guards, Gte::make(iterators[0].getIteratorVar(), bounds[0]));
    }
    return guards;
  }

  vector<Expr> result;
  for (const auto& iterator : iterators) {
    taco_iassert(!iterator.isFull()) << iterator
        << " - full iterators do not need to partake in merge loop bounds";
    Expr iterUnexhausted = Lt::make(iterator.getIteratorVar(),
                                    iterator.getEndVar());
    result.push_back(iterUnexhausted);
  }

  return (!result.empty())
         ? taco::ir::conjunction(result)
         : Lt::make(iterators[0].getIteratorVar(), iterators[0].getEndVar());
}


Expr LowererImpl::generateAssembleGuard(IndexExpr expr) {
  class GenerateGuard : public IndexExprVisitorStrict {
  public:
    GenerateGuard(const std::set<TensorVar>& guardedTemps,
                  const std::map<TensorVar,Expr>& tempToGuard)
        : guardedTemps(guardedTemps), tempToGuard(tempToGuard) {}

    Expr lower(IndexExpr expr) {
      this->expr = Expr();
      IndexExprVisitorStrict::visit(expr);
      return this->expr;
    }

  private:
    Expr expr;
    const std::set<TensorVar>& guardedTemps;
    const std::map<TensorVar,Expr>& tempToGuard;

    using IndexExprVisitorStrict::visit;

    void visit(const AccessNode* node) {
      expr = (util::contains(guardedTemps, node->tensorVar) &&
              node->tensorVar.getOrder() == 0)
             ? tempToGuard.at(node->tensorVar) : true;
    }

    void visit(const LiteralNode* node) {
      expr = true;
    }

    void visit(const NegNode* node) {
      expr = lower(node->a);
    }

    void visit(const AddNode* node) {
      expr = Or::make(lower(node->a), lower(node->b));
    }

    void visit(const SubNode* node) {
      expr = Or::make(lower(node->a), lower(node->b));
    }

    void visit(const MulNode* node) {
      expr = And::make(lower(node->a), lower(node->b));
    }

    void visit(const DivNode* node) {
      expr = And::make(lower(node->a), lower(node->b));
    }

    void visit(const SqrtNode* node) {
      expr = lower(node->a);
    }

    void visit(const CastNode* node) {
      expr = lower(node->a);
    }

    void visit(const CallIntrinsicNode* node) {
      Expr ret = false;
      for (const auto& arg : node->args) {
        ret = Or::make(ret, lower(arg));
      }
      expr = ret;
    }

    void visit(const ReductionNode* node) {
      taco_ierror
          << "Reduction nodes not supported in concrete index notation";
    }
  };

  return ir::simplify(GenerateGuard(guardedTemps, tempToBitGuard).lower(expr));
}


bool LowererImpl::isAssembledByUngroupedInsertion(TensorVar result) {
  return util::contains(assembledByUngroupedInsert, result);
}


bool LowererImpl::isAssembledByUngroupedInsertion(Expr result) {
  for (const auto& tensor : assembledByUngroupedInsert) {
    if (getTensorVar(tensor) == result) {
      return true;
    }
  }
  return false;
}


bool LowererImpl::hasStores(Stmt stmt) {
  if (!stmt.defined()) {
    return false;
  }

  struct FindStores : IRVisitor {
    bool hasStore;
    const std::map<TensorVar, Expr>& tensorVars;
    const std::map<TensorVar, Expr>& tempToBitGuard;

    using IRVisitor::visit;

    FindStores(const std::map<TensorVar, Expr>& tensorVars,
               const std::map<TensorVar, Expr>& tempToBitGuard)
        : tensorVars(tensorVars), tempToBitGuard(tempToBitGuard) {}

    void visit(const Store* stmt) {
      hasStore = true;
    }

    void visit(const Assign* stmt) {
      for (const auto& tensorVar : tensorVars) {
        if (stmt->lhs == tensorVar.second) {
          hasStore = true;
          break;
        }
      }
      if (hasStore) {
        return;
      }
      for (const auto& bitGuard : tempToBitGuard) {
        if (stmt->lhs == bitGuard.second) {
          hasStore = true;
          break;
        }
      }
    }

    bool hasStores(Stmt stmt) {
      hasStore = false;
      stmt.accept(this);
      return hasStore;
    }
  };
  return FindStores(tensorVars, tempToBitGuard).hasStores(stmt);
}


Expr LowererImpl::searchForStartOfWindowPosition(Iterator iterator, ir::Expr start, ir::Expr end) {
    taco_iassert(iterator.isWindowed());
    vector<Expr> args = {
            // Search over the `crd` array of the level,
            iterator.getMode().getModePack().getArray(1),
            // between the start and end position,
            start, end,
            // for the beginning of the window.
            iterator.getWindowLowerBound(),
    };
    return Call::make("taco_binarySearchAfter", args, Datatype::UInt64);
}


Expr LowererImpl::searchForEndOfWindowPosition(Iterator iterator, ir::Expr start, ir::Expr end) {
    taco_iassert(iterator.isWindowed());
    vector<Expr> args = {
            // Search over the `crd` array of the level,
            iterator.getMode().getModePack().getArray(1),
            // between the start and end position,
            start, end,
            // for the end of the window.
            iterator.getWindowUpperBound(),
    };
    return Call::make("taco_binarySearchAfter", args, Datatype::UInt64);
}


Stmt LowererImpl::upperBoundGuardForWindowPosition(Iterator iterator, ir::Expr access) {
  taco_iassert(iterator.isWindowed());
  return ir::IfThenElse::make(
    ir::Gte::make(access, ir::Div::make(ir::Sub::make(iterator.getWindowUpperBound(), iterator.getWindowLowerBound()), iterator.getStride())),
    ir::Break::make()
  );
}


Stmt LowererImpl::strideBoundsGuard(Iterator iterator, ir::Expr access, bool incrementPosVar) {
  Stmt cont = ir::Continue::make();
  // If requested to increment the iterator's position variable, add the increment
  // before the continue statement.
  if (incrementPosVar) {
    cont = ir::Block::make({
                               ir::Assign::make(iterator.getPosVar(),
                                                ir::Add::make(iterator.getPosVar(), ir::Literal::make(1))),
                               cont
                           });
  }
  // The guard makes sure that the coordinate being accessed is along the stride.
  return ir::IfThenElse::make(
      ir::Neq::make(ir::Rem::make(ir::Sub::make(access, iterator.getWindowLowerBound()), iterator.getStride()), ir::Literal::make(0)),
      cont
  );
}


Expr LowererImpl::projectWindowedPositionToCanonicalSpace(Iterator iterator, ir::Expr expr) {
  return ir::Div::make(ir::Sub::make(expr, iterator.getWindowLowerBound()), iterator.getStride());
}


Expr LowererImpl::projectCanonicalSpaceToWindowedPosition(Iterator iterator, ir::Expr expr) {
  return ir::Add::make(ir::Mul::make(expr, iterator.getStride()), iterator.getWindowLowerBound());
}

}
