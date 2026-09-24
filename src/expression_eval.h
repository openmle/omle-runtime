#ifndef OMLE_EXPRESSION_EVAL_H_
#define OMLE_EXPRESSION_EVAL_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "ast_types.h"
#include "omle/status.h"
#include "runtime_tensor.h"

namespace omle::rt::impl {

// -----------------------------------------------------------------------
// User-defined functions (compiled from DefineFunction proto messages).
// -----------------------------------------------------------------------

struct UserFunction {
  std::vector<std::string> params;  // parameter names in declaration order
  ExprPtr body;                     // compiled body expression
};

using UserFunctionMap = std::unordered_map<std::string, UserFunction>;

// RAII context guard: activates a UserFunctionMap for the calling thread.
// eval_apply resolves unknown function names through the active context.
// Contexts nest (destroying a guard restores the previous context).
class UserFunctionContext {
 public:
  explicit UserFunctionContext(const UserFunctionMap& map) noexcept;
  ~UserFunctionContext() noexcept;

  // Returns the currently active map for this thread, or nullptr.
  static const UserFunctionMap* current() noexcept;

 private:
  const UserFunctionMap* prev_;
};

// -----------------------------------------------------------------------
// Expression and predicate evaluation
// -----------------------------------------------------------------------

// Evaluate an expression over all N rows.  Returns a [N]-length double
// column. The DSL evaluates in double whatever the column's storage width:
// its 61 primitives compose, so a float32 rounding at every step compounds
// through the expression rather than happening once at the end.
omle::rt::StatusOr<std::vector<double>> eval_expr(const Expr& expr,
                                                  const ValueStore& vs,
                                                  int n_rows);

inline omle::rt::StatusOr<std::vector<double>> eval_expr(const ExprPtr& ep,
                                                         const ValueStore& vs,
                                                         int n_rows) {
  return eval_expr(*ep, vs, n_rows);
}

// Evaluate a predicate over all N rows.
// Returns a [N]-length byte vector: 1 = true, 0 = false.
omle::rt::StatusOr<std::vector<uint8_t>> eval_pred(const Pred& pred,
                                                   const ValueStore& vs,
                                                   int n_rows);

inline omle::rt::StatusOr<std::vector<uint8_t>> eval_pred(const PredPtr& pp,
                                                          const ValueStore& vs,
                                                          int n_rows) {
  return eval_pred(*pp, vs, n_rows);
}

}  // namespace omle::rt::impl

#endif  // OMLE_EXPRESSION_EVAL_H_
