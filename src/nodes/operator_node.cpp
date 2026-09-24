#include "operator_node.h"

#include "../operator_registry.h"

namespace omle::rt::impl {

// Operators used to be narrowed to float32 before dispatch unless they were
// known to handle float64, because reading a Float64 tensor through the
// float32 accessors (at(), row(), f32_ptr()) reinterprets double storage as
// float -- silent garbage, since the dtype assert in f32_ptr() compiles out of
// a release build.
//
// Every registered operator now reads its inputs through the dtype-generic
// get(), so there is nothing left to narrow and the shim is gone. What guards
// a newly added operator is the assert inside f32_ptr() itself: a debug build
// aborts by name the first time one reinterprets a float64 tensor, which is
// how the operators below were found in the first place.
omle::rt::Status OperatorNode::execute(ValueStore& vs, int n_rows) const {
  const OperatorFn* fn = OperatorRegistry::instance().find(domain, op);
  if (!fn)
    return {omle::rt::ErrorCode::UnknownOperator,
            "unknown operator: " + domain + "/" + op};
  return (*fn)(vs, n_rows, in_names, out_names, attrs);
}

}  // namespace omle::rt::impl
