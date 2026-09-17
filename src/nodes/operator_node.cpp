#include "operator_node.h"

#include "../operator_registry.h"

namespace omle::rt::impl {

omle::rt::Status OperatorNode::execute(ValueStore& vs, int n_rows) const {
  const OperatorFn* fn = OperatorRegistry::instance().find(domain, op);
  if (!fn)
    return {omle::rt::ErrorCode::UnknownOperator,
            "unknown operator: " + domain + "/" + op};
  return (*fn)(vs, n_rows, in_names, out_names, attrs);
}

}  // namespace omle::rt::impl
