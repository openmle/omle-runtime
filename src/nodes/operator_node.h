#ifndef OMLE_NODES_OPERATOR_NODE_H_
#define OMLE_NODES_OPERATOR_NODE_H_

#include <string>

#include "../ast_types.h"
#include "../graph_node.h"

namespace omle::rt::impl {

// Dispatches to a registered operator function by (domain, op).
class OperatorNode final : public GraphNode {
 public:
  std::string domain;
  std::string op;
  AttributeMap attrs;

  omle::rt::Status execute(ValueStore& vs, int n_rows) const override;
};

}  // namespace omle::rt::impl

#endif  // OMLE_NODES_OPERATOR_NODE_H_
