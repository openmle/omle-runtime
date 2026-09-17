#ifndef OMLE_NODES_COMPOSITE_NODE_H_
#define OMLE_NODES_COMPOSITE_NODE_H_

#include <string>
#include <utility>
#include <vector>

#include "../graph_node.h"

namespace omle::rt::impl {

// A CompositeNode runs a local sub-DAG in an isolated ValueStore scope.
// input_aliases:  (external_name → internal_name) renames on entry
// output_aliases: (internal_name → external_name) renames on exit
class CompositeNode final : public GraphNode {
 public:
  using Alias = std::pair<std::string, std::string>;

  std::vector<Alias> input_aliases;
  std::vector<std::unique_ptr<GraphNode>> nodes;  // topological order
  std::vector<Alias> output_aliases;

  omle::rt::Status execute(ValueStore& vs, int n_rows) const override;
};

}  // namespace omle::rt::impl

#endif  // OMLE_NODES_COMPOSITE_NODE_H_
