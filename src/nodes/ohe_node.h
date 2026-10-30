#ifndef OMLE_NODES_OHE_NODE_H_
#define OMLE_NODES_OHE_NODE_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "../graph_node.h"

namespace omle::rt::impl {

// OneHotEncoder with pre-built category hash maps.
// Hash maps are built once at model-load time from the constant-store
// categories tensor; inference uses O(1) lookup instead of the O(n_categories)
// linear scan of the generic OperatorNode path.
class OHENode final : public GraphNode {
 public:
  // true when in_names holds one tensor per feature (per-column mode).
  // false when in_names[0] is a wide multi-column string tensor.
  bool per_col = false;

  // cat_maps[f][category_string] = local column index within feature f's output
  // block.
  std::vector<std::unordered_map<std::string, int>> cat_maps;

  // Width (number of output columns) of each feature's block.
  std::vector<int> feat_widths;

  // Sum of feat_widths.
  int total_output_cols = 0;

  omle::rt::Status execute(ValueStore& vs, int n_rows) const override;
};

}  // namespace omle::rt::impl

#endif  // OMLE_NODES_OHE_NODE_H_
