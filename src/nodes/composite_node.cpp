#include "composite_node.h"

#include <unordered_map>

#include "../status_macros.h"

namespace omle::rt::impl {

omle::rt::Status CompositeNode::execute(ValueStore& vs, int n_rows) const {
  std::unordered_map<std::string, std::string> in_alias;
  std::unordered_map<std::string, std::string> out_alias;
  for (const auto& [f, t] : input_aliases) in_alias[f] = t;
  for (const auto& [f, t] : output_aliases) out_alias[t] = f;

  ValueStore child = vs.child_scope();
  for (const auto& ext_name : in_names) {
    auto it = in_alias.find(ext_name);
    const std::string& int_name =
        (it != in_alias.end()) ? it->second : ext_name;
    child.put(int_name, vs.get(ext_name));
  }

  for (const auto& node : nodes) {
    RETURN_IF_ERROR(node->execute(child, n_rows));
  }

  for (const auto& ext_name : out_names) {
    auto it = out_alias.find(ext_name);
    const std::string& int_name =
        (it != out_alias.end()) ? it->second : ext_name;
    vs.put(ext_name, child.get(int_name));
  }
  return {};
}

}  // namespace omle::rt::impl
