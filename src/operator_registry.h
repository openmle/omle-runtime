#ifndef OMLE_OPERATOR_REGISTRY_H_
#define OMLE_OPERATOR_REGISTRY_H_

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "ast_types.h"
#include "omle/status.h"
#include "runtime_tensor.h"

namespace omle::rt::impl {

// An operator takes named inputs from the ValueStore, a set of attributes,
// and produces one or more named outputs into the ValueStore.
using OperatorFn = std::function<omle::rt::Status(
    ValueStore& vs, int n_rows, const std::vector<std::string>& in_names,
    const std::vector<std::string>& out_names, const AttributeMap& attrs)>;

class OperatorRegistry {
 public:
  static OperatorRegistry& instance();

  void reg(const std::string& domain, const std::string& op, OperatorFn fn);
  const OperatorFn* find(const std::string& domain,
                         const std::string& op) const;

 private:
  std::unordered_map<std::string, OperatorFn> table_;  // "domain/op"
};

// Registers all built-in operators (called once at startup).
void register_builtin_operators();

}  // namespace omle::rt::impl

#endif  // OMLE_OPERATOR_REGISTRY_H_
