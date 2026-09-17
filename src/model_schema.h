#ifndef OMLE_MODEL_SCHEMA_H_
#define OMLE_MODEL_SCHEMA_H_

#include <limits>
#include <unordered_set>
#include <vector>

#include "graph_node.h"
#include "runtime_tensor.h"

namespace omle::rt::impl {

// -----------------------------------------------------------------------
// Per-feature preprocessing policies (mirror proto Feature enums)
// -----------------------------------------------------------------------

enum class MissingPolicy : uint8_t {
  Propagate,  // NaN stays NaN downstream
  AsValue,    // replace with missing_replacement
  AsInvalid,  // re-route through invalid_policy
};

enum class InvalidPolicy : uint8_t {
  ReturnInvalid,  // output NaN
  AsIs,           // keep the raw value
  AsMissing,      // output NaN
  AsValue,        // replace with invalid_replacement
};

enum class OutlierPolicy : uint8_t {
  AsIs,       // keep the outlier value
  AsMissing,  // output NaN
  AsExtreme,  // clamp to nearest declared interval boundary
};

// -----------------------------------------------------------------------
// Compiled domain
// -----------------------------------------------------------------------

struct CompiledInterval {
  float left = -std::numeric_limits<float>::infinity();
  float right = std::numeric_limits<float>::infinity();
  bool left_open = true;
  bool right_open = true;

  bool contains(float v) const noexcept {
    if (left_open ? v <= left : v < left) return false;
    if (right_open ? v >= right : v > right) return false;
    return true;
  }

  // Nearest point strictly inside (or on the boundary of) this interval.
  float clamp_to(float v) const noexcept {
    float lo =
        left_open ? std::nextafter(left, std::numeric_limits<float>::infinity())
                  : left;
    float hi = right_open ? std::nextafter(
                                right, -std::numeric_limits<float>::infinity())
                          : right;
    if (lo > hi) return (lo + hi) * 0.5f;
    return v < lo ? lo : (v > hi ? hi : v);
  }
};

struct CompiledDomain {
  enum class Kind : uint8_t { None, Continuous, Discrete } kind = Kind::None;

  // Continuous: declared allowed intervals (OR-ed; value must fall in at least
  // one).
  std::vector<CompiledInterval> intervals;

  // Discrete: sets keyed by their declared ValueProperty.
  std::unordered_set<float>
      valid_set;  // VALID entries (non-empty ⇒ closed-world)
  std::unordered_set<float> invalid_set;  // INVALID entries
  std::unordered_set<float> missing_set;  // MISSING entries (treated like NaN)
};

// -----------------------------------------------------------------------
// One compiled logical feature
// -----------------------------------------------------------------------

struct SchemaFeature {
  std::string name;    // logical name published to ValueStore
  std::string source;  // source input name in ValueStore
  int col = 0;         // column index within source tensor

  MissingPolicy missing_policy = MissingPolicy::Propagate;
  float missing_replacement = 0.0f;
  InvalidPolicy invalid_policy = InvalidPolicy::ReturnInvalid;
  float invalid_replacement = 0.0f;
  OutlierPolicy outlier_policy = OutlierPolicy::AsIs;

  CompiledDomain domain;
};

// -----------------------------------------------------------------------
// Schema node: runs before all graph nodes; binds logical features
// -----------------------------------------------------------------------

class ModelSchemaNode final : public GraphNode {
 public:
  std::vector<SchemaFeature> features;

  // When true: emit one [n_rows, n_features] tensor under batch_key instead of
  // n_features individual [n_rows, 1] tensors. Enabled when all features share
  // the same source and are in sequential column order (col == feature index).
  bool batch_mode = false;
  std::string batch_key;

  // Reads each feature from its source column, applies preprocessing
  // policies, and writes [n_rows, 1] tensors under each feature name.
  omle::rt::Status execute(ValueStore& vs, int n_rows) const override;
};

}  // namespace omle::rt::impl

#endif  // OMLE_MODEL_SCHEMA_H_
