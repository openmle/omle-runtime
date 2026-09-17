#include "model_schema.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "math_utils.h"

namespace omle::rt::impl {

namespace {

static constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

// Returns true iff v falls within at least one declared interval.
static bool in_any_interval(const CompiledDomain& dom, float v) noexcept {
  for (const auto& iv : dom.intervals)
    if (iv.contains(v)) return true;
  return false;
}

// Nearest boundary value across all intervals.
static float clamp_to_extreme(const CompiledDomain& dom, float v) noexcept {
  float best = kNaN;
  float best_dist = std::numeric_limits<float>::infinity();
  for (const auto& iv : dom.intervals) {
    float c = iv.clamp_to(v);
    float d = std::abs(c - v);
    if (d < best_dist) {
      best_dist = d;
      best = c;
    }
  }
  return best;
}

// Apply invalid_policy and return the resulting value.
static float handle_invalid(const SchemaFeature& f, float raw) noexcept {
  switch (f.invalid_policy) {
    case InvalidPolicy::AsIs:
      return raw;
    case InvalidPolicy::AsValue:
      return f.invalid_replacement;
    case InvalidPolicy::ReturnInvalid:
      return kNaN;
    case InvalidPolicy::AsMissing:
      return kNaN;
  }
  return kNaN;
}

static float apply_schema(const SchemaFeature& f, float raw) noexcept {
  const auto& dom = f.domain;

  // ── 1. Missing check ──────────────────────────────────────────────────
  {
    bool is_missing = is_nan_safe(raw);
    if (!is_missing && dom.kind == CompiledDomain::Kind::Discrete)
      is_missing = dom.missing_set.count(raw) > 0;

    if (is_missing) {
      switch (f.missing_policy) {
        case MissingPolicy::Propagate:
          return kNaN;
        case MissingPolicy::AsValue:
          return f.missing_replacement;
        case MissingPolicy::AsInvalid:
          return handle_invalid(f, raw);
      }
    }
  }

  // ── 2. Discrete invalid / valid-set check ─────────────────────────────
  if (dom.kind == CompiledDomain::Kind::Discrete) {
    bool in_invalid = dom.invalid_set.count(raw) > 0;
    bool in_valid = dom.valid_set.empty() || dom.valid_set.count(raw) > 0;
    if (in_invalid || !in_valid) return handle_invalid(f, raw);
  }

  // ── 3. Continuous outlier check ───────────────────────────────────────
  if (dom.kind == CompiledDomain::Kind::Continuous && !dom.intervals.empty()) {
    if (!in_any_interval(dom, raw)) {
      switch (f.outlier_policy) {
        case OutlierPolicy::AsIs:
          return raw;
        case OutlierPolicy::AsMissing:
          return kNaN;
        case OutlierPolicy::AsExtreme:
          return clamp_to_extreme(dom, raw);
      }
    }
  }

  return raw;
}

}  // anonymous namespace

omle::rt::Status ModelSchemaNode::execute(ValueStore& vs, int n_rows) const {
  // Batch mode: all features share one source, emit a single wide tensor.
  if (batch_mode && !features.empty()) {
    const Tensor& src = vs.get(features[0].source);
    if (!src.is_string()) {
      const int nf = static_cast<int>(features.size());
      Tensor batch(n_rows, nf);
      float* out = batch.f32_ptr();
      for (int r = 0; r < n_rows; ++r) {
        float* row = out + r * nf;
        for (int f = 0; f < nf; ++f)
          row[f] = apply_schema(features[f], src.at(r, features[f].col));
      }
      vs.put(batch_key, std::move(batch));
      return {};
    }
  }

  for (const auto& feat : features) {
    const Tensor& src = vs.get(feat.source);
    // String tensors have no numeric schema to apply — pass through as-is.
    if (src.is_string()) {
      if (feat.name != feat.source) vs.put(feat.name, src);
      continue;
    }
    Tensor out(n_rows, 1);
    for (int r = 0; r < n_rows; ++r)
      out.f32_ptr()[r] = apply_schema(feat, src.at(r, feat.col));
    vs.put(feat.name, std::move(out));
  }
  return {};
}

}  // namespace omle::rt::impl
