#include "ohe_node.h"

#include <cstring>
#include <vector>

namespace omle::rt::impl {

omle::rt::Status OHENode::execute(ValueStore& vs, int n_rows) const {
  if (out_names.empty()) return {};

  const int needed = n_rows * total_output_cols;
  const int nf = static_cast<int>(cat_maps.size());

  // Thread-local output buffer — zeroed each call, reused across calls.
  // We write directly here and expose a zero-copy view to the ValueStore,
  // avoiding the alloc+memcpy of an intermediate owned result tensor.
  thread_local std::vector<float> tl_scratch;
  if (static_cast<int>(tl_scratch.size()) < needed)
    tl_scratch.assign(needed, 0.0f);
  else
    std::memset(tl_scratch.data(), 0,
                static_cast<std::size_t>(needed) * sizeof(float));

  const bool is_str = per_col ? vs.get(in_names[0]).is_string()
                              : vs.get(in_names[0]).is_string();

  for (int r = 0; r < n_rows; ++r) {
    int out_off = 0;
    for (int f = 0; f < nf; ++f) {
      if (is_str) {
        const std::string& x = per_col ? vs.get(in_names[f]).str_at(r, 0)
                                       : vs.get(in_names[0]).str_at(r, f);
        auto it = cat_maps[f].find(x);
        if (it != cat_maps[f].end())
          tl_scratch[r * total_output_cols + out_off + it->second] = 1.0f;
      } else {
        // Numeric input: treat value as ordinal index within this feature's
        // categories.
        int idx = static_cast<int>(per_col ? vs.get(in_names[f]).get(r, 0)
                                           : vs.get(in_names[0]).get(r, f));
        if (idx >= 0 && idx < feat_widths[f])
          tl_scratch[r * total_output_cols + out_off + idx] = 1.0f;
      }
      out_off += feat_widths[f];
    }
  }

  if (out_names.size() > 1) {
    int out_off = 0;
    for (int f = 0; f < nf && f < (int)out_names.size(); ++f) {
      const int w = feat_widths[f];
      Tensor col(n_rows, w);
      float* dp = col.f32_ptr();
      for (int r = 0; r < n_rows; ++r)
        std::memcpy(dp + r * w,
                    tl_scratch.data() + r * total_output_cols + out_off,
                    static_cast<std::size_t>(w) * sizeof(float));
      out_off += w;
      vs.put(out_names[f], std::move(col));
    }
  } else {
    Tensor result(n_rows, total_output_cols);
    std::memcpy(result.f32_ptr(), tl_scratch.data(),
                static_cast<std::size_t>(needed) * sizeof(float));
    vs.put(out_names[0], std::move(result));
  }
  return {};
}

}  // namespace omle::rt::impl
