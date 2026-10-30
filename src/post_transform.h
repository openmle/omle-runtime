#ifndef OMLE_POST_TRANSFORM_H_
#define OMLE_POST_TRANSFORM_H_

#include <algorithm>
#include <cmath>

#include "model_base.h"
#include "omle/port.h"

// Shared post-transform application used by tree ensembles, linear models, etc.

namespace omle::rt::impl {

// Apply post-transform in-place on `scores[0 .. n_samples*n_outputs)`.
// T is either float or double.
template <typename T>
inline void apply_post_transform(T* OMLE_RESTRICT scores, int n_samples,
                                 int n_outputs, PostTransform pt) {
  const int total = n_samples * n_outputs;

  switch (pt) {
    case PostTransform::Identity:
      break;

    case PostTransform::Sigmoid:
      for (int i = 0; i < total; ++i)
        scores[i] = T(1) / (T(1) + std::exp(-scores[i]));
      break;

    case PostTransform::Softmax:
      for (int s = 0; s < n_samples; ++s) {
        T* row = scores + s * n_outputs;
        T mx = *std::max_element(row, row + n_outputs);
        T sum = T(0);
        for (int j = 0; j < n_outputs; ++j) {
          row[j] = std::exp(row[j] - mx);
          sum += row[j];
        }
        const T inv = T(1) / sum;
        for (int j = 0; j < n_outputs; ++j) row[j] *= inv;
      }
      break;

    case PostTransform::Exp:
      for (int i = 0; i < total; ++i) scores[i] = std::exp(scores[i]);
      break;

    case PostTransform::Logit:
      for (int i = 0; i < total; ++i)
        scores[i] = T(1) / (T(1) + std::exp(-scores[i]));
      break;

    case PostTransform::Probit:
      for (int i = 0; i < total; ++i) {
        T p = scores[i];
        p = std::max(T(1e-7), std::min(T(1) - T(1e-7), p));
        const T sign = p < T(0.5) ? T(-1) : T(1);
        const T t = std::sqrt(T(-2) * std::log(std::min(p, T(1) - p)));
        const T n = t - (T(2.515517) + T(0.802853) * t + T(0.010328) * t * t) /
                            (T(1) + T(1.432788) * t + T(0.189269) * t * t +
                             T(0.001308) * t * t * t);
        scores[i] = sign * n;
      }
      break;

    case PostTransform::CLogLog:
      for (int i = 0; i < total; ++i)
        scores[i] = T(1) - std::exp(-std::exp(scores[i]));
      break;

    case PostTransform::Cauchit:
      for (int i = 0; i < total; ++i)
        scores[i] = T(0.5) + std::atan(scores[i]) * T(0.318309886);
      break;

    case PostTransform::LogLog:
      for (int i = 0; i < total; ++i)
        scores[i] = std::exp(-std::exp(-scores[i]));
      break;

    case PostTransform::SigmoidBinary:
      // n_outputs == 2; col 0 holds the raw sum, col 1 is zero-initialised.
      // Apply sigmoid to col 0 then set col 1 = 1 - p.
      for (int s = 0; s < n_samples; ++s) {
        T p = T(1) / (T(1) + std::exp(-scores[s * 2]));
        scores[s * 2] = T(1) - p;
        scores[s * 2 + 1] = p;
      }
      break;
  }
}

}  // namespace omle::rt::impl

#endif  // OMLE_POST_TRANSFORM_H_
