#ifndef OMLE_MODEL_BASE_H_
#define OMLE_MODEL_BASE_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "omle/tensor.h"

namespace omle::rt::impl {

// ----------------------------------------------------------------
// Shared enumerations used by tree ensembles, linear models, etc.
// ----------------------------------------------------------------
enum class SplitOp : uint8_t {
  LessThan = 1,
  LessOrEqual = 2,
  GreaterThan = 3,
  GreaterOrEqual = 4,
  Equal = 5,
  NotEqual = 6,
  InSet = 7,
  NotInSet = 8,
  IsMissing = 9,
};

enum class PostTransform : uint8_t {
  Identity = 0,
  Sigmoid = 1,
  Softmax = 2,
  Exp = 3,
  Logit = 4,
  Probit = 5,
  CLogLog = 6,
  Cauchit = 7,
  LogLog = 8,
  SigmoidBinary = 9,
};

enum class Aggregation : uint8_t {
  Sum = 0,
  Average = 1,
  WeightedSum = 2,
  WeightedAvg = 3,
  MajorityVote = 4,
  SoftVote = 5,
  Min = 6,
  Max = 7,
};

class ModelBase {
 public:
  virtual ~ModelBase() = default;

  virtual int num_inputs() const = 0;
  virtual int num_outputs() const = 0;

  // Internal precision of this executor.  Subclasses that operate natively
  // in float64 override this to return Float64 so that predict_named can
  // supply a double-precision flat buffer and get a double-precision result.
  virtual omle::rt::DataType dtype() const {
    return omle::rt::DataType::Float32;
  }

  // ----------------------------------------------------------------
  // Float32 flat path (pure virtual — every executor must provide it).
  // ----------------------------------------------------------------
  virtual void predict(const float* features, int n_samples,
                       float* output) const = 0;

  // Single-threaded float32 path; defaults to predict().
  virtual void predict_serial(const float* features, int n_samples,
                              float* output) const {
    predict(features, n_samples, output);
  }

  // ----------------------------------------------------------------
  // Float64 flat path.
  // Default: downcast inputs to float32, call predict(), upcast output.
  // Float64 executors override to run the native double code path.
  // ----------------------------------------------------------------
  virtual void predict(const double* features, int n_samples,
                       double* output) const {
    const int nf = num_inputs(), no = num_outputs();
    std::vector<float> flat_f(static_cast<std::size_t>(n_samples) * nf);
    std::vector<float> out_f(static_cast<std::size_t>(n_samples) * no);
    for (std::size_t i = 0; i < flat_f.size(); ++i)
      flat_f[i] = static_cast<float>(features[i]);
    predict(flat_f.data(), n_samples, out_f.data());
    for (std::size_t i = 0; i < out_f.size(); ++i)
      output[i] = static_cast<double>(out_f[i]);
  }

  // Single-threaded float64 path; defaults to predict(double*,...).
  virtual void predict_serial(const double* features, int n_samples,
                              double* output) const {
    predict(features, n_samples, output);
  }

  // ----------------------------------------------------------------
  // Named input → named output paths.
  // If output_filter is non-empty only names present in it are returned.
  // Branches on dtype() to use float32 or float64 buffers.
  // ----------------------------------------------------------------
  virtual std::unordered_map<std::string, omle::rt::Tensor> predict_named(
      const std::unordered_map<std::string, omle::rt::Tensor>& inputs,
      const std::vector<std::string>& output_filter = {}) const {
    return default_predict_named(inputs, false, output_filter);
  }

  virtual std::unordered_map<std::string, omle::rt::Tensor>
  predict_named_serial(
      const std::unordered_map<std::string, omle::rt::Tensor>& inputs,
      const std::vector<std::string>& output_filter = {}) const {
    return default_predict_named(inputs, true, output_filter);
  }

 private:
  std::unordered_map<std::string, omle::rt::Tensor> default_predict_named(
      const std::unordered_map<std::string, omle::rt::Tensor>& inputs,
      bool serial, const std::vector<std::string>& output_filter = {}) const {
    int n_rows = 0;
    const int n_feats = num_inputs();
    const int nout = num_outputs();
    for (auto& [k, t] : inputs) {
      n_rows = t.n_rows;
      break;
    }
    if (n_rows == 0) return {};

    omle::rt::Tensor result;

    if (dtype() == omle::rt::DataType::Float64) {
      // ---- float64 path ----
      std::vector<double> flat(static_cast<std::size_t>(n_rows) * n_feats, 0.0);
      if (inputs.size() == 1) {
        const auto& t = inputs.begin()->second;
        omle::rt::Tensor dense = t.to_float64();
        std::memcpy(
            flat.data(), dense.f64_ptr(),
            static_cast<std::size_t>(n_rows) * n_feats * sizeof(double));
      } else {
        int col = 0;
        for (auto& [k, t] : inputs) {
          omle::rt::Tensor dense = t.to_float64();
          const int w = dense.n_cols;
          for (int r = 0; r < n_rows; ++r)
            std::memcpy(flat.data() + r * n_feats + col, dense.f64_row(r),
                        w * sizeof(double));
          col += w;
        }
      }
      std::vector<double> out(static_cast<std::size_t>(n_rows) * nout);
      if (serial)
        predict_serial(flat.data(), n_rows, out.data());
      else
        predict(flat.data(), n_rows, out.data());
      result = omle::rt::Tensor::from_doubles(n_rows, nout, std::move(out));
    } else {
      // ---- float32 path (original behaviour) ----
      std::vector<float> flat(static_cast<std::size_t>(n_rows) * n_feats, 0.f);
      if (inputs.size() == 1) {
        const auto& t = inputs.begin()->second;
        omle::rt::Tensor dense = t.to_float32();
        std::memcpy(flat.data(), dense.f32_ptr(),
                    static_cast<std::size_t>(n_rows) * n_feats * sizeof(float));
      } else {
        int col = 0;
        for (auto& [k, t] : inputs) {
          omle::rt::Tensor dense = t.to_float32();
          const int w = dense.n_cols;
          for (int r = 0; r < n_rows; ++r)
            std::memcpy(flat.data() + r * n_feats + col, dense.row(r),
                        w * sizeof(float));
          col += w;
        }
      }
      std::vector<float> out(static_cast<std::size_t>(n_rows) * nout);
      if (serial)
        predict_serial(flat.data(), n_rows, out.data());
      else
        predict(flat.data(), n_rows, out.data());
      result = omle::rt::Tensor::from_floats(n_rows, nout, std::move(out));
    }

    if (!output_filter.empty() &&
        std::find(output_filter.begin(), output_filter.end(), "output") ==
            output_filter.end())
      return {};
    return {{"output", std::move(result)}};
  }
};

}  // namespace omle::rt::impl

#endif  // OMLE_MODEL_BASE_H_
