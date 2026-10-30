#include "linear_node.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <type_traits>
#include <vector>

#include "../post_transform.h"
#include "../simd_traits.h"
#include "../width_dispatch.h"
#include "omle/port.h"

namespace omle::rt::impl {

// -----------------------------------------------------------------------
// Abstract Impl base — type-erased boundary seen by LinearNode::execute().
// -----------------------------------------------------------------------
struct LinearNode::Impl {
  virtual int n_features() const noexcept = 0;
  virtual int n_outputs() const noexcept = 0;
  virtual omle::rt::DataType dtype() const noexcept = 0;
  virtual PostTransform post_transform() const noexcept = 0;
  // Compute raw scores. features and output must match dtype().
  virtual void compute(const void* features, int n_rows,
                       void* output) const = 0;
  virtual ~Impl() = default;
};

// -----------------------------------------------------------------------
// Typed model storage
// -----------------------------------------------------------------------
template <typename T>
struct LinearModelT {
  std::vector<T> coefficients;  // [n_outputs * n_features], row-major
  std::vector<T> intercept;     // [n_outputs], may be empty
  PostTransform post_transform = PostTransform::Identity;
  int n_features = 0;
  int n_outputs = 1;
};

// -----------------------------------------------------------------------
// Templated compute kernels
// -----------------------------------------------------------------------
template <typename T>
static T dot(const T* OMLE_RESTRICT a, const T* OMLE_RESTRICT b,
             int n) noexcept {
  using ST = SimdTraits<T>;

#ifdef OMLE_AVX2
  const int aw = ST::AVX_WIDTH;
  auto acc0 = ST::zero(), acc1 = ST::zero();
  auto acc2 = ST::zero(), acc3 = ST::zero();
  int k = 0;
  for (; k + 4 * aw <= n; k += 4 * aw) {
    acc0 = ST::fmadd(ST::loadu(a + k), ST::loadu(b + k), acc0);
    acc1 = ST::fmadd(ST::loadu(a + k + aw), ST::loadu(b + k + aw), acc1);
    acc2 =
        ST::fmadd(ST::loadu(a + k + 2 * aw), ST::loadu(b + k + 2 * aw), acc2);
    acc3 =
        ST::fmadd(ST::loadu(a + k + 3 * aw), ST::loadu(b + k + 3 * aw), acc3);
  }
  acc0 = ST::add(ST::add(acc0, acc1), ST::add(acc2, acc3));
  for (; k + aw <= n; k += aw)
    acc0 = ST::fmadd(ST::loadu(a + k), ST::loadu(b + k), acc0);
  T result = ST::hsum(acc0);
  for (; k < n; ++k) result += a[k] * b[k];
  return result;

#elif defined(OMLE_NEON)
  const int nw = ST::NEON_WIDTH;
  auto acc0 = ST::nzero(), acc1 = ST::nzero();
  int k = 0;
  for (; k + 2 * nw <= n; k += 2 * nw) {
    acc0 = ST::nfma(acc0, ST::nloadu(a + k), ST::nloadu(b + k));
    acc1 = ST::nfma(acc1, ST::nloadu(a + k + nw), ST::nloadu(b + k + nw));
  }
  T result = ST::nhsum(ST::nadd(acc0, acc1));
  for (; k < n; ++k) result += a[k] * b[k];
  return result;

#else
  T result = T(0);
  for (int k = 0; k < n; ++k) result += a[k] * b[k];
  return result;
#endif
}

template <typename T>
static void predict_single_output(const LinearModelT<T>& model,
                                  const T* OMLE_RESTRICT features,
                                  int n_samples, T* OMLE_RESTRICT output) {
  using ST = SimdTraits<T>;
  const T* w = model.coefficients.data();
  const int nf = model.n_features;
  const T b = model.intercept.empty() ? T(0) : model.intercept[0];

#ifdef OMLE_NEON
  const int nw = ST::NEON_WIDTH;
  int s = 0;
  for (; s + 4 <= n_samples; s += 4) {
    const T* x0 = features + (s + 0) * nf;
    const T* x1 = features + (s + 1) * nf;
    const T* x2 = features + (s + 2) * nf;
    const T* x3 = features + (s + 3) * nf;
    auto a0 = ST::nzero(), a1 = ST::nzero();
    auto b0 = ST::nzero(), b1 = ST::nzero();
    auto c0 = ST::nzero(), c1 = ST::nzero();
    auto d0 = ST::nzero(), d1 = ST::nzero();
    int k = 0;
    for (; k + 2 * nw <= nf; k += 2 * nw) {
      auto w0 = ST::nloadu(w + k);
      auto w1 = ST::nloadu(w + k + nw);
      a0 = ST::nfma(a0, ST::nloadu(x0 + k), w0);
      a1 = ST::nfma(a1, ST::nloadu(x0 + k + nw), w1);
      b0 = ST::nfma(b0, ST::nloadu(x1 + k), w0);
      b1 = ST::nfma(b1, ST::nloadu(x1 + k + nw), w1);
      c0 = ST::nfma(c0, ST::nloadu(x2 + k), w0);
      c1 = ST::nfma(c1, ST::nloadu(x2 + k + nw), w1);
      d0 = ST::nfma(d0, ST::nloadu(x3 + k), w0);
      d1 = ST::nfma(d1, ST::nloadu(x3 + k + nw), w1);
    }
    output[s + 0] = ST::nhsum(ST::nadd(a0, a1)) + b;
    output[s + 1] = ST::nhsum(ST::nadd(b0, b1)) + b;
    output[s + 2] = ST::nhsum(ST::nadd(c0, c1)) + b;
    output[s + 3] = ST::nhsum(ST::nadd(d0, d1)) + b;
    for (; k < nf; ++k) {
      output[s + 0] += x0[k] * w[k];
      output[s + 1] += x1[k] * w[k];
      output[s + 2] += x2[k] * w[k];
      output[s + 3] += x3[k] * w[k];
    }
  }
  for (; s < n_samples; ++s) output[s] = dot<T>(features + s * nf, w, nf) + b;

#elif defined(OMLE_AVX2)
  const int aw = ST::AVX_WIDTH;
  int s = 0;
  for (; s + 4 <= n_samples; s += 4) {
    const T* x0 = features + (s + 0) * nf;
    const T* x1 = features + (s + 1) * nf;
    const T* x2 = features + (s + 2) * nf;
    const T* x3 = features + (s + 3) * nf;
    auto a0 = ST::zero(), a1 = ST::zero();
    auto b0 = ST::zero(), b1 = ST::zero();
    auto c0 = ST::zero(), c1 = ST::zero();
    auto d0 = ST::zero(), d1 = ST::zero();
    int k = 0;
    for (; k + 2 * aw <= nf; k += 2 * aw) {
      auto w0 = ST::loadu(w + k);
      auto w1 = ST::loadu(w + k + aw);
      a0 = ST::fmadd(ST::loadu(x0 + k), w0, a0);
      a1 = ST::fmadd(ST::loadu(x0 + k + aw), w1, a1);
      b0 = ST::fmadd(ST::loadu(x1 + k), w0, b0);
      b1 = ST::fmadd(ST::loadu(x1 + k + aw), w1, b1);
      c0 = ST::fmadd(ST::loadu(x2 + k), w0, c0);
      c1 = ST::fmadd(ST::loadu(x2 + k + aw), w1, c1);
      d0 = ST::fmadd(ST::loadu(x3 + k), w0, d0);
      d1 = ST::fmadd(ST::loadu(x3 + k + aw), w1, d1);
    }
    output[s + 0] = ST::hsum(ST::add(a0, a1)) + b;
    output[s + 1] = ST::hsum(ST::add(b0, b1)) + b;
    output[s + 2] = ST::hsum(ST::add(c0, c1)) + b;
    output[s + 3] = ST::hsum(ST::add(d0, d1)) + b;
    for (; k < nf; ++k) {
      output[s + 0] += x0[k] * w[k];
      output[s + 1] += x1[k] * w[k];
      output[s + 2] += x2[k] * w[k];
      output[s + 3] += x3[k] * w[k];
    }
  }
  for (; s < n_samples; ++s) output[s] = dot<T>(features + s * nf, w, nf) + b;

#else
  for (int s = 0; s < n_samples; ++s)
    output[s] = dot<T>(features + s * nf, w, nf) + b;
#endif
}

template <typename T>
static void predict_multi_output(const LinearModelT<T>& model,
                                 const T* OMLE_RESTRICT features, int n_samples,
                                 T* OMLE_RESTRICT output) {
  const int nf = model.n_features;
  const int nout = model.n_outputs;
  const bool has_bias = !model.intercept.empty();

  std::fill(output, output + static_cast<std::size_t>(n_samples) * nout, T(0));

  for (int j = 0; j < nout; ++j) {
    const T* wj = model.coefficients.data() + j * nf;
    const T bj = has_bias ? model.intercept[j] : T(0);
    for (int s = 0; s < n_samples; ++s)
      output[s * nout + j] = dot<T>(features + s * nf, wj, nf) + bj;
  }
}

// -----------------------------------------------------------------------
// Typed Impl
// -----------------------------------------------------------------------
template <typename T>
struct LinearImplT final : LinearNode::Impl {
  LinearModelT<T> model;

  int n_features() const noexcept override { return model.n_features; }
  int n_outputs() const noexcept override { return model.n_outputs; }
  PostTransform post_transform() const noexcept override {
    return model.post_transform;
  }
  omle::rt::DataType dtype() const noexcept override {
    return std::is_same<T, double>::value ? omle::rt::DataType::Float64
                                          : omle::rt::DataType::Float32;
  }

  void compute(const void* features_raw, int n_rows,
               void* output_raw) const override {
    const T* features = static_cast<const T*>(features_raw);
    T* output = static_cast<T*>(output_raw);
    if (model.n_outputs == 1)
      predict_single_output(model, features, n_rows, output);
    else
      predict_multi_output(model, features, n_rows, output);
    apply_post_transform(output, n_rows, model.n_outputs, model.post_transform);
  }
};

// Explicit instantiations — linker finds these; no other TU instantiates them.
template struct LinearImplT<float>;
template struct LinearImplT<double>;

// -----------------------------------------------------------------------
// Factory functions
// -----------------------------------------------------------------------
std::unique_ptr<LinearNode::Impl> make_linear_impl(
    std::vector<float> coefficients, std::vector<float> intercept,
    PostTransform post_transform, int n_features, int n_outputs) {
  auto p = std::make_unique<LinearImplT<float>>();
  p->model.coefficients = std::move(coefficients);
  p->model.intercept = std::move(intercept);
  p->model.post_transform = post_transform;
  p->model.n_features = n_features;
  p->model.n_outputs = n_outputs;
  return p;
}

std::unique_ptr<LinearNode::Impl> make_linear_impl_f64(
    std::vector<double> coefficients, std::vector<double> intercept,
    PostTransform post_transform, int n_features, int n_outputs) {
  auto p = std::make_unique<LinearImplT<double>>();
  p->model.coefficients = std::move(coefficients);
  p->model.intercept = std::move(intercept);
  p->model.post_transform = post_transform;
  p->model.n_features = n_features;
  p->model.n_outputs = n_outputs;
  return p;
}

// Node-level wrappers — defined here where Impl is complete.
std::unique_ptr<LinearNode> make_linear_node(std::vector<float> coefficients,
                                             std::vector<float> intercept,
                                             PostTransform post_transform,
                                             int n_features, int n_outputs) {
  return std::make_unique<LinearNode>(
      make_linear_impl(std::move(coefficients), std::move(intercept),
                       post_transform, n_features, n_outputs));
}

std::unique_ptr<LinearNode> make_linear_node_f64(
    std::vector<double> coefficients, std::vector<double> intercept,
    PostTransform post_transform, int n_features, int n_outputs) {
  return std::make_unique<LinearNode>(
      make_linear_impl_f64(std::move(coefficients), std::move(intercept),
                           post_transform, n_features, n_outputs));
}

std::unique_ptr<LinearExecutor> make_linear_executor(
    std::vector<float> coefficients, std::vector<float> intercept,
    PostTransform post_transform, int n_features, int n_outputs) {
  return std::make_unique<LinearExecutor>(
      make_linear_impl(std::move(coefficients), std::move(intercept),
                       post_transform, n_features, n_outputs),
      n_features, n_outputs);
}

std::unique_ptr<LinearExecutor> make_linear_executor_f64(
    std::vector<double> coefficients, std::vector<double> intercept,
    PostTransform post_transform, int n_features, int n_outputs) {
  return std::make_unique<LinearExecutor>(
      make_linear_impl_f64(std::move(coefficients), std::move(intercept),
                           post_transform, n_features, n_outputs),
      n_features, n_outputs);
}

// -----------------------------------------------------------------------
// LinearNode
// -----------------------------------------------------------------------
LinearNode::LinearNode(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
LinearNode::~LinearNode() = default;

omle::rt::Status LinearNode::execute(ValueStore& vs, int n_rows) const {
  const omle::rt::DataType dt = impl_->dtype();
  auto features = gather_slots(vs, in_names, n_rows, dt);

  const int n_out = impl_->n_outputs();
  const bool f64 = (dt == omle::rt::DataType::Float64);
  omle::rt::Tensor scores = omle::rt::Tensor::dense(dt, n_rows, n_out);
  with_width(f64, [&](auto tag) {
    using T = decltype(tag);
    impl_->compute((const void*)data_w<T>(features), n_rows,
                   (void*)data_w<T>(scores));
  });

  if (out_names.size() == 2) {
    // Classification: [y_pred, y_prob]
    omle::rt::Tensor pred =
        omle::rt::Tensor::dense(omle::rt::DataType::Int64, n_rows, 1);
    int64_t* pp = pred.i64_ptr();
    if (n_out == 1) {
      // Binary: expand (n,1) → (n,2): [1-p, p]
      omle::rt::Tensor prob2 = omle::rt::Tensor::dense(dt, n_rows, 2);
      with_width(f64, [&](auto tag) {
        using T = decltype(tag);
        const T* sp = data_w<T>(scores);
        T* dp = data_w<T>(prob2);
        for (int r = 0; r < n_rows; ++r) {
          dp[r * 2] = T(1) - sp[r];
          dp[r * 2 + 1] = sp[r];
          // Strictly greater, not >=. sklearn's LinearClassifierMixin.predict
          // uses (decision_function > 0) for the binary case, and a decision
          // value of 0 is exactly p == 0.5, so a tie resolves to class 0.
          // That also matches np.argmax([0.5, 0.5]), which the multiclass
          // branch below relies on.
          pp[r] = sp[r] > T(0.5) ? 1 : 0;
        }
      });
      vs.put(out_names[1], std::move(prob2));
    } else {
      omle::rt::Tensor prob = scores;
      vs.put(out_names[1], std::move(prob));
      with_width(f64, [&](auto tag) {
        using T = decltype(tag);
        const T* sp = data_w<T>(scores);
        for (int r = 0; r < n_rows; ++r) {
          const T* row = sp + r * n_out;
          pp[r] = static_cast<int64_t>(
              std::distance(row, std::max_element(row, row + n_out)));
        }
      });
    }
    vs.put(out_names[0], std::move(pred));
  } else {
    // Single-output binary probability (e.g. OVR sub-model with PREDICTION
    // stripped): expand scalar sigmoid output [n, 1] → [n, 2] so
    // TakeSlots(indices=[1]) works.
    if (out_names.size() == 1 && n_out == 1 &&
        impl_->post_transform() == PostTransform::Sigmoid) {
      omle::rt::Tensor prob2 = omle::rt::Tensor::dense(dt, n_rows, 2);
      with_width(f64, [&](auto tag) {
        using T = decltype(tag);
        const T* sp = data_w<T>(scores);
        T* dp = data_w<T>(prob2);
        for (int r = 0; r < n_rows; ++r) {
          dp[r * 2] = T(1) - sp[r];
          dp[r * 2 + 1] = sp[r];
        }
      });
      vs.put(out_names[0], std::move(prob2));
      return {};
    }
    with_width(f64, [&](auto tag) {
      using T = decltype(tag);
      scatter_outputs(vs, out_names, data_w<T>(scores), n_rows, n_out);
    });
  }
  return {};
}

// -----------------------------------------------------------------------
// LinearExecutor (standalone ModelBase path)
// -----------------------------------------------------------------------
LinearExecutor::LinearExecutor(std::unique_ptr<LinearNode::Impl> impl,
                               int n_features, int n_outputs)
    : impl_(std::move(impl)),
      n_features_(n_features),
      n_outputs_(n_outputs),
      dtype_(impl_->dtype()) {}

LinearExecutor::~LinearExecutor() = default;

omle::rt::DataType LinearExecutor::dtype() const { return dtype_; }

void LinearExecutor::predict(const float* features, int n_samples,
                             float* output) const {
  if (dtype_ == omle::rt::DataType::Float64) {
    // float32 input to float64 model: upcast, compute, downcast.
    thread_local std::vector<double> feat_d, out_d;
    const int nf = n_features_;
    const int no = n_outputs_;
    feat_d.resize(static_cast<std::size_t>(n_samples) * nf);
    out_d.resize(static_cast<std::size_t>(n_samples) * no);
    upcast_f32_to_f64(features, feat_d.data(), n_samples * nf);
    impl_->compute(feat_d.data(), n_samples, out_d.data());
    downcast_f64_to_f32(out_d.data(), output, n_samples * no);
  } else {
    impl_->compute(features, n_samples, output);
  }
}

void LinearExecutor::predict(const double* features, int n_samples,
                             double* output) const {
  if (dtype_ == omle::rt::DataType::Float64) {
    // float64 input to float64 model: native path, no conversion.
    impl_->compute(features, n_samples, output);
  } else {
    // float64 input to float32 model: downcast, compute, upcast.
    thread_local std::vector<float> feat_f, out_f;
    const int nf = n_features_;
    const int no = n_outputs_;
    feat_f.resize(static_cast<std::size_t>(n_samples) * nf);
    out_f.resize(static_cast<std::size_t>(n_samples) * no);
    downcast_f64_to_f32(features, feat_f.data(), n_samples * nf);
    impl_->compute(feat_f.data(), n_samples, out_f.data());
    upcast_f32_to_f64(out_f.data(), output, n_samples * no);
  }
}

}  // namespace omle::rt::impl
