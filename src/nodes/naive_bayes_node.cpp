#include "naive_bayes_node.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <type_traits>
#include <vector>

#include "../post_transform.h"
#include "../simd_traits.h"
#include "../width_dispatch.h"

namespace omle::rt::impl {

// -----------------------------------------------------------------------
// Abstract Impl base
// -----------------------------------------------------------------------
struct NaiveBayesNode::Impl {
  virtual int n_features() const noexcept = 0;
  virtual int n_classes() const noexcept = 0;
  // dtype() is the input feature dtype; outputs are always Float32.
  virtual omle::rt::DataType dtype() const noexcept = 0;
  // features: [n_rows, n_features] of dtype(); pred_out: [n_rows, 1] float32;
  // prob_out: [n_rows, n_classes] float32 (softmax probabilities).
  virtual void compute(const void* features, int n_rows, float* pred_out,
                       float* prob_out) const = 0;
  virtual ~Impl() = default;
};

// -----------------------------------------------------------------------
// Typed model storage
// -----------------------------------------------------------------------
template <typename T>
struct NaiveBayesModelT {
  NaiveBayesVariant variant = NaiveBayesVariant::Gaussian;

  std::vector<T> class_log_priors;  // [n_classes]

  // Gaussian
  std::vector<T> means;      // [n_classes * n_features]
  std::vector<T> variances;  // [n_classes * n_features]
  std::optional<double> variance_epsilon;

  // Multinomial / Bernoulli
  std::vector<T> feature_log_prob;  // [n_classes * n_features]
  std::optional<double> binarize_threshold;

  // Categorical
  // category_log_prob is always stored as float because the look-up is by
  // integer category index (decoded from the input row treated as int).
  // The log-probs themselves are not feature-precision sensitive in the
  // same way as the continuous inputs.
  std::vector<float> category_log_prob;  // [n_classes * total_categories]
  std::vector<int32_t> category_offset;
  std::vector<int32_t> category_count;

  int n_classes = 0;
  int n_features = 0;
};

// -----------------------------------------------------------------------
// Templated compute kernels
// -----------------------------------------------------------------------
template <typename T>
static void gaussian_log_likelihoods_t(const NaiveBayesModelT<T>& model,
                                       const T* X,  // [n_rows, n_features]
                                       int n_rows,
                                       T* log_ll)  // [n_rows, n_classes]
{
  const T eps = static_cast<T>(model.variance_epsilon.value_or(1e-9));
  const int nc = model.n_classes;
  const int nf = model.n_features;

  for (int c = 0; c < nc; ++c) {
    const T* mu = model.means.data() + c * nf;
    const T* var = model.variances.data() + c * nf;
    for (int r = 0; r < n_rows; ++r) {
      const T* x = X + r * nf;
      T ll = T(0);
      for (int f = 0; f < nf; ++f) {
        T v = var[f] + eps;
        T d = x[f] - mu[f];
        ll += T(-0.5) *
              (d * d / v + std::log(T(2) * T(3.14159265358979323846) * v));
      }
      log_ll[r * nc + c] = ll;
    }
  }
}

template <typename T>
static void count_log_likelihoods_t(
    const std::vector<T>& log_prob,  // [n_classes, n_features]
    const T* X,                      // [n_rows, n_features]
    int n_rows, int nf, int nc,
    T* log_ll,  // [n_rows, n_classes]
    bool is_bernoulli, double binarize) {
  const T bin_thresh = static_cast<T>(binarize);
  for (int r = 0; r < n_rows; ++r) {
    const T* x = X + r * nf;
    for (int c = 0; c < nc; ++c) {
      const T* lp = log_prob.data() + c * nf;
      T ll = T(0);
      if (is_bernoulli) {
        for (int f = 0; f < nf; ++f) {
          T xi = (x[f] > bin_thresh) ? T(1) : T(0);
          T lp1m = std::log(T(1) - std::exp(lp[f]) + T(1e-9));
          ll += xi * lp[f] + (T(1) - xi) * lp1m;
        }
      } else {
        // Multinomial
        for (int f = 0; f < nf; ++f) ll += x[f] * lp[f];
      }
      log_ll[r * nc + c] = ll;
    }
  }
}

// Categorical: input values are treated as integer category indices.
// The log-prob table is always stored as float (it's a lookup table).
// We read the input as T and cast to int for indexing.
template <typename T>
static void categorical_log_likelihoods_t(const NaiveBayesModelT<T>& model,
                                          const T* X,  // [n_rows, n_features]
                                          int n_rows,
                                          T* log_ll)  // [n_rows, n_classes]
{
  const int nc = model.n_classes;
  const int nf = model.n_features;

  // Stride in category_log_prob per class row.
  int cat_row_stride = 0;
  for (int f = 0; f < nf; ++f) cat_row_stride += model.category_count[f];

  for (int r = 0; r < n_rows; ++r) {
    const T* x = X + r * nf;
    for (int c = 0; c < nc; ++c) {
      const float* class_row =
          model.category_log_prob.data() + c * cat_row_stride;
      float ll = 0.0f;
      for (int f = 0; f < nf; ++f) {
        int cat_idx = static_cast<int>(x[f]);
        if (cat_idx >= 0 && cat_idx < model.category_count[f])
          ll += class_row[model.category_offset[f] + cat_idx];
      }
      log_ll[r * nc + c] = static_cast<T>(ll);
    }
  }
}

// -----------------------------------------------------------------------
// Typed Impl
// -----------------------------------------------------------------------
template <typename T>
struct NaiveBayesImplT final : NaiveBayesNode::Impl {
  NaiveBayesModelT<T> model;

  int n_features() const noexcept override { return model.n_features; }
  int n_classes() const noexcept override { return model.n_classes; }
  omle::rt::DataType dtype() const noexcept override {
    return std::is_same<T, double>::value ? omle::rt::DataType::Float64
                                          : omle::rt::DataType::Float32;
  }

  void compute(const void* features_raw, int n_rows, float* pred_out,
               float* prob_out) const override {
    const T* X = static_cast<const T*>(features_raw);
    const int nc = model.n_classes;
    const int nf = model.n_features;

    // Compute log-likelihoods per class in T precision.
    std::vector<T> log_ll(static_cast<std::size_t>(n_rows) * nc, T(0));

    using V = NaiveBayesVariant;
    switch (model.variant) {
      case V::Gaussian:
        gaussian_log_likelihoods_t<T>(model, X, n_rows, log_ll.data());
        break;
      case V::Multinomial:
        count_log_likelihoods_t<T>(model.feature_log_prob, X, n_rows, nf, nc,
                                   log_ll.data(), false, 0.0);
        break;
      case V::Bernoulli:
        count_log_likelihoods_t<T>(model.feature_log_prob, X, n_rows, nf, nc,
                                   log_ll.data(), true,
                                   model.binarize_threshold.value_or(0.0));
        break;
      case V::Categorical:
        categorical_log_likelihoods_t<T>(model, X, n_rows, log_ll.data());
        break;
    }

    // Add log priors.
    for (int r = 0; r < n_rows; ++r)
      for (int c = 0; c < nc; ++c)
        log_ll[r * nc + c] += model.class_log_priors[c];

    // Prediction = argmax(log posterior); downcast result to float32.
    for (int r = 0; r < n_rows; ++r) {
      const T* row = log_ll.data() + r * nc;
      int best = 0;
      for (int c = 1; c < nc; ++c)
        if (row[c] > row[best]) best = c;
      pred_out[r] = static_cast<float>(best);
    }

    // Probability = softmax(log_posterior); downcast to float32.
    for (int r = 0; r < n_rows; ++r)
      for (int c = 0; c < nc; ++c)
        prob_out[r * nc + c] = static_cast<float>(log_ll[r * nc + c]);
    apply_post_transform(prob_out, n_rows, nc, PostTransform::Softmax);
  }
};

// Explicit instantiations — linker finds these; no other TU instantiates them.
template struct NaiveBayesImplT<float>;
template struct NaiveBayesImplT<double>;

// -----------------------------------------------------------------------
// Factory functions
// -----------------------------------------------------------------------
std::unique_ptr<NaiveBayesNode::Impl> make_naive_bayes_impl(
    NaiveBayesVariant variant, std::vector<float> class_log_priors,
    std::vector<float> means, std::vector<float> variances,
    std::optional<double> variance_epsilon, std::vector<float> feature_log_prob,
    std::optional<double> binarize_threshold,
    std::vector<float> category_log_prob, std::vector<int32_t> category_offset,
    std::vector<int32_t> category_count, int n_classes, int n_features) {
  auto p = std::make_unique<NaiveBayesImplT<float>>();
  p->model.variant = variant;
  p->model.class_log_priors = std::move(class_log_priors);
  p->model.means = std::move(means);
  p->model.variances = std::move(variances);
  p->model.variance_epsilon = variance_epsilon;
  p->model.feature_log_prob = std::move(feature_log_prob);
  p->model.binarize_threshold = binarize_threshold;
  p->model.category_log_prob = std::move(category_log_prob);
  p->model.category_offset = std::move(category_offset);
  p->model.category_count = std::move(category_count);
  p->model.n_classes = n_classes;
  p->model.n_features = n_features;
  return p;
}

std::unique_ptr<NaiveBayesNode::Impl> make_naive_bayes_impl_f64(
    NaiveBayesVariant variant, std::vector<double> class_log_priors,
    std::vector<double> means, std::vector<double> variances,
    std::optional<double> variance_epsilon,
    std::vector<double> feature_log_prob,
    std::optional<double> binarize_threshold,
    std::vector<double> category_log_prob, std::vector<int32_t> category_offset,
    std::vector<int32_t> category_count, int n_classes, int n_features) {
  auto p = std::make_unique<NaiveBayesImplT<double>>();
  p->model.variant = variant;
  p->model.class_log_priors = std::move(class_log_priors);
  p->model.means = std::move(means);
  p->model.variances = std::move(variances);
  p->model.variance_epsilon = variance_epsilon;
  p->model.feature_log_prob = std::move(feature_log_prob);
  p->model.binarize_threshold = binarize_threshold;
  // category_log_prob is always float in the model; narrow from double.
  // Done with an explicit transform rather than assign(): the iterator form
  // narrows inside <vector>, which MSVC reports as C4244 against its own
  // header, where it cannot be read or suppressed locally.
  p->model.category_log_prob.resize(category_log_prob.size());
  std::transform(category_log_prob.begin(), category_log_prob.end(),
                 p->model.category_log_prob.begin(),
                 [](double v) { return static_cast<float>(v); });
  p->model.category_offset = std::move(category_offset);
  p->model.category_count = std::move(category_count);
  p->model.n_classes = n_classes;
  p->model.n_features = n_features;
  return p;
}

// Node-level wrappers — defined here where Impl is complete.
std::unique_ptr<NaiveBayesNode> make_naive_bayes_node(
    NaiveBayesVariant variant, std::vector<float> class_log_priors,
    std::vector<float> means, std::vector<float> variances,
    std::optional<double> variance_epsilon, std::vector<float> feature_log_prob,
    std::optional<double> binarize_threshold,
    std::vector<float> category_log_prob, std::vector<int32_t> category_offset,
    std::vector<int32_t> category_count, int n_classes, int n_features) {
  return std::make_unique<NaiveBayesNode>(make_naive_bayes_impl(
      variant, std::move(class_log_priors), std::move(means),
      std::move(variances), variance_epsilon, std::move(feature_log_prob),
      binarize_threshold, std::move(category_log_prob),
      std::move(category_offset), std::move(category_count), n_classes,
      n_features));
}

std::unique_ptr<NaiveBayesNode> make_naive_bayes_node_f64(
    NaiveBayesVariant variant, std::vector<double> class_log_priors,
    std::vector<double> means, std::vector<double> variances,
    std::optional<double> variance_epsilon,
    std::vector<double> feature_log_prob,
    std::optional<double> binarize_threshold,
    std::vector<double> category_log_prob, std::vector<int32_t> category_offset,
    std::vector<int32_t> category_count, int n_classes, int n_features) {
  return std::make_unique<NaiveBayesNode>(make_naive_bayes_impl_f64(
      variant, std::move(class_log_priors), std::move(means),
      std::move(variances), variance_epsilon, std::move(feature_log_prob),
      binarize_threshold, std::move(category_log_prob),
      std::move(category_offset), std::move(category_count), n_classes,
      n_features));
}

// -----------------------------------------------------------------------
// NaiveBayesNode
// -----------------------------------------------------------------------
NaiveBayesNode::NaiveBayesNode(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
NaiveBayesNode::~NaiveBayesNode() = default;

omle::rt::Status NaiveBayesNode::execute(ValueStore& vs, int n_rows) const {
  const omle::rt::DataType dt = impl_->dtype();
  auto features = gather_slots(vs, in_names, n_rows, dt);

  const int nc = impl_->n_classes();

  // Both outputs are Float32.
  Tensor pred(n_rows, 1);
  Tensor prob =
      omle::rt::Tensor::dense(omle::rt::DataType::Float32, n_rows, nc);

  with_width(dt == omle::rt::DataType::Float64, [&](auto tag) {
    using T = decltype(tag);
    impl_->compute((const void*)data_w<T>(features), n_rows, pred.f32_ptr(),
                   prob.f32_ptr());
  });

  if (!out_names.empty()) vs.put(out_names[0], std::move(pred));
  if (out_names.size() >= 2) vs.put(out_names[1], std::move(prob));
  return {};
}

}  // namespace omle::rt::impl
